"""Parameter-free cellular-automaton genome interpreter and tissue morphogen engine on FCC N12.

Given a set of genomic loci (each with a promoter DNA sequence and coding DNA sequence):
  1. Builds the B-DNA duplex and Major Groove 4-slot barcode (`dna.py`),
  2. Translates each coding sequence via the 64 -> 21 Rumer code (`genetic_code.py`),
  3. Folds each polypeptide in O(L) time via local K_4 groove selection (`folding.py`),
  4. Scans the folded proteins for C2H2 (`|N_12|=12`), Trp-pin (`d_in(square)=4`), and
     Ig synaptic (`N_HB=63`) locks (`locks.py`),
  5. Evaluates promoter occupancy and runs discrete cellular-automaton steps on the 3D FCC N12
     lattice with exact link coupling kappa_link = 1 / |N_12| = 1 / 12.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .constants import CARRIER, CarrierConstants, FCC_N12_VECTORS
from .dna import BDNADuplexGeometry, build_bdna_duplex
from .genetic_code import translate_rna_or_dna
from .folding import FoldedProtein, fold_protein_cotranslational
from .locks import (
    C2H2FingerMatch,
    TrpPinMatch,
    IgSynapticLock,
    scan_c2h2_locks,
    scan_trp_pin_locks,
    scan_ig_synaptic_locks,
    score_tf_dna_binding_ev,
)


@dataclass(frozen=True)
class GeneLocus:
    """Genomic locus with upstream promoter DNA and coding DNA sequence (CDS)."""

    gene_id: str
    promoter_dna: str
    coding_dna: str


@dataclass(frozen=True)
class ExpressedGeneProduct:
    """Physical and structural product of a single gene locus."""

    gene_id: str
    promoter_duplex: BDNADuplexGeometry
    coding_duplex: BDNADuplexGeometry
    protein_sequence: str
    folded_protein: FoldedProtein
    c2h2_locks: tuple[C2H2FingerMatch, ...]
    trp_pin_locks: tuple[TrpPinMatch, ...]
    ig_synaptic_locks: tuple[IgSynapticLock, ...]


def express_gene_locus(locus: GeneLocus, c: CarrierConstants = CARRIER) -> ExpressedGeneProduct:
    """Execute the full DNA -> mRNA -> O(L) folded protein -> structural lock pipeline."""
    prom_duplex = build_bdna_duplex(locus.promoter_dna, c=c)
    cds_duplex = build_bdna_duplex(locus.coding_dna, c=c)
    prot_seq = translate_rna_or_dna(locus.coding_dna, stop_at_termination=True)
    if not prot_seq:
        raise ValueError(f"Gene {locus.gene_id!r} produced an empty protein (early Stop codon).")

    folded = fold_protein_cotranslational(prot_seq, c=c)
    c2h2 = tuple(scan_c2h2_locks(prot_seq, c=c))
    trp = tuple(scan_trp_pin_locks(prot_seq))
    ig = tuple(scan_ig_synaptic_locks(prot_seq, c=c))

    return ExpressedGeneProduct(
        gene_id=locus.gene_id,
        promoter_duplex=prom_duplex,
        coding_duplex=cds_duplex,
        protein_sequence=prot_seq,
        folded_protein=folded,
        c2h2_locks=c2h2,
        trp_pin_locks=trp,
        ig_synaptic_locks=ig,
    )


def compute_regulatory_network_matrix_ev(
    products: list[ExpressedGeneProduct],
    c: CarrierConstants = CARRIER,
) -> np.ndarray:
    """Compute the N_genes x N_genes regulatory binding energy matrix (in eV) via Major Groove barcodes."""
    n = len(products)
    mat = np.zeros((n, n), dtype=np.float64)

    for i, regulator in enumerate(products):
        for j, target in enumerate(products):
            best_ev = 0.0
            prom_seq = target.promoter_duplex.sequence
            # Check C2H2 3-bp recognition matches along the promoter Major Groove
            for finger in regulator.c2h2_locks:
                readers = finger.reader_residues
                for k in range(max(0, len(prom_seq) - 2)):
                    subsite = prom_seq[k : k + 3]
                    ev = score_tf_dna_binding_ev(readers, subsite, c=c)
                    if ev < best_ev:
                        best_ev = ev
            # Check Trp-pin 4-bp core matches along the promoter Major Groove
            for pin in regulator.trp_pin_locks:
                core = pin.predicted_dna_core
                if core in prom_seq:
                    ev = -len(core) * c.e_hb_ev
                    if ev < best_ev:
                        best_ev = ev
            mat[i, j] = best_ev
    return mat


def step_fcc_morphogen_automaton(
    state: np.ndarray,
    regulatory_matrix_ev: np.ndarray,
    n_steps: int = 10,
    c: CarrierConstants = CARRIER,
) -> np.ndarray:
    """Advance a 3D FCC N12 cellular-automaton morphogen field with link coupling kappa_link = 1/12.

    Args:
        state: array of shape (Lz, Ly, Lx, N_genes) representing expression levels in [0, 1].
        regulatory_matrix_ev: N_genes x N_genes binding energies in eV (negative = activation).
        n_steps: number of discrete ticks.

    Returns:
        Updated state tensor of shape (Lz, Ly, Lx, N_genes).
    """
    cur = np.array(state, dtype=np.float64, copy=True)
    kappa_link = 1.0 / c.n_12  # Exact 1/12 FCC link coupling (no arbitrary diffusion constant)
    # Dimensionless regulatory weight normalized by the ATP cellular quantum (2 * E_HB)
    w_reg = -regulatory_matrix_ev / (c.atp_quantum_ev * c.n_12)

    for _ in range(n_steps):
        # Isotropic 12-neighbor cuboctahedral average on the 3D torus
        neighbor_sum = np.zeros_like(cur)
        for dz, dy, dx in FCC_N12_VECTORS:
            neighbor_sum += np.roll(cur, shift=(int(dz), int(dy), int(dx)), axis=(0, 1, 2))
        fcc_avg = kappa_link * neighbor_sum

        # Local gene-regulatory drive from Major Groove H-bond locks
        drive = np.einsum("zyxg,gh->zyxh", fcc_avg, w_reg)
        cur = np.clip(fcc_avg + drive, 0.0, 1.0)

    return cur
