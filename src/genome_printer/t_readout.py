"""T-layer laboratory projector B: M <-> T (IUPAC codecs, SI unit bridge, and X-Ray formatting).

This is the ONLY module in `genome-printer` that contains human T-layer nomenclature:
  - IUPAC base letters ('U'/'T', 'C', 'A', 'G') <-> 2-bit M-layer registers `b in Z_4`,
  - IUPAC 20 amino acid + Stop symbols <-> M-layer shell-2 class indices `a in Z_21`,
  - Laboratory SI calibration anchors (`a_0` in Angstroms, `BE_H` in eV / kJ/mol, `T_HB` in K)
    that project dimensionless M-layer quantities (`dh`, `E_0`, `E_HB` quanta) onto human
    laboratory instruments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .m_carrier import M_CARRIER, MCarrierConstants
from .m_chemistry import (
    m_ionization_ratio_be_h,
    m_covalent_radius_in_a0,
    m_sp3_bond_angle_rad,
    m_compute_hbond_phase_ratios,
)
from .m_nucleic import MDuplexLatticeState
from .m_polypeptide import MFoldedChain
from .m_tissue import MTissueMorphology, MChromosomeScanResult, m_scan_chromosome_de_novo


# T-layer IUPAC base codec: b in Z_4 (0, 1, 2, 3) <-> ('T'/'U', 'C', 'A', 'G')
T_BASE_TO_Z4: dict[str, int] = {"U": 0, "T": 0, "C": 1, "A": 2, "G": 3}
T_Z4_TO_DNA_BASE: tuple[str, ...] = ("T", "C", "A", "G")
T_Z4_TO_RNA_BASE: tuple[str, ...] = ("U", "C", "A", "G")

# T-layer Major Groove slot symbols: (-1, +1, +2, 0) <-> ('-', '+', 'M', '0')
T_SLOT_CODE_TO_SYMBOL: dict[int, str] = {-1: "-", +1: "+", +2: "M", 0: "0"}

# T-layer IUPAC 1-letter and 3-letter amino acid symbols for Z_21 classes 0..20
T_Z21_TO_AA1: tuple[str, ...] = (
    "L", "S", "R",  # 0..2: Octet I 3_square bridge
    "A", "V", "P", "G", "T",  # 3..7: Octet I V/E 4-fold
    "F", "Y", "W", "C", "M", "I", "H", "Q", "N", "K", "D", "E",  # 8..19: Octet II
    "*",  # 20: Stop
)

T_AA1_TO_Z21: dict[str, int] = {sym: idx for idx, sym in enumerate(T_Z21_TO_AA1)}

T_GROOVE_NAMES: tuple[str, ...] = ("alpha_3.6_13", "beta_111", "turn_k4", "stop")
T_BOND_KIND_NAMES: tuple[str, ...] = (
    "backbone_alpha_i_minus_4",
    "backbone_beta_i_minus_2",
    "sidechain_hb_fork",
    "disulfide_ss_bridge",
)


def encode_dna_to_z4(seq_str: str) -> tuple[int, ...]:
    """Project a T-layer DNA/RNA string into a tuple of 2-bit M-layer registers `b in Z_4`."""
    return tuple(T_BASE_TO_Z4[ch] for ch in seq_str.strip().upper())


def decode_z4_to_dna(codes: tuple[int, ...] | list[int]) -> str:
    """Project 2-bit M-layer registers `b in Z_4` into a T-layer DNA string."""
    return "".join(T_Z4_TO_DNA_BASE[b & 3] for b in codes)


def encode_protein_to_z21(seq_str: str) -> tuple[int, ...]:
    """Project a T-layer 1-letter amino acid string into M-layer class indices `a in Z_21`."""
    return tuple(T_AA1_TO_Z21[ch] for ch in seq_str.strip().upper())


def decode_z21_to_protein(class_ids: tuple[int, ...] | list[int]) -> str:
    """Project M-layer class indices `a in Z_21` into a T-layer 1-letter amino acid string."""
    return "".join(T_Z21_TO_AA1[a] for a in class_ids)


@dataclass(frozen=True)
class TSIBridgeProjector:
    """T-layer SI calibration bridge projecting native M-layer units (dh, E_0, E_HB) to SI / Angstrom / eV."""

    a_0_angstrom: float = 0.529177210903
    be_h_ev: float = 13.605693122994
    ev_to_kj_mol: float = 96.48533212
    k_b_ev_per_kelvin: float = 8.617333262e-5

    def length_a0_to_angstrom(self, length_in_a0: float) -> float:
        """Convert dimensionless Bohr units a_0 to Angstroms."""
        return length_in_a0 * self.a_0_angstrom

    def length_dh_to_angstrom(self, length_in_dh: float, c: MCarrierConstants = M_CARRIER) -> float:
        """Convert native Planck lattice steps dh to Angstroms via N_a0."""
        return (length_in_dh / c.a_0_dh) * self.a_0_angstrom

    def energy_be_h_to_ev(self, ratio_to_be_h: float) -> float:
        """Convert dimensionless BE_H ratio to electron-volts (eV)."""
        return ratio_to_be_h * self.be_h_ev

    def energy_e0_to_ev(self, energy_in_e0: float, c: MCarrierConstants = M_CARRIER) -> float:
        """Convert native cell energy units E_0 = s_0 / dt to eV via BE_H."""
        return (energy_in_e0 / c.be_h_e0) * self.be_h_ev

    def quanta_ehb_to_ev(self, quanta: float, c: MCarrierConstants = M_CARRIER) -> float:
        """Convert E_HB quanta to eV."""
        return quanta * (self.be_h_ev / c.n_hb)

    def quanta_ehb_to_kj_mol(self, quanta: float, c: MCarrierConstants = M_CARRIER) -> float:
        """Convert E_HB quanta to kJ/mol."""
        return self.quanta_ehb_to_ev(quanta, c=c) * self.ev_to_kj_mol

    def ionization_energy_ev(self, z: int, c: MCarrierConstants = M_CARRIER) -> float:
        """Project M-layer ionization ratio IE_1(Z)/BE_H to eV."""
        return m_ionization_ratio_be_h(z, c=c) * self.be_h_ev

    def covalent_radius_angstrom(self, z: int, bond_order: int = 1, c: MCarrierConstants = M_CARRIER) -> float:
        """Project M-layer covalent radius r_cov(Z)/a_0 to Angstroms."""
        return m_covalent_radius_in_a0(z, bond_order=bond_order, c=c) * self.a_0_angstrom

    def sp3_angle_deg(self, lone_pairs: int = 0, c: MCarrierConstants = M_CARRIER) -> float:
        """Project M-layer K_4 bond angle in radians to degrees."""
        return math.degrees(m_sp3_bond_angle_rad(lone_pairs, c=c))

    def water_thermodynamics_si(self, c: MCarrierConstants = M_CARRIER) -> dict[str, float]:
        """Project M-layer tetrahedral H-bond phase ratios to SI (eV, kJ/mol, Kelvin)."""
        ratios = m_compute_hbond_phase_ratios(c=c)
        e_hb_ev = self.be_h_ev / ratios.n_hb_divisor
        e_hb_kj = e_hb_ev * self.ev_to_kj_mol
        t_hb_k = e_hb_ev / self.k_b_ev_per_kelvin
        return {
            "e_hb_ev": e_hb_ev,
            "e_hb_kj_mol": e_hb_kj,
            "t_hb_kelvin": t_hb_k,
            "delta_h_vap_kj_mol": ratios.vap_in_ehb_quanta * e_hb_kj,
            "delta_h_fus_kj_mol": ratios.fus_in_ehb_quanta * e_hb_kj,
            "t_melt_kelvin": ratios.t_melt_over_t_hb * t_hb_k,
            "t_boil_kelvin": ratios.t_boil_over_t_hb * t_hb_k,
            "t_crit_kelvin": ratios.t_crit_over_t_hb * t_hb_k,
            "bcs_gap_ratio": ratios.bcs_gap_ratio,
        }


T_PROJECTOR = TSIBridgeProjector()


def format_polypeptide_xray_table(
    folded: MFoldedChain,
    proj: TSIBridgeProjector = T_PROJECTOR,
    c: MCarrierConstants = M_CARRIER,
) -> str:
    """Format an X-Ray bond trace of an M-layer folded chain in both M-units and T-layer SI units."""
    seq_str = decode_z21_to_protein(folded.class_sequence)
    tot_kj = -proj.quanta_ehb_to_kj_mol(folded.total_quanta_ehb, c=c)
    lines = [
        f"=== X-RAY BOND TRACE (L={len(folded.class_sequence)}, bonds={len(folded.bond_records)}, "
        f"E = -{folded.total_quanta_ehb} E_HB = {folded.total_energy_e0:.3e} E_0 = {tot_kj:.2f} kJ/mol) ===",
        f"Sequence (T-projection): {seq_str}",
        f"{'Bond Kind':26s} | {'Pair (T)':10s} | {'d / a_0':9s} | {'d (A)':9s} | {'Quanta':8s} | {'Energy (T)'}",
        "-" * 92,
    ]
    for b in folded.bond_records:
        kind_str = T_BOND_KIND_NAMES[b.bond_kind]
        pair_str = f"{T_Z21_TO_AA1[b.class_i]}{b.residue_i+1}<->{T_Z21_TO_AA1[b.class_j]}{b.residue_j+1}"
        d_ang = proj.length_a0_to_angstrom(b.distance_a0)
        e_kj = -proj.quanta_ehb_to_kj_mol(b.quanta_ehb, c=c)
        lines.append(
            f"{kind_str:26s} | {pair_str:10s} | {b.distance_a0:7.2f}   | {d_ang:7.2f} A | "
            f"{b.quanta_ehb:2d} E_HB  | {e_kj:8.2f} kJ/mol"
        )
    return "\n".join(lines)


def format_tissue_xray_table(
    morph: MTissueMorphology,
    proj: TSIBridgeProjector = T_PROJECTOR,
    c: MCarrierConstants = M_CARRIER,
) -> str:
    """Format an X-Ray trace of an M-layer 3D tissue structure in both M-units and T-layer SI units."""
    tot_kj = -proj.quanta_ehb_to_kj_mol(morph.total_quanta_ehb, c=c)
    lines = [
        f"=== TISSUE MORPHOLOGY X-RAY TRACE (code={morph.morphology_code}, laminae={morph.n_laminae}, "
        f"cells={len(morph.living_cells)}, divisions={len(morph.division_lineage)}, "
        f"cavities={len(morph.lumen_nodes)}) ===",
        f"FCC N12 contacts: {len(morph.adhesion_network)} (Ig locks: {len(morph.ig_synaptic_network)}) | "
        f"E = -{morph.total_quanta_ehb} E_HB = {morph.total_energy_e0:.3e} E_0 = {tot_kj:.2f} kJ/mol",
        "-" * 92,
    ]
    lam_counts: dict[int, int] = {}
    for cell in morph.living_cells:
        lam_counts[cell.lamina_index] = lam_counts.get(cell.lamina_index, 0) + 1
    for lam, cnt in sorted(lam_counts.items()):
        lines.append(f"  lamina k={lam:2d} : {cnt:3d} cells")

    if morph.ig_synaptic_network:
        lines.append("-" * 92)
        for syn in morph.ig_synaptic_network[:6]:
            e_kj = -proj.quanta_ehb_to_kj_mol(syn.quanta_ehb, c=c)
            lines.append(
                f"  cell #{syn.cell_a_id:2d} (k={syn.lamina_a}) {syn.pos_a_ijk} "
                f"<--> cell #{syn.cell_b_id:2d} (k={syn.lamina_b}) {syn.pos_b_ijk} | "
                f"e_FCC={syn.fcc_link_vector} | {syn.quanta_ehb} E_HB ({e_kj:.2f} kJ/mol)"
            )
    return "\n".join(lines)


def scan_fasta_and_format_xray(
    fasta_dna: str,
    min_orf_codons: int = 8,
    proj: TSIBridgeProjector = T_PROJECTOR,
    c: MCarrierConstants = M_CARRIER,
) -> tuple[MChromosomeScanResult, str]:
    """Execute FASTA (T) -> Z_4 (M) -> Splicing + O(1) Fold + Locks (M) -> T(FASTA) + Blind Spots report."""
    clean_dna = "".join(
        line.strip().upper()
        for line in fasta_dna.splitlines()
        if not line.strip().startswith(">")
    )
    codes_z4 = encode_dna_to_z4(clean_dna)
    scan = m_scan_chromosome_de_novo(codes_z4, min_orf_codons=min_orf_codons, c=c)

    lines = [
        f"=== DE NOVO FASTA -> M -> T(FASTA) X-RAY SCAN ({scan.genome_length_bp} bp) ===",
        f"Open 147-bp chromatin windows: {scan.open_chromatin_bp_count} / {scan.genome_length_bp} bp "
        f"| Discovered ORFs: {len(scan.expressed_products)}",
        f"Spliced intron-rescued ORFs: {list(scan.spliced_loci_indices)} | "
        f"Lock-free ORFs (blind spots): {list(scan.unannotated_lock_free_loci)} | "
        f"Orphan promoters: {len(scan.orphan_promoter_positions)}",
        "-" * 96,
    ]
    for prod in scan.expressed_products:
        prot_str = decode_z21_to_protein(prod.class_sequence_z21)
        f = prod.folded_chain
        e_kj = -proj.quanta_ehb_to_kj_mol(f.total_quanta_ehb, c=c)
        locks_summary: list[str] = []
        if prod.spliced_introns:
            locks_summary.append(f"spliced_introns={len(prod.spliced_introns)}")
        if prod.c2h2_locks:
            locks_summary.append(f"C2H2(|N12|=12)={len(prod.c2h2_locks)}")
        if prod.trp_pin_locks:
            locks_summary.append(f"TrpPin={len(prod.trp_pin_locks)}")
        if prod.ig_locks:
            locks_summary.append(f"Ig(N_HB=63)={len(prod.ig_locks)}")
        if prod.catalytic_triads:
            locks_summary.append(f"CatalyticTriad={len(prod.catalytic_triads)}")
        if prod.atp_walker_locks:
            locks_summary.append(f"WalkerATP={len(prod.atp_walker_locks)}")
        if prod.transmembrane_rods:
            locks_summary.append(f"TM18={len(prod.transmembrane_rods)}")
        lock_str = ", ".join(locks_summary) if locks_summary else "BLIND_SPOT (no lock)"

        lines.append(
            f"Locus #{prod.locus_id:2d} | L={len(prot_str):3d} aa | "
            f"bonds={f.total_quanta_ehb:3d} E_HB ({e_kj:7.1f} kJ/mol) | "
            f"FCC_self_avoid={len(set(f.fcc_lattice_nodes)) == len(f.fcc_lattice_nodes)} | "
            f"Locks: {lock_str}"
        )
    if scan.intergenic_dark_spans:
        lines.append("-" * 96)
        lines.append(f"Intergenic non-coding spans (>=12 bp): {list(scan.intergenic_dark_spans)}")
    return scan, "\n".join(lines)
