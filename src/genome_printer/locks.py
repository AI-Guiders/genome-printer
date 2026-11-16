"""Universal DNA-binding transcription factor locks and Ig synaptic adhesion locks on FCC N12.

Implements Section 1.4 and Section 3.3 of Volume 6 (`biology`) of "Кипящий океан":
  1. Zinc Finger C2H2 Lock (`C-x(2,4)-C-x(12)-H-x(3,5)-H`):
     - Spaced by exactly |N_12| = 12 residues between Cys_2 and His_1,
     - Projects a recognition alpha-helix into the B-DNA Major Groove at relative positions
       (-1, +2, +3, +6) from the start of the helix, reading a 3-bp DNA subsite via exact
       H-bond fork pairing:
         Arg (+, +)       <-> G (-, -, +, 0)   [2 H-bonds]
         Gln/Asn (+, -)   <-> A (-, +, -, M)   [2 H-bonds]
         Asp/Glu (-, -)   <-> C (0, +, -, -)   [2 H-bonds]
         His (+ <-> -)    <-> G or T           [1 H-bond]
  2. Homeodomain / Forkhead / Helix-Turn-Helix Trp-Pin Lock (`W-x(1,2)-[N/Q]-x(0,2)-[R/K]`):
     - Anchored by the indole wedge of Tryptophan (W) in the hydrophobic core,
     - Presents Asn/Gln (+, -) and Arg/Lys (+) at the i + 4 = d_in(square) helical step
       to read the `TAAT` / `TGTTT` homeotic/forkhead core in the Major Groove.
  3. Synaptic Immunoglobulin Lock (`Ig_donor <-> Ig_acceptor`, Dpr <-> DIP):
     - Donor loop length L_loop = N_HB = 2^frac_bits - 1 = 63 residues (`[Cys_B .. Cys_F]`),
     - Inter-domain step Delta_Cys(donor) = N_shell(2) - 1 = 41 = Delta_phi_disc,
       Delta_Cys(acceptor) = N_shell(2) + 1 = 43 (sum = 2 * N_shell(2) = 84),
     - Seven cuboctahedral contact offsets from the conserved Trp (W) pin and Cys_F:
       {W + 3, W + 4, W + 6, W + 12, W + 13, Cys_F - 12, Cys_F - 5}.
"""

from __future__ import annotations

from dataclasses import dataclass
from .constants import CARRIER, CarrierConstants
from .dna import major_groove_barcode
from .genetic_code import AMINO_ACID_SPECS


@dataclass(frozen=True)
class C2H2FingerMatch:
    """Detected C2H2 zinc finger motif with |N_12| = 12 loop and its 3-bp DNA recognition key."""

    cys1_index: int
    cys2_index: int
    his1_index: int
    his2_index: int
    loop_12_sequence: str
    reader_residues: tuple[str, str, str]  # at helical positions (-1, +3, +6) -> (loop[11], loop[9], loop[6])
    predicted_dna_triplet: str


@dataclass(frozen=True)
class TrpPinMatch:
    """Detected Homeodomain / Forkhead / HTH Tryptophan-pin DNA reader lock."""

    trp_index: int
    amide_reader_index: int
    basic_reader_index: int
    motif_slice: str
    predicted_dna_core: str


@dataclass(frozen=True)
class IgSynapticLock:
    """Detected Immunoglobulin synaptic adhesion domain (Donor key or Acceptor socket)."""

    role: str  # 'donor' (Dpr-like, W-x-R ridge) or 'acceptor' (DIP-like, W-x-x-x-[D/E] groove)
    cys_b_index: int
    cys_f_index: int
    loop_length: int  # Canonical donor = N_HB = 63
    trp_pin_index: int
    contact_signature: tuple[str, ...]  # 7 cuboctahedral offsets


def _amino_acid_read_base(aa: str) -> str:
    """Map a Major Groove reader sidechain to its complementary DNA base via H-bond fork."""
    if aa in ("R", "K"):
        return "G"  # [+, +] guanidinium pairs with G [-, -, +, 0]
    if aa in ("Q", "N"):
        return "A"  # [+, -] amide pairs with A [-, +, -, M]
    if aa in ("D", "E"):
        return "C"  # [-, -] carboxylate pairs with C [0, +, -, -]
    if aa == "H":
        return "G"
    return "T"  # Hydrophobic/methyl contact


def scan_c2h2_locks(protein_seq: str, c: CarrierConstants = CARRIER) -> list[C2H2FingerMatch]:
    """Scan a protein sequence for C2H2 zinc-finger locks with exact |N_12| = 12 spacing."""
    seq = protein_seq.strip().upper()
    n = len(seq)
    matches: list[C2H2FingerMatch] = []

    for i in range(n):
        if seq[i] != "C":
            continue
        for cys_gap in (2, 4):
            j = i + 1 + cys_gap
            if j >= n or seq[j] != "C":
                continue
            # Exact cuboctahedral coordination star |N_12| = 12 residues between Cys_2 and His_1
            h1 = j + 1 + c.n_12
            if h1 >= n or seq[h1] != "H":
                continue
            for his_gap in (3, 4, 5):
                h2 = h1 + 1 + his_gap
                if h2 < n and seq[h2] == "H":
                    loop_12 = seq[j + 1 : h1]
                    # Recognition helix positions (-1, +3, +6) within the 12-residue loop
                    readers = (loop_12[11], loop_12[8], loop_12[5])
                    dna_triplet = "".join(_amino_acid_read_base(r) for r in readers)
                    matches.append(
                        C2H2FingerMatch(
                            cys1_index=i,
                            cys2_index=j,
                            his1_index=h1,
                            his2_index=h2,
                            loop_12_sequence=loop_12,
                            reader_residues=readers,
                            predicted_dna_triplet=dna_triplet,
                        )
                    )
                    break
    return matches


def scan_trp_pin_locks(protein_seq: str) -> list[TrpPinMatch]:
    """Scan a protein sequence for Homeodomain / Forkhead / HTH Tryptophan-pin DNA locks."""
    seq = protein_seq.strip().upper()
    n = len(seq)
    matches: list[TrpPinMatch] = []

    for i in range(n - 3):
        if seq[i] != "W":
            continue
        # Look for Asn/Gln within 1..4 steps and Arg/Lys within 2..6 steps (helical i -> i+4 turn)
        amide_idx = -1
        basic_idx = -1
        for d1 in (1, 2, 3, 4):
            if i + d1 < n and seq[i + d1] in ("N", "Q"):
                amide_idx = i + d1
                break
        for d2 in (2, 3, 4, 5, 6):
            if i + d2 < n and seq[i + d2] in ("R", "K"):
                basic_idx = i + d2
                break
        if amide_idx != -1 and basic_idx != -1:
            has_asn = any(seq[i + d1] == "N" for d1 in (1, 2, 3, 4) if i + d1 < n)
            core = "TAAT" if has_asn else "TGTTT"
            matches.append(
                TrpPinMatch(
                    trp_index=i,
                    amide_reader_index=amide_idx,
                    basic_reader_index=basic_idx,
                    motif_slice=seq[i : max(amide_idx, basic_idx) + 1],
                    predicted_dna_core=core,
                )
            )
    return matches


def scan_ig_synaptic_locks(protein_seq: str, c: CarrierConstants = CARRIER) -> list[IgSynapticLock]:
    """Scan a protein sequence for Immunoglobulin synaptic adhesion domains (Donor / Acceptor)."""
    seq = protein_seq.strip().upper()
    n = len(seq)
    cys_positions = [idx for idx, ch in enumerate(seq) if ch == "C"]
    locks: list[IgSynapticLock] = []

    for a in range(len(cys_positions)):
        cb = cys_positions[a]
        for b in range(a + 1, len(cys_positions)):
            cf = cys_positions[b]
            span = cf - cb
            # Ig domain disulfide loop span around N_HB = 63 (typically 50..72 residues)
            if not (c.n_shell_2 + 8 <= span <= c.n_hb + c.n_12):
                continue
            # Conserved Tryptophan pin inside [Cys_B + 10 .. Cys_B + 22]
            trp_idx = -1
            for k in range(cb + 8, min(cf, cb + 24)):
                if seq[k] == "W":
                    trp_idx = k
                    break
            if trp_idx == -1:
                continue

            # Determine Donor (W-x-R ridge, [+,+]) vs Acceptor (W-x-x-x-[D/E] groove, [-,-])
            role = "ig_domain"
            if trp_idx + 2 < n and seq[trp_idx + 2] in ("R", "K"):
                role = "donor"
            elif trp_idx + 4 < n and seq[trp_idx + 4] in ("D", "E"):
                role = "acceptor"

            # Extract the 7 cuboctahedral contact offsets:
            # {W + d(3), W + d_in(sq)(4), W + |sq|(6), W + |N_12|(12), W + N_phi(13), Cys_F - |N_12|(12), Cys_F - d_out(E)(5)}
            offsets = (
                trp_idx + c.d,
                trp_idx + c.d_in_sq,
                trp_idx + c.n_squares,
                trp_idx + c.n_12,
                trp_idx + c.n_phi,
                cf - c.n_12,
                cf - c.d_out_e,
            )
            sig = tuple(seq[idx] if 0 <= idx < n else "-" for idx in offsets)
            locks.append(
                IgSynapticLock(
                    role=role,
                    cys_b_index=cb,
                    cys_f_index=cf,
                    loop_length=span,
                    trp_pin_index=trp_idx,
                    contact_signature=sig,
                )
            )
    return locks


def score_ig_synaptic_pair_ev(
    donor: IgSynapticLock,
    acceptor: IgSynapticLock,
    c: CarrierConstants = CARRIER,
) -> float:
    """Compute binding free energy Delta_G_bind in eV between an Ig donor and Ig acceptor on {111}."""
    delta_n_hb = 0
    for aa_d, aa_a in zip(donor.contact_signature, acceptor.contact_signature):
        if aa_d == "-" or aa_a == "-":
            continue
        fork_d = AMINO_ACID_SPECS[aa_d].hb_fork
        fork_a = AMINO_ACID_SPECS[aa_a].hb_fork
        # Opposite polarities (+1 and -1) form H-bonds / salt bridges; hydrophobic (0 and 0) pack in {111}
        polar_pairs = sum(1 for s1 in fork_d for s2 in fork_a if s1 * s2 == -1)
        polar_clashes = sum(1 for s1 in fork_d for s2 in fork_a if s1 * s2 == +1)
        hydrophobic_contacts = 1 if (0 in fork_d and 0 in fork_a and polar_clashes == 0) else 0
        delta_n_hb += polar_pairs + hydrophobic_contacts - polar_clashes

    return -delta_n_hb * c.e_hb_ev


def score_tf_dna_binding_ev(
    reader_residues: tuple[str, ...],
    dna_subsite: str,
    c: CarrierConstants = CARRIER,
) -> float:
    """Compute Major Groove binding energy Delta_G in eV between reader sidechains and a DNA subsite."""
    barcodes = major_groove_barcode(dna_subsite)
    if len(reader_residues) != len(barcodes):
        raise ValueError("Length of reader_residues must match length of dna_subsite.")

    total_hb = 0
    for aa, bc in zip(reader_residues, barcodes):
        fork = AMINO_ACID_SPECS[aa.upper()].hb_fork
        # Count complementary pairings between sidechain fork (+1/-1/0) and 4-slot barcode ('+', '-', 'M', '0')
        for s in fork:
            if s == +1 and "-" in bc:
                total_hb += 1
            elif s == -1 and "+" in bc:
                total_hb += 1
            elif s == 0 and "M" in bc:
                total_hb += 1
    return -total_hb * c.e_hb_ev
