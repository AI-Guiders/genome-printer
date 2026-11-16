"""B-DNA Watson-Crick digital lock, Major Groove 4-slot barcode, and 147 bp nucleosome geometry.

Implements Chapter 1 of Volume 6 (`biology`) of "Кипящий океан":
  - 3 frontal Watson-Crick H-bond tracks (d = 3) with signs (+1 donor, -1 acceptor, 0 neutral):
      G = (-1, +1, +1) <-> C = (+1, -1, -1)  -> 3 * E_HB
      A = (+1, -1,  0) <-> T = (-1, +1, -1)  -> 2 * E_HB
  - 4 rear Major Groove slots (|K_4| = 4) exposed for O(1) transcription-factor readout without
    unwinding the duplex:
      A-T -> (-, +, -, M)
      T-A -> (M, -, +, -)
      G-C -> (-, -, +, 0)
      C-G -> (0, +, -, -)
  - Exact 3D B-DNA double-helix coordinates on the {111} slice:
      stacking step h_stack = 9 * kappa * a_0 (~3.368 A),
      turn period n_turn = |N_12| - 2 = 10 bp (36 deg per bp),
      glycosidic width R(C1'-C1') = 29 * kappa * a_0 (~10.85 A),
  - Nucleosome core particle capacity N_nuc = N_ball(3) = 1 + 12 + 42 + 92 = 147 bp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from .constants import CARRIER, CarrierConstants


# Frontal Watson-Crick 3-track H-bond polarity vectors (+1 donor -NH, -1 acceptor =O/=N-, 0 non-polar -CH)
WATSON_CRICK_TRACKS: dict[str, tuple[int, int, int]] = {
    "G": (-1, +1, +1),
    "C": (+1, -1, -1),
    "A": (+1, -1, 0),
    "T": (-1, +1, -1),
    "U": (-1, +1, -1),
}

# Rear Major Groove 4-slot barcode (|K_4| = 4 slots: '-', '+', 'M' thymine methyl, '0' cytosine C5-H)
MAJOR_GROOVE_BARCODE: dict[tuple[str, str], tuple[str, str, str, str]] = {
    ("A", "T"): ("-", "+", "-", "M"),
    ("T", "A"): ("M", "-", "+", "-"),
    ("G", "C"): ("-", "-", "+", "0"),
    ("C", "G"): ("0", "+", "-", "-"),
}

COMPLEMENT_BASE: dict[str, str] = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "U": "A",
}


def watson_crick_hbond_count(base1: str, base2: str) -> int:
    """Count complementary H-bonds between two bases across the d = 3 frontal tracks.

    Returns 3 for G-C/C-G, 2 for A-T/T-A/A-U/U-A, and negative (repulsive mismatch)
    when like-sign polar tracks (+/+ or -/-) collide.
    """
    t1 = WATSON_CRICK_TRACKS[base1.upper()]
    t2 = WATSON_CRICK_TRACKS[base2.upper()]
    bonds = 0
    clashes = 0
    for s1, s2 in zip(t1, t2):
        if s1 * s2 == -1:
            bonds += 1
        elif s1 * s2 == +1:
            clashes += 1
    if clashes > 0:
        return -clashes
    return bonds


def watson_crick_pair_energy_ev(base1: str, base2: str, c: CarrierConstants = CARRIER) -> float:
    """Pairing energy in eV: -n_HB * E_HB for valid pairs, +clashes * E_HB for mismatches."""
    count = watson_crick_hbond_count(base1, base2)
    return -count * c.e_hb_ev


def major_groove_barcode(coding_seq: str) -> list[tuple[str, str, str, str]]:
    """Return the sequence of 4-slot Major Groove barcodes along a double-stranded DNA segment."""
    barcodes: list[tuple[str, str, str, str]] = []
    for b in coding_seq.upper():
        if b not in COMPLEMENT_BASE or b == "U":
            raise ValueError(f"Expected DNA base in A,T,G,C; got {b!r}.")
        pair = (b, COMPLEMENT_BASE[b])
        barcodes.append(MAJOR_GROOVE_BARCODE[pair])
    return barcodes


@dataclass(frozen=True)
class BDNADuplexGeometry:
    """3D coordinates and physical metrics of a B-DNA double helix segment."""

    sequence: str
    strand1_c1_xyz: np.ndarray  # shape (L, 3) in Angstroms
    strand2_c1_xyz: np.ndarray  # shape (L, 3) in Angstroms
    total_hbonds: int
    total_energy_ev: float
    total_energy_kj_mol: float
    length_angstrom: float
    nucleosome_units: float


def build_bdna_duplex(sequence: str, c: CarrierConstants = CARRIER) -> BDNADuplexGeometry:
    """Construct exact 3D glycosidic C1' coordinates and H-bond energetics of a B-DNA duplex."""
    seq = sequence.strip().upper()
    n = len(seq)
    if n == 0:
        raise ValueError("DNA sequence must be non-empty.")

    r_duplex = 0.5 * c.bdna_width_c1_c1_angstrom
    h_step = c.bdna_stacking_step_angstrom
    twist_rad = (2.0 * math.pi) / c.bdna_bp_per_turn  # 36 deg = pi/5 rad
    # B-DNA glycosidic bond offset across the minor/major groove asymmetry (~120 deg = 2*pi/d)
    strand_offset_rad = (2.0 * math.pi) / c.d

    strand1 = np.zeros((n, 3), dtype=np.float64)
    strand2 = np.zeros((n, 3), dtype=np.float64)
    total_hb = 0

    for i, base in enumerate(seq):
        comp = COMPLEMENT_BASE[base]
        hb = watson_crick_hbond_count(base, comp)
        if hb < 0:
            raise ValueError(f"Invalid Watson-Crick pair at index {i}: {base}-{comp}")
        total_hb += hb

        phi1 = i * twist_rad
        phi2 = phi1 + strand_offset_rad
        z = i * h_step

        strand1[i] = [r_duplex * math.cos(phi1), r_duplex * math.sin(phi1), z]
        strand2[i] = [r_duplex * math.cos(phi2), r_duplex * math.sin(phi2), z]

    energy_ev = -total_hb * c.e_hb_ev
    return BDNADuplexGeometry(
        sequence=seq,
        strand1_c1_xyz=strand1,
        strand2_c1_xyz=strand2,
        total_hbonds=total_hb,
        total_energy_ev=energy_ev,
        total_energy_kj_mol=energy_ev * c.ev_to_kj_mol,
        length_angstrom=(n - 1) * h_step if n > 1 else 0.0,
        nucleosome_units=n / float(c.n_ball_3),
    )
