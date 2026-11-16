"""M-layer nucleic acid register (Z_4 tiles, Z_64 triplets, Z_21 classes, and Rumer bit-inversion).

Zero T-layer strings, zero IUPAC letters ('A','T','G','C'), zero amino acid names.
On M, a nucleobase is a 2-bit integer `b in {0, 1, 2, 3}`:
  - bit 1 (b >> 1): 0 = pyrimidine (1 ring), 1 = purine (2 rings);
  - bit 0 (b & 1):  keto/amino parity, such that Rumer involution is bitwise NOT `b ^ 3`
                    and Watson-Crick complement is `b ^ 2`!
Specifically:
  b = 0 (0b00): frontal tracks (-1, +1, -1)  [2-bond pyrimidine]
  b = 1 (0b01): frontal tracks (+1, -1, -1)  [3-bond pyrimidine]
  b = 2 (0b10): frontal tracks (+1, -1,  0)  [2-bond purine, pairs with b ^ 2 = 0]
  b = 3 (0b11): frontal tracks (-1, +1, +1)  [3-bond purine, pairs with b ^ 2 = 1]

Notice how on M:
  - Watson-Crick complement of `b` is `b ^ 0b10` (flips purine/pyrimidine bit, preserves H-bond count)!
  - Rumer antisymmetry transform of `b` is `b ^ 0b11` (flips both bits: 0 <-> 3, 1 <-> 2)!
  - A dinucleotide stem `s = (b1 << 2) | b2 in Z_16` transforms under Rumer as `s ^ 0b1111`!
  - A triplet codon `c = (b1 << 4) | (b2 << 2) | b3 in Z_64` maps deterministically into
    one of `N_shell(2) / 2 = 21` classes `a in {0, ..., 20}`!
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from .m_carrier import M_CARRIER, MCarrierConstants


# Frontal d = 3 Watson-Crick H-bond track signs (+1 donor, -1 acceptor, 0 neutral) indexed by b in 0..3
M_FRONTAL_TRACKS: tuple[tuple[int, int, int], ...] = (
    (-1, +1, -1),  # b = 0
    (+1, -1, -1),  # b = 1
    (+1, -1, 0),   # b = 2
    (-1, +1, +1),  # b = 3
)

# Rear |K_4| = 4 Major Groove slot codes (-1 acceptor, +1 donor, +2 methyl M, 0 neutral H)
# indexed by coding base b in 0..3 (paired with its Watson-Crick complement b ^ 2)
M_MAJOR_GROOVE_SLOTS: tuple[tuple[int, int, int, int], ...] = (
    (+2, -1, +1, -1),  # pair (0, 2)
    (0, +1, -1, -1),   # pair (1, 3)
    (-1, +1, -1, +2),  # pair (2, 0)
    (-1, -1, +1, 0),   # pair (3, 1)
)

# 8 4-bit stems of Rumer Octet I (4-fold degenerate): s = (b1 << 2) | b2 in Z_16
# b=0 (U/T), b=1 (C), b=2 (A), b=3 (G):
# GC(13=0b1101), GG(15=0b1111), CC(5=0b0101), CU(4=0b0100),
# CG(7=0b0111), GU(12=0b1100), UC(1=0b0001), AC(9=0b1001)
M_RUMER_OCTET_I_STEMS: frozenset[int] = frozenset({1, 4, 5, 7, 9, 12, 13, 15})

# 8 4-bit stems of Rumer Octet II (split by 3rd base): bitwise complement s ^ 0b1111 of Octet I!
M_RUMER_OCTET_II_STEMS: frozenset[int] = frozenset({0, 2, 3, 6, 8, 10, 11, 14})


def m_complement_base(b: int) -> int:
    """Watson-Crick complement on Z_4: flips the purine/pyrimidine bit `b ^ 2`."""
    return (b & 3) ^ 2


def m_rumer_transform_base(b: int) -> int:
    """Rumer involution on Z_4: bitwise complement `b ^ 3` (0<->3, 1<->2)."""
    return (b & 3) ^ 3


def m_rumer_transform_stem(stem_4bit: int) -> int:
    """Rumer involution on a 4-bit dinucleotide stem in Z_16: `stem_4bit ^ 0xF`."""
    return (stem_4bit & 0xF) ^ 0xF


def m_watson_crick_hbond_quanta(b1: int, b2: int) -> int:
    """Count integer E_HB quanta between two 2-bit bases across the d = 3 frontal tracks.

    Returns +3 for (1,3)/(3,1), +2 for (0,2)/(2,0), and negative for repulsive clashes.
    """
    t1 = M_FRONTAL_TRACKS[b1 & 3]
    t2 = M_FRONTAL_TRACKS[b2 & 3]
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


# 64-entry tuple mapping 6-bit codon `c = (b1 << 4) | (b2 << 2) | b3 in Z_64` to class `a in 0..20`
# Class indices 0..20 (21 = N_shell(2)/2 classes on the 2nd coordination shell):
#   0..2:   Octet I 3_square bridge (6 codons each)
#   3..7:   Octet I V/E 4-fold classes (4 codons each)
#   8..19:  Octet II 12_E/V split classes
#   20:     1_core termination class (Stop, 3 codons)
M_CODON_TO_CLASS_21: tuple[int, ...] = (
    # b1 = 0 (00..15)
    8,  8,  0,  0,   # s=0  (0000..0011)
    1,  1,  1,  1,   # s=1  (0100..0111) -> Octet I
    9,  9,  20, 20,  # s=2
    11, 11, 20, 10,  # s=3
    # b1 = 1 (16..31)
    0,  0,  0,  0,   # s=4  -> Octet I
    5,  5,  5,  5,   # s=5  -> Octet I
    14, 14, 15, 15,  # s=6
    2,  2,  2,  2,   # s=7  -> Octet I
    # b1 = 2 (32..47)
    13, 13, 13, 12,  # s=8
    7,  7,  7,  7,   # s=9  -> Octet I
    16, 16, 17, 17,  # s=10
    1,  1,  2,  2,   # s=11
    # b1 = 3 (48..63)
    4,  4,  4,  4,   # s=12 -> Octet I
    3,  3,  3,  3,   # s=13 -> Octet I
    18, 18, 19, 19,  # s=14
    6,  6,  6,  6,   # s=15 -> Octet I
)


def m_translate_codons(base_codes: tuple[int, ...] | list[int], stop_at_20: bool = True) -> tuple[int, ...]:
    """Translate a sequence of 2-bit bases `b in Z_4` into a sequence of class indices `a in Z_21`."""
    out: list[int] = []
    n = len(base_codes)
    for i in range(0, n - 2, 3):
        c6 = ((base_codes[i] & 3) << 4) | ((base_codes[i + 1] & 3) << 2) | (base_codes[i + 2] & 3)
        cls_id = M_CODON_TO_CLASS_21[c6]
        if cls_id == 20:
            if stop_at_20:
                break
            out.append(20)
        else:
            out.append(cls_id)
    return tuple(out)


@dataclass(frozen=True)
class MDuplexLatticeState:
    """M-layer B-DNA duplex state in native dh steps, a_0 ratios, and integer E_HB / E_0 units."""

    coding_bases: tuple[int, ...]
    template_bases: tuple[int, ...]
    major_groove_slots: tuple[tuple[int, int, int, int], ...]
    strand1_xyz_a0: np.ndarray  # shape (L, 3) in dimensionless Bohr units a_0
    strand2_xyz_a0: np.ndarray  # shape (L, 3) in dimensionless Bohr units a_0
    strand1_xyz_dh: np.ndarray  # shape (L, 3) in native lattice steps dh
    strand2_xyz_dh: np.ndarray  # shape (L, 3) in native lattice steps dh
    hbond_quanta_per_bp: tuple[int, ...]
    total_hbond_quanta: int
    total_energy_e0: float  # -total_hbond_quanta * E_HB in native E_0 = s_0 / dt units
    nucleosome_ball_fraction: float


def m_build_duplex(
    coding_bases: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> MDuplexLatticeState:
    """Construct M-layer duplex coordinates (in a_0 and dh units) and H-bond quanta from 2-bit bases."""
    codes = tuple(int(b) & 3 for b in coding_bases)
    n = len(codes)
    templates = tuple(m_complement_base(b) for b in codes)
    slots = tuple(M_MAJOR_GROOVE_SLOTS[b] for b in codes)
    quanta = tuple(m_watson_crick_hbond_quanta(b, t) for b, t in zip(codes, templates))

    r_a0 = 0.5 * c.bdna_width_in_a0_units
    h_a0 = c.bdna_stacking_step_in_a0_units
    twist = (2.0 * math.pi) / c.bdna_bp_per_turn
    offset = (2.0 * math.pi) / c.d

    s1 = np.zeros((n, 3), dtype=np.float64)
    s2 = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        p1 = i * twist
        p2 = p1 + offset
        z = i * h_a0
        s1[i] = [r_a0 * math.cos(p1), r_a0 * math.sin(p1), z]
        s2[i] = [r_a0 * math.cos(p2), r_a0 * math.sin(p2), z]

    tot_q = sum(quanta)
    return MDuplexLatticeState(
        coding_bases=codes,
        template_bases=templates,
        major_groove_slots=slots,
        strand1_xyz_a0=s1,
        strand2_xyz_a0=s2,
        strand1_xyz_dh=s1 * c.a_0_dh,
        strand2_xyz_dh=s2 * c.a_0_dh,
        hbond_quanta_per_bp=quanta,
        total_hbond_quanta=tot_q,
        total_energy_e0=-tot_q * c.e_hb_e0,
        nucleosome_ball_fraction=n / float(c.n_ball_3),
    )


@dataclass(frozen=True)
class MIntronSplicingSpan:
    """M-layer spliced intron lariat span on a 2-bit Z_4 transcript.

    Physical splicing gate on Z_4:
      - 5' donor stem: `s_donor = (3 << 2) | 0 = 12` (`(3, 0)` in Z_4, i.e., `GU`),
      - 3' acceptor stem: `s_acceptor = (2 << 2) | 3 = 11` (`(2, 3)` in Z_4, i.e., `AG`),
      - Branch-point adenosine `b = 2` preceded by a 2-bond pyrimidine tract (`b in {0, 1}`)
        located within the cuboctahedral shell window `[j - N_shell(2), j - N_shell(1)] = [j - 42, j - 12]`
        upstream of the 3' acceptor `j`,
      - Frame-rescue condition: excising `[i .. j + 1]` removes an in-frame termination
        triplet (`class 20`) that would otherwise truncate the open reading frame!
    """

    donor_start_idx: int
    branch_point_idx: int
    acceptor_end_idx: int  # Inclusive index of the final base `3` of the (2, 3) acceptor
    excised_length: int


def m_splice_pre_mrna(
    bases_z4: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> tuple[tuple[int, ...], tuple[MIntronSplicingSpan, ...]]:
    """Deterministic M-layer lariat splicing of a 2-bit Z_4 transcript."""
    seq = tuple(int(b) & 3 for b in bases_z4)
    n = len(seq)
    spliced: list[int] = []
    spans: list[MIntronSplicingSpan] = []

    i = 0
    while i < n:
        # Check for 5' donor dinucleotide (3, 0) [stem 12]
        if i + 1 < n and seq[i] == 3 and seq[i + 1] == 0:
            found_intron: MIntronSplicingSpan | None = None
            # Search for 3' acceptor (2, 3) [stem 11] within [i + N_shell(2) - 1 .. i + N_ring]
            min_end = i + c.n_shell_2 - 1
            max_end = min(n - 1, i + c.n_ring)
            for j in range(min_end, max_end):
                if seq[j - 1] == 2 and seq[j] == 3:
                    # Check if the candidate intron [i .. j] contains at least one IN-FRAME stop codon (class 20)
                    has_stop = False
                    for k in range(i, j - 2):
                        if (k % 3) != 0:
                            continue
                        c6 = (seq[k] << 4) | (seq[k + 1] << 2) | seq[k + 2]
                        if M_CODON_TO_CLASS_21[c6] == 20:
                            has_stop = True
                            break
                    if not has_stop:
                        continue
                    # Locate branch-point base `2` within [j - 42 .. j - 12] followed by a pure pyrimidine tract (0 or 1)
                    bp_idx = -1
                    w_lo = max(i + 6, j - c.n_shell_2)
                    w_hi = max(w_lo, j - c.n_shell_1)
                    for b_pos in range(w_lo, w_hi + 1):
                        if seq[b_pos] == 2:
                            tract = seq[b_pos + 1 : j - 1]
                            if len(tract) >= c.n_12 and all(b_tr in (0, 1) for b_tr in tract):
                                bp_idx = b_pos
                                break
                    if bp_idx != -1:
                        found_intron = MIntronSplicingSpan(i, bp_idx, j, j - i + 1)
                        break
            if found_intron is not None:
                spans.append(found_intron)
                i = found_intron.acceptor_end_idx + 1
                continue

        spliced.append(seq[i])
        i += 1

    return tuple(spliced), tuple(spans)


def m_nucleosome_accessibility_profile(
    bases_z4: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> tuple[int, ...]:
    """Compute 147-bp (N_ball(3) = 147) chromatin accessibility flag (1=open, 0=wrapped) per base.

    A 147-bp window wraps into a closed nucleosome core particle unless its average
    H-bond track stiffness exceeds 2.5 E_HB/bp (3-bond GC islands) or contains a rigid
    homopolymer tract of length >= d_out(V) = 7 that resists {111} superhelical bending.
    """
    seq = tuple(int(b) & 3 for b in bases_z4)
    n = len(seq)
    win = c.n_ball_3  # 147
    if n < win:
        return tuple(1 for _ in range(n))

    open_flags = [0] * n
    for start in range(0, n - win + 1, c.bdna_bp_per_turn):
        end = start + win
        sub = seq[start:end]
        # Count 3-bond pairs (b in {1, 3}) and max homopolymer run
        three_bond_count = sum(1 for b in sub if b in (1, 3))
        max_run = 1
        cur_run = 1
        for k in range(1, win):
            if sub[k] == sub[k - 1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1
        # Open promoter window if 3-bond fraction > 2/3 or homopolymer run >= d_out(V) = 7
        if three_bond_count * 3 >= 2 * win or max_run >= c.d_out_v:
            for k in range(start, end):
                open_flags[k] = 1

    return tuple(open_flags)
