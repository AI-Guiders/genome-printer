"""M-layer polypeptide folding on K_4 grooves (Z_3) and structural lock recognition on Z_21.

Zero T-layer strings, zero amino acid letters, zero SI dimensional constants.
All residues are integer class indices `a in {0, ..., 20}` (`Z_21`, `N_shell(2)/2 = 21`),
all K_4 backbone grooves are integers `g in {0, 1, 2}` (`0`: 3.6_13 helix, `1`: {111} strand,
`2`: K_4 turn/hinge, `3`: termination), all energies are integer quanta of `E_HB` and
native `E_0 = s_0 / dt`, and all coordinates are in `a_0` and native `dh` steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from .m_carrier import M_CARRIER, MCarrierConstants
from .m_nucleic import M_MAJOR_GROOVE_SLOTS


@dataclass(frozen=True)
class MClass21Spec:
    """Pure M-layer specification of one of the 21 classes on the 2nd coordination shell."""

    class_id: int  # 0..20
    rumer_octet: int  # 1, 2, or 0 (termination)
    channel_id: int  # 0: 3_square, 1: V/E, 2: 12_E/V, 3: 1_core
    k4_groove: int  # 0 (helix 3.6_13), 1 ({111} strand), 2 (turn/hinge), 3 (stop)
    hb_fork: tuple[int, ...]  # +1 donor, -1 acceptor, 0 hydrophobic methyl/aryl branch
    planar_111_ring: int  # 1 if aromatic {111} N_6 ring, 0 otherwise
    period3_thiol: int  # 1 if Z=16 thiol bridge capable (class 11), 0 otherwise
    cyclic_hinge: int  # 1 if cyclic/glycine hinge (classes 5, 6), 0 otherwise


# All 21 classes of the second coordination shell (N_shell(2)/2 = 21), indexed 0..20
M_CLASS_21_SPECS: tuple[MClass21Spec, ...] = (
    # Octet I: 3_square bridge (0, 1, 2)
    MClass21Spec(0, 1, 0, 0, (0, 0), 0, 0, 0),
    MClass21Spec(1, 1, 0, 2, (+1, -1), 0, 0, 0),
    MClass21Spec(2, 1, 0, 0, (+1, +1), 0, 0, 0),
    # Octet I: V/E 4-fold classes (3, 4, 5, 6, 7)
    MClass21Spec(3, 1, 1, 0, (0,), 0, 0, 0),
    MClass21Spec(4, 1, 1, 1, (0, 0), 0, 0, 0),
    MClass21Spec(5, 1, 1, 2, (), 0, 0, 1),
    MClass21Spec(6, 1, 1, 2, (), 0, 0, 1),
    MClass21Spec(7, 1, 1, 1, (+1, -1, 0), 0, 0, 0),
    # Octet II: 12_E/V split classes (8..19)
    MClass21Spec(8, 2, 2, 1, (0, 0, 0), 1, 0, 0),
    MClass21Spec(9, 2, 2, 1, (0, 0, +1, -1), 1, 0, 0),
    MClass21Spec(10, 2, 2, 1, (0, 0, 0, +1), 1, 0, 0),
    MClass21Spec(11, 2, 2, 1, (0,), 0, 1, 0),
    MClass21Spec(12, 2, 2, 0, (0, 0), 0, 0, 0),
    MClass21Spec(13, 2, 2, 1, (0, 0), 0, 0, 0),
    MClass21Spec(14, 2, 2, 0, (+1, -1), 1, 0, 0),
    MClass21Spec(15, 2, 2, 0, (+1, -1), 0, 0, 0),
    MClass21Spec(16, 2, 2, 2, (+1, -1), 0, 0, 0),
    MClass21Spec(17, 2, 2, 0, (+1,), 0, 0, 0),
    MClass21Spec(18, 2, 2, 2, (-1, -1), 0, 0, 0),
    MClass21Spec(19, 2, 2, 0, (-1, -1), 0, 0, 0),
    # Core termination class (20)
    MClass21Spec(20, 0, 3, 3, (), 0, 0, 0),
)


@dataclass(frozen=True)
class MBondRecord:
    """M-layer record of a single closed bond in integer E_HB quanta and native coordinates."""

    bond_kind: int  # 0: alpha i->i-4, 1: beta i->i-2, 2: sidechain fork, 3: thiol S-S bridge
    residue_i: int
    class_i: int
    residue_j: int
    class_j: int
    quanta_ehb: int
    energy_e0: float
    distance_a0: float
    distance_dh: float


@dataclass(frozen=True)
class MFoldedChain:
    """M-layer result of O(1)-per-residue co-translational folding on K_4 grooves."""

    class_sequence: tuple[int, ...]
    k4_grooves: tuple[int, ...]  # 0, 1, or 2 per residue
    ca_xyz_a0: np.ndarray  # shape (L, 3) in Bohr units a_0
    ca_xyz_dh: np.ndarray  # shape (L, 3) in native lattice steps dh
    bond_records: tuple[MBondRecord, ...]
    thiol_bridges: tuple[tuple[int, int], ...]
    total_quanta_ehb: int
    total_energy_e0: float
    radius_of_gyration_a0: float
    elementary_ops_count: int  # Strictly <= 36 * L (proving O(L) in class P)


def _select_k4_groove_m(seq: tuple[int, ...], idx: int) -> tuple[int, int]:
    """Select K_4 groove in Z_3 (0, 1, 2) using O(1) local cuboctahedral window."""
    n = len(seq)
    cls_id = seq[idx]
    spec = M_CLASS_21_SPECS[cls_id]
    ops = 1

    if spec.cyclic_hinge == 1:
        return 2, ops + 1

    w_start = max(0, idx - 4)
    w_end = min(n, idx + 5)
    votes_0 = 0
    votes_1 = 0
    votes_2 = 0

    for j in range(w_start, w_end):
        ops += 1
        g = M_CLASS_21_SPECS[seq[j]].k4_groove
        if g == 0:
            votes_0 += 1
        elif g == 1:
            votes_1 += 1
        else:
            votes_2 += 1

    if idx >= 4 and M_CLASS_21_SPECS[seq[idx - 4]].cyclic_hinge == 0:
        ops += 1
        if spec.k4_groove == 0:
            votes_0 += 2

    if votes_2 > votes_0 and votes_2 > votes_1 and spec.k4_groove == 2:
        return 2, ops
    if votes_1 > votes_0:
        return 1, ops
    return 0, ops


def m_fold_chain_cotranslational(
    class_sequence: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> MFoldedChain:
    """Fold a sequence of Z_21 class indices in O(L) time with full M-layer bond tracing."""
    seq = tuple(int(a) for a in class_sequence if int(a) != 20)
    n = len(seq)

    grooves: list[int] = []
    ca_a0 = np.zeros((n, 3), dtype=np.float64)
    bonds: list[MBondRecord] = []
    total_ops = 0

    h_alpha = c.alpha_helix_rise_in_a0_units  # 4 * kappa
    twist_alpha = (2.0 * math.pi) / c.alpha_helix_residues_per_turn  # 5*pi/9
    r_alpha = c.n_squares * c.kappa  # 6 * kappa

    h_beta = c.beta_sheet_rise_in_a0_units  # (28/3) * kappa
    pleat_amp = c.kappa

    pos = np.zeros(3, dtype=np.float64)
    axis_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    binormal_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    helix_phase = 0.0
    strand_step = 0

    for i in range(n):
        g, ops = _select_k4_groove_m(seq, i)
        grooves.append(g)
        total_ops += ops

        if i == 0:
            ca_a0[0] = pos.copy()
            continue

        if g == 0:
            helix_phase += twist_alpha
            pos = pos + h_alpha * axis_dir
            offset = r_alpha * (math.cos(helix_phase) * normal_dir + math.sin(helix_phase) * binormal_dir)
            ca_a0[i] = pos + offset
            if i >= c.d_in_sq and grooves[i - c.d_in_sq] == 0 and M_CLASS_21_SPECS[seq[i]].cyclic_hinge == 0:
                j = i - c.d_in_sq
                dist_a0 = float(np.linalg.norm(ca_a0[i] - ca_a0[j]))
                bonds.append(
                    MBondRecord(0, i, seq[i], j, seq[j], 1, -c.e_hb_e0, dist_a0, dist_a0 * c.a_0_dh)
                )
                total_ops += 1
        elif g == 1:
            strand_step += 1
            sign = 1.0 if (strand_step % 2 == 0) else -1.0
            pos = pos + h_beta * axis_dir
            ca_a0[i] = pos + sign * pleat_amp * normal_dir
            if i >= 2 and grooves[i - 2] == 1:
                j = i - 2
                dist_a0 = float(np.linalg.norm(ca_a0[i] - ca_a0[j]))
                bonds.append(
                    MBondRecord(1, i, seq[i], j, seq[j], 1, -c.e_hb_e0, dist_a0, dist_a0 * c.a_0_dh)
                )
                total_ops += 1
        else:
            cos_t = -1.0 / c.d
            sin_t = math.sqrt(1.0 - cos_t * cos_t)
            new_axis = cos_t * axis_dir + sin_t * normal_dir
            new_normal = -sin_t * axis_dir + cos_t * normal_dir
            axis_dir = new_axis / np.linalg.norm(new_axis)
            normal_dir = new_normal / np.linalg.norm(new_normal)
            binormal_dir = np.cross(axis_dir, normal_dir)
            step_ca = (c.n_12 - 2) * c.kappa
            pos = pos + step_ca * axis_dir
            ca_a0[i] = pos.copy()

        # Sidechain H-bond fork pairing in the 12-neighbor window
        fork_i = M_CLASS_21_SPECS[seq[i]].hb_fork
        if any(s != 0 for s in fork_i):
            for prev in range(max(0, i - c.n_12), i):
                total_ops += 1
                fork_p = M_CLASS_21_SPECS[seq[prev]].hb_fork
                if any(s1 * s2 == -1 for s1 in fork_i for s2 in fork_p):
                    dist_a0 = float(np.linalg.norm(ca_a0[i] - ca_a0[prev]))
                    bonds.append(
                        MBondRecord(2, i, seq[i], prev, seq[prev], 1, -c.e_hb_e0, dist_a0, dist_a0 * c.a_0_dh)
                    )
                    break

    # Period-3 thiol bridges (class 11)
    thiol_pos = [idx for idx, cls_id in enumerate(seq) if M_CLASS_21_SPECS[cls_id].period3_thiol == 1]
    bridges: list[tuple[int, int]] = []
    used: set[int] = set()
    for a_i in range(len(thiol_pos)):
        i_c = thiol_pos[a_i]
        if i_c in used:
            continue
        for b_i in range(a_i + 1, len(thiol_pos)):
            j_c = thiol_pos[b_i]
            if j_c in used:
                continue
            span = j_c - i_c
            if span in (3, 5) or (c.n_shell_2 <= span <= c.n_hb + c.n_12):
                bridges.append((i_c, j_c))
                used.add(i_c)
                used.add(j_c)
                dist_a0 = float(np.linalg.norm(ca_a0[j_c] - ca_a0[i_c]))
                bonds.append(
                    MBondRecord(3, j_c, seq[j_c], i_c, seq[i_c], 2, -2.0 * c.e_hb_e0, dist_a0, dist_a0 * c.a_0_dh)
                )
                break

    tot_q = sum(b.quanta_ehb for b in bonds)
    center = np.mean(ca_a0, axis=0) if n > 0 else np.zeros(3)
    rg_a0 = float(math.sqrt(np.mean(np.sum((ca_a0 - center) ** 2, axis=1)))) if n > 0 else 0.0

    return MFoldedChain(
        class_sequence=seq,
        k4_grooves=tuple(grooves),
        ca_xyz_a0=ca_a0,
        ca_xyz_dh=ca_a0 * c.a_0_dh,
        bond_records=tuple(bonds),
        thiol_bridges=tuple(bridges),
        total_quanta_ehb=tot_q,
        total_energy_e0=-tot_q * c.e_hb_e0,
        radius_of_gyration_a0=rg_a0,
        elementary_ops_count=total_ops,
    )


@dataclass(frozen=True)
class MC2H2Lock:
    """M-layer C2H2 lock with |N_12| = 12 loop spacing between thiol_2 (11) and histidine_1 (14)."""

    thiol1_idx: int
    thiol2_idx: int
    his1_idx: int
    his2_idx: int
    loop_12_classes: tuple[int, ...]
    reader_classes: tuple[int, int, int]
    target_bases_z4: tuple[int, int, int]


@dataclass(frozen=True)
class MTrpPinLock:
    """M-layer indole wedge (class 10) + amide (15/16) + cationic (2/17) Major Groove lock."""

    wedge_idx: int
    amide_idx: int
    cation_idx: int
    target_core_z4: tuple[int, ...]


@dataclass(frozen=True)
class MIgLock:
    """M-layer immunoglobulin lock with loop span L_loop = N_HB = 63 and 7 cuboctahedral offsets."""

    polarity_role: int  # +1 for donor ridge (W+2 in {2,17}), -1 for acceptor groove (W+4 in {18,19}), 0 generic
    thiol_b_idx: int
    thiol_f_idx: int
    loop_span: int
    wedge_pin_idx: int
    contact_classes_7: tuple[int, ...]  # Classes at {W+3, W+4, W+6, W+12, W+13, Cys_F-12, Cys_F-5}


def _m_reader_class_to_base_z4(cls_id: int) -> int:
    """Map a Z_21 reader class to its complementary Z_4 base via H-bond fork."""
    if cls_id in (2, 17, 14):
        return 3  # [+1, +1] pairs with b=3 (-1, -1, +1, 0)
    if cls_id in (15, 16):
        return 2  # [+1, -1] pairs with b=2 (-1, +1, -1, +2)
    if cls_id in (18, 19):
        return 1  # [-1, -1] pairs with b=1 (0, +1, -1, -1)
    return 0  # Hydrophobic pairs with methyl +2 of b=0


def m_scan_c2h2_locks(
    seq: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> tuple[MC2H2Lock, ...]:
    """Scan a Z_21 sequence for C2H2 locks with exact |N_12| = 12 loop."""
    n = len(seq)
    out: list[MC2H2Lock] = []
    for i in range(n):
        if seq[i] != 11:
            continue
        for gap_c in (2, 4):
            j = i + 1 + gap_c
            if j >= n or seq[j] != 11:
                continue
            h1 = j + 1 + c.n_12
            if h1 >= n or seq[h1] != 14:
                continue
            for gap_h in (3, 4, 5):
                h2 = h1 + 1 + gap_h
                if h2 < n and seq[h2] == 14:
                    loop12 = tuple(seq[j + 1 : h1])
                    readers = (loop12[11], loop12[8], loop12[5])
                    targets = tuple(_m_reader_class_to_base_z4(r) for r in readers)
                    out.append(
                        MC2H2Lock(i, j, h1, h2, loop12, readers, (targets[0], targets[1], targets[2]))
                    )
                    break
    return tuple(out)


def m_scan_trp_pin_locks(seq: tuple[int, ...] | list[int]) -> tuple[MTrpPinLock, ...]:
    """Scan a Z_21 sequence for indole-pin (class 10) Major Groove locks."""
    n = len(seq)
    out: list[MTrpPinLock] = []
    for i in range(n - 3):
        if seq[i] != 10:
            continue
        amide_idx = -1
        cation_idx = -1
        for d1 in (1, 2, 3, 4):
            if i + d1 < n and seq[i + d1] in (15, 16):
                amide_idx = i + d1
                break
        for d2 in (2, 3, 4, 5, 6):
            if i + d2 < n and seq[i + d2] in (2, 17):
                cation_idx = i + d2
                break
        if amide_idx != -1 and cation_idx != -1:
            has_16 = any(seq[i + d1] == 16 for d1 in (1, 2, 3, 4) if i + d1 < n)
            # Core (0, 2, 2, 0) [TAAT] or (0, 3, 0, 0, 0) [TGTTT] in Z_4
            core = (0, 2, 2, 0) if has_16 else (0, 3, 0, 0, 0)
            out.append(MTrpPinLock(i, amide_idx, cation_idx, core))
    return tuple(out)


def m_scan_ig_locks(
    seq: tuple[int, ...] | list[int],
    c: MCarrierConstants = M_CARRIER,
) -> tuple[MIgLock, ...]:
    """Scan a Z_21 sequence for Ig domains with loop span around N_HB = 63 and 7 cuboctahedral offsets."""
    n = len(seq)
    thiols = [idx for idx, cls_id in enumerate(seq) if cls_id == 11]
    out: list[MIgLock] = []

    for a in range(len(thiols)):
        cb = thiols[a]
        for b in range(a + 1, len(thiols)):
            cf = thiols[b]
            span = cf - cb
            if not (c.n_shell_2 + 8 <= span <= c.n_hb + c.n_12):
                continue
            w_idx = -1
            for k in range(cb + 8, min(cf, cb + 24)):
                if seq[k] == 10:
                    w_idx = k
                    break
            if w_idx == -1:
                continue

            role = 0
            if w_idx + 2 < n and seq[w_idx + 2] in (2, 17):
                role = +1
            elif w_idx + 4 < n and seq[w_idx + 4] in (18, 19):
                role = -1

            offsets = (
                w_idx + c.d,
                w_idx + c.d_in_sq,
                w_idx + c.n_squares,
                w_idx + c.n_12,
                w_idx + c.n_phi,
                cf - c.n_12,
                cf - c.d_out_e,
            )
            sig = tuple(seq[idx] if 0 <= idx < n else 20 for idx in offsets)
            out.append(MIgLock(role, cb, cf, span, w_idx, sig))
    return tuple(out)


def m_score_ig_lock_pair_quanta(donor: MIgLock, acceptor: MIgLock) -> int:
    """Count net complementary E_HB quanta across the 7 cuboctahedral offsets between two Ig locks."""
    net_q = 0
    for ca, cb in zip(donor.contact_classes_7, acceptor.contact_classes_7):
        if ca == 20 or cb == 20:
            continue
        fa = M_CLASS_21_SPECS[ca].hb_fork
        fb = M_CLASS_21_SPECS[cb].hb_fork
        polar_pairs = sum(1 for s1 in fa for s2 in fb if s1 * s2 == -1)
        polar_clashes = sum(1 for s1 in fa for s2 in fb if s1 * s2 == +1)
        hydro = 1 if (0 in fa and 0 in fb and polar_clashes == 0) else 0
        net_q += polar_pairs + hydro - polar_clashes
    return net_q


def m_score_tf_dna_quanta(
    reader_classes: tuple[int, ...],
    dna_subsite_z4: tuple[int, ...],
) -> int:
    """Count integer E_HB quanta between reader sidechain forks and Major Groove 4-slot vectors."""
    total_q = 0
    for cls_id, b_code in zip(reader_classes, dna_subsite_z4):
        fork = M_CLASS_21_SPECS[cls_id].hb_fork
        slots = M_MAJOR_GROOVE_SLOTS[b_code & 3]
        for s in fork:
            if s == +1 and (-1 in slots):
                total_q += 1
            elif s == -1 and (+1 in slots):
                total_q += 1
            elif s == 0 and (+2 in slots):
                total_q += 1
    return total_q
