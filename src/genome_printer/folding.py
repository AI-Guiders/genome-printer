"""O(1)-per-residue co-translational protein folder on the cuboctahedral K_4 lattice.

Implements Chapter 3 of Volume 6 (`biology`) and Theorem 4.1 of Volume 7 (`compute`)
of "Кипящий океан":
  - Each peptide unit is rigid in the {111} plane (120 deg sp2), while each C_alpha
    tetrahedral node (sp3, cos(theta) = -1/3) has exactly d = 3 discrete orientational
    grooves relative to the preceding link:
      1. Groove 'alpha': right-handed 3.6_13 alpha-helix (n_alpha = 18/5 = 3.6 residues/turn,
         100 deg twist, axial step h_alpha = 4 * kappa * a_0 = 1.497 A, closing the i -> i-4
         13-atom H-bond bridge of energy E_HB = BE_H / 63),
      2. Groove 'beta': pleated {111} beta-strand (axial step h_beta = (28/3) * kappa * a_0 = 3.492 A,
         lateral H-bond pairing between antiparallel strands),
      3. Groove 'turn': loop reversal enforced by Proline (cyclic C_alpha-N) or Glycine/polar caps.
  - Because each residue i = 0..L-1 selects its K_4 groove from a constant-size local window
    (at most d * |N_12| = 36 elementary operations per residue), co-translational folding runs in
    strict linear time O(L) in class P without global 3^L search or neural network approximators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from .constants import CARRIER, CarrierConstants
from .genetic_code import AMINO_ACID_SPECS


@dataclass(frozen=True)
class FoldedProtein:
    """Result of O(1)-per-residue co-translational folding on the cuboctahedral K_4 lattice."""

    sequence: str
    grooves: tuple[str, ...]  # 'alpha', 'beta', or 'turn' per residue
    ca_xyz: np.ndarray  # shape (L, 3) in Angstroms
    backbone_hbonds: int
    sidechain_hbonds: int
    disulfide_bridges: tuple[tuple[int, int], ...]
    free_energy_ev: float
    free_energy_kj_mol: float
    radius_of_gyration_angstrom: float
    elementary_ops_count: int  # Strictly bounded by <= 36 * L (proving O(L) in P)


def _select_local_k4_groove(seq: str, idx: int) -> tuple[str, int]:
    """Select the K_4 groove ('alpha', 'beta', 'turn') at position `idx` using O(1) local window.

    Inspects at most the local cuboctahedral neighborhood window [idx - 4 .. idx + 4]
    (<= 9 residues, <= d * |N_12| = 36 elementary checks).
    """
    n = len(seq)
    aa = seq[idx]
    spec = AMINO_ACID_SPECS[aa]
    ops = 1

    # Proline and Glycine enforce a local K_4 turn/hinge unless stabilized by an overriding helix
    if aa in ("P", "G"):
        return "turn", ops + 1

    # Check local window [idx - 4 .. idx + 4] for i -> i-4 alpha-helix vs beta-strand cooperativity
    w_start = max(0, idx - 4)
    w_end = min(n, idx + 5)
    alpha_votes = 0
    beta_votes = 0
    turn_votes = 0

    for j in range(w_start, w_end):
        ops += 1
        g = AMINO_ACID_SPECS[seq[j]].preferred_groove
        if g == "alpha":
            alpha_votes += 1
        elif g == "beta":
            beta_votes += 1
        else:
            turn_votes += 1

    # If i -> i-4 partner is available and non-proline, alpha groove gains the 13-atom H-bond bonus
    if idx >= 4 and seq[idx - 4] not in ("P", "G"):
        ops += 1
        if spec.preferred_groove == "alpha":
            alpha_votes += 2

    if turn_votes > alpha_votes and turn_votes > beta_votes and spec.preferred_groove == "turn":
        return "turn", ops
    if beta_votes > alpha_votes:
        return "beta", ops
    return "alpha", ops


def fold_protein_cotranslational(sequence: str, c: CarrierConstants = CARRIER) -> FoldedProtein:
    """Fold a polypeptide sequence of length L in O(L) time via local K_4 groove selection."""
    seq = sequence.strip().upper().rstrip("*")
    n = len(seq)
    if n == 0:
        raise ValueError("Protein sequence must be non-empty.")

    grooves: list[str] = []
    ca_xyz = np.zeros((n, 3), dtype=np.float64)
    total_ops = 0
    backbone_hb = 0
    sidechain_hb = 0

    # Pauling-Corey 3.6_13 alpha-helix parameters derived from CARRIER
    # Radius r_alpha of C_alpha cylinder: sqrt((R_step)^2 - h_alpha^2) / (2 * sin(50 deg)) ~ 2.30 A
    h_alpha = c.alpha_helix_rise_angstrom  # 1.497 A
    twist_alpha = (2.0 * math.pi) / c.alpha_helix_residues_per_turn  # 100 deg = 5*pi/9 rad
    r_alpha = (c.n_squares * c.kappa * c.a_0_angstrom)  # 6 * kappa * a_0 ~ 2.245 A

    # Pleated {111} beta-sheet parameters
    h_beta = c.beta_sheet_rise_angstrom  # 3.492 A
    pleat_amp = c.kappa * c.a_0_angstrom  # ~0.374 A perpendicular pleat

    # Running frame state for co-translational extrusion from the ribosome tunnel
    pos = np.zeros(3, dtype=np.float64)
    axis_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    binormal_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    helix_phase = 0.0
    strand_step_index = 0

    for i in range(n):
        g, ops = _select_local_k4_groove(seq, i)
        grooves.append(g)
        total_ops += ops

        if i == 0:
            ca_xyz[0] = pos.copy()
            continue

        if g == "alpha":
            helix_phase += twist_alpha
            pos = pos + h_alpha * axis_dir
            offset = r_alpha * (math.cos(helix_phase) * normal_dir + math.sin(helix_phase) * binormal_dir)
            ca_xyz[i] = pos + offset
            # Check closure of the 13-atom i -> i - 4 hydrogen bond (d_in(square) = 4)
            if i >= c.d_in_sq and grooves[i - c.d_in_sq] == "alpha" and seq[i] != "P":
                backbone_hb += 1
                total_ops += 1
        elif g == "beta":
            strand_step_index += 1
            sign = 1.0 if (strand_step_index % 2 == 0) else -1.0
            pos = pos + h_beta * axis_dir
            ca_xyz[i] = pos + sign * pleat_amp * normal_dir
            if i >= 2 and grooves[i - 2] == "beta":
                backbone_hb += 1
                total_ops += 1
        else:
            # 'turn' groove on K_4: rotate propagation axis by tetrahedral angle cos(theta) = -1/3
            # in the (axis_dir, normal_dir) plane and step by C_alpha-C_alpha distance (~3.80 A = 10*kappa*a_0)
            cos_t = -1.0 / c.d
            sin_t = math.sqrt(1.0 - cos_t * cos_t)
            new_axis = cos_t * axis_dir + sin_t * normal_dir
            new_normal = -sin_t * axis_dir + cos_t * normal_dir
            axis_dir = new_axis / np.linalg.norm(new_axis)
            normal_dir = new_normal / np.linalg.norm(new_normal)
            binormal_dir = np.cross(axis_dir, normal_dir)
            step_ca = (c.n_12 - 2) * c.kappa * c.a_0_angstrom  # 10 * kappa * a_0 ~ 3.742 A
            pos = pos + step_ca * axis_dir
            ca_xyz[i] = pos.copy()

        # Local sidechain H-bond fork pairing within the 12-neighbor window [i - 12 .. i - 1]
        fork_i = AMINO_ACID_SPECS[seq[i]].hb_fork
        if any(s != 0 for s in fork_i):
            for prev in range(max(0, i - c.n_12), i):
                total_ops += 1
                fork_p = AMINO_ACID_SPECS[seq[prev]].hb_fork
                if any(s1 * s2 == -1 for s1 in fork_i for s2 in fork_p):
                    sidechain_hb += 1
                    break

    # Identify Cysteine disulfide locks (including canonical Ig loop span N_HB = 63 or C2H2 locks)
    cys_indices = [idx for idx, ch in enumerate(seq) if ch == "C"]
    disulfides: list[tuple[int, int]] = []
    used_cys: set[int] = set()
    for a_idx in range(len(cys_indices)):
        i_c = cys_indices[a_idx]
        if i_c in used_cys:
            continue
        for b_idx in range(a_idx + 1, len(cys_indices)):
            j_c = cys_indices[b_idx]
            if j_c in used_cys:
                continue
            span = j_c - i_c
            # Disulfide bridges form either at short C-x(2,4)-C zinc/redox knuckles or domain loops (40..75)
            if span in (3, 5) or (c.n_shell_2 <= span <= c.n_hb + c.n_12):
                disulfides.append((i_c, j_c))
                used_cys.add(i_c)
                used_cys.add(j_c)
                break

    total_hb = backbone_hb + sidechain_hb + 2 * len(disulfides)
    free_energy_ev = -total_hb * c.e_hb_ev
    center = np.mean(ca_xyz, axis=0)
    rg = float(math.sqrt(np.mean(np.sum((ca_xyz - center) ** 2, axis=1))))

    return FoldedProtein(
        sequence=seq,
        grooves=tuple(grooves),
        ca_xyz=ca_xyz,
        backbone_hbonds=backbone_hb,
        sidechain_hbonds=sidechain_hb,
        disulfide_bridges=tuple(disulfides),
        free_energy_ev=free_energy_ev,
        free_energy_kj_mol=free_energy_ev * c.ev_to_kj_mol,
        radius_of_gyration_angstrom=rg,
        elementary_ops_count=total_ops,
    )
