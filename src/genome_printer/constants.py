"""Exact constants of the FCC N12 cuboctahedral carrier and scale-cancellation bridge.

Every quantity in this module is derived algebraically from:
  - Spatial dimension d = 3,
  - FCC causal star |N_12| = 12 with cuboctahedral sphere ratio kappa = R_in / R_out = 1 / sqrt(2),
  - {111} hexagonal slice |N_6| = 6 with ratio kappa_hex = sqrt(3) / 2,
  - Bekenstein cell bit budget B_V = 2 * pi / ln(2) -> mod_bits = floor(B_V) = 9,
  - Phase ring Z_512 (N_ring = 2^9 = 512), hierarchy steps N_hier = mod_bits - 1 = 8,
  - Amplitude fractional bits frac_bits = mod_bits - d = 6 (z_min = 2^-6 = 1/64, N_HB = 2^6 - 1 = 63),
  - Discrete Heisenberg threshold Delta_phi_disc = round(512 / (4 * pi)) = 41,
  - Phase cluster count N_phi = ceil(512 / 41) = 1 + |N_12| = 13.

By Theorem 1.1 (Scale-Cancellation Theorem, Vol. 5 `chemistry`), all atomic, covalent,
hydrogen-bond, and biological ratios depend only on these dimensionless integers and
factors (kappa, kappa_hex), with the Planck step ell_P cancelling identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


# 12 integer lattice displacement vectors of the 3D FCC cuboctahedral star N_12:
# (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1), each of Euclidean norm sqrt(2).
FCC_N12_VECTORS: np.ndarray = np.array(
    [
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],
    ],
    dtype=np.int64,
)

# 6 vectors in the {111} plane (x + y + z = 0 subset of N_12), forming the hexagonal ring N_6.
HEX_N6_VECTORS: np.ndarray = np.array(
    [
        [1, -1, 0],
        [0, 1, -1],
        [-1, 0, 1],
        [-1, 1, 0],
        [0, -1, 1],
        [1, 0, -1],
    ],
    dtype=np.int64,
)

# 4 vertices of the regular sp3 tetrahedron K_4 on FCC (mutual dot products = -1/3 after normalization).
K4_TETRAHEDRAL_SLOTS: np.ndarray = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=np.float64,
) / math.sqrt(3.0)


def fcc_shell_count(k: int) -> int:
    """Number of nodes on the k-th cuboctahedral coordination shell: N_shell(k) = 10 * k^2 + 2 for k >= 1."""
    if k < 0:
        raise ValueError("Shell index k must be non-negative.")
    if k == 0:
        return 1
    return 10 * k * k + 2


def fcc_ball_count(k: int) -> int:
    """Cumulative number of nodes in the cuboctahedral ball of radius k: sum_{m=0}^k N_shell(m)."""
    return sum(fcc_shell_count(m) for m in range(k + 1))


@dataclass(frozen=True)
class CarrierConstants:
    """Derived constants of the FCC N12 carrier and atomic/biological scale bridge."""

    # Fundamental topological and geometric counts
    d: int = 3
    n_12: int = 12
    n_6: int = 6
    n_squares: int = 6
    n_triangles: int = 8
    n_edges_cuboct: int = 24

    # Outward and inward branching numbers of the 4 cuboctahedral channels (V, E, square, triangle)
    d_in_v: int = 1
    d_lat_v: int = 4
    d_out_v: int = 7  # d_soft = (1 - kappa^6) / kappa^6 = 7

    d_in_e: int = 2
    d_lat_e: int = 5
    d_out_e: int = 5

    d_in_sq: int = 4
    d_lat_sq: int = 4
    d_out_sq: int = 4

    d_in_tri: int = 3
    d_lat_tri: int = 6
    d_out_tri: int = 3

    # Exact geometric ratios
    kappa: float = 1.0 / math.sqrt(2.0)  # R_in / R_out on 3D FCC cuboctahedron
    kappa_sq: float = 0.5
    kappa_hex: float = math.sqrt(3.0) / 2.0  # R_in / R_out on {111} hexagon

    # Bekenstein bit budget and Z_512 phase ring
    b_v: float = (2.0 * math.pi) / math.log(2.0)
    mod_bits: int = int(math.floor((2.0 * math.pi) / math.log(2.0)))  # 9
    n_ring: int = 1 << int(math.floor((2.0 * math.pi) / math.log(2.0)))  # 512
    n_hier: int = int(math.floor((2.0 * math.pi) / math.log(2.0))) - 1  # 8
    frac_bits: int = int(math.floor((2.0 * math.pi) / math.log(2.0))) - 3  # 6
    z_min: float = 1.0 / (1 << 6)  # 1/64
    n_hb: int = (1 << 6) - 1  # 63 = d * kappa^2 * N_shell(2) = (3/2) * 42
    delta_phi_disc: int = round(512 * 0.5 / (2.0 * math.pi))  # 41
    phi_pauli: int = 512 // 2  # 256
    n_phi: int = math.ceil(512 / 41)  # 13 = 1 + |N_12|
    m_hat: int = 1 + 12 * 8  # 97 = 4^2 + 9^2
    delta_pack: float = 0.5 / 12.0  # 1/24

    # Shell numbers
    n_shell_1: int = 12
    n_shell_2: int = 42
    n_shell_3: int = 92
    n_ball_3: int = 1 + 12 + 42 + 92  # 147 bp (nucleosome core particle)

    # Atomic SI bridge anchor (Bohr radius a_0 and Hydrogen ground-state binding energy BE_H)
    # In atomic units (a_0, BE_H), all Floor 3+ chemistry/biology ratios are pure carrier expressions;
    # these two numbers convert dimensionless lattice units into Angstroms and eV / kJ/mol.
    a_0_angstrom: float = 0.529177210903
    be_h_ev: float = 13.605693122994
    ev_to_kj_mol: float = 96.48533212

    @property
    def alpha_fs(self) -> float:
        """Exact algebraic fine-structure constant from (M_hat=97, d_soft=7, kappa=1/sqrt(2))."""
        u = self.d_out_v * self.m_hat  # 7 * 97 = 679
        num = u * self.m_hat * self.kappa * (u + self.kappa)
        den = (u * self.m_hat) ** 2 - u * (self.kappa ** 3) - self.n_hier
        return num / den

    @property
    def h2_bond_length_angstrom(self) -> float:
        """Equilibrium H2 bond length R_e(H2) = sqrt(2) * a_0 = (1 / kappa) * a_0."""
        return (1.0 / self.kappa) * self.a_0_angstrom

    @property
    def h2_dissociation_ev(self) -> float:
        """Ground-state dissociation energy D_0(H2) = BE_H / d = BE_H / 3."""
        return self.be_h_ev / self.d

    @property
    def e_hb_ev(self) -> float:
        """Universal hydrogen bond energy quantum E_HB = BE_H / 63 = D_0(H2) / (kappa^2 * N_shell(2))."""
        return self.be_h_ev / self.n_hb

    @property
    def e_hb_kj_mol(self) -> float:
        """Universal hydrogen bond energy quantum in kJ/mol (~20.84 kJ/mol)."""
        return self.e_hb_ev * self.ev_to_kj_mol

    @property
    def atp_quantum_ev(self) -> float:
        """Cellular energy quantum Delta_G(ATP) = (d - 1) * E_HB = 2 * E_HB (~0.432 eV)."""
        return (self.d - 1) * self.e_hb_ev

    @property
    def atp_quantum_kj_mol(self) -> float:
        """Cellular energy quantum Delta_G(ATP) in kJ/mol (~41.68 kJ/mol)."""
        return self.atp_quantum_ev * self.ev_to_kj_mol

    @property
    def peptide_cn_bond_angstrom(self) -> float:
        """Peptide backbone single bond R(C-N) = r_cov(6) + r_cov(7) = (52/13)*kappa*a_0 = 4*kappa*a_0."""
        return self.d_in_sq * self.kappa * self.a_0_angstrom

    @property
    def alpha_helix_residues_per_turn(self) -> float:
        """Pauling-Corey 3.6_13 alpha-helix residues per turn: (|N_12| + |square|) / d_out(E) = 18 / 5 = 3.6."""
        return (self.n_12 + self.n_squares) / self.d_out_e

    @property
    def alpha_helix_rise_angstrom(self) -> float:
        """Axial rise per residue in alpha-helix: h_alpha = R(C-N) = 4 * kappa * a_0 (~1.497 A)."""
        return self.peptide_cn_bond_angstrom

    @property
    def alpha_helix_pitch_angstrom(self) -> float:
        """Helical pitch P_alpha = n_alpha * h_alpha = (72/5) * kappa * a_0 (~5.388 A)."""
        return self.alpha_helix_residues_per_turn * self.alpha_helix_rise_angstrom

    @property
    def transmembrane_span_angstrom(self) -> float:
        """Length of an 18-residue (5-turn) transmembrane alpha-helix: 72 * kappa * a_0 (~26.94 A)."""
        return (self.n_12 + self.n_squares) * self.alpha_helix_rise_angstrom

    @property
    def beta_sheet_rise_angstrom(self) -> float:
        """Axial step per residue along antiparallel beta-strand: (floor(B_V) + 1/d) * kappa * a_0 = (28/3)*kappa*a_0 (~3.492 A)."""
        return (self.mod_bits + 1.0 / self.d) * self.kappa * self.a_0_angstrom

    @property
    def bdna_stacking_step_angstrom(self) -> float:
        """Axial stacking step of B-DNA in {111} slice: floor(B_V) * kappa * a_0 = 9 * kappa * a_0 (~3.368 A)."""
        return self.mod_bits * self.kappa * self.a_0_angstrom

    @property
    def bdna_bp_per_turn(self) -> int:
        """Base pairs per turn of B-DNA: |N_12| - 2 = 10 (quadratic growth coefficient of N_shell(k))."""
        return self.n_12 - 2

    @property
    def bdna_width_c1_c1_angstrom(self) -> float:
        """Glycosidic C1'-C1' width of B-DNA duplex: (1 + d_in(square)*d_out(V)) * kappa * a_0 = 29 * kappa * a_0 (~10.85 A)."""
        return (1 + self.d_in_sq * self.d_out_v) * self.kappa * self.a_0_angstrom


CARRIER = CarrierConstants()
