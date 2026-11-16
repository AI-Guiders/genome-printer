"""M-layer carrier and full scale ladder from (dh, dt, s_0) to N_a0, BE_H, and E_HB.

Zero T-layer strings, zero human labels, zero external empirical constants.
Every physical scale on Floors k = 0..13 is derived algebraically in native M-layer units:
  - Spatial step dh = 1, time tick dt = 1, link speed c_0 = dh / dt = 1, macro light speed c = kappa * c_0,
  - Action quantum s_0 = 1/2 (in units of hbar = 1 -> s_0 = hbar / 2),
  - Energy quantum E_0 = s_0 / dt, momentum quantum p_0 = s_0 / dh, force quantum F_0 = s_0 / (dh * dt),
  - Inertial mass quantum m_arg = 2 * s_0 * dt / dh^2 = E_0 / c^2,
  - Fine-structure constant alpha_fs from (M_hat = 97, d_soft = 7, kappa = 1/sqrt(2)),
  - Electroweak scale v = (1 / sqrt(r)) * alpha_fs^N_hier * E_0,
  - Higgs mass m_H, proton mass m_p, electron mass m_e = alpha_fs^2 * m_H / N_phi,
  - Electron Compton step count N_c = (E_0 / kappa) / m_e,
  - Integer Bohr step count N_a0 = round(N_c / alpha_fs) in Z (Floor k = 9, a_0 = N_a0 * dh),
  - Hydrogen binding energy BE_H = [alpha_fs / (2 * kappa * N_a0)] * E_0,
  - Universal hydrogen-bond quantum E_HB = BE_H / N_HB = BE_H / 63 in native E_0 units!
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


# 12 integer lattice displacement vectors of the 3D FCC cuboctahedral star N_12
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

# 6 integer vectors in the {111} hexagonal slice N_6 (dx + dy + dz == 0)
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

# 4 unit vectors of the regular sp3 tetrahedron K_4 on FCC (mutual dot products = -1/3)
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
    if k <= 0:
        return 1
    return 10 * k * k + 2


def fcc_ball_count(k: int) -> int:
    """Cumulative number of nodes in the cuboctahedral ball of radius k: sum_{m=0}^k N_shell(m)."""
    return sum(fcc_shell_count(m) for m in range(k + 1))


@dataclass(frozen=True)
class MCarrierConstants:
    """Pure M-layer native generalized coordinates (dh, dt, s_0) and closed k=0..13 scale ladder."""

    # Native M-layer cell steps (generalized coordinates dh, dt and action step s_0)
    dh: float = 1.0
    dt: float = 1.0
    s_0: float = 0.5  # Minimum phase jump Delta_phi_min = 1/2 rad in units of hbar = 1

    # Fundamental topological and geometric counts of the cuboctahedron
    d: int = 3
    n_12: int = 12
    n_6: int = 6
    n_squares: int = 6
    n_triangles: int = 8
    n_edges_cuboct: int = 24

    # Channel branching numbers (d_in, d_lat, d_out) for V(0), E(1), square(2), triangle(3)
    d_in_v: int = 1
    d_lat_v: int = 4
    d_out_v: int = 7

    d_in_e: int = 2
    d_lat_e: int = 5
    d_out_e: int = 5

    d_in_sq: int = 4
    d_lat_sq: int = 4
    d_out_sq: int = 4

    d_in_tri: int = 3
    d_lat_tri: int = 6
    d_out_tri: int = 3

    # Exact geometric sphere ratios
    kappa: float = 1.0 / math.sqrt(2.0)
    kappa_sq: float = 0.5
    kappa_hex: float = math.sqrt(3.0) / 2.0

    # Bekenstein bit budget and Z_512 phase ring
    b_v: float = (2.0 * math.pi) / math.log(2.0)
    mod_bits: int = int(math.floor((2.0 * math.pi) / math.log(2.0)))  # 9
    n_ring: int = 1 << int(math.floor((2.0 * math.pi) / math.log(2.0)))  # 512
    n_hier: int = int(math.floor((2.0 * math.pi) / math.log(2.0))) - 1  # 8
    frac_bits: int = int(math.floor((2.0 * math.pi) / math.log(2.0))) - 3  # 6
    n_codons: int = 1 << 6  # 64 = 2^frac_bits
    z_min: float = 1.0 / (1 << 6)  # 1/64
    n_hb: int = (1 << 6) - 1  # 63 = d * kappa^2 * N_shell(2)
    delta_phi_min: float = 0.5
    phase_residue_r: float = 0.5 / (2.0 * math.pi)  # 1 / (4*pi)
    delta_phi_disc: int = round(512 * 0.5 / (2.0 * math.pi))  # 41
    phi_pauli: int = 512 // 2  # 256
    n_phi: int = math.ceil(512 / 41)  # 13 = 1 + |N_12|
    m_hat: int = 1 + 12 * 8  # 97 = 4^2 + 9^2
    delta_pack: float = 0.5 / 12.0  # 1/24

    # Coordination shell invariants
    n_shell_1: int = 12
    n_shell_2: int = 42
    n_classes_21: int = 42 // 2  # 21 = N_shell(2) / 2
    n_shell_3: int = 92
    n_ball_3: int = 1 + 12 + 42 + 92  # 147

    # --- Theorem 5.1 Mechanical Ladder on (dh, dt, s_0) ---

    @property
    def c_0(self) -> float:
        """Link transfer speed on M: c_0 = dh / dt."""
        return self.dh / self.dt

    @property
    def speed_c(self) -> float:
        """Macroscopic wavefront speed: c = kappa * c_0."""
        return self.kappa * self.c_0

    @property
    def e_0(self) -> float:
        """Primary cell energy quantum: E_0 = s_0 / dt."""
        return self.s_0 / self.dt

    @property
    def p_0(self) -> float:
        """Primary cell momentum quantum: p_0 = s_0 / dh."""
        return self.s_0 / self.dh

    @property
    def f_0(self) -> float:
        """Primary cell force quantum: F_0 = s_0 / (dh * dt)."""
        return self.s_0 / (self.dh * self.dt)

    @property
    def m_arg(self) -> float:
        """Primary cell inertia quantum: m_arg = 2 * s_0 * dt / dh^2 = E_0 / c^2."""
        return (2.0 * self.s_0 * self.dt) / (self.dh * self.dh)

    # --- Full Scale Ladder k = 0 -> 11 in native (dh, E_0, m_arg) units ---

    @property
    def alpha_fs(self) -> float:
        """Exact algebraic fine-structure constant from (M_hat=97, d_soft=7, kappa=1/sqrt(2))."""
        u = self.d_out_v * self.m_hat
        num = u * self.m_hat * self.kappa * (u + self.kappa)
        den = (u * self.m_hat) ** 2 - u * (self.kappa ** 3) - self.n_hier
        return num / den

    @property
    def vev_e0(self) -> float:
        """Electroweak VEV scale v in native energy units E_0 (Floor k = 6): v = r^(-1/2) * alpha_fs^8 * E_0."""
        return (1.0 / math.sqrt(self.phase_residue_r)) * (self.alpha_fs ** self.n_hier) * self.e_0

    @property
    def m_higgs_bare_e0(self) -> float:
        """Bare Higgs mass in E_0 units: kappa^2 * v."""
        return self.kappa_sq * self.vev_e0

    @property
    def m_higgs_e0(self) -> float:
        """Dressed Higgs mass in E_0 units: m_H^(bare) * sqrt(1 + N_hier * alpha_fs / (pi * kappa^2))."""
        loop_factor = math.sqrt(1.0 + self.n_hier * self.alpha_fs / (math.pi * self.kappa_sq))
        return self.m_higgs_bare_e0 * loop_factor

    @property
    def m_proton_e0(self) -> float:
        """Proton K_3 mass in E_0 units (Floor k = 7): alpha_fs * m_H^(bare) * (1 + delta_pack)."""
        return self.alpha_fs * self.m_higgs_bare_e0 * (1.0 + self.delta_pack)

    @property
    def m_electron_e0(self) -> float:
        """Electron 13-cell cluster mass in E_0 units (Floor k = 8): alpha_fs^2 * m_H / N_phi."""
        return (self.alpha_fs * self.alpha_fs) * self.m_higgs_e0 / self.n_phi

    @property
    def n_compton_steps(self) -> float:
        """Electron Compton wavelength in integer lattice steps dh (Floor k = 8): N_c = (E_0 / kappa) / m_e."""
        e_planck_in_e0 = self.e_0 / self.kappa
        return e_planck_in_e0 / self.m_electron_e0

    @property
    def n_bohr_steps(self) -> int:
        """Exact integer number of lattice steps dh in the first Bohr radius a_0 (Floor k = 9): N_a0 = round(N_c / alpha_fs)."""
        return int(round(self.n_compton_steps / self.alpha_fs))

    @property
    def a_0_dh(self) -> float:
        """Bohr radius a_0 in native lattice steps dh: N_a0 * dh."""
        return float(self.n_bohr_steps) * self.dh

    @property
    def n_pack_atom(self) -> int:
        """Exact integer number of Planck Voronoi cells in one atomic Wigner-Seitz cell (Floor k = 10): N_a0^3."""
        return self.n_bohr_steps ** 3

    @property
    def be_h_e0(self) -> float:
        """Hydrogen ground-state binding energy BE_H in native energy units E_0: [alpha_fs / (2 * kappa * N_a0)] * E_0."""
        return (self.alpha_fs / (2.0 * self.kappa * self.n_bohr_steps)) * self.e_0

    @property
    def d0_h2_e0(self) -> float:
        """H2 dissociation energy D_0(H2) in native energy units E_0: BE_H / d."""
        return self.be_h_e0 / self.d

    @property
    def e_hb_e0(self) -> float:
        """Universal hydrogen-bond quantum E_HB in native energy units E_0: BE_H / N_HB = BE_H / 63."""
        return self.be_h_e0 / self.n_hb

    @property
    def atp_e0(self) -> float:
        """Cellular energy quantum Delta_G(ATP) in native energy units E_0: (d - 1) * E_HB = 2 * E_HB."""
        return (self.d - 1) * self.e_hb_e0

    @property
    def e_hb_in_be_h_units(self) -> float:
        """Ratio E_HB / BE_H = 1 / N_HB = 1 / 63."""
        return 1.0 / self.n_hb

    @property
    def atp_in_ehb_quanta(self) -> int:
        """Cellular energy quantum in integer units of E_HB: d - 1 = 2."""
        return self.d - 1

    # --- Molecular & Biological Step Lengths in both a_0 and native dh steps ---

    @property
    def peptide_cn_in_a0_units(self) -> float:
        """Peptide single bond length in units of a_0: d_in(sq) * kappa = 4 * kappa."""
        return self.d_in_sq * self.kappa

    @property
    def peptide_cn_dh(self) -> float:
        """Peptide single bond length in native lattice steps dh: 4 * kappa * N_a0 * dh."""
        return self.peptide_cn_in_a0_units * self.a_0_dh

    @property
    def alpha_helix_residues_per_turn(self) -> float:
        """Pauling-Corey 3.6_13 residues per turn: (|N_12| + |square|) / d_out(E) = 18 / 5 = 3.6."""
        return (self.n_12 + self.n_squares) / self.d_out_e

    @property
    def alpha_helix_rise_in_a0_units(self) -> float:
        """Axial rise per residue in alpha-helix in units of a_0: 4 * kappa."""
        return self.peptide_cn_in_a0_units

    @property
    def alpha_helix_rise_dh(self) -> float:
        """Axial rise per residue in alpha-helix in native steps dh: 4 * kappa * N_a0 * dh."""
        return self.alpha_helix_rise_in_a0_units * self.a_0_dh

    @property
    def beta_sheet_rise_in_a0_units(self) -> float:
        """Axial step per residue in beta-strand in units of a_0: (mod_bits + 1/d) * kappa = (28/3) * kappa."""
        return (self.mod_bits + 1.0 / self.d) * self.kappa

    @property
    def beta_sheet_rise_dh(self) -> float:
        """Axial step per residue in beta-strand in native steps dh."""
        return self.beta_sheet_rise_in_a0_units * self.a_0_dh

    @property
    def bdna_stacking_step_in_a0_units(self) -> float:
        """Axial stacking step of B-DNA in units of a_0: mod_bits * kappa = 9 * kappa."""
        return self.mod_bits * self.kappa

    @property
    def bdna_stacking_step_dh(self) -> float:
        """Axial stacking step of B-DNA in native lattice steps dh: 9 * kappa * N_a0 * dh."""
        return self.bdna_stacking_step_in_a0_units * self.a_0_dh

    @property
    def bdna_bp_per_turn(self) -> int:
        """Base pairs per turn of B-DNA: |N_12| - 2 = 10."""
        return self.n_12 - 2

    @property
    def bdna_width_in_a0_units(self) -> float:
        """Duplex C1'-C1' width in units of a_0: (1 + d_in(sq)*d_out(V)) * kappa = 29 * kappa."""
        return (1 + self.d_in_sq * self.d_out_v) * self.kappa

    @property
    def bdna_width_dh(self) -> float:
        """Duplex C1'-C1' width in native lattice steps dh: 29 * kappa * N_a0 * dh."""
        return self.bdna_width_in_a0_units * self.a_0_dh


M_CARRIER = MCarrierConstants()
