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
    MThermalMobilityWindow,
    m_ionization_ratio_be_h,
    m_covalent_radius_in_a0,
    m_sp3_bond_angle_rad,
    m_compute_hbond_phase_ratios,
    m_compute_life_thermal_windows,
)
from .m_nucleic import MDuplexLatticeState
from .m_polypeptide import MFoldedChain, MProteinSynthesisSpec
from .m_tissue import (
    MTissueMorphology,
    MChromosomeScanResult,
    MSynthesizedGeneProduct,
    MSynthesizedChromosome,
    m_scan_chromosome_de_novo,
    m_synthesize_gene_and_protein,
    m_synthesize_chromosome_for_morphology,
)


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

T_CARRIER_STAGE_NAMES: dict[int, str] = {
    0: "Zygote_Founder (V=1, E=0)",
    1: "Solid_Morula_B3 (N_K4>0, chi_3D=1)",
    2: "Hollow_Vesicle_S2 (N_K4=0, chi_2D=2, beta_2=1)",
    3: "Hollow_Hex_Tube_S1xI (N_K4=0, chi_2D=0, beta_1=1)",
    4: "Stratified_Cortical_Column (N_K4=6(L-1), E_Ig=15(L-1))",
}


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

    def life_thermal_windows_si(self, c: MCarrierConstants = M_CARRIER) -> list[dict[str, float | int]]:
        """Project the 3 dimensionless M-layer thermal mobility windows (rungs m = -1, 0, +1) to SI (eV, kJ/mol, K)."""
        ratios = m_compute_hbond_phase_ratios(c=c)
        e_hb_ev = self.be_h_ev / ratios.n_hb_divisor
        e_hb_kj = e_hb_ev * self.ev_to_kj_mol
        t_hb_k = e_hb_ev / self.k_b_ev_per_kelvin
        out: list[dict[str, float | int]] = []
        for w in m_compute_life_thermal_windows(c=c):
            out.append(
                {
                    "rung_m": w.rung_m,
                    "e_soft_ehb_quanta": w.e_soft_in_ehb_quanta,
                    "e_soft_ev": w.e_soft_in_ehb_quanta * e_hb_ev,
                    "e_soft_kj_mol": w.e_soft_in_ehb_quanta * e_hb_kj,
                    "min_e_hard_ehb_quanta": w.min_e_hard_in_ehb_quanta,
                    "min_e_hard_ev": w.min_e_hard_in_ehb_quanta * e_hb_ev,
                    "min_e_hard_kj_mol": w.min_e_hard_in_ehb_quanta * e_hb_kj,
                    "t_min_kelvin": w.t_min_over_t_hb * t_hb_k,
                    "t_max_kelvin": w.t_max_over_t_hb * t_hb_k,
                    "reduced_wavelength_a0": w.reduced_wavelength_in_a0,
                    "spectroscopic_wavelength_um": (2.0 * math.pi * w.reduced_wavelength_in_a0 * self.a_0_angstrom) * 1e-4,
                }
            )
        return out



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

    if morph.topology_snapshot is not None:
        s = morph.topology_snapshot
        stage_name = T_CARRIER_STAGE_NAMES.get(s.carrier_stage_code, "Unknown")
        lines.append("-" * 92)
        lines.append(
            f"  [M-Topology Readout] stage={stage_name} | "
            f"V={s.v_cells}, E={s.e_contacts}, F_tri={s.f_triangles}, F_sq={s.f_squares_surface}, "
            f"K4={s.k4_tetrahedra}, chi_2D={s.euler_chi_2d}, "
            f"Betti=({s.betti_0}, {s.betti_1_open_channels}, {s.betti_2_closed_cavities})"
        )
    if morph.growth_trajectory:
        lines.append("  [Growth Trajectory on FCC N12]:")
        for snap in morph.growth_trajectory:
            st_str = T_CARRIER_STAGE_NAMES.get(snap.carrier_stage_code, "Unknown")
            lines.append(
                f"    tick {snap.tick:2d}: V={snap.v_cells:2d}, E={snap.e_contacts:3d}, "
                f"F_tri={snap.f_triangles:3d}, F_sq={snap.f_squares_surface:2d}, "
                f"K4={snap.k4_tetrahedra:2d}, chi_2D={snap.euler_chi_2d:3d}, "
                f"Betti=({snap.betti_0},{snap.betti_1_open_channels},{snap.betti_2_closed_cavities}), "
                f"E_Ig={snap.ig_synapses_count:2d} -> {st_str}"
            )

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
    if scan.autonomous_tissue is not None and scan.autonomous_tissue.topology_snapshot is not None:
        s = scan.autonomous_tissue.topology_snapshot
        st_name = T_CARRIER_STAGE_NAMES.get(s.carrier_stage_code, "Unknown")
        lines.append("-" * 96)
        lines.append(
            f"Autonomous M-Carrier Tissue Grown: {st_name} | "
            f"V={s.v_cells}, E={s.e_contacts}, F_tri={s.f_triangles}, F_sq={s.f_squares_surface}, "
            f"K4={s.k4_tetrahedra}, chi_2D={s.euler_chi_2d}, "
            f"Betti=({s.betti_0},{s.betti_1_open_channels},{s.betti_2_closed_cavities}), E_Ig={s.ig_synapses_count}"
        )
    return scan, "\n".join(lines)


def synthesize_gene_and_protein_xray(
    target_dna_triplets: tuple[str, ...] = (),
    target_homeodomain_core: str = "",
    require_catalytic_triad: bool = False,
    require_atp_walker_motor: bool = False,
    require_transmembrane_rod: bool = False,
    ig_synaptic_polarity: int = 0,
    min_stability_quanta_ehb: int = 12,
    proj: TSIBridgeProjector = T_PROJECTOR,
    c: MCarrierConstants = M_CARRIER,
) -> tuple[MSynthesizedGeneProduct, str]:
    """Execute T(Design Spec) -> M(Synthesis Spec) -> M(Synthesize Z_21 & Z_4 + O(L) Verify) -> T(FASTA + X-Ray)."""
    triplets_z4: list[tuple[int, int, int]] = []
    for t_str in target_dna_triplets:
        z4 = encode_dna_to_z4(t_str)
        if len(z4) != 3:
            raise ValueError(f"Each C2H2 DNA target must be 3 bp, got {t_str!r}")
        triplets_z4.append((z4[0], z4[1], z4[2]))

    core_z4 = encode_dna_to_z4(target_homeodomain_core) if target_homeodomain_core else ()
    spec = MProteinSynthesisSpec(
        target_c2h2_dna_triplets_z4=tuple(triplets_z4),
        target_trp_pin_core_z4=core_z4,
        require_catalytic_triad=1 if require_catalytic_triad else 0,
        require_atp_walker_motor=1 if require_atp_walker_motor else 0,
        require_transmembrane_rod=1 if require_transmembrane_rod else 0,
        ig_synaptic_polarity=ig_synaptic_polarity,
        min_stability_quanta_ehb=min_stability_quanta_ehb,
    )
    res = m_synthesize_gene_and_protein(spec, c=c)
    prot_str = decode_z21_to_protein(res.synthesized_class_seq_z21)
    cds_dna_str = decode_z4_to_dna(res.synthesized_coding_dna_z4)
    prom_dna_str = decode_z4_to_dna(res.synthesized_promoter_z4)
    f = res.verified_expression.folded_chain
    e_kj = -proj.quanta_ehb_to_kj_mol(f.total_quanta_ehb, c=c)
    bind_kj = -proj.quanta_ehb_to_kj_mol(res.dna_binding_quanta_verified, c=c)

    lines = [
        f"=== DE NOVO M-LAYER PROTEIN & GENE SYNTHESIS CERTIFICATE (VERIFIED={res.all_constraints_satisfied}) ===",
        f"Synthesized Protein (L={len(prot_str)} aa): {prot_str}",
        f"Synthesized Coding Gene ({len(cds_dna_str)} bp): {cds_dna_str}",
        f"Target Promoter ({len(prom_dna_str)} bp): {prom_dna_str}",
        f"Folded Stability: -{f.total_quanta_ehb} E_HB ({e_kj:.2f} kJ/mol) [target >= {min_stability_quanta_ehb} E_HB] | "
        f"FCC_self_avoid={len(set(f.fcc_lattice_nodes)) == len(f.fcc_lattice_nodes)} | ops={f.elementary_ops_count}",
        f"Verified Locks: C2H2={[decode_z4_to_dna(lk.target_bases_z4) for lk in res.verified_expression.c2h2_locks]} | "
        f"TrpPin={[decode_z4_to_dna(lk.target_core_z4) for lk in res.verified_expression.trp_pin_locks]} | "
        f"CatalyticTriad={len(res.verified_expression.catalytic_triads)} | "
        f"WalkerATP={len(res.verified_expression.atp_walker_locks)} | "
        f"TM18={len(res.verified_expression.transmembrane_rods)} | "
        f"Ig(N_HB=63)={len(res.verified_expression.ig_locks)}",
        f"Verified Promoter Binding Affinity: -{res.dna_binding_quanta_verified} E_HB ({bind_kj:.2f} kJ/mol)",
    ]
    return res, "\n".join(lines)


def synthesize_chromosome_for_morphology_xray(
    target_stage_code: int = 4,
    min_stability_quanta_ehb: int = 48,
    proj: TSIBridgeProjector = T_PROJECTOR,
    c: MCarrierConstants = M_CARRIER,
) -> tuple[MSynthesizedChromosome, str]:
    """Synthesize a complete FASTA chromosome for a target 3D organ stage (2, 3, or 4) and verify via de novo scan."""
    res = m_synthesize_chromosome_for_morphology(
        target_stage_code=target_stage_code,
        min_stability_quanta_ehb=min_stability_quanta_ehb,
        c=c,
    )
    chr_dna = decode_z4_to_dna(res.chromosome_z4)
    scan = res.de_novo_scan_verification
    morph = scan.autonomous_tissue
    s = morph.topology_snapshot if morph is not None else None
    stage_str = T_CARRIER_STAGE_NAMES.get(s.carrier_stage_code, "Unknown") if s is not None else "None"
    tot_kj = -proj.quanta_ehb_to_kj_mol(morph.total_quanta_ehb, c=c) if morph is not None else 0.0

    lines = [
        f"=== DE NOVO CHROMOSOME -> 3D ORGAN SYNTHESIS CERTIFICATE (VERIFIED={res.morphology_verified}) ===",
        f"Target Stage Code: {target_stage_code} -> Grown: {stage_str}",
        f"Synthesized Chromosome Length: {len(chr_dna)} bp | Discovered ORFs on blind scan: {len(scan.expressed_products)}",
    ]
    if s is not None and morph is not None:
        lines.append(
            f"Grown 3D Tissue Invariants: V={s.v_cells} cells, E={s.e_contacts} contacts (E_Ig={s.ig_synapses_count}), "
            f"F_tri={s.f_triangles}, F_sq={s.f_squares_surface}, K4={s.k4_tetrahedra}, "
            f"chi_2D={s.euler_chi_2d}, Betti=({s.betti_0},{s.betti_1_open_channels},{s.betti_2_closed_cavities}) | "
            f"E_adhesion = -{morph.total_quanta_ehb} E_HB ({tot_kj:.2f} kJ/mol)"
        )
    return res, "\n".join(lines)
