"""Test suite for the two-layer (M -> T) cuboctahedral FCC N12 genome interpreter."""

from __future__ import annotations

import math
import numpy as np

from genome_printer import (
    M_CARRIER,
    FCC_N12_VECTORS,
    HEX_N6_VECTORS,
    K4_TETRAHEDRAL_SLOTS,
    fcc_shell_count,
    fcc_ball_count,
    m_step_g_reversible,
    m_step_g_reverse,
    m_cuboctahedral_subshells,
    m_ionization_ratio_be_h,
    m_ionization_energy_e0,
    m_covalent_radius_in_a0,
    m_covalent_radius_dh,
    m_find_hydrogen_commensurate_backbone_pairs,
    m_sp3_bond_angle_cos,
    m_compute_hbond_phase_ratios,
    m_compute_life_thermal_windows,
    M_FRONTAL_TRACKS,
    M_MAJOR_GROOVE_SLOTS,
    M_RUMER_OCTET_I_STEMS,
    M_RUMER_OCTET_II_STEMS,
    M_CODON_TO_CLASS_21,
    m_complement_base,
    m_rumer_transform_base,
    m_rumer_transform_stem,
    m_watson_crick_hbond_quanta,
    m_translate_codons,
    m_build_duplex,
    M_CLASS_21_SPECS,
    m_fold_chain_cotranslational,
    m_scan_c2h2_locks,
    m_scan_trp_pin_locks,
    m_scan_ig_locks,
    m_score_ig_lock_pair_quanta,
    MGeneLocus,
    m_express_locus,
    m_compute_regulatory_quanta_matrix,
    m_step_fcc_morphogen_automaton,
    m_grow_vesicle_s1,
    m_grow_hex_tube,
    m_grow_laminar_column_6,
    m_grow_tissue_autonomous,
    T_PROJECTOR,
    encode_dna_to_z4,
    decode_z4_to_dna,
    encode_protein_to_z21,
    decode_z21_to_protein,
    format_polypeptide_xray_table,
    format_tissue_xray_table,
    scan_fasta_and_format_xray,
    synthesize_gene_and_protein_xray,
    synthesize_chromosome_for_morphology_xray,
)


def test_m_carrier_native_ladder_dh_dt_s0() -> None:
    """Verify the closed scale ladder from (dh, dt, s_0) to N_a0, BE_H, and E_HB in M_CARRIER."""
    assert FCC_N12_VECTORS.shape == (12, 3)
    assert HEX_N6_VECTORS.shape == (6, 3)
    assert K4_TETRAHEDRAL_SLOTS.shape == (4, 3)

    # Mechanical ladder Theorem 5.1 on (dh=1, dt=1, s_0=1/2)
    assert M_CARRIER.c_0 == 1.0
    assert math.isclose(M_CARRIER.speed_c, 1.0 / math.sqrt(2.0), rel_tol=1e-12)
    assert M_CARRIER.e_0 == 0.5
    assert M_CARRIER.p_0 == 0.5
    assert M_CARRIER.f_0 == 0.5
    assert M_CARRIER.m_arg == 1.0

    # Alpha_fs and closed hierarchy v -> m_H -> m_p -> m_e -> N_c -> N_a0 -> BE_H -> E_HB
    assert math.isclose(1.0 / M_CARRIER.alpha_fs, 137.03599919, rel_tol=1e-7)
    assert isinstance(M_CARRIER.n_bohr_steps, int)
    assert M_CARRIER.n_bohr_steps > 3_000_000_000_000_000_000_000_000
    assert M_CARRIER.n_pack_atom == M_CARRIER.n_bohr_steps ** 3
    assert math.isclose(M_CARRIER.e_hb_e0 * M_CARRIER.n_hb, M_CARRIER.be_h_e0, rel_tol=1e-12)
    assert math.isclose(M_CARRIER.atp_e0, 2.0 * M_CARRIER.e_hb_e0, rel_tol=1e-12)

    # Three-way identity for N_HB = 63
    id1 = int(round(M_CARRIER.d * M_CARRIER.kappa_sq * fcc_shell_count(2)))
    id2 = int(round(((1.0 - M_CARRIER.kappa_sq**3) / (M_CARRIER.kappa_sq**3)) * M_CARRIER.mod_bits))
    id3 = (1 << M_CARRIER.frac_bits) - 1
    assert id1 == id2 == id3 == 63
    assert fcc_ball_count(3) == 147

    # Bit-exact reversibility of the shared mt_ca step operator `g` on 3D FCC Z_512[i]^2
    f0 = np.zeros((4, 4, 4, 4), dtype=np.int32)
    f0[2, 2, 2] = [64, 0, 0, 64]
    f0[2, 3, 2] = [32, 32, 16, 48]
    f_curr = f0.copy()
    f_past = f0.copy()
    impulses = []
    for _ in range(6):
        f_next, f_prev, imp = m_step_g_reversible(f_curr, f_past)
        impulses.append(imp)
        f_past = f_prev
        f_curr = f_next
    for imp in reversed(impulses):
        f_past, f_curr = m_step_g_reverse(f_past, f_curr, imp)
    assert np.array_equal(f_curr, f0)


def test_m_chemistry_and_t_projector() -> None:
    """Verify M-layer dimensionless chemistry ratios and T-layer SI projection."""
    subshells = m_cuboctahedral_subshells()
    assert [s.spin_capacity for s in subshells] == [2, 6, 10, 14]

    # In native E_0 units: IE_1(1) == BE_H and IE_1(8) == BE_H (oxygen exact screening sigma=|square|=6)
    assert math.isclose(m_ionization_energy_e0(1), M_CARRIER.be_h_e0, rel_tol=1e-12)
    assert math.isclose(m_ionization_energy_e0(8), M_CARRIER.be_h_e0, rel_tol=1e-12)

    # Exact commensurability of peptide C-N bond in both a_0 and native dh steps:
    # r_cov(6) + r_cov(7) = (27/13 + 25/13)*kappa*a_0 = 4*kappa*a_0 = 4*kappa*N_a0*dh
    rc_a0 = m_covalent_radius_in_a0(6, bond_order=1)
    rn_a0 = m_covalent_radius_in_a0(7, bond_order=1)
    assert math.isclose(rc_a0 + rn_a0, M_CARRIER.peptide_cn_in_a0_units, rel_tol=1e-12)
    assert math.isclose(
        m_covalent_radius_dh(6) + m_covalent_radius_dh(7),
        M_CARRIER.peptide_cn_dh,
        rel_tol=1e-12,
    )
    # Exactly two p-block square-channel pairs (Z1 + Z2 == N_phi == 13) are commensurate with r_cov(H):
    # (5, 8) [B-O] and (6, 7) [C-N], while period-3 Si (Z=14) has exact radius d * kappa * a_0
    comm_pairs = m_find_hydrogen_commensurate_backbone_pairs()
    assert len(comm_pairs) == M_CARRIER.d - 1
    assert all(z1 + z2 == M_CARRIER.n_phi for z1, z2 in comm_pairs)
    assert math.isclose(
        m_covalent_radius_in_a0(6 + M_CARRIER.n_hier, bond_order=1),
        M_CARRIER.d * M_CARRIER.kappa,
        rel_tol=1e-12,
    )

    # Exact cosines on K_4: -(N_hier - m_LP) / (d * N_hier) for m_LP in 0..d-1
    for m_lp in range(M_CARRIER.d):
        expected_cos = -(M_CARRIER.n_hier - m_lp) / float(M_CARRIER.d * M_CARRIER.n_hier)
        assert math.isclose(m_sp3_bond_angle_cos(m_lp), expected_cos, abs_tol=1e-12)

    # T-projector SI verification against NIST reference measurements
    nist_ie = {
        1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298,
        6: 11.260, 7: 14.534, 8: 13.618, 9: 17.423, 10: 21.565,
    }
    for z, ref in nist_ie.items():
        pred = T_PROJECTOR.ionization_energy_ev(z)
        assert abs(pred - ref) / ref < 0.01

    ratios = m_compute_hbond_phase_ratios()
    w_si = T_PROJECTOR.water_thermodynamics_si()
    assert math.isclose(
        w_si["bcs_gap_ratio"],
        (M_CARRIER.d_out_v / float(M_CARRIER.d - 1)) * (M_CARRIER.m_hat / (M_CARRIER.m_hat - 1.0)),
        abs_tol=1e-12,
    )

    # Three dimensionless & SI thermal mobility windows inherited directly from MHBondPhaseRatios
    windows_m = m_compute_life_thermal_windows()
    assert [w.rung_m for w in windows_m] == [-1, 0, 1]
    assert math.isclose(windows_m[0].e_soft_in_ehb_quanta, ratios.fus_in_ehb_quanta, rel_tol=1e-12)
    assert math.isclose(windows_m[1].e_soft_in_ehb_quanta, 1.0, rel_tol=1e-12)
    assert math.isclose(windows_m[2].e_soft_in_ehb_quanta, ratios.vap_in_ehb_quanta, rel_tol=1e-12)

    windows_si = T_PROJECTOR.life_thermal_windows_si()
    for w_m, w_s in zip(windows_m, windows_si):
        assert math.isclose(
            w_m.min_e_hard_in_ehb_quanta,
            M_CARRIER.n_12 * w_m.e_soft_in_ehb_quanta,
            rel_tol=1e-12,
        )
        assert math.isclose(
            w_m.t_min_over_t_hb,
            ratios.t_melt_over_t_hb * w_m.e_soft_in_ehb_quanta,
            rel_tol=1e-12,
        )
        assert math.isclose(
            w_m.t_max_over_t_hb,
            ratios.t_boil_over_t_hb * w_m.e_soft_in_ehb_quanta,
            rel_tol=1e-12,
        )
        assert math.isclose(
            float(w_s["t_min_kelvin"]),
            w_m.e_soft_in_ehb_quanta * w_si["t_melt_kelvin"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            float(w_s["t_max_kelvin"]),
            w_m.e_soft_in_ehb_quanta * w_si["t_boil_kelvin"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            float(w_s["min_e_hard_ev"]),
            M_CARRIER.n_12 * w_m.e_soft_in_ehb_quanta * w_si["e_hb_ev"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            float(w_s["min_e_hard_kj_mol"]),
            M_CARRIER.n_12 * w_m.e_soft_in_ehb_quanta * w_si["e_hb_kj_mol"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            w_m.reduced_wavelength_in_a0,
            ((M_CARRIER.d - 1.0) * M_CARRIER.n_hb) / (M_CARRIER.alpha_fs * w_m.e_soft_in_ehb_quanta),
            rel_tol=1e-12,
        )
        assert math.isclose(
            w_m.reduced_wavelength_dh,
            w_m.reduced_wavelength_in_a0 * M_CARRIER.a_0_dh,
            rel_tol=1e-12,
        )



def test_m_nucleic_z4_and_rumer_bit_inversion() -> None:
    """Verify Z_4 base registers, bitwise Rumer involution (s ^ 0xF), and 147-bp nucleosome duplex."""
    for b in range(4):
        comp = m_complement_base(b)
        assert m_complement_base(comp) == b
        q = m_watson_crick_hbond_quanta(b, comp)
        assert q in (2, 3)
        assert m_rumer_transform_base(m_rumer_transform_base(b)) == b

    # Bitwise complement `s ^ 0xF` bijectively maps Rumer Octet I (8 stems) <-> Octet II (8 stems)
    mapped_stems = {m_rumer_transform_stem(s) for s in M_RUMER_OCTET_I_STEMS}
    assert mapped_stems == set(M_RUMER_OCTET_II_STEMS)

    assert len(M_CODON_TO_CLASS_21) == 64 == M_CARRIER.n_codons
    assert len(set(M_CODON_TO_CLASS_21)) == 21 == M_CARRIER.n_classes_21
    assert len(M_MAJOR_GROOVE_SLOTS) == 4

    # 147-bp nucleosome duplex in M-units
    bases_147 = tuple(i % 4 for i in range(147))
    duplex = m_build_duplex(bases_147)
    assert math.isclose(duplex.nucleosome_ball_fraction, 1.0, abs_tol=1e-12)
    assert duplex.strand1_xyz_a0.shape == (147, 3)
    assert duplex.strand1_xyz_dh.shape == (147, 3)
    assert math.isclose(duplex.total_energy_e0, -duplex.total_hbond_quanta * M_CARRIER.e_hb_e0, rel_tol=1e-12)


def test_m_polypeptide_folding_and_locks() -> None:
    """Verify O(L) folding on Z_21 classes and C2H2 / Trp-pin / Ig lock scanning without strings."""
    assert len(M_CLASS_21_SPECS) == 21

    # Encode a test sequence via T-codec, then run pure M-layer folding and lock scanning
    z21_seq = encode_protein_to_z21("MACPECKSFSQSSNLQKHHQRTHGWFQNRRMKWKK")
    folded = m_fold_chain_cotranslational(z21_seq)

    assert folded.ca_xyz_a0.shape == (len(z21_seq), 3)
    assert folded.elementary_ops_count <= 36 * len(z21_seq)
    assert folded.total_quanta_ehb > 10
    assert math.isclose(folded.total_energy_e0, -folded.total_quanta_ehb * M_CARRIER.e_hb_e0, rel_tol=1e-12)

    c2h2_locks = m_scan_c2h2_locks(z21_seq)
    assert len(c2h2_locks) == 1
    assert len(c2h2_locks[0].loop_12_classes) == 12

    trp_locks = m_scan_trp_pin_locks(z21_seq)
    assert len(trp_locks) >= 1
    assert trp_locks[0].target_core_z4 in ((0, 2, 2, 0), (0, 3, 0, 0, 0))

    donor_z21 = encode_protein_to_z21("MA" + "C" + ("A" * 11) + "W" + "A" + "R" + ("A" * 48) + "C" + "GG")
    acceptor_z21 = encode_protein_to_z21("MA" + "C" + ("A" * 11) + "W" + "AAA" + "E" + ("A" * 46) + "C" + "GG")
    d_ig = m_scan_ig_locks(donor_z21)
    a_ig = m_scan_ig_locks(acceptor_z21)
    assert len(d_ig) == 1 and d_ig[0].polarity_role == +1 and d_ig[0].loop_span == 63
    assert len(a_ig) == 1 and a_ig[0].polarity_role == -1 and a_ig[0].loop_span == 63
    assert m_score_ig_lock_pair_quanta(d_ig[0], a_ig[0]) > 0

    xray_str = format_polypeptide_xray_table(folded)
    assert "X-RAY BOND TRACE" in xray_str


def test_m_tissue_organogenesis_and_fcc_automaton() -> None:
    """Verify pure M-layer gene expression, FCC morphogen step, and 3D tissue architectures."""
    cds_z4 = encode_dna_to_z4("ATGTGGTTCCAGAACCGTCGTGCGCTGGCGTAA")
    prom1_z4 = encode_dna_to_z4("GCGCGCTAATTGTTTGCGCGC")
    prom2_z4 = encode_dna_to_z4("GCGCGCTAATTGCGCGCGCGC")

    prod1 = m_express_locus(MGeneLocus(0, prom1_z4, cds_z4))
    prod2 = m_express_locus(MGeneLocus(1, prom2_z4, cds_z4))
    q_mat = m_compute_regulatory_quanta_matrix([prod1, prod2])
    assert q_mat.shape == (2, 2)
    assert int(np.max(q_mat)) >= 4

    grid = np.zeros((6, 6, 6, 2), dtype=np.float64)
    grid[3, 3, 3, 0] = 1.0
    out = m_step_fcc_morphogen_automaton(grid, q_mat, n_steps=4)
    assert float(np.sum(out[..., 1])) > 0.0

    # 1. S_1 vesicle: 12 cells, 24 cuboctahedral edges, 1 central cavity
    #    Trajectory: Zygote (V=1, code=0) -> Morula B^3 (V=13, E=36, K4=8, code=1) -> Vesicle S^2 (V=12, E=24, F_tri=8, F_sq=6, chi_2D=2, beta_2=1, code=2)
    vesicle = m_grow_vesicle_s1()
    assert len(vesicle.living_cells) == 12
    assert len(vesicle.adhesion_network) == 24
    assert len(vesicle.lumen_nodes) == 1
    assert len(vesicle.growth_trajectory) == 3
    assert vesicle.growth_trajectory[0].carrier_stage_code == 0
    assert vesicle.growth_trajectory[1].v_cells == 13
    assert vesicle.growth_trajectory[1].k4_tetrahedra == 8
    assert vesicle.growth_trajectory[1].carrier_stage_code == 1
    assert vesicle.topology_snapshot is not None
    assert vesicle.topology_snapshot.f_triangles == 8
    assert vesicle.topology_snapshot.f_squares_surface == 6
    assert vesicle.topology_snapshot.k4_tetrahedra == 0
    assert vesicle.topology_snapshot.euler_chi_2d == 2
    assert vesicle.topology_snapshot.betti_2_closed_cavities == 1
    assert vesicle.topology_snapshot.carrier_stage_code == 2

    # 2. N_6 tube: 4 rings x 6 = 24 cells, 4 central cavity nodes, chi_2D == 0 at every ring r=1..4
    tube = m_grow_hex_tube(axial_rings=4)
    assert len(tube.living_cells) == 24
    assert len(tube.lumen_nodes) == 4
    assert tube.topology_snapshot is not None
    assert tube.topology_snapshot.f_triangles == 24
    assert tube.topology_snapshot.f_squares_surface == 6
    assert tube.topology_snapshot.k4_tetrahedra == 0
    assert tube.topology_snapshot.euler_chi_2d == 0
    assert tube.topology_snapshot.betti_1_open_channels == 1
    assert tube.topology_snapshot.carrier_stage_code == 3
    for snap in tube.growth_trajectory:
        assert snap.euler_chi_2d == 0
        assert snap.k4_tetrahedra == 0
        assert snap.carrier_stage_code == 3

    # 3. |square|=6 laminar column: 42 = N_shell(2) cells, 41 = Delta_phi_disc divisions,
    #    147 = N_ball(3) total contacts, 75 inter-lamina Ig synaptic locks, 30 K_4 tetrahedra
    col = m_grow_laminar_column_6()
    assert len(col.living_cells) == 42
    assert len(col.division_lineage) == 41
    assert len(col.adhesion_network) == 147
    assert len(col.ig_synaptic_network) == 75
    assert col.topology_snapshot is not None
    assert col.topology_snapshot.k4_tetrahedra == 30
    assert col.topology_snapshot.ig_synapses_count == 75
    assert col.topology_snapshot.carrier_stage_code == 4

    # 4. Autonomous morphogenesis driven purely by expressed protein locks via the single branchless field:
    #    - lock-free proteome -> isotropic O_h -> grows hollow Vesicle S^2 (code=2, chi_2D=2)
    #    - prod1 carries Trp-pin -> D_3d axial symmetry without Ig rescue -> grows hollow Tube (code=3, chi_2D=0)
    #    - donor + acceptor Ig proteome -> D_3d with Ig synaptic rescue -> grows 6-layer Column (code=4, chi_2D=41)
    lock_free_cds = encode_dna_to_z4("ATGGCGCTGGCGGCGCTGGCGCTGGCGTAA")
    prod_iso = m_express_locus(MGeneLocus(2, prom1_z4, lock_free_cds))
    auto_vesicle = m_grow_tissue_autonomous([prod_iso])
    assert auto_vesicle.topology_snapshot is not None
    assert auto_vesicle.topology_snapshot.carrier_stage_code == 2
    assert auto_vesicle.topology_snapshot.euler_chi_2d == 2

    auto_tube = m_grow_tissue_autonomous([prod1], axial_steps=4)
    assert auto_tube.topology_snapshot is not None
    assert auto_tube.topology_snapshot.carrier_stage_code == 3
    assert auto_tube.topology_snapshot.euler_chi_2d == 0
    assert len(auto_tube.cavitation_events) == 4

    ig_donor_res, _ = synthesize_gene_and_protein_xray(ig_synaptic_polarity=+1, min_stability_quanta_ehb=40)
    ig_acc_res, _ = synthesize_gene_and_protein_xray(ig_synaptic_polarity=-1, min_stability_quanta_ehb=40)
    auto_col = m_grow_tissue_autonomous([ig_donor_res.verified_expression, ig_acc_res.verified_expression])
    assert auto_col.topology_snapshot is not None
    assert auto_col.topology_snapshot.carrier_stage_code == 4
    assert auto_col.topology_snapshot.euler_chi_2d == 41
    assert len(auto_col.living_cells) == 42
    assert len(auto_col.adhesion_network) == 147
    assert len(auto_col.cavitation_events) == 0

    t_str = format_tissue_xray_table(col)
    assert "TISSUE MORPHOLOGY X-RAY TRACE" in t_str


def test_de_novo_fasta_scanner_and_four_former_blind_spots() -> None:
    """Verify de novo FASTA -> M -> O(1) Fold -> T(FASTA) scan, splicing, self-avoidance, and locks."""
    # Construct a synthetic eukaryotic chromosome segment containing:
    #   - Upstream open promoter with TAAT core
    #   - Gene 0: Exon 1 ("ATGGCG") + 45-bp GU..A..polyY..AG intron containing an in-frame TAA stop
    #             + Exon 2 encoding a Walker ATP P-loop (GxxxxGKS), Catalytic Triad (S..H..D), and Trp-pin
    #   - Intergenic dark region
    #   - Gene 1: Short lock-free ORF (to verify blind-spot detection!)
    promoter_0 = "GCGCGCGCGCGCTAATGCGCGCGC"
    # Exon 1: M-A (ATG-GCA) -> no internal GT!
    exon_1 = "ATGGCA"
    # 45 bp intron: starts with GT (3,0), has in-frame TAA at offset +3 ("GTATAA"), branch-point A (2) at -21, 19-bp poly-T tract, ends with AG (2,3)
    intron_45 = "GTATAA" + ("C" * 17) + "A" + ("T" * 19) + "AG"
    # Exon 2: G-Q-Q-Q-Q-G-K-S (Walker ATP) + H-Q-D (Catalytic triad with S) + W-F-Q-N-R-R (Trp-pin) + TAA
    exon_2 = "GGACAGCAGCAGCAGGGAAAAAGCCATCAGGACTGGTTCCAGAACCGCCGCTAA"
    intergenic = "C" * 36
    orf_blind_spot = "ATGGCGCTGGCGGCGCTGGCGCTGGCGTAA"

    synthetic_fasta = f">synthetic_chr1\n{promoter_0}{exon_1}{intron_45}{exon_2}{intergenic}{orf_blind_spot}\n"
    scan, report = scan_fasta_and_format_xray(synthetic_fasta, min_orf_codons=8)

    assert len(scan.expressed_products) == 2
    # Locus 0 was rescued by M-layer lariat splicing of the 45-bp intron!
    assert 0 in scan.spliced_loci_indices
    prod0 = scan.expressed_products[0]
    assert len(prod0.spliced_introns) == 1
    assert prod0.spliced_introns[0].excised_length == 45
    assert len(prod0.atp_walker_locks) == 1
    assert len(prod0.catalytic_triads) >= 1
    assert len(prod0.trp_pin_locks) >= 1
    # Verify strict 3D FCC N12 self-avoidance (Pauli occupancy b(x) in {0, 1})
    fcc_nodes = prod0.folded_chain.fcc_lattice_nodes
    assert len(set(fcc_nodes)) == len(fcc_nodes)

    # Locus 1 has no structural lock and is flagged as a blind spot
    assert 1 in scan.unannotated_lock_free_loci
    assert "DE NOVO FASTA -> M -> T(FASTA) X-RAY SCAN" in report


def test_de_novo_protein_and_gene_synthesis_with_prescribed_properties() -> None:
    """Verify inverse folding / de novo synthesis of a protein and coding gene with prescribed locks and stability."""
    # Synthesize a multi-functional protein and gene required to:
    #   1) Bind the 6-bp promoter address ("GCG", "TAT") via a tandem 2x C2H2 finger array (|N_12|=12 loops),
    #   2) Bind the homeodomain core "TAAT" via a Trp-pin lock,
    #   3) Catalyze proton-relay cleavage via a Ser-His-Asp triad (3 E_HB),
    #   4) Couple ATP via a Walker P-loop motor (2 E_HB),
    #   5) Anchor in the membrane via an 18-residue (|N_12|+|square|=18) transmembrane rod,
    #   6) Present an Ig_donor (+1) synaptic lock with loop span N_HB = 63,
    #   7) Achieve a folding stability of at least 90 E_HB quanta (~1875 kJ/mol).
    synth_res, cert_str = synthesize_gene_and_protein_xray(
        target_dna_triplets=("GCG", "TAT"),
        target_homeodomain_core="TAAT",
        require_catalytic_triad=True,
        require_atp_walker_motor=True,
        require_transmembrane_rod=True,
        ig_synaptic_polarity=+1,
        min_stability_quanta_ehb=90,
    )

    assert synth_res.all_constraints_satisfied is True
    assert synth_res.verified_expression.folded_chain.total_quanta_ehb >= 90
    assert synth_res.dna_binding_quanta_verified >= 10
    assert len(synth_res.verified_expression.c2h2_locks) == 2
    assert len(synth_res.verified_expression.trp_pin_locks) == 1
    assert len(synth_res.verified_expression.catalytic_triads) == 1
    assert len(synth_res.verified_expression.atp_walker_locks) == 1
    assert len(synth_res.verified_expression.transmembrane_rods) == 1
    assert len(synth_res.verified_expression.ig_locks) == 1
    assert "VERIFIED=True" in cert_str

    # Also verify a minimal spec (with all optional locks disabled) has zero false-positive lock detections:
    min_res, min_cert = synthesize_gene_and_protein_xray(
        target_dna_triplets=("GCG",),
        target_homeodomain_core="",
        require_catalytic_triad=False,
        require_atp_walker_motor=False,
        require_transmembrane_rod=False,
        ig_synaptic_polarity=0,
        min_stability_quanta_ehb=60,
    )
    assert min_res.all_constraints_satisfied is True
    assert min_res.verified_expression.folded_chain.total_quanta_ehb >= 60
    assert len(min_res.verified_expression.c2h2_locks) == 1
    assert len(min_res.verified_expression.trp_pin_locks) == 0
    assert len(min_res.verified_expression.catalytic_triads) == 0
    assert len(min_res.verified_expression.atp_walker_locks) == 0
    assert len(min_res.verified_expression.transmembrane_rods) == 0
    assert len(min_res.verified_expression.ig_locks) == 0
    assert "VERIFIED=True" in min_cert

    # Full closed-loop chromosome synthesis for all three 3D organ architectures (stages 2, 3, and 4):
    # Target Stage -> Synthesize Z_4 Chromosome -> Blind De Novo Scan -> Autonomous 3D Organ Growth -> Euler/Betti Verify
    chr_vesicle, cert_v = synthesize_chromosome_for_morphology_xray(target_stage_code=2, min_stability_quanta_ehb=48)
    assert chr_vesicle.morphology_verified is True
    assert chr_vesicle.de_novo_scan_verification.autonomous_tissue is not None
    assert chr_vesicle.de_novo_scan_verification.autonomous_tissue.topology_snapshot is not None
    assert chr_vesicle.de_novo_scan_verification.autonomous_tissue.topology_snapshot.euler_chi_2d == 2
    assert len(chr_vesicle.de_novo_scan_verification.autonomous_tissue.living_cells) == 12
    assert "VERIFIED=True" in cert_v

    chr_tube, cert_t = synthesize_chromosome_for_morphology_xray(target_stage_code=3, min_stability_quanta_ehb=48)
    assert chr_tube.morphology_verified is True
    assert chr_tube.de_novo_scan_verification.autonomous_tissue is not None
    assert chr_tube.de_novo_scan_verification.autonomous_tissue.topology_snapshot is not None
    assert chr_tube.de_novo_scan_verification.autonomous_tissue.topology_snapshot.euler_chi_2d == 0
    assert len(chr_tube.de_novo_scan_verification.autonomous_tissue.living_cells) == 24
    assert "VERIFIED=True" in cert_t

    chr_col, cert_c = synthesize_chromosome_for_morphology_xray(target_stage_code=4, min_stability_quanta_ehb=48)
    assert chr_col.morphology_verified is True
    assert len(chr_col.de_novo_scan_verification.expressed_products) == 2
    assert chr_col.de_novo_scan_verification.autonomous_tissue is not None
    assert chr_col.de_novo_scan_verification.autonomous_tissue.topology_snapshot is not None
    assert chr_col.de_novo_scan_verification.autonomous_tissue.topology_snapshot.euler_chi_2d == 41
    assert len(chr_col.de_novo_scan_verification.autonomous_tissue.living_cells) == 42
    assert len(chr_col.de_novo_scan_verification.autonomous_tissue.adhesion_network) == 147
    assert "VERIFIED=True" in cert_c
