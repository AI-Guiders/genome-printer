"""Test suite for the parameter-free FCC N12 cuboctahedral genome interpreter."""

from __future__ import annotations

import math
import numpy as np
import pytest

from genome_printer import (
    CARRIER,
    FCC_N12_VECTORS,
    HEX_N6_VECTORS,
    K4_TETRAHEDRAL_SLOTS,
    fcc_shell_count,
    fcc_ball_count,
    cuboctahedral_subshells,
    first_ionization_energy_ev,
    covalent_radius_angstrom,
    sp3_bond_angle_deg,
    compute_water_thermodynamics,
    watson_crick_hbond_count,
    major_groove_barcode,
    build_bdna_duplex,
    AMINO_ACID_SPECS,
    RUMER_OCTET_I_STEMS,
    RUMER_OCTET_II_STEMS,
    CODON_TABLE,
    rumer_transform,
    translate_rna_or_dna,
    fold_protein_cotranslational,
    scan_c2h2_locks,
    scan_trp_pin_locks,
    scan_ig_synaptic_locks,
    score_ig_synaptic_pair_ev,
    GeneLocus,
    express_gene_locus,
    compute_regulatory_network_matrix_ev,
    step_fcc_morphogen_automaton,
    trace_bdna_bonds,
    trace_protein_folding,
    trace_tf_dna_contacts,
    trace_ig_synaptic_contacts,
    format_xray_report,
)


def test_carrier_identities_and_shells() -> None:
    """Verify exact integer and algebraic identities of the FCC N12 carrier."""
    assert FCC_N12_VECTORS.shape == (12, 3)
    assert HEX_N6_VECTORS.shape == (6, 3)
    assert K4_TETRAHEDRAL_SLOTS.shape == (4, 3)

    # All K_4 tetrahedral slot pairs have dot product -1/d = -1/3
    for i in range(4):
        for j in range(i + 1, 4):
            dot = float(np.dot(K4_TETRAHEDRAL_SLOTS[i], K4_TETRAHEDRAL_SLOTS[j]))
            assert math.isclose(dot, -1.0 / 3.0, abs_tol=1e-12)

    assert CARRIER.mod_bits == 9
    assert CARRIER.n_ring == 512
    assert CARRIER.n_hier == 8
    assert CARRIER.frac_bits == 6
    assert CARRIER.n_hb == 63
    assert CARRIER.delta_phi_disc == 41
    assert CARRIER.n_phi == 13
    assert CARRIER.m_hat == 97

    # Three-way identity for N_HB = 63:
    # d * kappa^2 * N_shell(2) = ((1 - kappa^6)/kappa^6) * floor(B_V) = 2^frac_bits - 1 = 63
    id1 = int(round(CARRIER.d * CARRIER.kappa_sq * fcc_shell_count(2)))
    id2 = int(round(((1.0 - CARRIER.kappa_sq**3) / (CARRIER.kappa_sq**3)) * CARRIER.mod_bits))
    id3 = (1 << CARRIER.frac_bits) - 1
    assert id1 == id2 == id3 == 63

    # Nucleosome 147 bp = N_ball(3) = 1 + 12 + 42 + 92
    assert fcc_shell_count(0) == 1
    assert fcc_shell_count(1) == 12
    assert fcc_shell_count(2) == 42
    assert fcc_shell_count(3) == 92
    assert fcc_ball_count(3) == 147


def test_chemistry_z1_to_16_and_water() -> None:
    """Verify atomic subshells, ionization energies, covalent radii, angles, and water phases."""
    subshells = cuboctahedral_subshells()
    assert [s.spin_capacity for s in subshells] == [2, 6, 10, 14]
    assert sum(s.spin_capacity for s in subshells[:2]) == 8  # Period 2

    # First ionization energies Z=1..10 within ~2% of NIST
    nist_ie = {
        1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298,
        6: 11.260, 7: 14.534, 8: 13.618, 9: 17.423, 10: 21.565,
    }
    for z, ref in nist_ie.items():
        pred = first_ionization_energy_ev(z)
        rel_err = abs(pred - ref) / ref
        assert rel_err < 0.025, f"Z={z}: pred={pred:.3f} vs ref={ref:.3f}"

    # Exact commensurability of peptide C-N bond: r_cov(6) + r_cov(7) = 4 * kappa * a_0 = 2 * R_e(H2)
    r_c = covalent_radius_angstrom(6, bond_order=1)
    r_n = covalent_radius_angstrom(7, bond_order=1)
    assert math.isclose(r_c + r_n, CARRIER.peptide_cn_bond_angstrom, rel_tol=1e-12)
    assert math.isclose(r_c + r_n, 2.0 * CARRIER.h2_bond_length_angstrom, rel_tol=1e-12)

    # Bond angles CH4 (L=0), NH3 (L=1), H2O (L=2)
    assert math.isclose(sp3_bond_angle_deg(0), 109.47122, abs_tol=1e-3)
    assert math.isclose(sp3_bond_angle_deg(1), 106.9577, abs_tol=1e-2)
    assert math.isclose(sp3_bond_angle_deg(2), 104.4775, abs_tol=1e-2)

    # Water phase transitions and BCS gap
    w = compute_water_thermodynamics()
    assert 20.5 <= w.e_hb_kj_mol <= 21.1
    assert 5.8 <= w.delta_h_fus_kj_mol <= 6.1
    assert 272.0 <= w.t_melt_kelvin <= 280.0
    assert 370.0 <= w.t_boil_kelvin <= 375.0
    assert 645.0 <= w.t_crit_kelvin <= 649.0
    assert math.isclose(w.bcs_gap_ratio, 3.536458, abs_tol=1e-4)


def test_bdna_watson_crick_and_barcodes() -> None:
    """Verify Watson-Crick 3-track lock, Major Groove 4-slot barcode, and 147 bp nucleosome."""
    assert watson_crick_hbond_count("G", "C") == 3
    assert watson_crick_hbond_count("A", "T") == 2
    assert watson_crick_hbond_count("A", "C") < 0
    assert watson_crick_hbond_count("G", "T") < 0

    barcodes = major_groove_barcode("ATGC")
    assert len(set(barcodes)) == 4  # All 4 pairs have distinct 4-slot barcodes

    # Build a 147 bp nucleosome-length B-DNA duplex
    seq_147 = ("ATGC" * 37)[:147]
    duplex = build_bdna_duplex(seq_147)
    assert math.isclose(duplex.nucleosome_units, 1.0, abs_tol=1e-12)
    assert duplex.strand1_c1_xyz.shape == (147, 3)


def test_rumer_genetic_code_partition() -> None:
    """Verify 64 -> 21 = N_shell(2)/2 and Rumer's 8 + (12 + 1) = 21 antisymmetry."""
    assert len(CODON_TABLE) == 64
    assert len(AMINO_ACID_SPECS) == 21 == fcc_shell_count(2) // 2
    assert len(RUMER_OCTET_I_STEMS) == 8
    assert len(RUMER_OCTET_II_STEMS) == 8

    # Rumer transformation (U<->G, A<->C) maps Octet I onto Octet II bijectively
    mapped = {rumer_transform(s) for s in RUMER_OCTET_I_STEMS}
    assert mapped == set(RUMER_OCTET_II_STEMS)


def test_cotranslational_o1_folding_and_locks() -> None:
    """Verify O(L) linear-time protein folding and structural lock recognition (C2H2, Trp, Ig)."""
    # Construct a synthetic transcription factor containing both a C2H2 finger (|N_12|=12 loop)
    # and a Homeodomain Trp-pin (W-F-Q-N-R-R)
    tf_protein = "MA" + "C" + "PE" + "C" + "KSFSQSSNLQKH" + "H" + "QRT" + "H" + "G" + "WFQNRRMKWKK"
    folded = fold_protein_cotranslational(tf_protein)

    assert folded.ca_xyz.shape == (len(tf_protein), 3)
    assert folded.elementary_ops_count <= 36 * len(tf_protein)
    assert folded.free_energy_ev < 0.0

    c2h2_hits = scan_c2h2_locks(tf_protein)
    assert len(c2h2_hits) == 1
    assert len(c2h2_hits[0].loop_12_sequence) == 12

    trp_hits = scan_trp_pin_locks(tf_protein)
    assert len(trp_hits) >= 1
    assert trp_hits[0].predicted_dna_core in ("TAAT", "TGTTT")

    # Construct a Donor and Acceptor Ig synaptic pair with canonical loop span N_HB = 63
    # cb at index 2, cf at index 2 + 63 = 65; W at index 14
    donor_seq = "MA" + "C" + ("A" * 11) + "W" + "A" + "R" + ("A" * 48) + "C" + "GG"
    acceptor_seq = "MA" + "C" + ("A" * 11) + "W" + "AAA" + "E" + ("A" * 46) + "C" + "GG"
    d_locks = scan_ig_synaptic_locks(donor_seq)
    a_locks = scan_ig_synaptic_locks(acceptor_seq)
    assert len(d_locks) == 1 and d_locks[0].role == "donor" and d_locks[0].loop_length == 63
    assert len(a_locks) == 1 and a_locks[0].role == "acceptor" and a_locks[0].loop_length == 63
    bind_ev = score_ig_synaptic_pair_ev(d_locks[0], a_locks[0])
    assert bind_ev < 0.0


def test_end_to_end_genome_interpreter_on_fcc() -> None:
    """Verify end-to-end gene expression, regulatory matrix, and 3D FCC N12 morphogen step."""
    # Coding DNA for a mini homeodomain regulator (M-W-F-Q-N-R-R-A-L-A-*)
    cds_reg = "ATGTGGTTCCAGAACCGTCGTGCGCTGGCGTAA"
    locus1 = GeneLocus("Hox_Regulator", promoter_dna="GCGCGCTAATTGTTTGCGCGC", coding_dna=cds_reg)
    locus2 = GeneLocus("Target_Morphogen", promoter_dna="GCGCGCTAATTGCGCGCGCGC", coding_dna=cds_reg)

    prod1 = express_gene_locus(locus1)
    prod2 = express_gene_locus(locus2)
    reg_mat = compute_regulatory_network_matrix_ev([prod1, prod2])
    assert reg_mat.shape == (2, 2)
    assert np.min(reg_mat) < 0.0  # Active Major Groove binding

    grid = np.zeros((6, 6, 6, 2), dtype=np.float64)
    grid[3, 3, 3, 0] = 1.0
    out = step_fcc_morphogen_automaton(grid, reg_mat, n_steps=4)
    assert out.shape == (6, 6, 6, 2)
    assert float(np.sum(out[..., 1])) > 0.0


def test_xray_bond_tracer() -> None:
    """Verify X-Ray bond tracing across B-DNA, O(L) protein folding, Major Groove, and Ig locks."""
    dna_traces = trace_bdna_bonds("ATGC")
    assert len(dna_traces) == 4
    assert dna_traces[0].hbond_count == 2 and dna_traces[0].active_track_indices == (0, 1)
    assert dna_traces[2].hbond_count == 3 and dna_traces[2].active_track_indices == (0, 1, 2)

    tf_protein = "MA" + "C" + "PE" + "C" + "KSFSQSSNLQKH" + "H" + "QRT" + "H" + "G" + "WFQNRRMKWKK"
    prot_report = trace_protein_folding(tf_protein)
    assert len(prot_report.bond_events) > 10
    assert prot_report.total_quanta_ehb > 10
    assert any(ev.bond_type == "backbone_alpha_i_minus_4" for ev in prot_report.bond_events)
    assert any(ev.bond_type == "sidechain_hb" for ev in prot_report.bond_events)
    assert any(ev.bond_type == "disulfide_ss" for ev in prot_report.bond_events)
    formatted = format_xray_report(prot_report)
    assert "X-RAY BOND TRACE" in formatted

    tf_contacts = trace_tf_dna_contacts(("R", "Q", "D"), "GAC")
    assert len(tf_contacts) == 3
    assert all(c.quanta_ehb >= 1 for c in tf_contacts)

    donor_seq = "MA" + "C" + ("A" * 11) + "W" + "A" + "R" + ("A" * 48) + "C" + "GG"
    acceptor_seq = "MA" + "C" + ("A" * 11) + "W" + "AAA" + "E" + ("A" * 46) + "C" + "GG"
    d_lock = scan_ig_synaptic_locks(donor_seq)[0]
    a_lock = scan_ig_synaptic_locks(acceptor_seq)[0]
    ig_contacts = trace_ig_synaptic_contacts(d_lock, a_lock)
    assert len(ig_contacts) == 7
    assert sum(ev.net_quanta_ehb for ev in ig_contacts) > 0
