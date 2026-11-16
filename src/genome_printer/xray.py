"""X-Ray Bond & Lock Tracer for the cuboctahedral FCC N12 genome interpreter.

Because every interaction in `genome-printer` is deterministic and quantized in integer
multiples of E_HB = BE_H / 63 (~0.2160 eV = 20.84 kJ/mol) on the FCC N12 lattice,
we can trace every single bond with exact 3D coordinates, track/slot indices, and
channel provenance ("X-Ray Trace") without black-box probabilities:

  1. B-DNA Duplex X-Ray Trace (`trace_bdna_bonds`):
     - Every base pair i = 0..L-1 with its 3 frontal Watson-Crick tracks (+1/-1/0),
       exact number of closed H-bonds (2 for A-T, 3 for G-C), energy -n_HB * E_HB,
       rear 4-slot Major Groove barcode, and 3D glycosidic C1'-C1' coordinates.
  2. Co-Translational Protein Folding X-Ray Trace (`trace_protein_folding`):
     - Step-by-step ribosome extrusion trace for i = 0..L-1: chosen K_4 groove
       ('alpha', 'beta', 'turn'), exact i -> i-4 (13-atom ring) or i -> i-2 backbone
       H-bond partner, sidechain fork (+1/-1/0) H-bond partner in the 12-neighbor
       cuboctahedral window, disulfide bridges, 3D C_alpha coordinates, and distance in A.
  3. Protein-DNA & Synaptic Lock X-Ray Trace (`trace_tf_dna_contacts`, `trace_ig_synaptic_contacts`):
     - Exact residue-to-base-pair Major Groove slot contacts (which sidechain fork branch
       locks onto which Major Groove slot '+', '-', 'M') and exact 7-offset cuboctahedral
       beta-face contacts between Ig_donor and Ig_acceptor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from .constants import CARRIER, CarrierConstants
from .dna import (
    WATSON_CRICK_TRACKS,
    COMPLEMENT_BASE,
    MAJOR_GROOVE_BARCODE,
    build_bdna_duplex,
)
from .genetic_code import AMINO_ACID_SPECS
from .folding import fold_protein_cotranslational
from .locks import IgSynapticLock


@dataclass(frozen=True)
class DNABasePairBondTrace:
    """X-Ray record of a single base pair in a B-DNA duplex."""

    bp_index: int
    base_coding: str
    base_template: str
    frontal_tracks_coding: tuple[int, int, int]
    frontal_tracks_template: tuple[int, int, int]
    active_track_indices: tuple[int, ...]  # Which of tracks (0, 1, 2) formed H-bonds
    hbond_count: int
    energy_ev: float
    energy_kj_mol: float
    major_groove_barcode: tuple[str, str, str, str]
    c1_coding_xyz: tuple[float, float, float]
    c1_template_xyz: tuple[float, float, float]
    c1_c1_distance_angstrom: float


@dataclass(frozen=True)
class ProteinBondEvent:
    """X-Ray record of an individual bond (backbone H-bond, sidechain H-bond, or S-S bridge)."""

    bond_type: str  # 'backbone_alpha_i_minus_4', 'backbone_beta_i_minus_2', 'sidechain_hb', 'disulfide_ss'
    residue_i: int
    aa_i: str
    residue_j: int
    aa_j: str
    quanta_ehb: int
    energy_ev: float
    energy_kj_mol: float
    ca_i_xyz: tuple[float, float, float]
    ca_j_xyz: tuple[float, float, float]
    ca_distance_angstrom: float
    mechanism: str


@dataclass(frozen=True)
class ProteinFoldingXRayReport:
    """Complete X-Ray trace of co-translational folding for a polypeptide chain."""

    sequence: str
    grooves: tuple[str, ...]
    bond_events: tuple[ProteinBondEvent, ...]
    total_quanta_ehb: int
    total_energy_ev: float
    total_energy_kj_mol: float


@dataclass(frozen=True)
class MajorGrooveContactEvent:
    """X-Ray record of a single amino acid sidechain reading a DNA base pair in the Major Groove."""

    reader_offset: int
    amino_acid: str
    aa_fork: tuple[int, ...]
    bp_index: int
    base_pair: str
    barcode: tuple[str, str, str, str]
    matched_slots: tuple[str, ...]
    quanta_ehb: int
    energy_ev: float


@dataclass(frozen=True)
class SynapticContactEvent:
    """X-Ray record of one of the 7 cuboctahedral offsets in an Ig_donor <-> Ig_acceptor lock."""

    offset_label: str
    donor_residue_index: int
    donor_aa: str
    donor_fork: tuple[int, ...]
    acceptor_residue_index: int
    acceptor_aa: str
    acceptor_fork: tuple[int, ...]
    interaction_kind: str  # 'polar_hbond', 'hydrophobic_pack', 'repulsive_clash', 'neutral'
    net_quanta_ehb: int
    energy_ev: float


def trace_bdna_bonds(sequence: str, c: CarrierConstants = CARRIER) -> tuple[DNABasePairBondTrace, ...]:
    """Generate an X-Ray bond-by-bond trace along a B-DNA double helix."""
    duplex = build_bdna_duplex(sequence, c=c)
    traces: list[DNABasePairBondTrace] = []

    for i, b1 in enumerate(duplex.sequence):
        b2 = COMPLEMENT_BASE[b1]
        t1 = WATSON_CRICK_TRACKS[b1]
        t2 = WATSON_CRICK_TRACKS[b2]
        active_tracks = tuple(idx for idx in range(3) if t1[idx] * t2[idx] == -1)
        hb = len(active_tracks)
        e_ev = -hb * c.e_hb_ev
        xyz1 = tuple(float(x) for x in duplex.strand1_c1_xyz[i])
        xyz2 = tuple(float(x) for x in duplex.strand2_c1_xyz[i])
        dist = float(np.linalg.norm(duplex.strand1_c1_xyz[i] - duplex.strand2_c1_xyz[i]))

        traces.append(
            DNABasePairBondTrace(
                bp_index=i,
                base_coding=b1,
                base_template=b2,
                frontal_tracks_coding=t1,
                frontal_tracks_template=t2,
                active_track_indices=active_tracks,
                hbond_count=hb,
                energy_ev=e_ev,
                energy_kj_mol=e_ev * c.ev_to_kj_mol,
                major_groove_barcode=MAJOR_GROOVE_BARCODE[(b1, b2)],
                c1_coding_xyz=xyz1,
                c1_template_xyz=xyz2,
                c1_c1_distance_angstrom=dist,
            )
        )
    return tuple(traces)


def trace_protein_folding(sequence: str, c: CarrierConstants = CARRIER) -> ProteinFoldingXRayReport:
    """Generate a complete X-Ray trace of every backbone, sidechain, and disulfide bond formed during O(L) folding."""
    folded = fold_protein_cotranslational(sequence, c=c)
    seq = folded.sequence
    grooves = folded.grooves
    ca = folded.ca_xyz
    events: list[ProteinBondEvent] = []

    for i in range(1, len(seq)):
        g = grooves[i]
        xyz_i = tuple(float(x) for x in ca[i])

        # 1. Backbone H-bond check
        if g == "alpha" and i >= c.d_in_sq and grooves[i - c.d_in_sq] == "alpha" and seq[i] != "P":
            j = i - c.d_in_sq
            xyz_j = tuple(float(x) for x in ca[j])
            dist = float(np.linalg.norm(ca[i] - ca[j]))
            events.append(
                ProteinBondEvent(
                    bond_type="backbone_alpha_i_minus_4",
                    residue_i=i,
                    aa_i=seq[i],
                    residue_j=j,
                    aa_j=seq[j],
                    quanta_ehb=1,
                    energy_ev=-c.e_hb_ev,
                    energy_kj_mol=-c.e_hb_kj_mol,
                    ca_i_xyz=xyz_i,
                    ca_j_xyz=xyz_j,
                    ca_distance_angstrom=dist,
                    mechanism="13-atom Pauling-Corey 3.6_13 C=O(i-4)...H-N(i) bridge",
                )
            )
        elif g == "beta" and i >= 2 and grooves[i - 2] == "beta":
            j = i - 2
            xyz_j = tuple(float(x) for x in ca[j])
            dist = float(np.linalg.norm(ca[i] - ca[j]))
            events.append(
                ProteinBondEvent(
                    bond_type="backbone_beta_i_minus_2",
                    residue_i=i,
                    aa_i=seq[i],
                    residue_j=j,
                    aa_j=seq[j],
                    quanta_ehb=1,
                    energy_ev=-c.e_hb_ev,
                    energy_kj_mol=-c.e_hb_kj_mol,
                    ca_i_xyz=xyz_i,
                    ca_j_xyz=xyz_j,
                    ca_distance_angstrom=dist,
                    mechanism="{111} pleated beta-strand register lock",
                )
            )

        # 2. Sidechain H-bond fork check in local 12-neighbor window
        fork_i = AMINO_ACID_SPECS[seq[i]].hb_fork
        if any(s != 0 for s in fork_i):
            for prev in range(max(0, i - c.n_12), i):
                fork_p = AMINO_ACID_SPECS[seq[prev]].hb_fork
                if any(s1 * s2 == -1 for s1 in fork_i for s2 in fork_p):
                    xyz_p = tuple(float(x) for x in ca[prev])
                    dist = float(np.linalg.norm(ca[i] - ca[prev]))
                    events.append(
                        ProteinBondEvent(
                            bond_type="sidechain_hb",
                            residue_i=i,
                            aa_i=seq[i],
                            residue_j=prev,
                            aa_j=seq[prev],
                            quanta_ehb=1,
                            energy_ev=-c.e_hb_ev,
                            energy_kj_mol=-c.e_hb_kj_mol,
                            ca_i_xyz=xyz_i,
                            ca_j_xyz=xyz_p,
                            ca_distance_angstrom=dist,
                            mechanism=f"Polar fork {seq[i]}{list(fork_i)} <-> {seq[prev]}{list(fork_p)} in |N_12| window",
                        )
                    )
                    break

    # 3. Disulfide bridges
    for i_c, j_c in folded.disulfide_bridges:
        xyz_i = tuple(float(x) for x in ca[i_c])
        xyz_j = tuple(float(x) for x in ca[j_c])
        dist = float(np.linalg.norm(ca[i_c] - ca[j_c]))
        events.append(
            ProteinBondEvent(
                bond_type="disulfide_ss",
                residue_i=j_c,
                aa_i="C",
                residue_j=i_c,
                aa_j="C",
                quanta_ehb=2,
                energy_ev=-2.0 * c.e_hb_ev,
                energy_kj_mol=-2.0 * c.e_hb_kj_mol,
                ca_i_xyz=xyz_j,
                ca_j_xyz=xyz_i,
                ca_distance_angstrom=dist,
                mechanism=f"Period-3 Cys-Cys disulfide lock (span={j_c - i_c})",
            )
        )

    total_q = sum(ev.quanta_ehb for ev in events)
    return ProteinFoldingXRayReport(
        sequence=seq,
        grooves=grooves,
        bond_events=tuple(events),
        total_quanta_ehb=total_q,
        total_energy_ev=-total_q * c.e_hb_ev,
        total_energy_kj_mol=-total_q * c.e_hb_kj_mol,
    )


def trace_tf_dna_contacts(
    reader_residues: tuple[str, ...],
    dna_subsite: str,
    start_bp_index: int = 0,
    c: CarrierConstants = CARRIER,
) -> tuple[MajorGrooveContactEvent, ...]:
    """Trace every individual sidechain-to-Major-Groove slot contact for a transcription factor."""
    seq = dna_subsite.strip().upper()
    if len(reader_residues) != len(seq):
        raise ValueError("Length of reader_residues must match length of dna_subsite.")

    contacts: list[MajorGrooveContactEvent] = []
    for k, (aa, base) in enumerate(zip(reader_residues, seq)):
        comp = COMPLEMENT_BASE[base]
        bc = MAJOR_GROOVE_BARCODE[(base, comp)]
        fork = AMINO_ACID_SPECS[aa.upper()].hb_fork
        matched: list[str] = []
        for s in fork:
            if s == +1 and "-" in bc:
                matched.append("donor(+) -> acceptor(-)")
            elif s == -1 and "+" in bc:
                matched.append("acceptor(-) -> donor(+)")
            elif s == 0 and "M" in bc:
                matched.append("hydrophobic(0) -> methyl(M)")
        q = len(matched)
        contacts.append(
            MajorGrooveContactEvent(
                reader_offset=k,
                amino_acid=aa.upper(),
                aa_fork=fork,
                bp_index=start_bp_index + k,
                base_pair=f"{base}-{comp}",
                barcode=bc,
                matched_slots=tuple(matched),
                quanta_ehb=q,
                energy_ev=-q * c.e_hb_ev,
            )
        )
    return tuple(contacts)


def trace_ig_synaptic_contacts(
    donor: IgSynapticLock,
    acceptor: IgSynapticLock,
    c: CarrierConstants = CARRIER,
) -> tuple[SynapticContactEvent, ...]:
    """Trace all 7 cuboctahedral contact offsets between an Ig_donor key and Ig_acceptor socket."""
    labels = (
        "W + d (3)",
        "W + d_in(sq) (4)",
        "W + |sq| (6)",
        "W + |N_12| (12)",
        "W + N_phi (13)",
        "Cys_F - |N_12| (-12)",
        "Cys_F - d_out(E) (-5)",
    )
    donor_indices = (
        donor.trp_pin_index + c.d,
        donor.trp_pin_index + c.d_in_sq,
        donor.trp_pin_index + c.n_squares,
        donor.trp_pin_index + c.n_12,
        donor.trp_pin_index + c.n_phi,
        donor.cys_f_index - c.n_12,
        donor.cys_f_index - c.d_out_e,
    )
    acceptor_indices = (
        acceptor.trp_pin_index + c.d,
        acceptor.trp_pin_index + c.d_in_sq,
        acceptor.trp_pin_index + c.n_squares,
        acceptor.trp_pin_index + c.n_12,
        acceptor.trp_pin_index + c.n_phi,
        acceptor.cys_f_index - c.n_12,
        acceptor.cys_f_index - c.d_out_e,
    )

    events: list[SynapticContactEvent] = []
    for lbl, idx_d, idx_a, aa_d, aa_a in zip(
        labels,
        donor_indices,
        acceptor_indices,
        donor.contact_signature,
        acceptor.contact_signature,
    ):
        if aa_d == "-" or aa_a == "-":
            continue
        fork_d = AMINO_ACID_SPECS[aa_d].hb_fork
        fork_a = AMINO_ACID_SPECS[aa_a].hb_fork
        polar_pairs = sum(1 for s1 in fork_d for s2 in fork_a if s1 * s2 == -1)
        polar_clashes = sum(1 for s1 in fork_d for s2 in fork_a if s1 * s2 == +1)
        hydro = 1 if (0 in fork_d and 0 in fork_a and polar_clashes == 0) else 0
        net_q = polar_pairs + hydro - polar_clashes

        if polar_pairs > 0 and polar_clashes == 0:
            kind = "polar_hbond"
        elif hydro > 0 and polar_clashes == 0:
            kind = "hydrophobic_pack"
        elif polar_clashes > polar_pairs:
            kind = "repulsive_clash"
        else:
            kind = "neutral"

        events.append(
            SynapticContactEvent(
                offset_label=lbl,
                donor_residue_index=idx_d,
                donor_aa=aa_d,
                donor_fork=fork_d,
                acceptor_residue_index=idx_a,
                acceptor_aa=aa_a,
                acceptor_fork=fork_a,
                interaction_kind=kind,
                net_quanta_ehb=net_q,
                energy_ev=-net_q * c.e_hb_ev,
            )
        )
    return tuple(events)


def format_xray_report(protein_report: ProteinFoldingXRayReport) -> str:
    """Format a human-readable X-Ray bond table for a folded protein."""
    lines = [
        f"=== X-RAY BOND TRACE ({len(protein_report.sequence)} aa, "
        f"{len(protein_report.bond_events)} bonds, "
        f"total = {protein_report.total_quanta_ehb} E_HB = {protein_report.total_energy_kj_mol:.2f} kJ/mol) ===",
        f"{'Type':26s} | {'Pair':12s} | {'d(CA-CA)':9s} | {'Energy':14s} | {'Mechanism'}",
        "-" * 96,
    ]
    for ev in protein_report.bond_events:
        pair_str = f"{ev.aa_i}{ev.residue_i+1}<->{ev.aa_j}{ev.residue_j+1}"
        lines.append(
            f"{ev.bond_type:26s} | {pair_str:12s} | {ev.ca_distance_angstrom:7.2f} A | "
            f"{ev.energy_kj_mol:8.2f} kJ/mol | {ev.mechanism}"
        )
    return "\n".join(lines)
