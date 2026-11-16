"""M-layer cellular automaton, gene regulatory network, and 3D tissue morphogenesis on FCC N12.

Zero T-layer strings, zero human labels, zero SI dimensional constants.
All gene loci, cell states, division lineages, and adhesion/synaptic contacts are
represented purely by integer indices, 3-bit octant masks (Z_8), integer shell/lamina
numbers, surface H-bond forks (-1, 0, +1), and integer E_HB / E_0 quanta.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .m_carrier import (
    M_CARRIER,
    MCarrierConstants,
    FCC_N12_VECTORS,
    HEX_N6_VECTORS,
)
from .m_nucleic import (
    MDuplexLatticeState,
    m_build_duplex,
    m_translate_codons,
)
from .m_polypeptide import (
    MFoldedChain,
    MC2H2Lock,
    MTrpPinLock,
    MIgLock,
    m_fold_chain_cotranslational,
    m_scan_c2h2_locks,
    m_scan_trp_pin_locks,
    m_scan_ig_locks,
    m_score_ig_lock_pair_quanta,
    m_score_tf_dna_quanta,
)


@dataclass(frozen=True)
class MGeneLocus:
    """Pure M-layer genomic locus with 2-bit promoter and 2-bit coding sequences in Z_4."""

    locus_id: int
    promoter_bases_z4: tuple[int, ...]
    coding_bases_z4: tuple[int, ...]


@dataclass(frozen=True)
class MExpressedLocus:
    """Pure M-layer expressed gene product."""

    locus_id: int
    promoter_duplex: MDuplexLatticeState
    coding_duplex: MDuplexLatticeState
    class_sequence_z21: tuple[int, ...]
    folded_chain: MFoldedChain
    c2h2_locks: tuple[MC2H2Lock, ...]
    trp_pin_locks: tuple[MTrpPinLock, ...]
    ig_locks: tuple[MIgLock, ...]


def m_express_locus(locus: MGeneLocus, c: MCarrierConstants = M_CARRIER) -> MExpressedLocus:
    """Execute the pure M-layer Z_4 -> Z_64 -> Z_21 -> O(L) K_4 fold -> lock scan pipeline."""
    prom = m_build_duplex(locus.promoter_bases_z4, c=c)
    cds = m_build_duplex(locus.coding_bases_z4, c=c)
    seq_z21 = m_translate_codons(locus.coding_bases_z4, stop_at_20=True)
    folded = m_fold_chain_cotranslational(seq_z21, c=c)
    c2h2 = m_scan_c2h2_locks(seq_z21, c=c)
    trp = m_scan_trp_pin_locks(seq_z21)
    ig = m_scan_ig_locks(seq_z21, c=c)
    return MExpressedLocus(
        locus_id=locus.locus_id,
        promoter_duplex=prom,
        coding_duplex=cds,
        class_sequence_z21=seq_z21,
        folded_chain=folded,
        c2h2_locks=c2h2,
        trp_pin_locks=trp,
        ig_locks=ig,
    )


def m_compute_regulatory_quanta_matrix(products: list[MExpressedLocus]) -> np.ndarray:
    """Compute the N_loci x N_loci integer E_HB binding quanta matrix on Major Groove slots."""
    n = len(products)
    mat = np.zeros((n, n), dtype=np.int64)
    for i, reg in enumerate(products):
        for j, tgt in enumerate(products):
            best_q = 0
            prom_z4 = tgt.promoter_duplex.coding_bases
            for finger in reg.c2h2_locks:
                readers = finger.reader_classes
                for k in range(max(0, len(prom_z4) - 2)):
                    sub = prom_z4[k : k + 3]
                    q = m_score_tf_dna_quanta(readers, sub)
                    if q > best_q:
                        best_q = q
            for pin in reg.trp_pin_locks:
                core = pin.target_core_z4
                clen = len(core)
                for k in range(max(0, len(prom_z4) - clen + 1)):
                    if prom_z4[k : k + clen] == core:
                        if clen > best_q:
                            best_q = clen
            mat[i, j] = best_q
    return mat


def m_step_fcc_morphogen_automaton(
    state: np.ndarray,
    regulatory_quanta_matrix: np.ndarray,
    n_steps: int = 10,
    c: MCarrierConstants = M_CARRIER,
) -> np.ndarray:
    """Advance a 3D FCC N12 morphogen field with exact link coupling kappa_link = 1 / |N_12| = 1 / 12."""
    cur = np.array(state, dtype=np.float64, copy=True)
    kappa_link = 1.0 / c.n_12
    w_reg = regulatory_quanta_matrix.astype(np.float64) / float(c.atp_in_ehb_quanta * c.n_12)

    for _ in range(n_steps):
        nb_sum = np.zeros_like(cur)
        for idx in range(c.n_12):
            dz = int(FCC_N12_VECTORS[idx, 0])
            dy = int(FCC_N12_VECTORS[idx, 1])
            dx = int(FCC_N12_VECTORS[idx, 2])
            nb_sum += np.roll(cur, shift=(dz, dy, dx), axis=(0, 1, 2))
        fcc_avg = kappa_link * nb_sum
        drive = fcc_avg @ w_reg
        cur = np.clip(fcc_avg + drive, 0.0, 1.0)
    return cur


@dataclass(frozen=True)
class MCellNode:
    """Pure M-layer cell occupying a rhombic dodecahedral node on the 3D FCC lattice."""

    cell_id: int
    parent_id: int
    birth_tick: int
    position_ijk: tuple[int, int, int]
    octant_mask: int  # 0..7 in Z_8
    lamina_index: int  # Integer coordination shell / layer index
    active_loci: tuple[int, ...]
    surface_fork: tuple[int, ...]
    surface_ig_lock: MIgLock | None
    atp_quanta: int


@dataclass(frozen=True)
class MDivisionEvent:
    """M-layer mitotic division record."""

    tick: int
    parent_id: int
    daughter_id: int
    parent_ijk: tuple[int, int, int]
    daughter_ijk: tuple[int, int, int]
    fcc_direction: tuple[int, int, int]
    in_111_slice: int  # 1 if in N_6 {111} plane, 0 if apical-basal normal
    atp_quanta_spent: int
    energy_spent_e0: float


@dataclass(frozen=True)
class MCellContact:
    """M-layer H-bond lock across a shared FCC N12 face between two cells."""

    cell_a_id: int
    cell_b_id: int
    lamina_a: int
    lamina_b: int
    pos_a_ijk: tuple[int, int, int]
    pos_b_ijk: tuple[int, int, int]
    fcc_link_vector: tuple[int, int, int]
    contact_mode: int  # 0: surface fork homodimer, 1: Ig_donor <-> Ig_acceptor 7-offset lock
    quanta_ehb: int
    binding_energy_e0: float


@dataclass(frozen=True)
class MCavitationEvent:
    """M-layer programmed lumen cavitation at an enclosed interior node."""

    tick: int
    cell_id: int
    position_ijk: tuple[int, int, int]
    enclosed_faces: int


@dataclass(frozen=True)
class MTissueMorphology:
    """Complete M-layer 3D tissue structure and developmental trace in native units."""

    morphology_code: int  # 0: S_1 vesicle, 1: N_6 tube, 2: |square|=6 cortical column
    n_laminae: int
    living_cells: tuple[MCellNode, ...]
    division_lineage: tuple[MDivisionEvent, ...]
    cavitation_events: tuple[MCavitationEvent, ...]
    adhesion_network: tuple[MCellContact, ...]
    ig_synaptic_network: tuple[MCellContact, ...]
    lumen_nodes: tuple[tuple[int, int, int], ...]
    total_quanta_ehb: int
    total_energy_e0: float


def m_octant_mask(pos_ijk: tuple[int, int, int], center_ijk: tuple[int, int, int] = (0, 0, 0)) -> int:
    """Compute the 3-bit octant mask in Z_8 (0..7) relative to a center node."""
    bx = 1 if pos_ijk[0] >= center_ijk[0] else 0
    by = 1 if pos_ijk[1] >= center_ijk[1] else 0
    bz = 1 if pos_ijk[2] >= center_ijk[2] else 0
    return (bz << 2) | (by << 1) | bx


def _m_build_adhesion_network(
    cells: list[MCellNode],
    c: MCarrierConstants = M_CARRIER,
) -> tuple[list[MCellContact], list[MCellContact]]:
    """Scan all 12-neighbor faces between living cells and compute integer E_HB locks."""
    pos_map: dict[tuple[int, int, int], MCellNode] = {cell.position_ijk: cell for cell in cells}
    all_contacts: list[MCellContact] = []
    ig_contacts: list[MCellContact] = []
    seen: set[tuple[int, int]] = set()

    for cell_a in cells:
        x, y, z = cell_a.position_ijk
        for idx in range(c.n_12):
            dz = int(FCC_N12_VECTORS[idx, 0])
            dy = int(FCC_N12_VECTORS[idx, 1])
            dx = int(FCC_N12_VECTORS[idx, 2])
            nb_pos = (x + dx, y + dy, z + dz)
            cell_b = pos_map.get(nb_pos)
            if cell_b is None:
                continue
            pair_key = (min(cell_a.cell_id, cell_b.cell_id), max(cell_a.cell_id, cell_b.cell_id))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            link_vec = (dx, dy, dz)

            if (
                cell_a.surface_ig_lock is not None
                and cell_b.surface_ig_lock is not None
                and cell_a.surface_ig_lock.polarity_role * cell_b.surface_ig_lock.polarity_role == -1
            ):
                donor = cell_a.surface_ig_lock if cell_a.surface_ig_lock.polarity_role == +1 else cell_b.surface_ig_lock
                acceptor = (
                    cell_b.surface_ig_lock if cell_b.surface_ig_lock.polarity_role == -1 else cell_a.surface_ig_lock
                )
                q_hb = max(1, m_score_ig_lock_pair_quanta(donor, acceptor))
                ev = MCellContact(
                    cell_a_id=cell_a.cell_id,
                    cell_b_id=cell_b.cell_id,
                    lamina_a=cell_a.lamina_index,
                    lamina_b=cell_b.lamina_index,
                    pos_a_ijk=cell_a.position_ijk,
                    pos_b_ijk=cell_b.position_ijk,
                    fcc_link_vector=link_vec,
                    contact_mode=1,
                    quanta_ehb=q_hb,
                    binding_energy_e0=-q_hb * c.e_hb_e0,
                )
                ig_contacts.append(ev)
                all_contacts.append(ev)
                continue

            # Trans-antiparallel pairing of surface H-bond forks across the shared rhombic face
            q_hb = sum(1 for s1, s2 in zip(cell_a.surface_fork, cell_b.surface_fork) if s1 * (-s2) == -1)
            if q_hb > 0:
                all_contacts.append(
                    MCellContact(
                        cell_a_id=cell_a.cell_id,
                        cell_b_id=cell_b.cell_id,
                        lamina_a=cell_a.lamina_index,
                        lamina_b=cell_b.lamina_index,
                        pos_a_ijk=cell_a.position_ijk,
                        pos_b_ijk=cell_b.position_ijk,
                        fcc_link_vector=link_vec,
                        contact_mode=0,
                        quanta_ehb=q_hb,
                        binding_energy_e0=-q_hb * c.e_hb_e0,
                    )
                )

    return all_contacts, ig_contacts


def m_grow_vesicle_s1(c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a hollow S_1 cuboctahedral vesicle (N_shell(1)=12 cells, 24 edges) in native M units."""
    fork_d3 = (+1, -1, +1)
    founder = MCellNode(0, -1, 0, (0, 0, 0), m_octant_mask((0, 0, 0)), 0, (0,), fork_d3, None, 24)
    cells: list[MCellNode] = [founder]
    divisions: list[MDivisionEvent] = []
    next_id = 1

    for idx in range(c.n_12):
        dz = int(FCC_N12_VECTORS[idx, 0])
        dy = int(FCC_N12_VECTORS[idx, 1])
        dx = int(FCC_N12_VECTORS[idx, 2])
        pos = (dx, dy, dz)
        in_111 = 1 if (dx + dy + dz) == 0 else 0
        cells.append(
            MCellNode(next_id, 0, 1, pos, m_octant_mask(pos), 1, (0, 1), fork_d3, None, c.atp_in_ehb_quanta)
        )
        divisions.append(MDivisionEvent(1, 0, next_id, (0, 0, 0), pos, pos, in_111, 1, c.atp_e0))
        next_id += 1

    pos_set = {cell.position_ijk for cell in cells}
    cavitations: list[MCavitationEvent] = []
    living: list[MCellNode] = []
    lumen: list[tuple[int, int, int]] = []

    for cell in cells:
        x, y, z = cell.position_ijk
        enc = sum(
            1
            for k in range(c.n_12)
            if (x + int(FCC_N12_VECTORS[k, 2]), y + int(FCC_N12_VECTORS[k, 1]), z + int(FCC_N12_VECTORS[k, 0]))
            in pos_set
        )
        if enc == c.n_12:
            cavitations.append(MCavitationEvent(2, cell.cell_id, cell.position_ijk, enc))
            lumen.append(cell.position_ijk)
        else:
            living.append(cell)

    adhesion, synapses = _m_build_adhesion_network(living, c=c)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        0, 1, tuple(living), tuple(divisions), tuple(cavitations), tuple(adhesion), tuple(synapses), tuple(lumen), tot_q, -tot_q * c.e_hb_e0
    )


def m_grow_hex_tube(axial_rings: int = 4, c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a hollow tube by stacking `axial_rings` hexagonal {111} N_6 rings along FCC (1, 1, 0)."""
    axial_step = (1, 1, 0)
    cells: list[MCellNode] = []
    divisions: list[MDivisionEvent] = []
    cavitations: list[MCavitationEvent] = []
    lumen: list[tuple[int, int, int]] = []
    next_id = 0

    for r_idx in range(axial_rings):
        center = (r_idx * axial_step[0], r_idx * axial_step[1], r_idx * axial_step[2])
        lumen.append(center)
        core_id = next_id
        next_id += 1

        for k in range(c.n_6):
            dx = int(HEX_N6_VECTORS[k, 0])
            dy = int(HEX_N6_VECTORS[k, 1])
            dz = int(HEX_N6_VECTORS[k, 2])
            pos = (center[0] + dx, center[1] + dy, center[2] + dz)
            fork = (+1, -1, +1) if dz < 0 else ((-1, +1, -1) if dz > 0 else (+1, +1, -1))
            loci = (0, 1) if dz < 0 else ((0, 2) if dz > 0 else (0, 3))
            cells.append(
                MCellNode(next_id, core_id, r_idx + 1, pos, m_octant_mask(pos, center), r_idx, loci, fork, None, 2)
            )
            divisions.append(
                MDivisionEvent(r_idx + 1, core_id, next_id, center, pos, (dx, dy, dz), 1, 1, c.atp_e0)
            )
            next_id += 1

        cavitations.append(MCavitationEvent(r_idx + 2, core_id, center, c.n_6))

    adhesion, synapses = _m_build_adhesion_network(cells, c=c)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        1, axial_rings, tuple(cells), tuple(divisions), tuple(cavitations), tuple(adhesion), tuple(synapses), tuple(lumen), tot_q, -tot_q * c.e_hb_e0
    )


def m_grow_laminar_column_6(c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a |square| = 6-lamina column (6 x 7 = 42 = N_shell(2) cells, 147 = N_ball(3) contacts)."""
    n_layers = c.n_squares  # 6
    donor_ig = MIgLock(+1, 2, 2 + c.n_hb, c.n_hb, 14, (0, 2, 0, 19, 1, 10, 3))
    acceptor_ig = MIgLock(-1, 2, 2 + c.n_hb, c.n_hb, 14, (4, 18, 8, 2, 19, 9, 6))

    cells: list[MCellNode] = []
    divisions: list[MDivisionEvent] = []
    next_id = 0
    vertical_step = (1, 1, 0)
    prev_center_id = -1
    prev_center_pos = (0, 0, 0)
    fork_d3 = (+1, -1, +1)

    for l_idx in range(n_layers):
        center = (l_idx * vertical_step[0], l_idx * vertical_step[1], l_idx * vertical_step[2])
        ig_lock = donor_ig if (l_idx % 2 == 0) else acceptor_ig

        center_id = next_id
        next_id += 1
        cells.append(
            MCellNode(center_id, prev_center_id, l_idx + 1, center, m_octant_mask(center), l_idx, (l_idx,), fork_d3, ig_lock, 4)
        )
        if prev_center_id != -1:
            divisions.append(
                MDivisionEvent(l_idx + 1, prev_center_id, center_id, prev_center_pos, center, vertical_step, 0, 1, c.atp_e0)
            )
        prev_center_id = center_id
        prev_center_pos = center

        for k in range(c.n_6):
            dx = int(HEX_N6_VECTORS[k, 0])
            dy = int(HEX_N6_VECTORS[k, 1])
            dz = int(HEX_N6_VECTORS[k, 2])
            pos = (center[0] + dx, center[1] + dy, center[2] + dz)
            cells.append(
                MCellNode(next_id, center_id, l_idx + 1, pos, m_octant_mask(pos, center), l_idx, (l_idx,), fork_d3, ig_lock, 2)
            )
            divisions.append(
                MDivisionEvent(l_idx + 1, center_id, next_id, center, pos, (dx, dy, dz), 1, 1, c.atp_e0)
            )
            next_id += 1

    adhesion, synapses = _m_build_adhesion_network(cells, c=c)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        2, n_layers, tuple(cells), tuple(divisions), (), tuple(adhesion), tuple(synapses), (), tot_q, -tot_q * c.e_hb_e0
    )
