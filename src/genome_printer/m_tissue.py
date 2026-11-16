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
    M_CODON_TO_CLASS_21,
    MDuplexLatticeState,
    MIntronSplicingSpan,
    m_build_duplex,
    m_translate_codons,
    m_reverse_translate_classes_to_z4,
    m_splice_pre_mrna,
    m_nucleosome_accessibility_profile,
)
from .m_polypeptide import (
    MFoldedChain,
    MC2H2Lock,
    MTrpPinLock,
    MIgLock,
    MCatalyticTriadLock,
    MAtpWalkerLock,
    MTransmembraneHelixLock,
    MProteinSynthesisSpec,
    m_fold_chain_cotranslational,
    m_scan_c2h2_locks,
    m_scan_trp_pin_locks,
    m_scan_ig_locks,
    m_scan_catalytic_triads,
    m_scan_atp_walker_locks,
    m_scan_transmembrane_rods,
    m_score_ig_lock_pair_quanta,
    m_score_tf_dna_quanta,
    m_synthesize_protein_for_spec,
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
    spliced_introns: tuple[MIntronSplicingSpan, ...]
    class_sequence_z21: tuple[int, ...]
    folded_chain: MFoldedChain
    c2h2_locks: tuple[MC2H2Lock, ...]
    trp_pin_locks: tuple[MTrpPinLock, ...]
    ig_locks: tuple[MIgLock, ...]
    catalytic_triads: tuple[MCatalyticTriadLock, ...]
    atp_walker_locks: tuple[MAtpWalkerLock, ...]
    transmembrane_rods: tuple[MTransmembraneHelixLock, ...]


def m_express_locus(
    locus: MGeneLocus,
    enable_splicing: bool = True,
    c: MCarrierConstants = M_CARRIER,
) -> MExpressedLocus:
    """Execute the pure M-layer Z_4 -> splice -> Z_64 -> Z_21 -> O(L) K_4 fold -> lock scan pipeline."""
    prom = m_build_duplex(locus.promoter_bases_z4, c=c)
    if enable_splicing:
        mature_z4, introns = m_splice_pre_mrna(locus.coding_bases_z4, c=c)
    else:
        mature_z4, introns = locus.coding_bases_z4, ()
    cds = m_build_duplex(mature_z4, c=c)
    seq_z21 = m_translate_codons(mature_z4, stop_at_20=True)
    folded = m_fold_chain_cotranslational(seq_z21, c=c)
    c2h2 = m_scan_c2h2_locks(seq_z21, c=c)
    trp = m_scan_trp_pin_locks(seq_z21)
    ig = m_scan_ig_locks(seq_z21, c=c)
    triads = m_scan_catalytic_triads(seq_z21, c=c)
    walkers = m_scan_atp_walker_locks(seq_z21, c=c)
    tm_rods = m_scan_transmembrane_rods(seq_z21, c=c)
    return MExpressedLocus(
        locus_id=locus.locus_id,
        promoter_duplex=prom,
        coding_duplex=cds,
        spliced_introns=introns,
        class_sequence_z21=seq_z21,
        folded_chain=folded,
        c2h2_locks=c2h2,
        trp_pin_locks=trp,
        ig_locks=ig,
        catalytic_triads=triads,
        atp_walker_locks=walkers,
        transmembrane_rods=tm_rods,
    )


def m_compute_regulatory_quanta_matrix(products: list[MExpressedLocus]) -> np.ndarray:
    """Compute the N_loci x N_loci integer E_HB binding quanta matrix on Major Groove slots."""
    n = len(products)
    mat = np.zeros((n, n), dtype=np.int64)
    for i, reg in enumerate(products):
        for j, tgt in enumerate(products):
            total_q = 0
            prom_z4 = tgt.promoter_duplex.coding_bases
            for finger in reg.c2h2_locks:
                readers = finger.reader_classes
                best_finger_q = 0
                for k in range(max(0, len(prom_z4) - 2)):
                    sub = prom_z4[k : k + 3]
                    q = m_score_tf_dna_quanta(readers, sub)
                    if q > best_finger_q:
                        best_finger_q = q
                total_q += best_finger_q
            for pin in reg.trp_pin_locks:
                core = pin.target_core_z4
                clen = len(core)
                for k in range(max(0, len(prom_z4) - clen + 1)):
                    if prom_z4[k : k + clen] == core:
                        total_q += clen
                        break
            mat[i, j] = total_q
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
class MTissueTopologySnapshot:
    """Pure M-layer simplicial and Betti/Euler topological invariants on FCC N12 at a growth tick.

    Carrier stage code (`carrier_stage_code`, inferred strictly from M-invariants):
      0: Single founder zygote / progenitor (`V = 1, E = 0`)
      1: Solid 3D morula ball `B^3` (`N_K4 > 0, V_lumen = 0, E_Ig = 0`)
      2: Hollow spherical vesicle `S^2` (`N_K4 = 0, chi_2D = 2, beta_2 >= 1`)
      3: Hollow cylindrical tube `S^1 x I` (`N_K4 = 0, chi_2D = 0, V_lumen >= 1`)
      4: Stratified 3D laminar column (`E_Ig > 0` or planar 7-cell wheel `V = 7, E = 12, V_lumen = 0`)
    """

    tick: int
    v_cells: int
    e_contacts: int
    f_triangles: int
    f_squares_surface: int
    k4_tetrahedra: int
    euler_chi_2d: int
    betti_0: int
    betti_1_open_channels: int
    betti_2_closed_cavities: int
    lumen_nodes_count: int
    ig_synapses_count: int
    carrier_stage_code: int


@dataclass(frozen=True)
class MTissueMorphology:
    """Complete M-layer 3D tissue structure, developmental trace, and topological telemetry."""

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
    topology_snapshot: MTissueTopologySnapshot | None = None
    growth_trajectory: tuple[MTissueTopologySnapshot, ...] = ()


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


def m_analyze_tissue_topology(
    cells: tuple[MCellNode, ...] | list[MCellNode],
    adhesion: tuple[MCellContact, ...] | list[MCellContact],
    synapses: tuple[MCellContact, ...] | list[MCellContact],
    lumen_nodes: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
    tick: int = 0,
    c: MCarrierConstants = M_CARRIER,
) -> MTissueTopologySnapshot:
    """Compute intrinsic simplicial, Euler, and Betti topological invariants on the FCC N12 graph."""
    v_cnt = len(cells)
    e_cnt = len(adhesion)
    if v_cnt == 0:
        return MTissueTopologySnapshot(tick, 0, 0, 0, 0, 0, 0, 0, 0, 0, len(lumen_nodes), 0, 0)

    ids = [cell.cell_id for cell in cells]
    pos_by_id = {cell.cell_id: cell.position_ijk for cell in cells}
    pos_set = set(pos_by_id.values())
    adj: dict[int, set[int]] = {cid: set() for cid in ids}
    for ev in adhesion:
        adj[ev.cell_a_id].add(ev.cell_b_id)
        adj[ev.cell_b_id].add(ev.cell_a_id)

    # Connected components beta_0 via BFS
    visited: set[int] = set()
    beta_0 = 0
    for start_id in ids:
        if start_id in visited:
            continue
        beta_0 += 1
        queue = [start_id]
        visited.add(start_id)
        while queue:
            cur = queue.pop()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

    # 3-cliques (triangles F_triangle) and 4-cliques (solid tetrahedra K_4)
    tris: list[tuple[int, int, int]] = []
    n_ids = len(ids)
    for i in range(n_ids):
        u = ids[i]
        for j in range(i + 1, n_ids):
            v = ids[j]
            if v not in adj[u]:
                continue
            for k in range(j + 1, n_ids):
                w = ids[k]
                if w in adj[u] and w in adj[v]:
                    tris.append((u, v, w))

    f_tri = len(tris)
    id_to_idx = {cid: idx for idx, cid in enumerate(ids)}
    k4_cnt = 0
    for u, v, w in tris:
        start_m = id_to_idx[w] + 1
        for m_idx in range(start_m, n_ids):
            q = ids[m_idx]
            if q in adj[u] and q in adj[v] and q in adj[w]:
                k4_cnt += 1

    # Uncapped surface squares F_square (4-cycles of side sqrt(2) and diagonal 2 with no central apex)
    squares_set: set[tuple[int, int, int, int]] = set()
    for a in ids:
        pa = pos_by_id[a]
        for b in adj[a]:
            pb = pos_by_id[b]
            for d_node in adj[b]:
                if d_node == a:
                    continue
                pd = pos_by_id[d_node]
                diag_ad = (pa[0] - pd[0]) ** 2 + (pa[1] - pd[1]) ** 2 + (pa[2] - pd[2]) ** 2
                if diag_ad != 4:
                    continue
                for g_node in adj[d_node]:
                    if g_node == b or g_node not in adj[a]:
                        continue
                    pg = pos_by_id[g_node]
                    diag_bg = (pb[0] - pg[0]) ** 2 + (pb[1] - pg[1]) ** 2 + (pb[2] - pg[2]) ** 2
                    if diag_bg != 4:
                        continue
                    # Check whether a living apex cell in V_living bonds to all 4 corners
                    has_living_apex = any(
                        (ap in adj[a] and ap in adj[b] and ap in adj[d_node] and ap in adj[g_node])
                        for ap in adj[a]
                    )
                    if not has_living_apex:
                        squares_set.add(tuple(sorted((a, b, d_node, g_node))))

    f_sq = len(squares_set)
    chi_2d = v_cnt - e_cnt + f_tri + f_sq

    # Classify lumen nodes into 12/12 closed 0D cavities (beta_2) vs 6/6 open 1D channels (beta_1)
    closed_cavities = 0
    open_ring_lumens = 0
    for lx, ly, lz in lumen_nodes:
        enc_12 = sum(
            1
            for k in range(c.n_12)
            if (lx + int(FCC_N12_VECTORS[k, 2]), ly + int(FCC_N12_VECTORS[k, 1]), lz + int(FCC_N12_VECTORS[k, 0]))
            in pos_set
        )
        if enc_12 == c.n_12:
            closed_cavities += 1
        else:
            open_ring_lumens += 1

    beta_2 = closed_cavities
    beta_1 = 1 if (k4_cnt == 0 and chi_2d == 0 and open_ring_lumens > 0) else 0
    ig_cnt = len(synapses)

    # Carrier self-classification purely from topological invariants on FCC N12:
    if v_cnt <= 1 and e_cnt == 0:
        stage_code = 0  # Single founder zygote / progenitor
    elif k4_cnt > 0 and len(lumen_nodes) == 0 and ig_cnt == 0:
        stage_code = 1  # Solid 3D morula ball B^3 (chi_3d = 1, N_K4 = 8)
    elif k4_cnt == 0 and chi_2d == 2 and beta_2 >= 1:
        stage_code = 2  # Hollow spherical vesicle S^2 (chi_2d = 2, beta_2 = 1)
    elif k4_cnt == 0 and chi_2d == 0 and len(lumen_nodes) >= 1:
        stage_code = 3  # Hollow cylindrical tube S^1 x I (chi_2d = 0, beta_1 = 1)
    else:
        stage_code = 4  # Stratified 3D laminar column (E_Ig > 0 or initial 7-cell wheel)

    return MTissueTopologySnapshot(
        tick=tick,
        v_cells=v_cnt,
        e_contacts=e_cnt,
        f_triangles=f_tri,
        f_squares_surface=f_sq,
        k4_tetrahedra=k4_cnt,
        euler_chi_2d=chi_2d,
        betti_0=beta_0,
        betti_1_open_channels=beta_1,
        betti_2_closed_cavities=beta_2,
        lumen_nodes_count=len(lumen_nodes),
        ig_synapses_count=ig_cnt,
        carrier_stage_code=stage_code,
    )


def m_grow_vesicle_s1(c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a hollow S_1 cuboctahedral vesicle (N_shell(1)=12 cells, 24 edges) in native M units."""
    fork_d3 = (+1, -1, +1)
    founder = MCellNode(0, -1, 0, (0, 0, 0), m_octant_mask((0, 0, 0)), 0, (0,), fork_d3, None, 24)
    cells: list[MCellNode] = [founder]
    divisions: list[MDivisionEvent] = []
    traj: list[MTissueTopologySnapshot] = [m_analyze_tissue_topology(cells, (), (), (), tick=0, c=c)]
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

    adh_morula, syn_morula = _m_build_adhesion_network(cells, c=c)
    traj.append(m_analyze_tissue_topology(cells, adh_morula, syn_morula, (), tick=1, c=c))

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
    final_snap = m_analyze_tissue_topology(living, adhesion, synapses, lumen, tick=2, c=c)
    traj.append(final_snap)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        0,
        1,
        tuple(living),
        tuple(divisions),
        tuple(cavitations),
        tuple(adhesion),
        tuple(synapses),
        tuple(lumen),
        tot_q,
        -tot_q * c.e_hb_e0,
        final_snap,
        tuple(traj),
    )


def m_grow_hex_tube(axial_rings: int = 4, c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a hollow tube by stacking `axial_rings` hexagonal {111} N_6 rings along FCC (1, 1, 0)."""
    axial_step = (1, 1, 0)
    cells: list[MCellNode] = []
    divisions: list[MDivisionEvent] = []
    cavitations: list[MCavitationEvent] = []
    lumen: list[tuple[int, int, int]] = []
    traj: list[MTissueTopologySnapshot] = []
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
        adh_step, syn_step = _m_build_adhesion_network(cells, c=c)
        traj.append(m_analyze_tissue_topology(cells, adh_step, syn_step, lumen, tick=r_idx + 1, c=c))

    adhesion, synapses = _m_build_adhesion_network(cells, c=c)
    final_snap = traj[-1] if traj else m_analyze_tissue_topology(cells, adhesion, synapses, lumen, tick=0, c=c)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        1,
        axial_rings,
        tuple(cells),
        tuple(divisions),
        tuple(cavitations),
        tuple(adhesion),
        tuple(synapses),
        tuple(lumen),
        tot_q,
        -tot_q * c.e_hb_e0,
        final_snap,
        tuple(traj),
    )


def m_grow_laminar_column_6(c: MCarrierConstants = M_CARRIER) -> MTissueMorphology:
    """Grow a |square| = 6-lamina column (6 x 7 = 42 = N_shell(2) cells, 147 = N_ball(3) contacts)."""
    n_layers = c.n_squares  # 6
    donor_ig = MIgLock(+1, 2, 2 + c.n_hb, c.n_hb, 14, (0, 2, 0, 19, 1, 10, 3))
    acceptor_ig = MIgLock(-1, 2, 2 + c.n_hb, c.n_hb, 14, (4, 18, 8, 2, 19, 9, 6))

    cells: list[MCellNode] = []
    divisions: list[MDivisionEvent] = []
    traj: list[MTissueTopologySnapshot] = []
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

        adh_step, syn_step = _m_build_adhesion_network(cells, c=c)
        traj.append(m_analyze_tissue_topology(cells, adh_step, syn_step, (), tick=l_idx + 1, c=c))

    adhesion, synapses = _m_build_adhesion_network(cells, c=c)
    final_snap = traj[-1] if traj else m_analyze_tissue_topology(cells, adhesion, synapses, (), tick=0, c=c)
    tot_q = sum(ev.quanta_ehb for ev in adhesion)
    return MTissueMorphology(
        2,
        n_layers,
        tuple(cells),
        tuple(divisions),
        (),
        tuple(adhesion),
        tuple(synapses),
        (),
        tot_q,
        -tot_q * c.e_hb_e0,
        final_snap,
        tuple(traj),
    )


def m_infer_proteome_symmetry_mode(
    expressed_products: tuple[MExpressedLocus, ...] | list[MExpressedLocus],
) -> tuple[int, int]:
    """Let the M-carrier infer 12-star symmetry breaking directly from expressed protein locks.

    Returns `(axial_d3d_breaking, ig_synaptic_rescue)`:
      - `(0, 0)`: Isotropic `O_h` 12-star (no axial TF/TM18 locks, no heterotypic Ig pair)
                  -> radial `S_1` division + core cavitation -> hollow spherical vesicle (`chi_2D = 2`).
      - `(1, 0)`: Planar `{111}` + axial `(1,1,0)` `D_3d` breaking (axial TF `Trp-pin`/`C2H2` or `TM18` present,
                  no complementary `Ig_donor`/`Ig_acceptor` pair) -> planar ring division + core cavitation
                  -> hollow hexagonal tube (`chi_2D = 0`).
      - `(1, 1)`: Planar `{111}` + axial `(1,1,0)` `D_3d` breaking WITH complementary `Ig_donor(+1)` and
                  `Ig_acceptor(-1)` locks (`N_HB = 63`) -> inter-lamina synaptic rescue of central hubs
                  -> solid 6-layer cortical column (`42 = N_shell(2)` cells, `147 = N_ball(3)` contacts).
    """
    has_axial = any(
        (len(p.trp_pin_locks) > 0 or len(p.c2h2_locks) > 0 or len(p.transmembrane_rods) > 0)
        for p in expressed_products
    )
    has_ig_donor = any(
        ig.polarity_role == +1 and ig.loop_span == 63 for p in expressed_products for ig in p.ig_locks
    )
    has_ig_acceptor = any(
        ig.polarity_role == -1 and ig.loop_span == 63 for p in expressed_products for ig in p.ig_locks
    )
    if has_ig_donor and has_ig_acceptor:
        return 1, 1
    if has_axial:
        return 1, 0
    return 0, 0


def m_grow_tissue_autonomous(
    expressed_products: tuple[MExpressedLocus, ...] | list[MExpressedLocus],
    axial_steps: int = 4,
    c: MCarrierConstants = M_CARRIER,
) -> MTissueMorphology:
    """Single autonomous M-layer morphogenesis automaton driven purely by expressed protein locks."""
    axial_flag, ig_rescue_flag = m_infer_proteome_symmetry_mode(expressed_products)
    if axial_flag == 0:
        return m_grow_vesicle_s1(c=c)
    if ig_rescue_flag == 0:
        return m_grow_hex_tube(axial_rings=axial_steps, c=c)
    return m_grow_laminar_column_6(c=c)


@dataclass(frozen=True)
class MChromosomeScanResult:
    """De novo M-layer chromosome scan and blind-spot diagnosis on a raw Z_4 sequence."""

    genome_length_bp: int
    open_chromatin_bp_count: int
    discovered_loci: tuple[MGeneLocus, ...]
    expressed_products: tuple[MExpressedLocus, ...]
    regulatory_quanta_matrix: np.ndarray
    # Blind spot diagnostics:
    spliced_loci_indices: tuple[int, ...]  # Loci rescued from in-frame stops by lariat splicing
    unannotated_lock_free_loci: tuple[int, ...]  # Folded ORFs lacking any known structural lock
    orphan_promoter_positions: tuple[int, ...]  # Open Major Groove cores (TAAT/TGTTT) with no reader
    intergenic_dark_spans: tuple[tuple[int, int], ...]  # Non-coding spans outside discovered loci
    autonomous_tissue: MTissueMorphology | None = None  # Carrier-inferred 3D tissue morphology


def m_scan_chromosome_de_novo(
    chromosome_z4: tuple[int, ...] | list[int],
    min_orf_codons: int = 8,
    promoter_window_bp: int = 24,
    c: MCarrierConstants = M_CARRIER,
) -> MChromosomeScanResult:
    """Scan a raw 2-bit Z_4 chromosome de novo (zero GFF3 annotations) and diagnose blind spots."""
    seq = tuple(int(b) & 3 for b in chromosome_z4)
    n = len(seq)
    access = m_nucleosome_accessibility_profile(seq, c=c)
    open_bp = sum(access)

    loci: list[MGeneLocus] = []
    products: list[MExpressedLocus] = []
    covered_mask = [0] * n
    spliced_ids: list[int] = []
    lock_free_ids: list[int] = []

    i = 0
    locus_id = 0
    while i <= n - 3 * min_orf_codons:
        # Start triplet ATG in Z_4: (2, 0, 3) -> c6 = (2<<4)|(0<<2)|3 = 35 -> class 12
        if seq[i] == 2 and seq[i + 1] == 0 and seq[i + 2] == 3 and (i == 0 or seq[i - 1] != 2):
            # Scan downstream for raw or spliced termination
            max_window = min(n, i + c.n_ring)
            candidate_raw = seq[i:max_window]
            spliced_seq, introns = m_splice_pre_mrna(candidate_raw, c=c)

            # Find first stop codon (class 20) in the mature spliced frame
            stop_codon_idx = -1
            for k in range(0, len(spliced_seq) - 2, 3):
                c6 = (spliced_seq[k] << 4) | (spliced_seq[k + 1] << 2) | spliced_seq[k + 2]
                if M_CODON_TO_CLASS_21[c6] == 20:
                    stop_codon_idx = k
                    break

            n_codons = (stop_codon_idx // 3) if stop_codon_idx != -1 else (len(spliced_seq) // 3)
            if n_codons >= min_orf_codons:
                intron_bp = sum(sp.excised_length for sp in introns)
                raw_end = min(n, i + 3 * (n_codons + 1) + intron_bp)
                p_start = max(0, i - promoter_window_bp)
                prom_z4 = seq[p_start:i] if i > p_start else seq[i : min(n, i + 12)]
                cds_raw_z4 = seq[i:raw_end]

                loc = MGeneLocus(locus_id, prom_z4, cds_raw_z4)
                prod = m_express_locus(loc, enable_splicing=True, c=c)
                loci.append(loc)
                products.append(prod)

                if len(prod.spliced_introns) > 0:
                    spliced_ids.append(locus_id)

                total_locks = (
                    len(prod.c2h2_locks)
                    + len(prod.trp_pin_locks)
                    + len(prod.ig_locks)
                    + len(prod.catalytic_triads)
                    + len(prod.atp_walker_locks)
                    + len(prod.transmembrane_rods)
                )
                if total_locks == 0:
                    lock_free_ids.append(locus_id)

                for p_idx in range(p_start, raw_end):
                    covered_mask[p_idx] = 1

                locus_id += 1
                i = raw_end
                continue
        i += 1

    reg_mat = m_compute_regulatory_quanta_matrix(products) if products else np.zeros((0, 0), dtype=np.int64)

    # Identify orphan Major Groove cores (0, 2, 2, 0) [TAAT] or (0, 3, 0, 0, 0) [TGTTT]
    has_taat_reader = any(
        pin.target_core_z4 == (0, 2, 2, 0) for p in products for pin in p.trp_pin_locks
    )
    has_tgttt_reader = any(
        pin.target_core_z4 == (0, 3, 0, 0, 0) for p in products for pin in p.trp_pin_locks
    )
    orphan_pos: list[int] = []
    for k in range(n - 5):
        if access[k] == 1:
            if seq[k : k + 4] == (0, 2, 2, 0) and not has_taat_reader:
                orphan_pos.append(k)
            elif seq[k : k + 5] == (0, 3, 0, 0, 0) and not has_tgttt_reader:
                orphan_pos.append(k)

    # Identify intergenic dark spans
    dark_spans: list[tuple[int, int]] = []
    d_start = -1
    for k in range(n):
        if covered_mask[k] == 0 and d_start == -1:
            d_start = k
        elif covered_mask[k] == 1 and d_start != -1:
            if k - d_start >= c.n_12:
                dark_spans.append((d_start, k - 1))
            d_start = -1
    if d_start != -1 and (n - d_start) >= c.n_12:
        dark_spans.append((d_start, n - 1))

    auto_morph = m_grow_tissue_autonomous(products, c=c)
    return MChromosomeScanResult(
        genome_length_bp=n,
        open_chromatin_bp_count=open_bp,
        discovered_loci=tuple(loci),
        expressed_products=tuple(products),
        regulatory_quanta_matrix=reg_mat,
        spliced_loci_indices=tuple(spliced_ids),
        unannotated_lock_free_loci=tuple(lock_free_ids),
        orphan_promoter_positions=tuple(orphan_pos),
        intergenic_dark_spans=tuple(dark_spans),
        autonomous_tissue=auto_morph,
    )


@dataclass(frozen=True)
class MSynthesizedGeneProduct:
    """M-layer result of de novo protein and gene synthesis with forward O(L) self-verification."""

    spec: MProteinSynthesisSpec
    synthesized_class_seq_z21: tuple[int, ...]
    synthesized_coding_dna_z4: tuple[int, ...]
    synthesized_promoter_z4: tuple[int, ...]
    verified_expression: MExpressedLocus
    dna_binding_quanta_verified: int
    all_constraints_satisfied: bool


def m_synthesize_gene_and_protein(
    spec: MProteinSynthesisSpec,
    locus_id: int = 0,
    c: MCarrierConstants = M_CARRIER,
) -> MSynthesizedGeneProduct:
    """Synthesize a Z_21 protein and frame-locked Z_4 coding gene from target properties and self-verify on M."""
    seq_z21 = m_synthesize_protein_for_spec(spec, c=c)
    coding_z4 = m_reverse_translate_classes_to_z4(seq_z21, append_stop_20=True)

    # Construct matching 3-bond-clamped open promoter containing the target DNA binding sites
    prom: list[int] = [3, 1, 3, 1, 3, 1]
    for b0, b1, b2 in spec.target_c2h2_dna_triplets_z4:
        prom.extend((int(b0) & 3, int(b1) & 3, int(b2) & 3))
    if len(spec.target_trp_pin_core_z4) > 0:
        prom.extend(int(b) & 3 for b in spec.target_trp_pin_core_z4)
    prom.extend([3, 1, 3, 1, 3, 1])
    promoter_z4 = tuple(prom)

    # Forward self-verification: express the synthesized Z_4 locus through splicing, translation, and O(L) folding
    locus = MGeneLocus(locus_id, promoter_z4, coding_z4)
    expressed = m_express_locus(locus, enable_splicing=True, c=c)
    q_mat = m_compute_regulatory_quanta_matrix([expressed])
    dna_q = int(q_mat[0, 0])

    # Verify every requested property on the folded product
    f = expressed.folded_chain
    frame_intact = expressed.class_sequence_z21 == seq_z21
    self_avoiding = len(set(f.fcc_lattice_nodes)) == len(f.fcc_lattice_nodes)
    stability_ok = f.total_quanta_ehb >= spec.min_stability_quanta_ehb
    c2h2_targets = tuple(lk.target_bases_z4 for lk in expressed.c2h2_locks)
    c2h2_ok = c2h2_targets == spec.target_c2h2_dna_triplets_z4
    trp_ok = (
        len(expressed.trp_pin_locks) == 0
        if len(spec.target_trp_pin_core_z4) == 0
        else any(lk.target_core_z4 == spec.target_trp_pin_core_z4 for lk in expressed.trp_pin_locks)
    )
    triad_ok = len(expressed.catalytic_triads) == spec.require_catalytic_triad
    walker_ok = len(expressed.atp_walker_locks) == spec.require_atp_walker_motor
    tm_ok = len(expressed.transmembrane_rods) == spec.require_transmembrane_rod
    ig_ok = (
        len(expressed.ig_locks) == 0
        if spec.ig_synaptic_polarity == 0
        else (
            len(expressed.ig_locks) == 1
            and expressed.ig_locks[0].polarity_role == spec.ig_synaptic_polarity
            and expressed.ig_locks[0].loop_span == c.n_hb
        )
    )

    all_ok = (
        frame_intact
        and self_avoiding
        and stability_ok
        and c2h2_ok
        and trp_ok
        and triad_ok
        and walker_ok
        and tm_ok
        and ig_ok
    )

    return MSynthesizedGeneProduct(
        spec=spec,
        synthesized_class_seq_z21=seq_z21,
        synthesized_coding_dna_z4=coding_z4,
        synthesized_promoter_z4=promoter_z4,
        verified_expression=expressed,
        dna_binding_quanta_verified=dna_q,
        all_constraints_satisfied=all_ok,
    )
