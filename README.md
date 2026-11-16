# Genome Printer (v3.0 — Strict Two-Layer $\mathcal{M} \to \mathcal{T}$ Cuboctahedral FCC $N_{12}$ Architecture)

Parameter-free cellular-automaton genome interpreter, $O(1)$-per-residue polypeptide folder, and 3D tissue morphogenesis engine built strictly on the two-layer $(\mathcal{M}, \mathcal{T}, \mathcal{B})$ architecture of *«Кипящий океан»* (`mt-lattice-fluid`, Volumes 1–7).

## Strict Separation of Layers ($\mathcal{M}$ vs $\mathcal{T}$)

1. **Primary Carrier Layer $\mathcal{M}$ (`m_carrier.py`, `m_chemistry.py`, `m_nucleic.py`, `m_polypeptide.py`, `m_tissue.py`):**
   - **Zero human string literals, zero IUPAC letters, zero SI dimensional constants** (enforced via AST inspection by the `M_layer_purity` gate in `verify_genome_printer.py`).
   - All lengths, times, and energies are expressed directly in the closed native mechanical ladder of the cell $(dh = 1,\ dt = 1,\ s_0 = \tfrac{1}{2})$:
     $$c_0 = \frac{dh}{dt} = 1,\quad c = \kappa c_0 = \frac{1}{\sqrt{2}},\quad E_0 = \frac{s_0}{dt},\quad p_0 = \frac{s_0}{dh},\quad F_0 = \frac{s_0}{dh\cdot dt},\quad m_{\mathrm{arg}} = \frac{2s_0 dt}{dh^2}.$$
   - From $\hat M = 1 + |N_{12}| N_{\mathrm{hier}} = 97$, $d_{\mathrm{soft}} = 7$, and $\kappa = 1/\sqrt{2}$, the carrier derives $\alpha_{\mathrm{fs}} \approx 1/137.03599919$, the electroweak scale $v = r^{-1/2}\alpha_{\mathrm{fs}}^8 E_0$, the electron mass $m_e = \alpha_{\mathrm{fs}}^2 m_H / N_\varphi$, the Compton step count $N_c$, the integer Bohr step count $N_{a0} = \mathrm{round}(N_c/\alpha_{\mathrm{fs}}) \in \mathbb{Z}$ ($a_0 = N_{a0}\cdot dh$, atomic packing $N_{\mathrm{pack}} = N_{a0}^3 \in \mathbb{Z}$), the Hydrogen binding energy $BE_H = \frac{\alpha_{\mathrm{fs}}}{2\kappa N_{a0}}E_0$, and the universal hydrogen-bond quantum:
     $$E_{\mathrm{HB}} = \frac{BE_H}{N_{\mathrm{HB}}} = \frac{BE_H}{63} = \frac{\alpha_{\mathrm{fs}}}{2\kappa N_{\mathrm{HB}} N_{a0}}\,E_0,\qquad \Delta G_{\mathrm{ATP}} = (d-1)\,E_{\mathrm{HB}} = 2\,E_{\mathrm{HB}}.$$
   - **Nucleic registers ($\mathbb{Z}_4$, $\mathbb{Z}_{16}$, $\mathbb{Z}_{64}$):** Each nucleobase tile in the $\{111\}$ slice is a 2-bit integer $b \in \mathbb{Z}_4 = \{0, 1, 2, 3\}$. Watson–Crick complementation is bitwise XOR `b ^ 2`; Rumer's antisymmetry transformation on a 4-bit dinucleotide stem $s \in \mathbb{Z}_{16}$ is bitwise inversion `s ^ 0xF`! Each 6-bit triplet $c \in \mathbb{Z}_{64}$ ($\mathrm{frac\_bits} = 6$) maps deterministically to one of $N_{\mathrm{shell}}(2)/2 = 21$ second-shell classes $a \in \mathbb{Z}_{21}$.
   - **$O(1)$ Folding & 3D Morphogenesis ($\mathbb{Z}_3$, $\mathbb{Z}_8$):** Each residue selects its $K_4$ groove $g \in \mathbb{Z}_3 = \{0, 1, 2\}$ in $O(1)$ local operations ($\le 36 L \in \mathrm{P}$). On Floor $k = 12\ldots 13$, cells occupy rhombic dodecahedra on the 3D `FCC N12` lattice with 3-bit octant masks $\mathrm{mask}_3 \in \mathbb{Z}_8$ and integer lamina indices $k \in \mathbb{Z}_{\ge 0}$:
     - $S_1$ cuboctahedral vesicle: $N_{\mathrm{shell}}(1) = 12$ cells, $|E_{\mathrm{cuboct}}| = 24$ contacts, $1$ central cavity;
     - $N_6$ hexagonal tube: stacked $\{111\}$ rings with axial lumen cavitation;
     - $|\square| = 6$-lamina column: $6 \times 7 = 42 = N_{\mathrm{shell}}(2)$ cells, $41 = \Delta\varphi_{\mathrm{disc}}$ divisions, $147 = N_{\mathrm{ball}}(3)$ total contacts, and $75$ inter-lamina `Ig` synaptic locks.

2. **Laboratory Readout Projector $\mathcal{B}\colon \mathcal{M} \leftrightarrow \mathcal{T}$ (`t_readout.py`):**
   - The sole module containing human IUPAC symbols (`T/U, C, A, G`, 20 amino acid letters), SI calibration anchors (`Å`, `eV`, `kJ/mol`, `K`), and X-Ray report tables.

## Verification

```bash
python verify_genome_printer.py
```
