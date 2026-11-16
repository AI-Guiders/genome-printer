# Genome Printer (v2.0 — Clean Cuboctahedral FCC N₁₂ Edition)

Parameter-free cellular-automaton genome interpreter, B-DNA Major Groove reader, and $O(1)$-per-residue co-translational protein folder built strictly on the discrete geometry of the cuboctahedral **FCC $N_{12}$** carrier (`mt-lattice-fluid`, Volumes 5–7 of *«Кипящий океан»*).

## Core Principles (Zero Fitted Parameters)

1. **Scale-Cancellation Bridge (`constants.py`, `chemistry.py`):**
   - Spatial dimension $d = 3$, cuboctahedral star $|N_{12}| = 12$, sphere ratio $\kappa = R_{\mathrm{in}}/R_{\mathrm{out}} = 1/\sqrt{2}$, hexagonal $\{111\}$ slice ratio $\kappa_{\mathrm{hex}} = \sqrt{3}/2$.
   - Bekenstein cell bit budget $\lfloor B_V \rfloor = \lfloor 2\pi/\ln 2 \rfloor = 9\text{ bits}$ ($\mathbb{Z}_{512}$ phase ring, $N_{\mathrm{hier}} = 8$, $\mathrm{frac\_bits} = 6$, $N_{\mathrm{HB}} = 2^6 - 1 = 63$, $N_\varphi = 1 + |N_{12}| = 13$).
   - Universal hydrogen-bond quantum:
     $$E_{\mathrm{HB}} = \frac{D_0(\text{H}_2)}{\kappa^2 N_{\mathrm{shell}}(2)} = \frac{BE_H}{63} \approx 0.2160\text{ eV} \approx 20.84\text{ kJ/mol}.$$
   - Exact commensurability of the peptide $\text{C–N}$ bond with the hydrogen lattice:
     $$R(\text{C–N}) = r_{\mathrm{cov}}(6) + r_{\mathrm{cov}}(7) = \frac{27 + 25}{13}\kappa a_0 = 4\kappa a_0 = 2 R_e(\text{H}_2) \approx 1.497\text{ \AA}.$$

2. **B-DNA Duplex, Major Groove 4-Slot Barcode, and 147 bp Nucleosome (`dna.py`):**
   - $d = 3$ frontal Watson–Crick H-bond tracks ($E(\text{G–C}) = 3 E_{\mathrm{HB}}$, $E(\text{A–T}) = 2 E_{\mathrm{HB}}$).
   - $|K_4| = 4$ rear Major Groove slots (`A-T -> [-,+,-,M]`, `T-A -> [M,-,+,-]`, `G-C -> [-,-,+,0]`, `C-G -> [0,+,-,-]`) enabling $O(1)$ sequence readout without duplex unwinding.
   - Axial stacking step $h_{\mathrm{stack}} = \lfloor B_V \rfloor \kappa a_0 = 9\kappa a_0 \approx 3.368\text{ \AA}$, helical period $n_{\mathrm{turn}} = |N_{12}| - 2 = 10\text{ bp}$, and nucleosome capacity $N_{\mathrm{nuc}} = N_{\mathrm{ball}}(3) = 1 + 12 + 42 + 92 = 147\text{ bp}$.

3. **Universal Genetic Code $64 \to 21 = N_{\mathrm{shell}}(2)/2$ and Rumer's Partition (`genetic_code.py`):**
   - $4^3 = 64 = 2^{\mathrm{frac\_bits}} = N_{\mathrm{HB}} + 1$ triplets map to $21 = N_{\mathrm{shell}}(2)/2$ semantic classes via Rumer's $8_{\mathrm{Octet\ I}} + (12_{\mathrm{Octet\ II}} + 1_{\mathrm{Stop}}) = 21$ partition.

4. **$O(1)$-per-Residue Co-Translational Folding and Structural Locks (`folding.py`, `locks.py`):**
   - Resolves Levinthal's paradox in class $\mathrm{P}$ ($O(L)$ time, $\le 36 L$ operations): each residue selects one of $d = 3$ discrete $K_4$ grooves (`alpha` $3.6_{13}$ helix with $n_\alpha = 18/5 = 3.6$, `beta` $\{111\}$ sheet, or `turn`).
   - Deterministic recognition of `C2H2` zinc fingers ($|N_{12}| = 12$ loop), Homeodomain/Forkhead `Trp`-pins ($d_{\mathrm{in}}(\square) = 4$), and `Ig` synaptic adhesion locks ($L_{\mathrm{loop}} = N_{\mathrm{HB}} = 63$).

## Verification

```bash
python verify_genome_printer.py
```
