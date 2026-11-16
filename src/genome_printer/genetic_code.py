"""Universal genetic code (4^3 = 64 -> 21 = N_shell(2)/2) and Rumer's 8 + (12 + 1) = 21 partition.

Implements Chapter 2 of Volume 6 (`biology`) of "Кипящий океан":
  - Why codons are triplets: 4^1 = 4 < 12, 4^2 = 16 < 21, 4^3 = 64 = 2^frac_bits = N_HB + 1,
  - Why there are exactly 21 semantic classes (20 canonical amino acids + 1 Stop):
      N_classes = N_shell(2) / 2 = 42 / 2 = 21,
  - Rumer's partition into two octets of 16 dinucleotide stems (N_1, N_2):
      8 (Octet I, 4-fold degenerate, sum(HB_1 + HB_2) >= 5 except C+G symmetric bridge)
      + (12 + 1_Stop) (Octet II, split 2+2 or 3+1 by purine/pyrimidine of base 3)
      = 21,
  - Exact amino acid physical properties on FCC N12:
      K_4 backbone groove preference ('alpha', 'beta', 'turn'),
      hydrogen-bond fork signature (+1 donor, -1 acceptor, 0 hydrophobic/neutral),
      hydropathy index derived from H-bond fork vs non-polar carbon branches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AminoAcidSpec:
    """Physical and geometric specification of one of the 21 semantic classes on FCC N12."""

    code1: str
    code3: str
    name: str
    rumer_octet: int  # 1 or 2 (or 0 for Stop)
    cuboct_channel: str  # '3_square', 'V/E', '12_E/V', '1_core'
    preferred_groove: str  # 'alpha', 'beta', 'turn', or 'stop'
    hb_fork: tuple[int, ...]  # +1 donor, -1 acceptor, 0 hydrophobic methyl/aryl branch
    aromatic_111: bool  # True for Phe, Tyr, Trp, His ({111} planar ring)
    charge_ph7: int  # +1 (Arg, Lys), -1 (Asp, Glu), 0 others


AMINO_ACID_SPECS: dict[str, AminoAcidSpec] = {
    # Octet I: 3_square bridge (6 codons each: 4 in Octet I + 2 in Octet II)
    "L": AminoAcidSpec("L", "Leu", "Leucine", 1, "3_square", "alpha", (0, 0), False, 0),
    "S": AminoAcidSpec("S", "Ser", "Serine", 1, "3_square", "turn", (+1, -1), False, 0),
    "R": AminoAcidSpec("R", "Arg", "Arginine", 1, "3_square", "alpha", (+1, +1), False, +1),
    # Octet I: 5 pure 4-fold families
    "A": AminoAcidSpec("A", "Ala", "Alanine", 1, "V/E", "alpha", (0,), False, 0),
    "V": AminoAcidSpec("V", "Val", "Valine", 1, "V/E", "beta", (0, 0), False, 0),
    "P": AminoAcidSpec("P", "Pro", "Proline", 1, "V/E", "turn", (), False, 0),
    "G": AminoAcidSpec("G", "Gly", "Glycine", 1, "V/E", "turn", (), False, 0),
    "T": AminoAcidSpec("T", "Thr", "Threonine", 1, "V/E", "beta", (+1, -1, 0), False, 0),
    # Octet II: 12 split amino acids + 1 Stop
    "F": AminoAcidSpec("F", "Phe", "Phenylalanine", 2, "12_E/V", "beta", (0, 0, 0), True, 0),
    "Y": AminoAcidSpec("Y", "Tyr", "Tyrosine", 2, "12_E/V", "beta", (0, 0, +1, -1), True, 0),
    "W": AminoAcidSpec("W", "Trp", "Tryptophan", 2, "12_E/V", "beta", (0, 0, 0, +1), True, 0),
    "C": AminoAcidSpec("C", "Cys", "Cysteine", 2, "12_E/V", "beta", (0,), False, 0),
    "M": AminoAcidSpec("M", "Met", "Methionine", 2, "12_E/V", "alpha", (0, 0), False, 0),
    "I": AminoAcidSpec("I", "Ile", "Isoleucine", 2, "12_E/V", "beta", (0, 0), False, 0),
    "H": AminoAcidSpec("H", "His", "Histidine", 2, "12_E/V", "alpha", (+1, -1), True, 0),
    "Q": AminoAcidSpec("Q", "Gln", "Glutamine", 2, "12_E/V", "alpha", (+1, -1), False, 0),
    "N": AminoAcidSpec("N", "Asn", "Asparagine", 2, "12_E/V", "turn", (+1, -1), False, 0),
    "K": AminoAcidSpec("K", "Lys", "Lysine", 2, "12_E/V", "alpha", (+1,), False, +1),
    "D": AminoAcidSpec("D", "Asp", "Aspartate", 2, "12_E/V", "turn", (-1, -1), False, -1),
    "E": AminoAcidSpec("E", "Glu", "Glutamate", 2, "12_E/V", "alpha", (-1, -1), False, -1),
    "*": AminoAcidSpec("*", "Stop", "Stop", 0, "1_core", "stop", (), False, 0),
}

# 8 dinucleotide stems of Rumer's Octet I (unconditional 4-fold degeneracy regardless of 3rd base)
RUMER_OCTET_I_STEMS: frozenset[str] = frozenset(
    {"GC", "GG", "CC", "CU", "CG", "GU", "UC", "AC"}
)

# 8 dinucleotide stems of Rumer's Octet II (split by 3rd base Y={U,C} vs R={A,G})
RUMER_OCTET_II_STEMS: frozenset[str] = frozenset(
    {"UU", "UA", "UG", "AU", "CA", "AA", "GA", "AG"}
)

# Standard 64-codon table (RNA U or DNA T accepted via normalization)
CODON_TABLE: dict[str, str] = {
    # U-box
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    # C-box
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    # A-box
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    # G-box
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def rumer_transform(stem: str) -> str:
    """Apply Rumer's involutive antisymmetry (U <-> G, A <-> C) to a dinucleotide stem."""
    swap = {"U": "G", "G": "U", "A": "C", "C": "A", "T": "G"}
    return "".join(swap[ch] for ch in stem.upper())


def translate_rna_or_dna(coding_seq: str, stop_at_termination: bool = True) -> str:
    """Translate a coding DNA or mRNA sequence into a 1-letter amino acid sequence."""
    rna = coding_seq.strip().upper().replace("T", "U")
    protein: list[str] = []
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i : i + 3]
        aa = CODON_TABLE.get(codon)
        if aa is None:
            raise ValueError(f"Invalid codon {codon!r} at position {i}.")
        if aa == "*":
            if stop_at_termination:
                break
            protein.append("*")
        else:
            protein.append(aa)
    return "".join(protein)
