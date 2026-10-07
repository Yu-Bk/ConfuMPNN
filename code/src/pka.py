"""Amino acid pKa values and charge-type constants.

The values are approximate intrinsic pKa values and do not account for local
structural environments. For environment-specific pKa estimates, use a tool
such as PypKa or PROPKA. Histidine is treated as a basic residue (pKa = 6.0);
cysteine and tyrosine can also become negatively charged at higher pH values.
"""

# Side-chain pKa values for residues that may carry charge or change charge state near physiological pH.
PKA_SIDECHAIN = {
    "D": 3.9,   # Asp side-chain -COOH
    "E": 4.3,   # Glu side-chain -COOH
    "H": 6.0,   # His imidazole group
    "C": 8.3,   # Cys side-chain -SH
    "Y": 10.1,  # Tyr phenol group
    "K": 10.5,  # Lys side-chain -NH3+
    "R": 12.5,  # Arg guanidinium group
}

# Terminal pKa values.
PKA_N_TERM = 9.7   # α-NH3+ (N-terminus)
PKA_C_TERM = 2.3   # α-COOH (C-terminus)

# One-letter codes for the 20 standard amino acids, in LigandMPNN restype order.
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AAS)}

# Side-chain charge types used by charge and filter logic.
ACIDIC = ("D", "E", "C", "Y")   # Negatively charged upon deprotonation.
BASIC = ("K", "R", "H")          # Positively charged upon protonation.

# Strongly charged residues used by the spatial filter.
STRONG_POSITIVE = ("K", "R")     # Usually +1 at physiological pH.
STRONG_NEGATIVE = ("D", "E")     # Usually -1 at physiological pH.
