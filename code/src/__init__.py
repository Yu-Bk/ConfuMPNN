"""ConfuMPNN core module library.

Modules (grouped by purpose):
- pka.py                     : amino-acid pKa table and charged-type constants
- differentiable_charge.py   : differentiable pH-aware net-charge calculation
- isoelectric_point.py       : binary search for pI (isoelectric point)
- structure_aware_filter.py  : structure-aware filter (logit-bias injection)
- condition_embedding.py     : pH-aware condition encoder (Soft Prompt)
- losses.py                  : composite loss functions
- guided_sampler.py          : guided sampler (wrapper around the LigandMPNN decoder)

Activate the environment before use:
    conda activate confumpnn
"""
