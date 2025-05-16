"""
LibACCPS package initializer.

Rende disponibili a livello di pacchetto tutte le classi e funzioni
definite in LibACCPS.py, così puoi fare ad esempio:

    from LibACCPS import DFA, compute_observer, EdmondsKarp
"""

from .LibACCPS import (
    # Classi
    DFA,
    NFA,

    # Funzioni di base sugli automi
    compute_D_eps,
    compute_alpha,
    compute_beta,
    compute_observer,
    mask,
    concurrent_composition,      # (versione più recente)

    # Observer / operatori d’attacco
    create_attack_observer,
    create_operator_observer,

    # Utility per generatori e costruzioni
    gn_creator,
    compute_forbidden,
    trim_joint_observer_v2,

    # Analisi di insiemi e percorsi
    # find_weakly_forbidden,
    # invert_delta,
    Reach,



    # Visualizzazione
    draw_dfa_graphviz,
    draw_nfa_graphviz,
)

__all__ = [
    # Classi
    "DFA",
    "NFA",
    "SupervisorV2",
    "Edge",

    # Funzioni di base
    "compute_D_eps",
    "compute_alpha",
    "compute_beta",
    "compute_observer",
    "mask",
    "concurrent_composition",

    # Observer / attack
    "create_attack_observer",
    "create_operator_observer",

    # Utility automi
    "gn_creator",
    "compute_forbidden",
    "trim_joint_observer_v2",

    # Analisi insiemi
    "find_weakly_forbidden",
    "invert_delta",
    "Reach",

    # Algoritmi flusso
    "NoneDict",
    "EdmondsKarp",
    "min_cut",
    "get_min_cut_edges",
    "extract_disabled_events",
    "add_edge",
    "add_transition",
    "ConvertDFA2Gliph",

    # Visualizzazione
    "draw_dfa_graphviz",
    "draw_nfa_graphviz"
]

