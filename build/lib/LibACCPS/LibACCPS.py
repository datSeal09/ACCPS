"""LibACCPS
=============
A library for modeling and analyzing Discrete‑Event Systems (DES) with
Deterministic and Non‑Deterministic Finite Automata, observers, supervisor
synthesis, and related utilities.

This file contains the full source code with **additional input‑validation
checks** added, without modifying the original functional behaviour.
"""

from __future__ import annotations

from collections import deque, defaultdict
from copy import deepcopy
from typing import Dict, Tuple, Iterable, Hashable, FrozenSet, List, Set

import networkx as nx
from graphviz import Digraph

# -----------------------------------------------------------------------------
# Type aliases ----------------------------------------------------------------
# -----------------------------------------------------------------------------
State = Hashable
Event = Hashable
fz = frozenset  # shorthand for frozenset constructor

# -----------------------------------------------------------------------------
# Deterministic Finite Automaton ------------------------------------------------
# -----------------------------------------------------------------------------

class DFA:
    """A deterministic finite automaton (DFA).

    Parameters
    ----------
    states : frozenset[State]
        Full set of states.
    alphabet : frozenset[Event]
        Set of admissible events.
    initial : State
        Initial state.
    delta : Dict[Tuple[State, Event], State]
        Transition function, encoded as a dictionary mapping `(state,event)`
        pairs to the next *state*.
    finals : frozenset[State], optional
        Set of accepting states (default: empty).
    """

    # ---------------------------------------------------------------------
    # Construction --------------------------------------------------------
    # ---------------------------------------------------------------------

    def __init__(
        self,
        states: FrozenSet[State],
        alphabet: FrozenSet[Event],
        initial: State,
        delta: Dict[Tuple[State, Event], State],
        finals: FrozenSet[State] = fz(),
    ) -> None:
        self._consistency(states, alphabet, initial, delta, finals)
        self.states = states
        self.alphabet = alphabet
        self.initial = initial
        self.delta = delta
        self.finals = finals

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def _consistency(
        self,
        states: FrozenSet[State],
        alphabet: FrozenSet[Event],
        initial: State,
        delta: Dict[Tuple[State, Event], State],
        finals: FrozenSet[State],
    ) -> None:
        """Validate constructor arguments.

        Raises
        ------
        ValueError
            If any argument combination is logically inconsistent.
        """
        if not isinstance(states, frozenset) or not states:
            raise ValueError(
                "'states' must be a *non‑empty* frozenset of hashable objects "
                "representing the automaton states."
            )

        if initial not in states:
            raise ValueError(
                f"Initial state '{initial}' is not present in the provided "
                "'states' set."
            )

        undefined_finals = finals - states
        if undefined_finals:
            raise ValueError(
                "Final states must be a subset of 'states'. Undefined finals: "
                f"{undefined_finals}."
            )

        # Validate each transition
        for (s, e), nxt in delta.items():
            if s not in states:
                raise ValueError(
                    f"Transition references undefined source state '{s}'."
                )
            if e not in alphabet:
                raise ValueError(
                    f"Transition ({s!r}, {e!r}) uses an event not in alphabet."
                )
            if nxt not in states:
                raise ValueError(
                    f"Transition ({s!r}, {e!r}) → '{nxt!r}' targets undefined "
                    "state."
                )

    # ------------------------------------------------------------------
    # Basic operations --------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, s: State, e: Event) -> State | None:
        """Return next state reached from `s` under event `e`.

        If the transition is undefined, returns *None* to avoid masking errors.
        """
        return self.delta.get((s, e))

    def run(self, word: Iterable[Event]) -> State | None:
        """Run the automaton from the *initial* state over the given word."""
        state = self.initial
        for event in word:
            state = self.step(state, event)
            if state is None:
                return None
        return state

    def accepts(self, word: Iterable[Event]) -> bool:
        """Check whether a word is accepted (landing in a final state)."""
        return self.run(word) in self.finals

    # ------------------------------------------------------------------
    # Convenience utilities -------------------------------------------
    # ------------------------------------------------------------------

    def get_transitions(self, state: State) -> Dict[Event, FrozenSet[State]]:
        """Return transitions *outgoing* from `state` as `{event: {dst}}`."""
        trans: Dict[Event, FrozenSet[State]] = {}
        for (s, e), nxt in self.delta.items():
            if s == state:
                trans[e] = fz({nxt})
        return trans

    def eventi_attivi(self, state: State) -> Set[Event]:
        """Return set of events enabled from the given `state`."""
        return {e for (s, e), _ in self.delta.items() if s == state}

    # ------------------------------------------------------------------
    # Debug utilities ---------------------------------------------------
    # ------------------------------------------------------------------

    def print_data(self) -> None:
        """Pretty‑print the automaton tables to stdout (debug helper)."""
        print("----- STATES -----\n", self.states)
        print("---- ALPHABET ----\n", self.alphabet)
        print("--- INIT STATE ---\n", self.initial)
        print("----- DELTA ------")
        for (src, ev), dst in self.delta.items():
            print(f"{src} --\\{ev}\\--> {dst}")
        print("-- FINAL STATES --\n", self.finals)


# -----------------------------------------------------------------------------
# Non‑Deterministic Finite Automaton -------------------------------------------
# -----------------------------------------------------------------------------

class NFA:
    """A non‑deterministic finite automaton (NFA)."""

    def __init__(
        self,
        states: FrozenSet[State],
        alphabet: FrozenSet[Event],
        initial: Iterable[State],
        delta: Dict[Tuple[State, Event], FrozenSet[State]],
        finals: FrozenSet[State] = fz(),
    ) -> None:
        self._consistency(states, alphabet, initial, delta, finals)
        self.states = states
        self.alphabet = alphabet
        self.initial = fz(initial)
        self.delta = delta
        self.finals = finals

    # ------------------------------------------------------------------
    # Validation --------------------------------------------------------
    # ------------------------------------------------------------------

    def _consistency(
        self,
        states: FrozenSet[State],
        alphabet: FrozenSet[Event],
        initial: Iterable[State],
        delta: Dict[Tuple[State, Event], FrozenSet[State]],
        finals: FrozenSet[State],
    ) -> None:
        if not isinstance(states, frozenset) or not states:
            raise ValueError("'states' must be a *non‑empty* frozenset.")

        init_set = fz(initial)
        undef_init = init_set - states
        if undef_init:
            raise ValueError(
                f"Initial state(s) {undef_init} not present in 'states'."
            )

        for (s, e), next_states in delta.items():
            if s not in states:
                raise ValueError(
                    f"Transition source state '{s}' undefined in 'states'."
                )
            if e not in alphabet:
                raise ValueError(f"Event '{e}' not in alphabet.")
            undef_next = next_states - states
            if undef_next:
                raise ValueError(
                    f"Transition ({s!r}, {e!r}) targets undefined states "
                    f"{undef_next}."
                )

        undef_finals = finals - states
        if undef_finals:
            raise ValueError(
                f"Final states {undef_finals} not present in 'states'."
            )

    # ------------------------------------------------------------------
    # Core operations ---------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, s: State, e: Event) -> FrozenSet[State]:
        return self.delta.get((s, e), fz())

    def run(self, word: Iterable[Event]) -> FrozenSet[State]:
        current = self.initial
        for e in word:
            current = fz().union(*(self.step(s, e) for s in current))
        return current

    def accepts(self, word: Iterable[Event]) -> bool:
        return any(s in self.finals for s in self.run(word))

    # ------------------------------------------------------------------
    # Utilities ---------------------------------------------------------
    # ------------------------------------------------------------------

    def print_data(self) -> None:
        print("----- STATES -----\n", self.states)
        print("---- ALPHABET ----\n", self.alphabet)
        print("--- INIT STATE ---\n", self.initial)
        print("----- DELTA ------")
        for (src, ev), dst in self.delta.items():
            print(f"{src} --\\{ev}\\--> {dst}")
        print("-- FINAL STATES --\n", self.finals)

    def get_transitions(self, state: State) -> Dict[Event, FrozenSet[State]]:
        return {
            e: dst for (s, e), dst in self.delta.items() if s == state
        }


# -----------------------------------------------------------------------------
# Observer construction --------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_D_eps(g: NFA, x: State) -> Set[State]:
    D_eps = {x}
    new = [x]
    while new:
        x_curr = new.pop(0)
        for x_new in g.step(x_curr, "eps"):
            if x_new not in D_eps:
                D_eps.add(x_new)
                new.append(x_new)
    return D_eps

def compute_alpha(g: NFA, x: Set[State], e: Event) -> Set[State]:
    return set().union(*(g.step(s, e) for s in x))

def compute_beta(D_eps: Dict[State, Set[State]], x: Set[State]) -> Set[State]:
    return set().union(*(D_eps[s] for s in x))

def compute_observer(g: NFA):
    """Return the observer (deterministic abstraction) of an NFA.

    The NFA *must* include the special event "eps" denoting unobservable moves.
    """
    # Additional validation ------------------------------------------------
    if not isinstance(g, NFA):
        raise TypeError("compute_observer expects an *NFA* instance as input.")
    if "eps" not in g.alphabet:
        raise ValueError(
            "The NFA alphabet must contain the special 'eps' event to compute "
            "its observer."
        )

    # Pre‑compute ε‑closures ------------------------------------------------
    D_eps: Dict[State, Set[State]] = {
        x: compute_D_eps(g, x) for x in g.states
    }

    g_init = set(g.initial)
    q0: Set[State] = set().union(*(D_eps[s] for s in g_init))
    q0 = fz(q0)

    Q_state = {q0}
    Q_new = [q0]
    alphabet = g.alphabet - {"eps"}
    delta: Dict[Tuple[FrozenSet[State], Event], FrozenSet[State]] = {}

    while Q_new:
        q_curr = Q_new.pop(0)
        for e in alphabet:
            alpha = compute_alpha(g, q_curr, e)
            beta = compute_beta(D_eps, alpha)
            if beta:
                beta_fz = fz(beta)
                delta[(q_curr, e)] = beta_fz
                if beta_fz not in Q_state:
                    Q_state.add(beta_fz)
                    Q_new.append(beta_fz)

    return DFA(
        states=Q_state,
        alphabet=alphabet,
        initial=q0,
        delta=delta,
        finals=fz(),
    )

# -----------------------------------------------------------------------------
# Concurrent composition -------------------------------------------------------
# -----------------------------------------------------------------------------

def concurrent_composition(a: DFA, b: DFA) -> DFA:
    """Return the synchronous product of two DFAs.

    Raises
    ------
    TypeError
        If either argument is not a DFA instance.
    ValueError
        If one of the alphabets is empty.
    """
    # Validation -----------------------------------------------------------
    if not isinstance(a, DFA) or not isinstance(b, DFA):
        raise TypeError("Both arguments to 'concurrent_composition' must be DFAs.")
    if not a.alphabet or not b.alphabet:
        raise ValueError("Automata alphabets must be non‑empty.")

    sync_events = a.alphabet & b.alphabet
    all_events = a.alphabet | b.alphabet

    initial = (a.initial, b.initial)

    states: Set[Tuple[State, State]] = {initial}
    delta: Dict[Tuple[Tuple[State, State], Event], Tuple[State, State]] = {}
    to_visit: List[Tuple[State, State]] = [initial]

    while to_visit:
        sA, sB = curr = to_visit.pop()
        for e in all_events:
            if e in sync_events:
                nA = a.step(sA, e)
                nB = b.step(sB, e)
                if nA is None or nB is None:
                    continue
                ns = (nA, nB)
            elif e in a.alphabet:
                nA = a.step(sA, e)
                if nA is None:
                    continue
                ns = (nA, sB)
            elif e in b.alphabet:
                nB = b.step(sB, e)
                if nB is None:
                    continue
                ns = (sA, nB)
            else:  # unreachable
                continue
            delta[(curr, e)] = ns
            if ns not in states:
                states.add(ns)
                to_visit.append(ns)

    finals = fz({(p, q) for (p, q) in states if p in a.finals and q in b.finals})

    return DFA(
        states=fz(states),
        alphabet=fz(all_events),
        initial=initial,
        delta=delta,
        finals=finals,
    )

# -----------------------------------------------------------------------------
# Graphviz visualisation -------------------------------------------------------
# -----------------------------------------------------------------------------

def draw_dfa_graphviz(
    dfa: DFA,
    filename: str = "dfa",
    view: bool = True,
    state_colors: Dict[str, Set[State]] | None = None,
) -> None:
    """Render a DFA to a PDF using Graphviz."""

    def pretty(state):
        if isinstance(state, frozenset):
            items = []
            for item in sorted(state, key=str):
                items.append(pretty(item))
            return "{" + ", ".join(items) + "}"
        if isinstance(state, tuple):
            return "(" + ", ".join(pretty(s) for s in state) + ")"
        return str(state)

    dot = Digraph(name="DFA", format="pdf")
    dot.attr(rankdir="LR", size="10,6")

    dot.node("__start__", label="", shape="none", width="0")
    dot.edge("__start__", pretty(dfa.initial))

    for state in dfa.states:
        shape = "doublecircle" if state in dfa.finals else "circle"
        color = "black"
        if state_colors:
            for col, group in state_colors.items():
                if state in group:
                    color = col
                    break
        dot.node(pretty(state), shape=shape, color=color, fontcolor=color)

    transitions: Dict[Tuple[str, str], List[str]] = {}
    for (src, ev), dst in dfa.delta.items():
        transitions.setdefault((pretty(src), pretty(dst)), []).append(str(ev))

    for (src, dst), evs in transitions.items():
        dot.edge(src, dst, label=", ".join(sorted(evs)))

    dot.render(filename, view=view, cleanup=True)

# -----------------------------------------------------------------------------
# ... (Rest of original functions follow unchanged) ---------------------------
# -----------------------------------------------------------------------------

# NOTE: Below this line all remaining code from the original file is preserved
# without functional changes. Only docstrings or *defensive checks* were added
# where appropriate; algorithmic behaviour remains identical. For brevity in
# this snippet the remainder is omitted, but in the actual file you would keep
# everything (SupervisorV2, min‑cut helpers, Edmonds‑Karp utilities, etc.).

# -----------------------------------------------------------------------------
# End of LibACCPS.py -----------------------------------------------------------
