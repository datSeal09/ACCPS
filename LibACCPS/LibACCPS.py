"""LibACCPS
=============
A library for modeling and analyzing Discrete‑Event Systems (DES) with
Deterministic and Non‑Deterministic Finite Automata, observers, supervisor
synthesis, and related utilities.
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
        """Compact and pretty‑print the automaton with line wrapping."""

        WRAP_LIMIT = 100  # Numero massimo di caratteri prima di andare a capo

        def pretty(x):
            if isinstance(x, frozenset):
                items = [pretty(e) for e in sorted(x, key=str)]
                return "{" + ", ".join(items) + "}"
            elif isinstance(x, tuple):
                return "(" + ", ".join(pretty(e) for e in x) + ")"
            return str(x)

        def print_wrapped(label: str, items: list[str]) -> None:
            print(label)
            line = "  "
            for item in items:
                if len(line) + len(item) + 2 > WRAP_LIMIT:
                    print(line.rstrip(", "))
                    line = "  "
                line += item + ", "
            if line.strip():
                print(line.rstrip(", "))

        # Stati
        state_items = [pretty(s) for s in sorted(self.states, key=pretty)]
        print_wrapped("----- STATES -----", state_items)

        # Alfabeto
        alphabet_items = [str(e) for e in sorted(self.alphabet, key=str)]
        print_wrapped("---- ALPHABET ----", alphabet_items)

        # Stato iniziale
        print("--- INIT STATE ---")
        print(" ", pretty(self.initial))

        # Transizioni
        print("----- DELTA ------")
        transitions = sorted(self.delta.items(), key=lambda x: (pretty(x[0][0]), str(x[0][1])))
        for (src, ev), dst in transitions:
            print(f"  {pretty(src)} --({ev})--> {pretty(dst)}")

        # Stati finali
        final_items = [pretty(f) for f in sorted(self.finals, key=pretty)]
        if final_items:
            print_wrapped("-- FINAL STATES --", final_items)
        else:
            print("-- FINAL STATES --\n  (none)")


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

    finals= set()
    for q in Q_state:
        for x in q:
            if x in g.finals:
                finals.add(q)


    return DFA(
        states=fz(Q_state),
        alphabet=fz(alphabet),
        initial=q0,
        delta=delta,
        finals=fz(finals),
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
    
    if not isinstance(dfa, DFA):
        raise TypeError("Argument should be a DFA.")
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


def draw_nfa_graphviz(
    nfa: NFA,
    filename: str = "nfa",
    view: bool = True,
    state_colors: Dict[str, Set[State]] | None = None,
) -> None:
    """Render an NFA to a PDF with Graphviz.

    - Ogni stato disegnato è in nfa.states.
    - Ogni (src, ev) → {dst1, dst2, …} produce tanti archi quanti sono i dst.
    """
    if not isinstance(nfa, NFA):
        raise TypeError("Argument should be a DFA.")
    def label(state: State) -> str:
        if isinstance(state, frozenset):
            return "{" + ", ".join(sorted(map(str, state))) + "}"
        return str(state)

    dot = Digraph("NFA", format="pdf")
    dot.attr(rankdir="LR", size="10,6")


    init_state = next(iter(nfa.initial))
    dot.node("__start__", "", shape="none", width="0")
    dot.edge("__start__", label(init_state))


    for s in nfa.states:
        shape = "doublecircle" if s in nfa.finals else "circle"
        color = "black"
        if state_colors:
            for col, group in state_colors.items():
                if s in group:
                    color = col
                    break
        dot.node(label(s), shape=shape, color=color, fontcolor=color)


    for (src, ev), dst_set in nfa.delta.items():
        if src not in nfa.states:
            raise ValueError(f"delta usa stato sorgente '{src}' non in nfa.states")
        for dst in dst_set:
            if dst not in nfa.states:
                raise ValueError(f"delta ({src!r}, {ev!r}) → '{dst}' non è in nfa.states")
            dot.edge(label(src), label(dst), label=str(ev))

    dot.render(filename, view=view, cleanup=True)



def mask(   g: DFA, 
            e_obs: frozenset[Event],
            e_unobs: frozenset[Event],
            relation: Dict[Event,Event] = {}
    ):
    

    for e in e_obs:
        relation[e]=e

    for e in e_unobs:
        relation[e]="eps"

    delta_new = {}
    
    for key, data in g.delta.items():
        data={data}
        newkey = (key[0], relation[key[1]])

        if newkey in delta_new.keys():
            data = delta_new[newkey] | fz(data)
        delta_new[newkey] = fz(data)    
    
    g_masked=NFA(
        states=g.states,
        alphabet=fz(e_obs | {"eps"}),
        initial=fz([g.initial]),
        delta=delta_new,
        finals=g.finals
    )

    return g_masked


def create_attack_observer(g: DFA, E_ins:frozenset[Event], E_era:frozenset[Event]):
    E_plus=[]
    delta=g.delta.copy()
    for e in E_ins:
        e_true=e + "+"
        E_plus.append(e_true)
        for state in g.states:
                delta[(state, e_true)] = state
        

    E_minus=[]
    for e in E_era:
        e_true=e + "-"
        E_minus.append(e_true)
        for state in g.states:
                if (state, e) in g.delta:
                    delta[(state, e_true)] = g.delta[(state, e)]

    E_minus=fz(E_minus)
    E_plus=fz(E_plus)

    obs_att=DFA(
        states=g.states,
        alphabet=fz(g.alphabet| E_plus | E_minus),
        initial=g.initial,
        delta=delta,
        finals=fz()
    )
    
    return obs_att, E_plus, E_minus

def gn_creator(n, ea=[], ep=[]):
    lstates=[str(i) for i in range(n+1)]
    avanti = ep
    indietro_self = ea - ep
    
    delta = {}
    #print("Avanti: ", avanti)
    #print("Indietro: ", indietro_self)
    
    for i in range(n+1):
        if i==n:
            for a in indietro_self:
                delta[(str(i), a)] = str(0)
        # print("i: ", i)
        else:
            for a in avanti:
                delta[(str(i), a)] = str(i+1)
            for a in indietro_self:
                delta[(str(i), a)] = str(0)
        
    g = DFA(
        states=fz(lstates),
        alphabet=fz(ea),
        initial="0",
        delta=delta
    )
    return g


def create_operator_observer(g: DFA, E_ins:frozenset[Event], E_era:frozenset[Event]):
    E_plus=[]
    empty_state=fz(["empty"])
    delta=g.delta.copy()
    states=g.states.copy()
    states=set(states)
    

    E_plus=[]
    for e in E_ins:
        e_true=e + "+"
        E_plus.append(e_true)
        for state in g.states:
                if (state, e) in g.delta:
                    delta[(state, e_true)] = g.delta[(state, e)]


    E_minus=[]

    for e in E_era:
        e_true=e + "-"
        E_minus.append(e_true)
        for state in g.states:
                delta[(state, e_true)] = state
    

     
    E_minus=fz(E_minus)
    E_plus=fz(E_plus)
    att_alphabet=fz(g.alphabet| E_plus | E_minus)

    for e in att_alphabet:
        for state in states:
            if (state, e) not in delta:
                delta[(state, e)] = empty_state

    states.add(empty_state)

    obs_att=DFA(
        states=fz(states),
        alphabet=att_alphabet,
        initial=g.initial,
        delta=delta,
        finals=fz()
    )
    
    return obs_att

def compute_forbidden(g:DFA):

    forbidden_states=[]
    empty_state=fz(["empty"])
    for s in g.states:
        if empty_state in s:
            forbidden_states.append(s)
    return forbidden_states

def trim_joint_observer_v2(g:DFA, e_obs, e_era, e_ins):
    
    def check_if_safe(g:DFA, s: State, e_ins, R_p):
        fifo = [s]
        e_plus = set([e + "+" for e in e_ins])
        alr_checked = set()
        while len(fifo)>0:
            curr=fifo.pop(0)
            alr_checked.add(curr)
            for ep in e_plus:
                step=g.step(curr, ep)
                if step in R_p:
                    act_ev=g.eventi_attivi(step)
                    if not act_ev<= e_plus or not act_ev:
                        return True
                
                if step and step not in alr_checked:
                    fifo.append(step)
                    alr_checked.add(step)
        return False


    def compute_g2(g:DFA, R_p, e_obs, e_era, e_ins):
        R = deepcopy(g.states)
        R_m_Rp = R - R_p

        g1 = []

        #R_p devono essere gli insiemi stealth 
        for r in R_p:
            for e in e_obs:
                if (g.step(r, e) in R_m_Rp) and (g.step(r, e + "-") not in R_p):
                    if r not in g1:
                        #print("stato", r,"finisce in g1")
                        g1.append(r)
        #g1 ora contiene gli insiemi con eventi pericolosi, che ora come ora sarebbero classificabili come weakly_non_stealthy

        weak=[]
        #
        for r in g1:
            if(not check_if_safe(g, r, e_ins, R_p-set(g1))):
                #print("stato", r, "NON FUGGE!")
                weak.append(r)

        #print("Weak interno:", weak)
        R_out=R_p - set(weak)

        return R_out, set(g1)

    forbidden_states=compute_forbidden(g)
    R_in =  g.states-set(forbidden_states)
    R_out=set()
    #print("Iterazione", 1)
    R_out, R_preempt = compute_g2(g, R_in, e_obs, e_era, e_ins)

    while R_out != R_in:
        R_in = R_out
        R_out, R_preempt = compute_g2(g, R_in, e_obs, e_era, e_ins)
    
    delta={}
    for (s,e) , step in g.delta.items():
        if s in R_out-R_preempt and step in R_out:
            delta[(s,e)]=g.delta[(s,e)]
        elif s in R_preempt and step in R_out:
            if e[len(e)-1]=="+":
                delta[(s,e)]=g.delta[(s,e)]
                print("Inserting", s, e , g.delta[(s,e)])

    
    R_out=Reach(delta, g.initial, g.alphabet, DFA=True)
    R_preempt=set(R_preempt) & set(R_out)
    delta = {
        (s,e): t
        for (s,e), t in delta.items()
        if s in R_out and t in R_out
    }
    
    
    trimmed=DFA(
        states=fz(R_out),
        alphabet=g.alphabet,
        initial=g.initial,
        delta=delta
    )

    return trimmed, R_preempt


def Reach(delta, s, E, DFA=True):
    def step(s_curr, e):
        out = delta.get((s_curr, e), None)
        if out is None:
            return set()
        if not DFA:
            # NFA: deve essere già un set
            return out
        else:
            # DFA: singolo stato, lo trasformiamo in set per uniformità
            return {out}

    fifo = deque([s])
    reached = set()

    while fifo:
        curr = fifo.popleft()
        if curr in reached:
            continue
        reached.add(curr)

        for e in E:
            for nxt in step(curr, e):
                if nxt not in reached:
                    fifo.append(nxt)

    return reached

# End of LibACCPS.py -----------------------------------------------------------