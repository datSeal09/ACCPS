from typing import Dict, Tuple, Iterable, Hashable, FrozenSet
from copy import deepcopy, copy
from graphviz import Digraph
from collections import deque
import networkx as nx
State = Hashable
Event = Hashable
fz=frozenset
class DFA:
    def __init__(self,
                 states: frozenset[State],
                 alphabet: frozenset[Event],
                 initial: State,
                 delta: Dict[Tuple[State, Event], State],
                 finals: frozenset[State] = frozenset()):
        # Salva i dati
        self._consistency(states, alphabet, initial, delta, finals)
        self.states = states
        self.alphabet = alphabet
        self.initial = initial
        self.delta = delta
        self.finals = finals


    def _consistency(self, states: frozenset[State], alphabet: frozenset[Event], initial: State, delta: Dict[Tuple[State, Event], State], finals: frozenset[State] = frozenset()):
        # Controlla che tutti gli stati siano validi
        if not states:
            raise ValueError("States set is empty")
        if initial not in states:
            raise ValueError("Initial state is not in states")
        if not finals.issubset(states):
            raise ValueError("Final states must be a subset of states")
        for (s, e), nxt in delta.items():
            if s not in states:
                raise ValueError(f"State {s} is not in states")
            if e not in alphabet:
                raise ValueError(f"Event {e} is not in alphabet")
            if nxt not in states:
                raise ValueError(f"Next state {nxt} is not in states")
            
    
    # Metodo: data una stato e un evento, restituisci il nuovo stato
    def step(self, s: State, e: Event) -> State:
        return self.delta.get((s, e), frozenset())

    # Metodo: fai partire l'automa da initial su una parola
    def run(self, word: Iterable[Event]) -> State:
        s = self.initial
        for e in word:
            s = self.step(s, e)
            if s is None:
                return None
        return s

    # Metodo: verifica se una parola è accettata
    def accepts(self, word: Iterable[Event]) -> bool:
        return self.run(word) in self.finals
    
    def get_transitions(self, state: State) -> Dict[Event, frozenset[State]]:
        """
        Rende omogenee DFA e NFA: per ogni evento restituisce
        un *insieme* (frozenset) di stati destinazione.
        """
        trans: Dict[Event, frozenset[State]] = {}
        for (s, e), nxt in self.delta.items():
            if s == state:
                trans[e] = frozenset([nxt])          # singleton‑set
        return trans
    def eventi_attivi(self, state: State) -> set:
        attivi = set()
        for (s, e), next_state in self.delta.items():
            if s == state:
                attivi.add(e)
        return attivi

    def print_data(self):
        print("----- STATES -----")
        print(self.states   )
        print("---- ALPHABET ----")
        print(self.alphabet )
        print("--- INIT STATE ---")
        print(self.initial  )
        print("----- DELTA ------")
        for i, j in self.delta:
            print(f"{i} --\\{j}\\--> {self.delta[i, j]}")
        print("-- FINAL STATES --")
        print(self.finals   )
    

class NFA:
    def __init__(self,
                 states: frozenset[State],
                 alphabet: frozenset[Event],
                 initial: Iterable[State],                    # ← accetta lista/insieme
                 delta: Dict[Tuple[State, Event], frozenset[State]],
                 finals: frozenset[State] = frozenset()):
        self.states   = states
        self.alphabet = alphabet
        self.initial  = initial                 
        self.delta    = delta
        self.finals   = finals

    # passo singolo
    def step(self, s: State, e: Event) -> frozenset[State]:
        return self.delta.get((s, e), frozenset())

    # esecuzione su parola
    def run(self, word: Iterable[Event]) -> frozenset[State]:
        current: frozenset[State] = self.initial
        for e in word:  
            next_states: frozenset[State] = frozenset().union(
                *(self.step(s, e) for s in current)
            )
            current = next_states
        return current

    # accettazione
    def accepts(self, word: Iterable[Event]) -> bool:
        return any(s in self.finals for s in self.run(word))
    
    def print_data(self):
        print("----- STATES -----")
        print(self.states   )
        print("---- ALPHABET ----")
        print(self.alphabet )
        print("--- INIT STATE ---")
        print(self.initial  )
        print("----- DELTA ------")
        for i, j in self.delta:
            print(f"{i} --\\{j}\\--> {self.delta[i, j]}")
        print("-- FINAL STATES --")
        print(self.finals   )
    
    def get_transitions(self, state: State) -> Dict[Event, frozenset[State]]:
        transitions = {}
        for (s, e), next_states in self.delta.items():
            if s == state:
                transitions[e] = next_states
        return transitions
    
    



def compute_D_eps(g: NFA, x: State) -> set:
    D_eps = set([x]) # D_eps inizia contenendo lo stato da cui partiamo
    new = [x]        # anche new è una lista che contiene solo lo stato iniziale
    while len(new) > 0:            # finchè new non diventa l'insieme vuoto
        x_curr = new.pop(0)        # Estrae l'elemento in posizione 0 e lo rimuove dalla lista, tipo pop stack
        step=g.step(x_curr, "eps") # Calcoliamo il delta, l'insieme di stati raggiungibili da x_curr
        for x_new in step:
            # Se dentro step ci sono gli x_new, se uno di loro è stato generato per la prima volta
            if x_new not in D_eps:
                # Aggiungiamo a D_eps lo stato nuovo!
                D_eps.add(x_new)
                # Aggiungiamolo anche all'insieme di stati nuovi così lo espandiamo
                new.append(x_new)
    return D_eps


def compute_alpha(g:NFA, x:set[State], e:Event) -> set:
    alpha=set()
    for s in x:
        alpha|=g.step(s, e)
    return alpha
    
def compute_beta(D_eps: dict, x:set[State]) -> set:
    beta=set()
    for s in x:
        beta|=D_eps[s]
    return beta
    



def compute_observer(   g:NFA
    ):
    
    # Abbozziamo dei check
    if "eps" not in g.alphabet:
        print("Amici controllate, eps non è nell'alfabeto")
    
    D_eps = {}
    temp_eps=set()
    # Primo step dell'algoritmo, costruiamo D_eps
    for x in g.states:
        D_eps[x]=compute_D_eps(g, x)
    
    g_init=set({g.initial})
    q0=set()

    # Costruiamo q0, l'insieme di stati iniziali
    while len(g_init)>0:
        q0|=D_eps[g_init.pop()]
    q0=fz(q0)
    Q_state={q0}
    
    Q_new=[q0]
    alphabet=g.alphabet-{"eps"}
    delta={}

    #state_idx=0
    #state_2_index={
    #    fz(q0):state_idx
    #}
    
    while len(Q_new)>0:
        q_curr=Q_new.pop(0)
        #idx_curr = state_2_index[q_curr]
        for e in alphabet:
            alpha=compute_alpha(g, q_curr, e)
            beta=compute_beta(D_eps, alpha)

            if len(beta)>0:
                delta[(fz(q_curr), e)]= fz(beta)
                if beta not in Q_state:
                    #state_idx+=1
                    #state_2_index[beta]=state_idx
                    Q_new.append(beta)
                    Q_state.add(fz(beta))
    
    
    obs=DFA(
        states=Q_state,
        alphabet=alphabet,
        initial=q0,
        delta=delta,
        finals=fz()
    )
    
    return obs

def mask(   g:DFA, 
            e_obs =    {},
            e_unobs =  {},
            e_rel =    {},
            relation = {}
    ):
    
    # Completa relation se data incompleta, solo con relabel
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
    
    g_masked = deepcopy(g)
    g_masked.delta = delta_new
    g_masked.alphabet = fz(e_obs | {"eps"})

    return g_masked


def concurrent_composition(a:NFA, b:NFA):
    initial_state = [a.initial[0], b.initial[0]]

    syncronized_events = a.alphabet & b.alphabet
    print("This is the syncronized events")
    print(syncronized_events)
    
    all_possible_events = a.alphabet | b.alphabet
    states_to_analyze = [initial_state]
    states_already_analyze = [initial_state]
    
    conc_delta = []
    
    while len(states_to_analyze) != 0: 
        state = states_to_analyze.pop()
        states_already_analyze.append(state)
        
        print("This is the state")
        print(state)
        tran_a = a.get_transitions(state[0])
        tran_b = b.get_transitions(state[1])
        print("This is the transitions of a")
        print(tran_a)
        print("This is the transitions of b")
        print(tran_b)
        for event in all_possible_events:
            if event in syncronized_events:
                try:
                    print("FLAG")
                    print(tran_a[event] + tran_b[event])
                    nstate =  tran_a[event] + tran_b[event]
                    print("this is the nstate")
                    print(nstate)
                    conc_delta.append([[state, event], nstate]) 
                except:
                    nstate = state
            else:
                try:
                    if event in a.alphabet:
                        nstate = [tran_a[event][0], state[1]]
                    else:
                        nstate = [state[0], tran_b[event][0]]
                    conc_delta.append([[state, event], nstate]) 
                except:
                    nstate = state
            if (nstate not in states_already_analyze) and (nstate not in states_to_analyze):
                states_to_analyze.append(nstate)
    return conc_delta




def create_attack_observer(g, E_ins, E_era):
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



def create_operator_observer(g, E_ins, E_era):
    E_plus=[]
    empty_state="empty"
    delta=g.delta.copy()
    states=g.states.copy()
    

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


    states.add("empty")
    obs_att=DFA(
        states=states,
        alphabet=att_alphabet,
        initial=g.initial,
        delta=delta,
        finals=fz()
    )
    
    return obs_att




from typing import Dict, Tuple, List, Set

def concurrent_composition(a: DFA, b: DFA) -> DFA:
    sync_events = a.alphabet & b.alphabet
    all_events  = a.alphabet | b.alphabet

    initial = (a.initial, b.initial)
    states: Set[Tuple[State, State]] = {initial}
    delta: Dict[Tuple[Tuple[State, State], Event], Tuple[State, State]] = {}

    to_visit: List[Tuple[State, State]] = [initial]
    while to_visit:
        sA, sB = curr = to_visit.pop()

        for e in all_events:
            # Evento sincronizzato
            if e in sync_events:
                nextA = a.step(sA, e)
                nextB = b.step(sB, e)
                if nextA != frozenset() and nextB != frozenset():
                    ns = (nextA, nextB)
                else:
                    continue  # uno dei due è indefinito → non aggiungo
            # Evento solo di A
            elif e in a.alphabet:
                nextA = a.step(sA, e)
                if nextA != frozenset():
                    ns = (nextA, sB)
                else:
                    continue
            # Evento solo di B
            elif e in b.alphabet:
                nextB = b.step(sB, e)
                if nextB != frozenset():
                    ns = (sA, nextB)
                else:
                    continue
            else:
                print("ERROREERROREERROREERROREERROREERROREERROREERROREERROREERRORE")  # evento non riconosciuto (non dovrebbe succedere)

            delta[(curr, e)] = ns
            if ns not in states:
                states.add(ns)
                to_visit.append(ns)

    finals = frozenset(
        (p, q) for (p, q) in states
        if p in a.finals and q in b.finals
    )

    return DFA(
        states   = frozenset(states),
        alphabet = frozenset(all_events),
        initial  = initial,
        delta    = delta,
        finals   = finals
    )


def draw_dfa_graphviz(dfa:DFA, filename="dfa", view=True, state_colors=None):
    def pretty(state):
        if isinstance(state, frozenset):
            items = []
            for item in sorted(state, key=str):
                if isinstance(item, tuple):
                    a, b = item
                    a_str = pretty(a)
                    b_str = pretty(b)
                    items.append(f"({a_str} , {b_str})")
                else:
                    items.append(pretty(item))
            return "{" + ", ".join(items) + "}"
        elif isinstance(state, tuple):
            return "(" + ", ".join(pretty(s) for s in state) + ")"
        else:
            return str(state)

    dot = Digraph(name="DFA", format='pdf')
    dot.attr(rankdir='LR', size='10,6')
    dot.attr('node', shape='circle', fontsize='12')

    dot.node('__start__', label='', shape='none', width='0')
    dot.edge('__start__', pretty(dfa.initial))

    for state in dfa.states:
        shape = 'doublecircle' if state in dfa.finals else 'circle'
        state_str = pretty(state)

        color = 'black'  # default
        if state_colors:
            for col, states in state_colors.items():
                if state in states:
                    color = col
                    break  # usa il primo colore trovato

        dot.node(state_str, shape=shape, color=color, fontcolor=color)

    transitions = {}
    for (src, symbol), dst in dfa.delta.items():
        key = (pretty(src), pretty(dst))
        transitions.setdefault(key, []).append(str(symbol))

    for (src, dst), labels in transitions.items():
        label = ", ".join(sorted(labels))
        dot.edge(src, dst, label=label)

    dot.render(filename, view=view, cleanup=True)


fz=frozenset
def gn_creator(n, ea=[], ep=[]):
    # g=lib.DFA(
    lstates=[str(i) for i in range(n+1)]
    states=fz(lstates),
    initial="0",
    # print("States: ", states)
    alphabet=fz(ea),
    # initial="0",
    avanti = ep
    indietro_self = ea - ep
    
    delta = {}
    print("Avanti: ", avanti)
    print("Indietro: ", indietro_self)
    
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



def compute_forbidden(g:DFA):

    forbidden_states=[]
    for s in g.states:
        if "empty" in s:
            forbidden_states.append(s)
    return forbidden_states


def vedi_se_fugge(g:DFA, s: State, e_ins, R_p):
    fifo = [s]
    alr_checked = {s}
    while len(fifo)>0:
        curr=fifo.pop(0)
        for e in e_ins:
            step=g.step(curr, e + "+")
            if step in R_p:
                return True
            
            if step and step not in alr_checked:
                fifo.append(step)
                alr_checked.add(curr)
    return False




def compute_idk(g:DFA, R_p, e_obs, e_era, e_ins):
    R = deepcopy(g.states)
    R_m_Rp = R - R_p

    g1 = []

    #R_p devono essere gli insiemi stealth 
    for r in R_p:
        for e in e_obs:
            if (g.step(r, e) in R_m_Rp) and (g.step(r, e + "-") not in R_p):
                if r not in g1:
                    print("stato", r,"finisce in g1")
                    g1.append(r)
    #g1 ora contiene gli insiemi con eventi pericolosi, che ora come ora sarebbero classificabili come weakly_non_stealthy

    weak=[]
    #
    for r in g1:
        if(not vedi_se_fugge(g, r, e_ins, R_p-set(g1))):
            print("stato", r, "NON FUGGE!")
            weak.append(r)

    print("Weak interno:", weak)
    R_out=R_p - set(weak)

    return R_out, g1


def trim_joint_observer(g:DFA, e_obs, e_era, e_ins):
    forbidden_states=compute_forbidden(g)
    R_in =  g.states-set(forbidden_states)
    R_out=set()
    print("Iterazione", 1)
    R_out, R_preempt = compute_idk(g, R_in, e_obs, e_era, e_ins)
    i = 2

    while R_out != R_in:
        print("Iterazione", i)
        R_in = R_out
        R_out, R_preempt = compute_idk(g, R_in, e_obs, e_era, e_ins)
        i += 1
    
    delta={}
    for (s,e) , step in g.delta.items():
        if s in R_out and step in R_out:
            delta[(s,e)]=g.delta[(s,e)]
    

    trimmed=DFA(
        states=R_out,
        alphabet=g.alphabet,
        initial=g.initial,
        delta=delta
    )

    return trimmed, R_preempt
        
        







def trim_joint_observer_v2(g:DFA, e_obs, e_era, e_ins):
    
    def vedi_se_fugge2(g:DFA, s: State, e_ins, R_p):
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


    def compute_idk2(g:DFA, R_p, e_obs, e_era, e_ins):
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
            if(not vedi_se_fugge2(g, r, e_ins, R_p-set(g1))):
                #print("stato", r, "NON FUGGE!")
                weak.append(r)

        #print("Weak interno:", weak)
        R_out=R_p - set(weak)

        return R_out, g1

    forbidden_states=compute_forbidden(g)
    R_in =  g.states-set(forbidden_states)
    R_out=set()
    #print("Iterazione", 1)
    R_out, R_preempt = compute_idk2(g, R_in, e_obs, e_era, e_ins)
    i = 2

    while R_out != R_in:
        #print("Iterazione", i)
        R_in = R_out
        R_out, R_preempt = compute_idk2(g, R_in, e_obs, e_era, e_ins)
        i += 1
    
    delta={}
    for (s,e) , step in g.delta.items():
        if s in R_out and step in R_out:
            delta[(s,e)]=g.delta[(s,e)]
    print("R_out", R_out)
    R_out=Reach(delta, g.initial, g.alphabet, DFA=True)
    print("R_out", R_out)
    R_preempt=set(R_preempt) & set(R_out)
    trimmed=DFA(
        states=R_out,
        alphabet=g.alphabet,
        initial=g.initial,
        delta=delta
    )

    return trimmed, R_preempt
        
        








def find_weakly_forbidden(g: DFA, forb: Set[State], e_uc: Set[Event]):
    delta_inv=invert_delta(g.delta)
    weakly_forbidden=set()
    for s in forb:
        weakly_forbidden|=Reach(delta_inv, s, e_uc, DFA=False)
    return weakly_forbidden


def invert_delta(delta: dict[tuple, State]) -> dict[tuple, State]:
    delta_inv = {}
    for (s, e), next in delta.items():
        key = (next, e)
        if key not in delta_inv:
            delta_inv[key] = set()
        delta_inv[key].add(s)
    return delta_inv

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


class SupervisorV2:
    def __init__(self,
                 g: DFA,
                 e_c: frozenset[Event],
                 e_uc: frozenset[Event],
                 MisRelation,
                 costi,):
        self.costi=costi
        self.g=g
        self.e_c=e_c
        self.MisRelation=MisRelation
        self.e_uc=e_uc
        weakly=find_weakly_forbidden(g, MisRelation, e_uc)
        delta_inv=invert_delta(g.delta)
        self.forbidden=weakly|MisRelation
        if g.initial in self.forbidden:
            raise ValueError("Initial state is forbidden")
        potential=set()
        for s in weakly|MisRelation:
            potential|=Reach(delta_inv, s, g.alphabet, DFA=False)
        
        delta={}
        safe_potential=potential-self.forbidden
        for (s,e) , step in g.delta.items():
            if s in safe_potential and step in potential:
                delta[(s,e)]=g.delta[(s,e)]

        self.trimmed=DFA(
            states=Reach(delta, g.initial, g.alphabet, DFA=True),
            alphabet=g.alphabet,
            initial=g.initial,
            delta=delta
        )

        draw_dfa_graphviz(self.trimmed, filename="reversed")

    


    def calcola_min_cut(self):
        Gf = nx.DiGraph()
        SRC, SNK = 'SOURCE', 'SINK'
        Gf.add_node(SRC); Gf.add_node(SNK)

        # SOURCE → initial
        Gf.add_edge(SRC, self.trimmed.initial, capacity=float('inf'))

        # gadget per ogni (q,e)->qn
        for (q, e), qn in self.trimmed.delta.items():
            tn = f"{q}#{e}"
            Gf.add_edge(q, tn, capacity=float('inf'))
            cap = self.costi[e] if e in self.e_c else float('inf')
            Gf.add_edge(tn, qn, capacity=cap)

        # bad → SINK
        for qb in self.forbidden:
            Gf.add_edge(qb, SNK, capacity=float('inf'))

        # calcola min-cut
        cut_value, (reachable, non_reachable) = nx.minimum_cut(
            Gf, SRC, SNK, capacity='capacity'
        )

        # estrai i blocchi: guarda '#' su u, non su v
        blocchi = set()
        for u, v, dat in Gf.edges(data=True):
            if u in reachable and v in non_reachable and dat['capacity'] < float('inf'):
                if '#' in u:
                    q, e = u.split('#', 1)
                    blocchi.add((q, e))

        return cut_value, blocchi

    def OptimalSupervisor(self):
        INFTY=float('inf')
        graph, SOURCE, SINK = ConvertDFA2Gliph(self.trimmed, self.forbidden, self.e_c, self.costi)

        rev_graph, flow = EdmondsKarp(graph, SOURCE, SINK)
        if flow == INFTY:
            return INFTY, set()
        reachable_nodes = min_cut(rev_graph, SOURCE)


        cut_edges = get_min_cut_edges(rev_graph, reachable_nodes)

        disabled = extract_disabled_events(cut_edges)

        return flow, disabled




from collections import defaultdict

class Edge:
    def __init__(self, s, t, cap, flow):
        self.s = s
        self.t = t
        self.cap = cap
        self.flow = flow
        self.rev = None

def add_edge(graph, u, v, cap, flow=0):
    fwd = Edge(u, v, cap, flow)
    rev = Edge(v, u, 0, flow)
    fwd.rev=rev
    rev.rev=fwd
    graph[u].append(fwd)
    graph[v].append(rev)


def add_transition(graph, from_state, to_state, cap, event):
    intra_node=f"{from_state}->{event}"
    add_edge(graph, from_state, intra_node, cap=float('inf'))
    add_edge(graph, intra_node, to_state, cap)



def ConvertDFA2Gliph(g:DFA, forbidden_states:Set[State], e_c, cost_dict):
    INFTY=float('inf')
    graph=defaultdict(list)
    for (s, e), step in g.delta.items():
        if e in e_c:
            cost=cost_dict[e]
        else:
            cost=INFTY
        add_transition(graph, s, step, cost, e)
    SOURCE="SOURCE"
    SINK="SINK"


    add_edge(graph, SOURCE, g.initial, INFTY)

    for s_f in forbidden_states:
        add_edge(graph, s_f, SINK, INFTY)
    
    return graph, SOURCE, SINK

def NoneDict():
    return None

def EdmondsKarp(graph, SOURCE, SINK):
    flow=0
    INFTY=float('inf')
    flag=True

    while flag:
        fifo = deque()
        fifo.append(SOURCE)
        pred=defaultdict(NoneDict)
        while fifo and pred[SINK] == None:
            curr=fifo.popleft()
            for edge in graph[curr]:
                next=edge.t
                if pred[next] is None and next != SOURCE and edge.cap>edge.flow:
                    pred[next] = edge
                    fifo.append(next)
        
        if pred[SINK] != None:
            df=INFTY
            edge= pred[SINK]
            while edge:
                df = min (df, edge.cap - edge.flow)
                edge = pred[edge.s]
            edge= pred[SINK]
            while edge:
                edge.flow=edge.flow+df
                edge.rev.flow=edge.rev.flow-df
                edge = pred[edge.s]
            flow=flow+df

        if pred[SINK]==None:
            flag=False
    
    return graph, flow

        

# Idea: Partiamo da source e andiamo a vedere tutti i possibili archi uscenti
# Se l'arco uscente non è saturo allora andiamo ad aggiungere il nodo in cui arriviamo a reach_source
# Se è saturo allora tagliandolo separareremo al minimo costo un percorso tra source e sink
def min_cut(graph, SOURCE):
    Reach_SOURCE = set()
    queue = deque([SOURCE])
    while queue:
        state = queue.popleft()
        if state in Reach_SOURCE:
            continue
        Reach_SOURCE.add(state)
        for edge in graph[state]:
            # Una volta finito se abbiamo ancora spazio in capacità vuol dire che lui non è quello minimo da bloccare.
            if edge.cap > edge.flow and edge.t not in Reach_SOURCE:
                queue.append(edge.t)
    return Reach_SOURCE

def get_min_cut_edges(graph, reachable):
    cut_edges = []
    for u in reachable:
        for edge in graph[u]:
            if edge.t not in reachable and edge.cap > 0: # Importante qua prendo solo forward edges
                cut_edges.append(edge)
    return cut_edges

def extract_disabled_events(cut_edges):
    disabled = set()
    for edge in cut_edges:
        if "->" in edge.s and edge.cap > 0:
            split=edge.s.split("->")
            remove=(split[0], split[1])
            disabled.add(remove)
        else:
            print(f"Arco ignorato nel cut: {edge.s} → {edge.t}")
    return disabled