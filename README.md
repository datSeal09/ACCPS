# LibACCPS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**LibACCPS** is a Python library for modeling, analyzing, and visualizing Discrete-Event Systems (DES). It provides robust tools for working with Deterministic Finite Automata (DFA) and Non-Deterministic Finite Automata (NFA), including support for observer construction, concurrent composition, supervisor synthesis utilities, and attack modeling in the context of cybersecurity for automated systems.

The library is built with a focus on clear API design and practical application in academic research and system analysis, particularly for systems requiring formal verification and control.

Most of the alghorithms have been inspired by these 2 works:
* \[1] Q. Zhang, C. Seatzu, Z. W. Li, and A. Giua. Joint state estimation under attack of discrete event systems. *IEEE Access*, 9:168068–168079, 2021.
* \[2] Q. Zhang, C. Seatzu, Z. W. Li, and A. Giua. Selection of a stealthy and harmful attack function in discrete event systems. *Scientific Reports*, 12, 2022.


## Core Concepts

LibACCPS allows you to formally define and manipulate automata:

* **Deterministic Finite Automata (DFA):** Automata where each state has exactly one transition for each symbol in the alphabet.
* **Non-Deterministic Finite Automata (NFA):** Automata where states can have zero, one, or multiple transitions for a given symbol, including epsilon (ε) transitions for unobservable moves.
* **Observers:** Deterministic abstractions of NFAs, crucial for understanding system behavior based on observable events. LibACCPS can compute observers from NFAs, correctly handling ε-closures.
* **Concurrent Composition:** Combining multiple DFAs to model the joint behavior of interacting systems.
* **Masking:** Transforming a DFA into an NFA by defining certain events as unobservable (mapping them to "eps").
* **Attack Modeling:** Creating specialized observers (attack observers, operator observers) to analyze system behavior under event insertion or erasure attacks.
* **State-Space Analysis:** Utilities like `Reach` to find reachable states and `trim_joint_estimator` for reducing state-space based on observability and attack scenarios.

## Key Features

* **Robust Automata Construction:** Create and validate DFAs and NFAs with clear state, alphabet, transition, initial, and final state definitions.
* **Epsilon Transition Handling:** Full support for ε-transitions in NFAs and their correct handling during observer computation.
* **Observer Computation:** Generate a deterministic observer automaton from an NFA (e.g., after a `mask` operation).
* **Concurrent Composition:** Synchronous product of two DFAs to model parallel system execution.
* **Event Masking:** Convert DFAs to NFAs by specifying observable and unobservable events.
* **Attack Scenario Modeling:**
    * `create_attack_observer`: Models an attacker's view, considering event insertion and erasure.
    * `create_operator_observer`: Models an operator's view under potential attacks, including an 'empty' or failure state.
* **State Reachability and Trimming:**
    * `Reach`: Determine all reachable states from an initial state.
    * `compute_NS`: Identify non-stealthy states (states indicating an error or unexpected observation).
    * `trim_joint_estimator`: Advanced function to prune a joint observer automaton based on stealthiness and preemption criteria under attack scenarios.
* **Utility Generators:**
    * `gn_creator`: Helper to create specific types of bounded counter automata.
* **Graphviz Visualization:** Generate graphical representations of DFAs and NFAs.
    * Output to PDF files for documentation.
    * Direct inline rendering in Jupyter Notebooks and Colab.
* **Typed and Validated:** Uses Python type hints and performs consistency checks during automaton construction.

## Installation
First of all install graphviz in your local machine from https://graphviz.org/download/ (if you run it in colab this step is unnecessary)

Than you are ready to download the library!

```bash
wget https://raw.githubusercontent.com/datSeal09/ACCPS/main/dist/libaccps-<version>.tar.gz -O libaccps-<version>.tar.gz
```
and installing it using pip
```bash
pip install libaccps-<version>.tar.gz
```
For example:
```bash
wget https://raw.githubusercontent.com/datSeal09/ACCPS/main/dist/libaccps-1.0.0.tar.gz -O libaccps-1.0.0.tar.gz
pip install libaccps-1.0.0.tar.gz
```
