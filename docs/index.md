# Parallel DPLL SAT Solver
**Benjamin Colby**
**Griffin Teller**

griffinteller.github.io/parallelSAT



## Project Proposal

### Summary

We are going to implement a parallel SAT solver on a multicore CPU, an NVIDIA GPU, and an FPGA, and compare both performance characteristics (speed and power) as well as qualitatively compare design implications.

### Background

The DPLL algorithm solves CNF-SAT problems by assigning literals, identifying unit clauses and pure literals, and backtracking after reaching a contradiction. The psuedocode for DPLL is described below:

```
Algorithm DPLL
    Input: A set of clauses Φ.
    Output: A truth value indicating whether Φ is satisfiable.

function DPLL(Φ)
    // unit propagation:
    while there is a unit clause {l} in Φ do
        Φ ← unit-propagate(l, Φ);

    // pure literal elimination:
    while there is a literal l that occurs pure in Φ do
        Φ ← pure-literal-assign(l, Φ);

    // stopping conditions:
    if Φ is empty then
        return true;
    if Φ contains an empty clause then
        return false;

    // DPLL procedure:
    l ← choose-literal(Φ);
    return DPLL(Φ ∧ {l}) or DPLL(Φ ∧ {¬l});
```
(citation for psuedocode) [https://en.wikipedia.org/wiki/DPLL_algorithm#The_algorithm]

### The Challenge

### Resources

### Goals and Deliverables

### Schedule