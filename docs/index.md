# Parallel DPLL SAT Solver
**Benjamin Colby**
**Griffin Teller**

griffinteller.github.io/parallelSAT



## Project Proposal

### Summary

We are going to implement a parallel SAT solver on a multicore CPU, an NVIDIA GPU, and an FPGA, and compare both performance characteristics (speed and power) as well as qualitatively compare design implications.

### Background

Many computational problems (in fact, all verifiable The DPLL algorithm solves CNF-SAT problems by assigning literals, identifying unit clauses and pure literals, and backtracking after reaching a contradiction. The psuedocode for DPLL is described below:

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

While still EXPTIME, DPLL terminates efficiently for many natural problem instances. Additionally, the DPLL algorithm provides two possibles modes of parallelism: first, over the recursive call tree, and second over the inner routines (`unit-propogate` and `pure-literal-assign`).

### The Challenge

The parallelism in the outer call-tree lends itself well to a fork-join model of parallelism, while the inner routines may operate over many thousands of literals, and may benefit massively data-parallel computation. Exploiting fork-join parallelism is simple on a multicore CPU, but the parallelism over the inner routines is limited. We expect better exploitation of the inherent parallism in the inner routines on a GPU, but task-scheduling may come with substantially greater overhead (from additional kernel launches or communication between threads to dynamically assign work). We hope iterating, measuring, and comparing GPU implementations will reveal interesting memory and scheduler characteristics, as well as deepen our understanding of memory and scheduling on GPUs.

We will also need to design data structures for the formula and assignment state that lend themselves to efficient parallel implementations. In particular, at every decision point in the algorithm, if the two children tasks are going to be parallelized over, then the assignment state must be copied. These can be large, potentially, and many efficient SAT data structures rely on linked-lists, making copying even more expensive. We will have to explore data structures which can share information between structures while retaining good caching properties for each machine type.

Additionally, we see several opportunities to augment the DPLL algorithm with communication to potentially improve performance. For instance, while searching for unit literals, as soon as one processor finds such a literal, it should be returned from the subroutine and all other processors can stop. Exploring the synchronization and/or communication tradeoff on different machine types will potentially be illuminating.


### Resources

There are many resources online explaining SAT solving algorithms and associated data structures, but Ruben Martins lecture notes on SAT solvers from 15-414: Bug Catching are particularly helpful:
https://www.cs.cmu.edu/~15414/s22/s21/lectures/12-sat-solving.pdf
https://www.cs.cmu.edu/~15414/f18/2018/lectures/20-sat-techniques.pdf

We'll be starting from scratch for our implementations. 

In terms of hardware, we will start with the GHC machines for both multicore CPUs and NVIDIA GPUs. Depending on our early results, if we feel it would add substantively to our analysis, we may want to test our CPU implementation on larger clusters (e.g. PSC), or on a heterogenous CPU like the Apple M1 (we have access personally).

### Goals and Deliverables

### Schedule