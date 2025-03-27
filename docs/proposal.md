# Project Proposal

**Benjamin Colby**, **Griffin Teller**

### Summary

We are going to implement a parallel SAT solver on a multicore CPU, an NVIDIA GPU, and an FPGA, and compare both performance characteristics (speed and power) as well as qualitatively compare design implications.

### Background

All efficiently verifiable computational problems can be transformed efficiently to an instance of CNF-SAT: given a conjunction of disjunctive clauses, does a satisfying assignment exist? And if so, what assignment? The DPLL algorithm solves CNF-SAT problems by assigning literals, identifying unit clauses and pure literals, and backtracking after reaching a contradiction. The psuedocode for DPLL is described below:

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
[citation for psuedocode](https://en.wikipedia.org/wiki/DPLL_algorithm#The_algorithm)

While still potentially exponential in time complexity, DPLL terminates efficiently for many natural problem instances. Additionally, the DPLL algorithm provides two possible modes of parallelism: first, over the recursive call tree, and second over the inner routines (`unit-propogate` and `pure-literal-assign`).

### The Challenge

The parallelism in the outer call-tree lends itself well to a fork-join model of parallelism, while the inner routines may operate over many thousands of literals, and may benefit massively data-parallel computation. Exploiting fork-join parallelism is simple on a multicore CPU, but the parallelism over the inner routines is limited. We expect better exploitation of the inherent parallism in the inner routines on a GPU, but task-scheduling may come with substantially greater overhead (from additional kernel launches or communication between threads to dynamically assign work). We hope iterating, measuring, and comparing GPU implementations will reveal interesting memory and scheduler characteristics, as well as deepen our understanding of memory and scheduling on GPUs.

We will also need to design data structures for the formula and assignment state that lend themselves to efficient parallel implementations. In particular, at every decision point in the algorithm, if the two children tasks are going to be parallelized over, then the assignment state must be copied. These can be large, potentially, and many efficient SAT data structures rely on linked-lists, making copying even more expensive. We will have to explore data structures which can share information between structures while retaining good caching properties for each machine type.

Additionally, we see several opportunities to augment the DPLL algorithm with communication to potentially improve performance. For instance, while searching for unit literals, as soon as one processor finds such a literal, it should be returned from the subroutine and all other processors can stop. Exploring the synchronization and/or communication tradeoff on different machine types will potentially be illuminating.


### Resources

There are many resources online explaining SAT solving algorithms and associated data structures, but Ruben Martins lecture notes on SAT solvers from 15-414: Bug Catching are particularly helpful: <br>
https://www.cs.cmu.edu/~15414/s22/s21/lectures/12-sat-solving.pdf <br>
https://www.cs.cmu.edu/~15414/f18/2018/lectures/20-sat-techniques.pdf

We'll be starting from scratch for our implementations. 

In terms of hardware, we will start with the GHC machines for both multicore CPUs and NVIDIA GPUs. Depending on our early results, if we feel it would add substantively to our analysis, we may want to test our CPU implementation on larger clusters (e.g. PSC), or on a heterogenous CPU like the Apple M1 Pro (we have access to this CPU personally). It would be a surprising but welcome result if we were able to utilize the NVIDIA 2080 effectively enough to warrant testing on a more powerful GPU, in which case we may want to test on a more powerful GPU to observe performance scaling. 

### Goals and Deliverables

**Plan to achieve**
- Sequential implementation of DPLL
- Multi-core implementation of DPLL
- GPU implementation of DPLL
- Augmentation of multi-core implementation with inter-process communication
- Create testing framework and benchmarks
- Analysis of the characteristics of various data structures for the CNF formula and assignment state
- Analysis of performance bottlenecks on various implementations
- Analysis of various implementations under different workloads

**Hope to achieve**
- *Ahead of schedule*
    - FPGA implementation of DPLL
    - Experiment with heterogeneous hardware
    - Analysis of power consumption of various implementations
- *Behind schedule*
    - Forgo inter-process communication
    - Shorter analysis of data structures

### Schedule

**Deliverable schedule**

| Date | Deliverable              |
| ---- | ------------------------ |
| 3/26 | Project Proposal         |
| 4/15 | Project Milestone Report |
| 4/28 | Final Project Report     |
| 4/29 | Poster Session           |

**Project schedule**

| Week        | Tasks                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| 3/24 - 3/30 | - finish project proposal <br> - study existing research <br> - write sequential reference implementation        |
| 3/31 - 4/6  | - write multi-core implementation <br> - create testing framework <br> - create benchmarks                       |
| 4/7 - 4/13  | - write GPU implementation <br> - perform bottleneck and workload analysis <br> - write project milestone report |
| 4/14 - 4/20 | - augment multi-core implementation with inter-process communication <br> - heterogeneous hardware analysis      |
| 4/21 - 4/27 | - perform final performance analysis <br> - write the final report <br> - prepare for poster session             |