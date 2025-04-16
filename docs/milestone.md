# Project Milestone

**Benjamin Colby**, **Griffin Teller**

### Summary of Completed Work

We have implemented a working DPLL SAT solver for each of the three target architectures:
- Single-Core:
    A sequential reference implementation of DPLL that runs on a single core.

- Multi-Core: 
    A parallel version that launches new threads when the algorithm recurses, assigning two pthreads to explore independent branches of the two literal assignment cases (assign literal to true, assign literal to false).

- GPU:
    A parallel version that takes advantage of data parallelism within the inner steps of DPLL (unit literal propogation and unassigned literal search). It does not currently take exploit the parallelism between assignment decisions, though this is what we plan to explore next.

We have also built a testing framework that enables us to analyze the performance of the different implementations under different workloads. Our benchmarks are sourced from [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), which contains a database of Uniform Random-3-SAT CNF problems. The benchmarks vary in the total number of literals, the number of clauses, and the satisfiability of the problem.

### Preliminary Results

**Multi-Core**

We gathered the execution time and cache miss rate for both the sequential reference solution and the multi-core implementation on four different benchmarks. The benchmarks are all satisfiable formulas, with {50, 75, 150, 175} literals and {218, 325, 645, 753} clauses, respectively. Data was gathered on an Intel Xeon Silver 4208 with a max fork depth of 3 (after which it calls the sequential implementation).

| Benchmark | time (ref) | time (multi) | cache miss % (ref) | cache miss % (multi) |
| --------- | ---------- | ------------ | ------------------ | -------------------- |
| uf50-218  | 0.078370 s | 0.173443 s   | 46.866 %           | 28.971 %             |
| uf75-325  | 0.373130 s | 0.258275 s   | 44.734 %           | 26.721 %             |
| uf150-645 | 858.7970 s | 868.7310 s   | 28.927 %           | 29.129 %             |
| uf175-753 | Did not terminate

The multi-core implementation exhibits mariginal speedup over the reference implementation for medium-sized benchmarks, and has a higher cache hit rate. Further analysis is needed to determine why the multi-core speedup is so poor, and to characterize the performance with different max fork depths and problem sizes. We need to determine if the process is memory bound for high thread counts and examine cache performance at different levels within the memory heirarchy. A potential improvement over the current implementation is to create a thread pool instead of limiting thread launches to the shallow recursive iterations.

**GPU** 

| Benchmark | time (init) | time (compute) | time (total) | 
| --------- | ----------- | -------------- | ------------ |
| uf50-218  | 0.254 s     | 0.126 s        | 0.380 s
| uf75-325  | 0.257 s     | 0.007 s        | 264 ms
| uf150-645 | 0.260 s     | 25.215 s       | 25.475 s
| uf175-753 | 0.253 s     | 573.333 s      | 573.586 s

Gathered on an NVIDIA RTX 2070

We see a substantial speedup already over the CPU implementations, though one that is inconsistent between test cases (between ~1.5x ad ~30x, above). Still, are current approach dramatically underutilizes the GPU, with Nsight Compute reporting 1.1% compute throuput, and 2.3% memory throughput. The throughput is currently limited by the relatively small number of clauses in the problems we're solving (100s). To fully utilize GPU resources, we will need to run many different decisions in parallel across different kernels. After implementing this, we will also evaluate the impact of kernel launch overhead, and determine whether it is worth attempting to reassign threads within running kernels instead. We also plan on examining the latency between the host and device during checks

### Schedule

We are roughly on track with our original schedule, in which we wanted to have completed a first iteration for single-core, multi-core, and GPU versions of DPLL. We still plan on spending the next few days digging into current performance bottlenecks, and experimenting with the future approaches outlined above. For our deliverables, we expect to have an analysis of speedup and scaling across the three approaches and clause size and literal count. Time permitting, we also plan on demoing a live sodoku solver. We do not plan on exploring an FPGA implementation anymore, which was one of our nice to haves, and we also no longer plan on exploring the unit-propagation early termination idea, since we've since learned that real-world SAT problems have far fewer clauses than can saturate a GPU grid.

Beyond the next steps metioned in the prelimary results, an important step we plan to take is updating our benchmarking to average across problems of a similar size. We learned in our initial benchmarking that the same problem across different approaches can vary wildly in termination time, so averaging across problems of a similar size will give us more robust results.

| Timeframe   | Tasks                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| 4/14 - 4/17 | - analyze bottlenecks in multi-core implementation (Ben) <br> - analyze bottlenecks in gpu implementation (Griffin) <br> - research how to better benchmark SAT solving (Ben, Griffin) |
| 4/18 - 4/20 | - improve multi-core implementation performance (Ben) <br> - improve GPU implementation performance (Griffin) <br> - improve testing framework (Ben, Griffin) <br> |
| 4/21 - 4/24 | - create sudoku demo (Ben, Griffin) <br> - analyze sudoku performance (Ben, Griffin) <br> |
| 4/25 - 4/27 | - gather performance data (Ben) <br> - gather instrumentation data (Griffin) <br> - write final report (Ben, Griffin) <br> - prepare for poster session (Griffin, Ben) |