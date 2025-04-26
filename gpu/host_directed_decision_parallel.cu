#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <fstream>
#include <sstream>
#include <queue>
#include <thread>
#include <stack>
#include <mutex>

#define CC(x) { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct Formula {
    int numLiterals;
    int numClauses;
    int totalLiterals;
    int* clauseCounts;
    int* clauseStarts;
    int* clauseLiterals;
};

struct CheckResult {
    int allClausesSatisfied;
    int anyClausesFalse;
    
    // guaranteed to be valid if the above are both false
    int unassignedLiteral; 
};

enum AssignedValue : int {
    UNASSIGNED = 0,
    TRUE,
    FALSE
};

constexpr int maxInstances = 512;
constexpr int maxAssignments = 1 << 20;

__constant__ Formula devFormula;
__device__ AssignedValue* instanceAssignments[maxInstances];
__device__ AssignedValue assignmentArena[maxAssignments];
__device__ CheckResult checkResults[maxInstances];

static Formula hostFormula;
static int assignmentArenaHead = 0;
static std::vector<AssignedValue*> assignmentRecycling;
static std::vector<AssignedValue*> assignmentStack;
static const AssignedValue lTrue = TRUE;
static const AssignedValue lFalse = FALSE;
static AssignedValue* assignmentArenaPtr;

__device__ AssignedValue flipTF(AssignedValue value) {
    return value == TRUE ? FALSE : value == FALSE ? TRUE : UNASSIGNED;
}

__device__ AssignedValue getAssig(const volatile AssignedValue* assignment, int literal) {
    if (literal > 0) {
        return assignment[literal - 1];
    } else {
        return flipTF(assignment[-literal - 1]);
    }
}

__device__ void setAssig(volatile AssignedValue* assignment, int literal, AssignedValue value) {
    if (literal > 0) {
        assignment[literal - 1] = value;
    } else {
        assignment[-literal - 1] = flipTF(value);
    }
}

// print formula on one line in () ^ () format, with assignment underneath
__device__ void printFormula(const volatile AssignedValue* assignment) {
    printf("Formula:\n");
    for (int i = 0; i < devFormula.numClauses; i++) {
        const int clauseStart = devFormula.clauseStarts[i];
        const int clauseCount = devFormula.clauseCounts[i];
        const int* clause = devFormula.clauseLiterals + clauseStart;
        printf("(");
        for (int j = 0; j < clauseCount; j++) {
            int literal = clause[j];
            AssignedValue value = getAssig(assignment, literal);
            if (j > 0) printf(" v ");
            if (value == UNASSIGNED) {
                printf("%i ", literal);
            } else if (value == TRUE) {
                printf("%i(T) ", literal);
            } else {
                printf("%i(F) ", literal);
            }
        }
        printf(")\n");
    }
    printf("\n");
}


__global__ void propagateAndCheck() {
    volatile __shared__ int anyPropagated;
    volatile __shared__ CheckResult result;

    extern volatile __shared__ int sharedMem[];
    volatile int* clauseLiterals = sharedMem;
    volatile AssignedValue* assignment = 
        (AssignedValue*)(clauseLiterals + devFormula.totalLiterals);

    const int numClauses = devFormula.numClauses;

    int instance = blockIdx.x;
    int clauseIdx = threadIdx.x;
    if (clauseIdx >= numClauses) return;

    const int totalLiterals = devFormula.totalLiterals;
    const int numLiterals = devFormula.numLiterals;
    const int* clauseStarts = devFormula.clauseStarts;
    const int* clauseCounts = devFormula.clauseCounts;
    const int* devClauseLiterals = devFormula.clauseLiterals;
    AssignedValue* devAssignment = instanceAssignments[instance];

    // setup

    anyPropagated = 0;

    for (int i = clauseIdx; i < totalLiterals; i += numClauses) {
        clauseLiterals[i] = devClauseLiterals[i];
    }

    for (int i = clauseIdx; i < numLiterals; i += numClauses) {
        assignment[i] = devAssignment[i];
    }

    const int clauseStart = clauseStarts[clauseIdx];
    const int clauseCount = clauseCounts[clauseIdx];

    // we can remove the volatile qualifier because we will no longer write here
    const int* clause = const_cast<const int*>(clauseLiterals + clauseStart);

    __syncthreads();

    // propagate unit cluases
    int clauseSatisfied = 0;
    while (true) {
        if (!clauseSatisfied) {
            int numUnassigned = 0;
            int lastUnassigned;
            for (int i = 0; i < clauseCount; i++) {
                int literal = clause[i];
                AssignedValue value = getAssig(assignment, literal);

                if (value == TRUE) {
                    clauseSatisfied = 1;
                    break;
                }
                if (value == UNASSIGNED) {
                    numUnassigned++;
                    lastUnassigned = clause[i];
                    if (numUnassigned > 1) break;
                }
            }

            if (!clauseSatisfied && numUnassigned == 1) {
                setAssig(assignment, lastUnassigned, TRUE);
                anyPropagated = 1;
            }
        }
        
        __syncthreads();

        // printf("anyPropagated End: %i\n", anyPropagated);
        if (!anyPropagated) break;
        if (clauseIdx == 0) anyPropagated = 0;

        __syncthreads();
    }

    // get clause result
    int clauseFalse = 1;
    int foundUnassigned = 0;
    int unassignedLiteral = -2;
    for (int i = 0; i < clauseCount; i++) {
        int literal = clause[i];
        AssignedValue value = getAssig(assignment, literal);

        if (value == TRUE) {
            clauseSatisfied = 1;
            clauseFalse = 0;
            break;
        }

        if (value == UNASSIGNED) {
            clauseFalse = 0;
            foundUnassigned = 1;
            unassignedLiteral = abs(literal);
            break;
        }
    }

    // printf("Clause False: %i, Clause Satisfied: %i, Found Unassigned: %i, Unassigned Literal: %i Clause Start: %i Clause Count: %i\n",
    //        clauseFalse, clauseSatisfied, foundUnassigned, devClauseLiterals[clauseStart], clauseStart, clauseCount);
    
    if (clauseIdx == 0) {
        // first clause in block, initialize result
        result.allClausesSatisfied = clauseSatisfied;
        result.anyClausesFalse = clauseFalse;
        result.unassignedLiteral = 
            foundUnassigned ? unassignedLiteral : -1;
    }

    __syncthreads();

    if (clauseIdx > 0) {
        // atomics not needed,
        // incoherence between threads doesn't affect correctness here
        // if (clauseFalse) atomicCAS((int*)&result.anyClausesFalse, 0, 1);
        // if (!clauseSatisfied) atomicCAS((int*)&result.allClausesSatisfied, 1, 0);
        // if (foundUnassigned) atomicCAS((int*)&result.unassignedLiteral, -1, unassignedLiteral);
        if (clauseFalse) result.anyClausesFalse = 1;
        if (!clauseSatisfied) result.allClausesSatisfied = 0;
        if (foundUnassigned) result.unassignedLiteral = unassignedLiteral;
    }

    __syncthreads();

    // if (instance == 0) {
        // if (result.allClausesSatisfied) {
        //     printf("Instance: %i Clause %i: allClausesSatisfied: %i, anyClausesFalse: %i, unassignedLiteral: %i\n",
        //         instance, clauseIdx, result.allClausesSatisfied, result.anyClausesFalse, result.unassignedLiteral);
        // }
    // }
    
    if (clauseIdx == 0) {
        // complier is unhappy if we copy the whole struct
        checkResults[instance].allClausesSatisfied = result.allClausesSatisfied;
        checkResults[instance].anyClausesFalse = result.anyClausesFalse;
        checkResults[instance].unassignedLiteral = result.unassignedLiteral;
    }

    // copy assignment back to global memory
    for (int i = clauseIdx; i < numLiterals; i += numClauses) {
        devAssignment[i] = assignment[i];
    }
}

void setupGlobals(const Formula& formula) {
    hostFormula = formula;

    // copy formula to device
    int* clauseStarts;
    int* clauseCounts;
    int* clauseLiterals;
    CC(cudaMalloc(&clauseStarts, sizeof(int) * formula.numClauses));
    CC(cudaMalloc(&clauseCounts, sizeof(int) * formula.numClauses));
    CC(cudaMalloc(&clauseLiterals, sizeof(int) * formula.totalLiterals));

    CC(cudaMemcpy(clauseStarts, formula.clauseStarts, 
                  sizeof(int) * formula.numClauses, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(clauseCounts, formula.clauseCounts,
                  sizeof(int) * formula.numClauses, cudaMemcpyHostToDevice));
    CC(cudaMemcpy(clauseLiterals, formula.clauseLiterals,
                  sizeof(int) * formula.totalLiterals, cudaMemcpyHostToDevice));

    Formula copy = {
        .numLiterals = formula.numLiterals,
        .numClauses = formula.numClauses,
        .totalLiterals = formula.totalLiterals,
        .clauseCounts = clauseCounts,
        .clauseStarts = clauseStarts,
        .clauseLiterals = clauseLiterals
    };
    CC(cudaMemcpyToSymbol(devFormula, &copy, sizeof(Formula)));

    // get arena pointer
    CC(cudaGetSymbolAddress((void**)&assignmentArenaPtr, assignmentArena));
}

AssignedValue* allocAssignment() {
    if (assignmentRecycling.empty()) {
        if (assignmentArenaHead > maxAssignments) {
            std::cerr << "Too many assignments allocated" << std::endl;
            exit(EXIT_FAILURE);
        }

        AssignedValue* devAssignment = &assignmentArenaPtr[assignmentArenaHead];
        assignmentArenaHead += hostFormula.numLiterals;

        return devAssignment;
    } else {
        AssignedValue* assig = assignmentRecycling.back();
        assignmentRecycling.pop_back();
        return assig;
    }
}

void dpllSetup(const Formula& formula) {
    setupGlobals(formula);

    std::vector<AssignedValue> blankAssignment(formula.numLiterals, UNASSIGNED);
    AssignedValue* firstAssignment = allocAssignment();
    CC(cudaMemcpy(
        firstAssignment, blankAssignment.data(), 
        sizeof(AssignedValue) * formula.numLiterals,
        cudaMemcpyHostToDevice));
    assignmentStack.push_back(firstAssignment);
}

bool dpllMain() {
    std::vector<AssignedValue*> hostInstanceAssignments;
    hostInstanceAssignments.reserve(maxInstances);

    CheckResult results[maxInstances];

    // print formula
    // std::cout << "Formula:" << std::endl;
    // for (int i = 0; i < hostFormula.numClauses; i++) {
    //     const int clauseStart = hostFormula.clauseStarts[i];
    //     const int clauseCount = hostFormula.clauseCounts[i];
    //     const int* clause = hostFormula.clauseLiterals + clauseStart;
    //     std::cout << "(";
    //     for (int j = 0; j < clauseCount; j++) {
    //         int literal = clause[j];
    //         std::cout << literal << " ";
    //     }
    //     std::cout << ")" << std::endl;
    // }
    // std::cout << std::endl;

    while (!assignmentStack.empty()) {
        // std::cout << "----------------" << std::endl;
        // std::cout << "Stack size: " << assignmentStack.size() << std::endl;
        // std::cout << "Recycling size: " << assignmentRecycling.size() << std::endl;
        // std::cout << "Arena head: " << assignmentArenaHead << std::endl;
        // std::cout << std::endl;

        // std::cout << "Grabbing assignments" << std::endl;

        // pop and copy instance assignments
        hostInstanceAssignments.clear();
        for (int i = 0; i < maxInstances && !assignmentStack.empty(); i++) {
            AssignedValue* assignment = assignmentStack.back();
            assignmentStack.pop_back();
            hostInstanceAssignments.push_back(assignment);
        }
        int activeInstances = hostInstanceAssignments.size();
        // std::cout << "Stack size: " << assignmentStack.size() << std::endl;

        // std::cout << "Active instances: " << activeInstances << std::endl;

        CC(cudaMemcpyToSymbol(
            instanceAssignments, hostInstanceAssignments.data(), 
            activeInstances * sizeof(AssignedValue*)));

        // std::cout << "Propagating and checking" << std::endl;
        int numClausesCeil32 = ((hostFormula.numClauses - 1) / 32 + 1) * 32;
        int sharedMemSize = sizeof(int) * hostFormula.totalLiterals +
            sizeof(AssignedValue) * hostFormula.numLiterals;
        // std::cout << "Shared memory alloc: " << sharedMemSize << std::endl;
        propagateAndCheck<<<activeInstances, numClausesCeil32, sharedMemSize>>>();

        CC(cudaMemcpyFromSymbol(
            results, checkResults, 
            sizeof(CheckResult) * activeInstances));

        // std::cout << "Results:" << std::endl;
        // for (int i = 0; i < activeInstances; i++) {
        //     const CheckResult& result = results[i];
        //     std::cout << "Instance " << i << ": "
        //               << "allClausesSatisfied: " << result.allClausesSatisfied
        //               << ", anyClausesFalse: " << result.anyClausesFalse
        //               << ", unassignedLiteral: " << result.unassignedLiteral
        //               << std::endl;
        // }
        // std::cout << std::endl;

        // handle results
        for (int i = 0; i < activeInstances; i++) {
            const CheckResult& result = results[i];

            if (result.allClausesSatisfied) {
                return true;
            }

            if (result.anyClausesFalse) {
                // dead end; put assignment in recycling
                assignmentRecycling.push_back(hostInstanceAssignments[i]);
            }

            // handle decisions afterward to utilize recycling
        }

        for (int i = 0; i < activeInstances; i++) {
            const CheckResult& result = results[i];

            if (!result.anyClausesFalse) {
                AssignedValue* a1 = hostInstanceAssignments[i];
                AssignedValue* a2 = allocAssignment();

                // copy a1 to a2
                CC(cudaMemcpy(
                    a2, a1, 
                    sizeof(AssignedValue) * hostFormula.numLiterals,
                    cudaMemcpyDeviceToDevice));

                // set unassignedLiteral to TRUE on device in a1
                CC(cudaMemcpy(
                    &a1[result.unassignedLiteral - 1], &lTrue,
                    sizeof(AssignedValue), cudaMemcpyHostToDevice));

                // set unassignedLiteral to FALSE on device in a2
                CC(cudaMemcpy(
                    &a2[result.unassignedLiteral - 1], &lFalse,
                    sizeof(AssignedValue), cudaMemcpyHostToDevice));

                // push a1 and a2 to stack
                assignmentStack.push_back(a1);
                assignmentStack.push_back(a2);
            }
        }
    }

    return false;
}

bool dpll(const Formula& formula) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    dpllSetup(formula);

    const auto initializationEndTime = std::chrono::high_resolution_clock::now();
    const auto initializationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initializationEndTime - startTime);
    std::cout << "Initialization time: " << initializationDuration.count() << " ms" << std::endl;

    bool res = dpllMain();

    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - initializationEndTime);
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;

    return res;
}

int main(int argc, char** argv) {
    // usage: solver [-P] <benchmark_file_path>
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <benchmark_file_path>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];
    std::ifstream infile(file_path);
    if (!infile) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return 1;
    }

    // parse CNF file to generate formula
    std::vector<int> clauseCounts;
    std::vector<int> clauseStarts;
    std::vector<int> clauseLiterals;

    std::string line;
    bool pLineFound = false;
    int numLiterals;
    int totalLiterals = 0;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == 'c') {
            continue;
        }
        if (line[0] == 'p') {
            pLineFound = true;
            // parse the problem line
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> numLiterals;
            continue;
        }
        if (pLineFound) {
            std::istringstream iss(line);
            int lit;
            clauseStarts.push_back(clauseLiterals.size());
            while (iss >> lit) {
                if (lit == 0) {
                    break;
                }
                clauseLiterals.push_back(lit);
            }
            if (clauseStarts.back() == clauseLiterals.size()) {
                // clause is empty
                clauseStarts.pop_back();
            } else {
                int count = clauseLiterals.size() - clauseStarts.back();
                clauseCounts.push_back(count);
                totalLiterals += count;
            }
        }
    }
    infile.close();

    assert(clauseCounts.size() == clauseStarts.size());

    Formula formula;
    formula.numLiterals = numLiterals;
    formula.numClauses = clauseCounts.size();
    formula.totalLiterals = totalLiterals;
    formula.clauseCounts = clauseCounts.data();
    formula.clauseStarts = clauseStarts.data();
    formula.clauseLiterals = clauseLiterals.data();

    bool sat = dpll(formula);
    if (sat) {
        std::cout << "SAT" << std::endl;
    } else {
        std::cout << "UNSAT" << std::endl;
    }

    return 0;
}
