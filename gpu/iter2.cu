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

struct Clause {
    int count;
    int* literals;
};

struct Formula {
    int numLiterals;
    int numClauses;
    Clause* clauses;
};

struct CheckResult {
    bool allClausesSatisfied;
    bool anyClausesFalse;
    
    // guaranteed to be valid if the above are both false
    int unassignedLiteral; 
};

enum AssignedValue {
    UNASSIGNED = 0,
    TRUE,
    FALSE
};

constexpr int maxInstances = 64;
constexpr int maxAssignments = 1 << 20;

__constant__ Formula devFormula;
__constant__ AssignedValue* instanceAssignments[maxInstances];
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
__device__ void printFormula() {
    printf("Formula:\n");
    for (int i = 0; i < devFormula.numClauses; i++) {
        const Clause& clause = devFormula.clauses[i];
        printf("(");
        for (int j = 0; j < clause.count; j++) {
            int literal = clause.literals[j];
            AssignedValue value = getAssig(instanceAssignments[0], literal);
            if (value == UNASSIGNED) {
                printf("%d ", literal);
            } else if (value == TRUE) {
                printf("%d(T) ", literal);
            } else {
                printf("%d(F) ", literal);
            }
        }
        printf(")\n");
    }
    printf("\n");
}


// any propogated must be cleared to false
__global__ void propagateAndCheck() {
    int instance = blockIdx.x;
    int clauseIdx = threadIdx.x;
    if (clauseIdx >= devFormula.numClauses) return;

    const Clause& clause = devFormula.clauses[clauseIdx];
    volatile AssignedValue* assignment = instanceAssignments[instance];

    __shared__ bool anyPropagated;

    // propagate unit cluases
    bool clauseSatisfied = false;
    while (true) {
        if (clauseIdx == 0) anyPropagated = false;
        __syncthreads();

        // if (clauseIdx == 0) {
        //     printFormula();
        //     printf("|--------:\n");
        // }
        // __syncthreads();

        if (!clauseSatisfied) {
            int numUnassigned = 0;
            int lastUnassigned;
            for (int i = 0; i < clause.count; i++) {
                int literal = clause.literals[i];
                AssignedValue value = getAssig(assignment, literal);

                if (value == TRUE) {
                    clauseSatisfied = true;
                    break;
                }
                if (value == UNASSIGNED) {
                    numUnassigned++;
                    lastUnassigned = clause.literals[i];
                    if (numUnassigned > 1) break;
                }
            }

            if (!clauseSatisfied && numUnassigned == 1) {
                setAssig(assignment, lastUnassigned, TRUE);
                anyPropagated = true;
            }
        }
        
        __syncthreads();
        if (!anyPropagated) break;
    }

    // get clause result
    bool clauseFalse = true;
    bool foundUnassigned = false;
    int unassignedLiteral;
    for (int i = 0; i < clause.count; i++) {
        int literal = clause.literals[i];
        AssignedValue value = getAssig(assignment, literal);

        if (value == TRUE) {
            clauseSatisfied = true;
            clauseFalse = false;
            break;
        }

        if (value == UNASSIGNED) {
            clauseFalse = false;
            foundUnassigned = true;
            unassignedLiteral = abs(literal);
            break;
        }
    }

    // printf("Clause False: %i, Clause Satisfied: %i, Found Unassigned: %i, Unassigned Literal: %i\n",
    //        clauseFalse, clauseSatisfied, foundUnassigned, unassignedLiteral);

    if (clauseIdx == 0) {
        // first clause in block, initialize result
        checkResults[instance].allClausesSatisfied = clauseSatisfied;
        checkResults[instance].anyClausesFalse = clauseFalse;
        checkResults[instance].unassignedLiteral = 
            foundUnassigned ? unassignedLiteral : -1;
    }
    __syncthreads();

    if (clauseIdx == 0) return;

    // atomics not needed,
    // incoherence between threads doesn't affect correctness here
    CheckResult& result = checkResults[instance];
    if (clauseFalse) result.anyClausesFalse = true;
    if (!clauseSatisfied) result.allClausesSatisfied = false;
    if (foundUnassigned) result.unassignedLiteral = unassignedLiteral;
    // (creates a ton of unnecessary traffic, anything we can do about this?)
}

void setupGlobals(const Formula& formula) {
    hostFormula = formula;

    // copy formula to device
    Clause* clauses;
    CC(cudaMalloc(&clauses, sizeof(Clause) * formula.numClauses));
    for (int i = 0; i < formula.numClauses; i++) {
        CC(cudaMemcpy(&clauses[i].count, &formula.clauses[i].count, sizeof(int), cudaMemcpyHostToDevice));

        int* literals;
        CC(cudaMalloc(&literals, sizeof(int) * formula.clauses[i].count));
        CC(cudaMemcpy(literals, formula.clauses[i].literals, sizeof(int) * formula.clauses[i].count, cudaMemcpyHostToDevice));
        CC(cudaMemcpy(&clauses[i].literals, &literals, sizeof(int*), cudaMemcpyHostToDevice));
    }

    Formula copy = {
        .numLiterals = formula.numLiterals,
        .numClauses = formula.numClauses,
        .clauses = clauses
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
    //     const Clause& clause = hostFormula.clauses[i];
    //     std::cout << "Clause " << i << ": ";
    //     for (int j = 0; j < clause.count; j++) {
    //         std::cout << clause.literals[j] << " ";
    //     }
    //     std::cout << std::endl;
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
        propagateAndCheck<<<activeInstances, numClausesCeil32>>>();

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
    std::vector<Clause> clauses;
    std::string line;
    bool pLineFound = false;
    int numLiterals;
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
            std::vector<int> clause;
            while (iss >> lit) {
                if (lit == 0) {
                    break;
                }
                clause.push_back(lit);
            }
            if (!clause.empty()) {
                Clause c;
                c.count = clause.size();
                c.literals = new int[c.count];
                for (int i = 0; i < c.count; i++) {
                    c.literals[i] = clause[i];
                }
                clauses.push_back(c);
            }
        }
    }
    infile.close();

    Formula formula;
    formula.numLiterals = numLiterals;
    formula.numClauses = clauses.size();
    formula.clauses = new Clause[formula.numClauses];
    for (size_t i = 0; i < clauses.size(); i++) {
        formula.clauses[i].count = clauses[i].count;
        formula.clauses[i].literals = clauses[i].literals;
    }

    bool sat = dpll(formula);
    if (sat) {
        std::cout << "SAT" << std::endl;
    } else {
        std::cout << "UNSAT" << std::endl;
    }

    return 0;
}