#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <fstream>
#include <sstream>

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

enum AssignedValue {
    UNASSIGNED = 0,
    TRUE,
    FALSE
};

struct Assignment {
    AssignedValue* values;
};

__constant__ Formula devFormula;

__device__ AssignedValue flipTF(AssignedValue value) {
    return value == TRUE ? FALSE : value == FALSE ? TRUE : UNASSIGNED;
}

__device__ AssignedValue getAssig(const volatile Assignment* assignment, int literal) {
    if (literal > 0) {
        return assignment->values[literal - 1];
    } else {
        return flipTF(assignment->values[-literal - 1]);
    }
}

__device__ void setAssig(volatile Assignment* assignment, int literal, AssignedValue value) {
    if (literal > 0) {
        assignment->values[literal - 1] = value;
    } else {
        assignment->values[-literal - 1] = flipTF(value);
    }
}


// any propogated must be cleared to false
__global__ void propagateUnits(volatile Assignment* assignment, volatile bool* anyPropagated) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= devFormula.numClauses) return;

    // if (tid == 0) {
    //     // print formula for debugging
    //     printf("Formula: \n");
    //     printf("numLiterals: %d\n", devFormula.numLiterals);
    //     printf("numClauses: %d\n", devFormula.numClauses);
    //     for (int i = 0; i < devFormula.numClauses; i++) {
    //         printf("Clause %d: ", i);
    //         for (int j = 0; j < devFormula.clauses[i].count; j++) {
    //             printf("%d ", devFormula.clauses[i].literals[j]);
    //         }
    //         printf("\n");
    //     }
    // }

    Clause clause = devFormula.clauses[tid];
    int numUnassigned = 0;
    int lastUnassigned;
    for (int i = 0; i < clause.count; i++) {
        int literal = clause.literals[i];
        AssignedValue value = getAssig(assignment, literal);
        // printf("literal: %d  value: %d\n", literal, value);
        if (value == TRUE) {
            // clause is satisfied
            return;
        }
        if (value == UNASSIGNED) {
            numUnassigned++;
            lastUnassigned = clause.literals[i];
        }
        if (numUnassigned > 1) return;
    }

    if (numUnassigned != 1) return;
    setAssig(assignment, lastUnassigned, TRUE);
    // printf("lastUnassigned: %d  numUnassigned: %d\n", lastUnassigned, numUnassigned);
    *anyPropagated = true;
}

__global__ void checkClauses(const volatile Assignment* assignment, volatile bool* anyFalseClauses, volatile bool* allClausesSatisfied) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= devFormula.numClauses) return;

    Clause clause = devFormula.clauses[tid];
    bool clauseSatisfied = false;
    bool clauseFalse = true;
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
            break;
        }
    }

    if (clauseFalse) *anyFalseClauses = true;
    if (!clauseSatisfied) *allClausesSatisfied = false;
}

void copyFormulaToDevice(const Formula& formula) {
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
}

bool dpllHostDirected(const Formula& formula) {
    const auto startTime = std::chrono::high_resolution_clock::now();

    std::cout << "dpllHostDirected" << std::endl;
    std::cout << "numLiterals: " << formula.numLiterals << std::endl;
    std::cout << "numClauses: " << formula.numClauses << std::endl;

    copyFormulaToDevice(formula);

    Assignment devAssignmentView;
    CC(cudaMalloc(&devAssignmentView.values, sizeof(int) * formula.numLiterals));

    std::vector<AssignedValue> literalZeros(formula.numLiterals, UNASSIGNED);
    CC(cudaMemcpy(devAssignmentView.values, literalZeros.data(), sizeof(int) * formula.numLiterals, cudaMemcpyHostToDevice));

    Assignment* devAssignment;
    CC(cudaMalloc(&devAssignment, sizeof(Assignment)));
    CC(cudaMemcpy(devAssignment, &devAssignmentView, sizeof(Assignment), cudaMemcpyHostToDevice));

    bool* anyPropagated;
    bool* allClausesSatisfied;
    bool* anyFalseClauses;
    CC(cudaMalloc(&allClausesSatisfied, sizeof(bool)));
    CC(cudaMalloc(&anyFalseClauses, sizeof(bool)));
    CC(cudaMalloc(&anyPropagated, sizeof(bool)));

    std::function<bool()> inner = [=, &inner, &formula] () -> bool {
        bool f = false;
        bool t = true;

        constexpr int blockSize = 256;
        int numBlocks = (formula.numClauses - 1) / blockSize + 1;
        // std::cout << "numBlocks: " << numBlocks << std::endl;
        // std::cout << "blockSize: " << blockSize << std::endl;

        while (true) {
            CC(cudaMemcpy(anyPropagated, &f, sizeof(bool), cudaMemcpyHostToDevice));

            propagateUnits<<<numBlocks, blockSize>>>(devAssignment, anyPropagated);

            bool shouldContinue;
            CC(cudaMemcpy(&shouldContinue, anyPropagated, sizeof(bool), cudaMemcpyDeviceToHost));

            // std::cout << "shouldContinue: " << shouldContinue << std::endl;
            if (!shouldContinue) break;
        }

        // copy and print assignment
        std::vector<AssignedValue> retAssignment(formula.numLiterals);
        cudaMemcpy(retAssignment.data(), devAssignmentView.values, sizeof(int) * formula.numLiterals, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < formula.numLiterals; i++) {
        //     std::cout << retAssignment[i] << " ";
        // }

        // std::cout << "done propagating" << std::endl;

        CC(cudaMemcpy(allClausesSatisfied, &t, sizeof(bool), cudaMemcpyHostToDevice));
        CC(cudaMemcpy(anyFalseClauses, &f, sizeof(bool), cudaMemcpyHostToDevice));
        checkClauses<<<numBlocks, blockSize>>>(devAssignment, anyFalseClauses, allClausesSatisfied);

        bool allTrue;
        CC(cudaMemcpy(&allTrue, allClausesSatisfied, sizeof(bool), cudaMemcpyDeviceToHost));
        if (allTrue) {
            // std::cout << "found all clauses satisfied" << std::endl;
            return true;
        }

        bool anyFalse;
        CC(cudaMemcpy(&anyFalse, anyFalseClauses, sizeof(bool), cudaMemcpyDeviceToHost));
        if (anyFalse) {
            // std::cout << "found false clause" << std::endl;
            return false;
        }

        // find unassigned literal
        int unassignedLiteral = 0;
        for (int i = 0; i < formula.numLiterals; i++) {
            if (retAssignment[i] == UNASSIGNED) {
                unassignedLiteral = i + 1;
                break;
            }
        }

        assert(unassignedLiteral > 0);

        retAssignment[unassignedLiteral - 1] = TRUE;
        // std::cout << "found unassigned literal " << unassignedLiteral << std::endl;
        // std::cout << "assigning " << unassignedLiteral << " to TRUE" << std::endl;
        CC(cudaMemcpy(devAssignmentView.values, retAssignment.data(), sizeof(int) * formula.numLiterals, cudaMemcpyHostToDevice));
        if (inner()) return true;

        retAssignment[unassignedLiteral - 1] = FALSE;
        // std::cout << "assigning " << unassignedLiteral << " to FALSE" << std::endl;
        CC(cudaMemcpy(devAssignmentView.values, retAssignment.data(), sizeof(int) * formula.numLiterals, cudaMemcpyHostToDevice));
        return inner();
    };

    const auto initializationEndTime = std::chrono::high_resolution_clock::now();
    const auto initializationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(initializationEndTime - startTime);
    std::cout << "Initialization time: " << initializationDuration.count() << " ms" << std::endl;

   bool res = inner();

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

    bool sat = dpllHostDirected(formula);
    if (sat) {
        std::cout << "SAT" << std::endl;
    } else {
        std::cout << "UNSAT" << std::endl;
    }

    return 0;
}
