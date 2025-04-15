#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

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
        printf("literal: %d  value: %d\n", literal, value);
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
    printf("lastUnassigned: %d  numUnassigned: %d\n", lastUnassigned, numUnassigned);
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
        std::cout << "numBlocks: " << numBlocks << std::endl;
        std::cout << "blockSize: " << blockSize << std::endl;

        while (true) {
            CC(cudaMemcpy(anyPropagated, &f, sizeof(bool), cudaMemcpyHostToDevice));

            propagateUnits<<<numBlocks, blockSize>>>(devAssignment, anyPropagated);

            bool shouldContinue;
            CC(cudaMemcpy(&shouldContinue, anyPropagated, sizeof(bool), cudaMemcpyDeviceToHost));

            std::cout << "shouldContinue: " << shouldContinue << std::endl;
            if (!shouldContinue) break;
        }

        // copy and print assignment
        std::vector<AssignedValue> retAssignment(formula.numLiterals);
        cudaMemcpy(retAssignment.data(), devAssignmentView.values, sizeof(int) * formula.numLiterals, cudaMemcpyDeviceToHost);
        for (int i = 0; i < formula.numLiterals; i++) {
            std::cout << retAssignment[i] << " ";
        }

        std::cout << "done propagating" << std::endl;

        CC(cudaMemcpy(allClausesSatisfied, &t, sizeof(bool), cudaMemcpyHostToDevice));
        CC(cudaMemcpy(anyFalseClauses, &f, sizeof(bool), cudaMemcpyHostToDevice));
        checkClauses<<<numBlocks, blockSize>>>(devAssignment, anyFalseClauses, allClausesSatisfied);

        bool allTrue;
        CC(cudaMemcpy(&allTrue, allClausesSatisfied, sizeof(bool), cudaMemcpyDeviceToHost));
        if (allTrue) {
            std::cout << "found all clauses satisfied" << std::endl;
            return true;
        }

        bool anyFalse;
        CC(cudaMemcpy(&anyFalse, anyFalseClauses, sizeof(bool), cudaMemcpyDeviceToHost));
        if (anyFalse) {
            std::cout << "found false clause" << std::endl;
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
        std::cout << "found unassigned literal " << unassignedLiteral << std::endl;
        std::cout << "assigning " << unassignedLiteral << " to TRUE" << std::endl;
        CC(cudaMemcpy(devAssignmentView.values, retAssignment.data(), sizeof(int) * formula.numLiterals, cudaMemcpyHostToDevice));
        if (inner()) return true;

        retAssignment[unassignedLiteral - 1] = FALSE;
        std::cout << "assigning " << unassignedLiteral << " to FALSE" << std::endl;
        CC(cudaMemcpy(devAssignmentView.values, retAssignment.data(), sizeof(int) * formula.numLiterals, cudaMemcpyHostToDevice));
        return inner();
    };

    return inner();
}

int main() {
    Formula formula;
    formula.numLiterals = 10;
    formula.numClauses = 5;
    formula.clauses = new Clause[formula.numClauses];

    formula.clauses[0].count = 3;
    formula.clauses[0].literals = new int[3] {1, 2, 3};

    formula.clauses[1].count = 1;
    formula.clauses[1].literals = new int[1] {-1};

    formula.clauses[2].count = 1;
    formula.clauses[2].literals = new int[1] {-2};

    formula.clauses[3].count = 2;
    formula.clauses[3].literals = new int[2] {-3, -2};

    formula.clauses[4].count = 2;
    formula.clauses[4].literals = new int[2] {5, 7};

    dpllHostDirected(formula);

    return 0;
}