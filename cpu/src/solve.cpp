#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <chrono>

#include "sat.h"
#include "pool.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char **argv) {
    bool parallel = false;
    int numThreads = 0;
    int argIndex = 1;
    
    // usage: solver [-P <numThreads>] <benchmark_file_path>
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [-P <numThreads>] <benchmark_file_path>" << endl;
        return 1;
    }
    if (string(argv[argIndex]) == "-P") {
        parallel = true;
        argIndex++;
        if (argc <= argIndex) {
            cerr << "Error: Number of threads missing after -P" << endl;
            return 1;
        }
        numThreads = std::atoi(argv[argIndex]);
        argIndex++;
    }
    if (argc <= argIndex) {
        cerr << "Error: Benchmark file path missing" << endl;
        return 1;
    }

    string file_path = argv[argIndex];
    ifstream infile(file_path);
    if (!infile) {
        cerr << "Error: Unable to open file " << file_path << endl;
        return 1;
    }

    // parse CNF file to generate formula
    Formula formula;
    string line;
    int numLiterals, numClauses;
    bool pLineFound = false;
    while (getline(infile, line)) {
        if (line.empty() || line[0] == 'c') {
            continue;
        }
        if (line[0] == 'p') {
            // p cnf <num literals> <num clauses>
            istringstream pIss(line);
            string p, cnf;
            pIss >> p >> cnf >> numLiterals >> numClauses;
            pLineFound = true;
            std::cerr
                << "[parser] CNF header: "
                << numLiterals << " literals, "
                << numClauses  << " clauses\n";
            continue;
        }
        if (pLineFound) {
            istringstream iss(line);
            int lit;
            vector<int> clause;
            while (iss >> lit) {
                if (lit == 0) {
                    break;
                }
                clause.push_back(lit);
            }
            if (!clause.empty()) {
                formula.push_back(clause);
            }
        }
    }
    infile.close();

    Assignment assignment(numLiterals + 1, litAssign::UNASSIGNED);

    bool result;
    duration<double> elapsed;
    
    if (parallel) {
        // create thread pool
        ThreadPool pool;
        if (parallel) {
            threadPoolInit(&pool, numThreads);
        }

        cout << "Running parallel algorithm..." << endl;

        auto start = steady_clock::now();
        result = dpll_parallel(formula, assignment, pool, 0);
        auto end = steady_clock::now();
        elapsed = end - start;

        // reap worker threads
        threadPoolDestroy(&pool);

        double unitMs   = totalUnitNs.load()   * 1e-6;
        double pureMs   = totalPureNs.load()   * 1e-6;
        double copyMs   = totalCopyNs.load()   * 1e-6;
        double submitMs = totalSubmitNs.load() * 1e-6;
        double spinMs   = totalSpinNs.load() * 1e-6;
        double lockMs   = totalLockNs.load() * 1e-6;
        double workMs   = totalWorkNs.load() * 1e-6;

        cout << "\n=== Aggregate timings ===\n"
                << "Total unit time:        " << unitMs   << " ms\n"
                << "Total pure time:        " << pureMs   << " ms\n"
                << "Total copy time:        " << copyMs   << " ms\n"
                << "Total task submit time: " << submitMs << " ms\n"
                << "Total task spin time:   " << spinMs << " ms\n"
                << "Total worker lock time: " << lockMs << " ms\n"
                << "Total local work time:  " << workMs << " ms\n";
    } else {
        cout << "Running sequential algorithm..." << endl;
        auto start = steady_clock::now();
        result = dpll(formula, assignment);
        auto end = steady_clock::now();
        elapsed = end - start;
    }

    cout << "The formula is " << (result ? "SATISFIABLE." : "UNSATISFIABLE.") << endl;
    if (result) {
        cout << "Satisfying assignment:" << endl;
        for (int i = 0; i < numLiterals; i++) {
            cout << "   Variable " << i << " = " 
                 << ((assignment[i] == litAssign::TRUE) ? "true" : 
                    (assignment[i]== litAssign::FALSE) ? "false" : 
                                                        "unassigned") << endl;
        }
    }

    bool expected;
    if (file_path.find("uuf") != string::npos) {
        expected = false;
    }
    else {
        expected = true;
    }
    if (result == expected) {
        cout << "TEST PASSED" << endl;
    } else {
        cout << "TEST FAILED" << endl;
    }
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}