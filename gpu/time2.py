#!/usr/bin/python3

import subprocess
import pandas as pd
import argparse

def test(exec_path, tests_per_dir, runs_per_test):
    test_dirs = [
        "uf50-218",
        "uf75-325",
        "uf100-430",
        "uf125-538",
        "uf150-645",
        "uuf50-218",
        "uuf75-325",
        "uuf100-430",
        "uuf125-538",
        "uuf150-645",
    ]

    # create dataframe with columns ["test", "avg time"]
    df_detailed = pd.DataFrame(columns=["test", "avg time"])
    df_summary = pd.DataFrame(columns=["test", "avg time"])

    for test_dir in test_dirs:
        avg_times = []
        for i in range(1, tests_per_dir + 1):
            # run the command and get the output
            dir_prefix = test_dir.split("-")[0]
            command = [exec_path, f"../benchmarks/{test_dir}/{dir_prefix}-0{i}.cnf"]
            print(f"Running command: {command}")
            times = []
            for j in range(runs_per_test):
                result = subprocess.run(command, capture_output=True, text=True)
                # get the time from the output
                output = result.stdout
                # find "Total time: " in the output
                start_index = output.find("Total time: ")
                if start_index == -1:
                    print(f"Error: 'Total time: ' not found in output for {command}")
                    continue
                start_index += len("Total time: ")
                sec = output[start_index:output.find(" ", start_index)]
                # convert sec to float
                times.append(float(sec))

            # calculate the average time
            avg_time = sum(times) / len(times) / 1000.0
            avg_times.append(avg_time)
            # add the average time to the dataframe
            df_detailed = pd.concat([df_detailed, pd.DataFrame({"test": [f"{test_dir} {i}"], "avg time": [avg_time]})], ignore_index=True)
            print(f"Average time for {test_dir} {i}: {avg_time:.2f} seconds")

        avg_avg_time = sum(avg_times) / len(avg_times)

        # add the average of the averages to the summary dataframe
        df_summary = pd.concat([df_summary, pd.DataFrame({"test": [test_dir], "avg time": [avg_avg_time]})], ignore_index=True)
        print(f"Average time for {test_dir}: {avg_avg_time:.2f} seconds")

    # save the dataframe to a csv file
    name = "-".join(exec_path.split("/")[1:])
    df_detailed.to_csv(f"{name}_detailed.csv", index=False)
    df_summary.to_csv(f"{name}_summary.csv", index=False)


if __name__ == "__main__":
    # first argument is the path to the executable, second is tests_per_dir, third is runs_per_test
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Test the SAT solver.")
    parser.add_argument("exec_path", type=str, help="Path to the SAT solver executable")
    parser.add_argument("tests_per_dir", type=int, help="Number of tests per directory")
    parser.add_argument("runs_per_test", type=int, help="Number of runs per test")
    args = parser.parse_args()

    exec_path = args.exec_path
    tests_per_dir = args.tests_per_dir
    runs_per_test = args.runs_per_test

    # test the SAT solver
    test(exec_path, tests_per_dir, runs_per_test)
