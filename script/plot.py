import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


def read_csv_files(folder_path):
    file_data = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, newline="") as csvfile:
                csvreader = csv.reader(csvfile)

                # Skip the first line (header)
                next(csvreader)

                column_sums = None
                row_count = 0

                # Read the data
                for row in csvreader:
                    row_count += 1
                    if column_sums is None:
                        column_sums = [0] * len(row)

                    for i, value in enumerate(row):
                        column_sums[i] += float(value)

                if column_sums is not None:
                    file_info = filename.rstrip(".csv").split("_")
                    file_data.append((file_info, column_sums, row_count))

    return file_data


def calculate_averages(file_data):
    file_averages = []

    for filename, column_sums, row_count in file_data:
        if row_count > 0:
            averages = [sum_value / row_count for sum_value in column_sums]
            file_averages.append((filename, averages))

    return file_averages


def fill_arrays_with_averages(arrays, averages):
    for file_info, avg in averages:
        # Extract the combination key from file_info
        combination_key = tuple(
            [file_info[0], int(float(file_info[1])), int(float(file_info[2]))]
        )

        # import pdb
        # pdb.set_trace()
        # Check if the combination key is in the arrays dictionary
        if combination_key in arrays:
            if file_info[0] == "wastar":
                # For wastar, fill the entire array with the value of the first index
                arrays[combination_key]['cost'][:] = avg[1]
                arrays[combination_key]['length'][:] = avg[2]
                arrays[combination_key]['eval'][:] = avg[3]
            else:
                # Get the index for the arrays to fill in
                index = round(float(file_info[3]) / 0.05)
                arrays[combination_key]['cost'][index] = avg[1]
                arrays[combination_key]['length'][index] = avg[2]
                arrays[combination_key]['eval'][index] = avg[3]


# Specify the folder path
folder_path_di = "../logs/exp/dijkstra/"
folder_path_eu = "../logs/exp/euclidean/"

# Specify the folder path
folder_path = "/path/to/your/folder"

# Read the CSV files
file_data = read_csv_files(folder_path_di)
# file_data = read_csv_files(folder_path_eu)

# Calculate averages
averages = calculate_averages(file_data)

# Create three lists of possible planner setup
planner = ["epase", "qpase", "wastar"]
threads = [1, 10]
w = [1, 10]

# Generate all possible combinations
combinations = [
    combo
    for combo in itertools.product(planner, threads, w)
    if not (combo[0] == "wastar" and combo[1] == 10)
]

# Create three NumPy arrays (cost, length, eval) for each combination with a size of 11
arrays = {
    combo: {"cost": np.zeros(11), "length": np.zeros(11), "eval": np.zeros(11)}
    for combo in combinations
}

# Print the combinations and their corresponding arrays
# for combo, array_dict in arrays.items():
#     print(f"Combination: {combo}")
#     for array_name, array in array_dict.items():
#         print(f"{array_name.capitalize()} Array: {array}")
#     print("\n")
#
# import pdb
#
# pdb.set_trace()
#
# exit()

# # Display the averages for each file
# for file_info, avg in averages:
#     if file_info[0] == "epase":
#         print(f"Averages of columns for file {'_'.join(file_info)}:")
#         print(f"File Info: {file_info}")
#         print(f"Averages: {avg}")
#         print("\n")

# Fill arrays with averages
fill_arrays_with_averages(arrays, averages)
# Display the averages for each file
for combo, array_dict in arrays.items():
    print(f"Combination: {combo}")
    for array_name, array in array_dict.items():
        print(f"{array_name.capitalize()} Array: {array}")
    print("\n")

# X-axis: vector from 0 to 0.5 inclusive, incremented by 0.05
x = np.arange(0, 0.55, 0.05)


def plot_data_cost(arrays, w_value):
    plt.figure(figsize=(12, 8))

    for (planner, threads, w), array_dict in arrays.items():
        if w == w_value:
            if planner == "wastar":
                plt.plot(x, array_dict['cost'], label=f'{planner} (threads={threads}, w={w})', linestyle='--', color='black', linewidth=2.5)
            elif planner == "epase":
                color_ = 'blue'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['cost'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)
            else:
                color_ = 'red'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['cost'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)

    plt.xlabel("Heuristic Noise (incremented by 0.05)")
    plt.ylabel("Cost")
    plt.title(f"Cost Plot for w={w_value}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_length(arrays, w_value):
    plt.figure(figsize=(12, 8))

    for (planner, threads, w), array_dict in arrays.items():
        if w == w_value:
            if planner == "wastar":
                plt.plot(x, array_dict['length'], label=f'{planner} (threads={threads}, w={w})', linestyle='--', color='black', linewidth=2.5)
            elif planner == "epase":
                color_ = 'blue'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['length'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)
            else:
                color_ = 'red'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['length'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)

    plt.xlabel("Heuristic Noise (incremented by 0.05)")
    plt.ylabel("Length")
    plt.title(f"Length Plot for w={w_value}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_data_eval(arrays, w_value):
    plt.figure(figsize=(12, 8))

    for (planner, threads, w), array_dict in arrays.items():
        if w == w_value:
            if planner == "wastar":
                plt.plot(x, array_dict['eval'], label=f'{planner} (threads={threads}, w={w})', linestyle='--', color='black', linewidth=2.5)
            elif planner == "epase":
                color_ = 'blue'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['eval'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)
            else:
                color_ = 'red'
                if (threads==1):
                    linestyle_ = '-'
                else:
                    linestyle_ = '-.'
                plt.plot(x, array_dict['eval'], label=f'{planner} (threads={threads}, w={w})', linestyle=linestyle_, color=color_, linewidth=2.5)

    plt.xlabel("Heuristic Noise (incremented by 0.05)")
    plt.ylabel("Edge Evaluations")
    plt.title(f"Edge Evaluations Plot for w={w_value}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot data for w=1
plot_data_cost(arrays, 1)

# Plot data for w=10
plot_data_cost(arrays, 10)

# Plot data for w=1
plot_data_eval(arrays, 1)

# Plot data for w=10
plot_data_eval(arrays, 10)
