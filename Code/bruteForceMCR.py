#!/usr/bin/env python3

from itertools import permutations
from helper_funcs_null import read_cities, score
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def brute_force(nodes, time_limit):
    """
    Brute force TSP solution with a time limit.
    """
    best_path = None
    shortest_distance = float("inf")
    indicies = np.arange(1, len(nodes))  # Force starting at node 0
    start_time = time.time()

    for p in permutations(indicies):
        if time.time() - start_time > time_limit:
            return None, None, True  # Timeout, return without a solution

        p = [0] + list(p)  # Start at node 0
        dist = score(nodes, p)
        if dist < shortest_distance:
            shortest_distance = dist
            best_path = p

    return best_path, shortest_distance, False  # No timeout

def graph(cities, best_path, shortest_distance, data, algo, start=0):
    order = np.array(best_path)  # These are the indices of the cities

    # Reorder points based on the specified order
    ordered_points = cities[order]

    # Add the starting city to the end to complete the circuit
    ordered_points = np.vstack([ordered_points, ordered_points[start]])

    x = ordered_points[:, 0]
    y = ordered_points[:, 1]

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the path with markers and arrows for clarity
    plt.plot(x, y, marker='o', markersize=8, color='blue', linestyle='-', label=f'Total Distance: {round(shortest_distance, 2)}')

    # Add direction arrows
    for i in range(len(x) - 1):
        plt.annotate('', xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Add city labels
    for i, (x_coord, y_coord) in enumerate(zip(x, y)):
        plt.text(x_coord, y_coord, f'{i}', fontsize=12, ha='right')

    # Add labels and title
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'{algo}: {data} Path')
    plt.legend()

    # Save the plot to a file
    name, _ = data.split('.')
    plt.savefig(f'{algo}_{name}_matplotlib.png')

    # Show the plot for debugging purposes
    #plt.show()
def test_and_time(dataset_paths, time_limit=15000):  # time_limit in milliseconds
    times = []
    vertices = []
    timeouts = []
    estimated_times = []

    for data in dataset_paths:
        cities = read_cities(f'../data/{data}')
        num_vertices = len(cities)

        # Time the brute force TSP execution
        start_time = time.time() * 1000  # Start time in milliseconds
        best_path, shortest_distance, timeout = brute_force(cities, time_limit/1000)
        elapsed_time = (time.time() * 1000) - start_time  # Elapsed time in milliseconds

        # Estimate the time for timeout cases
        if timeout:
            num_permutations = math.factorial(num_vertices - 1)
            estimated_time = time_limit / (elapsed_time / num_permutations)
            estimated_times.append(estimated_time)  # Keep in milliseconds
        else:
            estimated_times.append(elapsed_time)  # Keep in milliseconds
            graph(cities, best_path, shortest_distance, data, 'brute_force')

        # Store the results
        times.append(elapsed_time)  # Keep in milliseconds
        vertices.append(num_vertices)
        timeouts.append(timeout)

        if timeout:
            print(f'Dataset: {data}, Timeout after {time_limit / 1000 :.2f} seconds, Vertices: {num_vertices}')
        else:
            print(f'Dataset: {data}, Time: {elapsed_time:.2f} milliseconds, Vertices: {num_vertices}')

    return vertices, times, timeouts, estimated_times

def plot_time_vs_vertices(vertices, times, timeouts, estimated_times, time_limit=15000):
    """
    Plots the number of vertices against the computation time in milliseconds.
    Indicate timeouts on the graph.
    """
    plt.figure(figsize=(10, 6))

    estimated_time_label_used = False
    completed_label_used = False
    total_time = []

    for i in range(len(vertices)):
        if timeouts[i]:
            if not estimated_time_label_used:
                plt.scatter(vertices[i], estimated_times[i], color='red', marker='x', s=100, label='Estimated Time')
                estimated_time_label_used = True
            else:
                plt.scatter(vertices[i], estimated_times[i], color='red', marker='x', s=100)
            total_time.append(estimated_times[i])
        else:
            if not completed_label_used:
                plt.scatter(vertices[i], times[i], color='green', label='Completed')
                completed_label_used = True
            else:
                plt.scatter(vertices[i], times[i], color='green')

            total_time.append(times[i])

    plt.plot(vertices, total_time, linestyle='-', color='blue', label='Computation Time')

    # Add horizontal line for timeout (in milliseconds)
    plt.axhline(y=time_limit, color='red', linestyle='dotted', label=f'Timeout Limit ({time_limit:.2f} milliseconds)')

    plt.xlabel('Number of Vertices')
    plt.ylabel('Computation Time (milliseconds)')
    plt.title(f'TSP Brute Force: Time vs Number of Vertices (Timeout={time_limit :.2f} ms)')
    plt.grid(True, which='both')  # Grid for both major and minor ticks
    plt.legend()
    plt.xticks(vertices)
 #   plt.ylim(None, 7000)
    plt.yscale('log')
    # Save and show the plot
 #   plt.tight_layout()
    plt.savefig('brute_force_plot.png')
    plt.show()

if __name__ == '__main__':
    # List of datasets to test
    dataset_paths = ['3_tiny_null.csv','5_tiny_null.csv','7_tiny_null.csv','10_tiny_null.csv','12_tiny_null.csv']


    # Run the tests and time them
    vertices, times, timeouts, estimated_times = test_and_time(dataset_paths)

    # Plot the results with timeouts
    plot_time_vs_vertices(vertices, times, timeouts, estimated_times)
