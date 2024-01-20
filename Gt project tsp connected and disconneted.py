#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import networkx as nx
import time

class TSPSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver")

        # Adjusting the style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="blue", foreground="red")  # Set blue background with white text

        self.root.configure(bg='black')  # Set the background color of the main window to black
        self.time_label = ttk.Label(root, text="Execution Time:", foreground='white', background='black')
        self.time_label.pack(pady=10)

        self.time_text = tk.Text(root, height=1, width=40, state="disabled")
        self.time_text.pack(pady=10)
        self.city_input_label = ttk.Label(root, text="Enter city graph (comma-separated):", foreground='white', background='black')
        self.city_input_label.pack(pady=10)

        self.city_input_entry = ttk.Entry(root, width=40)
        self.city_input_entry.pack(pady=10)
        self.upload_button = ttk.Button(root, text="Upload File", command=self.upload_file)
        self.upload_button.pack(side=tk.LEFT, padx=10)
        self.upload_button.configure(style="TButton")
        
        # Adding colors to buttons
        self.solve_button_exact = ttk.Button(root, text="Exact TSP (Connected Cities)", command=self.solve_exact_tsp_simulation)
        self.solve_button_exact.pack(side=tk.LEFT, padx=10)
        self.solve_button_exact.configure(style="TButton")  # Apply the configured style

        self.solve_button_disconnected_heuristic = ttk.Button(root, text="Heuristic TSP (Disconnected Cities)", command=self.solve_disconnected_heuristic_tsp_simulation)
        self.solve_button_disconnected_heuristic.pack(side=tk.LEFT, padx=10)
        self.solve_button_disconnected_heuristic.configure(style="TButton")  # Apply the configured style

        self.solve_button_disconnected_mst = ttk.Button(root, text="MST TSP (Disconnected Cities)", command=self.solve_disconnected_mst_tsp_simulation)
        self.solve_button_disconnected_mst.pack(side=tk.LEFT, padx=10)
        self.solve_button_disconnected_mst.configure(style="TButton")  # Apply the configured style

        self.reset_button = ttk.Button(root, text="Reset", command=self.reset_graph)
        self.reset_button.pack(side=tk.LEFT, padx=10)
        self.reset_button.configure(style="TButton")  # Apply the configured style

        self.next_step_button = ttk.Button(root, text="Heuristic Next Step", command=self.next_simulation_step, state=tk.DISABLED)
        self.next_step_button.pack(side=tk.LEFT, padx=10)
        self.next_step_button.configure(style="TButton")  # Apply the configured style

        self.result_label = ttk.Label(root, text="Results:", foreground='white', background='black')
        self.result_label.pack(pady=10)

        self.result_text = tk.Text(root, height=5, width=40, state="disabled")
        self.result_text.pack(pady=10)

        self.plot_canvas = FigureCanvasTkAgg(Figure(figsize=(8, 6)), master=root)
        self.plot_canvas.get_tk_widget().pack(pady=10)
        self.plot_ax = self.plot_canvas.figure.add_subplot(111)

        self.city_graph = None
        self.heuristic_simulation_step = 0
        self.heuristic_simulation_path = []

        self.mst_simulation_step = 0
        self.mst_simulation_path = []

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.city_input_entry.delete(0, tk.END)
                self.city_input_entry.insert(0, content)

    def solve_exact_tsp_simulation(self):
        start_time = time.time()
        city_input = self.city_input_entry.get()
        self.city_graph = [int(x) for x in city_input.split(',')]
        solution = self.solve_exact_tsp(self.city_graph)
        execution_time = time.time() - start_time
        self.plot_city_graph(self.city_graph, solution["Path"])
        self.display_time(execution_time)
        self.display_results(solution, execution_time)

    def solve_disconnected_heuristic_tsp_simulation(self):
        start_time = time.time()
        city_input = self.city_input_entry.get()
        self.city_graph = [int(x) for x in city_input.split(',')]
        self.heuristic_simulation_step = 0
        self.heuristic_simulation_path = []
        self.next_step_button.configure(state=tk.NORMAL)
        self.solve_disconnected_heuristic_tsp()
        execution_time = time.time() - start_time
        self.display_time(execution_time)
        self.display_results({"Path": self.heuristic_simulation_path[-1], "Total Distance": self.calculate_total_distance(self.city_graph, self.heuristic_simulation_path[-1])}, execution_time)

    def solve_disconnected_mst_tsp_simulation(self):
        start_time = time.time()
        city_input = self.city_input_entry.get()
        self.city_graph = [int(x) for x in city_input.split(',')]
        self.mst_simulation_step = 0
        self.mst_simulation_path = []
        self.next_step_button.configure(state=tk.NORMAL)
        self.solve_disconnected_mst_tsp()
        execution_time = time.time() - start_time
        self.display_time(execution_time)
        self.display_results({"Path": self.mst_simulation_path[-1], "Total Distance": self.calculate_total_distance(self.city_graph, self.mst_simulation_path[-1])}, execution_time)

    def reset_graph(self):
        self.city_input_entry.delete(0, 'end')
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, "end")
        self.result_text.config(state="disabled")

        self.next_step_button.configure(state=tk.DISABLED)

        self.plot_ax.clear()
        self.plot_canvas.draw()
        self.reset_time()

    def display_results(self, solution, execution_time):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, "end")
        self.result_text.insert("end", f"Path: {solution['Path']}\nTotal Distance: {solution['Total Distance']:.6f}")
        self.result_text.config(state="disabled")
        self.display_time(execution_time)

    def reset_time(self):
        self.time_text.config(state="normal")
        self.time_text.delete(1.0, "end")
        self.time_text.config(state="disabled")

    def display_time(self, execution_time):
        self.time_text.config(state="normal")
        self.time_text.delete(1.0, "end")
        self.time_text.insert("end", f"{execution_time:.6f} seconds")
        self.time_text.config(state="disabled")

    def solve_exact_tsp(self, city_graph):
        n = len(city_graph) // 2
        graph = self.create_graph(city_graph)
        tour = list(nx.approximation.traveling_salesman_problem(graph, cycle=True))
        path = [node for node in tour]

        return {"Path": path, "Total Distance": self.calculate_total_distance(city_graph, path)}

    def solve_disconnected_heuristic_tsp(self):
        n = len(self.city_graph) // 2
        visited = [False] * n
        path = [0]  # Start from the first city

        for _ in range(n - 1):
            current_city = path[-1]
            nearest_city = self.find_nearest_neighbor(current_city, visited)
            path.append(nearest_city)
            visited[nearest_city] = True
            self.heuristic_simulation_path.append(path.copy())

        self.next_simulation_step()

    def solve_disconnected_mst_tsp(self):
        n = len(self.city_graph) // 2
        graph = self.create_graph(self.city_graph)
        mst = nx.minimum_spanning_tree(graph)
        path = list(nx.dfs_preorder_nodes(mst, source=0))

        for i in range(n - 1):
            self.mst_simulation_path.append(path[:i + 2])
            self.next_simulation_step()

    def next_simulation_step(self):
        self.heuristic_simulation_step += 1
        self.mst_simulation_step += 1

        if self.heuristic_simulation_step <= len(self.heuristic_simulation_path):
            self.plot_city_graph(self.city_graph, self.heuristic_simulation_path[self.heuristic_simulation_step - 1])
        elif self.mst_simulation_step <= len(self.mst_simulation_path):
            self.plot_city_graph(self.city_graph, self.mst_simulation_path[self.mst_simulation_step - 1])
        else:
            self.next_step_button.configure(state=tk.DISABLED)

    def find_nearest_neighbor(self, current_city, visited):
        x, y = self.city_graph[current_city * 2], self.city_graph[current_city * 2 + 1]
        n = len(self.city_graph) // 2
        min_distance = float('inf')
        nearest_city = -1

        for i in range(n):
            if not visited[i]:
                distance = np.sqrt((x - self.city_graph[i * 2])**2 + (y - self.city_graph[i * 2 + 1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_city = i

        return nearest_city

    def create_graph(self, city_graph):
        n = len(city_graph) // 2
        graph = nx.Graph()
        for i in range(n):
            graph.add_node(i, pos=(city_graph[i * 2], city_graph[i * 2 + 1]))

        for i in range(n):
            for j in range(i + 1, n):
                distance = np.sqrt((city_graph[i * 2] - city_graph[j * 2])**2 + (city_graph[i * 2 + 1] - city_graph[j * 2 + 1])**2)
                graph.add_edge(i, j, weight=distance)

        return graph

    def calculate_total_distance(self, city_graph, path=None):
        total_distance = 0

        if path is None:
            path = list(range(len(city_graph) // 2))

        for i in range(len(path) - 1):
            current_city = path[i]
            next_city = path[i + 1]
            x1, y1 = city_graph[current_city * 2], city_graph[current_city * 2 + 1]
            x2, y2 = city_graph[next_city * 2], city_graph[next_city * 2 + 1]
            total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Adding the distance from the last city to the first city for a closed path
        first_city = path[0]
        x1, y1 = city_graph[first_city * 2], city_graph[first_city * 2 + 1]
        last_city = path[-1]
        x2, y2 = city_graph[last_city * 2], city_graph[last_city * 2 + 1]
        total_distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return total_distance

    def plot_city_graph(self, city_graph, path):
        self.plot_ax.clear()
        x = city_graph[::2]
        y = city_graph[1::2]

        # Plot the city graph
        self.plot_ax.scatter(x, y, marker='o', color='b', label='City Graph')
         # Plot the TSP path
        path_x = [x[i] for i in path + [path[0]]]
        path_y = [y[i] for i in path + [path[0]]]
        self.plot_ax.plot(path_x, path_y, marker='o', linestyle='-', color='r', label='TSP Path')

        # Plot the TSP path with arrows
        for i in range(len(path) - 1):
            self.plot_ax.annotate("", xy=(x[path[i + 1]], y[path[i + 1]]), xytext=(x[path[i]], y[path[i]]),
                                  arrowprops=dict(arrowstyle="->", linewidth=2, color='r'), color='r')

        # Annotate vertices with their indices
        for i, (x_coord, y_coord) in enumerate(zip(x, y)):
            self.plot_ax.text(x_coord, y_coord, str(i), color='black', fontsize=10, ha='center', va='center', fontweight='bold')

        self.plot_ax.set_title("City Graph with TSP Path")
        self.plot_ax.set_xlabel("X-axis")
        self.plot_ax.set_ylabel("Y-axis")
        self.plot_ax.legend()

        # Adjust layout to prevent overlap of labels
        self.plot_ax.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        self.plot_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPSolverApp(root)
    root.mainloop()


# In[ ]:




