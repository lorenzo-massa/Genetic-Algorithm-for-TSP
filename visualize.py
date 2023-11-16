import csv
import matplotlib.pyplot as plt
import networkx as nx

# Read the CSV file
csv_file = 'cost_matrix.csv'  # Replace with your file path
show_labels = False # Set to True to show cost labels
cost_matrix = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        cost_matrix.append([float(cell) if cell != 'inf' else float('inf') for cell in row])

# Calculate the number of cities
num_cities = len(cost_matrix) - 1

# Create a graph for cities with costs higher than 0 and not infinity
G = nx.Graph()
for i in range(num_cities):
    for j in range(num_cities):
        cost = cost_matrix[i][j]
        if i != j and cost > 0 and cost != float('inf'):
            if show_labels:
                G.add_edge(i, j, weight=cost)
            else:
                G.add_edge(i, j)

# Create city labels
city_labels = {i: str(i + 1) for i in range(num_cities)}

# Plot the graph
pos = nx.spring_layout(G)  # Layout algorithm
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, labels=city_labels, node_size=300, node_color='skyblue', font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Traveling Salesman Problem')
plt.show()
