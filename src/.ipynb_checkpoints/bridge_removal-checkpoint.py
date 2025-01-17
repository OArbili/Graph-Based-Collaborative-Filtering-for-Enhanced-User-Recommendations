from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx

class bridge_removal():
    
    def __init__(self):
        self.G = None
        self.G_no_bridges = None
        self.adj_sparse_mtx = None
        self.bridge_edges = None
    
    def load_networkx_g(self,G):
        self.G = G

    def load_adj_matrix(self,adj_matrix):
        self.G = to_networkx(adj_matrix)
        
    def find_bridge_edges(self):
        self.bridge_edges = list(nx.bridges(self.G))

    def remove_bridge_edges(self):
        self.G_no_bridges = self.G.copy()
        self.G_no_bridges.remove_edges_from(self.bridge_edges)
    
    def get_no_bridge_adj_matrix(self):
        return from_networkx(self.G_no_bridges)
    
    def get_adj_matrix(self):
        data = from_networkx(self.G)
        import torch_geometric.transforms as T

        sparse_tensor = T.ToSparseTensor()(data)

        # sparse_tensor = torch.sparse_coo_tensor(adj_matrix, edge_weights, (data.num_nodes, data.num_nodes))
        return sparse_tensor
    
    def display_graph(self,title,which='full'):
        if which == 'full':
            pos = nx.spring_layout(self.G)
            nx.draw(self.G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=12, font_weight='bold')
            plt.title(title)
            plt.axis('off')
            plt.show()
        else:
            pos = nx.spring_layout(self.G_no_bridges)
            nx.draw(self.G_no_bridges, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=12, font_weight='bold')
            plt.title(title)
            plt.axis('off')
            plt.show()