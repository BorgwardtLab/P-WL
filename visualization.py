from collections import defaultdict
import numpy as np

def count_triangles(graph):
    return len(graph.cliques(min=3, max=3))

def cluster_coef(graph):
    return graph.transitivity_undirected(mode='zero')

def count_edges(graph):
    return len(graph.es)

def count_vertices(graph):
    return len(graph.vs)

stat_funcs = {'num_tri': count_triangles, 'cluster_coef': cluster_coef,
              'edge_count': count_edges, 'vertex_count': count_vertices}


def visualize_graph_stats(graphs: list, labels: list, stats: list=['num_tri',
                                                                   'cluster_coef',
                                                                  'edge_count',
                                                                  'vertex_count']):
    result = defaultdict(dict)

    for stat in stats:
        current_stats = np.array([stat_funcs[stat](g) for g in graphs])
        for label in labels:
            result[stat][label] = current_stats[np.where(labels == label)]

    return result
