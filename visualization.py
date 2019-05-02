from collections import defaultdict
import numpy as np
#import leidenalg
from numpy import inf

def count_triangles(graph):
    return len(graph.cliques(min=3, max=3))

def cluster_coef(graph):
    return graph.transitivity_avglocal_undirected(mode='zero')

def count_edges(graph):
    return len(graph.es)

def count_vertices(graph):
    return len(graph.vs)

def char_path_length(graph):
    shortest_paths = np.asarray(graph.shortest_paths())
    upper_tri = shortest_paths[np.triu_indices(len(shortest_paths))]
    upper_tri[upper_tri == inf] = 0
    return np.mean(upper_tri)

def modularity(graph):
    partitions = leidenalg.find_partition(graph,
                                          leidenalg.ModularityVertexPartition)

    # Calculate v
    v = 0
    for i in graph.vs:
        # Since we assigned all edges the weight 1, we can use the number of
        # neighbors as sum over weights
        neighbors_of_i = graph.neighbors(i)
        v += len(neighbors_of_i)

    a = []
    deltas = []
    for i in graph.vs:
        neighbors_i = graph.neighbors(i)
        s_i = len(neighbors_of_i)

        for j in neighbors_of_i:
            s_j = len(graph.neighbors(j))
            a.append((s_i*s_j)/v)

            deltas.append(np.all([i.index in community_i for community_i in
                                  list(partitions)] == [j in community_j for
                                                       community_j in
                                                       list(partitions)]))

        deltas = np.array(deltas)
        ones = np.ones(len(deltas))
        return v*np.sum((ones-a)*deltas)


stat_funcs = {'num_tri': count_triangles, 'cluster_coef': cluster_coef,
              'edge_count': count_edges, 'vertex_count': count_vertices,
              'char_path': char_path_length, 'modularity': modularity}


def visualize_graph_stats(graphs: list, labels: list, stats: list=['num_tri',
                                                                   'cluster_coef',
                                                                  'edge_count',
                                                                  'vertex_count',
                                                                  'char_path',
                                                                   ]):
    result = defaultdict(dict)

    for stat in stats:
        current_stats = np.array([stat_funcs[stat](g) for g in graphs])
        for label in labels:
            result[stat][label] = current_stats[np.where(labels == label)]

    return result
