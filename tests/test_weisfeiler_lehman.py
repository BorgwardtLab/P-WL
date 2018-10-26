import unittest

class WeisfeilerLehmanTests(unittest.TestCase):
    
    def test_relabeling_graph_single(self):
        wl = WL()
        graph = ig.Graph([(0,1), (0,2), (0,3), (1,2), (3,4), (3,5) ])
        graph_labels = [5,2,3,4,1,1]
        graph.vs['label'] = graph_labels
        
        graph_transformed = wl.fit_transform([graph], num_iterations=1)

        # Check relabeling
        expected_preprocessing_relabeling = { 5: 0, 2: 1, 3: 2, 4: 3, 1: 4 }
        self.assertEqual(wl._preprocess_relabel_dict, expected_preprocessing_relabeling)

        # First iteration relabeling
        expected_relabeling = { '0-1-2-3': 0, '1-0-2': 1, '2-0-1': 2, '3-0-4-4': 3, '4-3': 4 }
        self.assertEqual(expected_relabeling, wl._label_dict)

        # Expected labels of transformed graph
        expected_labels = [0,1,2,3,4,4]
        self.assertEqual(expected_labels, graph_transformed[1][0][1])

        # Expected sorted label counts before  and after WL iteration
        unique_label_counts_before = sorted(Counter([ '-'.join(np.asarray(x).astype(str)) for x in graph_transformed[1][0][0]]).values())
        unique_label_counts_after = sorted(Counter(graph_transformed[1][0][1]).values())
        self.assertEqual(unique_label_counts_before, unique_label_counts_after)

    def test_relabeling_graph_double(self):
        wl = WL()
        graph_1 = ig.Graph([(0,1), (0,2), (0,3), (1,2), (2,3), (3,4), (3,5) ])
        graph_labels_1 = [5,2,3,4,1,1]
        graph_1.vs['label'] = graph_labels_1 

        graph_2 = ig.Graph([(0,1), (0,3), (1,2), (1,3), (2,3), (2,5), (3,4)])
        graph_labels_2 = [2,5,3,4,1,2]
        graph_2.vs['label'] = graph_labels_2

        graph_transformed = wl.fit_transform([graph_1, graph_2], num_iterations=1)

        # Check relabeling
        expected_preprocessing_relabeling = { 5: 0, 2: 1, 3: 2, 4: 3, 1: 4 }
        self.assertEqual(wl._preprocess_relabel_dict, expected_preprocessing_relabeling)

        # First iteration relabeling
        expected_relabeling = { '0-1-2-3': 0, '1-0-2': 1, '2-0-1-3': 2, '3-0-2-4-4': 3, '4-3': 4, '1-0-3': 5, '3-0-1-2-4': 6, '1-2': 7}
        self.assertEqual(expected_relabeling, wl._label_dict)

        # Check results
        expected_results_graph_1 = ([[0,1,2,3], [1,0,2], [2,0,1,3], [3,0,2,4,4], [4,3], [4,3]], [0,1,2,3,4,4])
        expected_results_graph_2 = ([[1,0,3], [0,1,2,3], [2,0,1,3], [3,0,1,2,4], [4,3], [1,2]], [5,0,2,6,4,7])
        self.assertEqual( {0: expected_results_graph_1, 1: expected_results_graph_2}, graph_transformed[1] )

        # Expected labels of transformed graph
        expected_labels_graph_1 = [0,1,2,3,4,4]
        self.assertEqual(expected_labels_graph_1, graph_transformed[1][0][1])

        expected_labels_graph_2 = [5,0,2,6,4,7]
        self.assertEqual(expected_labels_graph_2, graph_transformed[1][1][1])
        
        # Expected sorted label counts before  and after WL iteration
        unique_label_counts_before = sorted(Counter([ '-'.join(np.asarray(x).astype(str)) for x in graph_transformed[1][0][0]]).values())
        unique_label_counts_after = sorted(Counter(graph_transformed[1][0][1]).values())
        self.assertEqual(unique_label_counts_before, unique_label_counts_after)

        # Check feature mapping
        possible_original_labels = wl._preprocess_relabel_dict.values()
        counts_orig_labels_graph_1 = []
        orig_label_counts_graph_1 = Counter([0,1,2,3,4,4])
        counts_orig_labels_graph_2 = []
        orig_label_counts_graph_2 = Counter([1,0,2,3,4,1])
        for label in possible_original_labels:
            counts_orig_labels_graph_1.append(orig_label_counts_graph_1[label])
            counts_orig_labels_graph_2.append(orig_label_counts_graph_2[label])
       
        dot_product = np.dot(np.array(counts_orig_labels_graph_1), np.array(counts_orig_labels_graph_2).T)
        self.assertEqual(dot_product, 7)
        
        # Assemble mapped vector
        possible_mapped_labels = wl._label_dict.keys()
        counts_mapped_labels_graph_1 = []
        mapped_label_counts_graph_1 = Counter([ '-'.join(np.asarray(x).astype(str)) for x in graph_transformed[1][0][0]])
        counts_mapped_labels_graph_2 = []
        mapped_label_counts_graph_2 = Counter([ '-'.join(np.asarray(x).astype(str)) for x in graph_transformed[1][1][0]])

        for label in possible_mapped_labels:
            counts_mapped_labels_graph_1.append(mapped_label_counts_graph_1[label])
            counts_mapped_labels_graph_2.append(mapped_label_counts_graph_2[label])

        dot_product = np.dot(np.array(counts_mapped_labels_graph_1), np.array(counts_mapped_labels_graph_2).T)
        self.assertEqual(dot_product, 4)

        # Asses self-similarity
        concat_graph_1 = counts_orig_labels_graph_1 + counts_mapped_labels_graph_1
        dot_product = np.dot(concat_graph_1, concat_graph_1)
        self.assertEqual(dot_product, 16)

        concat_graph_2 = counts_orig_labels_graph_2 + counts_mapped_labels_graph_2
        dot_product = np.dot(concat_graph_2, concat_graph_2)
        self.assertEqual(dot_product, 14)


class MUTAGTest(unittest.TestCase):
    def test(self):
        G = ig.read('data/MUTAG/000.gml')
        G = [G]

        import graphkernels as gk
        K = gk.CalculateWLKernel(G, 1)

        labels_dict_0 = WL().fit_transform(G, num_iterations=0)
        labels_dict_1 = WL().fit_transform(G, num_iterations=1)

        # Check zeroth iteration: original 'relabelled' labels should
        # coincide.
        self.assertEqual(labels_dict_0[0], labels_dict_1[0])

        print(labels_dict_1[1])


if __name__== '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from weisfeiler_lehman import WL
    import igraph as ig
    import numpy as np
    from collections import Counter
    unittest.main()
