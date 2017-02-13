# Unit Testing code courtesy one of our former students -  Mac Chan
# 
# The following unit tests will check for all pairs on romania and random points on atlanta.
# Comment out any tests that you haven't implemented yet.
 
# If you failed on bonnie because of non-optimal path, make sure you pass all the local tests.
# Change ntest=-1 if you failed the path test on bonnie, it will run tests on atlanta until it finds a set of points that fail.
 
# If you failed on bonnie because of your explored set is too large, there is no easy way to test without a reference implementation. 
# But you can read the pdf slides for the optimized terminal condition.

# To run,
# nosetests --nocapture -v map_test.py:HW2Tests
# nosetests --nocapture -v map_test.py:HW2Tests.test_tucs_romania

import unittest
import pickle
import random

import sys
sys.path.append('./lib')
sys.path.append('./workspace/lib')
import networkx

# if you run this from a separate script, uncomment the following
from search_submission import breadth_first_search, uniform_cost_search, null_heuristic, euclidean_dist_heuristic, a_star, bidirectional_ucs, bidirectional_a_star, tridirectional_search, tridirectional_upgraded


class HW2Tests(unittest.TestCase):
    margin_of_error = 1.0e-6

    def setUp(self):
        self.romania = pickle.load(open('romania_graph.pickle', 'rb'))
        self.romania.reset_search()
        self.atlanta = pickle.load(open('atlanta_osm.pickle', 'rb'))
        self.atlanta.reset_search()

    def reference_path(self, g, src, dst, weight='weight'):
        g.reset_search()
        p = networkx.shortest_path(g, src, dst, weight=weight)
        c = self.sum_weight(g, p)
        return c, p

    def reference_bfs_path(self, g, src, dst):
        return self.reference_path(g, src, dst, weight=None)

    def sum_weight(self, g, path):
        pairs = zip(path, path[1:])
        return sum([g.get_edge_data(a, b)['weight'] for a, b in pairs])

    def romania_test(self, ref_method, method, assert_explored=False, **kwargs):
        keys = self.romania.node.keys()
        pairs = zip(keys, keys[1:])
        for src, dst in pairs:
            self.romania.reset_search()
            path = method(self.romania, src, dst, **kwargs)
            explored = len(self.romania.get_explored_nodes())
            ref_len, ref_path = ref_method(self.romania, src, dst)
            ref_explored = len(self.romania.get_explored_nodes())
            if path != ref_path:
                print src, dst
            assert path == ref_path
            if assert_explored:
                assert explored <= ref_explored

    def romania_tri_test(self, method):
        import itertools
        keys = self.romania.node.keys()
        triplets = zip(keys, keys[1:], keys[2:])
        for goals in triplets:
            for allcombo in itertools.permutations(goals):
                self.romania.reset_search()
                path = method(self.romania, allcombo)
                explored = len(self.romania.get_explored_nodes())
                pathlen = self.sum_weight(self.romania, path)
                s1len, s1 = self.reference_path(self.romania, allcombo[0], allcombo[1])
                s2len, s2 = self.reference_path(self.romania, allcombo[2], allcombo[1])
                s3len, s3 = self.reference_path(self.romania, allcombo[0], allcombo[2])
                minlen = min(s1len + s2len, s1len + s3len, s3len + s2len)
                if pathlen != minlen:
                    print allcombo
                assert pathlen == minlen

    def atlanta_bi_test(self, method, ntest=10, assert_explored=False, **kwargs):
        keys = list(networkx.connected_components(self.atlanta).next())
        random.shuffle(keys)
        for src, dst in zip(keys, keys[1:])[::2]:
            self.atlanta.reset_search()
            path = method(self.atlanta, src, dst, **kwargs)
            explored = len(self.atlanta.get_explored_nodes())
            pathlen = self.sum_weight(self.atlanta, path)
            ref_len, ref_path = self.reference_path(self.atlanta, src, dst)
            ref_explored = len(self.atlanta.get_explored_nodes())
            if abs(pathlen - ref_len) > self.margin_of_error:
                print src, dst
            assert abs(pathlen - ref_len) <= self.margin_of_error
            if assert_explored:
                assert explored <= ref_explored
            ntest -= 1
            if ntest == 0:
                break

    def atlanta_tri_test(self, method, ntest=10):
        keys = list(networkx.connected_components(self.atlanta).next())
        random.shuffle(keys)
        for goals in zip(keys, keys[1:], keys[2:])[::3]:
            self.atlanta.reset_search()
            path = method(self.atlanta, goals)
            explored = len(self.atlanta.get_explored_nodes())
            pathlen = self.sum_weight(self.atlanta, path)
            s1len, s1 = self.reference_path(self.atlanta, goals[0], goals[1])
            s2len, s2 = self.reference_path(self.atlanta, goals[2], goals[1])
            s3len, s3 = self.reference_path(self.atlanta, goals[0], goals[2])
            minlen = min(s1len + s2len, s1len + s3len, s3len + s2len)
            if abs(pathlen - minlen) > self.margin_of_error:
                print goals
            assert abs(pathlen - minlen) <= self.margin_of_error
            ntest -= 1
            if ntest == 0:
                break

    def same_node_bi_test(self, graph, method, ntest=10, **kwargs):
        keys = list(networkx.connected_components(graph).next())
        random.shuffle(keys)
        for i in range(ntest):
            path = method(graph, keys[i], keys[i], **kwargs)
            assert path == []

    def same_node_tri_test(self, graph, method, ntest=10):
        keys = list(networkx.connected_components(graph).next())
        random.shuffle(keys)
        for i in range(ntest):
            path = method(graph, [keys[i], keys[i], keys[i]])
            assert path == []

    def test_same_node_bi(self):
        self.same_node_bi_test(self.romania, breadth_first_search)
        self.same_node_bi_test(self.romania, uniform_cost_search)
        self.same_node_bi_test(self.romania, a_star, heuristic=null_heuristic)
        self.same_node_bi_test(self.romania, a_star, heuristic=euclidean_dist_heuristic)
        self.same_node_bi_test(self.romania, bidirectional_ucs)
        self.same_node_bi_test(self.romania, bidirectional_a_star, heuristic=null_heuristic)
        self.same_node_bi_test(self.romania, bidirectional_a_star, heuristic=euclidean_dist_heuristic)

    def test_same_node_tri(self):
        self.same_node_tri_test(self.romania, tridirectional_search)
        self.same_node_tri_test(self.romania, tridirectional_upgraded)

    def test_bfs_romania(self):
        self.romania_test(self.reference_bfs_path, breadth_first_search)

    def test_uni_romania(self):
        self.romania_test(self.reference_path, uniform_cost_search)

    def test_astar_nul_romania(self):
        self.romania_test(self.reference_path, a_star, heuristic=null_heuristic)

    def test_astar_euc_romania(self):
        self.romania_test(self.reference_path, a_star, heuristic=euclidean_dist_heuristic)

    def test_bucs_romania(self):
        self.romania_test(self.reference_path, bidirectional_ucs)

    def test_bucs_atlanta(self):
        # put ntest = -1 to run forever until it breaks
        self.atlanta_bi_test(bidirectional_ucs, ntest=10)

    def test_bastar_nul_romania(self):
        self.romania_test(self.reference_path, bidirectional_a_star, heuristic=null_heuristic)

    def test_bastar_nul_atlanta(self):
        # put ntest = -1 to run forever until it breaks
        self.atlanta_bi_test(bidirectional_a_star, heuristic=null_heuristic, ntest=10)

    def test_bastar_euc_romania(self):
        self.romania_test(self.reference_path, bidirectional_a_star, heuristic=euclidean_dist_heuristic)

    def test_bastar_euc_atlanta(self):
        # put ntest = -1 to run forever until it breaks
        self.atlanta_bi_test(bidirectional_a_star, heuristic=euclidean_dist_heuristic, ntest=10)

    def test_tucs_romania(self):
        self.romania_tri_test(tridirectional_search)

    def test_tucs_atlanta(self):
        # put ntest = -1 to run forever until it breaks
        self.atlanta_tri_test(tridirectional_search, ntest=10)

    def test_tri_upgraded_romania(self):
        self.romania_tri_test(tridirectional_upgraded)

    def test_tri_upgraded_atlanta(self):
        # put ntest = -1 to run forever until it breaks
        self.atlanta_tri_test(tridirectional_upgraded, ntest=10)


if __name__ == '__main__':
    unittest.main()
