# Yonathan Lim
# yhalim3
# CS6601
# Assignment 2

# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.
from __future__ import division
import math
from osm2networkx import *
import random
import pickle
import sys
import os
# Comment the next line when submitting to bonnie
# import matplotlib.pyplot as plt

# Implement a heapq backed priority queue (accompanying the relevant question)
import heapq

class PriorityQueue():
    # HINT look up/use the module heapq.

    REMOVED = 'REMOVED'

    def __init__(self, remove_existing=False):
        self.remove_existing = remove_existing
        # entry map concept from https://docs.python.org/2/library/heapq.html
        self.entries = {}
        self.queue = []
        self.current = 0

    def next(self):
        if self.current >= len(self.queue):
            self.current
            raise StopIteration

        out = self.queue[self.current]
        self.current += 1

        return out

    def pop(self):
        if self.remove_existing:
            while self.queue:
                entry = heapq.heappop(self.queue)
                cost, state = entry[:2]
                if state != self.REMOVED:
                    self.entries.pop(state)
                    return entry
            raise KeyError('Poping empty priority queue')

        return heapq.heappop(self.queue)

    def remove(self, nodeId):
        node = self.entries.pop(nodeId)
        node[1] = self.REMOVED

    def append(self, entry):
        add = True
        cost, state = entry[:2]
        if self.remove_existing and self.has_state(state):
            [prev_cost] = self.entries[state][:1]
            add = cost < prev_cost
            if add:
                self.remove(state)
        if add:
            self.entries[state] = entry
            heapq.heappush(self.queue, entry)

    def has_state(self, state):
        return state in self.entries

    def get_entry(self, state):
        if self.has_state(state):
            return self.entries[state]

    def __iter__(self):
        return self

    def __str__(self):
        return 'PQ:[%s]' % (', '.join([str(i) for i in self.queue]))

    def __contains__(self, key):
        self.current = 0
        return key in [n for v,n in self.queue]

    def __eq__(self, other):
        self.current = 0
        return self == other

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue = []

    def top(self):
        return self.queue[0]

    __next__ = next


#Warmup exercise: Implement breadth-first-search
def breadth_first_search(graph, start, goal):
    level = 0
    pq = PriorityQueue(remove_existing=True)
    visited = {}
    path = []
    end_node = None

    if start == goal or graph[start] is None:
        return path

    node = {'state': start, 'weight': 0, 'parent': None}
    pq.append([level, start, node])
    while end_node is None:
        entry = pq.pop()
        cost, state, node = entry
        visited[state] = node
        level += 1
        neighbors = graph[state]
        for next_state in neighbors.keys():
            next_node = {
                'state': next_state,
                'weight': node['weight'],
                'parent': node['state']
            }
            if next_state not in visited and not pq.has_state(next_state):
                if next_state == goal:
                    end_node = next_node
                    break
                pq.append([level, next_state, next_node])

    if end_node:
        node = end_node
        while node['parent']:
            path.insert(0, node['state'])
            node = visited[node['parent']]
        path.insert(0, node['state'])

    return path


#Warmup exercise: Implement uniform_cost_search
def uniform_cost_search(graph, start, goal):
    """Run uniform-cost search from start
    to goal and return the path"""
    return a_star(graph, start, goal, heuristic=null_heuristic)


# Warmup exercise: Implement A*
def null_heuristic(graph, v, goal ):
    return 0


# Warmup exercise: Implement the euclidean distance heuristic
def euclidean_dist_heuristic(graph, v, goal):
    if 'pos' in graph.node[v]:
        cur_pos = graph.node[v]['pos']
        goal_pos = graph.node[goal]['pos']
    else:
        cur_pos = graph.node[v]['position']
        goal_pos = graph.node[goal]['position']
    return math.sqrt((goal_pos[0] - cur_pos[0])**2 + (goal_pos[1] - cur_pos[1])**2)


# Warmup exercise: Implement A* algorithm
def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    pq = PriorityQueue(remove_existing=True)
    visited = {}
    path = []
    end_node = None

    if start == goal or graph[start] is None:
        return path

    init_weight = heuristic(graph, start, goal)
    node = {'state': start, 'weight': init_weight, 'true_cost': 0, 'parent': None}
    pq.append([init_weight, start, node])
    while end_node is None and pq.size():
        entry = pq.pop()
        cost, state, node = entry
        if state == goal:
            end_node = node
            break
        visited[state] = node

        neighbors = graph[state]
        for next_state in neighbors.keys():
            next_node = {
                'state': next_state,
                'weight': neighbors[next_state]['weight'],
                'true_cost': neighbors[next_state]['weight'] + node['true_cost'],
                'parent': node['state']
            }
            estimated_cost = heuristic(graph, next_state, goal)
            if next_state not in visited:
                pq.append([next_node['true_cost'] + estimated_cost, next_state, next_node])

    if end_node:
        node = end_node
        while node['parent']:
            path.insert(0, node['state'])
            node = visited[node['parent']]
        path.insert(0, node['state'])

    return path


# Exercise 1: Bidirectional Search
def bidirectional_ucs(graph, start, goal):
    return bidirectional_a_star(graph, start, goal, heuristic=null_heuristic)


def get_path(state, forward_node, backward_node, forward_visited, backward_visited):
    # TODO: update to check the intersection rather than using nodes that have been explored from forward and backward
    forward_path = []
    backward_path = []

    node = forward_node
    forward_path.insert(0, state)
    while node['parent']:
        forward_path.insert(0, node['parent'])
        node = forward_visited[node['parent']]

    node = backward_node
    while node['parent']:
        backward_path.append(node['parent'])
        node = backward_visited[node['parent']]

    return forward_path + backward_path


def expand_node(graph, goal, forward_state, forward_node, forward_frontier, forward_visited, heuristic):
    forward_visited[forward_state] = forward_node
    for forward_state_next, forward_neighbor in graph[forward_state].iteritems():
        forward_node_next = {
            'state': forward_state_next,
            'weight': forward_neighbor['weight'],
            'true_cost': forward_neighbor['weight'] + forward_node['true_cost'],
            'parent': forward_node['state']
        }
        estimated_cost = heuristic(graph, forward_state_next, goal)
        if forward_state_next not in forward_visited:
            forward_frontier.append(
                [forward_node_next['true_cost'] + estimated_cost, forward_state_next, forward_node_next])
    return forward_frontier, forward_visited


# Exercise 2: Bidirectional A*
def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    forward_frontier = PriorityQueue(remove_existing=True)
    backward_frontier = PriorityQueue(remove_existing=True)

    forward_visited = {}
    backward_visited = {}
    path = []

    if start == goal or graph[start] is None or graph[goal] is None:
        return path

    start_weight = 0  # heuristic(graph, start, goal)
    goal_weight = 0  # heuristic(graph, goal, start)
    start_node = {'state': start, 'weight': start_weight, 'true_cost': start_weight, 'parent': None}
    goal_node = {'state': goal, 'weight': goal_weight, 'true_cost': goal_weight, 'parent': None}
    forward_frontier.append([start_weight, start, start_node])
    backward_frontier.append([goal_weight, goal, goal_node])
    while forward_frontier.size() or backward_frontier.size():
        if forward_frontier.size():
            forward_entry = forward_frontier.pop()
            forward_cost, forward_state, forward_node = forward_entry
            path_found = forward_state in backward_visited
            if path_found:
                backward_node = backward_visited[forward_state]
                return get_path(forward_state, forward_node, backward_node, forward_visited, backward_visited)

            forward_frontier, forward_visited = expand_node(
                graph, goal, forward_state, forward_node, forward_frontier, forward_visited, heuristic)

        if backward_frontier.size():
            backward_entry = backward_frontier.pop()
            backward_cost, backward_state, backward_node = backward_entry
            path_found = backward_state in forward_visited
            if path_found:
                forward_node = forward_visited[backward_state]
                return get_path(backward_state, forward_node, backward_node, forward_visited, backward_visited)

            backward_frontier, backward_visited = expand_node(
                graph, start, backward_state, backward_node, backward_frontier, backward_visited, heuristic)

    return path


# Exercise 3: Tridirectional UCS Search
def tridirectional_search(graph, goals):
    return tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic)


def get_cost(graph, path):
    cost = 0
    for i in xrange(len(path) - 1):
        source = path[i]
        target = path[i+1]
        cost += graph.get_edge_data(source, target)['weight']

    return cost


def half_euclidean_heuristic(graph, v, goal):
    return euclidean_dist_heuristic(graph, v, goal) / 2


# Exercise 4: Present an improvement on tridirectional search in terms of nodes explored
def tridirectional_upgraded(graph, goals, heuristic=half_euclidean_heuristic):
    pairs = range(len(goals))
    combo = []  # create combination (0 -> 1, 1 -> 2, 2 -> 0)
    for i in pairs:
        if i == len(goals) - 1:
            pairs[i] = goals[i:] + goals[:1]
            combo.append([i, 0])
        else:
            pairs[i] = goals[i:i+2]
            combo.append([i, i+1])

    frontier_dict = {}
    visited_dict = {}
    path_dict = {}
    best_path_found = {}
    for i, pair in enumerate(pairs):
        start, goal = pair
        start_weight = 0  # heuristic(graph, start, goal)
        start_node = {'state': start, 'weight': start_weight, 'true_cost': start_weight, 'parent': None}
        frontier = PriorityQueue(remove_existing=True)
        frontier.append([start_weight, start, start_node])
        frontier_dict[i] = frontier
        visited_dict[i] = {}
        best_path_found[i] = False

    while not all(best_path_found.values()) and sum([frontier.size() for frontier in frontier_dict.values()]):
        for i, pair in enumerate(pairs):
            start, goal = pair
            target_index = i+1 if i < len(pairs) - 1 else 0
            frontier = frontier_dict[i]
            visited = visited_dict[i]
            target_visited = visited_dict[target_index]

            if frontier.size() and not best_path_found[i]:
                entry = frontier.pop()
                cost, state, node = entry
                path_found = state in target_visited
                if path_found:
                    target_node = target_visited[state]
                    path_dict[i] = get_path(state, node, target_node, visited, target_visited)
                    best_path_found[i] = True

                frontier, visited = expand_node(graph, goal, state, node, frontier, visited, heuristic)
                frontier_dict[i] = frontier
                visited_dict[i] = visited

    path_list = []
    for i, pair in enumerate(pairs):
        path = path_dict[i]
        cost = get_cost(graph, path)
        path_list.append([cost, path])

    final_path = []
    for i, pair in enumerate(combo):
        path1 = path_list[pair[0]]
        path2 = path_list[pair[1]]
        total_cost = path1[0] + path2[0]
        connected_path = path1[1] + path2[1][1:]
        final_path.append([total_cost, connected_path])

    path = min(final_path)[1]
    return path


# Extra Credit: Your best search method for the race
# Loads data from data.pickle and return the data object that is passed to the custom_search method. Will be called only once. Feel free to modify. 
def load_data():
    data = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.pickle"), 'rb'))
    return data

def custom_search(graph, goals, data=None):
    raise NotImplementedError
