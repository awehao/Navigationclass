
import heapq

import cv2
import numpy as np

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    def preloop(self):
        self.visited_nodes: set[PathNode] = set()
        self.g: dict[PathNode, float] = {}
        self.h: dict[PathNode, float] = {}
        self.counter = 0  # 用來在 f 值相同時打破平局

        # 初始化起點
        self.g[self.start_node] = 0
        self.h[self.start_node] = calculate_node_distance(self.start_node, self.goal_node)
        f_start = self.g[self.start_node] + self.h[self.start_node]

        # open_heap: (f值, 計數器, 節點)
        self.open_heap: list = [(f_start, self.counter, self.start_node)]

    def step(self):
        # 佇列空了，無解
        if not self.open_heap:
            self.is_done.set()
            return

        # 取出 f 值最小的節點
        _, _, current_node = heapq.heappop(self.open_heap)

        # 已訪問過就跳過（堆裡可能有過時的副本）
        if current_node in self.visited_nodes:
            return
        self.visited_nodes.add(current_node)

        # 檢查是否到達目標
        if calculate_node_distance(current_node, self.goal_node) <= self.goal_threshold:
            self.goal_node.parent = current_node
            self.goal_node.cost = self.g[current_node] + calculate_node_distance(current_node, self.goal_node)
            self.visited_nodes.add(self.goal_node)
            self.is_done.set()
            return

        # 展開鄰居節點
        for neighbor in self.get_neighbor_nodes(current_node):
            if neighbor in self.visited_nodes:
                continue

            tentative_g = self.g[current_node] + calculate_node_distance(current_node, neighbor)

            # 找到更好的路徑才更新
            if neighbor not in self.g or tentative_g < self.g[neighbor]:
                self.g[neighbor] = tentative_g
                self.h[neighbor] = calculate_node_distance(neighbor, self.goal_node)
                neighbor.parent = current_node
                neighbor.cost = tentative_g

                f = tentative_g + self.h[neighbor]
                self.counter += 1
                heapq.heappush(self.open_heap, (f, self.counter, neighbor))

    def postloop(self):
        return (
            collect_path(self.goal_node),
            self.visited_nodes
        )
