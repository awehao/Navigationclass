
import cv2
import numpy as np

from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner


class RRTStarImplementation(RRTStarPlanner):
    def preloop(self):
        self.visited_nodes: set[PathNode] = set()
        self.visited_nodes.add(self.start_node)
        self.node_list: list[PathNode] = [self.start_node]  # 樹中所有節點

    def step(self):
        # 1. 取樣隨機節點（30% 機率直接取終點）
        random_node = self.sample_random_node()

        # 2. 找樹中距離隨機節點最近的節點
        nearest_node = min(
            self.node_list,
            key=lambda n: calculate_node_distance(n, random_node)
        )

        # 3. 從 nearest 朝 random 方向延伸，最多走 step_size
        dist = calculate_node_distance(nearest_node, random_node)
        if dist < 1e-6:
            return
        ratio = min(self.step_size, dist) / dist
        new_x = nearest_node.coordinates.x + ratio * (random_node.coordinates.x - nearest_node.coordinates.x)
        new_y = nearest_node.coordinates.y + ratio * (random_node.coordinates.y - nearest_node.coordinates.y)
        new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))

        # 4. 確認新節點在地圖內且路徑無碰撞
        if not check_inside_map(self.occupancy_map, new_node):
            return
        if not check_collision_free(self.occupancy_map, nearest_node, new_node):
            return

        # 5. 找 search_radius 範圍內的近鄰節點
        near_nodes = [
            n for n in self.node_list
            if calculate_node_distance(n, new_node) <= self.search_radius
        ]

        # 6. 從近鄰中選成本最低的父節點（reparent）
        best_parent = nearest_node
        best_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
        for near in near_nodes:
            if not check_collision_free(self.occupancy_map, near, new_node):
                continue
            cost = near.cost + calculate_node_distance(near, new_node)
            if cost < best_cost:
                best_cost = cost
                best_parent = near

        # 7. 加入新節點
        new_node.parent = best_parent
        new_node.cost = best_cost
        self.node_list.append(new_node)
        self.visited_nodes.add(new_node)

        # 8. Rewire：若經過 new_node 能讓近鄰成本更低，就更新它們的父節點
        for near in near_nodes:
            if not check_collision_free(self.occupancy_map, new_node, near):
                continue
            new_cost = new_node.cost + calculate_node_distance(new_node, near)
            if new_cost < near.cost:
                near.parent = new_node
                near.cost = new_cost

        # 9. 檢查是否到達終點
        if calculate_node_distance(new_node, self.goal_node) <= self.goal_threshold:
            cost_via_new = new_node.cost + calculate_node_distance(new_node, self.goal_node)
            if self.goal_node.parent is None or cost_via_new < self.goal_node.cost:
                self.goal_node.parent = new_node
                self.goal_node.cost = cost_via_new
                self.visited_nodes.add(self.goal_node)
            self.is_done.set()

    def postloop(self):
        return (
            collect_path(self.goal_node),
            self.visited_nodes
        )
