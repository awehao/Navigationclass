# RRT* 實作解說 (rrt_star_implementation.py)

---

## RRT* 與 A* 的根本差異

| | A* | RRT* |
|---|---|---|
| 搜尋方式 | 在固定網格上系統性地展開 | 隨機取樣，逐步長出一棵樹 |
| 路徑品質 | 在網格解析度下最佳 | 隨迭代次數增加而趨近最佳 |
| 適用情境 | 結構化、低維空間 | 高維、複雜障礙物空間 |

RRT* 的核心想法：**從起點長出一棵隨機探索樹**，每次加入新節點時透過 **reparent（選最佳父節點）** 和 **rewire（重新接線）** 來不斷改善路徑品質。

---

## 第 1–6 行：import

```python
import cv2        # OpenCV（父類別視覺化用）
import numpy as np  # NumPy（同上）

from path_planning import *                        # PathNode、PixelCoordinates、calculate_node_distance 等
from path_planning.rrt_star_planner import RRTStarPlanner  # 父類別
```

父類別 `RRTStarPlanner` 已提供：
- `sample_random_node()` — 隨機取樣節點（30% 機率直接取終點，加速收斂）
- `self.step_size` — 每步最大延伸距離（50px）
- `self.search_radius` — reparent/rewire 的搜尋半徑（200px）
- `self.start_node`、`self.goal_node`、`self.occupancy_map`、`self.goal_threshold`

---

## 第 10–13 行：`preloop()` — 初始化

```python
self.visited_nodes: set[PathNode] = set()
self.visited_nodes.add(self.start_node)
self.node_list: list[PathNode] = [self.start_node]
```

- `visited_nodes`：所有已加入樹的節點，用於最終視覺化（灰色點）
- `node_list`：同上，但用 list 是因為需要用 `min()` 和 list comprehension 來搜尋
- 起點直接加入，`start_node.cost = 0`（PathNode 預設值）、`start_node.parent = None`

---

## 第 15–79 行：`step()` — 每次迭代

**這個函式會被外層迴圈反覆呼叫，每次長出樹的一個新節點。**

---

### 步驟 1：隨機取樣（第 17 行）

```python
random_node = self.sample_random_node()
```

`sample_random_node()` 內部邏輯：
- 70% 機率：在地圖範圍內完全隨機取一點
- 30% 機率：直接回傳 `goal_node`（goal biasing，讓樹更快往終點生長）

---

### 步驟 2：找最近節點（第 20–23 行）

```python
nearest_node = min(
    self.node_list,
    key=lambda n: calculate_node_distance(n, random_node)
)
```

在目前樹的所有節點中，找離 `random_node` 最近的那個。這是樹生長的「錨點」。

---

### 步驟 3：延伸（Steer）（第 26–32 行）

```python
dist = calculate_node_distance(nearest_node, random_node)
if dist < 1e-6:
    return
ratio = min(self.step_size, dist) / dist
new_x = nearest_node.coordinates.x + ratio * (random_node.coordinates.x - nearest_node.coordinates.x)
new_y = nearest_node.coordinates.y + ratio * (random_node.coordinates.y - nearest_node.coordinates.y)
new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))
```

從 `nearest_node` 朝 `random_node` 方向前進，但**最多只走 `step_size`（50px）**。

- 若 `random_node` 很近（< `step_size`）：`ratio = 1.0`，直接走到 `random_node`
- 若 `random_node` 很遠（> `step_size`）：`ratio < 1.0`，只走固定步長

```
nearest ──[step_size]──> new_node ····· random
```

---

### 步驟 4：碰撞檢查（第 35–38 行）

```python
if not check_inside_map(self.occupancy_map, new_node):
    return
if not check_collision_free(self.occupancy_map, nearest_node, new_node):
    return
```

- `check_inside_map`：確認新節點在地圖邊界內
- `check_collision_free`：用 Bresenham 直線演算法沿路徑逐像素檢查，確認沒有穿過障礙物

任一條件不符合就直接放棄這次取樣，進入下一次迭代。

---

### 步驟 5：找近鄰節點（第 41–44 行）

```python
near_nodes = [
    n for n in self.node_list
    if calculate_node_distance(n, new_node) <= self.search_radius
]
```

找出目前樹中，與 `new_node` 距離在 `search_radius`（200px）以內的所有節點。這些節點是 reparent 和 rewire 的候選對象。

---

### 步驟 6：Reparent — 選最佳父節點（第 47–55 行）

```python
best_parent = nearest_node
best_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
for near in near_nodes:
    if not check_collision_free(self.occupancy_map, near, new_node):
        continue
    cost = near.cost + calculate_node_distance(near, new_node)
    if cost < best_cost:
        best_cost = cost
        best_parent = near
```

RRT 只會把最近的節點當父節點，但 **RRT* 會在近鄰中找「從起點繞過來成本最低」的那個**。

```
start ──[低成本路徑]──> near_A ──> new_node  ← 選這個！
start ──[高成本路徑]──> nearest ──> new_node
```

`cost = near.cost（near 本身的累積成本）+ 到 new_node 的距離`

---

### 步驟 7：加入新節點（第 58–61 行）

```python
new_node.parent = best_parent
new_node.cost = best_cost
self.node_list.append(new_node)
self.visited_nodes.add(new_node)
```

- `new_node.cost` = **從起點到 new_node 的累積路徑總長**（不是兩點間距離）
- 加入 `node_list`（供之後搜尋用）和 `visited_nodes`（供視覺化用）

---

### 步驟 8：Rewire — 重新接線（第 64–70 行）

```python
for near in near_nodes:
    if not check_collision_free(self.occupancy_map, new_node, near):
        continue
    new_cost = new_node.cost + calculate_node_distance(new_node, near)
    if new_cost < near.cost:
        near.parent = new_node
        near.cost = new_cost
```

**這是 RRT* 比 RRT 更好的關鍵。** 新節點加入後，反過來問近鄰：「你繞過我再走，會不會比你現在的路徑更短？」

```
start ──...──> near（舊路徑，成本高）
                    ↑
start ──...──> new_node ──> near（新路徑，成本低）← 改接！
```

若改接成本更低，就更新 `near.parent` 和 `near.cost`。

---

### 步驟 9：終點檢查（第 72–79 行）

```python
if calculate_node_distance(new_node, self.goal_node) <= self.goal_threshold:
    cost_via_new = new_node.cost + calculate_node_distance(new_node, self.goal_node)
    if self.goal_node.parent is None or cost_via_new < self.goal_node.cost:
        self.goal_node.parent = new_node
        self.goal_node.cost = cost_via_new
        self.visited_nodes.add(self.goal_node)
    self.is_done.set()
```

若 `new_node` 距終點 ≤ `goal_threshold`（50px），則嘗試連接終點：
- 比較「經由 new_node 到終點」的成本與現有路徑
- 若更好才更新（保留最佳路徑）
- 無論是否更新，都呼叫 `self.is_done.set()` 結束迭代

---

## 第 81–85 行：`postloop()` — 回傳結果

```python
return (
    collect_path(self.goal_node),
    self.visited_nodes
)
```

- `collect_path(goal_node)`：從 `goal_node` 沿 `.parent` 追蹤回起點，回傳路徑節點列表
- `self.visited_nodes`：樹中所有節點，視覺化為灰色點

---

## 整體執行流程

```
preloop()
  └─ node_list = [start]

step() × N 次：
  random_node ← sample（70% 隨機 / 30% 終點）
  nearest ← 樹中最近節點
  new_node ← 從 nearest 朝 random 延伸 step_size
    ├─ 碰撞或出界 → 跳過
    ├─ Reparent：從 search_radius 內找成本最低的父節點
    ├─ 加入樹
    ├─ Rewire：若繞過 new_node 能讓近鄰更短 → 改接
    └─ 距終點 ≤ 50px → 連接終點，結束 ✓

postloop()
  └─ goal → parent → ... → start（回傳路徑）
```

---

## RRT vs RRT* 差異對照

| | RRT | RRT* |
|---|---|---|
| 父節點選擇 | 只用最近節點 | 近鄰中找成本最低的 |
| Rewire | 無 | 有，持續優化已有路徑 |
| 路徑品質 | 次優 | 隨迭代次數趨近最佳 |
