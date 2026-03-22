# A* 實作解說 (a_star_implementation.py)

---

## 第 1–8 行：import

```python
import heapq          # Python 內建最小堆，用來實作優先佇列
import cv2            # OpenCV（此檔案沒有直接用，但父類別需要）
import numpy as np    # NumPy（同上）

from path_planning import *                      # 匯入 PathNode、calculate_node_distance、collect_path 等工具
from path_planning.a_star_planner import AStarPlanner  # 匯入父類別
```

---

## 第 11 行：類別繼承

```python
class AStarImplementation(AStarPlanner):
```

繼承 `AStarPlanner`，它已經提供了：
- `get_neighbor_nodes(node)` — 回傳 8 個方向的鄰居（上下左右＋四個斜角），並自動過濾出界和碰到障礙的節點
- `self.start_node`、`self.goal_node`、`self.occupancy_map`、`self.goal_threshold` 等屬性（由更上層的 `Planner` 設定好）

---

## 第 12–24 行：`preloop()` — 初始化

```python
self.visited_nodes: set[PathNode] = set()
```
已「確認最短路徑」的節點集合（closed set）。

```python
self.g: dict[PathNode, float] = {}
self.h: dict[PathNode, float] = {}
```
- `g[node]`：從起點走到這個節點的**實際成本**（路徑長度）
- `h[node]`：從這個節點到終點的**預估成本**（用直線距離當啟發式）

```python
self.counter = 0
```
純粹是個流水號。因為 heap 裡存的是 `(f值, counter, node)`，當兩個節點 f 值相同時，Python 會繼續比較第二個元素（counter），避免直接比較 `PathNode` 物件而報錯。

```python
self.g[self.start_node] = 0
self.h[self.start_node] = calculate_node_distance(start, goal)
f_start = 0 + h
self.open_heap = [(f_start, 0, self.start_node)]
```
起點的 g=0（還沒走任何路），h=直線距離到終點，把起點推入 heap。

---

## 第 26–62 行：`step()` — 每次迭代做一步

**這個函式會被外層迴圈反覆呼叫，直到 `self.is_done` 被設定為止。**

```python
if not self.open_heap:
    self.is_done.set()
    return
```
heap 空了代表所有節點都探索過了，還沒到終點 → 無解，結束。

---

```python
_, _, current_node = heapq.heappop(self.open_heap)
```
從 heap 取出 **f 值最小**的節點。`heappop` 保證每次取出的是整個 heap 裡最小的元素。`_` 是 f 值和 counter，我們這裡不需要再用到。

---

```python
if current_node in self.visited_nodes:
    return
self.visited_nodes.add(current_node)
```
**Lazy Deletion（惰性刪除）**：同一個節點可能因為「找到更好路徑」而被多次推入 heap。第一次 pop 出來時才真正處理，後續重複出現的直接跳過。

---

```python
if calculate_node_distance(current_node, self.goal_node) <= self.goal_threshold:
    self.goal_node.parent = current_node
    self.is_done.set()
    return
```
不要求**剛好走到**終點座標，只要距離在 `goal_threshold`（50 像素）以內就算到了。
設定 `goal_node.parent = current_node`，讓之後的 `collect_path()` 能從終點往回追蹤。

---

```python
for neighbor in self.get_neighbor_nodes(current_node):
    if neighbor in self.visited_nodes:
        continue
```
取得 8 個方向的鄰居，跳過已確認路徑的節點。

```python
tentative_g = self.g[current_node] + calculate_node_distance(current_node, neighbor)
```
計算「從起點 → current → neighbor」的總成本。
- 直線方向（上下左右）距離 = `grid_size`（50）
- 斜角方向距離 = `grid_size × √2`（≈ 70.7）

```python
if neighbor not in self.g or tentative_g < self.g[neighbor]:
    self.g[neighbor] = tentative_g
    self.h[neighbor] = calculate_node_distance(neighbor, self.goal_node)
    neighbor.parent = current_node
    neighbor.cost = tentative_g
    f = tentative_g + self.h[neighbor]
    self.counter += 1
    heapq.heappush(self.open_heap, (f, self.counter, neighbor))
```
只有在「這條路比之前記錄的更短」時才更新：
- 更新 g、h
- 把 `current_node` 設為 `neighbor` 的父節點（記錄路徑）
- 把新的 f 值推入 heap

---

## 第 64–68 行：`postloop()` — 結束後回傳結果

```python
return (
    collect_path(self.goal_node),
    self.visited_nodes
)
```

- `collect_path(goal_node)`：從 `goal_node` 沿著每個節點的 `.parent` 往回追蹤到起點，回傳整條路徑的節點列表
- `self.visited_nodes`：所有探索過的節點，用來視覺化「搜尋範圍」（灰色點）

---

## 整體執行流程

```
preloop()
  └─ heap = [(f_start, 0, start)]

step() × N 次：
  heap 取出 f 最小的 current
    ├─ 已訪問 → 跳過
    ├─ 距終點 ≤ 50px → 連接終點，結束 ✓
    └─ 展開 8 個鄰居
         └─ 找到更短路徑 → 更新 parent，推入 heap

postloop()
  └─ goal → parent → parent → ... → start（回傳路徑）
```
