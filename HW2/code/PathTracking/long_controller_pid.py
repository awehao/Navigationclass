import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class PIDLongController(Controller):
    def __init__(self, model, a_range, kp=1.5, ki=5, kd=0):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.a_range = a_range  # 加速度上下限 [a_min, a_max]
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.current_idx = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 1:
            # Brake to 0 speed using PID when finishing the track
            v_ref = 0.0
            target = self.path[-1]
            return np.clip(-v, self.a_range[0], self.a_range[1]), target
        else:
            # Search Nearest Target Locally
            min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
            self.current_idx = min_idx
            target = self.path[min_idx]
            v_ref = target[4]  # 目標速度由速度剖面提供（waypoint 第 5 個欄位）

        # TODO 3.2: PID Control for Longitudinal Motion
        # 速度誤差 = 目標速度 - 當前速度
        ep = v_ref - v
        self.acc_ep += ep * self.dt  # 積分項累加
        # PID 輸出加速度指令
        next_a = self.kp * ep + self.ki * self.acc_ep + self.kd * (ep - self.last_ep) / self.dt
        self.last_ep = ep
        # 限制加速度在模型允許範圍內（防止超出物理限制）
        next_a = np.clip(next_a, self.a_range[0], self.a_range[1])
        # [end] TODO 3.2

        return next_a, target
