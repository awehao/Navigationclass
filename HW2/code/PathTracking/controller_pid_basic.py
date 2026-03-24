import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBasic(Controller):
    def __init__(self,
                 model,
                 # TODO 4.1.2: Tune PID Gains
                 # kp：比例增益；kd 有 1/dt 放大效果（dt=0.05 → ×20）
                 kp=1.5,
                 ki=0.001,
                 kd=0.3):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
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
            return None

        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # 計算車子到最近路徑點的方向角
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0]-x))
        theta_err = theta_target - yaw

        # Problem 4.1.1
        # 橫向誤差：車子偏離目標點連線的垂直距離（有正負號）
        err = min_dist * np.sin(np.deg2rad(theta_err))

        # PID 計算輸出角速度 next_w（單位：deg/s）
        self.acc_ep += err * self.dt
        next_w = self.kp * err + self.ki * self.acc_ep + self.kd * (err - self.last_ep) / self.dt
        self.last_ep = err

        return next_w
