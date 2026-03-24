import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, model,
                 # TODO 4.1.3: Tune PID Gains
                 # kp：比例增益，誤差越大轉向越大；太高會震盪
                 # ki：積分增益，消除穩態誤差；太高會積分飽和
                 # kd：微分增益，抑制誤差變化速率（注意實際效果 = kd/dt）
                 # k_heading：航向誤差權重，讓控制器在彎道入口提前轉向
                 kp=0.6,
                 ki=0.001,
                 kd=0.08,
                 k_heading=100.0):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.k_heading = k_heading
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
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx

        # TODO 4.1.3: PID Control for Bicycle Kinematic Model
        target = self.path[min_idx]

        # 計算車子朝向最近目標點的方向，與車頭方向的夾角
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        # 正規化到 [-180, 180] 避免角度跨越 0/360 時符號錯誤
        theta_err = ((theta_target - yaw) + 180) % 360 - 180

        # 橫向誤差（cross-track error）：車子偏離路徑的垂直距離
        # 正值 = 車子在路徑右側，負值 = 左側
        e_cte = min_dist * np.sin(np.deg2rad(theta_err))

        # 航向誤差（heading error）：車頭方向與路徑切線方向的夾角（弧度）
        # 彎道入口時 e_cte≈0，但 e_heading 已不為 0，可提前預知轉向需求
        e_heading = np.deg2rad(((target[2] - yaw) + 180) % 360 - 180)

        # 合成誤差 = 橫向誤差 + 航向誤差（以 k_heading 縮放至相同量級）
        err = e_cte + self.k_heading * e_heading

        # PID 計算：P 項 + I 項（累積誤差）+ D 項（誤差變化率）
        self.acc_ep += err * self.dt
        next_delta = self.kp * err + self.ki * self.acc_ep + self.kd * (err - self.last_ep) / self.dt
        self.last_ep = err
        # [end] TODO 4.1.3
        return next_delta
