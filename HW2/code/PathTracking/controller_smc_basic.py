import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerSMCBasic(Controller):
    """
    Sliding Mode Controller (SMC) for Basic Kinematic Model
    控制輸出：角速度 w (deg/s)
    狀態：x, y, yaw

    滑動面：s = theta_e + lambda * e
      - e      : 橫向誤差 (cross-track error)
      - theta_e: 航向誤差 (heading error)
      - lambda > 0: 調整滑動面斜率，越大越積極修正橫向誤差

    控制律：w = w_eq + w_sw
      - w_eq : 等效控制，維持系統在滑動面上
      - w_sw : 切換控制 = -k * tanh(s / phi)
               tanh 取代 sign 以避免高頻抖振 (chattering)
    """
    def __init__(self, model,
                 lambda_=0.5,   # 滑動面斜率：越大橫向誤差修正越積極
                 k=30.0,        # 切換增益：越大抗擾動越強，但過大會抖振
                 phi=0.1):      # 邊界層厚度：越大越平滑但精度略降
        self.path = None
        self.lambda_ = lambda_
        self.k = k
        self.phi = phi
        self.dt = model.dt
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None

        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(
            self.path, (x, y), self.current_idx, lookahead=50
        )
        self.current_idx = min_idx
        target = self.path[min_idx]

        # 計算車子朝向最近目標點的方向角
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        theta_err = ((theta_target - yaw) + 180) % 360 - 180

        # 橫向誤差 e（帶正負號）
        e = min_dist * np.sin(np.deg2rad(theta_err))

        # 航向誤差 theta_e（deg）：路徑切線方向 vs 車頭方向
        theta_e = ((target[2] - yaw) + 180) % 360 - 180

        # 滑動面 s = theta_e + lambda * e（單位：deg + lambda*m，混合但 lambda 可視為縮放）
        s = theta_e + self.lambda_ * e

        # 等效控制：在 basic model 中難以精確計算，設為 0
        w_eq = 0.0

        # 切換控制：+k * tanh(s/phi)
        # 符號分析：e > 0 且 theta_e > 0 → s > 0 → 需要正 w（逆時針轉向修正）
        # → 應為正號，與 PID (next_w = kp * err, err > 0 → w > 0) 一致
        w_sw = self.k * np.tanh(s / self.phi)

        next_w = w_eq + w_sw
        return next_w
