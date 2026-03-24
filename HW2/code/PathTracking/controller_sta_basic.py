import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerSTABasic(Controller):
    """
    Super Twisting Algorithm (STA) + SMC for Basic Kinematic Model
    控制輸出：角速度 w (deg/s)

    傳統 SMC 切換項：w_sw = k * sign(s)  → 高頻抖振
    STA 改良：將切換控制拆成兩項（平方根項 + 積分項），輸出連續無抖振

    STA 控制律：
        w = w_eq + w_sw
        w_sw = β * √|s| * sign(s) + u1
        u̇1  = α * sign(s)          ← u1 為積分項，使輸出連續

    滑動面：s = θ_e + λ * e
        θ_e > 0（需正 w）→ s > 0 → w_sw > 0 ✓
        e   > 0（需正 w）→ s > 0 → w_sw > 0 ✓
    """
    def __init__(self, model,
                 lambda_=0.5,   # 滑動面斜率
                 alpha=5.0,     # 積分增益（u1 的收斂速度，太大會超調）
                 beta=15.0):    # 平方根增益（初期誤差修正力道）
        self.path = None
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.dt = model.dt
        self.u1 = 0.0           # STA 積分項（需跨時步保存）
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.u1 = 0.0
        self.current_idx = 0

    # State: [x, y, yaw]
    def feedback(self, info):
        if self.path is None:
            print("No path !!")
            return None

        x, y, yaw = info["x"], info["y"], info["yaw"]

        if self.current_idx >= len(self.path) - 3:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(
            self.path, (x, y), self.current_idx, lookahead=50
        )
        self.current_idx = min_idx
        target = self.path[min_idx]

        # 橫向誤差 e（與 PID basic 相同計算方式）
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        theta_err = ((theta_target - yaw) + 180) % 360 - 180
        e = min_dist * np.sin(np.deg2rad(theta_err))

        # 航向誤差 θ_e（deg）
        theta_e = ((target[2] - yaw) + 180) % 360 - 180

        # 滑動面
        s = theta_e + self.lambda_ * e

        # Super Twisting 切換控制
        # β * √|s| * sign(s)：平方根項，大誤差時快速收斂
        # u1：積分項，消除穩態誤差並保持輸出連續性
        w_sw = self.beta * np.sqrt(abs(s)) * np.sign(s) + self.u1

        # 更新積分項
        self.u1 += self.alpha * np.sign(s) * self.dt

        # 等效控制（basic model 設為 0）
        w_eq = 0.0

        return w_eq + w_sw
