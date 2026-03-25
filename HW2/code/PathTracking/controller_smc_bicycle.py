import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerSMCBicycle(Controller):
    """
    Sliding Mode Controller (SMC) for Bicycle Kinematic Model
    控制輸出：前輪轉向角 delta (deg)
    狀態：x, y, yaw, delta, v

    滑動面：s = theta_e + lambda * e
      - e      : 橫向誤差 (cross-track error)，投影到路徑法線
      - theta_e: 航向誤差 (heading error)，路徑切線方向 vs 車頭方向
      - lambda > 0: 滑動面斜率，越大越積極修正橫向誤差

    控制律：delta = delta_eq + delta_sw
      - delta_eq : 等效控制（由自行車運動學推導）
      - delta_sw : 切換控制 = -k * tanh(s / phi)

    自行車模型運動學（線性化）：
      de/dt      = v * sin(theta_e) ≈ v * theta_e
      d(theta_e)/dt = -v/l * delta + v * kappa
        其中 kappa 為路徑曲率（path[i,3]），l 為軸距
    """
    def __init__(self, model,
                 lambda_=0.5,   # 滑動面斜率
                 k=0.3,         # 切換增益（弧度，k 太大→飽和在±30°→繞圈）
                 phi=0.3):      # 邊界層厚度（弧度，phi 太小→tanh 立即飽和）
        self.path = None
        self.lambda_ = lambda_
        self.k = k
        self.phi = phi
        self.dt = model.dt
        self.l = model.l        # 軸距 (wheelbase)
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None

        # Extract State
        x, y, yaw, delta, _ = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(
            self.path, (x, y), self.current_idx, lookahead=50
        )
        self.current_idx = min_idx
        target = self.path[min_idx].copy()
        target[2] = utils.angle_norm(target[2])

        # 轉換為弧度計算
        target_yaw_rad = np.deg2rad(target[2])
        yaw_rad = np.deg2rad(yaw)

        # 橫向誤差 e（投影到路徑法線方向，左正右負）
        e = -(x - target[0]) * np.sin(target_yaw_rad) + (y - target[1]) * np.cos(target_yaw_rad)

        # 航向誤差 theta_e（弧度，正規化到 [-pi, pi]）
        theta_e = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi

        # 路徑曲率 kappa（若路徑有第 4 欄曲率資訊則使用，否則設為 0）
        kappa = target[3] if len(target) > 3 else 0.0

        # 滑動面 s = lambda * e - theta_e
        # 方向驗證（對照 PID bicycle）：
        #   theta_e > 0（target 偏左，需向左 → delta > 0）
        #   → s = -theta_e < 0 → delta_sw = -k*tanh(s) > 0 ✓
        #   e > 0（car 在路徑右側，需向左 → delta > 0? 需再確認 e 方向）
        s = self.lambda_ * e - theta_e

        # 等效控制 delta_eq：由 ds/dt = 0 重新推導（滑動面換了，符號也換）
        # ds/dt = lambda * d(e)/dt - d(theta_e)/dt
        #       = lambda * (-v*sin(theta_e)) - (v*kappa - v/l * delta)
        # 令 ds/dt = 0 → delta_eq = l * (kappa + lambda * sin(theta_e))
        delta_eq = self.l * (kappa + self.lambda_ * np.sin(theta_e))

        # 切換控制 delta_sw：-k * tanh(s/phi)
        delta_sw = -self.k * np.tanh(s / self.phi)

        # 合成控制輸出（弧度），轉換為角度
        delta_rad = delta_eq + delta_sw
        next_delta = np.rad2deg(delta_rad)

        # 限制轉向角範圍 ±30 度
        next_delta = np.clip(next_delta, -30, 30)

        return next_delta
