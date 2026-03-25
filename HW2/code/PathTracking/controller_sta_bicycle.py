import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerSTABicycle(Controller):
    """
    Super Twisting Algorithm (STA) + SMC for Bicycle Kinematic Model
    控制輸出：前輪轉向角 delta (deg)

    傳統 SMC 切換項：delta_sw = -k * tanh(s/phi)  → 仍有輕微抖振
    STA 改良：輸出完全連續，抖振幾乎消除，且保有有限時間收斂

    STA 控制律（離散時間改良版）：
        delta = delta_eq + delta_sw
        delta_sw = -β * tanh(s/φ) + u1    ← tanh 取代 √|s|*sign(s)，消除離散抖振
        u̇1      = -α * sign(s)            ← 積分項，跨時步累積，確保有限時間收斂

    注意：理論 STA 使用 √|s|*sign(s)，但在 dt=0.05 的離散系統中，
    當 |s| 很小時 sign(s) 每步正負交替，振幅反比 |s| 更大 → 嚴重抖振。
    改用 tanh 後輸出有界且連續，保留積分項仍具 STA 收斂特性。

    滑動面：s = λ * e - θ_e
        θ_e > 0（需正 delta，向左轉）→ s = -θ_e < 0 → delta_sw > 0 ✓
        e   > 0（car 偏右）         → s > 0            → delta_sw < 0 ✓

    等效控制推導（由 ds/dt = 0）：
        ds/dt = λ * de/dt - dθ_e/dt
              = λ * (-v*sin(θ_e)) - (v*κ - v/l * delta) = 0
        → delta_eq = l * (κ + λ * sin(θ_e))
    """
    def __init__(self, model,
                 lambda_=0.5,  # 滑動面斜率（與 SMC 一致，使 s 有足夠大小讓 tanh 有效）
                 alpha=0.15,   # 積分增益（加快累積速度，約 0.65s 可達 0.1 rad 修正量）
                 beta=0.3,     # tanh 增益（與 SMC 的 k 相同，已驗證可行）
                 phi=0.3):     # tanh 邊界層（與 SMC 相同，控制切換平滑度）
        self.path = None
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.dt = model.dt
        self.l = model.l
        self.u1 = 0.0           # STA 積分項（弧度，跨時步保存）
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.u1 = 0.0
        self.current_idx = 0

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        if self.path is None:
            print("No path !!")
            return None

        x, y, yaw, _, _ = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        if self.current_idx >= len(self.path) - 3:
            self.u1 = 0.0
            return 0.0

        min_idx, _ = utils.search_nearest_local(
            self.path, (x, y), self.current_idx, lookahead=50
        )
        self.current_idx = min_idx
        target = self.path[min_idx].copy()
        target[2] = utils.angle_norm(target[2])

        target_yaw_rad = np.deg2rad(target[2])
        yaw_rad = np.deg2rad(yaw)

        # 橫向誤差 e（投影到路徑法線，與 LQR bicycle 定義相同）
        e = -(x - target[0]) * np.sin(target_yaw_rad) + (y - target[1]) * np.cos(target_yaw_rad)

        # 航向誤差 θ_e（弧度，正規化到 [-π, π]）
        theta_e = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi

        # 路徑曲率 κ
        kappa = target[3] if len(target) > 3 else 0.0

        # 滑動面 s = λ * e - θ_e
        s = self.lambda_ * e - theta_e

        # 等效控制（補償已知動態）
        delta_eq = self.l * (kappa + self.lambda_ * np.sin(theta_e))

        # STA 切換控制（離散改良：tanh 取代 √|s|*sign(s)）
        # tanh(s/φ)：|s| 大時趨近 ±1（有界），|s| 小時線性（連續），徹底消除離散抖振
        # u1 積分項：緩慢累積，補償持續性擾動，確保有限時間收斂（STA 核心特性）
        delta_sw = -self.beta * np.tanh(s / self.phi) + self.u1

        # 更新積分項 u̇1 = -α * sign(s)
        # sign(s) 在此處是合理的：u1 是積分，即使輸入不連續，輸出仍緩慢平滑變化
        self.u1 += -self.alpha * np.sign(s) * self.dt
        # 限幅防止 windup（u1 最大提供與 beta 相當的額外修正量）
        self.u1 = np.clip(self.u1, -self.beta, self.beta)
        # 緩慢衰減：防止 u1 在彎道大量累積後，出彎時帶著錯誤偏差繼續修正
        self.u1 *= 0.98

        # 合成並轉換為角度
        delta_rad = delta_eq + delta_sw
        next_delta = np.rad2deg(delta_rad)
        next_delta = np.clip(next_delta, -30, 30)

        return next_delta
