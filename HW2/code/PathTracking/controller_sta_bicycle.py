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

    STA 控制律：
        delta = delta_eq + delta_sw
        delta_sw = -β * √|s| * sign(s) + u1
        u̇1      = -α * sign(s)         ← 積分項，跨時步累積

    滑動面：s = λ * e - θ_e
        θ_e > 0（需正 delta，向左轉）→ s = -θ_e < 0 → delta_sw > 0 ✓
        e   > 0（car 偏右）         → s > 0            → delta_sw < 0 ✓

    等效控制推導（由 ds/dt = 0）：
        ds/dt = λ * de/dt - dθ_e/dt
              = λ * (-v*sin(θ_e)) - (v*κ - v/l * delta) = 0
        → delta_eq = l * (κ + λ * sin(θ_e))
    """
    def __init__(self, model,
                 lambda_=0.5,   # 滑動面斜率
                 alpha=0.05,    # 積分增益（rad/s²，太大→超調，太小→收斂慢）
                 beta=0.15):    # 平方根增益（rad，主要決定初期收斂速度）
        self.path = None
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
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

        x, y, yaw, _, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        if self.current_idx >= len(self.path) - 3:
            self.u1 = 0.0
            return 0.0

        min_idx, _ = utils.search_nearest_local(
            self.path, (x, y), self.current_idx, lookahead=100
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

        # Super Twisting 切換控制
        # -β/v * √|s| * sign(s)：速度越快切換量越小，避免高速時抖振放大
        v_safe = max(abs(v), 1.0)
        delta_sw = -(self.beta / v_safe) * np.sqrt(abs(s)) * np.sign(s) + self.u1

        # 更新積分項：改用 tanh 軟化 sign(s)，避免 s≈0 時符號頻繁跳動造成抖振
        self.u1 += -self.alpha * np.tanh(s / 0.1) * self.dt
        # 限幅防止積分累積過大（windup）
        self.u1 = np.clip(self.u1, -0.5, 0.5)

        # 合成並轉換為角度
        delta_rad = delta_eq + delta_sw
        next_delta = np.rad2deg(delta_rad)
        next_delta = np.clip(next_delta, -30, 30)

        return next_delta
