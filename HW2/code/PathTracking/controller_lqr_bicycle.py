import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            # Q 矩陣：狀態代價（越大代表越不容許該誤差存在）
            # R 矩陣：控制代價（越大代表越保守，轉向幅度越小）
            # 調參原則：Q[1,1] >> Q[0,0] → 優先修正航向（航向誤差不矯正，CTE 會持續增大）
            #           R 小 → 允許更大轉向，追蹤更積極；R 大 → 轉向平滑但慢
            self.Q[0,0] = 1.0    # 橫向誤差 e 的懲罰權重
            self.Q[1,1] = 10.0   # 航向誤差 theta_e 的懲罰權重（設大使控制器優先對齊路徑方向）
            self.R[0,0] = 0.1    # 控制輸入（轉向角 delta）的懲罰權重（設小允許更積極轉向）
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            # 狀態多了 delta（當前轉向角），控制輸入改為 delta_dot（轉向角速度）
            self.Q[0,0] = 1.0    # 橫向誤差 e 的懲罰權重
            self.Q[1,1] = 10.0   # 航向誤差 theta_e 的懲罰權重
            self.Q[2,2] = 1.0    # 轉向角 delta 的懲罰權重（避免過大的轉向角）
            self.R[0,0] = 1    # 控制輸入（delta_dot）的懲罰權重
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01):
        # 求解離散代數 Riccati 方程式（DARE）
        # 迭代收斂後得到最優代價矩陣 P，用以計算 LQR 增益 K
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None

        # Extract State
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
       # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        # 只在目前索引附近往前找，避免封閉路徑抓到後面的點
        min_idx, min_dist = utils.search_nearest_local(
        self.path, (x, y), self.current_idx, lookahead=50
        )
        self.current_idx = min_idx

        target = self.path[min_idx].copy()
        target[2] = utils.angle_norm(target[2])
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            # 將角度轉為弧度進行計算
            target_yaw_rad = np.deg2rad(target[2])
            yaw_rad = np.deg2rad(yaw)

            # 航向誤差（弧度，正規化到 [-pi, pi]）
            theta_e = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi

            # 橫向誤差：投影到路徑法線方向（路徑座標系下的 y 偏移）
            e = -(x - target[0]) * np.sin(target_yaw_rad) + (y - target[1]) * np.cos(target_yaw_rad)

            # 線性化自行車模型（狀態 = [e, theta_e]，控制 = delta）
            # 注意：此處 theta_e = path_yaw - vehicle_yaw，符號與標準定義相反
            # 正確離散化：
            #   e[k+1]       = e[k] - v*theta_e*dt   （theta_e>0 → 車頭偏右 → e 減少）
            #   theta_e[k+1] = theta_e[k] - v/l*delta*dt （delta>0 → vehicle_yaw 增加 → theta_e 減少）
            v_safe = max(abs(v), 0.1)
            A = np.array([[1.0, -v_safe * self.dt],
                          [0.0, 1.0]])
            B = np.array([[0.0],
                          [-v_safe * self.dt / self.l]])

            # 解 DARE 得 P，再算最優增益 K = (R + B'PB)^-1 * B'PA
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A

            # 最優控制律：u = -K * x_state（負號因為要讓誤差趨近 0）
            x_state = np.array([e, theta_e])
            u = -(K @ x_state)[0]  # 單位：弧度

            next_delta = np.rad2deg(u)
            # [end] TODO 4.4.1

        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            target_yaw_rad = np.deg2rad(target[2])
            yaw_rad = np.deg2rad(yaw)

            # 航向誤差（弧度）
            theta_e = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi

            # 橫向誤差
            e = -(x - target[0]) * np.sin(target_yaw_rad) + (y - target[1]) * np.cos(target_yaw_rad)

            delta_rad = np.deg2rad(delta)

            # 擴展狀態空間（狀態 = [e, theta_e, delta]，控制 = delta_dot）
            # 比 4.4.1 多了 delta 狀態，讓控制器考慮轉向角的平滑性
            # 符號與 4.4.1 相同（theta_e = path_yaw - vehicle_yaw，負號）
            v_safe = max(abs(v), 0.1)
            A = np.array([[1.0, -v_safe * self.dt, 0.0],
                          [0.0, 1.0,              -v_safe * self.dt / self.l],
                          [0.0, 0.0,               1.0]])
            B = np.array([[0.0],
                          [0.0],
                          [self.dt]])

            # 解 DARE 並計算最優增益 K
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A

            x_state = np.array([e, theta_e, delta_rad])
            delta_dot = -(K @ x_state)[0]  # 單位：rad/s

            # 將 delta_dot 積分得到新的轉向角
            next_delta = np.rad2deg(delta_rad + delta_dot * self.dt)
            next_delta = np.clip(next_delta, -30, 30)  # 限制 ±30度
            # [end] TODO 4.4.4

        return next_delta
