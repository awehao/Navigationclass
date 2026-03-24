import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, model,
                 # TODO 4.2.1: Tune Pure Pursuit Gain
                 # kp：前視距離的速度係數（速度越快，前看越遠，高速時更平穩）
                 # Lfc：最小前視距離（低速保底，避免 Ld 過小導致震盪）
                 kp=0.05, Lfc=1):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.dt = model.dt
        self.l = model.l  # 軸距（前後輪距離）
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None

        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # 找最近路徑點，更新當前路徑索引
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx

        # TODO 4.2.1: Pure Pursuit Control for Bicycle Kinematic Model
        # 前視距離 Ld = kp*v + Lfc
        # 速度越快看越遠（kp*v），同時保證低速時不低於 Lfc
        Ld = self.kp * v + self.Lfc

        # 從最近點往前掃描，找第一個與車子距離 >= Ld 的路徑點作為前視目標
        target_idx = min_idx
        for i in range(min_idx, len(self.path)):
            dist = np.sqrt((self.path[i, 0] - x)**2 + (self.path[i, 1] - y)**2)
            if dist >= Ld:
                target_idx = i
                break
        # 若路徑剩餘段全部比 Ld 近（接近終點），直接用最後一個點
        target = self.path[target_idx]

        # alpha：車頭方向（yaw）到前視目標點的夾角（弧度）
        # 正值 = 目標在車子左前方，負值 = 右前方
        alpha = np.arctan2(target[1] - y, target[0] - x) - np.deg2rad(yaw)
        # 正規化到 [-pi, pi]，避免角度跨越 ±180° 時符號錯誤
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Pure Pursuit 幾何轉向公式（由自行車運動學推導）：
        # delta = arctan(2 * L * sin(alpha) / Ld)
        # L = 軸距，alpha = 前視角，Ld = 前視距離
        next_delta = np.rad2deg(np.arctan2(2.0 * self.l * np.sin(alpha), Ld))
        # [end] TODO 4.2.1
        return next_delta
