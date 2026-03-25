import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, model,
                 # TODO 4.3.1: Tune Stanley Gain
                 # kp：橫向誤差增益；速度越大，arctan 內的值越小（自然抑制高速震盪）
                 kp=0.0000000000005):
        self.path = None
        self.kp = kp
        self.l = model.l
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
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Stanley 控制器以「前輪位置」為基準計算誤差
        front_x = x + self.l*np.cos(np.deg2rad(yaw))
        front_y = y + self.l*np.sin(np.deg2rad(yaw))
        # 前輪速度（由後輪速度透過幾何關係換算）
        vf = v / np.cos(np.deg2rad(delta)) if np.cos(np.deg2rad(delta)) != 0 else v

        min_idx, min_dist = utils.search_nearest_local(self.path, (front_x,front_y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # TODO 4.3.1: Stanley Control for Bicycle Kinematic Model
        # 航向誤差（heading error）：路徑切線方向 - 車頭方向（deg，已正規化）
        theta_e = utils.angle_norm(target[2] - yaw)

        # 前輪橫向誤差（cross-track error at front axle）
        theta_target = np.rad2deg(np.arctan2(target[1] - front_y, target[0] - front_x))
        theta_err = theta_target - yaw
        e = min_dist * np.sin(np.deg2rad(theta_err))

        # Stanley 公式：delta = theta_e + arctan(k * e / vf)
        # 第一項修正航向，第二項修正橫向偏差（速度大時自動減弱）
        v_f_safe = max(vf, 0.001)
        next_delta = theta_e + np.rad2deg(np.arctan2(-self.kp * e, v_f_safe))
        # [end] TODO 4.3.1

        return next_delta
