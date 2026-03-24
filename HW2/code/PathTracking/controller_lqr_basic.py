import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBasic(Controller):
    def __init__(self, model, Q=np.eye(2), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0,0] = 100
        self.Q[1,1] = 5
        self.R = R*2000
        self.pe = 0
        self.pth_e = 0
        self.dt = model.dt
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.current_idx = 0
    
    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]
        
        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx

        target = self.path[min_idx]
        
        # Optional TODO: LQR Control for Basic Kinematic Model
        # You can implement this if you want to use LQR for basic kinematic model in F1 Challenge
        # Compute heading and cross-track errors
        target_yaw_rad = np.deg2rad(target[2])
        yaw_rad = np.deg2rad(yaw)
        theta_e = (target_yaw_rad - yaw_rad + np.pi) % (2 * np.pi) - np.pi

        theta_target = np.arctan2(target[1] - y, target[0] - x)
        theta_err = theta_target - yaw_rad
        e = min_dist * np.sin(theta_err)

        # Linearized unicycle: state=[e, theta_e], control=w (rad/s)
        v_safe = max(abs(v), 0.1)
        A = np.array([[1.0, v_safe * self.dt],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [-self.dt]])

        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A

        x_state = np.array([e, theta_e])
        u = -(K @ x_state)[0]  # rad/s
        next_w = np.rad2deg(u)
        return next_w
