import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO 2.3.1: Bicycle Kinematic Model
        x, y, yaw = state.x, state.y, state.yaw
        v = state.v + cstate.a * self.dt
        x = x + v * np.cos(np.deg2rad(yaw)) * self.dt
        y = y + v * np.sin(np.deg2rad(yaw)) * self.dt
        w = np.rad2deg(v / self.l * np.tan(np.deg2rad(cstate.delta)))
        yaw = (yaw + w * self.dt) % 360
        # [end] TODO 2.3.1
        state_next = State(x, y, yaw, v, w)
        return state_next
