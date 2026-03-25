"""
compare.py - 同時跑兩個控制器並排比較
用法：python compare.py -t 400mRunningTrack
      python compare.py -t Silverstone -c1 smc -c2 sta
"""
import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from trajectory_generator import natural_cubic_spline, adaptive_sampling, generate_speed_profile
from navigation_utils import pos_int

# ──────────────────────────────────────────────
# 控制器顏色設定（BGR）
# ──────────────────────────────────────────────
COLOR = {
    "pid":          (200,  50,  50),
    "pure_pursuit": ( 50, 200,  50),
    "stanley":      ( 50,  50, 200),
    "lqr":          (200, 150,   0),
    "smc":          (  0, 180, 255),   # 橘
    "sta":          (180,   0, 255),   # 紫
}

def build_simulator_and_controllers(ctrl_name, track_data):
    from Simulation.simulator_bicycle import SimulatorBicycle
    from PathTracking.long_controller_pid import PIDLongController

    sim = SimulatorBicycle()
    long_ctrl = PIDLongController(model=sim.model, a_range=sim.a_range)

    if ctrl_name == "pid":
        from PathTracking.controller_pid_bicycle import ControllerPIDBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "pure_pursuit":
        from PathTracking.controller_pure_pursuit_bicycle import ControllerPurePursuitBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "stanley":
        from PathTracking.controller_stanley_bicycle import ControllerStanleyBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "lqr":
        from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "smc":
        from PathTracking.controller_smc_bicycle import ControllerSMCBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "sta":
        from PathTracking.controller_sta_bicycle import ControllerSTABicycle as C
        ctrl = C(model=sim.model)
    else:
        raise ValueError(f"未知控制器：{ctrl_name}")

    way_points, path = track_data
    sim.render_scale = _render_scale  # 共用 render_scale
    ctrl.set_path(way_points)
    long_ctrl.set_path(way_points)
    return sim, ctrl, long_ctrl

# ──────────────────────────────────────────────
# 軌跡紀錄與 minimap 繪製
# ──────────────────────────────────────────────
def draw_minimap(path, way_points, traj1, traj2, name1, name2, color1, color2, cte1, cte2, map_w=400, map_h=250):
    mm = np.ones((map_h, map_w, 3), dtype=np.uint8) * 30  # 深色背景

    # 計算路徑 bounding box 用於縮放
    all_x = path[:, 0]
    all_y = path[:, 1]
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    pad = 20

    def world2mm(wx, wy):
        px = int((wx - x_min) / (x_max - x_min + 1e-6) * (map_w - 2*pad) + pad)
        py = int((wy - y_min) / (y_max - y_min + 1e-6) * (map_h - 2*pad) + pad)
        return px, py

    # 畫參考路徑（灰）
    for i in range(len(path) - 1):
        cv2.line(mm, world2mm(path[i,0], path[i,1]),
                     world2mm(path[i+1,0], path[i+1,1]), (80,80,80), 1)

    # 畫兩條軌跡
    for traj, color in [(traj1, color1), (traj2, color2)]:
        for i in range(1, len(traj)):
            cv2.line(mm, world2mm(*traj[i-1]), world2mm(*traj[i]), color, 2)

    # 畫目前位置（圓點）
    if traj1:
        cv2.circle(mm, world2mm(*traj1[-1]), 5, color1, -1)
    if traj2:
        cv2.circle(mm, world2mm(*traj2[-1]), 5, color2, -1)

    # 圖例與 CTE
    cv2.putText(mm, f"{name1}  AvgCTE:{cte1:.2f}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color1, 1)
    cv2.putText(mm, f"{name2}  AvgCTE:{cte2:.2f}", (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color2, 1)
    cv2.putText(mm, "r:reset  ESC:quit", (5, map_h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
    return mm

def nearest_cte(path, x, y):
    dists = np.sqrt((path[:,0]-x)**2 + (path[:,1]-y)**2)
    return float(dists.min())

def load_track(track_name, map_w=2000, map_h=2000):
    global _render_scale
    filename = f"tracks/{track_name}.csv"
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    raw_x, raw_y = data[:,0], data[:,1]

    margin = 5
    tw = max(1e-5, raw_x.max() - raw_x.min())
    th = max(1e-5, raw_y.max() - raw_y.min())
    sx = (map_w - 2*margin*10) / tw
    sy = (map_h - 2*margin*10) / th
    _render_scale = min(sx, sy)

    cx = (raw_x.max() + raw_x.min()) / 2
    cy = (raw_y.max() + raw_y.min()) / 2
    scaled_x = (raw_x - cx) + (map_w/2) / _render_scale
    scaled_y = (raw_y - cy) + (map_h/2) / _render_scale

    t_anchors = np.linspace(0, 1, len(scaled_x))
    t_path = np.linspace(0, 1, 2000)
    path_x = natural_cubic_spline(t_anchors, scaled_x, t_path)
    path_y = natural_cubic_spline(t_anchors, scaled_y, t_path)

    v_ref, k = generate_speed_profile(path_x, path_y, max_v=85.0,
                                       max_lat_acc=30, max_long_acc=12, max_long_dec=18)
    wp_x, wp_y, wp_v = adaptive_sampling(path_x, path_y, k, v_ref=v_ref,
                                          min_ds=2.0, max_ds=10.0, k_gain=200.0)

    wp_yaw = np.zeros_like(wp_x)
    for i in range(len(wp_x)-1):
        wp_yaw[i] = np.rad2deg(np.arctan2(wp_y[i+1]-wp_y[i], wp_x[i+1]-wp_x[i]))
    wp_yaw[-1] = wp_yaw[-2]

    path_yaw = np.zeros_like(path_x)
    for i in range(len(path_x)-1):
        path_yaw[i] = np.rad2deg(np.arctan2(path_y[i+1]-path_y[i], path_x[i+1]-path_x[i]))
    path_yaw[-1] = path_yaw[-2]

    wp_k = np.zeros_like(wp_x)
    w_pts = np.vstack((wp_x, wp_y, wp_yaw, wp_k, wp_v)).T
    p = np.vstack((path_x, path_y, path_yaw, k, v_ref)).T
    return w_pts, p

_render_scale = 1.0

def step_one(sim, ctrl, long_ctrl, cmd):
    """執行一步模擬，回傳新指令"""
    sim.step(cmd)
    state = sim.state
    info = {"x": state.x, "y": state.y, "yaw": state.yaw,
            "v": state.v, "delta": sim.cstate.delta}
    next_a, _ = long_ctrl.feedback(info)
    info["v"] = info["v"] + next_a * sim.model.dt
    next_delta = ctrl.feedback(info)
    return ControlState("bicycle", next_a, next_delta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--track", default="400mRunningTrack",
                        choices=['400mRunningTrack','1000mStraight','Silverstone','Suzuka','Monza'])
    parser.add_argument("-c1", "--ctrl1", default="smc",
                        choices=['pid','pure_pursuit','stanley','lqr','smc','sta'])
    parser.add_argument("-c2", "--ctrl2", default="sta",
                        choices=['pid','pure_pursuit','stanley','lqr','smc','sta'])
    args = parser.parse_args()

    print(f"載入賽道：{args.track}")
    track_data = load_track(args.track)
    way_points, path = track_data

    # 建立兩組模擬器
    sim1, ctrl1, lc1 = build_simulator_and_controllers(args.ctrl1, track_data)
    sim2, ctrl2, lc2 = build_simulator_and_controllers(args.ctrl2, track_data)

    # 起始位姿
    start_yaw = np.rad2deg(np.arctan2(path[1,1]-path[0,1], path[1,0]-path[0,0]))
    start_pose = (path[0,0], path[0,1], start_yaw)
    sim1.init_pose(start_pose)
    sim2.init_pose(start_pose)

    cmd1 = ControlState("bicycle", None, None)
    cmd2 = ControlState("bicycle", None, None)

    traj1, traj2 = [], []
    cte_hist1, cte_hist2 = [], []

    color1 = COLOR[args.ctrl1]
    color2 = COLOR[args.ctrl2]

    cv2.namedWindow("Compare", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Compare", 900, 300)

    while True:
        # 步進兩個模擬器
        cmd1 = step_one(sim1, ctrl1, lc1, cmd1)
        cmd2 = step_one(sim2, ctrl2, lc2, cmd2)

        # 記錄軌跡
        traj1.append((sim1.state.x, sim1.state.y))
        traj2.append((sim2.state.x, sim2.state.y))

        # 計算 CTE
        cte_hist1.append(nearest_cte(path, sim1.state.x, sim1.state.y))
        cte_hist2.append(nearest_cte(path, sim2.state.x, sim2.state.y))

        avg_cte1 = float(np.mean(cte_hist1)) if cte_hist1 else 0.0
        avg_cte2 = float(np.mean(cte_hist2)) if cte_hist2 else 0.0

        # 繪製 minimap（左）和即時 CTE 柱狀（右）
        mm = draw_minimap(path, way_points,
                          traj1, traj2,
                          args.ctrl1, args.ctrl2,
                          color1, color2,
                          avg_cte1, avg_cte2,
                          map_w=600, map_h=300)

        # 即時 CTE 數字面板（右側）
        panel = np.ones((300, 300, 3), dtype=np.uint8) * 20
        cv2.putText(panel, "Avg CTE", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        # 柱狀圖
        max_cte = max(avg_cte1, avg_cte2, 1.0)
        bar_max_h = 180
        bar_w = 80

        b1_h = int(avg_cte1 / max_cte * bar_max_h)
        b2_h = int(avg_cte2 / max_cte * bar_max_h)

        cv2.rectangle(panel, (40, 240-b1_h), (40+bar_w, 240), color1, -1)
        cv2.rectangle(panel, (170, 240-b2_h), (170+bar_w, 240), color2, -1)

        cv2.putText(panel, f"{avg_cte1:.2f}", (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color1, 1)
        cv2.putText(panel, f"{avg_cte2:.2f}", (170, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color2, 1)
        cv2.putText(panel, args.ctrl1, (40, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 1)
        cv2.putText(panel, args.ctrl2, (170, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1)

        view = np.hstack((mm, panel))
        cv2.imshow("Compare", view)

        k = cv2.waitKey(1)
        if k == ord('r'):
            # 重置兩組模擬
            sim1.init_pose(start_pose); sim2.init_pose(start_pose)
            ctrl1.set_path(way_points); ctrl2.set_path(way_points)
            lc1.set_path(way_points);  lc2.set_path(way_points)
            cmd1 = ControlState("bicycle", None, None)
            cmd2 = ControlState("bicycle", None, None)
            traj1.clear(); traj2.clear()
            cte_hist1.clear(); cte_hist2.clear()
        elif k == 27:
            break

    cv2.destroyAllWindows()
    print(f"\n最終結果：")
    print(f"  {args.ctrl1:15s} Avg CTE = {np.mean(cte_hist1):.3f} px")
    print(f"  {args.ctrl2:15s} Avg CTE = {np.mean(cte_hist2):.3f} px")

if __name__ == "__main__":
    main()
