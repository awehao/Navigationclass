"""
compare.py - 同時跑最多 5 個控制器並排比較
用法：python compare.py -t 400mRunningTrack -c smc sta pid pure_pursuit stanley
      python compare.py -t Silverstone -c smc sta
      python compare.py -t 400mRunningTrack -c lqr_sa lqr_sav smc
控制器名稱：pid, pure_pursuit, stanley, lqr_sa, lqr_sav, smc, sta
  lqr_sa  = LQR (steering_angle)
  lqr_sav = LQR (steering_angular_velocity)
"""
import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from trajectory_generator import natural_cubic_spline, adaptive_sampling, generate_speed_profile

# ──────────────────────────────────────────────
# 各控制器顏色（BGR）
# ──────────────────────────────────────────────
COLOR = {
    "pid":          ( 50, 200,  50),   # 綠
    "pure_pursuit": (200, 200,   0),   # 青
    "stanley":      ( 50,  50, 220),   # 藍
    "lqr_sa":       (  0, 165, 255),   # 橘（steering_angle）
    "lqr_sav":      (  0, 100, 200),   # 深橘（steering_angular_velocity）
    "smc":          (  0, 200, 200),   # 黃
    "sta":          (200,   0, 200),   # 紫
}

_render_scale = 1.0

# ──────────────────────────────────────────────
# 載入賽道
# ──────────────────────────────────────────────
def load_track(track_name, map_w=2000, map_h=2000):
    global _render_scale
    data = np.loadtxt(f"tracks/{track_name}.csv", delimiter=',', skiprows=1)
    raw_x, raw_y = data[:,0], data[:,1]

    tw = max(1e-5, raw_x.max() - raw_x.min())
    th = max(1e-5, raw_y.max() - raw_y.min())
    _render_scale = min((map_w - 100) / tw, (map_h - 100) / th)

    cx = (raw_x.max() + raw_x.min()) / 2
    cy = (raw_y.max() + raw_y.min()) / 2
    sx = (raw_x - cx) + (map_w/2) / _render_scale
    sy = (raw_y - cy) + (map_h/2) / _render_scale

    t = np.linspace(0, 1, len(sx))
    t_path = np.linspace(0, 1, 2000)
    path_x = natural_cubic_spline(t, sx, t_path)
    path_y = natural_cubic_spline(t, sy, t_path)

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
    p     = np.vstack((path_x, path_y, path_yaw, k, v_ref)).T
    return w_pts, p

# ──────────────────────────────────────────────
# 建立單一控制器組合
# ──────────────────────────────────────────────
def build_agent(ctrl_name, way_points):
    from Simulation.simulator_bicycle import SimulatorBicycle
    from PathTracking.long_controller_pid import PIDLongController

    sim = SimulatorBicycle()
    sim.render_scale = _render_scale
    lc = PIDLongController(model=sim.model, a_range=sim.a_range)

    if ctrl_name == "pid":
        from PathTracking.controller_pid_bicycle import ControllerPIDBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "pure_pursuit":
        from PathTracking.controller_pure_pursuit_bicycle import ControllerPurePursuitBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "stanley":
        from PathTracking.controller_stanley_bicycle import ControllerStanleyBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "lqr_sa":
        from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as C
        ctrl = C(model=sim.model, control_state="steering_angle")
    elif ctrl_name == "lqr_sav":
        from PathTracking.controller_lqr_bicycle import ControllerLQRBicycle as C
        ctrl = C(model=sim.model, control_state="steering_angular_velocity")
    elif ctrl_name == "smc":
        from PathTracking.controller_smc_bicycle import ControllerSMCBicycle as C
        ctrl = C(model=sim.model)
    elif ctrl_name == "sta":
        from PathTracking.controller_sta_bicycle import ControllerSTABicycle as C
        ctrl = C(model=sim.model)
    else:
        raise ValueError(f"未知控制器：{ctrl_name}")

    ctrl.set_path(way_points)
    lc.set_path(way_points)
    return {"sim": sim, "ctrl": ctrl, "lc": lc,
            "cmd": ControlState("bicycle", None, None),
            "traj": [], "cte_hist": [], "finished": False}

def reset_agent(agent, way_points, start_pose):
    agent["sim"].init_pose(start_pose)
    agent["ctrl"].set_path(way_points)
    agent["lc"].set_path(way_points)
    agent["cmd"] = ControlState("bicycle", None, None)
    agent["traj"].clear()
    agent["cte_hist"].clear()
    agent["finished"] = False

# ──────────────────────────────────────────────
# 單步執行
# ──────────────────────────────────────────────
def step_agent(agent):
    sim, ctrl, lc = agent["sim"], agent["ctrl"], agent["lc"]
    sim.step(agent["cmd"])
    s = sim.state
    info = {"x": s.x, "y": s.y, "yaw": s.yaw,
            "v": s.v, "delta": sim.cstate.delta}
    next_a, _ = lc.feedback(info)
    info["v"] += next_a * sim.model.dt
    next_delta = ctrl.feedback(info)
    agent["cmd"] = ControlState("bicycle", next_a, next_delta)
    agent["traj"].append((s.x, s.y))

def nearest_cte(path, x, y):
    d = np.sqrt((path[:,0]-x)**2 + (path[:,1]-y)**2)
    idx = int(d.argmin())
    return float(d[idx]), idx

# ──────────────────────────────────────────────
# 繪製 minimap
# ──────────────────────────────────────────────
def draw_minimap(path, agents, ctrl_names, mm_w=700, mm_h=400):
    mm = np.ones((mm_h, mm_w, 3), dtype=np.uint8) * 25
    pad = 25

    x_min, x_max = path[:,0].min(), path[:,0].max()
    y_min, y_max = path[:,1].min(), path[:,1].max()

    def w2m(wx, wy):
        px = int((wx - x_min) / (x_max - x_min + 1e-6) * (mm_w - 2*pad) + pad)
        py = int((wy - y_min) / (y_max - y_min + 1e-6) * (mm_h - 2*pad) + pad)
        return (px, py)

    # 參考路徑
    for i in range(len(path)-1):
        cv2.line(mm, w2m(path[i,0], path[i,1]),
                     w2m(path[i+1,0], path[i+1,1]), (70,70,70), 1)

    # 各控制器軌跡
    for name, agent in zip(ctrl_names, agents):
        color = COLOR[name]
        traj = agent["traj"]
        for i in range(1, len(traj)):
            cv2.line(mm, w2m(*traj[i-1]), w2m(*traj[i]), color, 2)
        if traj:
            cv2.circle(mm, w2m(*traj[-1]), 6, color, -1)

    # 圖例
    for i, (name, agent) in enumerate(zip(ctrl_names, agents)):
        avg = np.mean(agent["cte_hist"]) if agent["cte_hist"] else 0.0
        suffix = " [DONE]" if agent["finished"] else ""
        cv2.putText(mm, f"{name}: {avg:.2f}px{suffix}", (8, 18 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR[name], 1)

    cv2.putText(mm, "r:reset  ESC:quit", (8, mm_h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
    return mm

# ──────────────────────────────────────────────
# 即時 CTE 柱狀面板
# ──────────────────────────────────────────────
def draw_bar_panel(agents, ctrl_names, panel_w=300, panel_h=400):
    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 20
    n = len(ctrl_names)
    avgs = [np.mean(a["cte_hist"]) if a["cte_hist"] else 0.0 for a in agents]
    max_cte = max(max(avgs), 1.0)

    bar_area_h = panel_h - 80
    bar_w = max(20, (panel_w - 20) // n - 10)
    spacing = (panel_w - 20) // n

    cv2.putText(panel, "Avg CTE (px)", (panel_w//2 - 60, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    for i, (name, avg) in enumerate(zip(ctrl_names, avgs)):
        color = COLOR[name]
        bh = int(avg / max_cte * bar_area_h)
        x0 = 10 + i * spacing
        y_base = panel_h - 45
        cv2.rectangle(panel, (x0, y_base - bh), (x0 + bar_w, y_base), color, -1)
        cv2.putText(panel, f"{avg:.1f}", (x0, y_base + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        # 短名
        short = name[:3]
        cv2.putText(panel, short, (x0, panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return panel

# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--track", default="400mRunningTrack",
                        choices=['400mRunningTrack','1000mStraight','Silverstone','Suzuka','Monza'])
    parser.add_argument("-c", "--controllers", nargs="+",
                        default=["smc", "sta"],
                        choices=['pid','pure_pursuit','stanley','lqr_sa','lqr_sav','smc','sta'],
                        help="最多 5 個控制器，例如：-c smc sta lqr_sa lqr_sav")
    args = parser.parse_args()

    ctrl_names = args.controllers[:5]   # 最多 5 個
    print(f"賽道：{args.track}  |  控制器：{ctrl_names}")

    way_points, path = load_track(args.track)

    # 起始位姿
    start_yaw = np.rad2deg(np.arctan2(path[1,1]-path[0,1], path[1,0]-path[0,0]))
    start_pose = (path[0,0], path[0,1], start_yaw)

    # 建立所有 agent
    agents = [build_agent(name, way_points) for name in ctrl_names]
    for a in agents:
        a["sim"].init_pose(start_pose)

    cv2.namedWindow("Compare", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Compare", 1000, 400)

    while True:
        # 所有 agent 同步步進
        for a in agents:
            if a["finished"]:
                continue
            step_agent(a)
            if a["traj"]:
                x, y = a["traj"][-1]
                cte, idx = nearest_cte(path, x, y)
                a["cte_hist"].append(cte)
                # 最近路徑點進入最後 1% → 視為到達終點，停止計算
                if idx >= len(path) - len(path) // 100:
                    a["finished"] = True

        # 繪製
        mm    = draw_minimap(path, agents, ctrl_names, mm_w=700, mm_h=400)
        panel = draw_bar_panel(agents, ctrl_names, panel_w=300, panel_h=400)
        view  = np.hstack((mm, panel))

        cv2.imshow("Compare", view)
        k = cv2.waitKey(1)

        if k == ord('r'):
            for a in agents:
                reset_agent(a, way_points, start_pose)
        elif k == 27:
            break

    cv2.destroyAllWindows()

    # 最終排名
    print("\n─── 最終結果 ───")
    results = [(name, np.mean(a["cte_hist"])) for name, a in zip(ctrl_names, agents) if a["cte_hist"]]
    for rank, (name, cte) in enumerate(sorted(results, key=lambda x: x[1]), 1):
        print(f"  #{rank}  {name:15s}  Avg CTE = {cte:.3f} px")

if __name__ == "__main__":
    main()
