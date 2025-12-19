from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from neoskidrl.config import load_config
from neoskidrl.rewards import aggregate_reward, compute_reward_terms
import importlib.resources as ir

try:
    import mujoco
except Exception as e:
    mujoco = None


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # yaw from quaternion (w, x, y, z) in Z-up frame
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _ray_aabb_2d(origin_xy: np.ndarray, dir_xy: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> float:
    # Slab intersection in 2D. Returns distance to entry or +inf if no hit.
    inv = 1.0 / np.where(np.abs(dir_xy) < 1e-9, 1e-9, dir_xy)
    t1 = (box_min - origin_xy) * inv
    t2 = (box_max - origin_xy) * inv
    tmin = np.maximum(np.minimum(t1, t2)[0], np.minimum(t1, t2)[1])
    tmax = np.minimum(np.maximum(t1, t2)[0], np.maximum(t1, t2)[1])
    if tmax < 0 or tmin > tmax:
        return float("inf")
    return float(max(tmin, 0.0))


class NeoSkidNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, config_path: str | Path | None = None, render_mode: str | None = None):
        super().__init__()
        if mujoco is None:
            raise ImportError("mujoco python package not found. Activate venv and `pip install mujoco`.")

        self.cfg = load_config(config_path)
        self.render_mode = render_mode

        # Load MJCF
        with ir.as_file(ir.files("neoskidrl.models").joinpath("omnicar_skid.xml")) as p:
            self.model = mujoco.MjModel.from_xml_path(str(p))
        self.data = mujoco.MjData(self.model)

        # Config
        self.dt = float(self.cfg["env"]["dt"])
        self.frame_skip = int(self.cfg["env"].get("frame_skip", 1))
        self.max_steps = int(self.cfg["env"]["episode_sec"] / (self.dt * self.frame_skip))
        self.model.opt.timestep = self.dt

        self.rays = int(self.cfg["sensors"]["lidar"]["rays"])
        self.lidar_range = float(self.cfg["sensors"]["lidar"]["range_m"])

        self.v_max = float(self.cfg["limits"]["v_max"])
        self.w_max = float(self.cfg["limits"]["w_max"])
        self.wheel_vel_max = float(self.cfg["limits"]["wheel_vel_max"])

        self.track = float(self.cfg["robot"]["track_m"])
        self.wheel_r = float(self.cfg["robot"]["wheel_radius_m"])

        self.action_space_mode = self.cfg["control"]["action_space"]
        if self.action_space_mode == "v_w":
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                           high=np.array([1.0, 1.0], dtype=np.float32),
                                           dtype=np.float32)
        elif self.action_space_mode == "wheel_velocities":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown control.action_space: {self.action_space_mode}")

        # obs: lidar + goal_rel(3) + speed(2)
        obs_dim = self.rays + 3 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.drive_mode = self.cfg["robot"]["drive"]["mode"]
        self.enforce_no_side_slip = bool(self.cfg["robot"]["drive"].get("enforce_no_side_slip", True))
        self.lateral_damping = float(self.cfg["robot"]["drive"].get("lateral_damping", 1.0))
        if self.drive_mode not in ("skid_steer", "mecanum"):
            raise ValueError(f"Unknown robot.drive.mode: {self.drive_mode}")
        if self.drive_mode == "mecanum":
            warnings.warn("robot.drive.mode=mecanum is a placeholder; falling back to skid-steer kinematics.")

        self.cylinder_prob = float(self.cfg["world"]["obstacles"].get("cylinder_prob", 0.4))

        # IDs
        self.base_qpos_adr = 0  # freejoint is first
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")

        # Robot geom ids
        self.robot_geoms = set()
        for name in ["chassis", "wheel_fl_geom", "wheel_fr_geom", "wheel_rl_geom", "wheel_rr_geom"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.robot_geoms.add(gid)

        # Obstacle ids: mocap bodies + geom ids
        self.max_obs = int(self.cfg["world"]["obstacles"]["max_count"])
        self.obs_body_ids = []
        self.obs_geom_ids = []
        self.obs_mocap_ids = []
        for i in range(self.max_obs):
            bname = f"obs_{i:02d}"
            gname = f"obs_{i:02d}_geom"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            self.obs_body_ids.append(bid)
            self.obs_geom_ids.append(gid)
            self.obs_mocap_ids.append(int(self.model.body_mocapid[bid]))

        # episode state
        self._prev_dist = None
        self._prev_action = None
        self._stuck_counter = 0
        self._stuck_limit_steps = int(self.cfg["task"]["failure"]["stuck_sec"] / (self.dt * self.frame_skip))

        # simple renderer (rgb_array)
        self._renderer = None
        cam_cfg = self.cfg["sensors"]["cameras"]
        if self.render_mode in ("rgb_array", "depth_array"):
            try:
                if self.render_mode == "rgb_array":
                    width = int(cam_cfg["rgb"]["width"])
                    height = int(cam_cfg["rgb"]["height"])
                else:
                    width = int(cam_cfg["depth"]["width"])
                    height = int(cam_cfg["depth"]["height"])
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            except Exception:
                self._renderer = None

        seed = int(self.cfg["env"].get("seed", 0))
        self.np_random = np.random.default_rng(seed)

    def _set_mocap_obstacle(
        self, idx: int, x: float, y: float, z: float, sx: float, sy: float, sz: float, shape: str = "box"
    ):
        mid = self.obs_mocap_ids[idx]
        self.data.mocap_pos[mid] = np.array([x, y, z], dtype=np.float64)
        self.data.mocap_quat[mid] = np.array([1, 0, 0, 0], dtype=np.float64)
        # resize geom in-place (xy half-sizes + z half-size)
        gid = self.obs_geom_ids[idx]
        if shape == "box":
            self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[gid] = np.array([sx, sy, sz], dtype=np.float64)
        elif shape == "cylinder":
            self.model.geom_type[gid] = mujoco.mjtGeom.mjGEOM_CYLINDER
            self.model.geom_size[gid] = np.array([sx, sz, 0.0], dtype=np.float64)
        else:
            raise ValueError(f"Unknown obstacle shape: {shape}")

    def _hide_obstacle(self, idx: int):
        self._set_mocap_obstacle(idx, 100.0, 100.0, 0.0, 0.01, 0.01, 0.01, shape="box")

    def _sample_xy(self, arena_x: float, arena_y: float, margin: float) -> Tuple[float, float]:
        x = self.np_random.uniform(-arena_x/2 + margin, arena_x/2 - margin)
        y = self.np_random.uniform(-arena_y/2 + margin, arena_y/2 - margin)
        return float(x), float(y)

    def _reset_robot_pose(self):
        # base freejoint qpos: [x,y,z,qw,qx,qy,qz]
        x, y = self._sample_xy(*self.cfg["world"]["arena_m"], margin=0.8)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        z = self.wheel_r + 0.001
        qw = math.cos(yaw/2); qz = math.sin(yaw/2)
        qx = 0.0; qy = 0.0
        self.data.qpos[0:7] = np.array([x, y, z, qw, qx, qy, qz], dtype=np.float64)
        self.data.qvel[:] = 0.0

    def _set_goal(self, x: float, y: float, yaw: float):
        # Goal is stored in env; the site is in world coordinates for visualization only.
        self.goal_xy = np.array([x, y], dtype=np.float64)
        self.goal_yaw = float(yaw)
        if self.goal_site_id != -1:
            goal_h = float(self.cfg["world"]["goal"].get("height_m", 0.01))
            self.model.site_pos[self.goal_site_id] = np.array([x, y, goal_h], dtype=np.float64)

    def _get_base_pose(self) -> Tuple[np.ndarray, float]:
        x, y, z = self.data.qpos[0:3]
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = _quat_to_yaw(qw, qx, qy, qz)
        return np.array([x, y], dtype=np.float64), float(yaw)

    def _update_goal_site_visual(self):
        # Move goal site visually by moving the site position inside base frame approximation:
        # easiest: do nothing fancy, we keep site at (2,0) in base. It's just a marker.
        # If you want accurate goal marker later, we can attach it to a separate mocap body.
        pass

    def _place_obstacles_and_goal(self):
        arena_x, arena_y = map(float, self.cfg["world"]["arena_m"])
        margin = float(self.cfg["world"]["obstacles"]["margin_m"])
        count_lo, count_hi = self.cfg["world"]["obstacles"]["count_range"]
        n_obs = int(self.np_random.integers(int(count_lo), int(count_hi) + 1))
        goal_cfg = self.cfg["world"].get("goal", {})
        goal_fixed = bool(goal_cfg.get("fixed", False))
        goal_xy = None
        goal_yaw = 0.0
        if goal_fixed:
            pos = goal_cfg.get("pos_m", [2.0, 0.0])
            goal_xy = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
            goal_yaw = math.radians(float(goal_cfg.get("yaw_deg", 0.0)))
            goal_clear = float(goal_cfg.get("clearance_m", 0.6))

        sx_lo, sx_hi = map(float, self.cfg["world"]["obstacles"]["size_xy_range_m"])
        h = float(self.cfg["world"]["obstacles"]["height_m"])
        sz = h / 2.0

        # hide all
        for i in range(self.max_obs):
            self._hide_obstacle(i)

        # sample robot start first
        self._reset_robot_pose()
        start_xy, _ = self._get_base_pose()

        # place obstacles (axis-aligned boxes)
        obs_xy = []
        for i in range(n_obs):
            for _tries in range(200):
                x, y = self._sample_xy(arena_x, arena_y, margin)
                shape = "cylinder" if self.np_random.random() < self.cylinder_prob else "box"
                if shape == "box":
                    sx = float(self.np_random.uniform(sx_lo/2, sx_hi/2))
                    sy = float(self.np_random.uniform(sx_lo/2, sx_hi/2))
                else:
                    sx = float(self.np_random.uniform(sx_lo/2, sx_hi/2))
                    sy = sx
                if goal_xy is not None and np.linalg.norm(np.array([x, y]) - goal_xy) < goal_clear:
                    continue
                if np.linalg.norm(np.array([x, y]) - start_xy) < 0.8:
                    continue
                ok = True
                for (ox, oy, osx, osy) in obs_xy:
                    if abs(x-ox) < (sx+osx+0.2) and abs(y-oy) < (sy+osy+0.2):
                        ok = False
                        break
                if not ok:
                    continue
                obs_xy.append((x, y, sx, sy))
                self._set_mocap_obstacle(i, x, y, sz, sx, sy, sz, shape=shape)
                break

        # fixed goal
        if goal_fixed and goal_xy is not None:
            self._set_goal(goal_xy[0], goal_xy[1], goal_yaw)
            return

        # sample goal away from obstacles
        for _tries in range(500):
            gx, gy = self._sample_xy(arena_x, arena_y, margin=0.8)
            g = np.array([gx, gy], dtype=np.float64)
            if np.linalg.norm(g - start_xy) < 2.0:
                continue
            ok = True
            for (ox, oy, osx, osy) in obs_xy:
                if abs(gx-ox) < (osx+0.4) and abs(gy-oy) < (osy+0.4):
                    ok = False
                    break
            if ok:
                gyaw = float(self.np_random.uniform(-math.pi, math.pi))
                self._set_goal(gx, gy, gyaw)
                return

        # fallback
        self._set_goal(2.0, 0.0, 0.0)

    def _compute_lidar(self) -> np.ndarray:
        base_xy, yaw = self._get_base_pose()
        angles = np.linspace(-math.pi, math.pi, self.rays, endpoint=False)
        dists = np.full((self.rays,), self.lidar_range, dtype=np.float32)

        # Build obstacle AABBs from current mocap pos + geom sizes
        aabbs = []
        for i in range(self.max_obs):
            mid = self.obs_mocap_ids[i]
            ox, oy, _ = self.data.mocap_pos[mid]
            # hidden?
            if abs(ox) > 50 or abs(oy) > 50:
                continue
            gid = self.obs_geom_ids[i]
            gtype = int(self.model.geom_type[gid])
            gsize = self.model.geom_size[gid]
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                sx = sy = float(gsize[0])
            else:
                sx = float(gsize[0])
                sy = float(gsize[1])
            box_min = np.array([ox - sx, oy - sy], dtype=np.float64)
            box_max = np.array([ox + sx, oy + sy], dtype=np.float64)
            aabbs.append((box_min, box_max))

        for k, a in enumerate(angles):
            th = yaw + float(a)
            dir_xy = np.array([math.cos(th), math.sin(th)], dtype=np.float64)
            best = float("inf")
            for (bmin, bmax) in aabbs:
                t = _ray_aabb_2d(base_xy, dir_xy, bmin, bmax)
                if t < best:
                    best = t
            if best < float("inf"):
                dists[k] = np.float32(min(best, self.lidar_range))

        # normalize to [0,1]
        return (dists / self.lidar_range).astype(np.float32)

    def _collision_happened(self) -> bool:
        # Check contacts between robot geoms and obstacle geoms
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in self.robot_geoms and g2 in self.obs_geom_ids) or (g2 in self.robot_geoms and g1 in self.obs_geom_ids):
                return True
        return False

    def _goal_obs(self) -> np.ndarray:
        base_xy, yaw = self._get_base_pose()
        rel = self.goal_xy - base_xy
        # rotate into robot frame
        c = math.cos(-yaw); s = math.sin(-yaw)
        dx = c * rel[0] - s * rel[1]
        dy = s * rel[0] + c * rel[1]
        dyaw = (self.goal_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        return np.array([dx, dy, dyaw], dtype=np.float32)

    def _speed_obs(self) -> np.ndarray:
        # Use base linear speed approx from qvel
        # For a freejoint: qvel[0:3]=linear vel, qvel[3:6]=angular vel
        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        wz = float(self.data.qvel[5])
        v = math.sqrt(vx*vx + vy*vy)
        return np.array([v, wz], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        lidar = self._compute_lidar()
        goal = self._goal_obs()
        spd = self._speed_obs()
        return np.concatenate([lidar, goal, spd]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(int(seed))

        mujoco.mj_resetData(self.model, self.data)
        self._place_obstacles_and_goal()

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        base_xy, _ = self._get_base_pose()
        self._prev_dist = float(np.linalg.norm(self.goal_xy - base_xy))
        self._prev_action = np.zeros((self.action_space.shape[0],), dtype=np.float32)
        self._stuck_counter = 0

        obs = self._get_obs()
        info = {"goal_xy": self.goal_xy.copy(), "goal_yaw": self.goal_yaw}
        return obs, info

    def _apply_action(self, action: np.ndarray):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # optional rate limit
        if self.cfg["control"]["rate_limit"]["enabled"]:
            md = float(self.cfg["control"]["rate_limit"]["max_delta_per_step"])
            action = np.clip(action, self._prev_action - md, self._prev_action + md)
            action = np.clip(action, -1.0, 1.0)

        if self.action_space_mode == "v_w":
            v = float(action[0]) * self.v_max
            w = float(action[1]) * self.w_max

            v_l = v - w * (self.track / 2.0)
            v_r = v + w * (self.track / 2.0)

            w_l = np.clip(v_l / self.wheel_r, -self.wheel_vel_max, self.wheel_vel_max)
            w_r = np.clip(v_r / self.wheel_r, -self.wheel_vel_max, self.wheel_vel_max)

            # ctrl order matches actuators: fl, fr, rl, rr
            self.data.ctrl[0] = w_l
            self.data.ctrl[1] = w_r
            self.data.ctrl[2] = w_l
            self.data.ctrl[3] = w_r

        else:  # wheel_velocities
            ws = action * self.wheel_vel_max
            ws = np.clip(ws, -self.wheel_vel_max, self.wheel_vel_max)
            self.data.ctrl[:] = ws

        self._prev_action = action

    def step(self, action):
        self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._apply_drive_constraints()

        self.steps += 1

        base_xy, yaw = self._get_base_pose()
        dist = float(np.linalg.norm(self.goal_xy - base_xy))
        dyaw = (self.goal_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        collided = self._collision_happened()

        # success
        pos_ok = dist <= float(self.cfg["task"]["success"]["pos_tol_m"])
        yaw_ok = abs(dyaw) <= math.radians(float(self.cfg["task"]["success"]["yaw_tol_deg"]))
        spd = self._speed_obs()[0]
        stop_ok = spd <= float(self.cfg["task"]["success"]["stop_speed_mps"])
        success = bool(pos_ok and yaw_ok and stop_ok)
        terms = compute_reward_terms(
            dist=dist,
            prev_dist=self._prev_dist,
            action=self._prev_action,
            collided=collided,
            success=success,
        )
        r = aggregate_reward(terms, self.cfg)

        # stuck check
        progress = float(terms.get("progress", 0.0))
        min_prog = float(self.cfg["task"]["failure"]["min_progress_m"])
        if progress < min_prog * 0.01:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        stuck = self._stuck_counter >= self._stuck_limit_steps

        terminated = False
        truncated = False

        if self.cfg["task"]["failure"]["collision"] and collided:
            terminated = True
        if stuck:
            terminated = True
        if success:
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        self._prev_dist = dist
        obs = self._get_obs()
        info = {
            "dist": dist,
            "dyaw": dyaw,
            "success": success,
            "collision": collided,
            "stuck": stuck,
            "steps": self.steps,
            "reward_terms": terms,
            "reward_total": float(r),
        }
        return obs, float(r), terminated, truncated, info

    def render(self):
        if self.render_mode not in ("rgb_array", "depth_array") or self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera="track")
        if self.render_mode == "depth_array":
            return self._renderer.render(depth=True)
        return self._renderer.render()

    def close(self):
        self._renderer = None

    def _apply_drive_constraints(self):
        if self.drive_mode != "skid_steer":
            return
        if not self.enforce_no_side_slip:
            return

        _xy, yaw = self._get_base_pose()
        vx_world = float(self.data.qvel[0])
        vy_world = float(self.data.qvel[1])
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        vx_body = c * vx_world - s * vy_world
        vy_body = s * vx_world + c * vy_world
        damp = max(0.0, min(1.0, self.lateral_damping))
        vy_body = vy_body * (1.0 - damp)
        c2 = math.cos(yaw)
        s2 = math.sin(yaw)
        self.data.qvel[0] = c2 * vx_body - s2 * vy_body
        self.data.qvel[1] = s2 * vx_body + c2 * vy_body
