import mujoco as mj
import numpy as np

from mink import Constraint
from mink import Limit


class GeomDistanceLimit(Limit):
    """Collision avoidance using mj_geomDistance between geoms."""

    def __init__(
        self,
        model,
        geom_pairs,
        gain=0.2,
        minimum_distance_from_collisions=0.02,
        collision_detection_distance=0.12,
        bound_relaxation=0.0,
        debug_pairs=None,
    ):
        self.model = model
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.debug_pairs = {frozenset(pair) for pair in (debug_pairs or [])}
        self.geom_pairs = self._resolve_geom_pairs(geom_pairs)

    def _resolve_geom_pairs(self, geom_pairs):
        resolved = []
        for group_a, group_b in geom_pairs:
            ids_a = [self._geom_id(name) for name in self._ensure_list(group_a)]
            ids_b = [self._geom_id(name) for name in self._ensure_list(group_b)]
            resolved.append((ids_a, ids_b, self._ensure_list(group_a), self._ensure_list(group_b)))
        return resolved

    @staticmethod
    def _ensure_list(group):
        if isinstance(group, str):
            return [group]
        return list(group)

    def _geom_id(self, name):
        geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            geom_names = self._list_geom_names()
            print(f"[GMR][geom] available geoms ({len(geom_names)}): {geom_names}")
            raise ValueError(f"[GMR] geom '{name}' not found in model")
        return geom_id

    def _list_geom_names(self):
        """Return all geom names (for debugging)."""
        names = []
        for i in range(self.model.ngeom):
            nm = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
            if nm is None:
                nm = f"<unnamed_{i}>"
            names.append(nm)
        return names

    def print_all_geoms(self):
        """Print all geom names with ids (debug helper)."""
        names = self._list_geom_names()
        for i, nm in enumerate(names):
            print(f"[GMR][geom] id={i}: {nm}")

    @staticmethod
    def _geom_centers(data, geom1, geom2):
        return data.geom_xpos[geom1].copy(), data.geom_xpos[geom2].copy()

    def _geom_distance(self, model, data, geom1, geom2, distmax):
        fromto = np.zeros(6, dtype=float)
        try:
            dist = mj.mj_geomDistance(model, data, geom1, geom2, distmax, fromto)
        except TypeError:
            try:
                dist = mj.mj_geomDistance(model, data, geom1, geom2, fromto)
            except TypeError:
                dist = mj.mj_geomDistance(model, data, geom1, geom2)
        if isinstance(dist, (tuple, list)):
            if len(dist) >= 1:
                dist_val = dist[0]
            else:
                dist_val = dist
            if len(dist) >= 2:
                fromto = np.asarray(dist[1], dtype=float).reshape(-1)
            dist = dist_val
        if fromto.shape[0] >= 6:
            p1 = fromto[:3].copy()
            p2 = fromto[3:6].copy()
        else:
            p1, p2 = self._geom_centers(data, geom1, geom2)
        return float(dist), p1, p2

    def _point_jacobian(self, model, data, body_id, point):
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        try:
            mj.mj_jac(model, data, jacp, jacr, point, body_id)
            return jacp
        except AttributeError:
            mj.mj_jacBody(model, data, jacp, jacr, body_id)
            body_pos = data.xpos[body_id]
            offset = point - body_pos
            # Point Jacobian = body Jacobian + angular Jacobian x offset.
            return jacp + np.cross(jacr.T, offset).T

    def compute_qp_inequalities(self, configuration, dt):
        model = configuration.model
        data = configuration.data
        nv = model.nv

        dist_min = self.minimum_distance_from_collisions
        dist_max = self.collision_detection_distance
        if dist_max <= dist_min:
            dist_max = dist_min + 1e-6

        G_rows = []
        h_rows = []

        for ids_a, ids_b, names_a, names_b in self.geom_pairs:
            best = None
            for geom1 in ids_a:
                for geom2 in ids_b:
                    dist, p1, p2 = self._geom_distance(model, data, geom1, geom2, dist_max)
                    if best is None or dist < best[0]:
                        best = (dist, p1, p2, geom1, geom2)

            if best is None:
                continue

            dist, p1, p2, geom1, geom2 = best
            if dist >= dist_max:
                continue
            # Debug print for selected pairs.
            for na in names_a:
                for nb in names_b:
                    if frozenset((na, nb)) in self.debug_pairs:
                        print(f"[GMR][geom_dist] {na} vs {nb}: {dist:.4f} m")

            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                p1, p2 = self._geom_centers(data, geom1, geom2)
                direction = p2 - p1
                norm = np.linalg.norm(direction)
                if norm < 1e-9:
                    continue
            direction /= norm

            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            J1 = self._point_jacobian(model, data, body1, p1)
            J2 = self._point_jacobian(model, data, body2, p2)
            Jd = direction @ (J2 - J1)

            phi = (dist - dist_min) / (dist_max - dist_min)
            h = self.gain * phi + self.bound_relaxation
            G_rows.append(-Jd)
            h_rows.append(h)

        if not G_rows:
            return Constraint(np.zeros((0, nv), dtype=float), np.zeros((0,), dtype=float))

        G = np.vstack(G_rows)
        h = np.asarray(h_rows, dtype=float)
        return Constraint(G, h)

class FootGroundContactLimit(Limit):
    """
    更精确的足部地面接触约束
    使用足部几何体的实际接触点
    """
    
    def __init__(self, model, foot_geom_names, min_height=0.0, margin=0.0075, gain=1.5):
        self.model = model
        self.foot_geom_names = foot_geom_names if isinstance(foot_geom_names, list) else [foot_geom_names]
        self.min_height = min_height
        self.margin = margin
        self.gain = gain
        self.geom_ids = []
        self.geom_lowest_offsets = []  # cached local-frame offset to lowest-z point per geom
        for name in self.foot_geom_names:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                self.geom_ids.append(geom_id)
            else:
                print(f"警告: 未找到几何体 {name}")

    def _lowest_point(self, model, data, geom_id):
        """Return (current_lowest_z, world_offset_from_center) for this geom.
        Computes the actual lowest world-frame z point of the geom volume rather
        than just the center, so half-thick boxes/cylinders/meshes don't sink
        through the floor before the constraint activates.
        """
        gtype = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        center = data.geom_xpos[geom_id]
        xmat = data.geom_xmat[geom_id].reshape(3, 3)

        if gtype == mj.mjtGeom.mjGEOM_SPHERE:
            offset = np.array([0.0, 0.0, -float(size[0])])
            return center[2] + offset[2], offset
        if gtype == mj.mjtGeom.mjGEOM_BOX:
            # support point along world -z direction in box-local coords
            local_z = xmat.T @ np.array([0.0, 0.0, -1.0])
            sup_local = np.sign(local_z) * size[:3]
            offset = xmat @ sup_local
            offset[0] = 0.0
            offset[1] = 0.0
            offset[2] = float(offset[2])
            offset[2] = min(offset[2], 0.0)
            offset_world = xmat @ sup_local
            return center[2] + float(offset_world[2]), offset_world
        if gtype == mj.mjtGeom.mjGEOM_CYLINDER:
            radius, half_len = float(size[0]), float(size[1])
            axis = xmat[:, 2]
            down = np.array([0.0, 0.0, -1.0])
            axial = float(down @ axis)
            radial = down - axial * axis
            norm = np.linalg.norm(radial)
            offset = half_len * np.sign(axial) * axis
            if norm > 1e-12:
                offset += radius * radial / norm
            return float(center[2] + offset[2]), offset
        if gtype == mj.mjtGeom.mjGEOM_CAPSULE:
            radius = float(size[0])
            half_len = float(size[1])
            axis = xmat[:, 2]
            # disk plane normal projected onto -z gives radial extent toward the floor
            radial = radius * float(np.linalg.norm(np.cross(axis, np.array([0.0, 0.0, 1.0]))))
            axial = half_len * float(abs(axis[2]))
            return center[2] - radial - axial, np.array([0.0, 0.0, -radial - axial])
        if model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[geom_id])
            start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            vertices = model.mesh_vert[start:start + count]
            world = vertices @ xmat.T + center
            idx = int(np.argmin(world[:, 2]))
            return float(world[idx, 2]), xmat @ vertices[idx]

        # mesh and others fall back to AABB rbound (sphere bound) — conservative
        rbound = float(model.geom_rbound[geom_id]) if model.geom_rbound[geom_id] > 0 else 0.0
        return center[2] - rbound, np.array([0.0, 0.0, -rbound])

    def compute_qp_inequalities(self, configuration, dt):
        model = configuration.model
        data = configuration.data
        nv = model.nv

        G_rows = []
        h_rows = []

        for idx, geom_id in enumerate(self.geom_ids):
            current_z, world_offset = self._lowest_point(model, data, geom_id)
            # legacy hack carried over from the original implementation
            if self.foot_geom_names[idx] == "pelvis_collision":
                current_z -= 0.075

            if current_z > self.min_height + self.margin:
                continue

            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            # v_point = v_center + omega x offset = v_center - [offset]_x omega.
            mj.mj_jacGeom(model, data, jacp, jacr, geom_id)
            ox, oy, oz = world_offset
            # cross-product matrix [offset]_x acting on rotational Jacobian
            jacp_pt = jacp - np.array([
                [ 0.0, -oz,  oy],
                [ oz,  0.0, -ox],
                [-oy,  ox,  0.0],
            ]) @ jacr

            Jz = jacp_pt[2:3, :]

            effective_min = self.min_height + self.margin
            G = -Jz
            h = self.gain * (current_z - effective_min)

            G_rows.append(G)
            h_rows.append(h)
            
        if not G_rows:
            return Constraint(np.zeros((0, nv)), np.zeros((0,)))
        
        G = np.vstack(G_rows)
        h = np.array(h_rows)
        return Constraint(G, h)


class ObstacleContactLimit(Limit):
    def __init__(self, model, robot_geom_names, obstacle_geom_names, min_surface_dist=1e-4, margin=0.005, gain=1000.0):
        self.model = model
        self.robot_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in robot_geom_names]
        self.obstacle_geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in obstacle_geom_names]
        self.min_surface_dist = min_surface_dist
        self.margin = margin
        self.gain = gain

    def compute_qp_inequalities(self, configuration, dt):
        model = configuration.model
        data = configuration.data
        nv = model.nv

        G_rows = []
        h_rows = []
        for rgid in self.robot_geom_ids:
            for ogid in self.obstacle_geom_ids:
                p_robot = data.geom_xpos[rgid].copy()
                p_obs = data.geom_xpos[ogid].copy()

                size_robot = model.geom_size[rgid].copy()
                size_obs = model.geom_size[ogid].copy()
                size_obs = np.array([size_obs[2], size_obs[0], size_obs[1]])

                dist_vec = p_robot - p_obs
                dist = np.linalg.norm(dist_vec)
                if dist < 1e-8:
                    n = np.array([1, 0, 0])
                else:
                    n = dist_vec / dist

                r_robot = np.dot(np.abs(n), size_robot)
                r_obs = np.dot(np.abs(n), size_obs)

                print(p_robot, size_obs, ((p_obs[0] - size_obs[0], p_obs[0] + size_obs[0]), (p_obs[1] - size_obs[1], p_obs[1] + size_obs[1]), (p_obs[2] - size_obs[2], p_obs[2] + size_obs[2])))
                if p_robot[0] > p_obs[0] + size_obs[0] + self.margin:
                    continue
                if p_robot[0] < p_obs[0] - size_obs[0] - self.margin:
                    continue
                if p_robot[1] > p_obs[1] + size_obs[1] + self.margin:
                    continue
                if p_robot[1] < p_obs[1] - size_obs[1] - self.margin:
                    continue
                if p_robot[2] - r_robot > p_obs[2] + size_obs[2] + self.min_surface_dist + self.margin:
                    continue
                if p_robot[2] < p_obs[2]:
                    continue
                surface_dist = p_robot[2] - (p_obs[2] + size_obs[2]) - r_robot
                print("z surface_dist", rgid, p_robot, surface_dist, p_obs[2] + size_obs[2], r_robot)
                
                if surface_dist > self.min_surface_dist + self.margin:
                    continue
                # 获取接触点 Jacobian（更准确）
                jacp = np.zeros((3, nv))
                jacr = np.zeros((3, nv))
                mj.mj_jacGeom(model, data, jacp, jacr, rgid)
                
                # 取 z 方向的 Jacobian
                Jz = jacp[2:3, :]  # 几何体位置在z方向的雅可比

                # 约束：表面距离 >= min_surface_dist + margin
                G = -self.gain * Jz
                h = (surface_dist - (self.min_surface_dist + self.margin))# * abs((surface_dist - (self.min_surface_dist + self.margin)))
                G_rows.append(G)
                h_rows.append(h)

        if not G_rows:
            return Constraint(np.zeros((0, nv)), np.zeros((0,)))

        return Constraint(np.vstack(G_rows), np.array(h_rows))
