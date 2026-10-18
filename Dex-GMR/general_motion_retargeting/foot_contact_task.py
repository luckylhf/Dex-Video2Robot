"""
自定义 mink Task：在脚着地时锁定水平位置，防止脚滑。

原理：
- Stance phase（脚着地）：记录接触点 (x,y)，error = current_xy - contact_xy，
  Jacobian 为脚部世界坐标系下的水平位置雅可比。高 cost 使得 IK 求解器
  优先保持脚的水平位置不变，而非跟踪可能漂移的动捕目标。
- Swing phase（脚离地）：error = [0, 0]，task 不产生任何约束，脚自由跟随动捕目标。

与 FootGroundContactLimit 的协同：
- FootGroundContactLimit (Limit):  不等式约束，防止脚穿透地面 (Z 方向)
- FootContactTask    (Task):      代价项，防止脚水平滑动 (X, Y 方向)
"""

import mujoco as mj
import numpy as np
from mink import Task


class FootContactTask(Task):
    """当脚与地面接触时锁定水平位置的自定义 IK task。

    在 stance phase 中，记录脚的接触点，并通过高权重代价项
    惩罚任何偏离该接触点的水平运动，从而消除脚滑效应。

    Attributes:
        body_id: MuJoCo body ID of the foot link.
        ground_height: World-frame Z coordinate of the ground plane.
        stance_threshold: Z distance below which the foot is considered in contact.
        hysteresis: Extra Z margin for exiting stance (prevents oscillation).
        _in_stance: Whether this foot is currently in stance phase.
        _contact_xy: World-frame (x, y) position where the foot first touched down.
    """

    def __init__(
        self,
        model: mj.MjModel,
        body_name: str,
        ground_height: float = 0.0,
        cost: float = 500.0,
        gain: float = 0.95,
        lm_damping: float = 0.0,
        stance_threshold: float = 0.04,
        hysteresis: float = 0.02,
    ):
        """初始化 FootContactTask。

        Args:
            model: MuJoCo model.
            body_name: 脚部 body 的名称（例如 "ankle_roll_l_link"）。
            ground_height: 世界坐标系下的地面 Z 高度（米）。
            cost: 接触约束的代价权重，应远大于 FrameTask 的位置权重。
            gain: Task gain，取值范围 [0, 1]。
            lm_damping: Levenberg-Marquardt 阻尼系数。
            stance_threshold: 脚进入 stance 的最大离地高度（米）。
            hysteresis: stance→swing 切换的迟滞余量（米）。
        """
        super().__init__(
            cost=np.array([cost, cost], dtype=float),
            gain=gain,
            lm_damping=lm_damping,
        )
        self.model = model
        self.body_name = body_name
        self.body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model")

        self.ground_height = ground_height
        self.stance_threshold = stance_threshold
        self.hysteresis = hysteresis

        self._in_stance: bool = False
        self._contact_xy: np.ndarray | None = None  # shape (2,) in world frame

    # ── public properties ───────────────────────────────────────────────────

    @property
    def in_stance(self) -> bool:
        """当前是否处于 stance phase。"""
        return self._in_stance

    @property
    def contact_point(self) -> np.ndarray | None:
        """当前接触点 (x, y) 世界坐标，仅在 stance phase 中有意义。"""
        return self._contact_xy.copy() if self._contact_xy is not None else None

    # ── stance detection ────────────────────────────────────────────────────

    def update_stance_state(self, foot_z: float) -> None:
        """根据脚的高度更新 stance/swing 状态。

        应在每次 IK solve 之前调用。

        Args:
            foot_z: 当前脚在世界坐标系下的 Z 坐标。
        """
        if not self._in_stance:
            # swing → stance: 脚高度低于 (ground + threshold)
            if foot_z < self.ground_height + self.stance_threshold:
                self._in_stance = True
                # 注意：contact_xy 在 update_contact_point() 中单独设置，
                # 由调用方根据策略（动捕目标/机器人实际位置）决定。
        else:
            # stance → swing: 脚高度高于 (ground + threshold + hysteresis)
            if foot_z > self.ground_height + self.stance_threshold + self.hysteresis:
                self._in_stance = False
                self._contact_xy = None

    def set_contact_point(self, xy: np.ndarray) -> None:
        """设置接触点位置。

        调用方负责决定用动捕目标位置还是机器人实际位置。
        仅在 stance entry 时调用一次。

        Args:
            xy: 世界坐标系下的 (x, y) 接触点。
        """
        self._contact_xy = np.asarray(xy, dtype=float)[:2].copy()

    def reset(self) -> None:
        """重置状态（用于 motion loop 重新开始时）。"""
        self._in_stance = False
        self._contact_xy = None

    # ── mink Task interface ─────────────────────────────────────────────────

    def compute_error(self, configuration) -> np.ndarray:
        """计算任务误差 e(q)。

        - swing phase: error = [0, 0]，任务不产生约束。
        - stance phase: error = current_foot_xy - contact_xy，
          将脚拉回接触点位置。

        Args:
            configuration: mink.Configuration 对象。

        Returns:
            2D 误差向量 [ex, ey]。
        """
        if not self._in_stance or self._contact_xy is None:
            return np.zeros(2, dtype=float)

        foot_xy = configuration.data.xpos[self.body_id][:2]
        return foot_xy - self._contact_xy

    def compute_jacobian(self, configuration) -> np.ndarray:
        """计算世界坐标系下脚部水平位置的雅可比 J(q) ∈ R^{2×nv}。

        始终计算完整雅可比（即使 swing phase 中 error=0），
        因为 mink 在构建 QP 时使用 J 与 error 的乘积。

        Args:
            configuration: mink.Configuration 对象。

        Returns:
            2×nv 雅可比矩阵（仅 x, y 行）。
        """
        model = configuration.model
        data = configuration.data
        nv = model.nv

        jacp = np.zeros((3, nv), dtype=float)
        jacr = np.zeros((3, nv), dtype=float)
        # 计算 body origin 处的世界坐标系位置雅可比
        mj.mj_jacBody(model, data, jacp, jacr, self.body_id)
        return jacp[:2, :]  # 仅水平 (x, y) 行
