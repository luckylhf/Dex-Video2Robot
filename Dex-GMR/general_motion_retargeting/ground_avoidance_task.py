import numpy as np
from mink import Task

class GroundAvoidanceTask(Task):
    """
    Soft constraint: enforce body.z >= min_height

    error = min_height - current_z   (目标是 error <= 0)
    Jacobian = - (row of frame Jacobian corresponding to z axis)
    """

    def __init__(self, model, body_name, min_height=0.0, weight=1.0, gain=0.2):
        super().__init__(
            cost=np.array([weight]),
            gain=gain,
            lm_damping=0.0,
        )
        self.model = model
        self.body_name = body_name
        self.min_height = min_height

    def compute_error(self, configuration) -> np.ndarray:
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.body_name, "body"
        )
        current_z = transform_frame_to_world.translation()[2]

        # 定义 error = min_height - current_z
        error = -self.min_height + current_z
        if error >= 0.0:
            error = 0
        # if self.body_name == "wrist_roll_r_link":
        #     print(f"[Ground Task] {self.body_name} current_z = {current_z}, error = {self.min_height - current_z}")
        return np.array([error])

    def compute_jacobian(self, configuration) -> np.ndarray:
        J = configuration.get_frame_jacobian(self.body_name, "body")
        # print("J.shape", J.shape)
        # print("J (first rows):\n", J[:4,:5]) 
        Jz = J[2:3, :]  # shape: (1, nv)
        return Jz
