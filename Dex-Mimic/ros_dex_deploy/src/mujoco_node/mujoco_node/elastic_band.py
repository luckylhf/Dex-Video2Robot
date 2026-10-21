import mujoco
import numpy as np
from typing import Literal


class ElasticBand:

    def __init__(self, property: Literal['spring', 'elastic'] = 'elastic', enable=True):
        # 刚度按本体重量反推：m=65.8 kg → 645 N；悬挂点 z=3 m；带子挂在 waist_pitch_link，
        # 该点比 pelvis 高 0.162 m。k=300 时 L=0 只有 300×(3-1.262)=521 N < 645 N，
        # 带子根本吊不住本体（脚一直贴地），解带时又瞬间抽走 521 N，策略会直接摔。
        # k=450 时 L=0 提供 782 N，平衡点在 pelvis≈1.40 m（脚离地约 0.4 m），
        # 每按一次 ▼DOWN 平衡降低 0.1 m，4 次后脚底贴地，与 GUI 的操作顺序一致。
        self.stiffness = 450
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = enable
        self.property = property

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        if distance < 1e-6:
            # 挂点与悬挂点重合时方向无定义，直接返回零力以避免除零。
            return np.zeros(3)
        direction = δx / distance
        v = np.dot(dx, direction)
        if self.property =="spring":
            f = (self.stiffness *
                (distance - self.length) - self.damping * v) * direction
        else:
            # 只有当距离大于自然长度时才产生拉力
            if distance > self.length:
                f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
            else:
                # 距离小于等于自然长度时不产生力
                f = np.zeros_like(direction)
        # f[0]=0
        # f[1]=0
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_UP:
            self.length -= 0.1
        if key == glfw.KEY_DOWN:
            self.length += 0.1
        if key == glfw.KEY_ESCAPE:
            self.enable = not self.enable
