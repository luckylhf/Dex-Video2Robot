"""仿真侧坐标一致性回归测试（函数级，不启 ROS / MuJoCo）。

覆盖的问题：`motor_state_to_robot_state()` 原先照搬真机编码器整圈逻辑，
关节到达合法限位（shoulder_roll_l 3.334 > π、hip_yaw_r -4.451 < -π）时
`zero_cnt` 被锁存成 ∓1 且永不复位，导致反馈偏移 ±2π、命令反向偏移 ∓2π。

用法：
    source .venv/bin/activate && source /opt/ros/humble/setup.bash \
      && source install/setup.bash && python -m pytest test/ -q
或直接：
    python test/test_sim_joint_coordinates.py
"""
import math
import os
import sys
import types

import numpy as np

# 与 rl_control_node.py 一致：把 src/rl_control 加进 sys.path，再以 common.* 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.robot_interface import RobotInterfaceImpl  # noqa: E402

BASE = 6            # floating_base_dof
N = 29              # whole_motor_nums
SHOULDER_L = 12     # shoulder_roll_l_joint，限位上限 3.334 (> π)
HIP_YAW_R = 63      # hip_yaw_r_joint，限位下限 -4.451 (< -π)


class _IdMap:
    def __init__(self, mapping):
        self._m = mapping

    def get_index_by_id(self, motor_id):
        return self._m.get(int(motor_id), -1)


class _Logger:
    def warn(self, msg):
        pass

    def info(self, msg):
        pass


class _FakeNode:
    def get_logger(self):
        return _Logger()


def _make_impl(joint_id, joint_index):
    """绕过 __init__，只补齐被测函数用到的字段。"""
    obj = RobotInterfaceImpl.__new__(RobotInterfaceImpl)
    obj.sim = True
    obj.config_path = '<test>'
    obj.node = _FakeNode()
    obj.floating_base_dof = BASE
    obj.whole_joint_nums = BASE + N
    obj.id_map = _IdMap({joint_id: joint_index})

    obj.zero_pos = np.zeros(N)
    obj.motor_dir = np.ones(N)
    obj.zero_offset = np.zeros(N)
    obj.zero_cnt = np.zeros(N)
    obj.ct_scale = np.ones(N)
    obj.temp_q = np.empty(N)
    obj._temp_zero_cnt = np.empty(N)
    obj.disable_joints_ = False
    obj.joint_pos_threshold = math.pi

    rd = types.SimpleNamespace()
    rd.q_a_ = np.zeros(BASE + N)
    rd.q_dot_a_ = np.zeros(BASE + N)
    rd.tau_a_ = np.zeros(BASE + N)
    rd.q_a_last = np.zeros(BASE + N)
    rd.qdot_a_last = np.zeros(BASE + N)
    rd.tor_a_last = np.zeros(BASE + N)
    rd.temperature_a = np.zeros(N)
    rd.q_d_ = np.zeros(BASE + N)
    rd.q_dot_d_ = np.zeros(BASE + N)
    rd.tau_d_ = np.zeros(BASE + N)
    rd.joint_kp_p_ = np.zeros(N)
    rd.joint_kd_p_ = np.zeros(N)
    obj.robot_data_ = rd
    return obj


def _status(motor_id, pos):
    return types.SimpleNamespace(name=motor_id, pos=pos, speed=0.0,
                                 current=0.0, temperature=0.0)


def _feed(impl, motor_id, target_pos):
    """按小步长给到目标角（单帧跳变 > π 会被当作异常丢弃，这里模拟真实连续运动）。"""
    n_steps = max(2, int(math.ceil(abs(target_pos) / (math.pi / 2))) + 1)
    first = True
    for i in range(n_steps):
        p = target_pos * i / (n_steps - 1)
        impl.motor_state_to_robot_state(_status(motor_id, p), received_flag=not first)
        first = False
    return impl.robot_data_.q_a_[BASE + impl.id_map.get_index_by_id(motor_id)]


def test_shoulder_reaches_limit_without_wrap():
    impl = _make_impl(SHOULDER_L, 18)
    q = _feed(impl, SHOULDER_L, 3.334)
    assert abs(q - 3.334) < 1e-9, f'反馈应保持 3.334，实际 {q}'
    assert impl.zero_cnt[18] == 0.0, '仿真下 zero_cnt 不应被置位'

    # 命令侧同样必须恒等：目标 0.10 不应变成 6.383
    impl.robot_data_.q_d_[BASE + 18] = 0.10
    impl.convert_to_motor_commands()
    assert abs(impl.robot_data_.q_d_[BASE + 18] - 0.10) < 1e-9


def test_hip_yaw_reaches_limit_without_wrap():
    impl = _make_impl(HIP_YAW_R, 20)
    q = _feed(impl, HIP_YAW_R, -4.451)
    assert abs(q + 4.451) < 1e-9, f'反馈应保持 -4.451，实际 {q}'
    assert impl.zero_cnt[20] == 0.0

    impl.robot_data_.q_d_[BASE + 20] = 0.0
    impl.convert_to_motor_commands()
    assert abs(impl.robot_data_.q_d_[BASE + 20]) < 1e-9


def test_repeated_cycles_do_not_drift():
    """每周期重写目标并重复转换，输出不得逐周期漂移。"""
    impl = _make_impl(SHOULDER_L, 18)
    _feed(impl, SHOULDER_L, 3.334)
    outs = []
    for _ in range(100):
        impl.robot_data_.q_d_[BASE + 18] = 0.10
        impl.convert_to_motor_commands()
        outs.append(float(impl.robot_data_.q_d_[BASE + 18]))
    assert max(abs(o - 0.10) for o in outs) < 1e-9, f'出现漂移：{outs[:5]} ... {outs[-3:]}'


def test_validator_rejects_nan_and_non_identity():
    impl = _make_impl(SHOULDER_L, 18)
    impl.zero_pos = np.zeros(N)
    impl._validate_sim_coordinate_contract()      # 恒等，应通过

    impl.zero_pos = np.zeros(N)
    impl.zero_pos[0] = np.nan
    try:
        impl._validate_sim_coordinate_contract()
    except ValueError as e:
        assert 'NaN' in str(e)
    else:
        raise AssertionError('NaN 应被拒绝')

    impl.zero_pos = np.zeros(N)
    impl.motor_dir = np.ones(N)
    impl.motor_dir[0] = -1.0
    try:
        impl._validate_sim_coordinate_contract()
    except ValueError as e:
        assert 'motor_dir' in str(e)
    else:
        raise AssertionError('非恒等 motor_dir 应被拒绝')


if __name__ == '__main__':
    for fn in (test_shoulder_reaches_limit_without_wrap,
               test_hip_yaw_reaches_limit_without_wrap,
               test_repeated_cycles_do_not_drift,
               test_validator_rejects_nan_and_non_identity):
        fn()
        print(f'PASS  {fn.__name__}')
    print('ALL PASS')
