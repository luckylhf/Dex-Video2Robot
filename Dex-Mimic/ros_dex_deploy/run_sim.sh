#!/usr/bin/env bash
# 启动仿真栈：activate venv → source ROS → colcon build → source install → ros2 launch sim.launch.py
# 不开 set -u —— ROS 的 setup.bash 引用未定义变量。
set -eo pipefail
export PYTHONNOUSERSITE=1

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_DIR"

# ---- venv（先激活，让 colcon 用 venv python 生成 entry-point shebang） ----
VENV_DIR="${VENV_DIR:-$WS_DIR/.venv}"
if [[ ! -r "$VENV_DIR/bin/activate" ]]; then
  echo "[run_sim] 找不到 $VENV_DIR/bin/activate；请在本目录运行 bash init_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export MIMIC_BUNDLE_DIR="$(realpath "${MIMIC_BUNDLE_DIR:-$WS_DIR/src/rl_control/policy/mimic/out}")"
python "$WS_DIR/check_bundle.py"
echo "[run_sim] venv: $VIRTUAL_ENV ($(python -V))"

# 显式使用 venv Python 执行 colcon，不依赖 PATH 或 colcon 自身的 shebang。
venv_py="$VIRTUAL_ENV/bin/python"
colcon_bin="$VIRTUAL_ENV/bin/colcon"
if [[ ! -x "$colcon_bin" ]]; then
  echo "[run_sim] 缺少 $colcon_bin；请重跑 ./init_env.sh" >&2
  exit 1
fi
echo "[run_sim] colcon: $venv_py $colcon_bin"

# ---- ROS 发行版（22.04→humble；24.04→jazzy） ----
if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    case "${VERSION_ID:-}" in
      22.04) ROS_DISTRO=humble ;;
      24.04) ROS_DISTRO=jazzy ;;
      *) echo "[run_sim] 未识别的 Ubuntu 版本 $VERSION_ID，请显式 export ROS_DISTRO=" >&2; exit 1 ;;
    esac
  fi
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
if [[ ! -r "$ROS_SETUP" ]]; then
  echo "[run_sim] 找不到 $ROS_SETUP，请先安装 ros-${ROS_DISTRO}-desktop" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ROS_SETUP"

# ---- build & launch ----
# 本项目三个 ament_python 包的五个 console_scripts，必须全部生成。
entry_paths=(
  "$WS_DIR/install/rl_control/lib/rl_control/rl_control_node"
  "$WS_DIR/install/rl_control/lib/rl_control/rl_control_node_sim"
  "$WS_DIR/install/mujoco_node/lib/mujoco_node/mujoco_node"
  "$WS_DIR/install/sim_joy/lib/sim_joy/teleop_gui"
  "$WS_DIR/install/sim_joy/lib/sim_joy/teleop_joy"
)

check_entry_shebangs() {
  local f sb bad=0
  for f in "${entry_paths[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[run_sim] 缺少入口脚本: $f" >&2
      bad=1
      continue
    fi
    sb="$(head -n1 "$f")"
    if [[ "$sb" != "#!$venv_py" ]]; then
      echo "[run_sim] shebang 未指向 venv python: $f -> $sb" >&2
      bad=1
    fi
  done
  return "$bad"
}

# 构建前：已有 install/ 但入口缺失或 shebang 不对则清理重建
if [[ -d "$WS_DIR/install" ]] && ! check_entry_shebangs; then
  echo "[run_sim] 检测到 install/ 内入口缺失或 shebang 不匹配 $venv_py；清理 build/install/log 后重建"
  rm -rf "$WS_DIR/install" "$WS_DIR/build" "$WS_DIR/log"
fi

echo "[run_sim] colcon build (--symlink-install)"
# -DPYTHON_EXECUTABLE 只影响 CMake 包；ament_python 的 shebang 由启动 colcon 的解释器决定
"$venv_py" "$colcon_bin" build --symlink-install --cmake-args -DPYTHON_EXECUTABLE="$venv_py"

# 构建后：五个入口必须齐全且全部使用 venv Python。
if ! check_entry_shebangs; then
  echo "[run_sim] 构建产物入口缺失或 shebang 不指向 $venv_py；请检查构建日志" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$WS_DIR/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# 关掉 Python 的 stdout 块缓冲，避免 ros2 launch 下 print 延迟刷新
export PYTHONUNBUFFERED=1
echo "[run_sim] ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[run_sim] ros2 launch rl_control sim.launch.py $*"
MIMIC_DIAG=1 exec ros2 launch rl_control sim.launch.py "$@"
