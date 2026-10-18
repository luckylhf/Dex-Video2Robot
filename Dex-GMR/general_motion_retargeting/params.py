from pathlib import Path

HERE = Path(__file__).resolve().parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE.parent / "assets"
ROBOT_XML_DICT = {"dex_evt": ASSET_ROOT / "dex/mjcf/dex_29.xml"}
ROBOT_URDF_DICT = {"dex_evt": ASSET_ROOT / "dex/urdf/dex_29.urdf"}
IK_CONFIG_DICT = {
    "smplx": {"dex_evt": IK_CONFIG_ROOT / "smplx_todex_v3.json"},
    "bvh_lafan1": {"dex_evt": IK_CONFIG_ROOT / "bvh_lafan1_to_dex_evt.json"},
    "bvh_soma": {"dex_evt": IK_CONFIG_ROOT / "bvh_soma_to_dex_evt.json"},
}
ROBOT_BASE_DICT = {"dex_evt": "pelvis"}
VIEWER_CAM_DISTANCE_DICT = {"dex_evt": 4.0}
