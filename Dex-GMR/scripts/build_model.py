"""从权威基础包派生锁头模型；只写 --output 及显式 --mimic-dir 指定的新包。"""
import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/dex')
parser.add_argument('--mimic-dir', type=Path)
args = parser.parse_args()
src, dst = args.source.resolve(), args.output.resolve()
for part in ['urdf', 'mjcf']:
    (dst / part).mkdir(parents=True, exist_ok=True)
for origin, target in [('meshes', 'meshes'), ('mujoco/meshes_convex', 'meshes_convex'), ('mujoco/meshes_simplify', 'meshes_visual')]:
    shutil.copytree(src / origin, dst / target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".DS_Store", "._*"))
shutil.copy2(src / 'LICENSE', dst / 'LICENSE')
shutil.copy2(src / 'urdf/tiangong3.urdf', dst / 'urdf/tiangong3_source.urdf')
urdf = ET.parse(src / 'urdf/tiangong3.urdf')
heads = {'head_yaw_joint', 'head_pitch_joint'}
for j in urdf.findall('joint'):
    if j.get('name') in heads:
        j.set('type', 'fixed')
        for tag in ['axis', 'limit', 'dynamics']:
            node = j.find(tag)
            if node is not None:
                j.remove(node)
for mesh in urdf.findall('.//mesh'):
    mesh.set('filename', '../meshes/' + Path(mesh.get('filename')).name)
urdf.write(dst / 'urdf/dex_29.urdf', encoding='utf-8', xml_declaration=True)

xml = ET.parse(src / 'mujoco/tiangong3_torq.xml')
root = xml.getroot()
root.set('model', 'tienkung_dex_29')
root.find('compiler').set('fusestatic', 'false')
for side in ['l', 'r']:
    name = f'ankle_roll_{side}_link'
    collisions = urdf.find(f"./link[@name='{name}']").findall('collision')
    geoms = [g for g in root.find(f".//body[@name='{name}']").findall('geom') if g.get('class') == 'collision']
    if len(collisions) != 7 or len(geoms) != 7:
        raise ValueError(f'{name}: expected seven source foot cylinders in URDF and MJCF')
    for collision, geom in zip(collisions, geoms):
        cylinder = collision.find('geometry/cylinder')
        if cylinder is None or geom.get('type') != 'cylinder':
            raise ValueError(f'{name}: source foot collisions must be cylinders')
        origin = collision.find('origin')
        axis = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()[:, 2]
        actual = Rotation.from_quat(np.fromstring(geom.get('quat'), sep=' '), scalar_first=True).as_matrix()[:, 2]
        if (not np.allclose(np.fromstring(origin.get('xyz'), sep=' '), np.fromstring(geom.get('pos'), sep=' '), atol=1e-9)
                or not np.allclose([float(cylinder.get('radius')), float(cylinder.get('length')) / 2],
                                   np.fromstring(geom.get('size'), sep=' '), atol=1e-9)
                or min(np.linalg.norm(axis - actual), np.linalg.norm(axis + actual)) > 1e-6):
            raise ValueError(f'{name}: source URDF/MJCF foot geometry mismatch')
ET.SubElement(root, 'option', timestep='0.002', gravity='0 0 -9.81')
asset = root.find('asset')
for mesh in list(asset.findall('mesh')):
    mesh.set('file', '../meshes_convex/' + Path(mesh.get('file')).name)
    visual = copy.deepcopy(mesh)
    visual.set('name', mesh.get('name') + '_visual')
    visual.set('file', '../meshes_visual/' + Path(mesh.get('file')).name)
    asset.append(visual)
links = {link.get('name'): link for link in urdf.findall('link')}
limits = {j.get('name'): j.find('limit').attrib for j in urdf.findall('joint') if j.get('type') == 'revolute'}
for body in root.findall('.//body'):
    link = links[body.get('name')]
    previous = body.find('inertial')
    if previous is not None:
        body.remove(previous)
    inertia = link.find('inertial')
    origin = inertia.find('origin')
    i = {k: float(v) for k, v in inertia.find('inertia').attrib.items()}
    tensor = np.array([[i['ixx'], i['ixy'], i['ixz']], [i['ixy'], i['iyy'], i['iyz']], [i['ixz'], i['iyz'], i['izz']]])
    rot = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    tensor = rot @ tensor @ rot.T
    full = tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]
    body.insert(0, ET.Element('inertial', pos=origin.get('xyz', '0 0 0'), mass=inertia.find('mass').get('value'), fullinertia=' '.join(format(v, '.16g') for v in full)))
    for j in list(body.findall('joint')):
        if j.get('name') in heads:
            body.remove(j)
        else:
            limit = limits[j.get('name')]
            j.set('range', limit['lower'] + ' ' + limit['upper'])
            j.set('actuatorfrcrange', '-' + limit['effort'] + ' ' + limit['effort'])
            j.set('damping', '0')
            j.set('frictionloss', '0')
    for geom in body.findall('geom'):
        if geom.get('class') == 'visual':
            geom.set('mesh', geom.get('mesh') + '_visual')
root.find('worldbody/geom').set('pos', '0 0 0')
pelvis = root.find(".//body[@name='pelvis']")
ET.SubElement(pelvis, 'site', name='imu', pos='0 0 0', size='0.005')
actuators = root.find('actuator')
for motor in list(actuators):
    actuators.remove(motor)
sensors = root.find('sensor')
if sensors is not None:
    root.remove(sensors)
sensors = ET.SubElement(root, 'sensor')
for name, limit in limits.items():
    ET.SubElement(actuators, 'motor', name='motor_' + name, joint=name, ctrllimited='true', ctrlrange='-' + limit['effort'] + ' ' + limit['effort'])
    ET.SubElement(sensors, 'jointpos', name=name + '_pos', joint=name)
    ET.SubElement(sensors, 'jointvel', name=name + '_vel', joint=name)
    ET.SubElement(sensors, 'actuatorfrc', name=name + '_torque', actuator='motor_' + name)
ET.SubElement(sensors, 'framequat', name='orientation', objtype='body', objname='pelvis')
ET.SubElement(sensors, 'framepos', name='position', objtype='body', objname='pelvis')
for tag, name in [('gyro', 'angular-velocity'), ('velocimeter', 'linear-velocity'), ('accelerometer', 'linear-acceleration'), ('magnetometer', 'magnetometer')]:
    ET.SubElement(sensors, tag, name=name, site='imu')
model_file = dst / 'mjcf/dex_29.xml'
ET.indent(xml, space='  ')
xml.write(model_file, encoding='utf-8', xml_declaration=True)
# Put the zero pose on z=0 using the actual collision geometry.
model = mujoco.MjModel.from_xml_path(str(model_file))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
lowest = np.inf
for g in range(model.ngeom):
    if 'ankle_roll_' not in model.body(int(model.geom_bodyid[g])).name or model.geom_group[g] != 1:
        continue
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half_len = model.geom_size[g, :2]
        axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
        z = data.geom_xpos[g, 2] - half_len * abs(axis_z) - radius * np.sqrt(max(0., 1 - axis_z**2))
    elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[g]
        vertices = model.mesh_vert[model.mesh_vertadr[mesh_id]:model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]]
        world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        z = float(world[:, 2].min())
    else:
        raise ValueError('Unsupported foot collision geometry')
    lowest = min(lowest, float(z))
assert np.isfinite(lowest)
pos = np.fromstring(pelvis.get('pos'), sep=' ')
pos[2] -= lowest
pelvis.set('pos', ' '.join(format(x, '.12g') for x in pos))
xml.write(model_file, encoding='utf-8', xml_declaration=True)
manifest = {
    'robot': 'tienkung_dex_29', 'head': 'fixed_at_urdf_zero', 'hands': False,
    'source_urdf_sha256': hashlib.sha256((dst / 'urdf/tiangong3_source.urdf').read_bytes()).hexdigest(),
    'urdf_sha256': hashlib.sha256((dst / 'urdf/dex_29.urdf').read_bytes()).hexdigest(),
    'model_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest(),
    'joint_names': list(limits), 'joint_limits': limits,
    'source': 'Tienkung_Dex_Model_All/urdf/tiangong3.urdf',
}
(dst / 'model.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
if args.mimic_dir:
    for peer in [args.mimic_dir / 'source/whole_body_tracking/whole_body_tracking/assets/dex_evt', args.mimic_dir / 'ros_dex_deploy/src/mujoco_node/resources/dex']:
        shutil.copytree(dst, peer, dirs_exist_ok=True)
print(json.dumps({'model': str(dst), 'nq': model.nq, 'nv': model.nv, 'nu': model.nu, 'mass_kg': float(model.body_mass.sum()), 'sha256': manifest['model_sha256']}, indent=2))
