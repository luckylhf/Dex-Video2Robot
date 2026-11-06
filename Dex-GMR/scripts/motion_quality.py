"""Backend-neutral, read-only kinematic quality report; never repairs motion."""
import hashlib
import io
import json
import pickle
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NumericUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='numpy' and name in ('ndarray','dtype'):return getattr(np,name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np.core.multiarray,name)
        if module in ('numpy.core.numeric','numpy._core.numeric') and name=='_frombuffer':return np.core.numeric._frombuffer
        raise ValueError(f'Unsupported motion PKL object: {module}.{name}')
    def persistent_load(self,pid):raise ValueError('External pickle references are unsupported')


def load_numeric(path):
    return NumericUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def validate_payload(payload,manifest):
    fps=float(payload['fps']);p=np.asarray(payload['root_pos']);q=np.asarray(payload['root_rot']);j=np.asarray(payload['dof_pos'])
    names=list(payload['joint_names']);n=len(p)
    if payload.get('model_sha256')!=manifest['model_sha256']:raise ValueError('Robot model SHA256 mismatch')
    if len(names)!=29 or len(set(names))!=29 or set(names)!=set(manifest['joint_names']):raise ValueError('Expected exactly the 29 named Dex joints')
    if payload.get('quat_order')!='xyzw':raise ValueError('Expected explicit xyzw root quaternions')
    if n<2 or p.shape!=(n,3) or q.shape!=(n,4) or j.shape!=(n,29):raise ValueError('Invalid motion array shape')
    if not np.isfinite(fps) or fps<=0 or not all(np.isfinite(a).all() for a in (p,q,j)):raise ValueError('Motion contains nonfinite values or invalid fps')
    if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-3):raise ValueError('Root quaternions are not normalized')
    return fps,p,q,j,names


def report(path,model_path,manifest,backend,upstream=None,payload=None,persist=True):
    import mujoco
    path=Path(path);model_path=Path(model_path)
    if sha(model_path)!=manifest['model_sha256']:raise ValueError('Model XML does not match manifest')
    if payload is None:payload=load_numeric(path)
    fps,p,q,j,names=validate_payload(payload,manifest)
    flags=[];limits=manifest['joint_limits']
    lower=np.array([float(limits[k]['lower']) for k in names]);upper=np.array([float(limits[k]['upper']) for k in names])
    speed=np.abs(np.diff(j,axis=0))*fps
    speed_limits=np.array([float(limits[k]['velocity']) for k in names])
    position_bad=(j<lower-1e-5)|(j>upper+1e-5)
    velocity_bad=speed>speed_limits+1e-5
    if position_bad.any():flags.append('joint_position_limit_exceeded')
    if velocity_bad.any():flags.append('nominal_urdf_velocity_limit_exceeded')
    root_speed=np.linalg.norm(np.diff(p,axis=0),axis=1)*fps
    root_angular=2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],axis=1)),0,1))*fps
    model=mujoco.MjModel.from_xml_path(str(model_path));data=mujoco.MjData(model)
    qadr=[model.joint(k).qposadr[0] for k in names]
    # Use the same visual foot meshes as the reference audit. Do not infer contact from height alone.
    feet=[]
    for side in ('l','r'):
        gid=model.geom('ankle_roll_'+side+'_link_visual').id;mid=model.geom_dataid[gid]
        if mid<0:raise ValueError('Expected mesh foot geometry')
        start=model.mesh_vertadr[mid];count=model.mesh_vertnum[mid]
        feet.append((gid,model.mesh_vert[start:start+count].copy()))
    heights=[]
    for i in range(len(p)):
        data.qpos[:3]=p[i];data.qpos[3:7]=q[i,[3,0,1,2]];data.qpos[qadr]=j[i]
        mujoco.mj_forward(model,data)
        heights.append([float((vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])[:,2].min()) for gid,vertices in feet])
    heights=np.asarray(heights);minimum=float(heights.min())
    if minimum<-.001:flags.append('visual_foot_penetration_over_1mm')
    if any(v for k,v in payload.get('ik_status_counts',{}).items() if k!='normal'):
        flags.append('ik_recovery_used')
    result=dict(schema='dex_motion_quality_v1',backend=backend,
        status='review' if flags else 'checked',scope='export_structure_limits_and_visual_foot_geometry',
        flags=flags,frames=len(p),fps=fps,model_sha256=manifest['model_sha256'],output_sha256=sha(path),
        metrics=dict(joint_limit_violations=int(position_bad.sum()),nominal_velocity_violations=int(velocity_bad.sum()),
            maximum_joint_velocity_rad_s=float(speed.max()),maximum_root_speed_m_s=float(root_speed.max()),
            maximum_root_angular_speed_rad_s=float(root_angular.max()),minimum_visual_foot_height_m=minimum,
            lower_visual_foot_height_percentiles_m=dict(zip(('p0','p50','p95','p100'),np.percentile(heights.min(1),[0,50,95,100]).tolist())),
            both_feet_above_5cm_fraction=float(np.mean(np.min(heights,axis=1)>.05))),
        thresholds=dict(penetration_m=.001,velocity='nominal URDF limits, independent of IK recovery settings'),
        unchecked=['source_pose_fidelity','support_contact_correctness','foot_sliding','full_body_self_collision','dynamics','real_robot'],
        warnings=['Airborne height is diagnostic only; genuine jumps must not be marked as failed grounding'],
        dynamic_validation=False,training_approved=False)
    if upstream is not None:
        result['backend_quality']=upstream
        result['flags']=sorted(set(flags+upstream.get('flags',[])))
        result['status']='review' if result['flags'] or upstream.get('status')!='accepted' else 'accepted'
        result['training_approved']=result['status']=='accepted'
        result['unchecked']=['dynamics','real_robot']
    if persist:
        path.with_suffix('.quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f"[quality] {result['status']}: {path.with_suffix('.quality.json')}; flags={result['flags']}",flush=True)
    return result


def report_npz(path,model_path,manifest,source_quality,checked_file,check_passed):
    """Audit the final arrays, including changes made by standing transitions."""
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:
        root=z['body_names'].tolist().index('pelvis')
        if str(z['quat_order'])!='wxyz':raise ValueError('Expected wxyz NPZ body quaternions')
        payload=dict(fps=float(z['fps']),root_pos=z['body_pos_w'][:,root],
                     root_rot=z['body_quat_w'][:,root][:,[1,2,3,0]],quat_order='xyzw',
                     dof_pos=z['joint_pos'],joint_names=z['joint_names'].tolist(),model_sha256=str(z['model_sha256']))
    result=report(path,model_path,manifest,source_quality['backend'],payload=payload,persist=False)
    result.update(scope='final_npz_geometry_limits_with_inherited_source_quality',source_quality=source_quality,
                  npz_check=dict(passed=check_passed,checked_file=str(checked_file)),
                  note='Source fidelity/contact judgments are inherited; final foot geometry and limits are recomputed after transitions')
    result['flags']=sorted(set(result['flags']+source_quality.get('flags',[])))
    if not check_passed:result['flags'].append('downstream_npz_check_failed')
    result['status']='review' if result['flags'] or source_quality['status']=='review' else 'checked'
    result['training_approved']=False
    result['unchecked'].append('added_transition_contact_and_source_fidelity')
    path.with_suffix('.quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"[quality NPZ] {result['status']}: {path.with_suffix('.quality.json')}; flags={result['flags']}",flush=True)
    return result
