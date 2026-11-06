import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from motion_quality import report,validate_payload,report_npz,sha


def payload():
    manifest=json.loads((ROOT/'assets/dex/model.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/dex/mjcf/dex_29.xml'))
    return dict(fps=50.,root_pos=np.tile(model.qpos0[:3],(8,1)),
                root_rot=np.tile(model.qpos0[[4,5,6,3]],(8,1)),dof_pos=np.tile(model.qpos0[7:],(8,1)),
                joint_names=manifest['joint_names'],quat_order='xyzw',model_sha256=manifest['model_sha256']),manifest


def test_airborne_height_is_not_automatically_failure(tmp_path):
    data,manifest=payload();data['root_pos'][:,2]+=.3
    path=tmp_path/'jump.pkl';path.write_bytes(pickle.dumps(data))
    result=report(path,ROOT/'assets/dex/mjcf/dex_29.xml',manifest,'gmr')
    assert result['metrics']['both_feet_above_5cm_fraction']==1
    assert result['status']=='checked' and result['training_approved'] is False
    assert 'support_contact_correctness' in result['unchecked']


def test_backend_review_is_preserved(tmp_path):
    data,manifest=payload();path=tmp_path/'motion.pkl';path.write_bytes(pickle.dumps(data))
    result=report(path,ROOT/'assets/dex/mjcf/dex_29.xml',manifest,'dex-retarget',
                  dict(status='review',flags=['sustained_arm_pose_mismatch']))
    assert result['status']=='review' and 'sustained_arm_pose_mismatch' in result['flags']
    assert not result['training_approved']


@pytest.mark.parametrize('key,value',[('model_sha256','wrong'),('quat_order','wxyz'),('joint_names',['wrong']*29)])
def test_export_contract_is_not_weakened(key,value):
    data,manifest=payload();data[key]=value
    with pytest.raises(ValueError):validate_payload(data,manifest)


def test_final_npz_is_audited_after_height_change(tmp_path):
    data,manifest=payload();data['root_pos'][:,2]+=.2
    path=tmp_path/'changed.npz'
    np.savez(path,fps=data['fps'],body_pos_w=data['root_pos'][:,None],
             body_quat_w=data['root_rot'][:,None,[3,0,1,2]],body_names=['pelvis'],
             quat_order='wxyz',joint_pos=data['dof_pos'],joint_names=data['joint_names'],
             model_sha256=data['model_sha256'])
    source=dict(status='accepted',flags=[],backend='dex-retarget')
    result=report_npz(path,ROOT/'assets/dex/mjcf/dex_29.xml',manifest,source,path,True)
    assert result['metrics']['both_feet_above_5cm_fraction']==1
    assert result['source_quality']['status']=='accepted'
    assert result['status']=='checked' and result['training_approved'] is False


def test_changed_quality_cannot_resume(tmp_path):
    from retarget_backend import verify_export
    output=tmp_path/'motion.pkl';output.write_bytes(b'fixture')
    source=tmp_path/'human.pt';source.write_bytes(b'fixture input')
    quality=output.with_suffix('.quality.json');quality.write_text('{}')
    receipt=dict(source_sha256=sha(source),output_sha256=sha(output),quality_sha256=sha(quality))
    output.with_suffix('.backend.json').write_text(json.dumps(receipt))
    verify_export(output,source)
    quality.write_text('{"status":"accepted"}')
    with pytest.raises(ValueError,match='quality report changed'):verify_export(output,source)
