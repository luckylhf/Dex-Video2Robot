"""只测试 MP4 调度、隔离环境及跨包路径；外部模型推理使用替身。"""
import runpy
import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class VideoPipelineTest(unittest.TestCase):
    def setUp(self):
        # These scheduling tests use fake videos/predictions, not licensed body resources.
        body_check = patch('check_body_models.require_body_models')
        self.body_check = body_check.start()
        self.addCleanup(body_check.stop)

    def test_video_defaults_to_bundled_body_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / 'clip.mp4').touch()
            (base / 'GVHMR/tools/demo').mkdir(parents=True)
            (base / 'GVHMR/tools/demo/demo.py').touch()
            (base / 'work/human/source_30fps').mkdir(parents=True)
            (base / 'work/human/source_30fps/hmr4d_results.pt').touch()
            argv = ['video_to_dex.py', str(base / 'clip.mp4'), '--gvhmr-root', str(base / 'GVHMR'),
                '--gvhmr-python', '/path/to/gvhmr/python', '--output', str(base / 'clip.pkl')]
            with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', argv), \
                patch('subprocess.run') as run, patch('tempfile.mkdtemp', return_value=str(base / 'work')):
                runpy.run_path(str(ROOT / 'scripts/video_to_dex.py'), run_name='__main__')
            command = run.call_args_list[2].args[0]
            self.assertEqual(command[command.index('--body-models') + 1], str(ROOT / 'assets/body_models'))
            self.assertEqual(command[command.index('--ik-recovery') + 1], 'strict')

    def test_mp4_calls_gvhmr_then_ik_then_mimic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / 'clip.mp4').touch()
            (base / 'GVHMR/tools/demo').mkdir(parents=True)
            (base / 'GVHMR/tools/demo/demo.py').touch()
            (base / 'models').mkdir()
            (base / 'work').mkdir()
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                if len(command) > 1 and command[1] == 'tools/demo/demo.py':
                    pred = base / 'work/human/source_30fps/hmr4d_results.pt'
                    pred.parent.mkdir(parents=True)
                    pred.touch()
            argv = ['video_to_dex.py', str(base / 'clip.mp4'), '--gvhmr-root', str(base / 'GVHMR'),
                '--gvhmr-python', '/path/to/gvhmr/python', '--body-models', str(base / 'models'),
                '--output', str(base / 'clip.pkl'), '--mimic-dir', str(base / 'Dex-Mimic'),
                '--static-camera', '--ik-recovery', 'table-retry']
            with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=run), patch('tempfile.mkdtemp', return_value=str(base / 'work')):
                runpy.run_path(str(ROOT / 'scripts/video_to_dex.py'), run_name='__main__')
            self.assertEqual(calls[0][0][0], 'ffmpeg')
            self.assertEqual(calls[1][0][0], '/path/to/gvhmr/python')
            self.assertIn('-s', calls[1][0])
            self.assertEqual(calls[1][1]['cwd'], base / 'GVHMR')
            self.assertTrue(calls[2][0][1].endswith('retarget_motion.py'))
            self.assertEqual(calls[2][0][calls[2][0].index('--ik-recovery') + 1], 'table-retry')
            self.assertEqual(json.loads((base / 'work/pipeline.json').read_text())['ik_recovery'], 'table-retry')
            self.assertTrue(calls[3][0][1].endswith('pkl_to_npz.py'))
            self.assertEqual(calls[3][1]['cwd'], base / 'Dex-Mimic')
            # Resuming this job with a different recovery policy must reject cached PKL reuse.
            resumed = [*argv, '--work-dir', str(base / 'work'), '--resume']
            resumed[resumed.index('--ik-recovery') + 1] = 'strict'
            with patch.object(sys, 'argv', resumed), patch('subprocess.run') as run:
                with self.assertRaises(SystemExit) as error:
                    runpy.run_path(str(ROOT / 'scripts/video_to_dex.py'), run_name='__main__')
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
