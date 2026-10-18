import os
import json
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from mujoco_node.policy_contract import asset_sha256, file_sha256, verify_model


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self._setup_noise_schedule()

    def _dex_contract(self):
        model_dir = Path(__file__).resolve().parents[1] / "assets/dex_evt"
        manifest = verify_model(model_dir)
        robot = self.env.unwrapped.scene["robot"]
        return dict(model_sha256=manifest["model_sha256"], asset_sha256=asset_sha256(model_dir),
                    control_dt=float(self.env.unwrapped.step_dt),
                    actuators={name: {field: getattr(cfg, field, None)
                                      for field in ("stiffness", "damping", "armature", "effort_limit_sim")}
                               for name, cfg in robot.cfg.actuators.items()})

    def load(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved = (checkpoint.get("infos") or {}).get("dex_contract")
        runtime = self._dex_contract()
        if saved is None:
            catalog_file = Path(__file__).resolve().parents[4] / "policy/dex/standing_baseline.json"
            catalog = json.loads(catalog_file.read_text())
            if file_sha256(path) != catalog["checkpoint_sha256"] or runtime != catalog["runtime_contract"]:
                raise ValueError("Unversioned checkpoint is not the validated standing baseline; use LOAD_CKPT to re-train it")
        elif saved != runtime:
            raise ValueError("Checkpoint model/dynamics differ from this task; use LOAD_CKPT for explicit re-training")
        return super().load(path, *args, **kwargs)

    def _setup_noise_schedule(self):
        schedule = self.cfg.get("noise_std_schedule", None)
        if schedule is None:
            return

        std_init = schedule.get("init_std", 0.5)
        std_final = schedule.get("final_std", 0.05)
        decay_iters = max(schedule.get("decay_iters", 10000), 1)
        entropy_init = schedule.get("entropy_init", None)
        entropy_final = schedule.get("entropy_final", None)
        mode = schedule.get("mode", "exp")

        original_update = self.alg.update
        runner = self

        def _scheduled_update():
            result = original_update()
            # current_learning_iteration is set to `it` after this call returns,
            # so we read it here (off-by-one is negligible over 15k iters)
            it = runner.current_learning_iteration
            t = min(it / decay_iters, 1.0)
            if mode == "linear":
                std = std_init + (std_final - std_init) * t
            else:
                std = std_init * (std_final / std_init) ** t
            with torch.no_grad():
                runner.alg.policy.std.fill_(std)
            if entropy_init is not None and entropy_final is not None:
                runner.alg.entropy_coef = entropy_init + (entropy_final - entropy_init) * t
            return result

        self.alg.update = _scheduled_update

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        infos = dict(infos or {})
        infos["dex_contract"] = self._dex_contract()
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
