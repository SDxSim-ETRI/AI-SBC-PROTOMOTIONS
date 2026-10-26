# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""`mlp.py` 프리셋이 보관 모델의 학습 config를 재현하는지 (2026-10-26 계보 평탄화의 충실도).

L1·R0·motion_leg(A0D0) 각각을 기본 인자로 조립해 그 모델의 `resolved_configs.pt`(학습 당시 pickle)와
env·agent.model config를 필드별로 비교한다. 체크포인트가 없는 환경에서는 건너뛴다.

    pytest protomotions/tests/test_pendulum_recipes.py
"""

import argparse
import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch

from protomotions.envs.mdp_component import MdpComponent
from protomotions.robot_configs.hip_pendulum import HipPendulumRobotConfig

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/experiments/assist_pendulum/mlp.py"
CASES = {
    "L1": ("checkpoints/L1_llp_pendulum", ["--recipe", "L1"]),
    "R0": ("results/llp_relative_R0_s0_20261020", ["--recipe", "R0"]),
    "motion_leg_A0D0": ("results/llp20_motion_A0D0_right_s0_20261021",
                        ["--recipe", "motion_leg", "--leg", "right", "--assist-effort-weight=0", "--action-rate-weight=0"]),
}
# train_agent가 실험 파일 밖에서 채우는 필드
IGNORE_ENV = {"save_dir", "experiment_name", "num_envs", "headless"}


def load_recipe():
    spec = importlib.util.spec_from_file_location("pendulum_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(extra):
    module = load_recipe()
    parser = argparse.ArgumentParser()
    module.additional_experiment_arguments(parser)
    args = parser.parse_args(extra)
    args.batch_size, args.training_max_steps, args.seed = 16384, 786432000, 0
    robot = HipPendulumRobotConfig()
    env = module.env_config(robot, args)
    return env, module.agent_config(robot, env, args)


def plain(x, ignore=()):
    if isinstance(x, MdpComponent):
        return {"compute_func": x.compute_func.__name__,
                "dynamic_vars": {k: getattr(v, "path", v) for k, v in x.dynamic_vars.items()},
                "static_params": plain(x.static_params)}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x) if f.name not in ignore}
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing' if k not in a else 'extra'}")
            else:
                out += differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_recipe_reproduces_recorded_training_config(name):
    folder, extra = CASES[name]
    recorded = ROOT / folder / "resolved_configs.pt"
    if not recorded.exists():
        pytest.skip(f"{recorded} not present")
    rec = torch.load(recorded, map_location="cpu", weights_only=False)
    env, agent = build(extra)
    env_diff = differences(plain(env, IGNORE_ENV), plain(rec["env"], IGNORE_ENV))
    assert not env_diff, "\n".join(env_diff)
    model_diff = differences(plain(agent.model), plain(rec["agent"].model))
    assert not model_diff, "\n".join(model_diff)
    for field in ("save_epoch_checkpoint_every", "save_last_checkpoint_every"):
        assert getattr(agent, field) == getattr(rec["agent"], field), field
    assert agent.evaluator.eval_metrics_every == rec["agent"].evaluator.eval_metrics_every


def test_recipes_differ_only_where_the_lineage_says():
    l1, _ = build(["--recipe", "L1"])
    r0, _ = build(["--recipe", "R0"])
    changed = set(differences(plain(l1), plain(r0)))
    assert {d.split(":")[0] for d in changed} == {".clamp_combined_injection", ".human_feedforward_inside_pd"}, changed
    with pytest.raises(ValueError, match="--leg"):
        build(["--recipe", "motion_leg"])
    leg, agent = build(["--recipe", "motion_leg", "--leg", "left"])
    assert tuple(leg.assist_active_joints) == (1,) and leg.assist_torque_limit == 20.0
    assert agent.model.actor.mu_model.joint_index == 1
    assert leg.reward_components["tracking"].static_params["joint_mask"] == (0.0, 1.0)
