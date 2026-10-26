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
"""LLP 20 N·m · 다리별 학습 · 실제 동작 의도 궤적 (2026-10-21).

R0 레시피(토크 경로 수정, 진자 관성·착용자 임피던스 무작위화, 입각 하중, HLP 예측 에뮬레이션)에서
세 가지를 바꾼다. 나머지는 R0와 같다.

1. 출력 20·tanh(u), 한계 20 N·m. 배포·평가 배율은 1이다.
2. 의도 궤적 θ_g. env를 세 집단에 고정 배정한다 (`INTENT_POOLS`).
   * 합성 20 %: 기존 합성 궤적, R0 관성 범위.
   * 가벼운 다리 40 %: 관성 배율 1–3.9 (hip 축 0.42–0.80 kg·m²), 달리기 60 · 조깅 30 · 점프 10 %.
   * 일반 40 %: R0 관성 범위, 걷기 35 · 조깅 20 · 점프 10 · 앉기/일어서기류 25 · 기타 10 %.
   동작은 SEED 고관절 굴곡 100 fps 데이터 중 H1 학습분할만 쓴다 (H1 미학습분은 평가용).
   비율은 클립 수가 아니라 쓰이는 시간 기준이다. 각 다리가 따라갈 수 없는 클립
   (|θ̈| 99 백분위 × hip 관성 > 0.8 × 착용자 토크 한계 150 N·m)은 그 env에서 뽑지 않는다.
   R0 관성 분포에서는 달리기를 따라갈 수 있는 다리가 2 % 남짓이라 가벼운 다리 집단을 따로 둔다.
3. 다리별 학습. `--leg right|left`: 그 관절에만 보조를 넣고, 보상·종료도 그 관절만 센다.
   actor는 그 관절의 관측만 읽는다 (`SingleJointMLP`). 배포·평가는 오른쪽·왼쪽 모델을 묶는다
   (`protomotions/assist/llp_pair.py`).

보조 토크² 벌점(A)과 명령 변화 벌점(Δ)은 `--assist-effort-weight`·`--action-rate-weight`로 정한다.
0이면 끄고, R0 값은 −2e-5·−0.005이다.

학습 (sbc env):
    python protomotions/train_agent.py --robot-name hip_pendulum --simulator newton \\
        --experiment-path examples/experiments/assist_pendulum/mlp_motion_leg.py \\
        --experiment-name llp20_motion_none_right_s0 --motion-file none \\
        --num-envs 4096 --batch-size 16384 --seed 0 --training-max-steps 786432000 \\
        --leg right --assist-effort-weight=0 --action-rate-weight=0
"""

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlp_relative as baseline  # noqa: E402
from mlp_relative import (  # noqa: E402, F401
    apply_inference_overrides,
    configure_robot_and_simulator,
    motion_lib_config,
    scene_lib_config,
    terrain_config,
)

SEED_ZARR = "/home/bak/Projects/ManiFlow_Policy/ManiFlow/data/hip-flexion-seed-locomotion-100fps.zarr"
SEED_METADATA = "data/seed/metadata/seed_metadata_v004.parquet"
LEGS = {"right": 0, "left": 1}  # hip_pendulum COMMON 순서 = 데이터 채널 순서 [r, l]

# (그룹, content_type_of_movement 정규식) — 처음 맞는 그룹
MOTION_GROUPS = (
    ("running", r"^running"),
    ("jogging", r"^jogging"),
    ("deep", r"sitting|kneel|transition|crouch"),
    ("jumping", r"^jumping"),
    ("walking", r"^walking(, turning)?$"),
    ("other", r".*"),
)
# hip 축 관성 I = 0.2863 + 0.1316 × 배율 → 배율 3.9 = 0.80 kg·m²
INTENT_POOLS = (
    ("synthetic", 0.2, "synthetic", 1.0, 25.0, ()),
    ("light_fast", 0.4, "motion", 1.0, 3.9,
     (("running", 0.6), ("jogging", 0.3), ("jumping", 0.1))),
    ("general", 0.4, "motion", 1.0, 25.0,
     (("walking", 0.35), ("jogging", 0.2), ("jumping", 0.1), ("deep", 0.25), ("other", 0.1))),
)
MOTION_ACCEL_FRAC = 0.8


def additional_experiment_arguments(parser):
    baseline.additional_experiment_arguments(parser)
    parser.set_defaults(assist_limit=20.0)
    g = parser.add_argument_group("LLP 20 N·m per-leg motion-intent study")
    g.add_argument("--leg", choices=sorted(LEGS), required=True, help="학습할 다리")
    g.add_argument("--motion-zarr", default=SEED_ZARR, help="SEED 고관절 굴곡 zarr (100 fps)")
    g.add_argument("--motion-metadata", default=SEED_METADATA, help="SEED metadata parquet")


def leg_joint_mask(leg: str, num_dofs: int = 2):
    joint = LEGS[leg]
    return joint, tuple(1.0 if j == joint else 0.0 for j in range(num_dofs))


def env_config(robot_cfg, args):
    from protomotions.envs.control.assist_disturbance_control import (
        DisturbedAssistTargetControlConfig,
    )

    if float(args.paired_excess_weight) != 0.0:
        raise ValueError("This study uses the R0 recipe: --paired-excess-weight must stay 0")
    cfg = baseline.env_config(robot_cfg, args)
    joint, mask = leg_joint_mask(args.leg, robot_cfg.number_of_actions)
    seed = int(getattr(args, "seed", 0) or 0)

    old = cfg.control_components["assist"]
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    values.update(
        intent_source="motion_data",
        motion_zarr_path=str(args.motion_zarr),
        motion_split="train",
        motion_metadata_parquet=str(Path(args.motion_metadata).resolve()),
        motion_seed=seed,
        intent_pools=INTENT_POOLS,
        motion_groups=MOTION_GROUPS,
        motion_accel_frac=MOTION_ACCEL_FRAC,
        motion_accel_check_dofs=(joint,),
        intent_pool_seed=seed,
    )
    cfg.control_components["assist"] = DisturbedAssistTargetControlConfig(**values)

    cfg.assist_active_joints = (joint,)
    for name in ("tracking", "human_power", "human_torque_sq", "assist_effort"):
        cfg.reward_components[name].static_params["joint_mask"] = mask
    for name in ("tracking_failure", "dof_vel_failure"):
        cfg.termination_components[name].static_params["joint_mask"] = mask
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    from protomotions.agents.common.shared_joint_mlp import SingleJointMLPConfig

    cfg = baseline.agent_config(robot_cfg, env_cfg, args)
    joint, _ = leg_joint_mask(args.leg, robot_cfg.number_of_actions)
    old = cfg.model.actor.mu_model
    values = {f.name: getattr(old, f.name) for f in fields(old) if f.name != "_target_"}
    cfg.model.actor.mu_model = SingleJointMLPConfig(**values, joint_index=joint)
    return cfg
