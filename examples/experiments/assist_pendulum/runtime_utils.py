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
"""assist_pendulum 체크포인트 로드/rollout 공용 유틸 (play.py, compare_rollout.py용).

inference_agent.py의 구성 경로를 최소한으로 복제: resolved_configs_inference.pt
로드 -> 컴포넌트 빌드 -> env/agent 생성 -> 체크포인트 로드.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from lightning.fabric import Fabric

from protomotions.utils.hydra_replacement import get_class


def build_env_agent(
    checkpoint: str,
    num_envs: int,
    headless: bool = True,
    overrides: Optional[Dict[str, object]] = None,
):
    """체크포인트에서 env + agent 재구성.

    Args:
        checkpoint: last.ckpt 경로.
        num_envs: 병렬 env 수.
        headless: False면 Newton viewer 창 표시.
        overrides: {"env.assist_torque_limit": 0.0} 형식의 config override.

    Returns:
        (env, agent)
    """
    ckpt = Path(checkpoint)
    resolved_path = ckpt.parent / "resolved_configs_inference.pt"
    assert resolved_path.exists(), f"resolved configs not found: {resolved_path}"
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = num_envs
    simulator_config.headless = headless

    if overrides:
        from protomotions.utils.config_utils import apply_config_overrides

        apply_config_overrides(
            overrides,
            env_config,
            simulator_config,
            robot_config,
            agent_config,
            terrain_config,
            motion_lib_config,
            scene_lib_config,
        )

    fabric = Fabric(
        accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
    )
    fabric.launch()

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(config=agent_config, env=env, fabric=fabric, root_dir=ckpt.parent)
    agent.setup()
    agent.load(str(ckpt), load_env=False)
    agent.model.eval()
    return env, agent


def rollout_recorded(
    env, agent, num_steps: int, seed: int
) -> Tuple[Dict[str, "torch.Tensor"], int]:
    """deterministic(mean_action) rollout을 돌며 상태/토크 시계열 기록.

    Returns:
        (logs, num_resets) — logs[key]: (num_steps, num_envs, num_dofs) CPU 텐서.
        키: dof_pos, dof_vel, theta_g, theta_d, tau_agent, tau_assist.
    """
    # lazy init(게인 랜덤화 등)이 첫 step에서 RNG를 소비하면 시딩 이후의
    # 난수 스트림이 rollout 간에 어긋난다 — 시딩 전에 미리 초기화해서
    # 같은 seed의 rollout들이 완전히 동일한 궤적을 받도록 보장한다.
    if hasattr(env, "_lazy_init_assist"):
        env._lazy_init_assist()

    torch.manual_seed(seed)
    obs, _ = env.reset()
    obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    keys = ("dof_pos", "dof_vel", "theta_g", "theta_d", "tau_agent", "tau_assist")
    logs = {k: [] for k in keys}
    num_resets = 0

    for _ in range(num_steps):
        with torch.no_grad():
            outs = agent.model(obs_td)
        actions = outs.get("mean_action", outs.get("action"))
        obs, _rew, done, _term, _extras = env.step(actions)

        ctx = env._current_context
        logs["dof_pos"].append(ctx.current.dof_pos.detach().cpu().clone())
        logs["dof_vel"].append(ctx.current.dof_vel.detach().cpu().clone())
        logs["theta_g"].append(ctx.assist.theta_g.detach().cpu().clone())
        logs["theta_d"].append(ctx.assist.theta_d.detach().cpu().clone())
        logs["tau_agent"].append(ctx.assist.tau_agent.detach().cpu().clone())
        logs["tau_assist"].append(ctx.assist.tau_assist.detach().cpu().clone())

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            num_resets += len(done_ids)
            env.reset(done_ids)
            obs = env.get_obs()

        obs_td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))

    return {k: torch.stack(v) for k, v in logs.items()}, num_resets
