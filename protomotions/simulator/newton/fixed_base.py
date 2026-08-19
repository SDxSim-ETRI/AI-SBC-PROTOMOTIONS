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
"""고정 베이스(fix_base_link=True) 로봇을 위한 Newton 시뮬레이터 지원.

Newton의 ``ArticulationView``는 FIXED root joint에 대해:
  * ``get_root_velocities()`` -> ``None`` 반환 (ProtoMotions 상태 배관에서 crash)
  * ``set_root_transforms()`` -> state가 아닌 **model**의 joint_X_p를 수정
    (solver 캐시와 불일치 위험)
  * ``set_root_velocities()`` -> no-op

``FixedBaseArticulationView``는 root velocity를 0으로 돌려주고 root 상태 쓰기를
무시하여(피벗은 월드 용접) 나머지 파이프라인이 그대로 동작하게 한다.
``FixedBaseNewtonSimulator``는 robot_view 생성 시 이 view 클래스를 쓰도록
``_setup_robot()``만 감싼다.

사용법: 실험 파일의 ``configure_robot_and_simulator``에서
``simulator_cfg._target_ = "protomotions.simulator.newton.fixed_base.FixedBaseNewtonSimulator"``
"""

import warp as wp
from newton.selection import ArticulationView

from protomotions.simulator.newton.simulator import NewtonSimulator


class FixedBaseArticulationView(ArticulationView):
    """FIXED root joint을 허용하는 ArticulationView.

    root velocity 읽기는 (num_worlds, 1, 6) 0-텐서를 반환하고, root
    transform/velocity 쓰기는 무시한다 (빌드 시점 포즈 유지). floating base면
    원본 동작 그대로.
    """

    def get_root_velocities(self, source):
        if self.is_floating_base:
            return super().get_root_velocities(source)
        if not hasattr(self, "_fixed_zero_root_velocities"):
            num_worlds = self.get_root_transforms(self.model).shape[0]
            self._fixed_zero_root_velocities = wp.zeros(
                (num_worlds, 1, 6),
                dtype=wp.float32,
                device=self.model.joint_q.device,
            )
        return self._fixed_zero_root_velocities

    def set_root_velocities(self, source, values, mask=None):
        if self.is_floating_base:
            super().set_root_velocities(source, values, mask=mask)

    def set_root_transforms(self, source, values, mask=None):
        if self.is_floating_base:
            super().set_root_transforms(source, values, mask=mask)


class FixedBaseNewtonSimulator(NewtonSimulator):
    """``asset.fix_base_link=True`` 로봇용 NewtonSimulator.

    ``_setup_robot()`` 동안에만 모듈의 ``ArticulationView``를
    ``FixedBaseArticulationView``로 교체하여 ``self.robot_view``가 고정 베이스를
    안전하게 다루도록 한다. 그 외 동작은 NewtonSimulator와 동일하다.
    """

    def _setup_robot(self) -> None:
        if not self.robot_config.asset.fix_base_link:
            super()._setup_robot()
            return

        import protomotions.simulator.newton.simulator as newton_sim_module

        original_view_cls = newton_sim_module.ArticulationView
        newton_sim_module.ArticulationView = FixedBaseArticulationView
        try:
            super()._setup_robot()
        finally:
            newton_sim_module.ArticulationView = original_view_cls
