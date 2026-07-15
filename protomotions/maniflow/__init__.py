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
"""Integration layer for ManiFlow policies trained outside this repo.

This package is the single seam between ProtoMotions and the ManiFlow
codebase (currently an external checkout, see ``loader.py`` for how the
``maniflow`` package is resolved). Simulation-side code should only import
from here — never from ``maniflow.*`` directly — so that vendoring the
ManiFlow model code into this repo later only requires changes inside this
package.

Modules:
    channels         — estimator action-channel contract (hip DOF names and
                       name-based COMMON-index derivation). Single source of
                       truth shared by collection, training, and inference.
    loader           — resolve/import the ``maniflow`` package and load
                       workspace checkpoints into inference-ready policies.
    torque_estimator — closed-loop wrapper turning ProtoMotions
                       ``RobotState`` into ManiFlow observations and hip
                       torque predictions.
    hybrid_control   — per-env/per-DOF torque override on top of Newton
                       BUILT_IN_PD, for applying ManiFlow torques to a
                       subset of joints while the rest stay PD-driven.
"""

from protomotions.maniflow.channels import (  # noqa: F401
    HIP_DOF_NAMES,
    hip_dof_indices,
)
from protomotions.maniflow.loader import (  # noqa: F401
    discover_best_checkpoint,
    ensure_maniflow_importable,
    load_maniflow_policy,
)
from protomotions.maniflow.torque_estimator import ManiFlowTorqueEstimator  # noqa: F401
from protomotions.maniflow.hybrid_control import JointTorqueOverride  # noqa: F401
