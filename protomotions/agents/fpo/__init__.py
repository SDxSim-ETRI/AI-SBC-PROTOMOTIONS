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
"""FPO++ (Flow Policy Optimization) agent with an optional Temporal Transformer context encoder.

Key Components:
    - FPO (agent.py): PPO subclass whose actor loss is the CFM-loss ratio of FPO++
    - FlowActor / TemporalTransformer (model.py): flow policy and its context encoder
    - FPOAgentConfig / FPOModelConfig / FlowActorConfig / TemporalTransformerConfig (config.py)

Example:
    >>> from protomotions.agents.fpo.agent import FPO
    >>> agent = FPO(fabric, env, config)
    >>> agent.setup(); agent.fit()
"""
