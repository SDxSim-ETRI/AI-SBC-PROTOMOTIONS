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
"""Continuous walking with assist motor output ON, OFF, then ON again."""
import argparse
import numpy as np
import torch
import presentation_off_capture as removal
import run_system


class ToggleHook(removal.RemovalHook):
    def __init__(self, *args, toggle, **kwargs):
        protocol=argparse.Namespace(mode='reference',off_at=toggle.off_at,reference=None)
        super().__init__(*args,protocol=protocol,**kwargs)
        self.toggle=toggle
        self.off_index=round(toggle.off_at/self.llp_dt)
        self.on_index=round(toggle.on_at/self.llp_dt)
        self.enabled=np.ones(self.n_calls,dtype=bool)

    @torch.no_grad()
    def __call__(self, sim, window):
        index=self.i
        super().__call__(sim,window)
        if self.off_index<=index<self.on_index:
            self.override.set_torques(torch.zeros(2,self.n_dofs,device=self.env.device))
            self.log['tau_assist'][index].zero_()
            self.enabled[index]=False

    def traces(self):
        result=super().traces()
        result.update(assist_motor_enabled=self.enabled[:self.i],assist_off_at_s=np.asarray(self.toggle.off_at),assist_on_at_s=np.asarray(self.toggle.on_at))
        assert not self.replay_active
        return result


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--off-at',type=float,default=10)
    p.add_argument('--on-at',type=float,default=15)
    toggle,remaining=p.parse_known_args()
    args=run_system.parse_args(remaining)
    assert 0<toggle.off_at<toggle.on_at<args.seconds
    assert args.human=='tracker'
    run_system.parse_args=lambda:args
    run_system.SystemHook=lambda *a,**kw:ToggleHook(*a,toggle=toggle,**kw)
    run_system.main()


if __name__=='__main__':main()
