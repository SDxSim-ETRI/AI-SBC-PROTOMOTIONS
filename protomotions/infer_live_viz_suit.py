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
"""Real-time live monitor for suit robot inference.

Extension of infer_live_viz.py for skeleton_torque_suit models.
With --active-cable, adds two extra panels:
  - Cable Displacement: slide2/slide4 actual positions vs DOFC A버전 targets
  - Hip & DOFC Balance: hip_r / hip_l angles + y = sin(-hip_r) - sin(-hip_l)

Joint selection
---------------
**GUI**: CheckButtons panels on the right let you toggle joint groups
for Torques and Positions plots independently.

**CLI**: --torque-dofs and --pos-dofs accept short DOF names (partial match OK).

Multiple envs
-------------
Overview strip shows root-z for all envs. Press ← / → (or 0-9) to switch detail env.

Example
-------
::

    # Active cable live viz (Newton, single env)
    python protomotions/infer_live_viz_suit.py \\
        --checkpoint tasks/mimic_suit_active_cable_motions14_23dof/output_newton/score_based.ckpt \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \\
        --simulator newton --num-envs 1 --active-cable

    # Passive cable live viz (no cable panels)
    python protomotions/infer_live_viz_suit.py \\
        --checkpoint tasks/mimic_suit_passive_cable_motions14_23dof/output_newton/score_based.ckpt \\
        --motion-file data/motion_for_trackers/skeleton_torque_suit_motions_11+koo_4.pt \\
        --simulator newton --num-envs 1
"""

from __future__ import annotations

# Ensure this project's protomotions package takes precedence over any installed version.
# torch.load(weights_only=False) unpickles class references that need the local robot configs.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import collections
from typing import Dict, List, Optional, Set

import colorsys
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

_GROUP_HUES = {
    "left_leg":  0.83,
    "right_leg": 0.50,
    "waist":     0.33,
    "left_arm":  0.08,
    "right_arm": 0.00,
    "other":     0.75,
}


def _joint_level(name: str) -> int:
    n = name.lower()
    # proximal / trunk (dark)
    if any(k in n for k in ("hip", "shoulder", "arm_flex", "arm_add", "arm_rot", "lumbar")):
        return 0
    # mid-limb (mid)
    if any(k in n for k in ("knee", "elbow", "elbow_flex", "slide")):
        return 1
    # distal (light)
    if any(k in n for k in ("ankle", "wrist", "pro_sup")):
        return 2
    return 1


def _axis_linestyle(name: str) -> str:
    """Linestyle by motion axis: flexion/extension=solid, adduction/bending=dashed, rotation=dotted."""
    n = name.lower()
    if any(k in n for k in ("flexion", "extension", "flex", "angle")):
        return "-"
    if any(k in n for k in ("adduction", "add", "bending")):
        return "--"
    if any(k in n for k in ("rotation", "rot", "pro_sup")):
        return ":"
    return "-"


def _dof_color(group: str, level: int):
    hue = _GROUP_HUES.get(group, 0.75)
    val = (0.40, 0.68, 0.92)[level]
    return colorsys.hsv_to_rgb(hue, 0.85, val)


def _group_dofs(dof_names: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {k: [] for k in _GROUP_HUES}
    for i, name in enumerate(dof_names):
        n = name.lower()
        # Side: explicit keyword OR _r/_l suffix (e.g. hip_flexion_r, arm_flex_l)
        right = "right" in n or n.endswith("_r") or "_r_" in n
        left  = "left"  in n or n.endswith("_l") or "_l_" in n
        leg   = any(k in n for k in ("hip", "knee", "ankle", "toe"))
        # arm: shoulder synonyms (arm_flex/add/rot) + elbow + wrist + pro_sup
        arm   = any(k in n for k in ("shoulder", "arm_flex", "arm_add", "arm_rot",
                                      "elbow", "wrist", "finger", "pro_sup"))
        waist = any(k in n for k in ("waist", "spine", "torso", "pelvis", "trunk", "lumbar"))
        if   right and leg:   groups["right_leg"].append(i)
        elif left  and leg:   groups["left_leg"].append(i)
        elif right and arm:   groups["right_arm"].append(i)
        elif left  and arm:   groups["left_arm"].append(i)
        elif waist:           groups["waist"].append(i)
        else:                 groups["other"].append(i)
    return {k: v for k, v in groups.items() if v}


def _foot_indices(body_names: List[str]) -> Dict[str, int]:
    keywords = ("ankle", "foot", "toe")
    return {
        name: i
        for i, name in enumerate(body_names)
        if any(kw in name.lower() for kw in keywords)
    }


def _short(name: str) -> str:
    return name.replace("_joint", "")


def _resolve_dof_filter(
    dof_names: List[str], requested: Optional[List[str]]
) -> Optional[Set[int]]:
    if not requested:
        return None
    matched: Set[int] = set()
    for req in requested:
        req_l = req.lower()
        for i, name in enumerate(dof_names):
            if req_l in _short(name).lower():
                matched.add(i)
    return matched or None


# ── LiveVisualizerSuit ────────────────────────────────────────────────────────


class LiveVisualizerSuit:
    """Real-time monitor for suit robot with optional cable panels.

    Keyboard shortcuts (click the figure window first):
        ← / →       select previous / next env
        0-9         jump to env N
        q           quit
    """

    def __init__(
        self,
        body_names: List[str],
        dof_names: List[str],
        num_envs: int = 1,
        window: int = 300,
        update_every: int = 5,
        torque_dof_filter: Optional[List[str]] = None,
        pos_dof_filter: Optional[List[str]] = None,
        active_cable: bool = False,
    ):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.body_names = list(body_names)
        self.dof_names  = list(dof_names)
        self.num_envs   = num_envs
        self.window     = window
        self.update_every = update_every
        self.active_cable = active_cable
        self.selected_env = 0
        self._step = 0
        self._quit = False

        self._dof_groups = _group_dofs(dof_names)
        self._foot_idx   = _foot_indices(body_names)

        self._torque_filter: Optional[Set[int]] = _resolve_dof_filter(dof_names, torque_dof_filter)
        self._dofpos_filter: Optional[Set[int]] = _resolve_dof_filter(dof_names, pos_dof_filter)

        buf_keys = [
            "root_z", "root_vx", "root_vy", "root_vz",
            "torques", "dof_pos", "actions", "foot_forces",
        ]
        if active_cable:
            buf_keys += ["cable_pos", "dofc_targets", "hip_angles", "dofc_balance"]

        self._bufs: List[Dict] = [
            {k: collections.deque(maxlen=window) for k in buf_keys}
            for _ in range(num_envs)
        ]

        self._setup_figure()

    # ── figure ───────────────────────────────────────────────────────────────

    def _setup_figure(self):
        from matplotlib.widgets import CheckButtons

        plt = self.plt
        plt.ion()

        n_rows = 5 if self.active_cable else 4
        fig_h  = 13 if self.active_cable else 10
        height_ratios = [0.45, 1, 1, 1, 1] if self.active_cable else [0.45, 1, 1, 1]

        self.fig = plt.figure(figsize=(17, fig_h))
        self.fig.canvas.manager.set_window_title("ProtoMotions Suit Live Monitor")

        gs = self.fig.add_gridspec(
            n_rows, 2,
            height_ratios=height_ratios,
            hspace=0.55, wspace=0.32,
            left=0.06, right=0.78, top=0.94, bottom=0.05,
        )

        # ── overview strip ────────────────────────────────────────────────
        self.ax_ov = self.fig.add_subplot(gs[0, :])
        self.ax_ov.set_title(
            "Root Height — All Envs  (← / → or 0-9 to select detail env)", fontsize=8
        )
        self.ax_ov.set_ylabel("z (m)", fontsize=7)
        self.ax_ov.tick_params(labelsize=7)
        cmap = plt.cm.tab20
        self._ov_lines = [
            self.ax_ov.plot(
                [], [], lw=0.9, color=cmap(e / max(self.num_envs, 1)),
                alpha=0.6, label=f"env{e}",
            )[0]
            for e in range(self.num_envs)
        ]
        self.ax_ov.legend(
            fontsize=6, loc="upper right",
            ncol=min(10, self.num_envs), framealpha=0.7,
        )

        # ── detail panels (rows 1-3) ──────────────────────────────────────
        self.ax_root    = self.fig.add_subplot(gs[1, 0])
        self.ax_contact = self.fig.add_subplot(gs[1, 1])
        self.ax_torque  = self.fig.add_subplot(gs[2, 0])
        self.ax_dofpos  = self.fig.add_subplot(gs[2, 1])
        self.ax_action  = self.fig.add_subplot(gs[3, 0])
        self.ax_vel     = self.fig.add_subplot(gs[3, 1])

        for ax in (self.ax_root, self.ax_contact, self.ax_torque,
                   self.ax_dofpos, self.ax_action, self.ax_vel):
            ax.tick_params(labelsize=7)

        self.ax_root.set_title("Root Height + Z-velocity", fontsize=8)
        self.ax_contact.set_title("Contact Forces (N)", fontsize=8)
        self.ax_torque.set_title("Joint Torques (N·m)", fontsize=8)
        self.ax_dofpos.set_title("Joint Positions (rad)", fontsize=8)
        self.ax_action.set_title("Actions", fontsize=8)
        self.ax_vel.set_title("Root Linear Velocity (m/s)", fontsize=8)

        # ── Root height / vz ──────────────────────────────────────────────
        (self._root_z_line,)  = self.ax_root.plot([], [], "b-",  lw=1.2, label="z (m)")
        (self._root_vz_line,) = self.ax_root.plot([], [], "r-",  lw=1.0, alpha=0.7, label="vz (m/s)")
        self.ax_root.legend(fontsize=7, loc="upper right", framealpha=0.7)
        self.ax_root.axhline(0, color="gray", lw=0.5, ls="--")

        # ── Contact forces ────────────────────────────────────────────────
        foot_side_color = {"left": "#1f77b4", "right": "#d62728"}
        self._contact_lines: Dict[str, object] = {}
        for fname in self._foot_idx:
            side  = "left" if "left" in fname.lower() else "right"
            color = foot_side_color.get(side, "gray")
            (line,) = self.ax_contact.plot([], [], lw=1.2, label=fname, color=color)
            self._contact_lines[fname] = line
        self.ax_contact.legend(fontsize=7, loc="upper right", framealpha=0.7)

        # ── Joint torque / position lines ─────────────────────────────────
        self._torque_lines: Dict[int, object] = {}
        self._dofpos_lines: Dict[int, object] = {}

        for gname, idxs in self._dof_groups.items():
            for dof_i in idxs:
                name  = self.dof_names[dof_i]
                color = _dof_color(gname, _joint_level(name))
                ls    = _axis_linestyle(name)
                label = _short(name)
                (lt,) = self.ax_torque.plot([], [], lw=1.0, color=color, ls=ls, label=label)
                (lp,) = self.ax_dofpos.plot([], [], lw=1.0, color=color, ls=ls, label=label)
                self._torque_lines[dof_i] = lt
                self._dofpos_lines[dof_i] = lp

        for dof_i, line in self._torque_lines.items():
            line.set_visible(self._torque_filter is None or dof_i in self._torque_filter)
        for dof_i, line in self._dofpos_lines.items():
            line.set_visible(self._dofpos_filter is None or dof_i in self._dofpos_filter)

        self._refresh_legend(self.ax_torque, self._torque_lines)
        self._refresh_legend(self.ax_dofpos, self._dofpos_lines)

        # ── Action lines (lazy) ───────────────────────────────────────────
        self._action_lines: List = []

        # ── Root velocity ─────────────────────────────────────────────────
        vel_colors = {"vx": "#1f77b4", "vy": "#ff7f0e", "vz": "#2ca02c"}
        self._vel_lines = {
            k: self.ax_vel.plot([], [], lw=1.2, label=k, color=vel_colors[k])[0]
            for k in ("vx", "vy", "vz")
        }
        self.ax_vel.legend(fontsize=7, loc="upper right", framealpha=0.7)
        self.ax_vel.axhline(0, color="gray", lw=0.5, ls="--")

        # ── Cable panels (row 4, only when --active-cable) ────────────────
        self.ax_cable   = None
        self.ax_hip     = None
        self._cable_lines: Dict = {}
        self._hip_lines: Dict = {}

        if self.active_cable:
            self.ax_cable = self.fig.add_subplot(gs[4, 0])
            self.ax_hip   = self.fig.add_subplot(gs[4, 1])

            self.ax_cable.tick_params(labelsize=7)
            self.ax_hip.tick_params(labelsize=7)
            self.ax_cable.set_title("Cable Displacement (m) — actual vs DOFC target", fontsize=8)
            self.ax_hip.set_title("Hip Angles (rad) & DOFC Balance y", fontsize=8)

            # slide2: blue, slide4: orange
            cable_colors = {"slide2": "#1f77b4", "slide4": "#ff7f0e"}
            for cname, color in cable_colors.items():
                (la,) = self.ax_cable.plot([], [], lw=1.3, color=color,  ls="-",  label=f"{cname} actual")
                (lt,) = self.ax_cable.plot([], [], lw=1.0, color=color,  ls="--", label=f"{cname} target",
                                           alpha=0.7)
                self._cable_lines[f"{cname}_actual"]  = la
                self._cable_lines[f"{cname}_target"]  = lt
            self.ax_cable.axhline(0, color="gray", lw=0.5, ls="--")
            self.ax_cable.legend(fontsize=7, loc="upper right", framealpha=0.7, ncol=2)

            hip_colors = {"hip_r": "#e74c3c", "hip_l": "#3498db", "balance_y": "#2ecc71"}
            (self._hip_lines["hip_r"],)     = self.ax_hip.plot([], [], lw=1.2, color=hip_colors["hip_r"],
                                                                label="hip_r (rad)")
            (self._hip_lines["hip_l"],)     = self.ax_hip.plot([], [], lw=1.2, color=hip_colors["hip_l"],
                                                                label="hip_l (rad)")
            (self._hip_lines["balance_y"],) = self.ax_hip.plot([], [], lw=1.0, color=hip_colors["balance_y"],
                                                                ls="--", label="y balance")
            self.ax_hip.axhline(0, color="gray", lw=0.5, ls="--")
            self.ax_hip.legend(fontsize=7, loc="upper right", framealpha=0.7)

        # ── CheckButtons: Torques & Positions ─────────────────────────────
        group_names = list(self._dof_groups.keys())

        def _group_is_visible(lines_d: Dict[int, object], gname: str) -> bool:
            return any(
                lines_d.get(i) and lines_d[i].get_visible()
                for i in self._dof_groups[gname]
            )

        ax_ct = self.fig.add_axes([0.80, 0.50, 0.185, 0.31], facecolor="#f5f5f5")
        ax_ct.set_title("Torques\n(joint groups)", fontsize=7, pad=3)
        self._chk_torque = CheckButtons(
            ax_ct, group_names,
            [_group_is_visible(self._torque_lines, g) for g in group_names]
        )
        for lbl in self._chk_torque.labels:
            lbl.set_fontsize(7)
        self._chk_torque.on_clicked(self._on_check_torque)

        ax_cd = self.fig.add_axes([0.80, 0.13, 0.185, 0.31], facecolor="#f5f5f5")
        ax_cd.set_title("Positions\n(joint groups)", fontsize=7, pad=3)
        self._chk_dofpos = CheckButtons(
            ax_cd, group_names,
            [_group_is_visible(self._dofpos_lines, g) for g in group_names]
        )
        for lbl in self._chk_dofpos.labels:
            lbl.set_fontsize(7)
        self._chk_dofpos.on_clicked(self._on_check_dofpos)

        # ── title + keyboard ──────────────────────────────────────────────
        self._title = self.fig.suptitle(
            f"Suit Live Monitor  |  env 0/{self.num_envs - 1}  |  step 0",
            fontsize=10,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.plt.show(block=False)
        self.fig.canvas.flush_events()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = event.key
        if key in [str(i) for i in range(10)]:
            self.selected_env = min(int(key), self.num_envs - 1)
        elif key == "right":
            self.selected_env = (self.selected_env + 1) % self.num_envs
        elif key == "left":
            self.selected_env = (self.selected_env - 1) % self.num_envs
        elif key == "q":
            self._quit = True

    def _on_check_torque(self, label: str):
        group_names = list(self._dof_groups.keys())
        is_vis = self._chk_torque.get_status()[group_names.index(label)]
        for dof_i in self._dof_groups.get(label, []):
            if dof_i in self._torque_lines:
                self._torque_lines[dof_i].set_visible(is_vis)
        self._refresh_legend(self.ax_torque, self._torque_lines)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _on_check_dofpos(self, label: str):
        group_names = list(self._dof_groups.keys())
        is_vis = self._chk_dofpos.get_status()[group_names.index(label)]
        for dof_i in self._dof_groups.get(label, []):
            if dof_i in self._dofpos_lines:
                self._dofpos_lines[dof_i].set_visible(is_vis)
        self._refresh_legend(self.ax_dofpos, self._dofpos_lines)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_legend(self, ax, lines_dict: Dict[int, object]):
        handles, labels = [], []
        for dof_i, line in lines_dict.items():
            if line.get_visible():
                handles.append(line)
                labels.append(_short(self.dof_names[dof_i]))
        if handles:
            ax.legend(
                handles, labels, fontsize=5, ncol=2, loc="upper right",
                framealpha=0.7, labelspacing=0.15, columnspacing=0.5,
                handlelength=1.0,
            )
        elif ax.get_legend():
            ax.get_legend().remove()

    @staticmethod
    def _relim(ax):
        ax.relim()
        ax.autoscale_view()

    # ── public update ─────────────────────────────────────────────────────────

    def update(self, robot_state, contact_forces, actions, dofc_fn=None,
               override_torques=None) -> bool:
        """Append new data and optionally redraw. Returns False when user presses q.

        override_torques: [n_env, n_dof] tensor — replaces robot_state.dof_forces when
            the simulator does not store applied torques (e.g. Newton BUILT_IN_PD).
        """
        rp = robot_state.rigid_body_pos.cpu().numpy()
        rv = robot_state.rigid_body_vel.cpu().numpy()
        if override_torques is not None:
            torques = override_torques.cpu().numpy()
        else:
            torques = (
                robot_state.dof_forces.cpu().numpy()
                if robot_state.dof_forces is not None else None
            )
        dof_pos = robot_state.dof_pos
        acts = actions.cpu().numpy()
        cf = contact_forces.cpu().numpy() if contact_forces is not None else None

        cable_data = None
        if self.active_cable and dofc_fn is not None:
            import torch
            s2, s4 = dofc_fn(dof_pos)
            s2 = s2.clamp(min=0.0, max=0.51)
            s4 = s4.clamp(min=0.0, max=0.51)
            hip_r = dof_pos[:, 0]
            hip_l = dof_pos[:, 5]
            balance_y = torch.sin(-hip_r) - torch.sin(-hip_l)
            cable_data = {
                "cable_pos":    dof_pos[:, 23:27].cpu().numpy(),  # [n_env, 4]
                "dofc_targets": torch.stack([s2, s4], dim=-1).cpu().numpy(),  # [n_env, 2]
                "hip_r":        hip_r.cpu().numpy(),
                "hip_l":        hip_l.cpu().numpy(),
                "balance_y":    balance_y.cpu().numpy(),
            }

        dof_pos_np = dof_pos.cpu().numpy()
        n_env = dof_pos_np.shape[0]

        for e in range(min(n_env, self.num_envs)):
            b = self._bufs[e]
            b["root_z"].append(float(rp[e, 0, 2]))
            b["root_vx"].append(float(rv[e, 0, 0]))
            b["root_vy"].append(float(rv[e, 0, 1]))
            b["root_vz"].append(float(rv[e, 0, 2]))
            if torques is not None:
                b["torques"].append(torques[e])
            b["dof_pos"].append(dof_pos_np[e])
            b["actions"].append(acts[e])
            if cf is not None:
                b["foot_forces"].append(
                    {fname: float(np.linalg.norm(cf[e, fidx]))
                     for fname, fidx in self._foot_idx.items()}
                )
            if cable_data is not None:
                b["cable_pos"].append(cable_data["cable_pos"][e])
                b["dofc_targets"].append(cable_data["dofc_targets"][e])
                b["hip_angles"].append(
                    np.array([cable_data["hip_r"][e], cable_data["hip_l"][e]])
                )
                b["dofc_balance"].append(float(cable_data["balance_y"][e]))

        self._step += 1
        if self._step % self.update_every == 0:
            self._redraw()
        return not self._quit

    # ── redraw ────────────────────────────────────────────────────────────────

    def _redraw(self):
        sel = self.selected_env
        b   = self._bufs[sel]
        T   = len(b["root_z"])
        if T == 0:
            return
        xs = np.arange(T)

        # Overview
        for e, line in enumerate(self._ov_lines):
            ob = self._bufs[e]
            ot = len(ob["root_z"])
            if ot:
                line.set_data(np.arange(ot), list(ob["root_z"]))
            line.set_linewidth(2.2 if e == sel else 0.8)
            line.set_alpha(1.0  if e == sel else 0.35)
        self._relim(self.ax_ov)

        # Root height + vz
        self._root_z_line.set_data(xs, list(b["root_z"]))
        self._root_vz_line.set_data(xs, list(b["root_vz"]))
        self._relim(self.ax_root)

        # Contact forces
        if b["foot_forces"]:
            for fname, line in self._contact_lines.items():
                vals = [f.get(fname, 0.0) for f in b["foot_forces"]]
                line.set_data(np.arange(len(vals)), vals)
            self._relim(self.ax_contact)

        # Torques
        if b["torques"]:
            tarr = np.array(list(b["torques"]))
            for dof_i, line in self._torque_lines.items():
                if line.get_visible():
                    line.set_data(xs[: len(tarr)], tarr[:, dof_i])
            self._relim(self.ax_torque)

        # DOF positions
        if b["dof_pos"]:
            parr = np.array(list(b["dof_pos"]))
            for dof_i, line in self._dofpos_lines.items():
                if line.get_visible():
                    line.set_data(xs[: len(parr)], parr[:, dof_i])
            self._relim(self.ax_dofpos)

        # Actions
        if b["actions"]:
            aarr = np.array(list(b["actions"]))
            n_act = aarr.shape[1]
            if len(self._action_lines) != n_act:
                self.ax_action.cla()
                self.ax_action.set_title("Actions", fontsize=8)
                self.ax_action.tick_params(labelsize=7)
                cmap = self.plt.cm.tab20
                self._action_lines = [
                    self.ax_action.plot(
                        [], [], lw=0.8,
                        color=cmap(i / max(n_act, 1)),
                        alpha=0.75, label=f"a{i}",
                    )[0]
                    for i in range(n_act)
                ]
                self.ax_action.legend(
                    fontsize=5, ncol=3, loc="upper right",
                    framealpha=0.7, labelspacing=0.15, columnspacing=0.5,
                    handlelength=1.0,
                )
            for i, line in enumerate(self._action_lines):
                line.set_data(xs[: len(aarr)], aarr[:, i])
            self._relim(self.ax_action)

        # Root velocity
        for k, key in zip(("vx", "vy", "vz"), ("root_vx", "root_vy", "root_vz")):
            vals = list(b[key])
            self._vel_lines[k].set_data(np.arange(len(vals)), vals)
        self._relim(self.ax_vel)

        # Cable panels
        if self.active_cable and b.get("cable_pos"):
            cpos  = np.array(list(b["cable_pos"]))    # [T, 4]
            ctgt  = np.array(list(b["dofc_targets"])) # [T, 2]
            # slide2 = cable[1], slide4 = cable[3]
            self._cable_lines["slide2_actual"].set_data(xs[: len(cpos)], cpos[:, 1])
            self._cable_lines["slide4_actual"].set_data(xs[: len(cpos)], cpos[:, 3])
            self._cable_lines["slide2_target"].set_data(xs[: len(ctgt)], ctgt[:, 0])
            self._cable_lines["slide4_target"].set_data(xs[: len(ctgt)], ctgt[:, 1])
            self._relim(self.ax_cable)

        if self.active_cable and b.get("hip_angles"):
            harr = np.array(list(b["hip_angles"]))  # [T, 2]
            yarr = list(b["dofc_balance"])
            self._hip_lines["hip_r"].set_data(xs[: len(harr)], harr[:, 0])
            self._hip_lines["hip_l"].set_data(xs[: len(harr)], harr[:, 1])
            self._hip_lines["balance_y"].set_data(np.arange(len(yarr)), yarr)
            self._relim(self.ax_hip)

        cable_suffix = " | Active Cable ON" if self.active_cable else ""
        self._title.set_text(
            f"Suit Live Monitor{cable_suffix}"
            f"  |  env {sel}/{self.num_envs - 1}  |  step {self._step}"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ── CLI setup (simulator must be imported before torch) ───────────────────────


def _create_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Run suit robot inference with real-time live visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--motion-file", type=str, default=None)
    p.add_argument(
        "--simulator", type=str, default="newton",
        choices=["isaacgym", "isaaclab", "newton", "genesis", "mujoco"],
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--window",   type=int, default=300,
                   help="Scrolling history length (steps)")
    p.add_argument("--update-every", type=int, default=5,
                   help="Redraw every N steps")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument(
        "--active-cable",
        action="store_true",
        default=False,
        help="Show cable displacement and DOFC A버전 target panels",
    )
    p.add_argument(
        "--torque-dofs", nargs="*", default=None, metavar="SHORT_NAME",
        help="DOF short names to show in the Torques plot (partial match OK).",
    )
    p.add_argument(
        "--pos-dofs", nargs="*", default=None, metavar="SHORT_NAME",
        help="DOF short names to show in the Positions plot.",
    )
    p.add_argument("--overrides", nargs="*", default=None)
    return p


import argparse as _ap

_parser = _create_parser()
_args, _ = _parser.parse_known_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

_AppLauncher = import_simulator_before_torch(_args.simulator)

import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    args = _args
    checkpoint_path = Path(args.checkpoint)

    for cfg_name in ("resolved_configs_inference.pt", "resolved_configs.pt"):
        cfg_path = checkpoint_path.parent / cfg_name
        if cfg_path.exists():
            break
    resolved_configs = torch.load(str(cfg_path), weights_only=False)

    robot_config      = resolved_configs["robot"]
    simulator_config  = resolved_configs["simulator"]
    terrain_config    = resolved_configs.get("terrain")
    scene_lib_config  = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config        = resolved_configs["env"]
    agent_config      = resolved_configs["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file

    if args.overrides:
        from protomotions.utils.config_utils import apply_config_overrides, parse_cli_overrides

        apply_config_overrides(
            parse_cli_overrides(args.overrides),
            env_config, simulator_config, robot_config,
            agent_config, terrain_config, motion_lib_config, scene_lib_config,
        )

    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric = Fabric(**asdict(FabricConfig(
        accelerator=accelerator, devices=1, num_nodes=1, loggers=[], callbacks=[]
    )))
    fabric.launch()

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = _AppLauncher({"headless": True, "device": str(fabric.device)})
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.utils.component_builder import build_all_components

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=None,
        **simulator_extra_params,
    )

    from protomotions.envs.base_env.env import BaseEnv

    env: BaseEnv = get_class(env_config._target_)(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    agent: BaseAgent = get_class(agent_config._target_)(
        config=agent_config, env=env, fabric=fabric,
        root_dir=checkpoint_path.parent,
    )
    agent.setup()
    agent.load(args.checkpoint, load_env=False)
    agent.eval()

    sim = env.simulator

    dofc_fn = None
    if args.active_cable:
        from protomotions.envs.base_env.active_cable_env import _dofc_a_target_pos

        dofc_fn = _dofc_a_target_pos
        log.info("Active cable panels enabled (DOFC A버전).")

    # For BUILT_IN_PD simulators (Newton), dof_forces is not stored after stepping.
    # Compute torques analytically: tau = kp*(target - pos) - kd*vel
    pd_kp = pd_kd = None
    from protomotions.robot_configs.base import ControlType

    if sim.control_type == ControlType.BUILT_IN_PD:
        ctrl_info = robot_config.control.control_info
        dof_names_ordered = list(sim._dof_names)
        pd_kp = torch.tensor(
            [ctrl_info[n].stiffness for n in dof_names_ordered], device=fabric.device
        )
        pd_kd = torch.tensor(
            [ctrl_info[n].damping for n in dof_names_ordered], device=fabric.device
        )
        log.info(
            "BUILT_IN_PD detected — torques computed as kp*(target-pos) - kd*vel."
        )

    viz = LiveVisualizerSuit(
        body_names=sim._body_names,
        dof_names=sim._dof_names,
        num_envs=args.num_envs,
        window=args.window,
        update_every=args.update_every,
        torque_dof_filter=args.torque_dofs,
        pos_dof_filter=args.pos_dofs,
        active_cable=args.active_cable,
    )

    log.info(
        "Live monitor running. "
        "Click figure → use ←/→ or 0-9 to switch env, q to quit."
    )

    done_indices = None
    try:
        with torch.no_grad():
            while True:
                obs, _ = env.reset(done_indices)
                obs = agent.add_agent_info_to_obs(obs)
                obs_td = agent.obs_dict_to_tensordict(obs)

                model_outs = agent.model(obs_td)
                actions = model_outs.get("mean_action", model_outs["action"])

                obs, _rewards, dones, _terminated, _extras = env.step(actions)

                robot_state = sim.get_robot_state()
                try:
                    contact_forces = sim.get_bodies_contact_buf().rigid_body_contact_forces
                except Exception:
                    contact_forces = None

                pd_torques = None
                if pd_kp is not None and hasattr(env, "_current_processed_action"):
                    pd_target = env._current_processed_action
                    pd_torques = (
                        pd_kp * (pd_target - robot_state.dof_pos)
                        - pd_kd * robot_state.dof_vel
                    )

                keep_running = viz.update(
                    robot_state, contact_forces, actions,
                    dofc_fn=dofc_fn, override_torques=pd_torques,
                )
                if not keep_running:
                    break

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        if hasattr(sim, "shutdown"):
            sim.shutdown()


if __name__ == "__main__":
    main()
