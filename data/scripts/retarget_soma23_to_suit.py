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
"""Retarget native SOMA23 clips to the 27-DOF suit, with an explicit audit.

The original timestamps and horizontal pelvis trajectory are retained. Bounded
whole-body orientation fitting projects the source onto the suit's joints. Arm
T-pose frames are mapped to the suit's arms-down frames. All exported body
transforms and velocities are recomputed from suit FK. A single vertical shift
per clip places the lowest foot collision surface on the floor; it does not
lock feet, stitch clips, rescale time, or certify dynamic feasibility.

Run with the sbc interpreter, from the repository root. The candidate manifest
comes from the source audit documented in docs/ai_sbc/NEXT_SESSION.md, section 2-Y.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

from data.scripts.extract_soma23_hip_imu_zarr import M_SOMA, SOMA23_BODY_NAMES
from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)

ASSET = Path("protomotions/data/assets/mjcf/skeleton_torque_suit.xml")
BODY_MAP = {
    "femur_r": "RightLeg", "tibia_r": "RightShin", "talus_r": "RightFoot",
    "femur_l": "LeftLeg", "tibia_l": "LeftShin", "talus_l": "LeftFoot",
    "torso": "Chest", "humerus_r": "RightArm", "ulna_r": "RightForeArm",
    "hand_r": "RightHand", "humerus_l": "LeftArm", "ulna_l": "LeftForeArm",
    "hand_l": "LeftHand",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model():
    # Only collision geometry is needed; avoid loading anatomical visual meshes.
    tree = ET.parse(ASSET)
    for parent in tree.iter():
        for node in list(parent):
            if node.tag == "mesh" or (node.tag == "geom" and node.get("mesh")):
                parent.remove(node)
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))


def foot_surfaces(model, data):
    """Exact bottom of each foot box/sphere, using the collision geometry."""
    values = []
    for side in ("r", "l"):
        bottoms = []
        for name in (f"calcn_{side}_col", f"toes_{side}_col", f"talus_{side}_col"):
            geom = model.geom(name).id
            rotation = data.geom_xmat[geom].reshape(3, 3)
            extent = (np.abs(rotation[2]) @ model.geom_size[geom]
                      if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                      else model.geom_size[geom, 0])
            bottoms.append(data.geom_xpos[geom, 2] - extent)
        values.append(min(bottoms))
    return np.asarray(values)


def retarget(source, model):
    fps = float(source["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be positive and finite")
    if source["rigid_body_pos"].shape[1:] != (23, 3) or source["dof_pos"].shape[1] != 66:
        raise ValueError("Expected SOMA23, 23 bodies / 66 DOFs")
    if len(source["rigid_body_pos"]) < 2:
        raise ValueError("At least two source frames are required")
    src_rot = Rotation.from_quat(source["rigid_body_rot"].numpy().reshape(-1, 4))
    src_rot = src_rot.as_matrix().reshape(-1, 23, 3, 3)
    src_pos = source["rigid_body_pos"].numpy()
    if not np.isfinite(src_pos).all() or not np.isfinite(src_rot).all():
        raise ValueError("Nonfinite source transforms")
    n = len(src_pos)
    mapping = M_SOMA.numpy().astype(float)
    root_rot = src_rot[:, 0] @ mapping
    target_ids = [model.body(name).id for name in BODY_MAP]
    target_rot = []
    for suit, soma in BODY_MAP.items():
        correction = mapping
        if suit.startswith(("humerus", "ulna", "hand")):
            angle = -np.pi / 2 if suit.endswith("_r") else np.pi / 2
            correction = mapping @ Rotation.from_euler("x", angle).as_matrix()
        target_rot.append(src_rot[:, SOMA23_BODY_NAMES.index(soma)] @ correction)
    target_rot = np.stack(target_rot, axis=1)
    weights = np.array([1.0 if name.startswith(("femur", "torso")) else 0.7
                        for name in BODY_MAP])[:, None, None]
    bounds = (model.jnt_range[1:24, 0], model.jnt_range[1:24, 1])
    assert model.nq == 34 and model.nv == 33
    qpos = np.tile(model.qpos0, (n, 1))
    qpos[:, :3] = src_pos[:, 0]
    qpos[:, :2] -= qpos[0, :2].copy()
    qpos[:, 3:7] = Rotation.from_matrix(root_rot).as_quat()[:, [3, 0, 1, 2]]
    data = mujoco.MjData(model)
    previous = np.zeros(23)
    errors, surfaces, statuses, nfev = [], [], [], []
    for t in range(n):
        data.qpos[:] = qpos[t]

        def residual(angles):
            data.qpos[7:30] = angles
            mujoco.mj_kinematics(model, data)
            actual = data.xmat[target_ids].reshape(-1, 3, 3)
            # Chordal rotation error plus a weak continuity term (no time filter).
            return np.concatenate((((actual - target_rot[t]) * weights).ravel(),
                                   0.01 * (angles - previous)))

        solution = least_squares(residual, np.clip(previous, *bounds), bounds=bounds,
                                 max_nfev=60, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        qpos[t, 7:30] = solution.x
        previous = solution.x.copy()
        residual(solution.x)
        actual = data.xmat[target_ids].reshape(-1, 3, 3)
        delta = np.swapaxes(target_rot[t], -1, -2) @ actual
        errors.append(np.rad2deg(Rotation.from_matrix(delta).magnitude()))
        surfaces.append(foot_surfaces(model, data))
        statuses.append(int(solution.status))
        nfev.append(int(solution.nfev))
    surfaces = np.asarray(surfaces)
    z_shift = -float(surfaces.min())
    qpos[:, 2] += z_shift
    surfaces += z_shift
    kin = extract_kinematic_info(str(ASSET))
    q = torch.tensor(qpos, dtype=torch.float32)
    motion = fk_batch_mjcf_with_velocities(kin, q, fps=fps, velocity_max_horizon=1)
    motion.dof_pos = q[:, 7:]
    motion.dof_vel = compute_cartesian_velocity(q[:, None, 7:], fps, velocity_max_horizon=1)[:, 0]
    # These are geometry-derived labels, never force measurements.
    motion.rigid_body_contacts = torch.zeros((n, kin.num_bodies), dtype=torch.bool)
    foot_speed = []
    source_contacts = []
    for side, soma in (("r", "RightFoot"), ("l", "LeftFoot")):
        idx = kin.body_names.index(f"calcn_{side}")
        speed = motion.rigid_body_vel[:, idx, :].norm(dim=-1).numpy()
        foot_speed.append(speed)
        contact = (surfaces[:, len(foot_speed)-1] < 0.035) & (speed < 0.2)
        for name in (f"calcn_{side}", f"toes_{side}"):
            motion.rigid_body_contacts[:, kin.body_names.index(name)] = torch.from_numpy(contact)
        source_contacts.append(source["rigid_body_contacts"][:, SOMA23_BODY_NAMES.index(soma)].numpy())
    foot_speed = np.stack(foot_speed, 1)
    source_contacts = np.stack(source_contacts, 1).astype(bool)
    # Independent MuJoCo vs ProtoMotions FK check on the exported samples.
    fk_errors = []
    for i in np.linspace(0, n-1, min(n, 20), dtype=int):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        fk_errors.append(np.max(np.abs(data.xpos[1:] - motion.rigid_body_pos[i].numpy())))
    errors = np.asarray(errors)
    at_bounds = np.isclose(qpos[:, 7:30], bounds[0], atol=1e-4) | np.isclose(qpos[:, 7:30], bounds[1], atol=1e-4)
    audit = {
        "fps": fps, "frames": n, "duration_s": (n-1)/fps,
        "root_xy_speed_m_s": float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=-1).sum()*fps/(n-1)),
        "root_xy_preservation_max_m": float(np.max(np.abs(np.diff(qpos[:, :2], axis=0)-np.diff(src_pos[:, 0, :2], axis=0)))),
        "vertical_shift_m": z_shift,
        "minimum_foot_collision_height_m": float(surfaces.min()),
        "lower_foot_height_p95_m": float(np.percentile(surfaces.min(axis=1), 95)),
        "source_contact_foot_speed_mean_m_s": float(foot_speed[source_contacts].mean()) if source_contacts.any() else None,
        "source_contact_foot_height_mean_m": float(surfaces[source_contacts].mean()) if source_contacts.any() else None,
        "source_contact_foot_height_p95_m": float(np.percentile(surfaces[source_contacts], 95)) if source_contacts.any() else None,
        "no_foot_contact_fraction": float((~motion.rigid_body_contacts.any(dim=1)).float().mean()),
        "joint_limit_violation_max_rad": float(max(0.0, np.max(bounds[0]-qpos[:, 7:30]), np.max(qpos[:, 7:30]-bounds[1]))),
        "at_limit_fraction_by_joint": {name: float(at_bounds[:, i].mean()) for i,name in enumerate(kin.dof_names[:23])},
        "orientation_rmse_deg": {name: float(np.sqrt((errors[:, i]**2).mean())) for i,name in enumerate(BODY_MAP)},
        "max_joint_speed_rad_s": float(motion.dof_vel[:, :23].abs().max()),
        "max_joint_frame_change_rad": float(np.abs(np.diff(qpos[:, 7:30], axis=0)).max()),
        "mujoco_fk_max_position_error_m": float(max(fk_errors)),
        "optimizer_failed_frames": sum(status <= 0 for status in statuses),
        "optimizer_nfev_max": max(nfev),
        "contact_method": "foot collision bottom <35mm and calcn origin speed <0.2m/s; kinematic heuristic",
        "validation_status": "kinematic_candidate_only_not_training_approved",
    }
    return motion, audit, qpos, surfaces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=["slow", "normal_pace"])
    p.add_argument("--actors", nargs="+")
    p.add_argument("--names", nargs="+")
    p.add_argument("--use-audited-window", action="store_true",
                   help="Crop to the source audit's most even 4-second window; preserve fps")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "retarget_audit.json").exists():
        raise FileExistsError(args.out / "retarget_audit.json")
    source_manifest = json.loads(args.manifest.read_text())
    selected = [c for c in source_manifest["candidates"] if c["label"] in args.labels
                and (not args.actors or c["actor"] in args.actors)
                and (not args.names or c["name"] in args.names)]
    model = load_model()
    report = {"source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": sha256(args.manifest),
              "asset_sha256": sha256(ASSET), "script_sha256": sha256(__file__), "clips": []}
    frames = {k: [] for k in ("gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts")}
    field_names = ["rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel", "rigid_body_contacts"]
    starts, lengths, dts, counts, names = [], [], [], [], []
    cursor = 0
    for candidate in selected:
        path = Path(candidate["motion_path"])
        if sha256(path) != candidate["motion_sha256"]:
            raise ValueError(f"Source changed: {path}")
        source = torch.load(path, map_location="cpu", weights_only=False)
        source_start = 0.0
        if args.use_audited_window:
            window = candidate["most_even_4s_window"]
            if window is None:
                raise ValueError(f"No audited window: {path}")
            source_start = float(window["start_s"])
            first = round(source_start * source["fps"])
            last = round(window["end_s"] * source["fps"]) + 1
            total = len(source["dof_pos"])
            source = {k: v[first:last].clone() if torch.is_tensor(v) and v.ndim and len(v) == total else v
                      for k, v in source.items()}
        motion, audit, qpos, surfaces = retarget(source, model)
        out_path = args.out / (candidate["name"] + ".motion")
        if out_path.exists():
            raise FileExistsError(out_path)
        torch.save(motion.to_dict(), out_path)
        np.savez_compressed(out_path.with_suffix(".npz"), qpos=qpos, foot_collision_bottom=surfaces)
        audit.update({"name": candidate["name"], "actor": candidate["actor"], "label": candidate["label"],
                      "source": str(path), "source_sha256": candidate["motion_sha256"],
                      "source_start_s": source_start,
                      "raw_bvh": candidate["raw_path"], "raw_bvh_sha256": candidate["raw_sha256"],
                      "output": str(out_path.resolve()), "output_sha256": sha256(out_path),
                      "hlp_source_pool_contains": candidate["hlp_source_pool_contains"]})
        report["clips"].append(audit)
        (args.out / "retarget_audit.json").write_text(json.dumps(report, indent=2))
        for key, attr in zip(frames, field_names):
            frames[key].append(getattr(motion, attr))
        starts.append(cursor); counts.append(audit["frames"]); cursor += audit["frames"]
        lengths.append(audit["duration_s"]); dts.append(1/audit["fps"]); names.append(str(out_path.resolve()))
        print(candidate["name"], json.dumps({k:audit[k] for k in ("root_xy_speed_m_s", "vertical_shift_m", "source_contact_foot_speed_mean_m_s", "lower_foot_height_p95_m", "optimizer_failed_frames")}), flush=True)
    if not selected:
        raise ValueError("No clips selected")
    package = {k: torch.cat(v) for k,v in frames.items()}
    package.update(motion_files=tuple(names), length_starts=torch.tensor(starts), motion_num_frames=torch.tensor(counts),
                   motion_lengths=torch.tensor(lengths), motion_dt=torch.tensor(dts), motion_weights=torch.ones(len(names))/len(names))
    torch.save(package, args.out / "suit_native_walk.pt")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
