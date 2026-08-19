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
"""휴머노이드 모션 → 5-IMU 관측 + hip 각도 타깃 ManiFlow zarr 추출기.

실착용 센서 구성(허벅지 L/R + 정강이 L/R + 몸통 IMU 5개)으로 관측 가능한
피처만 뽑아, hip 각도를 예측하는 ManiFlow lowdim 학습용 zarr를 만든다.
bones-seed(soma23)와 ETRI skeleton(suit) 두 임베디먼트를 지원하며, 피처와
타깃 모두 **skeleton 규약**(x-앞/y-왼쪽/z-위, hinge 각도)으로 통일한다 —
soma23으로 학습한 모델을 skeleton 모션/시뮬레이터에서 그대로 평가·배포 가능.

입력 (둘 중 하나 이상):
  --motion-file  MotionLib 패키지 .pt (gts/grs/gvs/gavs/dps + length_starts ...)
  --motion-dir   .motion 파일 트리 (convert_soma23_bvh_to_proto.py 출력, 재귀 탐색)
  --embodiment   soma23(기본) | skeleton_suit — 입력 모션의 신체 배치

관측 피처 (프레임당, 전부 world-yaw 불변 — 배포 시 헤딩 캘리브레이션 불필요):
  IMU 5개 각각 [몸통, 허벅지R, 정강이R, 허벅지L, 정강이L], 센서 프레임은
  임베디먼트 무관하게 skeleton 규약으로 정렬(soma23은 상수 회전 M으로 재매핑):
    grav  (3)  센서 좌표계에서 본 중력 방향 = R_sensor^T·[0,0,-1]
    gyro  (3)  센서 좌표계 각속도 = R_sensor^T·ω_world
    accel (3)  [--with-accel 시] 비력 = R_sensor^T·(dv/dt + [0,0,g])
  인접 세그먼트 상대 방향 4쌍 [(몸통→허벅지R),(허벅지R→정강이R),(몸통→허벅지L),
  (허벅지L→정강이L)]:
    rel6d (6)  R_rel = R_parent^T·R_child 의 앞 두 열 (연속 6D 표현)
  기본 상태 차원 = 5×6 + 4×6 = 54  (--with-accel 시 69)

타깃 규약 (--convention):
  skeleton (기본): hip 로컬 회전(R_pelvis^T·R_femur, global rot에서 직접 계산)을
    skeleton_torque_suit.xml의 hinge 순서로 분해한 6채널 각도(rad) =
    [hip_flexion_r, hip_adduction_r, hip_rotation_r, hip_flexion_l,
     hip_adduction_l, hip_rotation_l].
    합성 규약(MuJoCo 선언 순서, 수치 검증됨): R = R(flex)·R(add)·R(rot),
    right: flex(0,-1,0)/add(1,0,0)/rot(0,0,1), left: flex(0,-1,0)/add(-1,0,0)/
    rot(0,0,-1) — 즉 R = Ry(-θf)·Rx(σθa)·Rz(σθr), σ=+1(R)/-1(L).
    관절 한계로 클리핑하지 않는다(배포 시 PD가 클램프).
  soma23: 기존 exp-map 6채널(RightLeg_x/y/z + LeftLeg_x/y/z) — legacy 재현용.

출력 zarr (ManiFlow ReplayBuffer 레이아웃, zarr v2 강제 — maniflow env는 zarr 2.12):
  data/state        (N, D) float32
  data/action       (N, A) float32
  meta/episode_ends (num_episodes,) int64
  attrs: action_dof_names, obs_spec, convention, source 경로

사용 예 (sbc env, 저장소 루트에서):
  # seed locomotion → skeleton 규약 학습 데이터
  python data/scripts/extract_soma23_hip_imu_zarr.py \
      --motion-dir data/seed/soma23_proto_locomotion \
      --save-path ../ManiFlow_Policy/ManiFlow/data/hip-imu-seed-locomotion-skel.zarr
  # ETRI suit 모션 → cross-eval 데이터
  python data/scripts/extract_soma23_hip_imu_zarr.py \
      --motion-file data/motion_for_trackers/skeleton_torque_suit_motions14.pt \
      --embodiment skeleton_suit \
      --save-path ../ManiFlow_Policy/ManiFlow/data/hip-imu-etri-suit14.zarr
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRAVITY_MAG = 9.81

# soma23_humanoid.xml 의 고정 신체/DOF 배치 (extract_kinematic_info와 동일 순서).
# 패키지 .pt / .motion 파일에는 이름이 저장되지 않으므로 상수로 둔다.
SOMA23_BODY_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
]
SOMA23_DOF_NAMES = [
    f"{b}_{ax}" for b in SOMA23_BODY_NAMES[1:] for ax in ("x", "y", "z")
]

# skeleton_torque_suit.xml 의 신체 배치 (robot_config('skeleton_torque_suit') 순서)
SKELETON_SUIT_BODY_NAMES = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "RH_dump", "RH_dump2", "femur_l", "tibia_l", "talus_l", "calcn_l",
    "toes_l", "LH_dump", "LH_dump2", "torso", "humerus_r", "ulna_r",
    "radius_r", "hand_r", "humerus_l", "ulna_l", "radius_l", "hand_l",
    "slider1", "slider2", "slider3", "slider4",
]

# 좌표 규약: soma23 body 프레임 = (+x 왼쪽, +y 뒤, +z 위),
# skeleton = (+x 앞, +y 왼쪽, +z 위). 두 모델 모두 rest에서 world-aligned.
# M_SOMA 열 = skeleton 축을 soma 좌표로 표현: x_k=(0,-1,0), y_k=(1,0,0), z_k=(0,0,1)
M_SOMA = torch.tensor([[0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])
M_IDENTITY = torch.eye(3)

# 임베디먼트별: (body_names, IMU 5개 [몸통,허벅지R,정강이R,허벅지L,정강이L],
#               상대쌍 4개, 센서 프레임 재매핑 M, (pelvis, femur_r, femur_l))
# skeleton_suit 의 "fk_from_dofs": suit 모션 .pt의 grs는 retarget 소스 프레임이라
# MJCF body 프레임과 상수 회전만큼 어긋남(무릎 축 검증으로 확인). 각속도·선속도는
# 상수 오프셋에 불변이라 저장값을 쓰되, **회전은 dps(hinge 각도, 신뢰 가능) +
# 루트 자세로 closed-form FK 재계산**한다.
EMBODIMENTS = {
    "soma23": {
        "body_names": SOMA23_BODY_NAMES,
        "imu_bodies": ["Chest", "RightLeg", "RightShin", "LeftLeg", "LeftShin"],
        "rel_pairs": [("Chest", "RightLeg"), ("RightLeg", "RightShin"),
                      ("Chest", "LeftLeg"), ("LeftLeg", "LeftShin")],
        "frame_M": M_SOMA,
        "hip_bodies": ("Hips", "RightLeg", "LeftLeg"),
        "fk_from_dofs": False,
    },
    "skeleton_suit": {
        "body_names": SKELETON_SUIT_BODY_NAMES,
        "imu_bodies": ["torso", "femur_r", "tibia_r", "femur_l", "tibia_l"],
        "rel_pairs": [("torso", "femur_r"), ("femur_r", "tibia_r"),
                      ("torso", "femur_l"), ("femur_l", "tibia_l")],
        "frame_M": M_IDENTITY,
        "hip_bodies": ("pelvis", "femur_r", "femur_l"),
        "fk_from_dofs": True,
    },
}

SKELETON_HIP_NAMES = [
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
]
SOMA23_HIP_DOF_IDX = [42, 43, 44, 54, 55, 56]  # RightLeg_x/y/z + LeftLeg_x/y/z


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """xyzw 쿼터니언 → 회전행렬 (..., 3, 3)."""
    x, y, z, w = q.unbind(-1)
    row0 = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1
    )
    row1 = torch.stack(
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1
    )
    row2 = torch.stack(
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_expmap(R: torch.Tensor) -> torch.Tensor:
    """회전행렬 배치 (T,3,3) → exp-map (T,3) — Rodrigues 로그, 소각 안정화."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0))
    vec = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                       R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    s = torch.sin(ang)
    scale = torch.where(s.abs() > 1e-6, ang / (2 * s), torch.full_like(ang, 0.5))
    return vec * scale.unsqueeze(-1)


def expmap_to_matrix(em: torch.Tensor) -> torch.Tensor:
    """exp-map 배치 (T,3) → 회전행렬 (T,3,3) — Rodrigues, 소각 안정화."""
    ang = em.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ax = em / ang
    zeros = torch.zeros_like(ax[:, 0])
    K = torch.stack([
        torch.stack([zeros, -ax[:, 2], ax[:, 1]], -1),
        torch.stack([ax[:, 2], zeros, -ax[:, 0]], -1),
        torch.stack([-ax[:, 1], ax[:, 0], zeros], -1),
    ], dim=-2)
    a = ang[:, :, None]
    eye = torch.eye(3, dtype=em.dtype).expand(em.shape[0], 3, 3)
    return eye + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)


def decompose_hip_yxz(R: torch.Tensor, side_sign: float) -> torch.Tensor:
    """skeleton hip hinge 분해: R = Ry(-θf)·Rx(σθa)·Rz(σθr) → (θf, θa, θr).

    intrinsic YXZ (R = Ry(α)Rx(β)Rz(γ)) 추출 후 부호 보정:
      β = -asin(R[1,2]);  α = atan2(R[0,2], R[2,2]);  γ = atan2(R[1,0], R[1,1])
      θf = -α,  θa = σβ,  θr = σγ   (σ = +1 오른쪽 / -1 왼쪽)
    특이점은 adduction ±90°인데 인체 hip에선 도달 불가.
    """
    beta = -torch.asin(R[..., 1, 2].clamp(-1.0, 1.0))
    alpha = torch.atan2(R[..., 0, 2], R[..., 2, 2])
    gamma = torch.atan2(R[..., 1, 0], R[..., 1, 1])
    return torch.stack([-alpha, side_sign * beta, side_sign * gamma], dim=-1)


def hip_angles_skeleton(grs: torch.Tensor, emb: dict) -> torch.Tensor:
    """global rot에서 hip 로컬 회전을 구해 skeleton hinge 6채널(rad)로 분해."""
    idx = {n: i for i, n in enumerate(emb["body_names"])}
    pelvis, femur_r, femur_l = emb["hip_bodies"]
    M = emb["frame_M"]
    R_p = quat_to_matrix(grs[:, idx[pelvis]])
    out = []
    for femur, sign in ((femur_r, 1.0), (femur_l, -1.0)):
        R_f = quat_to_matrix(grs[:, idx[femur]])
        R_local = R_p.transpose(-1, -2) @ R_f          # 임베디먼트 좌표
        R_k = M.T @ R_local @ M                        # skeleton 좌표로 변환
        out.append(decompose_hip_yxz(R_k, sign))
    return torch.cat(out, dim=-1)  # (T, 6)


def _axis_rot(axis, theta: torch.Tensor) -> torch.Tensor:
    """고정축 회전행렬 배치 (T, 3, 3) — Rodrigues."""
    a = torch.tensor(axis, dtype=theta.dtype)
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]],
                     dtype=theta.dtype)
    s = torch.sin(theta)[:, None, None]
    c = torch.cos(theta)[:, None, None]
    eye = torch.eye(3, dtype=theta.dtype).expand(theta.shape[0], 3, 3)
    return eye + s * K + (1 - c) * (K @ K)


# skeleton_torque_suit.xml 관절 정의 (dof 인덱스, [(축, ...)] 선언 순서 합성)
SUIT_DOF_IDX = {"hip_r": [0, 1, 2], "knee_r": [3], "hip_l": [5, 6, 7],
                "knee_l": [8], "lumbar": [10, 11, 12]}
SUIT_HIP_AXES_R = [(0, -1, 0), (1, 0, 0), (0, 0, 1)]
SUIT_HIP_AXES_L = [(0, -1, 0), (-1, 0, 0), (0, 0, -1)]
SUIT_KNEE_AXIS = (0, -1, 0)
SUIT_LUMBAR_AXES = [(0, -1, 0), (1, 0, 0), (0, 0, 1)]


def suit_fk_rotations(grs: torch.Tensor, dps: torch.Tensor) -> dict:
    """suit IMU 신체들의 MJCF-프레임 회전을 dps + 루트 자세로 재계산.

    suit 모션 .pt의 grs(비루트)는 retarget 소스 프레임이라 그대로 못 쓴다.
    체인: pelvis(grs[:,0]) → femur(hip 3힌지) → tibia(knee), pelvis → torso(lumbar).
    """
    def chain(R_parent, axes, thetas):
        R = R_parent
        for ax, th in zip(axes, thetas):
            R = R @ _axis_rot(ax, th)
        return R

    R_p = quat_to_matrix(grs[:, 0])  # pelvis (루트 자세는 신뢰 가능 — 직립 검증)
    d = dps.to(R_p.dtype)
    out = {"pelvis": R_p}
    out["femur_r"] = chain(R_p, SUIT_HIP_AXES_R, [d[:, i] for i in SUIT_DOF_IDX["hip_r"]])
    out["femur_l"] = chain(R_p, SUIT_HIP_AXES_L, [d[:, i] for i in SUIT_DOF_IDX["hip_l"]])
    out["tibia_r"] = out["femur_r"] @ _axis_rot(SUIT_KNEE_AXIS, d[:, SUIT_DOF_IDX["knee_r"][0]])
    out["tibia_l"] = out["femur_l"] @ _axis_rot(SUIT_KNEE_AXIS, d[:, SUIT_DOF_IDX["knee_l"][0]])
    out["torso"] = chain(R_p, SUIT_LUMBAR_AXES, [d[:, i] for i in SUIT_DOF_IDX["lumbar"]])
    return out


def hinge6(grs: torch.Tensor, dps: torch.Tensor, emb: dict) -> torch.Tensor:
    """skeleton hinge 6채널: suit는 dps 직접, soma23은 분해."""
    if emb["fk_from_dofs"]:
        hip_idx = SUIT_DOF_IDX["hip_r"] + SUIT_DOF_IDX["hip_l"]
        return dps[:, hip_idx].float()
    return hip_angles_skeleton(grs, emb).float()


def compute_flexion_trunk_features(
    grs: torch.Tensor, dps: torch.Tensor, emb: dict
) -> tuple:
    """각도 직접 입력 모드: obs = [hip_flexion_r/l, trunk_pitch, trunk_roll].

    실기 가정: hip flexion은 슈트 관절 센서(엔코더 상당)로 직접 측정, 상체는
    몸통 IMU의 융합 각도(pitch=앞숙임+, roll=오른쪽 기울임+)만 사용.
    action = [hip_flexion_r, hip_flexion_l] (flexion 1자유도 × 좌우).
    """
    T = grs.shape[0]
    h6 = hinge6(grs, dps, emb)
    flex = h6[:, [0, 3]]  # hip_flexion_r, hip_flexion_l

    # 몸통(리스트 첫 IMU) 방향 → skeleton 규약(x-앞,y-왼,z-위) 중력 → pitch/roll
    idx = {n: i for i, n in enumerate(emb["body_names"])}
    torso = emb["imu_bodies"][0]
    if emb["fk_from_dofs"]:
        R_t = suit_fk_rotations(grs, dps)[torso]
    else:
        R_t = quat_to_matrix(grs[:, idx[torso]]) @ emb["frame_M"]
    down = torch.tensor([0.0, 0.0, -1.0]).expand(T, 3)
    g = torch.einsum("tij,tj->ti", R_t.transpose(-1, -2), down.to(R_t.dtype))
    pitch = torch.atan2(g[:, 0], -g[:, 2])   # 앞으로 숙이면 +
    roll = torch.atan2(-g[:, 1], -g[:, 2])   # 오른쪽으로 기울면 +
    state = torch.cat([flex, pitch[:, None], roll[:, None]], dim=-1)
    return state.float().numpy(), flex.float().numpy()


def compute_episode_features(
    grs: torch.Tensor,   # (T, B, 4) xyzw 글로벌 회전
    gavs: torch.Tensor,  # (T, B, 3) 글로벌 각속도
    gvs: torch.Tensor,   # (T, B, 3) 글로벌 선속도
    dps: torch.Tensor,   # (T, D)   dof_pos
    dt: float,
    emb: dict,
    convention: str,
    with_accel: bool,
) -> tuple:
    """한 에피소드의 (state, action) float32 numpy 반환."""
    T = grs.shape[0]
    idx = {n: i for i, n in enumerate(emb["body_names"])}
    # 센서 프레임 = body 프레임 · M. 목표 규약(convention)에 맞춰 정렬:
    #   skeleton 규약: soma23 body→M_SOMA, suit body→I
    #   soma23 규약(legacy run01 계약): soma23 body→I, suit body→M_SOMA^T
    if convention == "skeleton":
        M = emb["frame_M"]
    else:
        M = M_IDENTITY if not emb["fk_from_dofs"] else M_SOMA.T.contiguous()

    # IMU 신체 회전행렬: grs 직접(soma23) 또는 dps 기반 FK(skeleton_suit)
    if emb["fk_from_dofs"]:
        R_body = suit_fk_rotations(grs, dps)
    else:
        R_body = {n: quat_to_matrix(grs[:, idx[n]]) for n in emb["imu_bodies"]}

    def to_sensor(name: str, v_world: torch.Tensor) -> torch.Tensor:
        # R_sensor^T·v = M^T·(R_body^T·v)
        return torch.einsum("tij,tj->ti", R_body[name].transpose(-1, -2), v_world) @ M

    feats = []
    down = torch.tensor([0.0, 0.0, -1.0]).expand(T, 3)
    for name in emb["imu_bodies"]:
        feats.append(to_sensor(name, down))                    # grav (3)
        # 각속도·선속도는 프레임 상수 오프셋에 불변 → 저장값 사용
        feats.append(to_sensor(name, gavs[:, idx[name]]))      # gyro (3)
        if with_accel:
            v = gvs[:, idx[name]]
            dv = torch.zeros_like(v)
            dv[1:] = (v[1:] - v[:-1]) / dt
            dv[0] = dv[1] if T > 1 else 0.0
            spec = dv + torch.tensor([0.0, 0.0, GRAVITY_MAG])
            feats.append(to_sensor(name, spec))                # accel (3)

    for parent, child in emb["rel_pairs"]:
        R_rel = R_body[parent].transpose(-1, -2) @ R_body[child]
        R_rel = M.T @ R_rel @ M                                # 센서 프레임 정렬
        feats.append(R_rel[..., :2].transpose(-1, -2).reshape(T, 6))  # rel6d (6)

    state = torch.cat(feats, dim=-1).float().numpy()

    if convention == "skeleton":
        if emb["fk_from_dofs"]:
            # dps가 곧 skeleton hinge 각도 (hip 6채널 직접 사용)
            hip_idx = SUIT_DOF_IDX["hip_r"] + SUIT_DOF_IDX["hip_l"]
            action = dps[:, hip_idx].float().numpy()
        else:
            action = hip_angles_skeleton(grs, emb).float().numpy()
    else:  # soma23 exp-map (legacy run01 계약)
        if emb["fk_from_dofs"]:
            # suit hinge 각도 → 로컬 회전 합성 → soma 좌표 → exp-map
            d = dps.to(torch.float64)
            out = []
            for key, axes in (("hip_r", SUIT_HIP_AXES_R), ("hip_l", SUIT_HIP_AXES_L)):
                R_k = None
                for ax, di in zip(axes, SUIT_DOF_IDX[key]):
                    Ri = _axis_rot(ax, d[:, di])
                    R_k = Ri if R_k is None else R_k @ Ri
                R_s = M_SOMA.double() @ R_k @ M_SOMA.double().T
                out.append(matrix_to_expmap(R_s))
            action = torch.cat(out, dim=-1).float().numpy()
        else:
            action = dps[:, SOMA23_HIP_DOF_IDX].float().numpy()
    return state, action


def iter_packaged(path: str, num_bodies: int):
    """MotionLib 패키지 .pt → 에피소드 (grs, gavs, gvs, dps, dt, 이름) 이터레이터."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d["grs"].shape[1] == num_bodies, (
        f"{path}: {d['grs'].shape[1]} bodies != embodiment({num_bodies})"
    )
    starts = d["length_starts"].tolist()
    frames = d["motion_num_frames"].tolist()
    dts = d["motion_dt"].tolist()
    files = [str(f) for f in d["motion_files"]]
    for i, (s, n) in enumerate(zip(starts, frames)):
        e = s + n
        yield (
            d["grs"][s:e], d["gavs"][s:e], d["gvs"][s:e], d["dps"][s:e],
            float(dts[i]), files[i],
        )


def iter_motion_dir(root: str, num_bodies: int):
    """.motion 파일 트리 → 에피소드 이터레이터 (재귀, 정렬 순회)."""
    for dirpath, _dirs, files in sorted(os.walk(root)):
        for f in sorted(files):
            if not f.endswith(".motion"):
                continue
            p = os.path.join(dirpath, f)
            m = torch.load(p, map_location="cpu", weights_only=False)
            if m["rigid_body_rot"].shape[1] != num_bodies:
                print(f"  skip {p}: {m['rigid_body_rot'].shape[1]} bodies")
                continue
            yield (
                m["rigid_body_rot"], m["rigid_body_ang_vel"],
                m["rigid_body_vel"], m["dof_pos"],
                1.0 / float(m["fps"]), p,
            )


def main():
    parser = argparse.ArgumentParser(
        description="휴머노이드 모션 → 5-IMU/hip-각도 ManiFlow zarr"
    )
    parser.add_argument("--motion-file", nargs="*", default=[],
                        help="MotionLib 패키지 .pt (복수 가능)")
    parser.add_argument("--motion-dir", nargs="*", default=[],
                        help=".motion 파일 루트 디렉토리 (재귀 탐색, 복수 가능)")
    parser.add_argument("--save-path", required=True, help="출력 zarr 경로")
    parser.add_argument("--embodiment", choices=sorted(EMBODIMENTS),
                        default="soma23", help="입력 모션의 신체 배치")
    parser.add_argument("--convention", choices=["skeleton", "soma23"],
                        default="skeleton",
                        help="타깃 각도 규약 (기본: skeleton hinge)")
    parser.add_argument("--features", choices=["imu5", "flexion_trunk"],
                        default="imu5",
                        help="관측 피처: imu5(54ch 시뮬레이션 IMU) | "
                             "flexion_trunk(4ch = hip flexion L/R + 상체 pitch/roll, "
                             "타깃도 flexion 2ch로 고정)")
    parser.add_argument("--with-accel", action="store_true",
                        help="가속도계 피처 포함 (mocap 유한차분, 기본 제외)")
    parser.add_argument("--min-frames", type=int, default=16,
                        help="이보다 짧은 에피소드는 제외 (horizon 미만 방지)")
    parser.add_argument("--resample-fps", type=float, default=None,
                        help="에피소드를 이 fps로 시간 리샘플 (선형 보간, "
                             "flexion_trunk 전용 — 각도 채널이라 안전). "
                             "예: 40fps suit 모션 → 30fps 학습 데이터 정합")
    args = parser.parse_args()

    if not args.motion_file and not args.motion_dir:
        raise SystemExit("--motion-file 또는 --motion-dir 를 하나 이상 지정하세요")

    emb = EMBODIMENTS[args.embodiment]
    num_bodies = len(emb["body_names"])
    if args.features == "flexion_trunk":
        args.convention = "skeleton"  # 각도 모드는 skeleton hinge 고정
        action_names = ["hip_flexion_r", "hip_flexion_l"]
    elif args.convention == "skeleton":
        action_names = list(SKELETON_HIP_NAMES)
    else:
        action_names = [SOMA23_DOF_NAMES[i] for i in SOMA23_HIP_DOF_IDX]
    print(f"embodiment: {args.embodiment} ({num_bodies} bodies), "
          f"convention: {args.convention}, features: {args.features}")
    print(f"targets: {action_names}")

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    import zarr

    if zarr.__version__.startswith("3"):
        # maniflow 학습 env는 zarr 2.12 — 루트뿐 아니라 하위 그룹/배열까지
        # 전부 v2 포맷으로 강제해야 읽을 수 있다.
        zarr.config.set({"default_zarr_format": 2})
    root = zarr.open(args.save_path, mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")

    state_store = None
    action_store = None
    episode_ends = []
    motion_names = []
    total = 0
    skipped_short = 0

    def sources():
        for p in args.motion_file:
            print(f"Loading packaged {p}")
            yield from iter_packaged(p, num_bodies)
        for d in args.motion_dir:
            print(f"Walking {d}")
            yield from iter_motion_dir(d, num_bodies)

    for grs, gavs, gvs, dps, dt, name in sources():
        if grs.shape[0] < args.min_frames:
            skipped_short += 1
            continue
        if args.features == "flexion_trunk":
            state, action = compute_flexion_trunk_features(grs, dps, emb)
        else:
            state, action = compute_episode_features(
                grs, gavs, gvs, dps, dt, emb, args.convention, args.with_accel
            )
        if args.resample_fps and abs(dt * args.resample_fps - 1.0) > 1e-3:
            assert args.features == "flexion_trunk", (
                "--resample-fps 는 flexion_trunk(순수 각도 채널)에서만 지원")
            t_old = np.arange(state.shape[0]) * dt
            t_new = np.arange(0.0, t_old[-1], 1.0 / args.resample_fps)
            state = np.stack(
                [np.interp(t_new, t_old, state[:, c]) for c in range(state.shape[1])],
                axis=-1).astype(np.float32)
            action = state[:, :action.shape[1]].copy()
        if state_store is None:
            chunk_rows = 4096
            state_store = data.zeros(
                name="state", shape=(0, state.shape[1]),
                chunks=(chunk_rows, state.shape[1]), dtype="float32",
            )
            action_store = data.zeros(
                name="action", shape=(0, action.shape[1]),
                chunks=(chunk_rows, action.shape[1]), dtype="float32",
            )
            print(f"state_dim: {state.shape[1]}, action_dim: {action.shape[1]}")
        state_store.append(state)
        action_store.append(action)
        total += state.shape[0]
        episode_ends.append(total)
        motion_names.append(os.path.basename(name))
        if len(episode_ends) % 500 == 0:
            print(f"  {len(episode_ends)} episodes, {total} frames")

    if not episode_ends:
        raise SystemExit("에피소드가 하나도 없습니다 — 입력 경로를 확인하세요")

    ends = np.array(episode_ends, dtype=np.int64)
    ends_store = meta.zeros(
        name="episode_ends", shape=(0,), chunks=(65536,), dtype="int64"
    )
    ends_store.append(ends)

    root.attrs["action_dof_names"] = action_names
    root.attrs["convention"] = args.convention
    root.attrs["embodiment"] = args.embodiment
    root.attrs["features"] = args.features
    if args.features == "flexion_trunk":
        root.attrs["obs_spec"] = json.dumps({
            "channels": ["hip_flexion_r", "hip_flexion_l",
                         "trunk_pitch", "trunk_roll"],
            "units": "rad; pitch=앞숙임+, roll=오른쪽+",
            "trunk_body": emb["imu_bodies"][0],
            "conventions": "skeleton hinge (unclipped)",
        })
    else:
        imu_blocks = ["grav3", "gyro3"] + (["accel3"] if args.with_accel else [])
        root.attrs["obs_spec"] = json.dumps({
            "imu_bodies": emb["imu_bodies"],
            "per_imu_blocks": imu_blocks,
            "rel6d_pairs": [list(p) for p in emb["rel_pairs"]],
            "layout": "per-IMU blocks first (IMU order), then rel6d pairs",
            "sensor_frame": "skeleton convention (x-fwd, y-left, z-up), "
                            "world-yaw invariant",
            "conventions": "quat xyzw / z-up; skeleton targets = hinge rad "
                           "R=Ry(-f)Rx(sa)Rz(sr), s=+1 R/-1 L; unclipped",
        })
    root.attrs["source_paths"] = (
        [os.path.abspath(p) for p in args.motion_file]
        + [os.path.abspath(p) for p in args.motion_dir]
    )
    root.attrs["motion_names"] = motion_names

    print(f"Saved {args.save_path}")
    print(f"  episodes: {len(episode_ends)} (short-skipped {skipped_short}), frames: {total}")
    print(f"  state:  {state_store.shape}")
    print(f"  action: {action_store.shape} = {action_names}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
