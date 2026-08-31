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
"""ManiFlowAngleEstimator 오프라인 등가성 검증 (AI-SBC Phase B).

suit14 평가 zarr(40 fps)를 **한 프레임씩 스트리밍**으로
``ManiFlowAngleEstimator``에 넣고 receding-horizon 예측을 수행해, 오프라인
기준 스크립트(ManiFlow_Policy/scripts/compare_hip_imu_reference.py — 배치
window 방식)가 산출한 수치(캘리브레이션 후 MAE 0.95° 등)를 재현하는지
확인한다. 수치가 허용오차 안에서 일치하면 래퍼의 history 관리·프라이밍·
캘리브레이션·denoise 설정이 학습/평가 파이프라인과 등가라는 뜻이고, 이후
시뮬 통합(Phase C)에서 성능 문제가 생겨도 래퍼 구현은 원인 후보에서
제외할 수 있다.

denoise가 확률적(flow matching 초기 노이즈 샘플)이라 bit-exact 재현은
불가능 — 지표 단위 허용오차(MAE ±0.05° 등)로 판정한다.

Usage (sbc conda env, repo 루트에서):
  python -m protomotions.maniflow.verify_angle_estimator [--denoise 3]

기본 경로는 flexion40-run01 체크포인트 / suit14-40fps zarr / 2026-08-13
기준 metrics.json. 결과는 results/angle_estimator_verify/<ts>/에 저장.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from protomotions.maniflow.angle_estimator import (
    FLEXION_ACTION_CHANNELS,
    FLEXION_OBS_CHANNELS,
    ManiFlowAngleEstimator,
)
from protomotions.maniflow.loader import (
    DEFAULT_MANIFLOW_ROOT,
    discover_best_checkpoint,
    load_maniflow_policy,
)

FLEXION40_RUN_DIR = (
    DEFAULT_MANIFLOW_ROOT
    / "data"
    / "outputs"
    / "hip_flexion_seed-maniflow_lowdim_policy_walking-locomotion-flexion40-run01_seed42"
)
SUIT14_40FPS_ZARR = DEFAULT_MANIFLOW_ROOT / "data" / "hip-flexion-etri-suit14-40fps.zarr"
# 재현 대상으로 고정한 기준 평가 (suit14-40fps, denoise 3, cal MAE 0.949°)
REFERENCE_METRICS = (
    DEFAULT_MANIFLOW_ROOT.parent
    / "eval_results"
    / "hip_imu_reference_compare"
    / "2026-08-13_16-53-24"
    / "metrics.json"
)

# 등가성 판정 허용오차 (denoise 확률성 + 배치 구성 차이 흡수)
TOL_MAE_DEG = 0.05
TOL_R2 = 0.005
TOL_CORR = 0.005
TOL_LEAD_MAE_DEG = 0.10


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """기준 스크립트의 metrics()와 동일한 정의 (rad 입력)."""
    err = pred - gt
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err**2).mean(0))
    r2 = 1 - (err**2).sum(0) / ((gt - gt.mean(0)) ** 2).sum(0)
    corr = np.array(
        [np.corrcoef(pred[:, j], gt[:, j])[0, 1] for j in range(gt.shape[1])]
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def streaming_predict(estimator, state_padded, ep_lengths, log_every=4000):
    """전 에피소드를 병렬 env로 두고 프레임 단위 스트리밍 예측.

    Args:
        estimator: 프라이밍 안 된 ManiFlowAngleEstimator (num_envs=에피소드 수).
        state_padded: (n_ep, T_max, obs_dim) — 각 에피소드 마지막 프레임으로 패딩.
        ep_lengths: (n_ep,) 실제 길이.

    Returns:
        (pred, lead): pred (n_ep, T_max, action_dim) rad(무보정),
        lead (n_ep, T_max) int — chunk 내 위치(0=관측과 동일 프레임), 예측
        없는 프레임(워밍업 구간)은 -1. 패딩 프레임은 호출부에서 ep_lengths로
        걸러야 한다.
    """
    n_ep, t_max, _ = state_padded.shape
    n_obs, n_act = estimator.n_obs_steps, estimator.n_action_steps
    pred = torch.zeros(n_ep, t_max, estimator.action_dim)
    lead = torch.full((n_ep, t_max), -1, dtype=torch.long)

    estimator.reset()
    for t in range(t_max):
        estimator.observe(state_padded[:, t])
        if t >= n_obs - 1 and (t - (n_obs - 1)) % n_act == 0:
            chunk = estimator.predict()  # (n_ep, n_act, A)
            n = min(n_act, t_max - t)
            pred[:, t : t + n] = chunk[:, :n].cpu()
            lead[:, t : t + n] = torch.arange(n)
        if log_every and (t + 1) % log_every == 0:
            print(f"  streaming {t + 1}/{t_max} frames")
    return pred, lead


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", default=str(FLEXION40_RUN_DIR))
    p.add_argument("--ckpt", default=None, help="기본: run-dir의 best topk ckpt")
    p.add_argument("--eval-zarr", default=str(SUIT14_40FPS_ZARR))
    p.add_argument(
        "--train-zarr", default=None, help="관측 캘리브레이션 기준 (기본: ckpt cfg)"
    )
    p.add_argument("--reference-metrics", default=str(REFERENCE_METRICS))
    p.add_argument("--denoise", type=int, default=3)
    p.add_argument("--fps", type=float, default=40.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="results/angle_estimator_verify")
    args = p.parse_args()

    import zarr

    torch.manual_seed(args.seed)
    deg = math.degrees

    ckpt = args.ckpt or discover_best_checkpoint(args.run_dir)
    print(f"ckpt: {ckpt}")
    policy, cfg, info = load_maniflow_policy(ckpt, device=args.device)

    # --- 평가 zarr 로드 + 채널 계약 검증 ------------------------------------
    z = zarr.open(args.eval_zarr, mode="r")
    assert z.attrs.get("features") == "flexion_trunk", (
        f"flexion_trunk zarr가 아님: {args.eval_zarr}"
    )
    obs_channels = tuple(json.loads(z.attrs["obs_spec"])["channels"])
    assert obs_channels == FLEXION_OBS_CHANNELS, obs_channels
    assert tuple(z.attrs["action_dof_names"]) == FLEXION_ACTION_CHANNELS
    state = z["data/state"][:].astype(np.float32)  # (T, 4) rad
    ref = z["data/action"][:].astype(np.float32)  # (T, 2) rad
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    names = list(z.attrs.get("motion_names", [f"ep{i}" for i in range(len(ends))]))
    n_ep = len(ends)
    print(f"eval zarr: {args.eval_zarr} — {n_ep} eps, {state.shape[0]} frames")

    # --- 관측 캘리브레이션 (기준 스크립트와 동일: 학습 평균 − 평가 평균) ----
    train_zarr = args.train_zarr
    if train_zarr is None:
        train_zarr = cfg.walking_task.dataset.zarr_path
        if not Path(train_zarr).is_absolute():
            train_zarr = str(DEFAULT_MANIFLOW_ROOT / train_zarr)
    zt = zarr.open(train_zarr, mode="r")
    train_mean = zt["data/state"][:].mean(0)
    obs_offset = (train_mean - state.mean(0)).astype(np.float32)
    print(
        "obs offset (deg): "
        + ", ".join(
            f"{c}={deg(o):+.2f}" for c, o in zip(FLEXION_OBS_CHANNELS, obs_offset)
        )
    )

    # --- estimator 구성 + 스트리밍 예측 --------------------------------------
    estimator = ManiFlowAngleEstimator(
        policy,
        num_envs=n_ep,
        device=torch.device(args.device),
        obs_offset=obs_offset,
        denoise_steps=args.denoise,
    )
    print(
        f"estimator: n_obs={estimator.n_obs_steps} n_act={estimator.n_action_steps} "
        f"action_dim={estimator.action_dim} denoise={estimator.num_inference_steps} "
        f"(epoch {info['epoch']}, {info['state_key']})"
    )

    ep_lengths = (ends - starts).astype(np.int64)
    t_max = int(ep_lengths.max())
    state_padded = np.zeros((n_ep, t_max, state.shape[1]), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        state_padded[i, : e - s] = state[s:e]
        state_padded[i, e - s :] = state[e - 1]  # 마지막 프레임 반복 패딩
    state_padded_t = torch.from_numpy(state_padded).to(args.device)

    pred, lead = streaming_predict(estimator, state_padded_t, ep_lengths)
    pred, lead = pred.numpy(), lead.numpy()

    # --- 유효 프레임 수집 (기준과 동일: lead≥0, 에피소드 길이 내) ------------
    ep_pred, ep_ref, ep_lead = [], [], []
    for i, (s, e) in enumerate(zip(starts, ends)):
        n = e - s
        m = lead[i, :n] >= 0
        ep_pred.append(pred[i, :n][m])
        ep_ref.append(ref[s:e][m])
        ep_lead.append(lead[i, :n][m])
    pred_all = np.concatenate(ep_pred)
    ref_all = np.concatenate(ep_ref)
    lead_all = np.concatenate(ep_lead)

    bias = (pred_all - ref_all).mean(0)
    print(
        "output bias (deg): "
        + ", ".join(
            f"{c}={deg(b):+.2f}" for c, b in zip(FLEXION_ACTION_CHANNELS, bias)
        )
    )

    m_raw = compute_metrics(pred_all, ref_all)
    m_cal = compute_metrics(pred_all - bias, ref_all)
    per_lead = []
    for k in range(estimator.n_action_steps):
        sel = lead_all == k
        mk = compute_metrics(pred_all[sel] - bias, ref_all[sel])
        per_lead.append(
            {
                "lead_ms": round(k * 1000 / args.fps),
                "mae_deg": float(deg(mk["mae"].mean())),
                "r2": float(mk["r2"].mean()),
                "corr": float(mk["corr"].mean()),
            }
        )
    per_episode = []
    for i in range(n_ep):
        me = compute_metrics(ep_pred[i] - bias, ep_ref[i])
        per_episode.append(
            {
                "name": names[i],
                "mae_deg": float(deg(me["mae"].mean())),
                "r2": float(me["r2"].mean()),
            }
        )

    print("\n[streaming estimator, calibrated]")
    for j, c in enumerate(FLEXION_ACTION_CHANNELS):
        print(
            f"  {c:>14}: MAE {deg(m_cal['mae'][j]):.3f}° "
            f"R² {m_cal['r2'][j]:.4f} corr {m_cal['corr'][j]:.4f}"
        )
    for row in per_lead:
        print(
            f"  lead {row['lead_ms']:3d} ms: MAE {row['mae_deg']:.3f}° "
            f"R² {row['r2']:.4f} corr {row['corr']:.4f}"
        )

    # --- 기준 수치와 대조 ------------------------------------------------------
    verdict = None
    ref_path = Path(args.reference_metrics)
    if ref_path.is_file():
        refm = json.load(open(ref_path))
        checks = []
        for j, c in enumerate(FLEXION_ACTION_CHANNELS):
            checks += [
                (
                    f"{c} MAE°",
                    deg(refm["calibrated"]["mae"][j]),
                    deg(m_cal["mae"][j]),
                    TOL_MAE_DEG,
                ),
                (f"{c} R²", refm["calibrated"]["r2"][j], m_cal["r2"][j], TOL_R2),
                (
                    f"{c} corr",
                    refm["calibrated"]["corr"][j],
                    m_cal["corr"][j],
                    TOL_CORR,
                ),
            ]
        for rrow, mrow in zip(refm.get("per_lead", []), per_lead):
            checks.append(
                (
                    f"lead {rrow['lead_ms']} ms MAE°",
                    rrow["mae_deg"],
                    mrow["mae_deg"],
                    TOL_LEAD_MAE_DEG,
                )
            )
        print(f"\n[기준 대조] reference: {ref_path}")
        print(f"{'metric':>22} | {'reference':>9} | {'streaming':>9} | {'Δ':>7} | ok")
        all_ok = True
        for name_, rv, mv, tol in checks:
            ok = abs(mv - rv) <= tol
            all_ok &= ok
            print(
                f"{name_:>22} | {rv:9.3f} | {mv:9.3f} | {mv - rv:+7.3f} | "
                f"{'✓' if ok else '✗ (tol ' + str(tol) + ')'}"
            )
        verdict = "PASS" if all_ok else "FAIL"
        print(f"\n등가성 판정: {verdict}")
    else:
        print(f"\n기준 metrics.json 없음({ref_path}) — 대조 생략, 수치만 보고")

    # --- 저장 -----------------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "ckpt": str(ckpt),
                "eval_zarr": str(args.eval_zarr),
                "train_zarr": str(train_zarr),
                "denoise_steps": estimator.num_inference_steps,
                "seed": args.seed,
                "obs_offset_rad": obs_offset.tolist(),
                "bias_rad": bias.tolist(),
                "raw": {k: v.tolist() for k, v in m_raw.items()},
                "calibrated": {k: v.tolist() for k, v in m_cal.items()},
                "per_lead": per_lead,
                "per_episode": per_episode,
                "reference_metrics": str(ref_path) if ref_path.is_file() else None,
                "verdict": verdict,
            },
            f,
            indent=2,
        )
    print(f"saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
