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
"""`ManiFlowChunkSource` 등가성 검증 — AI-SBC Phase C 게이트.

시뮬레이션 쪽 chunk 소스가 오프라인 예측과 같은 값을 내는지 확인한다.
`verify_angle_estimator.py`가 estimator 래퍼를 검증하는 것과 같은 논리를
**chunk 소스(링 버퍼 + 모델 프레임 격자 재표집 + 부호 변환)** 층에 적용한다.

두 가지 모드를 돌린다:

- **모드 A (dt = 모델 프레임 간격)**: suit14 프레임을 그대로 흘린다. 재표집이
  프레임 정확 복원이므로, 오프라인 기준(`metrics.json`)에 있는 lead의 MAE가
  기준과 일치해야 한다 → **PASS/FAIL 판정 대상**.
- **모드 B (dt = 10 ms)**: 같은 데이터를 100 Hz로 선형 업샘플해 흘린다
  (제어 주기). 판정은 **한쪽 방향** — B가 A보다 나쁘지만 않으면 된다.
  모델 프레임이 10 ms의 정수배가 아니면(옛 40 fps 판의 25 ms) 일부 관측
  프레임이 두 샘플 평균으로 근사되는데, 이 평활은 실제 env에도 그대로 있다.
  H1처럼 모델 프레임이 10 ms이면 B는 A와 같으므로 건너뛴다.

오프라인 기준에 없는 lead는 판정하지 않고 **측정값을 보고**하며, 오차가 lead에
따라 단조 증가하는지만 확인한다.

기본값은 H1 저장소 사본 / suit14-100fps zarr / 같은 H1의 2026-11-05 오프라인 평가
(`verify_angle_estimator.REFERENCE_METRICS`, denoise 1).

실행:
    python -m protomotions.maniflow.verify_chunk_source            # sbc env
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
    FLEXION_FRAME_DT,
    FLEXION_OBS_CHANNELS,
    ManiFlowAngleEstimator,
    _frame_dt_from_cfg,
)
from protomotions.maniflow.chunk_source import ManiFlowChunkSource
from protomotions.maniflow.loader import (
    DEFAULT_H1_CHECKPOINT,
    DEFAULT_MANIFLOW_ROOT,
    load_maniflow_policy,
)
from protomotions.maniflow.verify_angle_estimator import (
    REFERENCE_DENOISE,
    REFERENCE_METRICS,
    SUIT14_100FPS_ZARR,
)

CONTROL_DT = 0.01  # 모드 B: 보조 제어 주기 [s]

# 판정 허용오차
TOL_LEAD_MAE_DEG = 0.15  # 오프라인 기준 대비 (모드 A)
TOL_MODE_DIFF_DEG = 0.15  # 모드 B가 A보다 **나빠진** 양의 상한 (한쪽 방향)


def _load_suit14(path: str):
    """suit14 zarr → 에피소드 병렬 배치 (마지막 프레임 반복 패딩)."""
    import zarr

    z = zarr.open(path, mode="r")
    assert z.attrs.get("features") == "flexion_trunk", path
    assert tuple(json.loads(z.attrs["obs_spec"])["channels"]) == FLEXION_OBS_CHANNELS
    assert tuple(z.attrs["action_dof_names"]) == FLEXION_ACTION_CHANNELS
    state = z["data/state"][:].astype(np.float32)
    ends = np.asarray(z["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    lengths = ends - starts
    t_max = int(lengths.max())
    padded = np.zeros((len(ends), t_max, state.shape[1]), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        padded[i, : e - s] = state[s:e]
        padded[i, e - s :] = state[e - 1]
    return padded, lengths, state


def _gt_at(padded: torch.Tensor, lengths: torch.Tensor, frames: torch.Tensor):
    """프레임 좌표(실수)에서 GT 4ch 선형보간. frames (n, S) → (n, S, 4)."""
    max_idx = (lengths - 1).unsqueeze(1).float()
    f = frames.clamp(min=0.0)
    f = torch.minimum(f, max_idx)
    i0 = f.floor().long()
    i1 = torch.minimum(i0 + 1, (lengths - 1).unsqueeze(1))
    w = (f - i0.float()).unsqueeze(-1)
    g0 = torch.gather(padded, 1, i0.unsqueeze(-1).expand(-1, -1, padded.shape[-1]))
    g1 = torch.gather(padded, 1, i1.unsqueeze(-1).expand(-1, -1, padded.shape[-1]))
    return g0 + (g1 - g0) * w


def run_mode(
    label: str,
    src: ManiFlowChunkSource,
    padded: torch.Tensor,
    lengths: torch.Tensor,
    *,
    dt: float,
    refresh_steps: int,
    flexion_sign: float,
    log_every: int = 4000,
):
    """한 모드를 스트리밍해 lead별 예측/GT 쌍을 모은다.

    Returns:
        (pred, gt) — 각각 (samples, num_knots, 2) rad, 데이터 규약.
    """
    n_ep = padded.shape[0]
    device = padded.device
    all_ids = torch.arange(n_ep, device=device)
    # 스트리밍 총 스텝: 가장 긴 에피소드를 dt 격자로 덮는다
    # 평가 zarr 프레임 간격 = 모델 프레임 간격 (main에서 확인)
    frame_dt = src.knot_dt
    total_steps = int(math.ceil((float(lengths.max()) - 1) * frame_dt / dt)) + 1

    # 워밍업: 모델 관측 창이 실제 데이터로 채워질 때까지 예측 무시
    warmup = int(math.ceil((src.n_obs - 1) * src.knot_dt / dt))
    knot_frames = torch.arange(src.num_knots, device=device, dtype=torch.float32)

    src.reset(all_ids, padded[:, 0, :2] * flexion_sign, padded[:, 0, 2:])
    preds, gts = [], []
    for step in range(total_steps):
        t = step * dt
        frame = torch.full((n_ep, 1), t / frame_dt, device=device)
        obs = _gt_at(padded, lengths, frame)[:, 0]  # (n, 4)
        src.observe(obs[:, :2] * flexion_sign, obs[:, 2:])

        if step >= warmup and (step - warmup) % refresh_steps == 0:
            t_now = torch.full((n_ep,), t, device=device)
            knots, _ = src.predict_knots(all_ids, t_now)  # (n, K, 2) 로봇 규약
            pred = knots * flexion_sign  # → 데이터 규약
            gt_frames = frame + knot_frames.unsqueeze(0)  # (n, K)
            gt = _gt_at(padded, lengths, gt_frames)[..., :2]
            # 패딩 구간(에피소드 끝 이후)은 제외
            valid = gt_frames[:, -1] <= (lengths - 1).float()
            if bool(valid.any()):
                preds.append(pred[valid].cpu())
                gts.append(gt[valid].cpu())
        if log_every and (step + 1) % log_every == 0:
            print(f"  [{label}] {step + 1}/{total_steps} steps")

    return torch.cat(preds).numpy(), torch.cat(gts).numpy()


def per_lead_mae_deg(pred: np.ndarray, gt: np.ndarray, bias_leads: int = 0):
    """lead별 MAE [deg] — 채널 평균, 출력 바이어스를 제거한 뒤 계산.

    Args:
        bias_leads: 바이어스를 추정할 lead 개수. 오프라인 기준 스크립트는
            `action`(4프레임)만 쓰므로 **4를 줘야 같은 정의**가 된다. 0이면
            전체 lead로 추정 (더 긴 lead의 큰 잔차가 섞여 lead 0의 작은
            MAE를 왜곡한다).
    """
    k_bias = pred.shape[1] if bias_leads <= 0 else min(bias_leads, pred.shape[1])
    bias = (pred[:, :k_bias] - gt[:, :k_bias]).reshape(-1, pred.shape[-1]).mean(0, dtype=np.float64)
    err = np.abs((pred - bias) - gt)  # (S, K, 2)
    return [float(math.degrees(err[:, k].mean(dtype=np.float64))) for k in range(pred.shape[1])], bias


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default=str(DEFAULT_H1_CHECKPOINT))
    p.add_argument("--eval-zarr", default=str(SUIT14_100FPS_ZARR))
    p.add_argument("--train-zarr", default=None)
    p.add_argument("--reference-metrics", default=str(REFERENCE_METRICS))
    p.add_argument("--denoise", type=int, default=REFERENCE_DENOISE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flexion-sign", type=float, default=-1.0)
    p.add_argument("--modes", default="AB", help="A(모델 프레임 간격) / B(10 ms) 조합")
    p.add_argument(
        "--refresh-frames",
        type=int,
        default=None,
        help="예측 주기 [모델 프레임] — 기본은 오프라인 기준과 같은 n_action_steps",
    )
    p.add_argument("--out-root", default="results/chunk_source_verify")
    args = p.parse_args()

    import zarr

    torch.manual_seed(args.seed)
    deg = math.degrees
    device = torch.device(args.device)

    ckpt = args.ckpt
    print(f"ckpt: {ckpt}")
    policy, cfg, info = load_maniflow_policy(ckpt, device=args.device)
    frame_dt = _frame_dt_from_cfg(cfg) or FLEXION_FRAME_DT
    eval_dt = float(zarr.open(args.eval_zarr, mode="r").attrs.get("frame_dt", frame_dt))
    assert abs(eval_dt - frame_dt) < 1e-9, f"평가 zarr 프레임 간격 {eval_dt} s ≠ 모델 {frame_dt} s"
    refresh_frames = args.refresh_frames or int(policy.n_action_steps)

    padded_np, lengths_np, state = _load_suit14(args.eval_zarr)
    n_ep = padded_np.shape[0]
    print(
        f"eval zarr: {args.eval_zarr} — {n_ep} eps, "
        f"{state.shape[0]} frames, t_max {padded_np.shape[1]}"
    )

    # 관측 캘리브레이션 (기준 스크립트와 동일: 학습 평균 − 평가 평균)
    train_zarr = args.train_zarr
    if train_zarr is None:
        train_zarr = cfg.walking_task.dataset.zarr_path
        if not Path(train_zarr).is_absolute():
            train_zarr = str(DEFAULT_MANIFLOW_ROOT / train_zarr)
    obs_offset = (
        zarr.open(train_zarr, mode="r")["data/state"][:].mean(0, dtype=np.float64)  # float32 누적은 1~2° 어긋남
        - state.mean(0, dtype=np.float64)
    ).astype(np.float32)
    print(
        "obs offset (deg): "
        + ", ".join(
            f"{c}={deg(o):+.2f}" for c, o in zip(FLEXION_OBS_CHANNELS, obs_offset)
        )
    )

    padded = torch.from_numpy(padded_np).to(device)
    lengths = torch.from_numpy(lengths_np).to(device)

    results = {}
    for mode in args.modes.upper():
        dt = frame_dt if mode == "A" else CONTROL_DT
        if mode == "B" and "A" in args.modes.upper() and abs(dt - frame_dt) < 1e-9:
            print(f"\n=== 모드 B: 모델 프레임이 {frame_dt * 1e3:.0f} ms라 모드 A와 같음 — 생략 ===")
            continue
        # 예측 주기를 모델 프레임 기준으로 맞춘다
        refresh_steps = max(1, int(round(refresh_frames * frame_dt / dt)))
        estimator = ManiFlowAngleEstimator(
            policy,
            num_envs=n_ep,
            device=device,
            obs_offset=obs_offset,
            denoise_steps=args.denoise,
            frame_dt=frame_dt,
        )
        src = ManiFlowChunkSource(
            estimator,
            num_envs=n_ep,
            dt=dt,
            device=device,
            flexion_sign=args.flexion_sign,
        )
        print(
            f"\n=== 모드 {mode}: dt={dt * 1e3:.0f} ms, 예측 주기 "
            f"{refresh_steps} step ({refresh_steps * dt * 1e3:.0f} ms) ==="
        )
        pred, gt = run_mode(
            mode,
            src,
            padded,
            lengths,
            dt=dt,
            refresh_steps=refresh_steps,
            flexion_sign=args.flexion_sign,
        )
        # 오프라인 기준과 같은 정의로 바이어스 추정 (action 4프레임)
        mae, bias = per_lead_mae_deg(pred, gt, bias_leads=estimator.n_action_steps)
        results[mode] = {
            "dt": dt,
            "samples": int(pred.shape[0]),
            "mae_deg": mae,
            "bias_deg": [float(deg(b)) for b in bias],
        }
        print(f"  samples={pred.shape[0]}  bias(deg)={[round(deg(b), 2) for b in bias]}")
        for k, v in enumerate(mae):
            print(f"  lead {k * src.knot_dt * 1e3:>5.0f} ms : MAE {v:.3f}°")

    # ── 판정 ────────────────────────────────────────────────────────────
    verdicts = []
    ref = json.loads(Path(args.reference_metrics).read_text())
    ref_lead = {int(r["lead_ms"]): float(r["mae_deg"]) for r in ref["per_lead"]}
    print("\n===== 판정 =====")

    if "A" in results:
        mae_a = results["A"]["mae_deg"]
        for k, v in enumerate(mae_a):
            lead_ms = int(round(k * frame_dt * 1e3))
            if lead_ms in ref_lead:
                d = abs(v - ref_lead[lead_ms])
                ok = d <= TOL_LEAD_MAE_DEG
                verdicts.append(ok)
                print(
                    f"  [{'PASS' if ok else 'FAIL'}] lead {lead_ms:>3d} ms: "
                    f"{v:.3f}° vs 기준 {ref_lead[lead_ms]:.3f}° (Δ{d:.3f}°, "
                    f"허용 {TOL_LEAD_MAE_DEG}°)"
                )
            else:
                print(f"  [보고] lead {lead_ms:>3d} ms: {v:.3f}° (오프라인 기준 없음)")
        mono = all(mae_a[i] <= mae_a[i + 1] + 1e-6 for i in range(len(mae_a) - 1))
        verdicts.append(mono)
        print(f"  [{'PASS' if mono else 'FAIL'}] lead에 따른 MAE 단조 증가")

    if "A" in results and "B" in results:
        # 판정은 **한쪽 방향**이다: 모드 B(실제 env 경로)가 모드 A보다 나쁘지만
        # 않으면 된다. 100 Hz 샘플로 25 ms 격자를 복원할 때, 지연이 policy
        # 스텝의 정수배인 프레임(0/50/100/... ms 전)은 오차 0이고 절반에 걸리는
        # 프레임(25/75/... ms 전)만 두 샘플의 평균으로 근사된다(실측 평균
        # 0.16°, 최대 1.34°). 이 평활은 실제 진자 env에도 똑같이 존재하는
        # 경로이고, suit14에서는 오히려 예측 오차를 **줄이는** 방향으로 나온다.
        worst_worse = max(
            b - a for a, b in zip(results["A"]["mae_deg"], results["B"]["mae_deg"])
        )
        best_gain = max(
            a - b for a, b in zip(results["A"]["mae_deg"], results["B"]["mae_deg"])
        )
        ok = worst_worse <= TOL_MODE_DIFF_DEG
        verdicts.append(ok)
        print(
            f"  [{'PASS' if ok else 'FAIL'}] 모드 B가 A보다 나빠진 최대치 "
            f"{worst_worse:+.3f}° (허용 {TOL_MODE_DIFF_DEG}°); "
            f"B가 좋아진 최대치 {best_gain:+.3f}° — 재표집 평활 효과, 정상"
        )

    verdict = "PASS" if all(verdicts) else "FAIL"
    print(f"\n종합: {verdict}")

    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "epoch": info["epoch"],
                "eval_zarr": args.eval_zarr,
                "denoise": args.denoise,
                "flexion_sign": args.flexion_sign,
                "refresh_frames": refresh_frames,
                "obs_offset_rad": [float(v) for v in obs_offset],
                "modes": results,
                "bias_leads": "n_action_steps (오프라인 기준과 동일 정의)",
                "reference_per_lead_deg": ref_lead,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"saved: {out_dir / 'metrics.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
