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
"""Replay recorded walking observations to inspect braking decisions on CPU.

Input substitutions are one-step sensitivity probes on fixed recorded states,
not closed-loop assistance evaluations and not deployable performance results.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from protomotions.assist.llp_policy import LLPPolicy
from protomotions.envs.obs.assist_obs import compute_assist_proprio_obs, compute_assist_target_obs


def probe(run_dir):
    path = Path(run_dir)
    meta = json.loads((path/"meta.json").read_text())
    with np.load(path/"traces.npz") as source:
        tr = {k: source[k] for k in source.files}
    policy = LLPPolicy(meta["llp_checkpoint"], torch.device("cpu"))
    lay = policy.layout
    assert lay.dt == .01 and lay.obs_future_samples == 3 and lay.obs_future_dt == .01
    # Exclude calibration reset and the subsequent entire history refill.
    idx = np.flatnonzero((tr["t"] >= max(meta["warmup"], .5)+.30)
                        & (np.arange(len(tr["t"])) < len(tr["t"])-2))
    sign = lay.flexion_sign  # Skeleton joint_sign is +1 in run_system.
    q = torch.from_numpy(tr["hip_pos"][:, 1]).float()*sign
    v = torch.from_numpy(tr["hip_vel"][:, 1]).float()*sign
    past = np.arange(1, lay.history_steps+1)
    # Controller observe() pushes action i-1 into slot 0; act() uses slots 1..P.
    action_past = np.arange(2, lay.prev_action_steps+2)
    hist_q, hist_v = q[idx[:, None]-past], v[idx[:, None]-past]
    pred = torch.from_numpy(tr["theta_d_future"][idx, 1]).float()*sign
    age = torch.from_numpy(tr["chunk_age"][idx, 1, None]).float()
    obs = {
        "assist_proprio": compute_assist_proprio_obs(q[idx], v[idx], hist_q, hist_v, lay.dof_vel_scale),
        "assist_target": compute_assist_target_obs(pred, q[idx], age),
        "previous_actions": torch.from_numpy(tr["raw_action"][idx[:, None]-action_past, 1]).reshape(len(idx), -1)}
    original = policy.act(obs)
    truth = torch.from_numpy(tr["raw_action"][idx, 1])
    torque = lambda raw: (policy.to_torque(raw)*sign*meta["assist_scale"]).numpy()
    applied_error = torque(original)-tr["tau_assist"][idx, 1]
    error = dict(raw_action_max=float((original-truth).abs().max()),
                 raw_action_rms=float((original-truth).square().mean().sqrt()),
                 applied_torque_max_Nm=float(np.abs(applied_error).max()),
                 applied_torque_rms_Nm=float(np.sqrt(np.mean(applied_error**2))))
    # CPU vs recorded CUDA GEMM is not bit-exact; an index/sign mistake is much larger.
    if error["applied_torque_max_Nm"] > .2:
        raise RuntimeError(f"Replay mismatch too large for sensitivity interpretation: {error}")
    variants = {"recorded": tr["tau_assist"][idx, 1], "replayed": torque(original)}
    actual = torch.stack([q[idx+k] for k in range(3)], dim=1)
    linear = pred[:, :1]+v[idx, None]*torch.arange(3)[None, :, None]*lay.obs_future_dt
    for name, future in (("actual_future_oracle", actual), ("constant_velocity_future", linear)):
        modified = {**obs, "assist_target": compute_assist_target_obs(future, q[idx], age)}
        variants[name] = torque(policy.act(modified))
    # Deliberately synthetic history: removes acceleration but keeps present q, v.
    history_dt = torch.arange(1, lay.history_steps+1)[None, :, None]*lay.dt
    constant_history = q[idx, None]-v[idx, None]*history_dt
    modified = {**obs, "assist_proprio": compute_assist_proprio_obs(
        q[idx], v[idx], constant_history, v[idx, None].expand_as(hist_v), lay.dof_vel_scale)}
    variants["constant_velocity_history"] = torque(policy.act(modified))
    phase = tr["gait_phase"][idx]
    q_joint, v_joint = tr["hip_pos"][:, 1], tr["hip_vel"][:, 1]
    rows = []
    for lo, hi in ((20, 30), (30, 40), (60, 70), (70, 80), (80, 90), (90, 100)):
        sel = (phase >= lo/100) & (phase < hi/100)
        mean = lambda x: float(x[sel].mean())
        row = dict(phase_pct=[lo, hi], samples=int(sel.sum()))
        row["past_deg"] = {str(-k*10): mean(np.degrees(q_joint[idx-k])) for k in (25, 10, 2)}
        row["current_deg"] = mean(np.degrees(q_joint[idx]))
        row["predicted_deg"] = {str(k*10): mean(np.degrees(tr["theta_d_future"][idx, 1, k])) for k in range(3)}
        row["actual_20ms_deg"] = mean(np.degrees(q_joint[idx+2]))
        row["prediction_20ms_mae_deg"] = mean(np.degrees(np.abs(tr["theta_d_future"][idx, 1, 2]-q_joint[idx+2])))
        row["velocity_rad_s"] = mean(v_joint[idx])
        row["past_100ms_acceleration_rad_s2"] = mean((v_joint[idx]-v_joint[idx-10])/.1)
        row["forecast_forward_fraction_pct"] = mean(100.*(tr["theta_d_future"][idx, 1, 2]>tr["theta_d_future"][idx, 1, 0]))
        row["braking_fraction_pct"] = mean(100.*(tr["tau_assist"][idx, 1]*v_joint[idx]<0))
        row["torque_mean_Nm"] = {name: mean(u) for name, u in variants.items()}
        row["torque_change_rms_Nm"] = {name: float(np.sqrt(np.mean((u[sel]-variants["replayed"][sel])**2)))
                                       for name, u in variants.items()}
        rows.append(row)
    return dict(run_dir=str(path), checkpoint=meta["llp_checkpoint"], replay_error=error,
                warmup_excluded_until_s=float(tr["t"][idx[0]]), phase_rows=rows,
                caveat="One-step input sensitivity on fixed traces. No causal burden or closed-loop improvement claim.")


def render(results, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    for col, (name, result) in enumerate(results.items()):
        for row_id, lo in enumerate((70, 80)):
            ax = axes[row_id, col]
            r = next(r for r in result["phase_rows"] if r["phase_pct"][0] == lo)
            past_x = [-250, -100, -20, 0]
            past_y = [r["past_deg"][str(i)] for i in past_x[:-1]]+[r["current_deg"]]
            ax.plot(past_x, past_y, "o-", color="#555b65", label="Recorded history")
            ax.plot([0, 10, 20], [r["predicted_deg"][str(i)] for i in (0, 10, 20)],
                    "o-", color="#306cb3", label="HLP prediction")
            ax.scatter([20], [r["actual_20ms_deg"]], marker="x", color="#d35e30", s=80, label="Actual +20 ms")
            ax.axvline(0, color="#888", lw=.8)
            ax.set_title(f"{name}: {lo}-{lo+10}% phase")
            ax.set_ylabel("Hip flexion [deg]")
            ax.set_xlabel("Time relative to decision [ms]")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def build_section(results, svg):
    parts = ['<section><h2>70–90%에서 정책이 실제로 본 과거와 미래</h2>',
             '<p>아래는 토크를 결정한 시점의 phase로 묶은 관측이며, 앞의 토크 RMS 표는 물리 창의 중점 phase다. '
             '두 표의 구간 경계는 반 step 차이가 있다. 착용자 B의 양쪽 hip을 각자의 phase로 모았고 '
             '영점 조정 이후 히스토리가 채워지는 1.30초까지 제외했다.</p>',
             '<p><b>미래가 뒤로 간다고 잘못 예측해서 제동하는 것이 아니다.</b> HLP는 평균적으로 앞으로 계속 가되 '
             '조금만 더 간다고 예측한다. 과거에는 다리가 빠르게 전진하다가 감속하는 흐름이 보인다. '
             '진자에서 학습한 감속 토크와 실제 전신에서 착용자가 부담하는 토크는 같지 않다.</p>', svg]
    for name, result in results.items():
        parts.append(f'<h3>{name}</h3><div class="scroll"><table><tr><th>phase</th><th>−250 ms</th>'
                     '<th>−100 ms</th><th>현재</th><th>예측 0 ms</th><th>예측 +10 ms</th><th>예측 +20 ms</th>'
                     '<th>실제 +20 ms</th><th>+20 ms MAE</th></tr>')
        for r in result['phase_rows']:
            if r['phase_pct'][0] not in (70, 80):
                continue
            vals = [r['past_deg']['-250'], r['past_deg']['-100'], r['current_deg'],
                    *(r['predicted_deg'][str(k)] for k in (0, 10, 20)),
                    r['actual_20ms_deg'], r['prediction_20ms_mae_deg']]
            parts.append(f'<tr><td>{r["phase_pct"][0]}–{r["phase_pct"][1]}%</td>'
                         + ''.join(f'<td>{x:.2f}°</td>' for x in vals)+'</tr>')
        parts.append('</table></div><p>기록 관측으로 정책 출력을 CPU에서 재현했을 때, 실제 적용 토크와의 최대 차이 '
                     f'{result["replay_error"]["applied_torque_max_Nm"]:.3f} N·m.</p>')
        parts.append('<div class="scroll"><table><tr><th>phase</th><th>원래 평균 보조</th>'
                     '<th>실제 미래각으로 교체</th><th>등속 미래로 교체</th><th>등속 과거로 교체</th></tr>')
        for r in result['phase_rows']:
            if r['phase_pct'][0] not in (70, 80):
                continue
            vals = [r['torque_mean_Nm'][k] for k in (
                'recorded', 'actual_future_oracle', 'constant_velocity_future', 'constant_velocity_history')]
            parts.append(f'<tr><td>{r["phase_pct"][0]}–{r["phase_pct"][1]}%</td>'
                         + ''.join(f'<td>{x:+.2f} N·m</td>' for x in vals)+'</tr>')
        parts.append('</table></div>')
    parts.append('<p><b>예측을 완벽한 실제 미래각으로 바꿔도 제동은 남는다.</b> 이는 HLP 오차만 줄여서 '
                 '70–90% 문제가 해결될 것이라는 설명에 반하는 결과다. 한편 과거를 등속으로 바꾸면 L9의 제동은 '
                 '오히려 더 커진다. 따라서 감속 이력 하나가 제동의 단독 원인이라고도 단정할 수 없다. '
                 '현재 각도·속도·과거 상태·이전 action·짧은 미래창의 결합에 학습된 출력이다.</p>'
                 '<p>이 교체 실험은 고정된 기록 관측에서 한 번의 정책 출력을 비교한 민감도 검사다. '
                 '미래 실제값은 진단용 특권 정보이며, 인공 과거는 학습 분포 밖일 수 있다. '
                 '폐루프 부담 감소나 배포 가능한 개선으로 해석하지 않는다.</p></section>')
    return ''.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/cli_walk_diagnostic_20261009/walking_inputs")
    args = p.parse_args()
    torch.set_num_threads(4)
    records = json.loads(Path("results/llp_L11_opposition_sequence/comparison.json").read_text())
    results = {name: probe(records[key]["run_dir"]) for name, key in (
        ("L1 x0.5", "reference_L1_x0.5_m0"), ("L9 @2000", "eval_llp_L9_opposition_e2000_m0"))}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    svg = render(results, out.with_suffix(".png"))
    out.with_suffix(".svg").write_text(svg)
    out.with_suffix(".html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>걷기 정책의 과거·미래 입력</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:16px/1.7 system-ui;padding:0 20px}'
        'svg{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}'
        'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}</style>'
        + build_section(results, svg) + '</html>')
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
