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
"""Walking-first diagnosis of preserved W1 traces; no new performance claims."""
import argparse
import html
import io
import json
from pathlib import Path

import numpy as np

from report_torque_html import aligned_torques, label


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def diagnose(tr, meta, bins=10):
    data = aligned_torques(tr, meta)
    idx = np.flatnonzero((tr["t"] >= meta["warmup"]) & (np.arange(len(tr["t"])) >= 1))
    q = (tr["hip_pos"][idx] + tr["hip_pos"][idx - 1]) / 2
    w = (tr["hip_vel"][idx] + tr["hip_vel"][idx - 1]) / 2
    rows = []
    for i in range(bins):
        sel = (data["phase"] >= i / bins) & (data["phase"] < (i + 1) / bins)
        a, b, u = (data[k][sel].astype(np.float64) for k in ("agent_A", "agent_B", "assist_B"))
        residual = b + u - a
        fixed = u*u - 2*a*u
        response = residual*residual + 2*(a-u)*residual
        measured = b*b - a*a
        # Descriptive identity, not a separately integrated counterfactual.
        np.testing.assert_allclose(fixed + response, measured, atol=1e-9)
        row = dict(phase_start_pct=100*i/bins, samples=int(sel.sum()),
                   agent_A_rms_Nm=rms(a), agent_B_rms_Nm=rms(b), assist_rms_Nm=rms(u),
                   rms_increase_Nm=rms(b)-rms(a), agent_A_mean_Nm=float(a.mean()),
                   agent_B_mean_Nm=float(b.mean()), assist_mean_Nm=float(u.mean()),
                   angle_A_deg=float(np.degrees(q[:, 0][sel]).mean()),
                   omega_A_rad_s=float(w[:, 0][sel].mean()),
                   opposite_to_A_pct=float(100*np.mean(a*u < 0)),
                   same_sign_over_twice_A_pct=float(100*np.mean((a*u >= 0) & (np.abs(u)>2*np.abs(a)))),
                   fixed_demand_delta_Nm2=float(fixed.mean()),
                   response_delta_Nm2=float(response.mean()), actual_delta_Nm2=float(measured.mean()),
                   total_torque_residual_rms_Nm=rms(residual),
                   angle_AB_rms_deg=rms(np.degrees((q[:, 1]-q[:, 0])[sel])))
        row["side_rms_increase_Nm"] = [
            rms(data["agent_B"][sel[:, side], side])-rms(data["agent_A"][sel[:, side], side])
            for side in range(2)]
        rows.append(row)
    return rows


def make_figure(results, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for (key, result), color in zip(results.items(), ["#555b65", "#306cb3", "#a14d0f", "#168071"]):
        rows = result["phase"]
        x = np.array([r["phase_start_pct"]+5 for r in rows])
        name = result["plot_name"]
        ax[0].plot(x, [r["rms_increase_Nm"] for r in rows], "o-", color=color, label=name)
        ax[1].plot(x, [r["assist_mean_Nm"] for r in rows], "o-", color=color, label=name)
    rows = next(iter(results.values()))["phase"]
    ax[1].plot(x, [r["agent_A_mean_Nm"] for r in rows], "k--", label="Unassisted wearer A")
    ax[2].plot(x, [r["omega_A_rad_s"] for r in rows], "o-", color="#6949aa", label="Unassisted hip velocity")
    for axes in ax:
        axes.axhline(0, color="#999999", lw=.8)
        axes.axvspan(20, 40, color="#c77b12", alpha=.09)
        axes.axvspan(70, 90, color="#bb4455", alpha=.09)
        axes.set_xlim(0, 100)
        axes.set_xticks(np.arange(0, 101, 10))
        axes.grid(alpha=.16)
        axes.legend(fontsize=8, ncol=2)
    ax[0].set_ylabel("Wearer RMS: B - A [Nm]\nPositive = worse")
    ax[1].set_ylabel("Signed torque [Nm]\nPositive = flexion")
    ax[2].set_ylabel("Hip velocity [rad/s]\nPositive = flexion")
    ax[2].set_xlabel("Reference hip phase [%], 0 = maximum reference flexion")
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return buf.getvalue()[buf.getvalue().index("<svg"):]


INTRO = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>걷기 문제 구간 진단</title><style>
body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#242b35;background:#faf9f6}
svg{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;background:white}
td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}p{overflow-wrap:anywhere}
</style><h1>걷기: 어디서, 왜 방해하는가</h1>
<p>2026-10-09 원본 W1 40초 기록 재분석. 새 정책 성능 결과가 아니다. 토크는 같은 10 ms 창에 정렬했다.
그래프·표 모두 내장되어 이 HTML 하나로 확인할 수 있다.</p>
<p><b>20–40%:</b> 다리는 뒤로 움직이는 중이지만 착용자 토크는 굴곡 쪽으로 바뀐다.
L6·L8은 신전 보조를 늦게까지 내서 착용자와 맞선다. L1은 더 일찍 부호를 바꾼다.
움직임 방향과 필요한 토크 방향은 같지 않다.</p>
<p><b>70–90%:</b> 앞으로 나온 다리가 감속하는 구간이며 무보조 착용자 RMS는 약 17–18 N·m.
L1의 보조 16–20 N·m도 여기서는 작은 힘이 아니다. 80–90% RMS가 16.64→25.47 N·m로 증가한다.</p>
<p><b>큰 이득이 손해를 가린다:</b> 0–10%의 무보조 RMS는 약 76 N·m여서 이 구간 개선이 전체 τ² 평균을 지배한다.
L6가 평균 −25%여도 20–30% RMS는 22.41→38.90 N·m다.</p>
<p><b>확정과 가설:</b> 위 부호·크기·RMS는 측정 사실이다. 진자에서 학습한 제동과 절대 추종 보상이
전신 착용자에 잘못 전이됐다는 설명은 가설이다. 발 접촉 기록이 없어 phase를 실제 입각·유각으로 단정하지 않는다.
0%는 참조 hip 최대 굴곡이며 착지 검출값이 아니다.</p>
<p><b>다음 판단:</b> L11의 추종 보상을 무보조 진자 대비 비악화 보상으로 바꾼 L12 코드는 준비했지만
실제 물리 검증·학습은 미실행이다. 사용자 후속 가설에 따라 짧은 관측창과 단순 진자 플랜트의 한계를 먼저 분리한다.
보상 수정만으로 해결된다고 가정하지 않으며 L9도 걷기 후보로 보존한다.</p>"""


def phase_information_section(path):
    results = json.loads(Path(path).read_text())
    parts = ['<section><h2>짧은 창 때문에 보행 구간을 구별할 수 없는가</h2>',
             '<p>이 걷기의 한 주기는 중앙값 1.12초다. 과거 250 ms는 주기의 약 22%, 미래 20 ms는 약 1.8%다. '
             '구간 정보가 현재 관측에 있는지만 확인하기 위해, 앞부분 2–20초의 측정 hip 각도·각속도로 '
             '뒷부분 21–40초의 참조 위상을 추정했다. 시각·phase·참조 각도·토크는 입력에서 제외했다. '
             '학습/평가 사이 1초를 비웠으며 같은 모션 반복에서의 최근접 이웃 진단이다.</p>',
             '<div class="scroll"><table><tr><th>정책 기록</th><th>관측</th><th>평균 위상 오차 (% 주기)</th>'
             '<th>70–90% precision</th><th>70–90% recall</th></tr>']
    for name, result in results.items():
        for feature, row in result['results'].items():
            parts.append(f'<tr><td>{name}</td><td>{feature}</td><td>{row["phase_mae_percent_cycle"]:.2f}</td>'
                         f'<td>{100*row["precision_70_90"]:.1f}%</td><td>{100*row["recall_70_90"]:.1f}%</td></tr>')
    parts.append('</table></div><p><b>해석 범위:</b> 이 기록에서는 짧은 창에도 구간을 구별하는 정보가 상당히 있다. '
                 '1초 이력 확장만으로 뚜렷한 개선은 없었다. 이것은 LLP가 그 정보를 올바르게 쓰고 있다는 증거도, '
                 '다른 보행·착용자에 일반화된다는 증거도 아니다. 긴 창의 결과는 이 진단 추정기의 특성에도 의존한다.</p>'
                 '<p><b>더 근본적인 차이:</b> 같은 hip 감속이어도 진자와 전신 보행은 착용자가 직접 부담할 토크가 다를 수 있다. '
                 '진자는 무릎·골반·지면과의 상호작용을 실제로 적분하지 않으며 외란 사건은 목표 궤적과 독립적으로 섞인다. '
                 '구간을 알아보는 것과 힘을 풀어야 함을 학습하는 것은 별개다. 다음에는 진단용 전신 계측으로 '
                 '이 차이를 확인하고, 창 확장과 플랜트 개선을 한 번에 섞지 않는다. L3/L5처럼 하중을 일괄 제거하는 실험은 반복하지 않는다.</p></section>')
    return ''.join(parts)


def build_page(results, svg):
    sections = [INTRO, svg]
    for result in results.values():
        sections.append(f'<h2>{html.escape(result["name"])}</h2><p>전체 τ² 변화 {result["global_tau2_change_pct"]:+.2f}% · {html.escape(result["run_dir"])}</p>')
        sections.append('<div class="scroll"><table><tr><th>phase</th><th>착용자 RMS A→B</th><th>증가 B−A</th><th>보조 RMS</th><th>평균 착용자A / 보조</th><th>A와 반대방향 비율</th><th>R/L RMS 증가</th></tr>')
        for r in result["phase"]:
            values = [f'{r["phase_start_pct"]:.0f}–{r["phase_start_pct"]+10:.0f}%',
                      f'{r["agent_A_rms_Nm"]:.2f} → {r["agent_B_rms_Nm"]:.2f}',
                      f'{r["rms_increase_Nm"]:+.2f}', f'{r["assist_rms_Nm"]:.2f}',
                      f'{r["agent_A_mean_Nm"]:+.2f} / {r["assist_mean_Nm"]:+.2f}',
                      f'{r["opposite_to_A_pct"]:.1f}%',
                      ' / '.join(f'{v:+.2f}' for v in r["side_rms_increase_Nm"])]
            sections.append('<tr>' + ''.join('<td>'+v+'</td>' for v in values) + '</tr>')
        sections.append('</table></div>')
    sections.append('<p>RMS 증가 B−A는 양수일 때 악화.')
    sections.append('A와 반대방향 비율 자체는 인과효과가 아니며 실제 A/B 부담과 함께 해석한다. JSON의 고정 요구토크 항과 전신 반응 잔차는 정확한 대수 분해이며 반사실 시뮬레이션이 아니다.</p></html>')
    return ''.join(sections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="results/llp_L11_opposition_sequence/comparison.json")
    p.add_argument("--out", default="results/cli_walk_diagnostic_20261009/walking_diagnostic")
    p.add_argument("--input-diagnostic", help="Optional JSON from probe_walk_observations.py")
    p.add_argument("--phase-information", help="Optional JSON from probe_walk_phase_information.py")
    p.add_argument("--pendulum-balance", help="Optional JSON from probe_pendulum_balance.py")
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    selected = [("reference_L1_x0.5_m0", "L1 x0.5"),
                ("eval_llp_wide100_L6_e6000_m0", "L6 @6000"),
                ("eval_llp_L8_opposition_e4000_m0", "L8 @4000"),
                ("eval_llp_L9_opposition_e2000_m0", "L9 @2000")]
    results = {}
    for key, name in selected:
        record = manifest[key]
        path = Path(record["run_dir"])
        meta = json.loads((path/"meta.json").read_text())
        with np.load(path/"traces.npz") as tr:
            rows = diagnose(tr, meta)
        results[key] = dict(name=label(key), plot_name=name, run_dir=str(path),
                            global_tau2_change_pct=record["τ²"], phase=rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    svg = make_figure(results, out.with_suffix(".png"))
    out.with_suffix(".svg").write_text(svg)
    page = build_page(results, svg)
    if args.input_diagnostic:
        from probe_walk_observations import build_section
        source = Path(args.input_diagnostic)
        section = build_section(json.loads(source.read_text()), source.with_suffix(".svg").read_text())
        page = page.replace('</html>', section + '</html>')
    if args.phase_information:
        page = page.replace('</html>', phase_information_section(args.phase_information) + '</html>')
    if args.pendulum_balance:
        from probe_pendulum_balance import build_section as balance_section
        page = page.replace('</html>', balance_section(json.loads(Path(args.pendulum_balance).read_text())) + '</html>')
    out.with_suffix(".html").write_text(page)
    print(out.with_suffix(".html"))


if __name__ == "__main__":
    main()
