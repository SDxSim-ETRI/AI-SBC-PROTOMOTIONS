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
"""Standalone, evidence-derived HTML for completed and ongoing pendulum trials."""

import argparse
from datetime import datetime
from html import escape
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
INITIAL = ROOT / "results/pendulum_research_nativeff_20261014"
FOLLOWUP = ROOT / "results/pendulum_research_followup_20261015"
ADAPTATION = ROOT / "results/pendulum_adaptation_20261015"
LIMIT_CONTROL = ROOT / "results/pendulum_limit_control_20261015"
CONDITIONS = {"legacy": "기존 외란 · 목표 10 ms", "structured": "운동 의존 외란 · 목표 10 ms",
              "hold50": "기존 외란 · 목표 50 ms"}
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(), "svg.fonttype": "none",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>'
            + ''.join(f'<th>{escape(x)}</th>' for x in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def numbers(label, report):
    m = report["metrics"]["all"]
    return [label, f"{m['reference_tau_sq_Nm2']:.1f} → {m['assisted_tau_sq_Nm2']:.1f}",
            f"{m['reference_tracking_rmse_deg']:.2f} → {m['assisted_tracking_rmse_deg']:.2f}",
            f"{m['assisted_abs_work_W']:.1f}",
            f"{report['failures']['reference']} / {report['failures']['assisted']}"]


def figure(reports, l1, h1):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    names = ["무보조", "L1 × 0.5", "C0", "S1", "A1 · 실패 증가", "T1"]
    values = [l1['metrics']['all']['reference_tau_sq_Nm2'],
              l1['metrics']['all']['assisted_tau_sq_Nm2']] + [
        reports[f'eval_{v}_e2000_legacy']['metrics']['all']['assisted_tau_sq_Nm2']
        for v in ('C0', 'S1', 'A1', 'T1')]
    bars = axes[0].barh(names, values, color=['#a9b1bb', '#27765d', '#377bb5', '#c18e37', '#c35454', '#8672a6'])
    bars[4].set_hatch('///')
    axes[0].invert_yaxis()
    axes[0].bar_label(bars, fmt='%.0f', padding=4)
    axes[0].set_xlim(0, 3300)
    axes[0].set_title('첫 2,000 epoch 비교 · 기존 외란')
    axes[0].set_xlabel('사람 토크 제곱 평균 [N·m²/관절] · 낮을수록 작음')
    old = [l1['metrics']['all']['assisted_tau_sq_Nm2'],
           reports['eval_C0_e2000_legacy']['metrics']['all']['assisted_tau_sq_Nm2']]
    real = [h1[name]['metrics']['all']['assisted_tau_sq_Nm2'] for name in ('L1', 'C0')]
    a = axes[1].bar([-.18, .82], old, width=.36, color='#6796b6', label='합성 의도 예측')
    b = axes[1].bar([.18, 1.18], real, width=.36, color='#d59a45', label='실제 H1')
    axes[1].bar_label(a, fmt='%.0f', padding=3)
    axes[1].bar_label(b, fmt='%.0f', padding=3)
    axes[1].set_xticks([0, 1], ['L1 × 0.5', 'C0 @ 2,000'])
    axes[1].set_ylim(0, 2900)
    axes[1].axhline(values[0], color='#8b95a3', linestyle='--', label='무보조 2386')
    axes[1].set_title('예측기 교체 시 보조 효과 감소')
    axes[1].set_ylabel('사람 토크 제곱 평균 [N·m²/관절]')
    axes[1].legend(loc='upper left', fontsize=8, ncol=2)
    buffer = io.StringIO()
    fig.savefig(buffer, format='svg')
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index('<svg'):]


def generate(output):
    sys.path.insert(0, str(ROOT / 'docs' / 'ai_sbc'))
    from organize_reports import shell

    reports = read(INITIAL / 'comparison.json')
    l1_dirs = {'legacy': 'eval_L1_combined_human', 'structured': 'eval_L1_native_structured',
               'hold50': 'eval_L1_native_hold50'}
    l1 = {key: read(ROOT / 'results/pendulum_research_20261014' / name / 'summary.json')
          for key, name in l1_dirs.items()}
    h1 = {name: read(FOLLOWUP / folder / 'summary.json') for name, folder in
          [('L1', 'eval_L1_H1_legacy'), ('C0', 'eval_C0_e2000_H1_legacy')]}
    completed_followup = read(FOLLOWUP / 'followup_comparison.json', {})
    active = ADAPTATION if (ADAPTATION / 'followup_plan.json').exists() else FOLLOWUP
    state = read(active / 'followup_status.json', {'stage': '학습 시작 · 평가 실행기 준비'})
    live = read(active / 'followup_comparison.json', {})
    stage = state.get('stage', '상태 확인 중')
    ongoing = not (stage.startswith('complete:') or stage == 'stopped with error')
    headline = '첫 비교는 완료됐습니다. 기존 L1보다 좋은 새 모델은 아직 검증되지 않았습니다.'
    if stage.startswith('complete:'):
        headline = ('이번 후속 비교가 끝났습니다. 기존 L1을 유지합니다.' if 'retain L1' in stage
                    else '후보의 독립 진자 검증과 동결 W1 전이 평가를 완료했습니다. 배포 모델은 변경하지 않았습니다.')
    elif stage == 'stopped with error':
        headline = '후속 실행에서 오류가 발생했습니다. 아래에는 오류 전 완료한 결과만 표시합니다.'
    if active == ADAPTATION and ongoing:
        headline = 'C0 연장·AB1 비교는 10:42에 완료됐고 L1을 넘지 못했습니다. L1에서 출발하는 진자 추가학습을 진행합니다.'
    now = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    body = f'''<header><p class="eyebrow">AI-SBC · 진자만 학습 · {now} 갱신</p>
<h1>진자 사전학습 개선 실험</h1>
<p class="lead">{headline}</p>
<div class="cards"><div><b>4개 모델 · 24건 평가</b><span>각 2,000 epoch 학습 · 10월 15일 01:46 완료</span></div>
<div><b>실제 H1 연결 비교 완료</b><span>L1·C0 각 256회 · 보조 추종 실패 0회</span></div>
<div><b>후속 평가 29건 완료</b><span>C0 총 6,000 / AB1 2,000 epoch · 10월 15일 10:42 종료</span></div></div></header>
<nav class="toc"><a href="#findings">확인된 결론</a><a href="#matrix">첫 비교</a><a href="#h1">실제 H1</a><a href="#live">후속 결과</a><a href="#method">조건·근거</a></nav>
<section id="findings"><h2>현재 확인된 결론</h2>
<div class="pending"><b>이 결과는 20/40 N·m 학습 한계의 효과를 분리한 비교가 아닙니다.</b>
L1은 기존 토크 처리에서 40 N·m로 학습했고, C0는 사람 PD+FF와 외부 주입 토크 처리를 수정한 환경에서 20 N·m로 학습했습니다.
아래 수치는 공통 평가 환경에서 저장 모델을 비교한 결과입니다. 이 수치로 “40으로 학습한 뒤 반으로 줄이는 방식이 더 낫다”거나 “20 직접 학습이 실패했다”고 결론 내릴 수 없습니다.
진자 학습 개선과 학습 한계의 인과 효과는 아직 확정되지 않았습니다.</div>
<p>새 기본형 C0는 진자 무보조 대비 부담을 줄였지만 기존 L1에는 못 미쳤습니다.
외란을 바꾼 S1은 C0와 비슷했고, 사람 목표 갱신을 50 ms로 바꾼 T1은 더 나빴습니다.
공유 관절 정책 A1은 보조 토크가 한계에 붙고 실패가 늘어 후보에서 제외했습니다.</p>
<p><strong>실제 H1을 연결하면 보조 효과가 줄었습니다.</strong> 합성 예측 점수만으로 후속 모델을 고르지 않습니다.
W1은 이후 전이 평가에만 쓰며, 전신 보행 학습이나 W1 토크 정답을 사용하지 않습니다.</p>
{figure(reports, l1['legacy'], h1)}
<p class="note">L1은 기존 6,000 epoch 모델입니다. 새 모델과 학습 예산·학습 환경이 달라, 이 그림만으로 구조의 우위를 판단하지 않습니다.
A1은 실패로 생존 구간도 달라 단순 순위 비교에서 제외합니다. 아래 표에는 실패 횟수를 함께 표시했습니다.</p></section>
<section id="matrix"><h2>첫 비교: 같은 진자 조건에서 무엇이 달랐나</h2>
<p>C0: 새 기본형 / S1: 운동에 따라 달라지는 외란 / A1: 좌우에 같은 단일 관절 신경망 적용 /
T1: 사람 PD·속도 보상 목표를 50 ms마다 갱신. 모든 새 모델은 보조 한계 20 N·m입니다.</p>'''
    headers = ['모델', '사람 τ² [N·m²/관절] 무보조 → 보조', '추종 RMSE [°] 무보조 → 보조',
               '보조 시 |τ·ω| [W/관절]', '실패 무보조 / 보조 (각 256회)']
    for condition, title in CONDITIONS.items():
        rows = [numbers('L1 × 0.5 · 기존 기준', l1[condition])]
        rows += [numbers(v + ' @ 2,000', reports[f'eval_{v}_e2000_{condition}'])
                 for v in ('C0', 'S1', 'A1', 'T1')]
        body += f'<h3>{title}</h3>' + table(headers, rows)
    body += '''<p class="note">τ²는 시뮬레이션 부담 지표이며 실제 근활성·대사량이 아닙니다.
조건별 값은 공통 생존 구간의 평균입니다. 조건 간 외란 분포가 다르므로 서로 다른 표의 차이를 모델 개선으로 읽지 않습니다.</p></section>
<section id="h1"><h2>실제 예측기 H1을 연결한 결과</h2>
<p>합성 예측은 미래 목표 운동을 알려주지만, H1은 실제 운동 이력으로 미래 각도를 예측합니다.
같은 목표·관성·사람 게인을 유지하고 정책에 들어가는 예측만 교체했습니다.</p>'''
    body += table(headers, [numbers('L1 · 합성 예측', l1['legacy']), numbers('L1 · 실제 H1', h1['L1']),
                            numbers('C0 · 합성 예측', reports['eval_C0_e2000_legacy']),
                            numbers('C0 · 실제 H1', h1['C0'])])
    if 'eval_C0_e6000_H1_legacy' in completed_followup:
        body += '<h3>연장 학습까지 끝낸 실제 H1 비교</h3>'
        body += table(headers, [numbers(label, completed_followup[key]) for label, key in [
            ('기존 L1 × 0.5', 'eval_L1_H1_legacy'),
            ('C0 @ 5,000 · C0 최선', 'eval_C0_e5000_H1_legacy'),
            ('C0 @ 6,000', 'eval_C0_e6000_H1_legacy'),
            ('AB1 @ 2,000 · 실패로 제외', 'eval_AB1_e2000_H1_legacy')]])
        body += '<p><b>이 후속 실행에서 통과한 후보는 없습니다.</b> C0는 학습 연장으로 개선됐지만 L1보다 부담·추종 오차가 컸습니다. AB1은 포화와 추종 실패가 남았습니다. 조건부 추가 seed·W1 전이 평가는 실행하지 않았습니다.</p>'
    body += '''<p>L1의 사람 τ²는 1549.0에서 1878.7 N·m²/관절로 늘었습니다.
차이 329.7의 plant cluster bootstrap 95% 구간은 315.7–343.8입니다.
이 구간은 고정 평가 표본의 산포이며 학습 seed 간 재현성을 뜻하지 않습니다.</p>
<p class="note">오프라인 20 ms 각도 예측 MAE는 H1 0.41°, 합성 예측 1.35°였지만 보조 성능은 H1에서 더 낮았습니다.
예측 오차와 예측 정보의 의미 차이를 이번 교체만으로 분리할 수는 없습니다.</p></section>
<section id="live"><h2>추가 실험의 현재 진행과 완료 결과</h2>'''
    progress = []
    plan = read(active / 'followup_plan.json', {'jobs': []})
    for job in plan['jobs']:
        label, target = job['label'], max(job['epochs'])
        log = ('C0_extend_train.log' if label == 'C0' else f'{label}_train.log')
        if not (active / log).exists():
            continue
        with (active / log).open('rb') as stream:
            stream.seek(max(0, (active / log).stat().st_size - 16000))
            epochs = re.findall(r'Epoch (\d+),', stream.read().decode(errors='replace'))
        finished = (ROOT / job['directory'] / f'epoch_{target}.ckpt').exists()
        current = target if finished else (epochs[-1] if epochs else '준비')
        progress.append([label, f"{current} / {target}", '학습 완료' if finished else '학습 로그의 최근 epoch'])
    body += table(['실험', '진행 / 목표 epoch', '읽는 법'], progress)
    if active == ADAPTATION:
        body += '<p>PA_E는 기존 L1 actor에서 합성 예측으로, PA_H1은 같은 L1 actor에서 동결 H1의 실제 예측으로 추가학습합니다. 두 조건 모두 진자만 사용하며 512환경·보조 ±20 N·m·새 critic/optimizer·actor 학습률 3×10⁻⁵를 사용합니다. 100/300/600/1,000/2,000 epoch를 평가합니다.</p><p>예비검사 통과: H1 연결 전후 무보조 물리·목표·외란 난수 동일, 평가와 학습 예측 차이 최대 1.53×10⁻⁷ rad, 부분 reset의 H1 이력 분리, 초기 L1 actor 출력 동일. H1/W1은 학습하지 않습니다.</p>'
    body += f'<p><b>실행 단계:</b> <code>{escape(stage)}</code></p>'
    body += '''<p>저장 모델이 나오면 같은 평가를 이어갑니다. 실제 H1 조건에서 부담·추종·일률이 L1보다 개선되고 실패가 없는 후보는
다른 학습 seed와 별도 평가 조건으로 확인합니다. 그 검증을 통과한 경우 고정 W1 걷기·달리기 전이를 평가합니다.</p>
<p class="note">위 진행 수치는 실제 로그와 저장 파일에서 읽습니다. 완료 전 성능을 예상하지 않으며, 후보 검증이 끝나도 전체 개선 작업이 끝났다는 뜻은 아닙니다.</p>'''
    extra = {name: report for name, report in live.items() if name != 'eval_L1_H1_legacy'}
    body += table(headers, [numbers(name, report) for name, report in extra.items()]) if extra else (
        '<p class="pending">후속 모델의 저장·평가를 기다리는 중입니다. 아직 끝나지 않은 학습의 성능값은 표시하지 않습니다.</p>')
    if state.get('selected'):
        body += '<p>선별된 후속 후보: <code>' + escape(state['selected']['checkpoint']) + '</code>. 최종 배포 성공 판정은 아닙니다.</p>'
    body += f'<p>{"이 파일은 실행 중 60초마다 새로 생성됩니다. 브라우저에서 새로고침하면 갱신된 수치를 볼 수 있습니다." if ongoing else "이 실행의 마지막 결과입니다. 새 실험을 시작하면 다시 갱신합니다."}</p></section>'
    if active == ADAPTATION:
        body += '<p><a href="../results/pendulum_adaptation_20261015/followup_report.md">현재 추가학습 상태·평가 원본</a> · <a href="../results/pendulum_adaptation_20261015/preflight.json">예비검사 결과</a></p>'
    limit_state = read(LIMIT_CONTROL / 'status.json', {})
    if limit_state:
        body += '<section id="limit-control"><h2>학습 한계의 효과: 빠져 있던 대조 실험</h2><p>기존 토크 처리 20/40, 수정 토크 처리 20/40의 네 조건을 처음부터 학습합니다. 각 조건은 4,096환경·6,000 epoch·학습 seed 0/1이며, 마지막 6,000 epoch를 사전에 고정합니다. 평가에서는 20 학습은 ×1, 40 학습은 ×0.5로 모두 20 N·m를 적용합니다.</p>'
        body += '<p><b>현재 상태:</b> <code>' + escape(limit_state['stage']) + '</code></p>'
        audit = read(LIMIT_CONTROL / 'config_audit.json', {})
        if audit.get('passed'):
            body += '<p>저장 설정 비교 통과: 각 20/40 쌍의 차이는 실험명, 보조 한계, action→토크 배율뿐입니다. 예측·보상·사람 제어·외란·관성·신경망·학습 예산은 같습니다.</p>'
        limit_results = read(LIMIT_CONTROL / 'comparison.json', {})
        if limit_results:
            body += table(headers, [numbers(key, value) for key, value in limit_results.items()])
        else:
            body += '<p><b>이 대조 실험의 성능 결과는 아직 없습니다.</b> 기존 L1/C0 값으로 빈 결과를 대신하지 않습니다.</p>'
        body += '<p class="note">기존 처리 경로는 당시 저장 설정을 현재 코드로 재구성한 것입니다. 당시 실행의 비트 단위 재현은 아닙니다. 두 seed만으로 통계적 우월성을 확정하지 않습니다. PA_E/PA_H1 추가학습은 이 대조 실험을 대신하지 않습니다.</p><p><a href="../results/pendulum_limit_control_20261015/report.md">대조 실험 상태·결과</a> · <a href="../results/pendulum_limit_control_20261015/plan.json">고정 실행 계획</a></p></section>'
    body += '''
<section id="method"><h2>평가 조건과 재현 근거</h2>
<ul><li>각 조건은 128환경 × 2코호트 × 10초, 초기 0.5초 제외. 실패 후 다른 목표를 다시 샘플링하지 않습니다.</li>
<li>물리 2.5 ms, 보조 10 ms. 50 ms는 사람 역할 제어기의 목표 갱신 비교이며 물리 한 틱이 아닙니다.</li>
<li>사람 PD+속도 보상 전체를 ±150 N·m로 제한하고 보조·외란은 별도로 더합니다. 실제 토크 주입 오차는 0이었습니다.</li>
<li>L1 출력은 40·tanh(action) × 0.5, 새 정책은 20·tanh(action)입니다.</li>
<li>모델별 목표·관성·기본 게인과 무보조 첫 실패가 같음을 원본 배열에서 확인했습니다.</li>
<li>실제 H1은 고정 모델, denoise 1, 몸통 pitch/roll 0입니다. H1 난수 때문에 목표·외란이 바뀌지 않는 검사도 통과했습니다.</li></ul>
<details><summary>1,000/2,000 epoch 평가 24건 전체</summary>'''
    body += table(headers, [numbers(name, report) for name, report in reports.items()])
    body += '''</details><h3>원본 파일</h3><ul>
<li><a href="../results/pendulum_research_nativeff_20261014/comparison.json">첫 비교 24건 JSON</a> 및 각 평가 폴더의 rollout.npz</li>
<li><a href="../results/pendulum_research_followup_20261015/H1_vs_emulated_L1.json">H1 교체 비교·bootstrap</a></li>
<li><a href="../results/pendulum_research_followup_20261015/followup_report.md">후속 실행 상태와 완료 수치</a></li>
<li><a href="pendulum_pretrain_discussion.md">설계·가설·실행안</a> · <a href="HANDOFF.md">현재 작업 상태</a> · <a href="AI_SBC_VERSIONS.md">모델 경로</a></li>
</ul></section>'''
    styles = '''<style>.eyebrow,.note{color:#596b7d;font-size:13px}.lead{font-size:19px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.cards>div{background:#fff;border:1px solid #d8e0e7;border-top:4px solid #357862;border-radius:8px;padding:18px}.cards b,.cards span{display:block}.cards span{font-size:13px;color:#596b7d;margin-top:8px}.toc{display:flex;flex-wrap:wrap;gap:18px;padding:12px 0;border-bottom:1px solid #d8e0e7}.toc a{text-decoration:none}section{margin-top:32px}table{border-collapse:collapse;width:100%;background:#fff;font-size:13px}th,td{border:1px solid #d8e0e7;padding:9px 12px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf2f6}svg{width:100%;height:auto;background:white;border-radius:8px}code{font-size:12px;overflow-wrap:anywhere}.pending{padding:16px;background:#fff5db;border-left:4px solid #d59a45}details{margin:24px 0}summary{cursor:pointer;font-weight:bold}@media(max-width:700px){.cards{grid-template-columns:1fr}.lead{font-size:17px}}</style>'''
    temporary = output.with_suffix('.tmp')
    temporary.write_text(shell('진자 사전학습 개선 실험', body, styles))
    temporary.replace(output)
    subprocess.run([sys.executable, 'docs/ai_sbc/organize_reports.py'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f'{now}: wrote {output}', flush=True)
    adaptation_finished = stage.startswith('complete:') or stage == 'stopped with error'
    limit_finished = (not limit_state or limit_state['stage'].startswith('complete:')
                      or limit_state['stage'] == 'stopped with error')
    return adaptation_finished and limit_finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/ai_sbc/pendulum_research_report.html')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        finished = generate(ROOT / args.out)
        if not args.watch or finished:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
