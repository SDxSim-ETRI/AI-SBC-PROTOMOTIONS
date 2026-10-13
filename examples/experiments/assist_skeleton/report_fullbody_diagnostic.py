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
"""Self-contained report of measured walking dynamics and a privileged ablation."""

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

from report_torque_html import aligned_torques, phase_profile


def summarize(path):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text())
    summary = json.loads((path / 'metrics.json').read_text())['summary']
    with np.load(path / 'traces.npz') as source:
        data = aligned_torques(source, meta)
    a, b = data['agent_A'], data['agent_B']
    regional = {}
    for lo, hi in ((20, 40), (70, 80), (80, 90), (70, 90)):
        mask = (data['phase'] >= lo/100) & (data['phase'] < hi/100)
        rms = lambda value: float(np.sqrt(np.mean(value[mask]**2)))
        regional[f'{lo}-{hi}'] = dict(A_rms_Nm=rms(a), B_rms_Nm=rms(b),
                                      increase_Nm=rms(b)-rms(a), assist_rms_Nm=rms(data['assist_B']))
    return dict(run_dir=str(path), tau2_change_pct=float(100*(np.mean(b*b)/np.mean(a*a)-1)),
                region=regional, phase=phase_profile(data, 10),
                body_error_A_m=summary['body_err_m_A'], body_error_B_m=summary['body_err_m_B'],
                speed_change_pct=summary['walk_speed_change_pct'], survived_s=meta['survived_s'],
                dynamics=json.loads((path / 'fullbody_analysis.json').read_text()))


def cell(value):
    return '<td>' + html.escape(str(value)) + '</td>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/walk_fullbody_diagnostic_20261010')
    args = parser.parse_args()
    out = Path(args.out)
    status = json.loads((out / 'sequence_status.json').read_text())
    if status['state'] != 'complete':
        raise RuntimeError('Do not report an unfinished or unvalidated sequence as complete')
    result = {name: summarize(item['run_dir']) for name, item in status['completed'].items()}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    labels = {'L9_fullbody40': 'L9 원래 출력', 'L9_fullbody40_mask70_90': 'L9 + 70–90% 보조 차단 (특권 진단)'}
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>걷기 전신 계측과 70–90% 제동 진단</title><style>'
             'body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#242b35;background:#faf9f6}'
             'svg{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;background:white}'
             'td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}pre{white-space:pre-wrap}</style>'
             '<h1>걷기: 전신을 실제로 기록해서 확인한 제동 구간</h1>'
             '<p>W1 + 실제 HLP + L9@2000, 걷기 40초, seed42. 무보조 A와 보조 B를 함께 실행했다. '
             '기존 정책/입력/게인/보상은 그대로다. 두 번째 실행만 참조 phase 70–90%에서 B의 보조를 0으로 했다. '
             '<b>참조 phase는 실기기에 없는 특권 정보다. 이 차단 실험은 원인 진단이며 새 배포 정책이 아니다.</b></p>'
             '<p>전체 관절·골반·PD 목표는100 Hz, 동역학과 발 접촉은400 Hz로 별도 계측했다. '
             '예비8초 실행에서 기존 반복실행 산포보다 계측 유무의 차이가 작음을 확인했다. '
             '모든 원본과 초기 계측 오류 로그도 보존했다. 표·그림은 전부 내장되어 HTML 하나로 볼 수 있다.</p>'
             '<h2>부담과 보행 유지</h2><div class="scroll"><table><tr><th>조건</th><th>전체 τ² 변화</th>'
             '<th>70–80 RMS A→B</th><th>80–90 RMS A→B</th><th>몸체 오차 A→B</th><th>속도 변화</th></tr>']
    for name, row in result.items():
        values = [labels[name], f'{row["tau2_change_pct"]:+.2f}%',
                  *(f'{row["region"][key]["A_rms_Nm"]:.2f} → {row["region"][key]["B_rms_Nm"]:.2f} Nm'
                    for key in ('70-80', '80-90')),
                  f'{100*row["body_error_A_m"]:.3f} → {100*row["body_error_B_m"]:.3f} cm',
                  f'{row["speed_change_pct"]:+.3f}%']
        parts.append('<tr>' + ''.join(map(cell, values)) + '</tr>')
    parts.append('</table></div><p>위 표의 RMS는 기존 보고서와 같은 10 ms 평균 토크 기준이다. '
                 '동역학 분해 표는400 Hz 항의 부호 있는 평균이며 RMS와 다르다. '
                 '낙상 여부는 각 meta의 survived_s로 검증했다. 한 모션·한 seed의 진단이며 일반화 입증은 아니다.</p>'
                 '<h2>다른 구간의 변화도 함께 확인</h2><div class="scroll"><table><tr><th>phase</th>'
                 '<th>원래 L9: B−A RMS</th><th>차단 진단: B−A RMS</th></tr>')
    for a, b in zip(result['L9_fullbody40']['phase'], result['L9_fullbody40_mask70_90']['phase']):
        values = [f'{a["phase_start_pct"]:.0f}–{a["phase_end_pct"]:.0f}%',
                  f'{-a["rms_reduction_Nm"]:+.2f} Nm', f'{-b["rms_reduction_Nm"]:+.2f} Nm']
        parts.append('<tr>' + ''.join(map(cell, values)) + '</tr>')
    parts.append('</table></div><p>양수는 무보조보다 악화. 40–50%에서는 절감 폭이 줄어드는 교환이 남았다. '
                 '70–80%의 보조 RMS가 완전히0이 아닌 것은 시간창 중점으로 구간을 나누기 때문이다. '
                 '70% 직전에 결정한 토크가 경계를 걸치는 창에 남으며, 결정 시점70–90%의 보조는 정확히0임을 검증했다.</p>'
                 '<h2>어떤 구간인가: 접촉·무릎·착용자 목표</h2><div class="scroll"><table><tr>'
                 '<th>조건</th><th>A/B</th><th>phase</th><th>발 접촉 비율 (&gt;20 N)</th>'
                 '<th>무릎 각도</th><th>무릎 각속도</th><th>hip 각도</th><th>착용자 hip PD 목표</th>'
                 '<th>착용자 평균 토크</th><th>평균 보조</th></tr>')
    for name, row in result.items():
        for r in row['dynamics']['phase_rows']:
            if r['phase_pct'][0] not in (70, 80):
                continue
            values = [labels[name], r['environment'], f'{r["phase_pct"][0]}–{r["phase_pct"][1]}%',
                      f'{100*r["foot_contact_fraction"]["20"]:.2f}%', f'{r["knee_angle_deg"]:.2f}°',
                      f'{r["knee_velocity_rad_s"]:.2f} rad/s', f'{r["hip_angle_deg"]:.2f}°',
                      f'{r["hip_target_deg"]:.2f}°', f'{r["wearer_mean_Nm"]:+.2f} Nm',
                      f'{r["assist_mean_Nm"]:+.2f} Nm']
            parts.append('<tr>' + ''.join(map(cell, values)) + '</tr>')
    parts.append('</table></div><p>무릎 각도는 자산 규약상 굴곡이 음수다. 각속도 양수는 이 구간에서 펴지는 방향이다. '
                 '접촉은 실제 발 힘으로 확인했다. 참조 phase0% 자체를 착지로 가정하지 않았다. '
                 'PD 목표는 착용자 정책의 토크 생성 목표이며 HLP가 예측하는 실제 미래각과 같은 변수가 아니다.</p>'
                 '<h2>무보조 전신의 고관절 토크 분해</h2><p>관성행렬 M의 hip 행을 사용해 '
                 '고관절 자체·같은 쪽 무릎·골반·나머지 관절 가속도의 기여를 나눴다. '
                 '중력·속도항과 접촉/기타 구속력도 분리했다. '
                 '<b>같은 궤적에서의 운동방정식 항 분해다. 무릎을 고정하거나 제거한 인과효과가 아니다.</b></p>'
                 '<div class="scroll"><table><tr><th>phase</th><th>hip 자체 관성</th><th>무릎 결합</th>'
                 '<th>골반 결합</th><th>다른 관절</th><th>중력</th><th>속도항</th><th>−접촉항</th><th>착용자 합계</th></tr>')
    for r in result['L9_fullbody40']['dynamics']['phase_rows']:
        if r['environment'] != 'A' or r['phase_pct'][0] not in (70, 80):
            continue
        keys = ('hip_inertia', 'ipsilateral_knee_coupling', 'pelvis_coupling', 'other_joint_coupling',
                'gravity', 'velocity_bias', 'minus_contact_constraint')
        values = [f'{r["phase_pct"][0]}–{r["phase_pct"][1]}%'] + [f'{r["terms_mean_Nm"][k]:+.2f}' for k in keys]
        values += [f'{r["wearer_mean_Nm"]:+.2f}']
        parts.append('<tr>' + ''.join(map(cell, values)) + '</tr>')
    parts.append('</table></div><p>단위 Nm, 부호 있는 평균. 위 표에서 생략한 수동항·기타구속·수치잔차는 JSON에 모두 보존했다. '
                 '가속도는 솔버의 힘 평형 단계 값으로, implicit 적분 후 Δ속도/Δt와 구별한다.</p>'
                 '<h2>계측의 신뢰 범위</h2><p>GPU 관성행렬·편향항은 별도 CPU MuJoCo120상태와 대조했다. '
                 'MJWarp sparse solver는 일부 world가 수렴한 뒤 qfrc_constraint 작업 버퍼를 지울 수 있어 '
                 '그 배열을 그대로 해석하지 않고 최종 활성 제약의 Jᵀf로 재구성했다. 원본 작업 버퍼도 저장했다. '
                 '남은 솔버 수치 잔차를 별도 항으로 기록했으며 생체역학 기여로 해석하지 않았다.</p>')
    for name, row in result.items():
        parts.append('<h3>' + labels[name] + '</h3><pre>'
                     + html.escape(json.dumps(row['dynamics']['validation'], indent=2, ensure_ascii=False)) + '</pre>')
        source = (Path(row['run_dir']) / 'fullbody_report.html').read_text()
        parts.append(re.search(r'<svg\b.*?</svg>', source, flags=re.S).group())
    parts.append('</html>')
    (out / 'comparison.html').write_text(''.join(parts))
    print(out / 'comparison.html')
    print(json.dumps({k: {f: v for f, v in row.items() if f not in ('phase', 'dynamics')}
                      for k, row in result.items()}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
