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
"""Organize the AI-SBC HTML reports without changing their evidence arrays.

Run from the repository root after regenerating an individual report:
    python docs/ai_sbc/organize_reports.py
HTML reports remain standalone. Retired URLs become explicit archive links.
"""
from collections import Counter
from dataclasses import dataclass, field
from html import escape, unescape
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re

DOCS = Path(__file__).resolve().parent
MARKER = '<!-- AI-SBC report presentation v1 -->'
CATALOG = [
    dict(file='261022_llp20_motion_shared/report.html', group='current', title='LLP 20 N·m · 단일 관절 학습·양다리 공유·실제 동작 궤적 · A·Δ 2×2 (그래프)', date='2026-10-22',
         purpose='R0 레시피를 20·tanh, SEED 실제 고관절 동작(가벼운 다리 집단 포함), 단일 관절 학습 후 양다리 가중치 공유로 다시 학습하고, 보조 토크² 벌점 A·명령 변화 벌점 Δ를 켜고 끈 네 모델을 W1·진자(SEED 미학습 동작, ETRI 13동작, 합성 궤적)에서 L1·R0와 비교',
         result='네 모델 모두 W1에서 L1 ±20보다 덜 줄였다(기존 걷기 149/−123/−132/81, 달리기 637/413/229/155 vs L1 278·756). 학습 분포 진자에서는 에뮬레이션 예측으로 τ² 32–36 % 감소(R0 30 %)지만 실제 H1 예측이면 7–14 %로 떨어져, 학습 때 예측 신호와 배포 H1의 차이가 원인으로 보인다. A·Δ 효과는 평가마다 방향이 다르다.',
         limit='학습 seed 하나·W1 하나·걷기와 달리기 기록 하나씩. 세 공통 변경을 함께 적용해 R0와의 차이를 하나의 원인으로 나눌 수 없다. 10 Hz 저역통과는 모터 모델이 아닌 민감도 점검.'),
    dict(file='261021_r0_generalization/report.html', group='current', title='R0 일반화 · 달리기와 다른 걷기 구간에서 L1 대비 개선 유지 여부 (그래프)', date='2026-10-21',
         purpose='R0가 평가된 적 없는 조건(달리기, 같은 걷기 기록의 400–440 s 구간)에서 L1 ±10/15/20·R0 ±15/20 N·m를 같은 W1·H1으로 평가하고, 같은 보조 RMS·같은 저부담 손해의 L1 곡선과 비교',
         result='같은 보조 RMS에서 R0 ±20 이득은 L1 곡선보다 기존 걷기 +18 %, 400–440 s 구간 +35 %로 유지됐지만 달리기는 −34 %(±20 이득 436 vs L1 756)로 유지되지 않았다. 달리기에서 R0 보조는 필요 토크가 큰 구간에서 오히려 작아진다. 낙상 0.',
         limit='W1 하나·걷기와 달리기 기록 하나씩·R0 학습 seed 하나. W1이 학습하지 않은 동작은 무보조에서도 넘어져 평가하지 못했고, 0.85·1.15배 재생 걷기는 실제 속도 변화가 아니라 제외했다.'),
    dict(file='261020_relative/phase_report.html', group='current', title='상위 5개 LLP와 L1 · 걷기 위상별 도움과 방해 (그래프)', date='2026-10-20',
         purpose='학습 전 고정한 점수로 고른 상위 5개(R0·F1·R1·F2·G0)와 L1을 ±20 N·m에서 위상별·필요 토크별·다른 관절·진자 일반화 그래프로 비교',
         result='R0만 L1보다 점수가 높다. 저부담·최악 위상 방해가 절반 이하지만 보조 RMS도 절반이며, 배율 1로 쓰면 L1과 같은 이득·방해로 돌아간다. 필요 토크 5 N·m 미만에서 보조 방향은 모든 모델이 무작위 수준이다.',
         limit='저장 기록만 사용한 그래프 요약이다. 기존 걷기 한 클립·동결 W1/H1·학습 seed 하나.'),
    dict(file='261020_relative/report.html', group='current', title='진자 사전학습 R0/R1 · 토크 경로 대조와 무보조 대비 토크 증가 비용', date='2026-10-20',
         purpose='L1→R0(올바른 토크 경로만)→R1(R0+무보조 진자 대비 착용자 토크 증가 비용)로 한 번에 한 가지씩 바꿔 새 PPO 2개를 학습하고 W1·진자에서 L1/G0/G4/F1/F2와 같은 출력으로 비교',
         result='R0 ±20 N·m: 고관절 τ² 감소 262.03(L1 278.04), 저부담 감속 RMS 증가 3.27(L1 7.67), 최악 위상 7.57(L1 14.94) N·m, 포화 0.5 %. 과거 토크 경로가 L1 포화 출력의 주원인. R1은 출력만 줄고(이득 135.59) 방향 정확도가 떨어져 채택하지 않음.',
         limit='학습 seed 하나, 기존 걷기 한 클립·동결 W1/H1. 엄격한 공동 기준(평균 이득 ≥ L1)은 R0도 미달. 실기기 검증 아님.'),
    dict(file='261020_feasible/report.html', group='current', title='진자 사전학습 F1/F2 · 실제 PPO와 W1 전이', date='2026-10-20',
         purpose='올바른 토크 경로+관성 제한 C2 전환(F1)과 흡수 파워 비용 추가(F2)를 처음부터 학습해 L1/G0/G4와 같은 출력에서 비교',
         result='F2 ±20 N·m는 작은 보조 RMS로 비슷한 고관절 이득(284.35)을 냈지만 저부담 손해(8.07)와 진자 일반화가 L1보다 나빴다. F1은 손해는 작지만 이득이 크게 줄었다. L1 대체 모델 없음.',
         limit='F1 대 L1은 토크 경로와 궤적을 함께 바꾼 비교다. 학습 seed 하나, 동결 W1/H1 평가.'),
    dict(file='261016_output_scale/report.html', group='current', title='공통 출력 크기와 저부담 감속 진단', date='2026-10-16',
         purpose='L1·G0·G4의 실제 출력 범위를 맞춰 크기 효과와 저부담 감속 방해를 분리', result='24건 완료. L1의 20→10 N·m에서 저부담 감속 RMS 증가 7.67→2.27 N·m. 같은 20 N·m에서도 G4는 11.29/11.01 N·m로 차이가 남았다.', limit='기존 걷기 한 클립·동결 W1/H1 평가이며 새 학습이나 실기기 검증이 아니다.'),
    dict(file='experiment_summary.html', group='start', title='전체 실험 결과 요약', date='2026-10-20',
         purpose='연구의 흐름과 현재 해결할 문제', result='R0(올바른 토크 경로만)가 같은 ±20 N·m에서 L1 이득의 94 %를 유지하며 저부담·최악 위상 손해를 절반 이하로 줄였다. 범용 L1 대체 확정은 아니며 반복 seed 검증이 남았다.', limit='범용 보행이나 실기기 성공을 입증한 결과는 아니다.'),
    dict(file='261015_1815/training_plan.html', group='current', title='23 N·m LLP 일반화 학습 · 설계와 설명', date='2026-10-15',
         purpose='G1 기본·G0 대조를 포함한 다섯 조건, PPO·tanh와 scale·H1 평가의 역할 정리', result='무작위 초기화 ±23 N·m, PA 제외, L1 ×0.5 유지. 7개 학습·평가를 완료했으며 설계와 최종 결과를 구분해 기록했다.', limit='설계 문서이며 성능 결과가 아니다. 짧은 충격 외란은 순간 관성 변화나 heel strike 접촉의 재현이 아니다.'),
    dict(file='261015_1815/walking_phase_report.html', group='current', title='23 N·m LLP · 보행 phase별 도움과 방해', date='2026-10-16',
         purpose='각 후보와 L1의 좌우 위상별 착용자 부담·보조 토크를 시각적으로 비교', result='8개 모두 40초 완주. G4의 보행 후반 방해는 L1보다 컸고 G0는 평균 개선에도 다른 구간 방해가 남았다.', limit='기존 걷기 한 클립의 동결 W1/H1 진단이다. 실기기 성공이나 최종 순위를 뜻하지 않는다.'),
    dict(file='261015_1815/FINAL_RESULTS.html', group='current', title='23 N·m LLP · 최종 판단과 남은 문제', date='2026-10-16',
         purpose='계산 완료와 일반 모델 목표 달성을 구분', result='진자 선정 G4도 별도 난수 평가에서 L1 기준 미달. 조건부 달리기는 생략했고 모든 후보의 걷기는 완료했다.', limit='L1을 대체할 일반 모델을 확보했다는 결론이 아니다.'),
    dict(file='261015_1815/pendulum_results.html', group='current', title='23 N·m LLP · 진자 평가와 음수 reward 기여', date='2026-10-16',
         purpose='40개 평가의 부담·추종·일률과 실제 보상 기여 비교', result='보조 비용 강화는 포화를 줄였으나 L1 대체 실패. 사람 비용 2배만으로 개선되지 않았다.', limit='보상 기여는 평가 궤적에 학습 가중치를 적용한 값이며 학습 전체 평균이 아니다.'),
    dict(file='pendulum_candidates_report.html', group='current', title='진자 LLP 후보 3종 · 그래프 비교', date='2026-10-15',
         purpose='PA_E @100·L1 ×0.5·C0 @5000의 실제 H1 진자 평가를 부담·추종·일률 그래프로 비교', result='고정 평가에서 PA_E @100은 L1보다 부담·추종·일률을 소폭 개선했다. 더 긴 PA_E 학습은 부담을 줄여도 추종 오차가 커졌다.', limit='앞서 선정한 세 계열의 고정 비교다. 전체 모델의 확정 순위나 여러 seed·W1 전이 검증 결과가 아니다.'),
    dict(file='pendulum_research_report.html', group='current', title='진자 사전학습 개선 실험', date='2026-10-15',
         purpose='진자만 학습한 네 모델의 첫 비교와 실제 H1 연결, 후속 검증 상태', result='각 2,000 epoch·24평가 완료. 첫 비교에서 새 모델은 L1 기준에 미달했고, 실제 H1 연결 시 보조 효과도 줄었다. 추가학습·검증 결과를 본문에 갱신한다.', limit='진자 내 결과다. 후속 후보의 재현성과 W1 전이는 각각 완료 여부를 확인해야 한다.'),
    dict(file='walking_retarget_report.html', group='current', title='실제 보행 변환과 착용자 추종', date='2026-10-13',
         purpose='새 원본 보행을 현재 가상 착용자가 따라갈 수 있는지 확인', result='보통·느림 16조건에서 15낙상, 1골반 하강 중 종료. 같은 기존 걷기도 참조 표현을 바꾸면 낙상했다.', limit='보조를 끈 평가다. 데이터 표현·발 접촉·착용자 호환성을 먼저 해결해야 한다.'),
    dict(file='walking_finetune_report.html', group='current', title='전신 걷기 보조 학습 결과', date='2026-10-11',
         purpose='진자에서 배운 보조 모델을 전신 걷기에서 개선', result='기본 걷기의 고관절 부담은 약 69% 줄었다. 느리게 재생한 조건과 허리·무릎 부담은 해결하지 못했다.', limit='0.85–1.15배는 같은 동작의 재생 속도다. 실제 속도별 보행의 보조 효과와 구분한다.'),
    dict(file='walking_data_audit.html', group='current', title='실제 보행 자료의 범위와 속도', date='2026-10-12',
         purpose='재생 속도 변형과 실제로 다르게 걸은 자료를 구분', result='실제 보통·느림 자료를 확인했고, 빠른 후보의 정지·회전 구간을 선별할 필요가 있었다.', limit='자료 점검 보고서다. 후속 변환·무보조 평가 결과는 착용자 추종 보고서에서 확인한다.'),
    dict(file='l1_w1_report.html', group='baseline', title='기본 보조 모델의 비교 기준', date='2026-10-01',
         purpose='사람 범위 근력의 가상 착용자에서 기존 성능을 재확인', result='출력 0.5배의 기본 모델은 걷기 부담 19.3%, 달리기 부담 23.7%를 줄였고 각각 40초 유지했다.', limit='가상 착용자 한 모델의 개발 기준이다. 출력 배율·토크 한계가 새 후보와 다르다.'),
    dict(file='walking_swing_gate.html', group='current', title='걷기 유각 브레이크 분석과 관절각 게이트', date='2026-10-14',
         purpose='기본 모델 L1이 걷기에서 착용자를 방해하는 구간의 원인을 찾고 관절각만으로 막기', result='방해 구간은 유각 중·후반(68–92 %) 하나다. 관절각 게이트로 걷기 부담 감소가 18.7 %에서 26.5 %, 달리기는 24.3 %에서 31.0 %가 됐다.', limit='가상 착용자 W1, 걷기·달리기 각 한 클립이다. 20–40 %는 손해 구간이 아니었다. 게이트 비율은 W1 걷기에서 정했다.'),
    dict(file='walking_fullbody_diagnostic.html', group='diagnostic', title='무릎·골반 결합과 보조 차단 진단', date='2026-10-10',
         purpose='단순 진자와 전신 걷기의 차이를 실제 물리에서 계측', result='문제 구간의 보조를 끄면 부담이 일부 줄었다. 무릎·골반 운동이 고관절 토크에 크게 기여했다.', limit='참조 보행 구간을 아는 진단이다. 그대로 배포하는 제어 방법이 아니다.'),
    dict(file='walking_diagnostic.html', group='diagnostic', title='보조가 방해하는 구간과 관측 정보', date='2026-10-09',
         purpose='토크 방향·크기와 정책이 받은 과거·미래 입력을 분석', result='전체 평균의 이득이 특정 구간의 부담 증가를 가렸다. 예측 오차나 관측창 길이 하나로 원인을 확정할 수 없었다.', limit='기존 기록과 정책 재생 분석이다. 후속 전신 계측·학습으로 이어진 과거 진단이다.'),
    dict(file='l6_l7_torque_report.html', group='diagnostic', title='진자 학습 후보의 구간별 토크 비교', date='2026-10-09까지',
         purpose='기본 모델과 진자 학습 후보의 착용자·장치 토크를 비교', result='일부 평균 부담은 줄었지만 구간별 악화가 남았다. 표시된 후보 선정은 전체 기준 통과를 뜻하지 않는다.', limit='기본 모델과 L6·L8·L9의 당시 비교다. 최신 전신 학습 모델의 순위표가 아니다.'),
    dict(file='versions.html', group='reference', title='모델·데이터 이름과 경로', date='2026-10-13',
         purpose='원래 모델 이름과 보존 경로를 찾는 표', result='비교 기준과 연구 후보를 구분하고 원본 데이터 코드를 설명한다.', limit='버전 번호가 크다고 더 좋은 모델은 아니다.'),
    dict(file='llp_v13_report.html', group='archive', title='이전 착용자에서의 기본 모델 평가', date='2026-09-28',
         purpose='현재 기준 이전의 강한 착용자 W0에서 진자 정책 평가', result='기본 모델을 보관하게 된 과거 근거다.', limit='현재 W1보다 강한 착용자의 결과다. 현재 성능표와 직접 혼합하지 않는다.'),
    dict(file='transfer_report.html', group='archive', title='초기 진자 정책의 전신 전이 시험', date='2026-09-23',
         purpose='초기 진자 정책을 이전 착용자 W0에서 시험', result='진자에서 학습한 정책을 전신에 연결한 초기 기록이다.', limit='뒤에 수행한 기본 모델 선정·W1 검증으로 갱신된 과거 자료다.'),
    dict(file='skeleton_il_hip_torque_review.html', group='archive', title='초기 토크 모방학습 데이터 검토', date='초기 연구 · 평가일 미표기',
         purpose='토크 직접 예측 연구 당시 수집 배열과 인덱스 점검', result='코드 검토로 데이터 정렬과 필드의 의미를 정리했다.', limit='종료한 토크 직접 예측 연구의 자료다. 현재 각도 예측·보조 제어 성능을 검증하지 않는다.'),
]
CSS = '''
/* Shared shell; original plot and control styles remain above this block. */
html{scroll-behavior:smooth}body{margin:0!important;padding:0!important;max-width:none!important;background:#f4f6f8!important;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif!important}
main.report-shell{display:block;max-width:1180px!important;margin:0 auto!important;padding:30px 28px 70px!important}
.report-shell h1{font-size:30px!important;line-height:1.4!important;letter-spacing:-.025em;margin:10px 0 18px!important}.report-shell h2{font-size:22px;line-height:1.45;margin:24px 0 12px}.report-shell h3{line-height:1.5}.report-shell p,.report-shell li{overflow-wrap:anywhere}.report-shell a{color:#245aa0}
.report-nav{display:flex;gap:10px 20px;flex-wrap:wrap;border-bottom:1px solid #d8e0e7;padding:0 0 15px;margin-bottom:24px;font-size:14px}.report-nav a{font-weight:600;text-decoration:none}.report-header{margin:0 0 24px;padding:0;background:transparent;border:0}.report-meta{font-size:13px;color:#5d6c7a;margin:0}.report-outcome{background:white;border:1px solid #d8e0e7;border-left:5px solid #357862;border-radius:7px;padding:16px 22px;margin:18px 0}.report-outcome p{margin:7px 0}.report-outcome strong{color:#1b5945}.report-outcome.archive{border-left-color:#8a7360}.report-outcome.archive strong{color:#6c533d}.report-limit{color:#526474;font-size:14px}.report-toc{display:flex;gap:8px 16px;flex-wrap:wrap;font-size:14px;margin:20px 0 30px}.report-toc a{background:#e9eef4;padding:5px 10px;border-radius:4px;text-decoration:none}
.report-section{margin:22px 0!important;padding:0!important;border:0;background:transparent}.report-fold{border:1px solid #d8e0e7!important;border-radius:7px;background:white!important;padding:13px 18px!important;margin:16px 0!important;color:inherit!important;font-size:inherit!important}.report-fold>summary{font-size:16px;font-weight:650;cursor:pointer;line-height:1.65}.report-fold[open]>summary{border-bottom:1px solid #e1e6eb;margin-bottom:16px;padding-bottom:10px}.report-fold .report-inner{padding:0 2px}.report-shell details>summary{cursor:pointer}.report-shell .scroll{overflow-x:auto;max-width:100%}.report-shell table{font-variant-numeric:tabular-nums}.report-shell pre{max-width:100%;overflow-x:auto;white-space:pre-wrap;overflow-wrap:anywhere}.report-shell img{max-width:100%;height:auto}.report-shell [hidden]{display:none!important}.report-shell .controls{top:0}.report-shell :target{scroll-margin-top:16px}.report-footer{margin-top:38px;border-top:1px solid #d8e0e7;padding-top:14px;font-size:13px;color:#5c6d7b}.report-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.report-card{background:white;border:1px solid #d8e0e7;border-radius:8px;padding:20px}.report-card h3{margin:4px 0 10px;font-size:19px}.report-card p{margin:8px 0}.report-card .report-meta{font-size:12px}.report-card a{text-decoration:none}.report-card .report-purpose{font-size:14px;color:#526474}.report-shell .archive-list{margin:16px 0;padding-left:24px}.report-shell .archive-list li{padding:8px 0}
@media(max-width:700px){main.report-shell{padding:20px 14px 45px!important}.report-shell h1{font-size:25px!important}.report-cards{grid-template-columns:1fr}.report-outcome{padding:14px 16px}.report-fold{padding:12px!important}.report-shell table{font-size:13px}.report-shell .controls{position:static}.report-nav{gap:8px 14px}}
@media print{body{background:white!important}main.report-shell{max-width:none!important;padding:0!important}.report-nav,.report-toc{display:none}.report-cards{display:block}.report-fold{break-inside:auto}details.report-fold:not([open])::after{content:'세부 내용은 HTML에서 펼쳐 확인할 수 있습니다.';display:block;color:#666;font-size:12px}.report-header{break-after:avoid}}
'''

@dataclass
class Node:
    tag: str
    start: int
    open_end: int
    attrs: dict = field(default_factory=dict)
    parent: object = None
    end_start: int = 0
    end: int = 0
    children: list = field(default_factory=list)


class Tree(HTMLParser):
    """Offset tree: use original slices to preserve SVG, arrays and scripts bytewise."""
    VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.offsets = [0]
        self.offsets += [m.end() for m in re.finditer('\n', source)]
        self.root = Node('root', 0, 0, end_start=len(source), end=len(source))
        self.stack = [self.root]
        self.nodes = []
        self.feed(source)
        for node in self.stack[1:]:
            node.end_start = node.end = len(source)
    def position(self):
        line, col = self.getpos()
        return self.offsets[line-1]+col
    def handle_starttag(self, tag, attrs):
        pos = self.position()
        n = Node(tag, pos, pos+len(self.get_starttag_text()), dict(attrs), self.stack[-1])
        self.stack[-1].children.append(n)
        self.nodes.append(n)
        if tag in self.VOID:
            n.end_start = n.end = n.open_end
        else:
            self.stack.append(n)
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            n = self.stack.pop(); n.end_start = n.end = n.open_end
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag == tag:
                pos = self.position()
                n = self.stack[i]
                n.end_start = pos
                n.end = self.source.index('>',pos)+1
                for missing in self.stack[i+1:]:
                    missing.end_start = missing.end = pos
                self.stack = self.stack[:i]
                break
    def raw(self,node):
        return self.source[node.start:node.end]
    def descendants(self,node,tag):
        return [n for n in self.nodes if n.tag==tag and node.start<=n.start<n.end<=node.end]


def plain(text):
    return unescape(re.sub('<[^>]*>', '', text)).strip()


def shell(title, body, styles='', prefix=''):
    return '<!doctype html>\n<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+f'<title>{escape(title)} · AI-SBC</title>'+styles+'<style>'+CSS+'</style></head><body>'+MARKER+'<main class="report-shell">'+nav(prefix)+body+'</main></body></html>\n'


def nav(prefix):
    return '<nav class="report-nav" aria-label="보고서 탐색">'+''.join(f'<a href="{prefix}{url}">{label}</a>' for url,label in [('reports.html','보고서 목록'),('experiment_summary.html','전체 실험 요약'),('versions.html','모델·데이터 경로')])+'</nav>'


def header(meta):
    state = '과거 참고 · 현재 평가 기준 아님' if meta['group']=='archive' else {'current':'최근 결과','baseline':'비교 기준','diagnostic':'원인 진단'}.get(meta['group'],'연구 자료')
    return '<header class="report-header"><p class="report-meta">'+escape(state)+' · '+escape(meta['date'])+'</p><h1>'+escape(meta['title'])+'</h1><p>'+escape(meta['purpose'])+'</p><div class="report-outcome '+('archive' if meta['group']=='archive' else '')+'"><p><strong>결론</strong> '+escape(meta['result'])+'</p><p class="report-limit">'+escape(meta['limit'])+'</p></div></header>'


def footer(prefix):
    return f'<footer class="report-footer">표·그래프·원본 수치는 해당 실험의 기록입니다. 현재 작업 순서는 <a href="{prefix}experiment_summary.html#next">전체 요약의 다음 작업</a>을 기준으로 읽습니다. 상세 항목은 제목을 눌러 펼칠 수 있습니다.</footer>'


def protected(tree):
    # Evidence and interaction contracts must survive layout changes verbatim.
    result = {}
    for tag in ('table','svg','img','script','pre','select','button','input'):
        nodes = [n for n in tree.nodes if n.tag==tag]
        result[tag] = Counter(hashlib.sha256(tree.raw(n).encode()).hexdigest() for n in nodes)
    return result


def patch_spans(source, changes):
    changes = sorted(changes, reverse=True)
    boundary = len(source)+1
    for start,end,value in changes:
        assert end<=boundary, 'overlapping edits'
        source = source[:start]+value+source[end:]
        boundary = start
    return source


def read_content(source):
    tree = Tree(source)
    styles = ''.join(tree.raw(n) for n in tree.nodes if n.tag=='style')
    node = next((n for n in tree.nodes if n.tag=='body'),None)
    if node:
        body = source[node.open_end:node.end_start]
    else:
        # Old fragment reports can put tooltips and scripts after </main>.
        # Strip head metadata only, retaining SVG titles and trailing scripts.
        head = next((n for n in tree.nodes if n.tag=='head'),None)
        if head:
            removals = [(head.start,head.end,'')]
        else:
            removals = []
            for n in tree.nodes:
                if n.tag not in ('meta','title','style','link'):
                    continue
                parent = n.parent
                in_svg = False
                while parent:
                    in_svg = in_svg or parent.tag=='svg'
                    parent = parent.parent
                if not in_svg:
                    removals.append((n.start,n.end,''))
        body = patch_spans(source,removals)
        body = re.sub(r'<!doctype[^>]*>|</?(?:html|head|body)[^>]*>','',body,flags=re.I)
    inner = Tree(body)
    main = next((n for n in inner.root.children if n.tag=='main'),None)
    if main:
        body = body[:main.start]+body[main.open_end:main.end_start]+body[main.end:]
    return body,styles


OPEN = {
 'l1_w1_report.html':['핵심 숫자'],
 'walking_finetune_report.html':['기본 속도의 결과','고관절 밖의 부담','재생을 늦춘 걷기의 문제','이 결과로 아직'],
 'walking_data_audit.html':['기존 시험은','현재 suit용'],
 'walking_retarget_report.html':['무보조 착용자 평가','기존 걷기의 참조 표현','확인한 것과'],
 'walking_fullbody_diagnostic.html':['부담과 보행 유지','계측의 신뢰'],
 'walking_diagnostic.html':[],
}


def organize(meta, source, prefix):
    # These standalone sharing pages already keep their glossary at the top and
    # figures in the body. Generic folding would hide both behind another toggle.
    if meta['file'].startswith('261015_1815/') or meta['file'] in ('261020_relative/phase_report.html', '261021_r0_generalization/report.html', '261022_llp20_motion_shared/report.html'):
        return source
    if MARKER in source:
        return source
    body, styles = read_content(source)
    tree = Tree(body)
    original = protected(Tree(source))
    if meta['file'] in ('experiment_summary.html','versions.html'):
        return shell(meta['title'],body+footer(prefix),styles,prefix)
    # Retire old action items in prose, while keeping any tables/plots/raw details.
    edits = []
    for n in tree.nodes:
        if n.tag=='h1':
            edits.append((n.start,n.end,''))
        if n.tag=='h2' and plain(tree.raw(n)) in ('다음 학습 방향','다음에 확인할 내용','다음 단계','다음 순서'):
            edits.append((n.open_end,n.end_start,'현재 작업과의 관계'))
            siblings = n.parent.children
            for nxt in siblings[siblings.index(n)+1:]:
                if nxt.tag in ('h2','details','section','table','svg'):break
                if not any(nxt.start<=x.start<x.end<=nxt.end for x in tree.nodes if x.tag in ('table','svg','img','pre','script','select','button','input')):
                    edits.append((nxt.start,nxt.end,''))
            edits.append((n.end,n.end,f'<p>이 보고서는 당시 실험의 기록이다. 현재 순서는 <a href="{prefix}experiment_summary.html#next">전체 요약의 다음 작업</a>에 모았다. 이전 추가학습 제안을 현재 실행 지시로 읽지 않는다.</p>'))
        if meta['file']=='walking_diagnostic.html' and n.tag=='p' and '다음 판단:' in plain(tree.raw(n)):
            edits.append((n.start,n.end,f'<p>이후 전신 계측과 걷기 추가학습을 수행했다. L12는 학습하지 않았다. <a href="{prefix}walking_fullbody_diagnostic.html">전신 계측</a>과 <a href="{prefix}walking_finetune_report.html">추가학습 결과</a>를 확인한다.</p>'))
    body = patch_spans(body,edits)
    # Legacy prose correction; no numeric evidence or script change.
    body = body.replace('τ²는 착용자 hip 근토크의 제곱 평균이라 다리를 덜 움직여서는 못 내린다 — 기준 지표.', 'τ²는 착용자 고관절 토크의 제곱 평균이다. 동작이 바뀌면 이 값도 달라질 수 있어 추종·속도·낙상을 함께 본다.')
    tree = Tree(body)
    if meta['file']=='l6_l7_torque_report.html':
        # Fold the old introduction, but leave the selector-driven panels together.
        intro = next((n for n in tree.root.children if n.tag=='header'),None)
        content = body
        if intro:
            content = body[:intro.start]+'<details class="report-fold"><summary>이 비교의 선정 기준과 토크 해석</summary>'+tree.raw(intro)+'</details>'+body[intro.end:]
    else:
        children = tree.root.children
        starts = []
        for n in children:
            h = n if n.tag=='h2' else next(iter(tree.descendants(n,'h2')),None) if n.tag in ('section','article') else None
            if h:starts.append((n,h))
        trailing = [n for n in children if n.tag=='script' or n.attrs.get('id')=='tip']
        end_content = min([n.start for n in trailing] or [len(body)])
        groups=[]
        for i,(node,h) in enumerate(starts):
            end = starts[i+1][0].start if i+1<len(starts) else end_content
            raw = body[node.start:end]
            title = plain(tree.raw(h))
            # Heading moves into the wrapper; an existing ID remains reachable.
            raw = raw[:h.start-node.start]+raw[h.end-node.start:]
            ident = h.attrs.get('id') or f'report-part-{i+1}'
            opened = any(title.startswith(x) for x in OPEN.get(meta['file'],[]))
            groups.append((opened,ident,title,raw))
        before = body[:starts[0][0].start] if starts else body[:end_content]
        # Remove now-redundant link to the overview inside the old introduction.
        before = re.sub(r'<p[^>]*>\s*<a href="experiment_summary.html">← 全体[^<]*</a>\s*</p>','',before)
        content=''
        if groups:
            content += '<nav class="report-toc" aria-label="보고서 내용">'+''.join(f'<a href="#{ident}">{escape(title)}</a>' for opened,ident,title,raw in groups if opened)+'</nav>'
            for opened,ident,title,raw in sorted(groups,key=lambda g:not g[0]):
                if opened:content+=f'<section class="report-section" id="{ident}"><h2>{escape(title)}</h2>{raw}</section>'
            if plain(before):content+='<details class="report-fold"><summary>평가 배경과 조건</summary><div class="report-inner">'+before+'</div></details>'
            for opened,ident,title,raw in groups:
                if not opened:content+=f'<details class="report-fold" id="{ident}"><summary>{escape(title)}</summary><div class="report-inner">'+raw+'</div></details>'
            content += body[end_content:]
        else:
            content=body
            if meta['group']=='archive':
                content='<details class="report-fold"><summary>과거 자료 원문 펼치기</summary>'+body+'</details>'
    common = ''
    if meta['file']!='skeleton_il_hip_torque_review.html':
        common='<p class="report-limit">무보조 A와 보조 B는 보조 유무의 비교 조건이다. 토크²는 시뮬레이션 부담 지표이며 실제 근활성·대사량 측정값이 아니다. RMS는 토크 크기를 나타내는 제곱 평균의 제곱근이다. 보행 진행률 0%는 참조 고관절 최대 굴곡이며 발 착지 검출값이 아니다.</p>'
    if meta['file'] in ('walking_data_audit.html','walking_retarget_report.html'):
        common+='<p class="report-limit">A443 같은 Axxx는 원본 데이터의 동작 수행자 코드다. 실험 번호나 모델 버전이 아니다. 모든 후보는 같은 suit 체형과 같은 가상 착용자로 평가했다.</p>'
    result=shell(meta['title'],header(meta)+common+content+footer(prefix),styles,prefix)
    assert protected(Tree(result))==original, f"Evidence/controls changed: {meta['file']}"
    return result


def link_for(meta):
    return 'archive/reports/'+meta['file'] if meta['group']=='archive' else meta['file']


def build_catalog():
    body='<header class="report-header"><p class="report-meta">AI-SBC · '+max(m['date'] for m in CATALOG if m['group']!='archive')+' 결과 반영</p><h1>보고서 목록</h1><p>처음에는 전체 요약을 읽고, 필요한 결과와 근거만 선택한다. 과거 참고 자료는 아래에 분리했다.</p><div class="report-outcome"><p><strong>최신 결과</strong> 20 N·m·실제 동작·단일 관절 공유로 다시 학습한 네 모델(A·Δ 2×2)은 W1에서 L1보다 덜 줄였다. 학습 때 예측 신호와 배포 H1의 차이가 원인으로 보인다(2026-10-22).</p><p><a href="261022_llp20_motion_shared/report.html">결과 그래프 읽기 →</a> · <a href="experiment_summary.html">전체 실험 요약</a></p></div></header>'
    for group,title in [('current','최근 결과'),('baseline','계속 사용할 비교 기준'),('diagnostic','원인을 확인한 진단'),('reference','모델과 데이터 찾기')]:
        body+=f'<section class="report-section"><h2>{title}</h2><div class="report-cards">'
        for m in CATALOG:
            if m['group']!=group:continue
            body+='<article class="report-card"><p class="report-meta">'+escape(m['date'])+'</p><h3><a href="'+link_for(m)+'">'+escape(m['title'])+'</a></h3><p>'+escape(m['result'])+'</p><p class="report-purpose">'+escape(m['limit'])+'</p></article>'
        body+='</div></section>'
    body+='<details class="report-fold" id="past"><summary>과거 참고 자료 · 현재 결과와 분리</summary><ul class="archive-list">'
    for m in CATALOG:
        if m['group']=='archive':body+='<li><a href="'+link_for(m)+'">'+escape(m['title'])+'</a> — '+escape(m['limit'])+'</li>'
    body+='</ul><p>수치와 그래프는 보존했다. 이전 주소에도 보관 위치 안내가 남아 있다.</p></details>'
    return shell('보고서 목록',body+footer(''))


def render_versions():
    """Rebuild versions.html from AI_SBC_VERSIONS.md so the nav never shows a stale table."""
    import markdown
    source=(DOCS/'AI_SBC_VERSIONS.md').read_text(encoding='utf-8')
    found=re.search(r'(\d{4}-\d{2}-\d{2}) 기준',source)
    meta=next(m for m in CATALOG if m['file']=='versions.html')
    if found:meta['date']=found.group(1)
    content=markdown.markdown(source,extensions=['tables','fenced_code'])
    (DOCS/'versions.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>모델·데이터 이름과 경로</title></head><body><main>'
        +'<p class="report-meta">AI_SBC_VERSIONS.md에서 생성 · 수정은 원본 Markdown에서 한다</p>'+content+'</main></body></html>\n',encoding='utf-8')


def main():
    render_versions()
    archive=DOCS/'archive/reports';archive.mkdir(parents=True,exist_ok=True)
    record=[]
    for meta in CATALOG:
        p=DOCS/meta['file']
        if meta['group']=='archive':
            target=archive/meta['file']
            # A regenerated old URL replaces the archived presentation on the next pass.
            source=target.read_text() if target.exists() and 'AI-SBC archived report pointer' in p.read_text() else p.read_text()
            output=organize(meta,source,'../../')
            target.write_text(output)
            pointer=header(meta)+f'<p>이 보고서는 과거 참고 자료로 옮겼습니다.</p><p><a href="archive/reports/{meta["file"]}">원래 수치와 그래프가 있는 보관본 열기 →</a></p>'
            p.write_text(shell(meta['title'],'<!-- AI-SBC archived report pointer -->'+pointer+footer('')))
        else:
            target=p
            p.write_text(organize(meta,p.read_text(),'../' * (len(Path(meta['file']).parts) - 1)))
        record.append(dict(meta,path=link_for(meta),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (DOCS/'reports.html').write_text(build_catalog())
    updated=max(m['date'] for m in CATALOG if m['group']!='archive')
    (DOCS/'report_catalog.json').write_text(json.dumps({'updated':updated,'reports':record},ensure_ascii=False,indent=2)+'\n')
    print('Organized',len(CATALOG),'reports; 3 historical reports archived; index: docs/ai_sbc/reports.html')


if __name__=='__main__':
    main()
