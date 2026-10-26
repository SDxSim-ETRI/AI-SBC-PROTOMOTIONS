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
"""Render self-contained study explanations and the requested sharing directory."""
from pathlib import Path
import json
from html import escape
import re
import markdown

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "docs/ai_sbc/261015_1815"
STUDY = ROOT / "results/pendulum_general23_20261015_all"
MODEL_DESCRIPTIONS = {
    'L1': '기존 비교 모델',
    'G0': '양측 공동 정책',
    'G1': '관절별 공유 정책',
    'G2': '구간 외란 추가',
    'G3': '사람 부담 비용 강화',
    'G4': '보조 비용 강화',
}
CSS = """html{scroll-behavior:smooth}body{margin:0;background:#f4f6f8;color:#1c2d3d;font:16px/1.8 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:28px 26px 70px}h1{font-size:32px;line-height:1.4}h2{font-size:23px;margin:34px 0 14px}a{color:#245aa0}table{width:100%;border-collapse:collapse;background:white;font-size:14px;line-height:1.65}th{background:#e5edf5;text-align:left}td,th{padding:11px 13px;border:1px solid #dce3eb;vertical-align:top}strong{color:#185b4a}pre{background:#e8eef5;padding:20px;border-radius:8px;overflow-x:auto}code{font-size:.88em;overflow-wrap:anywhere}nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid #d8e1ea;padding-bottom:15px}details{background:white;border:1px solid #d8e1ea;border-radius:8px;padding:15px 20px;margin:20px 0}summary{cursor:pointer;font-weight:650}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:white;border:1px solid #dce3eb;border-radius:10px;padding:20px}.note{color:#62748a;font-size:14px}.plot svg{width:100%;height:auto}.status{padding:16px;background:#e5eef9;border-radius:8px}@media(max-width:750px){main{padding:20px 14px}.cards{grid-template-columns:1fr}table{display:block;overflow-x:auto}h1{font-size:26px}}"""


CSS += '.training-key{border-left:4px solid #245aa0;padding:12px 18px;background:#eaf0f7;border-radius:4px}select{max-width:100%;font:inherit;padding:5px 8px}'


def shell(title, body):
    return f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main><nav><a href="index.html">공유 자료</a><a href="FINAL_RESULTS.html">최종 판단</a><a href="training_plan.html">설계·용어 설명</a><a href="REQUESTS_AND_TODO.html">요청사항·TODO</a><a href="walking_phase_report.html">보행 phase 결과</a><a href="pendulum_results.html">진자·보상 진단</a><a href="EXECUTION_MONITOR.html">실행 감시</a></nav>{body}</main></body></html>'


def model_label(name, *, multiline=False, compact=False):
    """Keep raw result IDs stable while spelling out model and training identity."""
    model, _, seed = name.partition('_s')
    label = model if compact and seed else model+' · '+MODEL_DESCRIPTIONS[model]
    if seed:
        training = {'0': '첫 학습 (seed 0)', '1': '독립 재학습 (seed 1)'}[seed]
        label += ('\n' if multiline else ' · ') + training
    return label


def training_identity_note():
    return '<p class="training-key"><strong>모델 이름 읽기:</strong> <b>첫 학습 (seed 0, 원본 표기 s0)</b>과 <b>독립 재학습 (seed 1, 원본 표기 s1)</b>은 같은 설계를 서로 다른 난수 초기값으로 각각 처음부터 학습한 두 모델이다. seed는 가중치 초기화·표본 추출 등에 쓰는 난수열의 시작값이다. s1은 s0를 이어서 학습한 모델이 아니다. G0–G4는 모두 첫 학습을 했고 G1·G4만 독립 재학습도 했다. 좌우 다리는 그림의 오른쪽·왼쪽으로 구분한다. 보행 <b>평가</b>의 난수 초기값은 모든 모델에 42를 사용했다.</p>'


def glossary():
    source=(FOLDER/'training_plan.md').read_text()
    start=source.index('<details')
    end=source.index('</details>',start)+len('</details>')
    return markdown.markdown(source[start:end],extensions=['md_in_html'])+training_identity_note()


def main():
    comparison_path=STUDY/'comparison.json'
    comparison=json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    walking_path=STUDY/'walking/runs.json'
    walking=json.loads(walking_path.read_text()) if walking_path.exists() else {}
    todo=FOLDER/'REQUESTS_AND_TODO.md'
    source=todo.read_text()
    for label in ('G0','G1','G2','G3','G4'):
        if sum(k.startswith(label+'_s0_dev_') for k in comparison)==4:
            source=source.replace(f'- [ ] {label} 학습·진자 평가 완료',f'- [x] {label} 학습·진자 평가 완료')
    if all(f'G{i}_s0' in walking for i in range(5)):
        source=source.replace('- [ ] 각 후보의 동결 W1 걷기 평가·phase별 도움/방해 분석','- [x] 각 후보의 동결 W1 걷기 평가·phase별 도움/방해 분석')
    todo.write_text(source)
    for name,title in [('training_plan','23 N·m LLP 일반화 학습 계획'),('FINAL_RESULTS','23 N·m LLP 최종 판단'),('REQUESTS_AND_TODO','요청사항과 TODO'),('EXECUTION_MONITOR','실행 감시와 자동 복구')]:
        source=(FOLDER/(name+'.md')).read_text()
        body=markdown.markdown(source,extensions=['tables','fenced_code','toc','md_in_html'])
        body=re.sub(r'<li>\[([ x])\] ', lambda m: '<li><input type="checkbox" disabled'+(' checked' if m[1]=='x' else '')+'> ', body)
        if name != 'training_plan':
            body=body.replace('</h1>', '</h1>'+glossary(), 1)
        else:
            body=body.replace('</details>', '</details>'+training_identity_note(), 1)
        (FOLDER/(name+'.html')).write_text(shell(title,body))
    status=json.loads((STUDY/'status.json').read_text()) if (STUDY/'status.json').exists() else {}
    body='<h1>23 N·m LLP 일반화 실험</h1><p>각도 예측을 받아 보조 토크를 내는 정책을 처음부터 학습하고, 보행 주기의 어느 구간에서 사람 역할 제어기의 부담을 줄이거나 늘리는지 확인한다.</p>'+glossary()
    if status.get('stage','').startswith('complete'):
        body+='<p class="status"><strong>실험 완료 · L1을 대체할 일반 모델은 확보하지 못했다.</strong><br>7개 모델 각 3,000 epoch, 진자 평가 40건, W1 걷기 8건 완료. G4는 새 설계 중 진자 선정 점수가 가장 좋았지만 별도 평가의 부담·추종에서 L1에 미달했고 보행 방해도 남았다. <a href="FINAL_RESULTS.html">최종 판단과 남은 문제</a></p>'
    body+='<div class="cards"><article class="card"><h2><a href="walking_phase_report.html">보행 phase 결과</a></h2><p>좌우 구간별 도움·방해, 착용자 토크, 장치 보조 토크를 그래프로 확인한다. L1·G0–G4 및 G1/G4 첫 학습·독립 재학습의 완료 결과.</p></article><article class="card"><h2><a href="training_plan.html">학습 설계와 설명</a></h2><p>다섯 조건, PPO, tanh와 scale, L1과 G1의 차이, H1 평가 목적. <a href="training_plan.md">Markdown 원문</a></p></article><article class="card"><h2><a href="REQUESTS_AND_TODO.html">요청사항·TODO</a></h2><p>사용자 요구사항과 완료 여부. <a href="REQUESTS_AND_TODO.md">Markdown 원문</a></p></article><article class="card"><h2>실행 상태</h2><p>'+escape('완료 · L1 대체 기준 미달 · 걷기 8건 별도 완료' if status.get('stage','').startswith('complete') else status.get('stage','시작 전'))+'</p><p>최근 확인 epoch: '+escape('3,000 · 새 모델 7개 모두 완료' if status.get('stage','').startswith('complete') else str(status.get('observed_epoch','—')))+'</p><p class="note">'+escape(status.get('updated_utc',''))+'</p><p><a href="../../results/pendulum_general23_20261015_all/report.html">원시 실행·진자 평가 모니터</a></p></article></div>'
    body+='<p class="note">공유 설명과 결과는 이 폴더에 모은다. 학습 가중치·원본 로그·평가 배열은 results/에 보존한다. HTML 결과 그래프와 핵심 수치는 파일 안에 포함해 별도 서버 없이 읽을 수 있게 한다.</p>'
    (FOLDER/'index.html').write_text(shell('23 N·m LLP 실험 공유 자료',body))
    report=FOLDER/'walking_phase_report.html'
    if not report.exists():
        report.write_text(shell('보행 phase 결과 · 평가 준비', '<h1>보행할 때 어디서 돕고, 어디서 방해하는가</h1>'+glossary()+'<p class="status">학습·평가 진행 중이다. 아직 이번 후보의 보행 결과가 없으므로 성능 그래프나 순위를 표시하지 않는다.</p><p>완료된 평가부터 좌우 phase별 토크 RMS 감소량, 착용자 사용 토크, 장치 보조 토크를 표시한다. 각 phase는 참조 고관절 최대 굴곡을 0%로 정의한다.</p>'))
    from report_general_pendulum import main as render_pendulum
    render_pendulum()


if __name__ == '__main__':
    main()
