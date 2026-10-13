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
    python docs/organize_reports.py
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
    dict(file='experiment_summary.html', group='start', title='전체 실험 결과 요약', date='2026-10-13',
         purpose='연구의 흐름과 현재 해결할 문제', result='기존 걷기의 보조는 개선했다. 현재는 새 실제 보행의 무보조 추종을 확보하는 단계다.', limit='범용 보행이나 실기기 성공을 입증한 결과는 아니다.'),
    dict(file='walking_retarget_report.html', group='current', title='실제 보행 변환과 착용자 추종', date='2026-10-13',
         purpose='새 원본 보행을 현재 가상 착용자가 따라갈 수 있는지 확인', result='보통·느림 16조건에서 15낙상, 1골반 하강 중 종료. 같은 기존 걷기도 참조 표현을 바꾸면 낙상했다.', limit='보조를 끈 평가다. 데이터 표현·발 접촉·착용자 호환성을 먼저 해결해야 한다.'),
    dict(file='walking_finetune_report.html', group='current', title='전신 걷기 보조 학습 결과', date='2026-10-11',
         purpose='진자에서 배운 보조 모델을 전신 걷기에서 개선', result='기본 걷기의 고관절 부담은 약 69% 줄었다. 느리게 재생한 조건과 허리·무릎 부담은 해결하지 못했다.', limit='0.85–1.15배는 같은 동작의 재생 속도다. 실제 속도별 보행의 보조 효과와 구분한다.'),
    dict(file='walking_data_audit.html', group='current', title='실제 보행 자료의 범위와 속도', date='2026-10-12',
         purpose='재생 속도 변형과 실제로 다르게 걸은 자료를 구분', result='실제 보통·느림 자료를 확인했고, 빠른 후보의 정지·회전 구간을 선별할 필요가 있었다.', limit='자료 점검 보고서다. 후속 변환·무보조 평가 결과는 착용자 추종 보고서에서 확인한다.'),
    dict(file='l1_w1_report.html', group='baseline', title='기본 보조 모델의 비교 기준', date='2026-10-01',
         purpose='사람 범위 근력의 가상 착용자에서 기존 성능을 재확인', result='출력 0.5배의 기본 모델은 걷기 부담 19.3%, 달리기 부담 23.7%를 줄였고 각각 40초 유지했다.', limit='가상 착용자 한 모델의 개발 기준이다. 출력 배율·토크 한계가 새 후보와 다르다.'),
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
            edits.append((n.end,n.end,f'<p>이 보고서는 당시 실험의 기록이다. 현재 순서는 <a href="{prefix}experiment_summary.html#next">전체 실험 요약</a>에 모았다. 이전 추가학습 제안을 현재 실행 지시로 읽지 않는다.</p>'))
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
    body='<header class="report-header"><p class="report-meta">AI-SBC · 2026-10-13 정리</p><h1>보고서 목록</h1><p>처음에는 전체 요약을 읽고, 필요한 결과와 근거만 선택한다. 과거 참고 자료는 아래에 분리했다.</p><div class="report-outcome"><p><strong>지금 확인할 것</strong> 새 실제 보행을 현재 가상 착용자가 보조 없이 따라가는지 확인하는 단계다.</p><p><a href="experiment_summary.html">전체 실험의 결론과 다음 작업 읽기 →</a></p></div></header>'
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


def main():
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
            p.write_text(organize(meta,p.read_text(),''))
        record.append(dict(meta,path=link_for(meta),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (DOCS/'reports.html').write_text(build_catalog())
    (DOCS/'report_catalog.json').write_text(json.dumps({'updated':'2026-10-13','reports':record},ensure_ascii=False,indent=2)+'\n')
    print('Organized',len(CATALOG),'reports; 3 historical reports archived; index: docs/reports.html')


if __name__=='__main__':
    main()
