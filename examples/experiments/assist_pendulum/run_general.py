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
"""Run fresh 23 Nm ablations, confirm selected recipe, then compare frozen L1.

The manifest fixes budgets, seeds, last-epoch selection, and all source hashes.
No PA checkpoint is loaded. Repeated invocations resume only their own run.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RECIPE = "examples/experiments/assist_pendulum/mlp_general.py"
EVALUATOR = "examples/experiments/assist_pendulum/evaluate_general.py"
BASE = "results/llp_pendulum20_C0_20261014_nativeff/epoch_2000.ckpt"
VARIANTS = {
    "G0": ["--joint-actor"],
    "G1": ["--shared-joint-actor"],
    "G2": ["--shared-joint-actor", "--phase-probability", ".5", "--structured-disturbance-prob", ".5"],
    "G3": ["--shared-joint-actor", "--phase-probability", ".5", "--structured-disturbance-prob", ".5", "--human-cost-multiplier", "2"],
    "G4": ["--shared-joint-actor", "--phase-probability", ".5", "--structured-disturbance-prob", ".5", "--regularization-multiplier", "4"],
}
DESCRIPTIONS = {
    "G0": "공동 actor · 기본 외란 · 기본 보상 (구조 대조군)",
    "G1": "관절별 공유 actor · 기본 외란 · 기본 보상",
    "G2": "G1 + 운동 구간 외란·각도 하중·감쇠",
    "G3": "G2 + 사람 토크²·일률 비용 2배",
    "G4": "G2 + 보조 절약·명령 변화 비용 4배",
}
SOURCES = [RECIPE, EVALUATOR, "examples/experiments/assist_pendulum/run_general.py",
    *["examples/experiments/assist_pendulum/" + s for s in (
        "mlp.py", "mlp_v3.py", "mlp_v4.py", "mlp_research.py", "runtime_utils.py", "evaluate_research.py")],
    *["protomotions/envs/control/" + s for s in (
        "pendulum_general_control.py", "pendulum_research_control.py", "assist_disturbance_control.py",
        "assist_motion_control.py", "assist_target_control.py")],
    "protomotions/envs/base_env/assist_env.py", "protomotions/envs/base_env/paired_assist_env.py",
    "protomotions/envs/rewards/assist.py", "protomotions/agents/common/shared_joint_mlp.py",
    "protomotions/agents/ppo/agent.py", "protomotions/assist/llp_policy.py"]


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hlp-checkpoint", required=True)
    parser.add_argument("--out", default="results/pendulum_general23_20261015_all")
    parser.add_argument("--tag", default="20261015_all")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "study.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    plan = dict(epochs=args.epochs, num_envs=args.num_envs, training_seed=0, confirmation_seed=1,
        variants=VARIANTS, torque_limit_Nm=23, L1_scale=.5, L1_effective_limit_Nm=20,
        hlp_checkpoint=args.hlp_checkpoint, initialization="random; no warm start",
        screening="Fixed final epoch for all five recipes. H1 legacy/phase/hold50, emulator phase diagnostic.",
        selection="No assisted failures; lower burden/tracking/work than unassisted in each H1 domain. Rank equal-domain mean burden ratio; no L1 used for recipe selection.",
        confirmation="Selected recipe and G1, fresh seed 1. All selected/G1 seeds evaluated on held-out plant/trajectory/H1 seeds before L1 comparison.",
        transfer="Frozen W1 walking/running only if selected recipe beats L1 in burden and does not worsen tracking/work/failures across held-out H1 domains in BOTH training seeds. Different device limits are explicit.",
        phase_model="Dissipative synthetic deceleration/reversal pulses, NOT foot contact or changing inertia",
        source_sha256=hashes)
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == plan, "Study changed: use a new output directory"
    else:
        write_json(manifest, plan)
        for name in SOURCES:
            dst = out / "sources" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT/name).read_bytes())
    records_path = out / "comparison.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    active = {}

    def status(stage, **extra):
        state = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(), completed_evaluations=len(records), **extra)
        write_json(out / "status.json", state)
        rows = []
        for name, report in records.items():
            m = report["metrics"]["all"]
            rows.append(f"<tr><td>{escape(name)}</td><td>{m['assisted_tau_sq_Nm2']:.1f}</td><td>{m['assisted_tracking_rmse_deg']:.3f}</td><td>{m['assisted_abs_work_W']:.2f}</td><td>{report['failures']['assisted']}/{report['total_episodes']}</td></tr>")
        arms = ''.join(f"<li><b>{k}</b>: {v}</li>" for k,v in DESCRIPTIONS.items())
        html = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>23 Nm 진자 일반화 실험</title><style>body{font:16px/1.7 system-ui;background:#f4f6f8;color:#1c2d3d;max-width:1120px;margin:32px auto;padding:20px}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #dde3ea;text-align:right}td:first-child,th:first-child{text-align:left}code{overflow-wrap:anywhere}.status{padding:16px;background:#e5eef9;border-radius:8px}</style><body>'
        html += f'<h1>23 N·m 진자 일반화 실험</h1><p class="status">{escape(stage)}<br>{escape(state["updated_utc"])}<br>{escape(str(extra))}</p>'
        html += f'<p>무작위 초기화 · {args.num_envs:,} 환경 · 고정 마지막 {args.epochs:,} epoch · PA 계열 제외. 새 정책 ±23 N·m, L1 ×0.5 ±20 N·m.</p><ul>{arms}</ul>'
        html += '<p>미래 목표 각도 입력을 유지한다. 학습은 합성 의도 기반 잡음 예측, 주 평가는 실제 동결 H1이다. 관절별 actor는 같은 가중치를 각 관절에 독립 적용한다. 외란 위상·사람 토크는 actor에 주지 않는다.</p><p>관성은 환경별 고정이다. swing 유사 감쇠와 짧은 반전 충격은 heel strike 접촉·급변 관성의 재현이 아니다. 완전한 일반화 성공을 보장하지 않는다.</p>'
        html += '<table><tr><th>모델 / 평가</th><th>사람 τ² [N·m²]</th><th>추종 RMSE [°]</th><th>|일률| [W]</th><th>실패</th></tr>' + ''.join(rows) + '</table>'
        html += '<p>단위는 관절당 평균. 실패 전 공통 생존 구간의 수치이며 실패와 함께 판단한다. 빈 표는 평가 미완료다. 보상 크기·clip·좌우 구간 지표는 각 summary.json에 있다.</p><p><a href="manifest.json">고정 계획·소스 해시</a> · <a href="comparison.json">평가 원본 요약</a> · <a href="status.json">실행 상태</a></p></body></html>'
        (out / "report.html").write_text(html)
        print(stage, extra, flush=True)

    def run(command, label):
        assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in hashes.items()), "Source changed during study"
        log = out / (label + ".log")
        with log.open("a") as stream:
            stream.write(json.dumps(command) + "\n"); stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            active.update(child_pid=child.pid, command=command, log=str(log))
            while child.poll() is None:
                with log.open('rb') as inp:
                    inp.seek(max(0, log.stat().st_size-6000))
                    epochs = re.findall(r'Epoch (\d+)', inp.read().decode(errors='replace'))
                status(label, **active, observed_epoch=int(epochs[-1]) if epochs else None)
                time.sleep(30)
            if child.returncode:
                raise RuntimeError(f"{label}: exit {child.returncode}; see {log}")
        active.clear()

    def train(label, seed):
        name = f"llp_general23_{label}_s{seed}_{args.tag}"
        directory = ROOT / "results" / name
        ckpt = directory / f"epoch_{args.epochs}.ckpt"
        command = [sys.executable, "-u", "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
            "--simulator", "newton", "--experiment-path", RECIPE, "--experiment-name", name,
            "--motion-file", "none", "--num-envs", str(args.num_envs), "--batch-size", "16384",
            "--seed", str(seed), "--training-max-steps", str(args.epochs*args.num_envs*32),
            "--assist-limit", "23", *VARIANTS[label], "--overrides", "agent.save_epoch_checkpoint_every=500"]
        if not (directory / "resolved_configs.pt").exists():
            run(command + ["--create-config-only"], name + "_config")
        import torch
        cfg = torch.load(directory / "resolved_configs.pt", map_location="cpu", weights_only=False)
        assert cfg['env'].assist_torque_limit == cfg['env'].action_config['torque_scale'] == 23
        assert cfg['agent']._target_ == 'protomotions.agents.ppo.agent.PPO'
        assert cfg['agent'].training_max_steps == args.epochs*args.num_envs*32
        if args.preflight_only:
            return ckpt
        if not ckpt.exists():
            run(command, name + "_train")
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert saved['epoch'] == args.epochs
        return ckpt

    def evaluate(label, checkpoint, heldout=False, l1=False):
        reports = {}
        for condition, real_h1 in (("legacy",True),("phase",True),("hold50",True),("phase",False)):
            suffix = ("test" if heldout else "dev") + ("_H1_" if real_h1 else "_emulator_") + condition
            key = label + "_" + suffix
            dest = out / key
            command = [sys.executable, EVALUATOR, "--base-checkpoint", BASE, "--checkpoint", str(checkpoint),
                "--independent-limits", "--combined-human-limit", "--condition", condition,
                "--evaluation-limit", "20" if l1 else "23", "--output-scale", ".5" if l1 else "1",
                "--out", str(dest)]
            if real_h1:
                command += ["--hlp-checkpoint", args.hlp_checkpoint]
            if heldout:
                command += ["--plant-seed", "63031", "--trajectory-seed", "52021", "--hlp-seed", "991"]
            summary = dest / "summary.json"
            if not summary.exists():
                run(command, key)
            report = json.loads(summary.read_text())
            assert Path(report['protocol']['checkpoint']).resolve() == Path(checkpoint).resolve()
            assert report['assist_max_abs_Nm'] <= (20 if l1 else 23) + 1e-5
            assert report['injection_error_max_Nm'] < 1e-4
            reports[(condition,real_h1)] = report
            records[key] = report
            write_json(records_path, records)
            status("evaluated " + key)
        return reports

    try:
        if args.preflight_only:
            for label in VARIANTS:
                train(label,0)
            status("preflight configs passed; no full training started")
            return
        screen = {}
        for label in VARIANTS:
            checkpoint = train(label,0)
            screen[label] = evaluate(label+'_s0',checkpoint)
        candidates=[]
        for label,reports in screen.items():
            h1=[r for (condition,real),r in reports.items() if real]
            qualifies=all(r['failures']['assisted']==0 and all(
                r['metrics']['all']['assisted_'+metric] <= r['metrics']['all']['reference_'+metric]
                for metric in ('tau_sq_Nm2','tracking_rmse_deg','abs_work_W')) for r in h1)
            score=sum(r['metrics']['all']['assisted_tau_sq_Nm2']/r['metrics']['all']['reference_tau_sq_Nm2'] for r in h1)/len(h1)
            candidates.append(dict(label=label,qualifies=qualifies,score=score))
        eligible=[r for r in candidates if r['qualifies']]
        write_json(out/'selection.json',dict(candidates=candidates,rule=plan['selection']))
        if not eligible:
            evaluate('L1', ROOT/'checkpoints/L1_llp_pendulum/last.ckpt', heldout=True, l1=True)
            status("complete: no qualifying fresh candidate; no W1 transfer")
            return
        selected=min(eligible,key=lambda r:r['score'])['label']
        write_json(out/'selection.json',dict(candidates=candidates,selected=selected,rule=plan['selection']))
        for label in dict.fromkeys(('G1',selected)):
            train(label,1)
        test={}
        for label in dict.fromkeys(('G1',selected)):
            for seed in (0,1):
                ckpt=ROOT/'results'/f'llp_general23_{label}_s{seed}_{args.tag}'/f'epoch_{args.epochs}.ckpt'
                test[(label,seed)]=evaluate(f'{label}_s{seed}',ckpt,heldout=True)
        reference=evaluate('L1', ROOT/'checkpoints/L1_llp_pendulum/last.ckpt',heldout=True,l1=True)
        qualified=True
        for seed in (0,1):
            for domain,report in test[(selected,seed)].items():
                if not domain[1]: continue
                baseline=reference[domain]
                m,b=report['metrics']['all'],baseline['metrics']['all']
                qualified &= report['failures']['assisted']==0 and all(m['assisted_'+k] <= b['assisted_'+k] for k in ('tau_sq_Nm2','tracking_rmse_deg','abs_work_W'))
        write_json(out/'confirmation.json',dict(selected=selected,passes_L1=bool(qualified),new_limit=23,L1_limit=20))
        if qualified:
            for label,ckpt,scale in [('L1',ROOT/'checkpoints/L1_llp_pendulum/last.ckpt',.5), *[(f'{selected}_s{s}',ROOT/'results'/f'llp_general23_{selected}_s{s}_{args.tag}'/f'epoch_{args.epochs}.ckpt',1.) for s in (0,1)]]:
                for motion in (0,1):
                    name=f'W1_{label}_m{motion}'
                    if list((out/'transfer').glob(f'*_{name}/metrics.json')): continue
                    run([sys.executable,'examples/experiments/assist_skeleton/run_system.py','--llp-checkpoint',str(ckpt),'--assist-scale',str(scale),'--motion-id',str(motion),'--seconds','40','--seed','42','--tag',name,'--out',str(out/'transfer')],name)
        status("complete: confirmed comparison and transfer finished" if qualified else "complete: held-out L1 criterion failed; W1 transfer not justified")
    except BaseException as error:
        status("stopped with error",error=repr(error),**active)
        raise


if __name__ == '__main__':
    main()
