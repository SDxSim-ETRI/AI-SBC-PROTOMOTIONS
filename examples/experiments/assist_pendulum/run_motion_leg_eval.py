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
"""LLP 20 N·m 다리별·동작 의도 연구의 평가 실행기 (2026-10-21).

가중치 공유: 오른쪽 관절 하나로 학습한 단일 관절 모델(학습 seed 0)을 양쪽 다리에 같은 가중치로
적용한다 (`LLPSharedLeg`; 사용자 결정 2026-10-21, `decision_weight_sharing.json`). 학습 실행기
(`run_motion_leg.py`)가 끝낸 조건부터 학습 순서대로 평가한다.
* 진자: 합성 궤적 legacy/phase/hold50 (2026-10-20 R0와 같은 seed; L1·R0는 저장본), SEED 미학습 동작,
  ETRI 13동작 (L1·R0도 새로 평가). 모두 동결 H1 예측.
* W1: 기존 걷기 0–40 s, 같은 걷기 400–440 s, 달리기 2.59 m/s × 평가 seed 42·7 (L1·R0는 저장본).
* 모터 응답 민감도: 기존 걷기·달리기 × seed 42·7에 출력 1차 저역통과 10 Hz를 건 W1 평가 (L1·R0 포함).
새 모델은 출력 배율 1 (20·tanh), L1·R0는 ×0.5 (40·tanh → ±20 N·m).

    python examples/experiments/assist_pendulum/run_motion_leg_eval.py
"""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
STUDY = ROOT / "results/llp20_motion_20261021"
PENDULUM_EVAL = "examples/experiments/assist_pendulum/evaluate_motion_leg.py"
RUNNER = "examples/experiments/assist_skeleton/run_system.py"
BASE = "results/llp_relative_R0_s0_20261020/epoch_6000.ckpt"  # 플랜트·관측 config의 출처 (2026-10-26: 보관 체크포인트로 교체)
CONDITION_ORDER = ("A0D0", "A1D1", "A1D0", "A0D1")
BASELINES = {
    "L1": dict(checkpoint="checkpoints/L1_llp_pendulum/last.ckpt", scale=0.5),
    "R0": dict(checkpoint="results/llp_relative_R0_s0_20261020/epoch_6000.ckpt", scale=0.5),
}
PENDULUM_SYNTHETIC = ("legacy", "phase", "hold50")
PENDULUM_MOTION = ("seed_val", "etri")
PENDULUM_SEEDS = dict(plant=63031, trajectory=52021, hlp=991)
GAIT = "data/motion_for_trackers/skeleton_torque_suit_gait.pt"
W1_CONDITIONS = {
    "walk": dict(motion_id=0, start=0.0),
    "walk_seg400": dict(motion_id=0, start=400.0),
    "run": dict(motion_id=1, start=0.0),
}
LPF_CONDITIONS = ("walk", "run")
LPF_HZ = 10.0
W1_SEEDS = (42, 7)
EVAL_SOURCES = (
    "examples/experiments/assist_pendulum/run_motion_leg_eval.py",
    PENDULUM_EVAL,
    "examples/experiments/assist_pendulum/evaluate_research.py",
    "examples/experiments/assist_pendulum/runtime_utils.py",
    "examples/experiments/assist_pendulum/prepare_motion_eval_data.py",
    RUNNER,
    "examples/experiments/assist_skeleton/fullbody_diagnostic.py",
    "examples/experiments/assist_skeleton/analysis.py",
    "protomotions/assist/controller.py",
    "protomotions/assist/llp_policy.py",
    "protomotions/tests/test_llp_shared_leg.py",
    "protomotions/assist/skeleton_features.py",
    "protomotions/agents/common/shared_joint_mlp.py",
    "protomotions/envs/base_env/paired_assist_env.py",
    "protomotions/envs/base_env/assist_env.py",
    "protomotions/envs/control/assist_motion_control.py",
    "protomotions/envs/control/assist_disturbance_control.py",
    "protomotions/envs/control/motion_intent.py",
    "protomotions/maniflow/angle_estimator.py",
    "protomotions/maniflow/chunk_source.py",
    "protomotions/maniflow/hybrid_control.py",
    "protomotions/simulator/newton/substep_hook.py",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def trained_models():
    """학습이 끝난 조건 → 단일 관절(오른쪽 관절로 학습) 체크포인트 (학습 순서)."""
    path = STUDY / "execution_right.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    return {c: records[f"{c}_right"]["checkpoint"] for c in CONDITION_ORDER if f"{c}_right" in records}


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    lock = (STUDY / "eval_worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hlp = json.loads((ROOT / "results/llp_output_scale_20261016/manifest.json").read_text())["assets"]["H1"]["path"]
    w1 = "checkpoints/W1_wearer_human/last.ckpt"
    plan = dict(
        sources={s: sha(ROOT / s) for s in EVAL_SOURCES},
        assets={name: dict(path=path, sha256=sha(ROOT / path)) for name, path in
                dict(W1=w1, H1=hlp, base=BASE, gait=GAIT, L1=BASELINES["L1"]["checkpoint"],
                     R0=BASELINES["R0"]["checkpoint"]).items()},
        pendulum=dict(synthetic=PENDULUM_SYNTHETIC, motion=PENDULUM_MOTION, seeds=PENDULUM_SEEDS,
                      num_envs=128, cohorts=2, seconds=10, limit_Nm=20,
                      reused="L1/R0 synthetic-condition summaries from results/llp_relative_20261020/pendulum "
                             "and the stored L1 equivalent; not re-run"),
        w1=dict(conditions=W1_CONDITIONS, seeds=W1_SEEDS, seconds=40,
                reused="L1 and R0 ±20 N·m records from results/llp_output_scale_20261016, "
                       "results/llp_relative_20261020/walking and results/llp_r0_generalization_20261021"),
        motor_sensitivity=dict(conditions=LPF_CONDITIONS, seeds=W1_SEEDS, lowpass_hz=LPF_HZ,
                               models="L1 ×0.5, R0 ×0.5 and every new pair"),
        scale_contract="new single-joint models 20·tanh at scale 1, one set of weights on both legs; L1/R0 40·tanh at scale 0.5",
        weight_sharing="right-lane single-joint models (training seed 0) applied to both legs; left lane stopped by user decision",
    )
    manifest = STUDY / "eval_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old == plan, "evaluation sources or plan changed; use a new study directory"
    else:
        save(manifest, plan)
        for s in EVAL_SOURCES:
            dest = STUDY / "eval_sources" / s
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / s).read_bytes())
    records_path = STUDY / "eval_runs.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    failures_path = STUDY / "eval_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONFAULTHANDLER="1")

    def status(stage, **extra):
        save(STUDY / "eval_status.json", dict(stage=stage, pid=os.getpid(), completed=len(records),
                                             updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def execute(identity, command, folder, kind):
        if identity in records:
            return records[identity]
        for s, digest in plan["sources"].items():
            assert sha(ROOT / s) == digest, f"frozen evaluation source changed: {s}"
        folder.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            log = folder / f"attempt_{attempt}.log"
            started = time.time()
            with log.open("a") as stream:
                stream.write(json.dumps(command) + "\n")
                stream.flush()
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status("evaluating", current=identity, attempt=attempt, child_pid=child.pid,
                           elapsed_s=time.time() - started)
                    if time.time() - started > 3600:
                        child.kill()
                        child.wait()
                        raise RuntimeError(f"evaluation timeout: {identity}")
                    time.sleep(5)
            elapsed = time.time() - started
            if child.returncode == 0:
                if kind == "w1":
                    outputs = [p.parent for p in folder.glob("*/metrics.json") if (p.parent / "fullbody.npz").exists()]
                    assert len(outputs) == 1, outputs
                    path = outputs[0]
                    meta = json.loads((path / "meta.json").read_text())
                    assert not meta["oracle"] and not meta["swing_gate"] and not meta["mask_window"]
                else:
                    path = folder
                    assert (folder / "summary.json").exists()
                records[identity] = dict(kind=kind, command=command, path=str(path.relative_to(ROOT)),
                                         elapsed_s=elapsed, attempt=attempt)
                save(records_path, records)
                print(f"DONE {identity} {elapsed:.0f}s", flush=True)
                return records[identity]
            failures.append(dict(id=identity, attempt=attempt, returncode=child.returncode, elapsed_s=elapsed,
                                 log=str(log.relative_to(ROOT))))
            save(failures_path, failures)
            if attempt == 3 or child.returncode not in (-11, -6):
                raise RuntimeError(f"{identity}: exit {child.returncode}; see {log}")
            print(f"RETRY {identity}: {child.returncode}", flush=True)
        raise AssertionError("unreachable")

    def pendulum(model, condition, checkpoint=None, shared=False, scale=1.0):
        identity = f"pendulum_{condition}_{model}"
        command = [PYTHON, "-u", PENDULUM_EVAL, "--base-checkpoint", BASE, "--condition", condition,
                   "--evaluation-limit", "20", "--output-scale", str(scale), "--hlp-checkpoint", hlp,
                   "--plant-seed", str(PENDULUM_SEEDS["plant"]), "--trajectory-seed", str(PENDULUM_SEEDS["trajectory"]),
                   "--hlp-seed", str(PENDULUM_SEEDS["hlp"]), "--out", str(STUDY / "pendulum" / identity)]
        if checkpoint:
            command += ["--checkpoint", str(ROOT / checkpoint)]
        if shared:
            command += ["--share-legs"]
        return execute(identity, command, STUDY / "pendulum" / identity, "pendulum")

    def walking(model, condition, seed, checkpoint, shared=False, scale=1.0, lpf=False):
        identity = f"w1_{condition}{'_lpf10' if lpf else ''}_{model}_seed{seed}"
        spec = W1_CONDITIONS[condition]
        command = [PYTHON, "-u", RUNNER, "--human", "tracker", "--tracker-checkpoint", str(ROOT / w1),
                   "--motion-file", str(ROOT / GAIT), "--llp-checkpoint", str(ROOT / checkpoint),
                   "--hlp-checkpoint", hlp, "--assist-scale", str(scale), "--motion-id", str(spec["motion_id"]),
                   "--start-time", str(spec["start"]), "--seconds", "40", "--seed", str(seed),
                   "--tag", identity, "--out", str(STUDY / "walking" / identity), "--fullbody-diagnostic"]
        if shared:
            command += ["--llp-share-legs"]
        if lpf:
            command += ["--torque-lpf-hz", str(LPF_HZ)]
        return execute(identity, command, STUDY / "walking" / identity, "w1")

    try:
        # 무보조 짝 동일성 점검 (동작 조건)
        for condition in PENDULUM_MOTION:
            pendulum("zero_action", condition)
        done = set()
        while True:
            trained = trained_models()
            for condition in CONDITION_ORDER:
                if condition in done or condition not in trained:
                    continue
                checkpoint = trained[condition]
                model = f"M_{condition}"
                for c in W1_CONDITIONS:
                    for seed in W1_SEEDS:
                        walking(model, c, seed, checkpoint, shared=True)
                for c in PENDULUM_MOTION + PENDULUM_SYNTHETIC:
                    pendulum(model, c, checkpoint, shared=True)
                for c in LPF_CONDITIONS:
                    for seed in W1_SEEDS:
                        walking(model, c, seed, checkpoint, shared=True, lpf=True)
                done.add(condition)
                if condition == CONDITION_ORDER[0]:
                    # 기준 모델: 새 조건만 (첫 새 모델 뒤에 돌려 새 결과를 먼저 낸다)
                    for name, base in BASELINES.items():
                        for c in PENDULUM_MOTION:
                            pendulum(name, c, base["checkpoint"], scale=base["scale"])
                        for c in LPF_CONDITIONS:
                            for seed in W1_SEEDS:
                                walking(name, c, seed, base["checkpoint"], scale=base["scale"], lpf=True)
            if done == set(CONDITION_ORDER):
                break
            status("waiting for training", done=sorted(done), trained=sorted(trained))
            time.sleep(120)
        status("all evaluations complete", done=sorted(done))
        print("COMPLETE", flush=True)
    except BaseException as error:
        status("stopped with error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
