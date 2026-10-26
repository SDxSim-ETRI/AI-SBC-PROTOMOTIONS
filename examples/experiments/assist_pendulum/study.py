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
"""LLP 20 N·m 다리별·동작 의도 연구 실행기 — 학습 → 짝 평가 → W1 전이 → 진단 → 검증 (`MotionLegStudy`).

다리마다 한 줄(lane)로 네 조건(A·Δ 없음 → 모두 → A만 → Δ만)을 정한 순서로 학습한다. 오른쪽은 학습 seed 0,
왼쪽은 seed 1로 고정해 같은 다리 안의 조건 비교는 초기값·환경 난수가 같은 짝 비교가 된다. 모델은 마지막
epoch로 고정한다. 시뮬레이터가 SIGSEGV/SIGABRT로 죽으면 같은 실험 이름으로 다시 실행해 last.ckpt에서
이어 학습한다. 평가는 학습이 끝난 조건부터 학습 순서대로 한다 (가중치 공유: 오른쪽 관절 하나로 학습한
단일 관절 모델을 양쪽 다리에 같은 가중치로 적용; 사용자 결정 2026-10-21).
* 진자: 합성 궤적 legacy/phase/hold50 (2026-10-20 R0와 같은 seed; L1·R0는 저장본), SEED 미학습 동작,
  ETRI 13동작 (L1·R0도 새로 평가). 모두 동결 H1 예측.
* W1: 기존 걷기 0–40 s, 같은 걷기 400–440 s, 달리기 2.59 m/s × 평가 seed 42·7 (L1·R0는 저장본).
* 모터 응답 민감도: 기존 걷기·달리기 × seed 42·7에 출력 1차 저역통과 10 Hz를 건 W1 평가 (L1·R0 포함).
새 모델은 출력 배율 1 (20·tanh), L1·R0는 ×0.5 (40·tanh → ±20 N·m).

연구 폴더의 manifest는 소스 파일의 SHA-256을 고정한다. 2026-10-21 연구(`results/llp20_motion_20261021`)는
통합 전 파일명으로 고정돼 있어 닫힌 상태이며, 새 실행은 `--study results/llp20_motion_<날짜>`로 시작한다.

    python examples/experiments/assist_pendulum/study.py --study results/llp20_motion_261101 prepare
    python examples/experiments/assist_pendulum/study.py --study ... train --lane right
    python examples/experiments/assist_pendulum/study.py --study ... eval
    python examples/experiments/assist_pendulum/study.py --study ... diagnostics
    python examples/experiments/assist_pendulum/study.py --study ... verify
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_STUDY = "results/llp20_motion_20261021"
RECIPE = "examples/experiments/assist_pendulum/mlp.py"
PENDULUM_EVAL = "examples/experiments/assist_pendulum/evaluate.py"
RUNNER = "examples/experiments/assist_skeleton/run_system.py"
BASE = "results/llp_relative_R0_s0_20261020/epoch_6000.ckpt"  # 진자 평가의 플랜트·관측 config 출처
H1_CHECKPOINT = os.path.expanduser(
    "~/Projects/ManiFlow_Policy/ManiFlow/data/outputs/"
    "hip_flexion_seed-maniflow_lowdim_policy_walking-flexion100-h15-run01_seed42/checkpoints/epoch=0006-val_loss=0.000762.ckpt"
)
W1 = "checkpoints/W1_wearer_human/last.ckpt"
GAIT = "data/motion_for_trackers/skeleton_torque_suit_gait.pt"
EPOCHS = 6000
NUM_ENVS, BATCH, HORIZON = 4096, 16384, 32
# (이름, 설명, 인자) — 학습 순서 그대로
CONDITIONS = (
    ("A0D0", "A·Δ 없음", ["--assist-effort-weight=0", "--action-rate-weight=0"]),
    ("A1D1", "A·Δ 모두 (R0 값)", ["--assist-effort-weight=-2e-5", "--action-rate-weight=-0.005"]),
    ("A1D0", "A만", ["--assist-effort-weight=-2e-5", "--action-rate-weight=0"]),
    ("A0D1", "Δ만", ["--assist-effort-weight=0", "--action-rate-weight=-0.005"]),
)
CONDITION_ORDER = tuple(c[0] for c in CONDITIONS)
LANES = {"right": 0, "left": 1}  # 다리 → 학습 seed
LANES_JOINT = {"right": 0, "left": 1}
BASELINES = {
    "L1": dict(checkpoint="checkpoints/L1_llp_pendulum/last.ckpt", scale=0.5),
    "R0": dict(checkpoint="results/llp_relative_R0_s0_20261020/epoch_6000.ckpt", scale=0.5),
}
PENDULUM_SYNTHETIC = ("legacy", "phase", "hold50")
PENDULUM_MOTION = ("seed_val", "etri")
PENDULUM_SEEDS = dict(plant=63031, trajectory=52021, hlp=991)
W1_CONDITIONS = {
    "walk": dict(motion_id=0, start=0.0),
    "walk_seg400": dict(motion_id=0, start=400.0),
    "run": dict(motion_id=1, start=0.0),
}
LPF_CONDITIONS = ("walk", "run")
LPF_HZ = 10.0
W1_SEEDS = (42, 7)
TRAINING_SOURCES = (
    RECIPE,
    "examples/experiments/assist_pendulum/study.py",
    "protomotions/train_agent.py",
    "protomotions/envs/base_env/assist_env.py",
    "protomotions/envs/control/assist_target_control.py",
    "protomotions/envs/control/assist_motion_control.py",
    "protomotions/envs/control/assist_disturbance_control.py",
    "protomotions/envs/control/motion_intent.py",
    "protomotions/envs/rewards/assist.py",
    "protomotions/envs/terminations/assist.py",
    "protomotions/envs/obs/assist_obs.py",
    "protomotions/envs/context_views.py",
    "protomotions/agents/common/shared_joint_mlp.py",
    "protomotions/agents/common/mlp.py",
    "protomotions/agents/ppo/agent.py",
    "protomotions/agents/ppo/model.py",
    "protomotions/agents/base_agent/agent.py",
    "protomotions/robot_configs/hip_pendulum.py",
    "protomotions/tests/test_llp_motion_leg.py",
    "protomotions/tests/test_pendulum_recipes.py",
)
EVAL_SOURCES = (
    "examples/experiments/assist_pendulum/study.py",
    PENDULUM_EVAL,
    "examples/experiments/assist_pendulum/runtime.py",
    RUNNER,
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


def sha_or_missing(path):
    return sha(path) if Path(path).exists() else None


def read(path):
    return json.loads(Path(path).read_text())


class MotionLegStudy:
    """연구 폴더 하나(`results/llp20_motion_<날짜>`)의 학습·평가·진단·검증."""

    def __init__(self, study_dir: str = DEFAULT_STUDY):
        self.study = (ROOT / study_dir).resolve()
        suffix = self.study.name.rsplit("_", 1)[-1]
        self.date = suffix if re.fullmatch(r"\d{8}", suffix) else datetime.now().strftime("%Y%m%d")
        self.environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONFAULTHANDLER="1")

    # ------------------------------------------------------------------
    def run_name(self, condition, leg):
        return f"llp20_motion_{condition}_{leg}_s{LANES[leg]}_{self.date}"

    def train_command(self, condition, leg, extra):
        return [PYTHON, "-u", "protomotions/train_agent.py", "--robot-name", "hip_pendulum",
                "--simulator", "newton", "--experiment-path", RECIPE, "--recipe", "motion_leg",
                "--experiment-name", self.run_name(condition, leg), "--motion-file", "none",
                "--num-envs", str(NUM_ENVS), "--batch-size", str(BATCH), "--seed", str(LANES[leg]),
                "--training-max-steps", str(EPOCHS * NUM_ENVS * HORIZON), "--leg", leg, *extra]

    def status(self, name, stage, **extra):
        save(self.study / f"{name}.json", dict(stage=stage, pid=os.getpid(),
                                               updated_utc=datetime.now(timezone.utc).isoformat(), **extra))

    def _run_with_retry(self, command, log_for_attempt, poll_s, timeout_s=None, on_poll=None, failures=None, label=""):
        """자식 프로세스를 최대 3회 돌린다 (SIGSEGV/SIGABRT만 재시도). (outcome dict) 반환, 실패면 RuntimeError."""
        for attempt in range(1, 4):
            log = log_for_attempt(attempt)
            log.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log.open("a") as stream:
                stream.write(json.dumps(command) + "\n")
                stream.flush()
                child = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if on_poll is not None:
                        on_poll(child, log, attempt, started)
                    if timeout_s is not None and time.time() - started > timeout_s:
                        child.kill()
                        child.wait()
                        raise RuntimeError(f"timeout: {label}")
                    time.sleep(poll_s)
            outcome = dict(command=command, returncode=child.returncode, attempt=attempt, started_unix=started,
                           elapsed_s=time.time() - started, log=str(log.relative_to(ROOT)))
            if child.returncode == 0:
                return outcome
            if failures is not None:
                failures.append(dict(label=label, **outcome))
            if child.returncode not in (-11, -6) or attempt == 3:
                raise RuntimeError(f"{label} failed: exit {child.returncode}; see {log}")
            print(f"RETRY {label}: {child.returncode}", flush=True)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def prepare(self):
        self.study.mkdir(parents=True, exist_ok=True)
        manifest = self.study / "manifest.json"
        sources = {name: sha(ROOT / name) for name in TRAINING_SOURCES}
        if manifest.exists():
            old = read(manifest)
            assert old["training_sources"] == sources, "training sources changed after prepare; use a new study directory"
            print("manifest exists, unchanged")
            return
        sys.path.insert(0, str(HERE))
        import mlp as recipe

        plan = dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            purpose=("LLP 20 N·m (20·tanh) pendulum training with per-leg models and natural-motion "
                     "intents covering running-level hip speeds; 2×2 comparison of the assist-torque "
                     "squared cost A and the command-change cost Δ"),
            requested_order=list(CONDITION_ORDER),
            conditions={c[0]: dict(label=c[1], arguments=c[2]) for c in CONDITIONS},
            lanes={leg: dict(seed=seed, runs=[self.run_name(c[0], leg) for c in CONDITIONS])
                   for leg, seed in LANES.items()},
            epochs=EPOCHS, num_envs=NUM_ENVS, batch_size=BATCH,
            transitions_per_model=EPOCHS * NUM_ENVS * HORIZON,
            recipe=dict(
                base="R0 (mlp.py --recipe R0): corrected torque path, v4 plant and disturbance randomization",
                preset="mlp.py --recipe motion_leg",
                action="20*tanh(u), assist limit 20 N·m, deployment scale 1",
                per_leg="separate run per leg; only that joint assisted; rewards/terminations count only that joint; actor reads only that joint",
                intent_pools=[list(p) for p in recipe.INTENT_POOLS],
                motion_groups=[list(g) for g in recipe.MOTION_GROUPS],
                motion_accel_frac=recipe.MOTION_ACCEL_FRAC,
                motion_data=dict(zarr=recipe.SEED_ZARR, metadata=recipe.SEED_METADATA,
                                 split="train (H1 training split; H1 validation clips stay unseen)"),
            ),
            selection="Fixed final epoch for every run; no checkpoint chosen by evaluation outcomes",
            training_sources=sources,
        )
        save(manifest, plan)
        for name in TRAINING_SOURCES:
            dest = self.study / "sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        save(self.study / "initial_git_status.json",
             subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines())
        (self.study / "initial_nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
        print("prepared", manifest)

    def train_lane(self, leg):
        import torch

        manifest = read(self.study / "manifest.json")
        sources = manifest["training_sources"]
        lock = (self.study / f"worker_{leg}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records_path = self.study / f"execution_{leg}.json"
        records = read(records_path) if records_path.exists() else {}
        errors_path = self.study / f"runtime_failures_{leg}.json"
        errors = read(errors_path) if errors_path.exists() else []

        def status(stage, **extra):
            self.status(f"status_{leg}", stage, lane=leg, completed=sorted(records), **extra)

        for condition, _, extra in CONDITIONS:
            label = f"{condition}_{leg}"
            name = self.run_name(condition, leg)
            checkpoint = ROOT / "results" / name / f"epoch_{EPOCHS}.ckpt"
            if label in records:
                continue
            for path, digest in sources.items():
                assert sha(ROOT / path) == digest, f"frozen training source changed: {path}"

            def on_poll(child, log, attempt, started):
                with log.open("rb") as inp:
                    inp.seek(max(0, log.stat().st_size - 4000))
                    epochs = re.findall(r"Epoch (\d+)", inp.read().decode(errors="replace"))
                status("training", current=label, attempt=attempt, child_pid=child.pid,
                       job_elapsed_s=time.time() - started, observed_epoch=int(epochs[-1]) if epochs else None)

            try:
                outcome = self._run_with_retry(
                    self.train_command(condition, leg, extra),
                    lambda attempt: self.study / "logs" / f"{label}_attempt{attempt}.log",
                    poll_s=15, on_poll=on_poll, failures=errors, label=label)
            finally:
                save(errors_path, errors)
            if outcome["returncode"] != 0:
                status("stopped with error", current=label, returncode=outcome["returncode"])
                raise RuntimeError(label)
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert ckpt["epoch"] == EPOCHS, ckpt["epoch"]
            assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in ckpt["model"].values())
            env_cfg = torch.load(checkpoint.parent / "resolved_configs.pt", map_location="cpu", weights_only=False)["env"]
            joint = LANES_JOINT[leg]
            assert tuple(env_cfg.assist_active_joints) == (joint,)
            assert float(env_cfg.assist_torque_limit) == 20.0
            assert float(env_cfg.action_config["torque_scale"]) == 20.0
            assert env_cfg.clamp_combined_injection is False and env_cfg.human_feedforward_inside_pd is True
            weights = {k: float(env_cfg.reward_components[k].static_params["weight"]) for k in ("assist_effort", "action_rate")}
            expected = {"assist_effort": -2e-5 if condition[1] == "1" else 0.0,
                        "action_rate": -0.005 if condition[3] == "1" else 0.0}
            assert weights == expected, (weights, expected)
            records[label] = dict(**outcome, run=name, checkpoint=str(checkpoint.relative_to(ROOT)),
                                  checkpoint_sha256=sha(checkpoint), reward_weights=weights)
            save(records_path, records)
            print(f"DONE {label} {outcome['elapsed_s']:.0f}s", flush=True)
        status("lane complete")
        print("LANE COMPLETE", leg, flush=True)

    def trained_models(self):
        """학습이 끝난 조건 → 단일 관절(오른쪽 관절로 학습) 체크포인트 (학습 순서)."""
        path = self.study / "execution_right.json"
        records = read(path) if path.exists() else {}
        return {c: records[f"{c}_right"]["checkpoint"] for c in CONDITION_ORDER if f"{c}_right" in records}

    # ------------------------------------------------------------------
    # 평가 (진자 짝 평가 + W1 전이)
    # ------------------------------------------------------------------
    def evaluate(self):
        self.study.mkdir(parents=True, exist_ok=True)
        lock = (self.study / "eval_worker.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hlp = H1_CHECKPOINT
        plan = dict(
            sources={s: sha(ROOT / s) for s in EVAL_SOURCES},
            assets={name: dict(path=path, sha256=sha(ROOT / path if not Path(path).is_absolute() else path)) for name, path in
                    dict(W1=W1, H1=hlp, base=BASE, gait=GAIT, L1=BASELINES["L1"]["checkpoint"],
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
        manifest = self.study / "eval_manifest.json"
        if manifest.exists():
            assert read(manifest) == plan, "evaluation sources or plan changed; use a new study directory"
        else:
            save(manifest, plan)
            for s in EVAL_SOURCES:
                dest = self.study / "eval_sources" / s
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT / s).read_bytes())
        records_path = self.study / "eval_runs.json"
        records = read(records_path) if records_path.exists() else {}
        failures_path = self.study / "eval_failures.json"
        failures = read(failures_path) if failures_path.exists() else []

        def status(stage, **extra):
            self.status("eval_status", stage, completed=len(records), **extra)

        def execute(identity, command, folder, kind):
            if identity in records:
                return records[identity]
            for s, digest in plan["sources"].items():
                assert sha(ROOT / s) == digest, f"frozen evaluation source changed: {s}"
            folder.mkdir(parents=True, exist_ok=True)

            def on_poll(child, log, attempt, started):
                status("evaluating", current=identity, attempt=attempt, child_pid=child.pid, elapsed_s=time.time() - started)

            try:
                outcome = self._run_with_retry(command, lambda attempt: folder / f"attempt_{attempt}.log",
                                               poll_s=5, timeout_s=3600, on_poll=on_poll, failures=failures, label=identity)
            finally:
                save(failures_path, failures)
            if kind == "w1":
                outputs = [p.parent for p in folder.glob("*/metrics.json") if (p.parent / "fullbody.npz").exists()]
                assert len(outputs) == 1, outputs
                path = outputs[0]
                meta = read(path / "meta.json")
                assert not meta["oracle"] and not meta["swing_gate"] and not meta["mask_window"]
            else:
                path = folder
                assert (folder / "summary.json").exists()
            records[identity] = dict(kind=kind, command=command, path=str(path.relative_to(ROOT)),
                                     elapsed_s=outcome["elapsed_s"], attempt=outcome["attempt"])
            save(records_path, records)
            print(f"DONE {identity} {outcome['elapsed_s']:.0f}s", flush=True)
            return records[identity]

        def pendulum(model, condition, checkpoint=None, shared=False, scale=1.0):
            identity = f"pendulum_{condition}_{model}"
            command = [PYTHON, "-u", PENDULUM_EVAL, "paired", "--base-checkpoint", BASE, "--condition", condition,
                       "--evaluation-limit", "20", "--output-scale", str(scale), "--hlp-checkpoint", hlp,
                       "--plant-seed", str(PENDULUM_SEEDS["plant"]), "--trajectory-seed", str(PENDULUM_SEEDS["trajectory"]),
                       "--hlp-seed", str(PENDULUM_SEEDS["hlp"]), "--etri-dir", str(self.study / "eval_data"),
                       "--out", str(self.study / "pendulum" / identity)]
            if checkpoint:
                command += ["--checkpoint", str(ROOT / checkpoint)]
            if shared:
                command += ["--share-legs"]
            return execute(identity, command, self.study / "pendulum" / identity, "pendulum")

        def walking(model, condition, seed, checkpoint, shared=False, scale=1.0, lpf=False):
            identity = f"w1_{condition}{'_lpf10' if lpf else ''}_{model}_seed{seed}"
            spec = W1_CONDITIONS[condition]
            command = [PYTHON, "-u", RUNNER, "--human", "tracker", "--tracker-checkpoint", str(ROOT / W1),
                       "--motion-file", str(ROOT / GAIT), "--llp-checkpoint", str(ROOT / checkpoint),
                       "--hlp-checkpoint", hlp, "--assist-scale", str(scale), "--motion-id", str(spec["motion_id"]),
                       "--start-time", str(spec["start"]), "--seconds", "40", "--seed", str(seed),
                       "--tag", identity, "--out", str(self.study / "walking" / identity), "--fullbody-diagnostic"]
            if shared:
                command += ["--llp-share-legs"]
            if lpf:
                command += ["--torque-lpf-hz", str(LPF_HZ)]
            return execute(identity, command, self.study / "walking" / identity, "w1")

        try:
            if not (self.study / "eval_data" / "etri_suit13_metadata.parquet").exists():
                subprocess.run([PYTHON, PENDULUM_EVAL, "prepare-etri", "--out", str(self.study / "eval_data")],
                               cwd=ROOT, env=self.environment, check=True)
            # 무보조 짝 동일성 점검 (동작 조건)
            for condition in PENDULUM_MOTION:
                pendulum("zero_action", condition)
            done = set()
            while True:
                trained = self.trained_models()
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

    # ------------------------------------------------------------------
    def diagnostics(self):
        """네 조건 학습이 끝나면 새 모델 4개와 L1·R0를 학습 분포 진자에서 에뮬레이션 / 실제 H1 예측으로 평가."""
        records = self.study / "execution_right.json"
        while True:
            done = read(records) if records.exists() else {}
            if all(f"{c}_right" in done for c in CONDITION_ORDER):
                break
            time.sleep(120)
        config = done["A0D0_right"]["checkpoint"]
        models = {c: done[f"{c}_right"]["checkpoint"] for c in CONDITION_ORDER}
        models.update(L1=BASELINES["L1"]["checkpoint"], R0=BASELINES["R0"]["checkpoint"])
        for name, checkpoint in models.items():
            for source in ("emulated", "h1"):
                out = self.study / "diagnostics" / f"chunk_{name}_{source}.json"
                if out.exists():
                    continue
                command = [PYTHON, "-u", PENDULUM_EVAL, "chunk", "--config-checkpoint", config, "--checkpoint", checkpoint,
                           "--out", str(out)] + (["--hlp-checkpoint", H1_CHECKPOINT] if source == "h1" else [])
                self._run_with_retry(command, lambda attempt, n=name, s=source: self.study / "diagnostics" / f"chunk_{n}_{s}_attempt{attempt}.log",
                                     poll_s=5, label=f"{name} {source}")
                print("DONE", name, source, flush=True)
        print("COMPLETE", flush=True)

    # ------------------------------------------------------------------
    def verify(self, strict=False):
        """기록 무결성 점검 → verification.json. 통합 전 연구는 소스 파일이 보관돼 missing으로 표시된다."""
        import torch

        study = self.study
        manifest = read(study / "manifest.json")
        eval_manifest = read(study / "eval_manifest.json")
        execution = read(study / "execution_right.json")
        runs = read(study / "eval_runs.json")
        checks = dict(training_sources_unchanged={k: sha_or_missing(ROOT / k) == v for k, v in manifest["training_sources"].items()},
                      evaluation_sources_unchanged={k: sha_or_missing(ROOT / k) == v for k, v in eval_manifest["sources"].items()},
                      evaluation_assets_unchanged={k: sha_or_missing(ROOT / v["path"]) == v["sha256"] for k, v in eval_manifest["assets"].items()})
        training = {}
        for c in CONDITION_ORDER:
            rec = execution[f"{c}_right"]
            ckpt = torch.load(ROOT / rec["checkpoint"], map_location="cpu", weights_only=False)
            env = torch.load(ROOT / rec["checkpoint"].replace(f"epoch_{EPOCHS}.ckpt", "resolved_configs.pt"),
                             map_location="cpu", weights_only=False)["env"]
            training[c] = dict(checkpoint=rec["checkpoint"], epoch=ckpt["epoch"], sha256_matches_record=sha(ROOT / rec["checkpoint"]) == rec["checkpoint_sha256"],
                               all_parameters_finite=all(not torch.is_tensor(v) or bool(torch.isfinite(v).all()) for v in ckpt["model"].values()),
                               active_joints=list(env.assist_active_joints), limit=float(env.assist_torque_limit),
                               reward_weights={k: float(env.reward_components[k].static_params["weight"]) for k in ("assist_effort", "action_rate")},
                               elapsed_h=rec["elapsed_s"] / 3600, attempts=rec["attempt"])
        walking, pendulum = {}, {}
        for identity, r in runs.items():
            path = ROOT / r["path"]
            if r["kind"] == "w1":
                meta = read(path / "meta.json")
                walking[identity] = dict(fall=meta["survived_s"], share_legs=meta.get("llp_share_legs"), assist_scale=meta["assist_scale"],
                                         traces=(path / "traces.npz").exists(), fullbody=(path / "fullbody.npz").exists(),
                                         lowpass_hz=meta.get("torque_lpf_hz"))
            else:
                s = read(path / "summary.json")
                pendulum[identity] = dict(failures=s["failures"], rollout=(path / "rollout.npz").exists(),
                                          zero_action_max_q_difference_rad=s.get("zero_action_max_q_difference_rad"),
                                          injection_error_max_Nm=s["injection_error_max_Nm"], assist_max_abs_Nm=s["assist_max_abs_Nm"])
        new_w1 = {k: v for k, v in walking.items() if "_M_" in k}
        checks.update(
            training=training,
            all_training_epoch_final=all(t["epoch"] == EPOCHS and t["all_parameters_finite"] and t["sha256_matches_record"] for t in training.values()),
            evaluation_records=len(runs), w1_runs=len(walking), pendulum_runs=len(pendulum),
            w1_falls=sum(v["fall"] is not None for v in walking.values()),
            new_models_shared_on_both_legs=all(v["share_legs"] for v in new_w1.values()),
            w1_outputs_complete=all(v["traces"] and v["fullbody"] for v in walking.values()),
            pendulum_outputs_complete=all(v["rollout"] for v in pendulum.values()),
            pendulum_assist_within_20Nm=all(v["assist_max_abs_Nm"] <= 20.0001 for v in pendulum.values()),
            zero_action_identity={k: v["zero_action_max_q_difference_rad"] for k, v in pendulum.items() if "zero_action" in k},
            diagnostics=sorted(p.name for p in (study / "diagnostics").glob("chunk_*.json")),
            runtime_failures=dict(training=read(study / "runtime_failures_right.json") if (study / "runtime_failures_right.json").exists() else [],
                                  evaluation=read(study / "eval_failures.json") if (study / "eval_failures.json").exists() else []),
            walking=walking, pendulum=pendulum)
        checks["sources_unchanged"] = (all(checks["training_sources_unchanged"].values()) and all(checks["evaluation_sources_unchanged"].values())
                                       and all(checks["evaluation_assets_unchanged"].values()))
        checks["records_consistent"] = (checks["all_training_epoch_final"] and checks["w1_falls"] == 0 and checks["new_models_shared_on_both_legs"]
                                        and checks["w1_outputs_complete"] and checks["pendulum_outputs_complete"]
                                        and checks["pendulum_assist_within_20Nm"] and len(checks["diagnostics"]) == 12
                                        and checks["evaluation_records"] == 74)
        checks["passed"] = checks["sources_unchanged"] and checks["records_consistent"]
        (study / "verification.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
        print(json.dumps({k: checks[k] for k in ("passed", "sources_unchanged", "records_consistent", "all_training_epoch_final",
                                                  "evaluation_records", "w1_runs", "pendulum_runs", "w1_falls",
                                                  "new_models_shared_on_both_legs", "zero_action_identity")}, indent=1))
        if strict:
            assert checks["passed"]
        return checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", default=DEFAULT_STUDY, help="연구 폴더 (results/llp20_motion_<날짜>)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="manifest·소스 사본 고정")
    t = sub.add_parser("train", help="한 다리(lane)의 네 조건을 순서대로 학습")
    t.add_argument("--lane", choices=sorted(LANES), required=True)
    sub.add_parser("eval", help="학습이 끝난 조건부터 진자 짝 평가·W1 전이·저역통과 민감도")
    sub.add_parser("diagnostics", help="에뮬레이션 vs 실제 H1 예측 진단")
    v = sub.add_parser("verify", help="기록 무결성 점검 → verification.json")
    v.add_argument("--strict", action="store_true", help="통과하지 못하면 실패로 종료")
    args = p.parse_args(argv)
    study = MotionLegStudy(args.study)
    if args.command == "prepare":
        study.prepare()
    elif args.command == "train":
        study.train_lane(args.lane)
    elif args.command == "eval":
        study.evaluate()
    elif args.command == "diagnostics":
        study.diagnostics()
    else:
        study.verify(strict=args.strict)


if __name__ == "__main__":
    main()
