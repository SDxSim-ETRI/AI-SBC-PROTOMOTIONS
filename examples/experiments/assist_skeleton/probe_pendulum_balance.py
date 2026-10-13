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
"""Compare prescribed W1 hip motion with load-free, fixed-base pendulum balance.

This is inverse dynamics of a different plant, not a simulated counterfactual,
an estimate of W1 hidden torques, or a reproduction of randomized training loads.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ASSET = Path("protomotions/data/assets/mjcf/hip_pendulum.xml")
SCALES = (1.0, 13.0, 25.0)


def parameters():
    thigh = ET.parse(ASSET).find(".//body[@name='right_thigh']")
    inertial, joint = thigh.find("inertial"), thigh.find("joint")
    mass = float(inertial.get("mass"))
    distance = abs(float(inertial.get("pos").split()[2]))
    com_inertia = float(inertial.get("diaginertia").split()[1])
    armature = float(joint.get("armature"))
    return dict(mass_kg=mass, com_distance_m=distance, com_inertia_kg_m2=com_inertia,
                parallel_inertia_kg_m2=mass * distance**2, armature_kg_m2=armature,
                gravity_m_s2=9.81, frictionloss_omitted_Nm=float(joint.get("frictionloss")))


def balance(q, acceleration, scale, params):
    inertia = (params["parallel_inertia_kg_m2"] + scale * params["com_inertia_kg_m2"]
               + params["armature_kg_m2"])
    gravity = params["mass_kg"] * params["gravity_m_s2"] * params["com_distance_m"] * np.sin(q)
    return inertia * acceleration + gravity


def verify_against_mujoco(params):
    """Independent inverse-dynamics check of geometry, inertia and torque sign."""
    import mujoco

    rng = np.random.default_rng(42)
    worst = 0.0
    for scale in SCALES:
        xml = ET.parse(ASSET).getroot()
        pelvis = xml.find(".//body[@name='pelvis']")
        pelvis.remove(pelvis.find("freejoint"))
        for side in ("right", "left"):
            body = xml.find(f".//body[@name='{side}_thigh']")
            inertial = body.find("inertial")
            values = np.fromstring(inertial.get("diaginertia"), sep=" ") * scale
            inertial.set("diaginertia", " ".join(map(str, values)))
            body.find("joint").set("frictionloss", "0")
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        assert model.nq == model.nv == 2
        data = mujoco.MjData(model)
        for _ in range(20):
            # Physical flexion is -pendulum joint rotation; torque changes sign too.
            q = rng.uniform(-0.7, 0.7, 2)
            acceleration = rng.uniform(-40, 40, 2)
            data.qpos[:] = -q
            data.qvel[:] = rng.uniform(-3, 3, 2)
            data.qacc[:] = -acceleration
            mujoco.mj_inverse(model, data)
            error = np.max(np.abs(-data.qfrc_inverse - balance(q, acceleration, scale, params)))
            worst = max(worst, float(error))
    if worst > 1e-8:
        raise RuntimeError(f"Pendulum balance does not match inverse dynamics: {worst}")
    return dict(engine=f"MuJoCo {mujoco.__version__}", cases=60, max_torque_error_Nm=worst)


def probe(run_dir, params):
    with np.load(Path(run_dir) / "traces.npz") as tr:
        idx = np.flatnonzero((tr["t"] >= 1.30) & (np.arange(len(tr["t"])) > 0))
        dt = tr["t"][idx].astype(float) - tr["t"][idx - 1].astype(float)
        q = (tr["hip_pos"][idx].astype(float) + tr["hip_pos"][idx - 1]) / 2
        acceleration = (tr["hip_vel"][idx].astype(float) - tr["hip_vel"][idx - 1]) / dt[:, None, None]
        before, after = tr["gait_phase"][idx - 1], tr["gait_phase"][idx]
        phase = (before + 0.5 * ((after - before + 0.5) % 1 - 0.5)) % 1
        wearer = (tr["tau_pd_prev"] + tr["tau_ff_prev"])[idx]
        assist = tr["tau_assist"][idx - 1]
    rows = []
    for lo in range(0, 100, 10):
        mask = (phase >= lo / 100) & (phase < (lo + 10) / 100)
        for env, name in enumerate(("unassisted_A", "assisted_B")):
            mean = lambda x: float(x[mask].mean())
            torques = {str(int(s)): balance(q[:, env], acceleration[:, env], s, params) for s in SCALES}
            rows.append(dict(phase_pct=[lo, lo + 10], environment=name, samples=int(mask.sum()),
                             interval_acceleration_mean_rad_s2=mean(acceleration[:, env]),
                             wearer_mean_Nm=mean(wearer[:, env]),
                             wearer_rms_Nm=float(np.sqrt(np.mean(wearer[:, env][mask]**2))),
                             assist_mean_Nm=mean(assist[:, env]),
                             pendulum_mean_Nm={s: mean(t) for s, t in torques.items()},
                             pendulum_rms_Nm={s: float(np.sqrt(np.mean(t[mask]**2))) for s, t in torques.items()}))
    return dict(run_dir=str(run_dir), phase_rows=rows)


def build_section(result):
    rows = result["runs"]["L9 @2000"]["phase_rows"]
    parts = ['<section><h2>같은 hip 운동에 단순 진자는 얼마의 토크를 요구하는가</h2>',
             '<p>무보조 W1의 기록된 각도·각속도를 사용해 10 ms 구간별 가속도를 계산하고, '
             '그 궤적을 고정 골반 진자에 그대로 요구했을 때의 토크를 계산했다. '
             'τ = I·각가속도 + m·g·d·sin(각도). 질량은 고정하고 질량중심 관성 배율을 '
             '학습 범위의 1/13/25로 놓았다. 13은 범위의 중간값이며 W1에서 추정한 관성이 아니다. '
             '외란·마찰은 제외했으며 MuJoCo 역동역학 60개 상태로 식과 부호를 검증했다.</p>',
             '<div class="scroll"><table><tr><th>phase</th><th>평균 가속도 (rad/s²)</th>'
             '<th>실제 W1 평균 토크</th><th>진자 ×1</th><th>진자 ×13</th><th>진자 ×25</th></tr>']
    for row in rows:
        if row['environment'] != 'unassisted_A' or row['phase_pct'][0] not in (20, 30, 70, 80):
            continue
        values = [row['interval_acceleration_mean_rad_s2'], row['wearer_mean_Nm'],
                  *(row['pendulum_mean_Nm'][str(s)] for s in (1, 13, 25))]
        parts.append(f'<tr><td>{row["phase_pct"][0]}–{row["phase_pct"][1]}%</td>'
                     + ''.join(f'<td>{v:+.2f}</td>' for v in values) + '</tr>')
    parts.append('</table></div><p><b>70–80%:</b> 무거운 진자는 큰 제동을 요구하지만 전신 W1은 '
                 '평균 토크 부호부터 다르다. 같은 hip 운동을 보고도 필요한 국소 토크가 다를 수 있다. '
                 '<b>80–90%:</b> 이 구간 평균 가속도는 작고 이 단순 계산은 지속적인 음의 토크를 '
                 '요구하지 않는다. 정책의 지속 제동 전체를 진자 관성 한 가지로 설명할 수는 없다.</p>'
                 '<p>과거 100 ms 가속도와 위 10 ms 가속도는 서로 다른 시간창이다. '
                 '평균 토크와 RMS도 구별해야 한다. 이 계산은 다른 플랜트의 요구 토크 비교이며 '
                 'W1의 무릎·골반·지면 토크를 분해한 결과가 아니다. 학습에는 무작위 외란도 있었으므로 '
                 '진자 학습 분포 전체 또는 정책 행동을 재현한 것으로 해석하지 않는다. '
                 'L2의 질량 확대와 L3/L5의 인위적 반전 토크는 이미 실패했다. '
                 '실제 전신 상호작용을 계측하지 않고 이를 다시 조절할 근거는 없다.</p></section>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/cli_walk_diagnostic_20261009/pendulum_balance.json")
    args = parser.parse_args()
    params = parameters()
    result = dict(parameters=params, scales=SCALES, validation=verify_against_mujoco(params),
                  interpretation="Prescribed kinematics of a different, load-free plant; not W1 hidden torque decomposition.",
                  runs={})
    records = json.loads(Path("results/llp_L11_opposition_sequence/comparison.json").read_text())
    for name, key in (("L1 x0.5", "reference_L1_x0.5_m0"),
                      ("L9 @2000", "eval_llp_L9_opposition_e2000_m0")):
        result["runs"][name] = probe(records[key]["run_dir"], params)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(result["validation"]))
    print(out)


if __name__ == "__main__":
    main()
