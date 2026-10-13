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
"""Validate and decompose recorded Newton hip dynamics, without changing motion."""

import argparse
import io
import json
from pathlib import Path

import mujoco
import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def analyze(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'fullbody_meta.json').read_text())
    run_meta = json.loads((run_dir / 'meta.json').read_text())
    with np.load(run_dir / 'fullbody.npz') as source:
        data = {k: source[k].astype(np.float64) for k in source.files}
    with np.load(run_dir / 'traces.npz') as source:
        traces = {k: source[k] for k in source.files}
    hips = np.array(meta['hip_mj_dofs'])
    common_hips = [meta['dof_names_common'].index('hip_flexion_' + side) for side in ('r', 'l')]
    common_knees = [meta['dof_names_common'].index('knee_angle_' + side) for side in ('r', 'l')]
    knees = np.array(meta['common_dof_to_mj_dof'])[common_knees]
    every = meta['substeps_per_window']
    count = len(traces['t'])
    valid_control = traces['t'] >= max(run_meta['warmup'], 1.3)
    valid_physics = np.repeat(valid_control, every)
    physics = lambda key: data['physics_' + key]
    control = lambda key: data['control_' + key]
    mass = physics('hip_mass_rows')
    acc = physics('qacc')
    inertia_terms = mass * acc[:, :, None, :]
    total_inertia = inertia_terms.sum(axis=-1)
    hip_field = lambda key: physics(key)[:, :, hips]
    actuator, applied = hip_field('qfrc_actuator'), hip_field('qfrc_applied')
    bias, passive, raw_constraint = (hip_field(k) for k in ('qfrc_bias', 'qfrc_passive', 'qfrc_constraint'))
    contact_constraint = physics('hip_constraint_contact')
    other_constraint = physics('hip_constraint_other')
    # MJWarp's sparse solver zeroes qfrc_constraint for all worlds on each
    # iteration, then skips worlds already marked done. That scratch array may
    # be zero for a converged world. Reconstruct from FINAL active J and force.
    constraint = contact_constraint + other_constraint
    constraint_error = raw_constraint - constraint
    balance_error = total_inertia + bias - passive - constraint - actuator - applied
    mean_actuator = physics('qfrc_actuator').reshape(count, every, 2, -1).mean(axis=1)
    aligned_actuator = mean_actuator[:-1, :, meta['common_dof_to_mj_dof']]
    pd_readback_error = aligned_actuator - control('tau_pd_prev')[1:]
    applied_mean = applied.reshape(count, every, 2, 2).mean(axis=1)
    applied_error = applied_mean - traces['tau_assist']
    kin_error = control('dof_pos')[:, :, common_hips] - traces['hip_pos']
    validation = dict(
        finite=all(np.isfinite(v).all() for v in data.values()),
        force_balance_rms_Nm=rms(balance_error[valid_physics]),
        force_balance_max_Nm=float(np.abs(balance_error[valid_physics]).max()),
        force_balance_p99_9_Nm=float(np.percentile(np.abs(balance_error[valid_physics]), 99.9)),
        force_balance_relative_rms=rms(balance_error[valid_physics]) / rms(actuator[valid_physics]),
        raw_constraint_scratch_max_disagreement_Nm=float(np.abs(constraint_error[valid_physics]).max()),
        pd_window_alignment_max_Nm=float(np.abs(pd_readback_error).max()),
        injected_torque_alignment_max_Nm=float(np.abs(applied_error).max()),
        hip_position_alignment_max_rad=float(np.abs(kin_error).max()),
    )
    model = mujoco.MjModel.from_binary_path(str(run_dir / 'diagnostic_model.mjb'))
    cpu = mujoco.MjData(model)
    full_mass = np.zeros((model.nv, model.nv))
    independent_errors = []
    samples = np.linspace(np.flatnonzero(valid_physics)[0], len(acc)-1, 60).astype(int)
    for i in samples:
        for env in range(2):
            cpu.qpos[:] = physics('qpos_before')[i, env]
            cpu.qvel[:] = physics('qvel_before')[i, env]
            mujoco.mj_fwdPosition(model, cpu)
            mujoco.mj_fwdVelocity(model, cpu)
            mujoco.mj_fullM(model, full_mass, cpu.qM)
            independent_errors.append((np.abs(full_mass[hips]-mass[i, env]).max(),
                                       np.abs(cpu.qfrc_bias[hips]-bias[i, env]).max()))
    independent_errors = np.max(independent_errors, axis=0)
    validation['cpu_mass_rows_max_error_kg_m2'] = float(independent_errors[0])
    validation['cpu_bias_max_error_Nm'] = float(independent_errors[1])
    validation['constraint_source'] = 'Final active constraint Jacobian transpose times force; raw scratch retained.'
    # Iterative contact solver residuals remain explicit in the decomposition.
    # Require <1 Nm and <2% actuator RMS, not fictitious exact force balance.
    if (not validation['finite'] or validation['force_balance_rms_Nm'] > 1
            or validation['force_balance_relative_rms'] > .02
            or validation['cpu_mass_rows_max_error_kg_m2'] > .005
            or validation['cpu_bias_max_error_Nm'] > .02
            or validation['pd_window_alignment_max_Nm'] > .005
            or validation['injected_torque_alignment_max_Nm'] > .005
            or validation['hip_position_alignment_max_rad'] > 1e-6):
        (run_dir / 'fullbody_validation_failed.json').write_text(json.dumps(validation, indent=2))
        raise RuntimeError(f'Diagnostic consistency checks failed: {validation}')

    # Only gravity is recomputed on CPU; M, total bias and all applied/constraint
    # terms are direct GPU solver readbacks. Zero velocity removes Coriolis terms.
    gravity = np.zeros_like(actuator)
    gravity_all = np.zeros(model.nv)
    for i in np.flatnonzero(valid_physics):
        for env in range(2):
            cpu.qpos[:] = physics('qpos_before')[i, env]
            cpu.qvel[:] = 0
            mujoco.mj_fwdPosition(model, cpu)
            mujoco.mj_fwdVelocity(model, cpu)
            mujoco.mj_rne(model, cpu, 0, gravity_all)
            gravity[i, env] = gravity_all[hips]
    diagonal = np.stack([inertia_terms[:, :, h, d] for h, d in enumerate(hips)], axis=-1)
    knee = np.stack([inertia_terms[:, :, h, d] for h, d in enumerate(knees)], axis=-1)
    root = inertia_terms[:, :, :, meta['root_mj_dofs']].sum(axis=-1)
    other = total_inertia - diagonal - knee - root
    terms = dict(hip_inertia=diagonal, ipsilateral_knee_coupling=knee, pelvis_coupling=root,
                 other_joint_coupling=other, gravity=gravity, velocity_bias=bias-gravity,
                 minus_passive=-passive, minus_contact_constraint=-contact_constraint,
                 minus_other_constraint=-other_constraint, minus_assist=-applied,
                 minus_solver_residual=-balance_error)
    np.testing.assert_allclose(sum(terms.values()), actuator, atol=1e-8)

    # Same midpoint phase convention as the original torque report; a solver
    # substep inherits its 10 ms window bin, so terms retain an additive average.
    next_phase = np.concatenate([traces['gait_phase'][1:], traces['gait_phase'][-1:]], axis=0)
    phase = (traces['gait_phase'] + .5 * ((next_phase - traces['gait_phase'] + .5) % 1 - .5)) % 1
    phase = np.repeat(phase, every, axis=0)
    foot_force = physics('contact_force_world')
    side_force = []
    for side in ('r', 'l'):
        ids = [j for j, name in enumerate(meta['contact_body_names']) if name.endswith('_' + side)]
        side_force.append(foot_force[:, :, ids].sum(axis=2))
    side_force = np.stack(side_force, axis=2)
    side_force_norm = np.linalg.norm(side_force, axis=-1)
    knee_q = np.repeat(control('dof_pos')[:, :, common_knees], every, axis=0)
    knee_v = np.repeat(control('dof_vel')[:, :, common_knees], every, axis=0)
    target = np.repeat(control('pd_target')[:, :, common_hips], every, axis=0)
    hip_q = np.repeat(control('dof_pos')[:, :, common_hips], every, axis=0)
    hip_v = np.repeat(control('dof_vel')[:, :, common_hips], every, axis=0)
    rows = []
    for lo in range(0, 100, 10):
        mask = (phase >= lo / 100) & (phase < (lo + 10) / 100) & valid_physics[:, None]
        for env, name in enumerate(('A', 'B')):
            mean = lambda value: float(value[:, env][mask].mean())
            rows.append(dict(phase_pct=[lo, lo+10], environment=name, samples=int(mask.sum()),
                             wearer_mean_Nm=mean(actuator), wearer_rms_Nm=rms(actuator[:, env][mask]),
                             assist_mean_Nm=mean(applied), terms_mean_Nm={k: mean(v) for k, v in terms.items()},
                             foot_contact_fraction={str(threshold): mean((side_force_norm > threshold).astype(float))
                                                    for threshold in (5, 20, 50)},
                             foot_force_mean_N=float(np.mean(side_force_norm[:, env][mask])),
                             foot_vertical_force_mean_N=mean(side_force[:, :, :, 2]),
                             knee_angle_deg=mean(np.degrees(knee_q)), knee_velocity_rad_s=mean(knee_v),
                             hip_angle_deg=mean(np.degrees(hip_q)), hip_velocity_rad_s=mean(hip_v),
                             hip_target_deg=mean(np.degrees(target)),
                             hip_target_error_deg=mean(np.degrees(target-hip_q))))
    result = dict(run_dir=str(run_dir), validation=validation, phase_rows=rows,
                  caveats=['Coordinate-based inertial decomposition is not a causal knee-lock experiment.',
                           'Phase zero is reference hip maximum flexion, not heel strike.',
                           'Contact split uses constraint type; passive and other constraints are kept separate.',
                           'Sparse qfrc_constraint scratch may be invalid after convergence; final J-transpose-force is used.',
                           'Iterative solver balance residual is reported as a separate term, not physical coupling.',
                           'One gait, one seed; no new policy or generalization claim.'])
    (run_dir / 'fullbody_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def render(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = result['phase_rows']
    a, b = [[r for r in rows if r['environment'] == name] for name in ('A', 'B')]
    x = np.arange(5, 100, 10)
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True, layout='constrained')
    axes[0].plot(x, [r['wearer_mean_Nm'] for r in a], 'o-', label='Wearer A')
    axes[0].plot(x, [r['wearer_mean_Nm'] for r in b], 'o-', label='Wearer B')
    axes[0].plot(x, [r['assist_mean_Nm'] for r in b], 'o-', label='Assist B')
    axes[0].set_ylabel('Mean torque [Nm]')
    for key, label in [('hip_inertia', 'Hip own acceleration'), ('ipsilateral_knee_coupling', 'Knee coupling'),
                       ('pelvis_coupling', 'Pelvis coupling'), ('other_joint_coupling', 'Other joints'),
                       ('gravity', 'Gravity'), ('velocity_bias', 'Velocity bias'),
                       ('minus_contact_constraint', 'Minus contact contribution')]:
        axes[1].plot(x, [r['terms_mean_Nm'][key] for r in a], 'o-', label=label)
    axes[1].set_ylabel('A force-balance terms [Nm]')
    axes[2].plot(x, [100*r['foot_contact_fraction']['20'] for r in a], 'o-', label='A: foot force >20 N')
    axes[2].plot(x, [100*r['foot_contact_fraction']['20'] for r in b], 'o-', label='B: foot force >20 N')
    axes[2].set_ylabel('Foot contact [% samples]')
    axes[3].plot(x, [r['hip_target_error_deg'] for r in a], 'o-', label='A: PD target - hip')
    axes[3].plot(x, [r['hip_target_error_deg'] for r in b], 'o-', label='B: PD target - hip')
    axes[3].set_ylabel('Hip target error [deg]')
    for ax in axes:
        ax.axvspan(70, 90, color='#bb4455', alpha=.08)
        ax.axhline(0, color='#777', lw=.6)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xticks(np.arange(0, 101, 10))
    axes[-1].set_xlabel('Reference hip phase [%], 0 = maximum flexion')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    fig.savefig(out.with_suffix('.png'), dpi=140)
    plt.close(fig)
    svg = buf.getvalue(); svg = svg[svg.index('<svg'):]
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>전신 걷기 동역학 진단</title>'
            '<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;padding:0 20px}'
            'svg{width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>전신 걷기 계측 결과</h1>'
            '<p>기존 L9 정책을 그대로 실행한 진단이다. 새 정책 성능 결과가 아니다. '
            '관성항 분해는 같은 궤적의 운동방정식 분해이며 무릎을 고정한 인과 실험이 아니다.</p>'
            + svg + '<h2>계측 검증</h2><pre>' + json.dumps(result['validation'], indent=2)
            + '</pre></html>')
    out.with_suffix('.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    result = analyze(args.run_dir)
    render(result, Path(args.run_dir) / 'fullbody_report')
    print(json.dumps(result['validation'], indent=2))
    for r in result['phase_rows']:
        if r['phase_pct'][0] in (70, 80):
            print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
