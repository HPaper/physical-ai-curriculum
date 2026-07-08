# -*- coding: utf-8 -*-
# Lec R30 실습 검증: 실험 (a)를 MuJoCo 플랜트로 재현 — gen_figs.py(numpy RK4)와 대조
# 실행: python3 check_mujoco.py
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline

# ---- 제어기 쪽 모델 (gen_figs.py와 동일: R10 WE-3 / R19 파라미터) ----
m1, m2, l1, l2, lc1, lc2 = 1.0, 1.0, 1.0, 1.0, 0.5, 0.5
I1 = I2 = 1.0 / 12
grav = 9.81

def M_mat(q):
    c2 = np.cos(q[1])
    return np.array([[I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2),
                      I2 + m2 * (lc2**2 + l1 * lc2 * c2)],
                     [I2 + m2 * (lc2**2 + l1 * lc2 * c2), I2 + m2 * lc2**2]])

def C_mat(q, qd):
    h = m2 * l1 * lc2 * np.sin(q[1])
    return np.array([[-h * qd[1], -h * (qd[0] + qd[1])], [h * qd[0], 0.0]])

def g_vec(q):
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return grav * np.array([(m1 * lc1 + m2 * l1) * c1 + m2 * lc2 * c12, m2 * lc2 * c12])

def fk(q):
    return np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
                     l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])])

def ik2r(p):
    x, y = p
    c2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])

# ---- MuJoCo 플랜트: 명시적 <inertial>이 geom 유래 관성을 덮어쓴다 ----
XML = """
<mujoco model="arm2r">
  <option timestep="0.001" gravity="0 -9.81 0" integrator="RK4"/>
  <worldbody>
    <body name="l1">
      <joint name="q1" type="hinge" axis="0 0 1"/>
      <inertial pos="0.5 0 0" mass="1" diaginertia="0.0833333 0.0833333 0.0833333"/>
      <geom type="capsule" fromto="0 0 0  1 0 0" size="0.03" contype="0" conaffinity="0"/>
      <body name="l2" pos="1 0 0">
        <joint name="q2" type="hinge" axis="0 0 1"/>
        <inertial pos="0.5 0 0" mass="1" diaginertia="0.0833333 0.0833333 0.0833333"/>
        <geom type="capsule" fromto="0 0 0  1 0 0" size="0.03" contype="0" conaffinity="0"/>
        <site name="ee" pos="1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

# ---- 0) 동역학 일치 확인: MuJoCo의 M·bias vs 손 유도 (R10·R11 회수) ----
q_test, qd_test = np.array([0.3, 0.5]), np.array([-0.2, 0.4])
d.qpos[:], d.qvel[:] = q_test, qd_test
mujoco.mj_forward(m, d)
Mfull = np.zeros((2, 2))
mujoco.mj_fullM(m, Mfull, d.qM)
bias_ours = C_mat(q_test, qd_test) @ qd_test + g_vec(q_test)
print("M 최대 편차      :", np.abs(Mfull - M_mat(q_test)).max())
print("C·qd+g 최대 편차 :", np.abs(d.qfrc_bias - bias_ours).max())

# ---- 공통 기준 궤적 (gen_figs.py와 동일) ----
def ref(t):
    t = np.asarray(t, dtype=float)
    x = 1.0 + 0.15 * (1 - np.cos(2 * np.pi * 0.4 * t))
    y = 0.45 * np.cos(2 * np.pi * 0.2 * t)
    return np.stack([x, y], axis=-1)

dt, T_end = 1e-3, 5.0
Kp, Kd = 400.0, 40.0
ts = np.arange(0, T_end, dt)
ref_fine = ref(ts)
t_wp = np.arange(0, T_end + 1e-9, 0.1)
q_wp = np.array([ik2r(p) for p in ref(t_wp)])

def build_setpoints(mode):
    if mode == 'zoh':
        idx = np.minimum((ts / 0.1).astype(int), len(t_wp) - 1)
        qd_ = q_wp[idx]
        return qd_, np.zeros_like(qd_), np.zeros_like(qd_)
    cs = CubicSpline(t_wp, q_wp, axis=0)
    return cs(ts), cs(ts, 1), cs(ts, 2)

# ---- 실험 (a) 재현: 플랜트만 MuJoCo로 교체 ----
print("\n실험 (a) 재현 — 플랜트: MuJoCo (integrator=RK4, 1kHz)")
for mode, name in [('zoh', 'ZOH'), ('cub', '3차 스플라인')]:
    qd_, vd_, ad_ = build_setpoints(mode)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q_wp[0]
    err = np.zeros(len(ts))
    tau_hist = np.zeros((len(ts), 2))
    for i in range(len(ts)):
        q, qdv = d.qpos.copy(), d.qvel.copy()
        e, ed = qd_[i] - q, vd_[i] - qdv
        tau = M_mat(q) @ (ad_[i] + Kd * ed + Kp * e) + C_mat(q, qdv) @ qdv + g_vec(q)
        d.qfrc_applied[:] = tau
        tau_hist[i] = tau
        err[i] = np.linalg.norm(fk(q) - ref_fine[i]) * 1000
        mujoco.mj_step(m, d)
    w = ts >= 0.5
    print(f"  {name:10s} RMS EEF 오차 {np.sqrt(np.mean(err[w]**2)):8.2f} mm | "
          f"피크 오차 {err[w].max():7.2f} mm | 피크 |tau| {np.abs(tau_hist[w]).max():6.1f} N·m")
print("\n(비교: gen_figs.py numpy RK4 — ZOH 74.37 mm / 3차 0.01 mm)")
