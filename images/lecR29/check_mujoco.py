# Lec R29 실습 재현 스크립트 — MuJoCo 2R + CBF 참조 필터 + computed torque(R19)
# 실행: python3 check_mujoco.py  (numpy/mujoco 필요)
import numpy as np
import mujoco

# ---------- 2링크 모델 (R10 WE-3와 동일) ----------
XML = f"""
<mujoco><option gravity="0 -9.81 0" timestep="0.001"/><worldbody>
  <body><joint type="hinge" axis="0 0 1"/>
    <inertial pos="0.5 0 0" mass="1" diaginertia="0.001 {1/12} {1/12}"/>
    <body pos="1 0 0"><joint type="hinge" axis="0 0 1"/>
      <inertial pos="0.5 0 0" mass="1" diaginertia="0.001 {1/12} {1/12}"/>
      <site name="ee" pos="1 0 0"/>
    </body></body></worldbody></mujoco>"""
l1 = l2 = 1.0

def ik2r(x, y):
    c2 = (x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1 + l2*np.cos(q2))
    return np.array([q1, q2])

# ---------- 기준 태스크: 원 (중심 (1.1, 0.3), 반경 0.3, 2 s/바퀴, 2바퀴) ----------
cx, cy, R, T_lap = 1.1, 0.3, 0.3, 2.0
w = 2*np.pi/T_lap
dt, T_end = 1e-3, 4.0
ts = np.arange(0, T_end, dt)
p_circle = np.stack([cx + R*np.cos(w*ts + np.pi), cy + R*np.sin(w*ts + np.pi)], axis=1)
v_circle = np.stack([-R*w*np.sin(w*ts + np.pi), R*w*np.cos(w*ts + np.pi)], axis=1)

X_WALL = 1.3                      # 금지 영역: EEF x > 1.3 (원의 최대 x = 1.4)

def make_ref(alpha):
    """참조 속도 레벨 CBF 필터: v_x <= alpha * (X_WALL - p_x). alpha=None이면 무필터."""
    p = p_circle[0].copy()
    P = np.empty_like(p_circle)
    k_r = 5.0
    for i in range(len(ts)):
        v = v_circle[i] + k_r*(p_circle[i] - p)
        if alpha is not None:
            v[0] = min(v[0], alpha*(X_WALL - p[0]))   # CBF-QP 해석해 (E3)
        P[i] = p
        p = p + dt*v
    return P

def run(alpha):
    P_ref = make_ref(alpha)
    q_d = np.array([ik2r(*p) for p in P_ref])
    qd_d = np.gradient(q_d, dt, axis=0)
    qdd_d = np.gradient(qd_d, dt, axis=0)

    m_true = mujoco.MjModel.from_xml_string(XML)
    m_nom = mujoco.MjModel.from_xml_string(XML)
    d_true, d_nom = mujoco.MjData(m_true), mujoco.MjData(m_nom)
    d_true.qpos[:], d_true.qvel[:] = q_d[0], qd_d[0]
    mujoco.mj_forward(m_true, d_true)
    Mn = np.zeros((2, 2))
    Kp, Kd = 100*np.eye(2), 20*np.eye(2)

    ee = np.empty((len(ts), 2)); qerr = np.empty(len(ts))
    for i in range(len(ts)):
        q, qd = d_true.qpos.copy(), d_true.qvel.copy()
        d_nom.qpos[:], d_nom.qvel[:] = q, qd
        mujoco.mj_forward(m_nom, d_nom)
        mujoco.mj_fullM(m_nom, Mn, d_nom.qM)
        e, ed = q_d[i] - q, qd_d[i] - qd
        tau = Mn@(qdd_d[i] + Kd@ed + Kp@e) + d_nom.qfrc_bias   # computed torque (R19)
        d_true.qfrc_applied[:] = tau
        mujoco.mj_step(m_true, d_true)
        ee[i] = d_true.site('ee').xpos[:2]
        qerr[i] = np.linalg.norm(e)
    margin = (X_WALL - ee[:, 0].max())*1000                     # 실제 EEF의 벽 여유 [mm]
    dist_circle = np.sqrt(np.mean(np.sum((ee - p_circle)**2, axis=1)))*1000
    track = np.sqrt(np.mean(np.sum((ee - P_ref)**2, axis=1)))*1000
    return margin, dist_circle, track

print(f"벽: x = {X_WALL}, 원 최대 x = {cx + R}")
print(f"{'설정':>10s} | {'벽 여유(min) mm':>15s} | {'원 대비 RMS mm':>14s} | {'참조 추종 RMS mm':>15s}")
for label, alpha in [('무필터', None), ('α=1', 1.0), ('α=5', 5.0), ('α=20', 20.0)]:
    margin, dist, track = run(alpha)
    print(f"{label:>10s} | {margin:15.2f} | {dist:14.2f} | {track:15.3f}")
