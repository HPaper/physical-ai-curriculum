# Lec R28 실습 검증: MuJoCo에서 "진짜" 파라미터를 숨기고 궤적 데이터만으로 동정
#   → 동정 모델로 computed torque(R19) 폐루프 성능 회복 확인
# 실행: python3 check_mujoco.py  (numpy / scipy / mujoco 3.2.5)
import numpy as np
import mujoco
from scipy.signal import butter, filtfilt

G = 9.81
DT = 0.002  # 500 Hz

# ------------------------------------------------------------------
# 1. "진짜" 로봇 (학생에게는 비공개라고 가정): 비균일 질량 + 관절 점성마찰
#    링크를 x-y 평면(수직면)에 두고 중력을 -y로 → R10의 해석 모델과 동일 기하
# ------------------------------------------------------------------
m1t, lc1t, I1t = 1.30, 0.45, 0.110   # I는 COM 기준 z축 관성
m2t, lc2t, I2t = 0.75, 0.55, 0.070
l1 = 1.0
d1t, d2t = 0.15, 0.09                # 관절 점성 damping [N·m·s/rad]

XML = f"""
<mujoco model="arm2r_hidden">
  <option timestep="{DT}" gravity="0 -{G} 0" integrator="implicitfast"/>
  <worldbody>
    <body name="l1">
      <joint name="q1" type="hinge" axis="0 0 1" damping="{d1t}"/>
      <geom type="capsule" fromto="0 0 0  {l1} 0 0" size="0.04" density="1"/>
      <inertial pos="{lc1t} 0 0" mass="{m1t}" diaginertia="0.002 {I1t} {I1t}"/>
      <body name="l2" pos="{l1} 0 0">
        <joint name="q2" type="hinge" axis="0 0 1" damping="{d2t}"/>
        <geom type="capsule" fromto="0 0 0  0.8 0 0" size="0.035" density="1"/>
        <inertial pos="{lc2t} 0 0" mass="{m2t}" diaginertia="0.002 {I2t} {I2t}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="q1" ctrlrange="-300 300"/>
    <motor joint="q2" ctrlrange="-300 300"/>
  </actuator>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)

# 참 base parameters (채점용 — 실습에서는 마지막에만 공개)
TH_TRUE = np.array([I1t + I2t + m1t * lc1t**2 + m2t * (l1**2 + lc2t**2),
                    m2t * l1 * lc2t,
                    I2t + m2t * lc2t**2,
                    m1t * lc1t + m2t * l1,
                    m2t * lc2t,
                    d1t, d2t])
# 명목(CAD) 모델: R10의 균일 막대 가정 (마찰 없음)
TH_NOM = np.array([5 / 3, 0.5, 1 / 3, 1.5, 0.5, 0.0, 0.0])


def regressor(q, qd, qdd):
    """마찰(점성) 열을 포함한 2R 회귀자: τ = Y·[θ1..θ5, d1, d2]"""
    c1, c2, s2, c12 = np.cos(q[0]), np.cos(q[1]), np.sin(q[1]), np.cos(q[0] + q[1])
    return np.array([
        [qdd[0], c2 * (2 * qdd[0] + qdd[1]) - s2 * (qd[1]**2 + 2 * qd[0] * qd[1]),
         qdd[1], G * c1, G * c12, qd[0], 0.0],
        [0.0, c2 * qdd[0] + s2 * qd[0]**2, qdd[0] + qdd[1], 0.0, G * c12, 0.0, qd[1]]])


def model_terms(th, q, qd):
    """θ로부터 M, C q̇ + g + 마찰 재구성 (computed torque용)"""
    c2 = np.cos(q[1])
    M = np.array([[th[0] + 2 * th[1] * c2, th[2] + th[1] * c2],
                  [th[2] + th[1] * c2, th[2]]])
    h = th[1] * np.sin(q[1])
    Cqd = np.array([-h * qd[1] * qd[0] - h * (qd[0] + qd[1]) * qd[1], h * qd[0]**2])
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    g = G * np.array([th[3] * c1 + th[4] * c12, th[4] * c12])
    return M, Cqd + g + th[5:7] * qd


def ref(t, seed=2, f0=0.4, H=5, amp=0.5):
    """멀티사인 참조 궤적 (해석적 q, q̇, q̈)"""
    rng = np.random.default_rng(seed)
    q = np.zeros((len(t), 2)); qd = np.zeros_like(q); qdd = np.zeros_like(q)
    for j in range(2):
        for k in range(1, H + 1):
            w = 2 * np.pi * k * f0
            ph = rng.uniform(0, 2 * np.pi)
            a = amp / k
            q[:, j] += a * np.sin(w * t + ph)
            qd[:, j] += a * w * np.cos(w * t + ph)
            qdd[:, j] += -a * w**2 * np.sin(w * t + ph)
    return q, qd, qdd


# 설정 검증: 해석 회귀자와 MuJoCo 역동역학의 일치 (τ_mj = Yθ* + damping은 qfrc_passive)
_d = mujoco.MjData(model)
_rng = np.random.default_rng(0)
_errs = []
for _ in range(20):
    _q, _qd, _qdd = _rng.uniform(-np.pi, np.pi, 2), _rng.uniform(-3, 3, 2), _rng.uniform(-5, 5, 2)
    _d.qpos[:] = _q; _d.qvel[:] = _qd; _d.qacc[:] = _qdd
    mujoco.mj_inverse(model, _d)
    _errs.append(np.abs(_d.qfrc_inverse - regressor(_q, _qd, _qdd) @ TH_TRUE).max())
print(f"[설정] max|mj_inverse - Yθ*| (랜덤 20 상태) = {max(_errs):.2e}")

# ------------------------------------------------------------------
# 2. 데이터 수집: PD로 여기 궤적 추종, (q 측정 + 지령 토크) 기록
# ------------------------------------------------------------------
T_ID = 20.0
t_id = np.arange(0, T_ID, DT)
qr, qdr, _ = ref(t_id, seed=2)
Kp, Kd = np.diag([120.0, 60.0]), np.diag([24.0, 12.0])

data = mujoco.MjData(model)
data.qpos[:] = qr[0]; data.qvel[:] = qdr[0]
mujoco.mj_forward(model, data)
Q_LOG, TAU_LOG = [], []
for i in range(len(t_id)):
    tau = Kp @ (qr[i] - data.qpos) + Kd @ (qdr[i] - data.qvel)
    data.ctrl[:] = tau
    Q_LOG.append(data.qpos.copy()); TAU_LOG.append(tau.copy())
    mujoco.mj_step(model, data)
Q_LOG, TAU_LOG = np.array(Q_LOG), np.array(TAU_LOG)

# 엔코더 잡음 + 수치 미분 + zero-phase 저역 필터 (실기 흉내)
rng = np.random.default_rng(11)
Qm = Q_LOG + rng.normal(0, 5e-5, Q_LOG.shape)
b, a = butter(4, 8.0, fs=1 / DT)      # 8 Hz 저역 (여기 대역 2 Hz의 4배)
Qf = filtfilt(b, a, Qm, axis=0)
Qdf = np.gradient(Qf, DT, axis=0)
Qddf = np.gradient(Qdf, DT, axis=0)

# 필터 과도 구간 버리고 Y·τ 스택
sl = slice(500, len(t_id) - 500)
Y = np.vstack([regressor(Qf[i], Qdf[i], Qddf[i]) for i in range(len(t_id))[sl]])
TAU = np.hstack([TAU_LOG[i] for i in range(len(t_id))[sl]])
th_hat = np.linalg.lstsq(Y, TAU, rcond=None)[0]

print("=== 동정 결과 (500 Hz, 엔코더 잡음 5e-5 rad, 8 Hz filtfilt) ===")
print("κ(Y) =", f"{np.linalg.cond(Y):.1f}")
names = ["θ1", "θ2", "θ3", "θ4", "θ5", "d1", "d2"]
for n, tt, th, tn in zip(names, TH_TRUE, th_hat, TH_NOM):
    print(f"  {n}: 참 {tt:8.4f} | 추정 {th:8.4f} ({100*(th-tt)/tt if tt else np.nan:+.2f}%) | CAD 명목 {tn:8.4f}")
rel = np.linalg.norm(th_hat - TH_TRUE) / np.linalg.norm(TH_TRUE)
rel_nom = np.linalg.norm(TH_NOM - TH_TRUE) / np.linalg.norm(TH_TRUE)
print(f"전체 상대오차 ‖θ̂-θ*‖/‖θ*‖ = {rel*100:.2f}%  (CAD 명목 모델은 {rel_nom*100:.1f}%)")


# ------------------------------------------------------------------
# 3. 폐루프 검증: computed torque (R19) — 명목 vs 동정 vs 참 모델
# ------------------------------------------------------------------
def run_ct(th, T=10.0, seed=9, kp=100.0, kd=20.0):
    t = np.arange(0, T, DT)
    qr, qdr, qddr = ref(t, seed=seed, f0=0.5, H=4, amp=0.5)
    d = mujoco.MjData(model)
    d.qpos[:] = qr[0]; d.qvel[:] = qdr[0]
    mujoco.mj_forward(model, d)
    E = []
    for i in range(len(t)):
        M, hvec = model_terms(th, d.qpos, d.qvel)
        v = qddr[i] + kd * (qdr[i] - d.qvel) + kp * (qr[i] - d.qpos)
        d.ctrl[:] = M @ v + hvec
        E.append(qr[i] - d.qpos)
        mujoco.mj_step(model, d)
    return np.sqrt(np.mean(np.array(E)**2))


rms_nom = run_ct(TH_NOM)
rms_id = run_ct(th_hat)
rms_true = run_ct(TH_TRUE)
print("\n=== computed torque 추종 RMS (검증 궤적, Kp=100, Kd=20) ===")
print(f"  CAD 명목 모델: {rms_nom*1e3:.3f} mrad")
print(f"  동정 모델:     {rms_id*1e3:.3f} mrad")
print(f"  참 모델(오라클): {rms_true*1e3:.3f} mrad")
print(f"  명목 대비 개선: {rms_nom/rms_id:.1f}배 / 오라클 대비 {rms_id/rms_true:.2f}배")

# 저게인에서 모델 오차가 더 아프게 드러나는지 (R19의 논점 회수)
rms_nom_lo = run_ct(TH_NOM, kp=25.0, kd=10.0)
rms_id_lo = run_ct(th_hat, kp=25.0, kd=10.0)
print(f"\n저게인(Kp=25): 명목 {rms_nom_lo*1e3:.2f} mrad vs 동정 {rms_id_lo*1e3:.3f} mrad "
      f"({rms_nom_lo/rms_id_lo:.0f}배)")
