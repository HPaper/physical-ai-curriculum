"""Lec R27 실습 검증 스크립트 — MuJoCo IMU 시뮬레이션과 자세 추정 파이프라인.
① MuJoCo 센서 의미론 검증: accelerometer = 비력(specific force), gyro = 몸체좌표 각속도
② 흔들리는 몸통(1축)에서 노이즈 주입 → 상보 필터/EKF로 pitch 추정 → 참값과 RMSE
(humanoid.xml이 있으면 같은 파이프라인을 그대로 적용할 수 있다 — 본문 실습 4단계)
"""
import numpy as np
import mujoco

XML = """
<mujoco model="tilting_torso">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="torso" pos="0 0 0.5">
      <joint name="pitch" type="hinge" axis="0 1 0" stiffness="10" damping="1.0"/>
      <geom type="capsule" fromto="0 0 0  0 0 0.4" size="0.04" density="800"/>
      <site name="imu" pos="0 0 0.3"/>
    </body>
  </worldbody>
  <actuator><motor joint="pitch" gear="1" ctrlrange="-10 10"/></actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <gyro name="gyr" site="imu"/>
    <framequat name="quat_true" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

# ---------- ① 센서 의미론 검증 ----------
print("=== ① MuJoCo IMU 센서 의미론 ===")
# (a) 정지·직립: 가속도계는 '비력' — 자유낙하가 아니라 지지되고 있으므로 +g를 읽는다
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)          # qacc까지 계산해야 센서가 채워진다
mujoco.mj_step(m, d)             # 한 스텝 진행해 정상 상태 센서값
print(f"정지 직립 가속도계: {np.round(d.sensor('acc').data, 4)}  (기대: [0, 0, +9.81])")
print(f"정지 직립 자이로  : {np.round(d.sensor('gyr').data, 6)}  (기대: [0, 0, 0])")

# (b) 30° 기울여 고정(속도 0): 사이트 좌표계로 회전된 중력 반력
mujoco.mj_resetData(m, d)
th0 = np.radians(30)
d.qpos[0] = th0
mujoco.mj_forward(m, d)
# 이 순간은 정적 평형이 아니므로(회전낙하 시작) 비력 = Rᵀ(a_lin − g_vec)
acc = d.sensor('acc').data.copy()
# 손계산: 축 y 힌지, 사이트 위치 p = r(sinθ, 0, cosθ) → p̈ = r·q̈(cosθ, 0, −sinθ) (q̇=0)
qdd = d.qacc[0]
r = 0.3
a_lin_world = r*qdd*np.array([np.cos(th0), 0, -np.sin(th0)])
g_vec = np.array([0, 0, -9.81])
Ry = np.array([[np.cos(th0), 0, np.sin(th0)], [0, 1, 0], [-np.sin(th0), 0, np.cos(th0)]])
f_pred = Ry.T @ (a_lin_world - g_vec)
print(f"30° 기울임 직후 가속도계: {np.round(acc, 3)} / 손계산 예측: {np.round(f_pred, 3)}")

# ---------- ② 자세 추정 파이프라인 (본문 WE-1·WE-2와 동일 구조) ----------
print("\n=== ② 흔들리는 몸통 자세 추정 (MuJoCo 데이터) ===")
rng = np.random.default_rng(7)
mujoco.mj_resetData(m, d)
n_sub = 10                       # 물리 1 kHz, IMU 100 Hz
T = 30.0
n = int(T/(m.opt.timestep*n_sub))
dt = m.opt.timestep*n_sub

sig_g, sig_f, bias_g = 0.01, 0.2, 0.02
th_true = np.zeros(n); gyro = np.zeros(n); f_meas = np.zeros((n, 2))
for k in range(n):
    for _ in range(n_sub):
        d.ctrl[0] = 3.0*np.sin(2*np.pi*0.3*d.time)   # 몸통을 계속 흔드는 외란
        mujoco.mj_step(m, d)
    th_true[k] = d.qpos[0]
    gyro[k] = d.sensor('gyr').data[1] + bias_g + sig_g*rng.standard_normal()
    acc = d.sensor('acc').data
    f_meas[k] = acc[[0, 2]] + sig_f*rng.standard_normal(2)   # x_b, z_b 축

th_acc = np.arctan2(-f_meas[:, 0], f_meas[:, 1])  # 정지 시 f_b = g(−sinθ, cosθ)

# 상보 필터 (본문 WE-1의 α)
alpha = 0.99
th_cf = np.zeros(n); th_cf[0] = th_acc[0]
for k in range(1, n):
    th_cf[k] = alpha*(th_cf[k-1] + gyro[k]*dt) + (1-alpha)*th_acc[k]

# EKF (본문 WE-2와 완전 동일: h(θ) = g(−sinθ, cosθ))
x = np.array([th_acc[0], 0.0]); P = np.diag([0.1**2, 0.05**2])
Qw = np.diag([(sig_g*dt)**2, (1e-4*dt)**2]); Rv = np.eye(2)*1.2**2
F = np.array([[1., -dt], [0., 1.]])
est = np.zeros((n, 2))
for k in range(1, n):
    x = np.array([x[0] + (gyro[k]-x[1])*dt, x[1]])
    P = F @ P @ F.T + Qw
    h = 9.81*np.array([-np.sin(x[0]), np.cos(x[0])])
    H = np.array([[-9.81*np.cos(x[0]), 0.], [-9.81*np.sin(x[0]), 0.]])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rv)
    x = x + K @ (f_meas[k] - h)
    P = (np.eye(2) - K @ H) @ P
    est[k] = x

sl = slice(int(3/dt), None)
rm = lambda a: np.sqrt(np.mean((a[sl] - th_true[sl])**2))
print(f"자이로 적분만  RMSE: {rm(np.cumsum(gyro)*dt + th_acc[0]):.4f} rad")
print(f"가속도계만    RMSE: {rm(th_acc):.4f} rad")
print(f"상보 필터     RMSE: {rm(th_cf):.4f} rad")
print(f"EKF          RMSE: {rm(est[:, 0]):.4f} rad (바이어스 추정 {est[-1,1]:.4f} rad/s)")
