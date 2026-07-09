"""Lec R27 그림 생성 스크립트.
fig1: 한 장 요약 — 상보 필터의 주파수 분리(자이로 고주파 + 가속도계 저주파) + 시계열
fig2: 상보 필터 α 스윕 — 컷오프의 트레이드오프 (바이어스 누설 vs 가속 오염)
fig3: EKF vs 상보 필터 — 바이어스 추정, 정확도, 선형화 지점의 위험(초기 오차 스윕)
fig4: 모멘텀 옵저버 — 2R 충돌 감지 잔차 스파이크, K_O 스윕, q̈ 미분 방식과 비교
fig5: 오도메트리 드리프트 — 적분 오차의 성장 법칙 (√t vs t vs t^1.5)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lec59"
GRAV = 9.81

# =====================================================================
# 공통 시뮬레이션: 흔들리는 몸통(1축 기울기)에 달린 IMU (WE-1, WE-2)
# =====================================================================
rng = np.random.default_rng(27)
dt = 0.01                               # 100 Hz
T = 60.0
t = np.arange(0, T, dt)
N = t.size

w1, w2 = 2*np.pi*0.3, 2*np.pi*1.1       # 흔들림의 두 주파수 성분
th_true = 0.6*np.sin(w1*t) + 0.2*np.sin(w2*t)
om_true = 0.6*w1*np.cos(w1*t) + 0.2*w2*np.cos(w2*t)
al_true = -0.6*w1**2*np.sin(w1*t) - 0.2*w2**2*np.sin(w2*t)

# --- 자이로: 각속도 + 상수 바이어스 + 백색 노이즈 ---
bias_g = 0.02                            # rad/s (MEMS 전원 인가 시 바이어스)
sig_g = 0.01                             # rad/s (샘플당)
gyro = om_true + bias_g + sig_g*rng.standard_normal(N)

# --- 가속도계: 비력 f_b = Rᵀ(p̈ − g_vec), IMU는 피벗에서 r 위치 ---
r_imu = 0.3
# 세계좌표 IMU 가속도 (피벗 중심 회전): p = r(sinθ, cosθ)
pxdd = r_imu*(al_true*np.cos(th_true) - om_true**2*np.sin(th_true))
pzdd = r_imu*(-al_true*np.sin(th_true) - om_true**2*np.cos(th_true))
fw = np.stack([pxdd - 0.0, pzdd + GRAV])            # p̈ − g_vec, g_vec=(0,−g)
c, s = np.cos(th_true), np.sin(th_true)
fb = np.stack([c*fw[0] - s*fw[1], s*fw[0] + c*fw[1]])  # Rᵀ f_w
sig_f = 0.2                                          # m/s² (축당)
fb_meas = fb + sig_f*rng.standard_normal(fb.shape)
th_acc = np.arctan2(-fb_meas[0], fb_meas[1])         # 가속도계 기울기 (정지 시 정확)

def rmse(a, b, skip=int(5/dt)):
    return np.sqrt(np.mean((a[skip:] - b[skip:])**2))

# --- 단독 사용의 실패 ---
th_gyro_only = np.cumsum(gyro)*dt                    # 순수 적분 → 드리프트
print("=== WE-1 사전: 단독 센서의 실패 ===")
print(f"자이로 적분 60초 후 오차: {abs(th_gyro_only[-1]-th_true[-1]):.3f} rad "
      f"(이론 바이어스 드리프트 b·t = {bias_g*T:.2f} rad)")
print(f"자이로 적분 RMSE: {rmse(th_gyro_only, th_true):.4f} rad")
print(f"가속도계 단독 RMSE: {rmse(th_acc, th_true):.4f} rad "
      f"(정지 구간이라면 σ≈{sig_f/GRAV:.3f} rad — 운동 가속이 지배)")

# =====================================================================
# WE-1: 상보 필터와 α 스윕
# =====================================================================
def complementary(alpha):
    th = np.zeros(N); th[0] = th_acc[0]
    for k in range(1, N):
        th[k] = alpha*(th[k-1] + gyro[k]*dt) + (1-alpha)*th_acc[k]
    return th

alphas_table = [0.9, 0.99, 0.995, 0.999, 0.9999]
print("\n=== WE-1: 상보 필터 α 스윕 ===")
print(f"{'α':>8} {'τ [s]':>8} {'f_c [Hz]':>9} {'RMSE [rad]':>11}")
for a in alphas_table:
    tau = a*dt/(1-a)
    e = rmse(complementary(a), th_true)
    print(f"{a:>8} {tau:>8.2f} {1/(2*np.pi*tau):>9.4f} {e:>11.4f}")

alpha_fine = 1 - np.logspace(np.log10(2e-4), np.log10(0.5), 60)
rmse_fine = np.array([rmse(complementary(a), th_true) for a in alpha_fine])
i_opt = np.argmin(rmse_fine)
a_opt = alpha_fine[i_opt]
tau_opt = a_opt*dt/(1-a_opt)
print(f"최적 α = {a_opt:.4f} (τ = {tau_opt:.2f} s, f_c = {1/(2*np.pi*tau_opt):.3f} Hz), "
      f"RMSE = {rmse_fine[i_opt]:.4f} rad")
print(f"바이어스 누설 예측 τ·b: α=0.999 → {0.999*dt/(1-0.999)*bias_g:.3f} rad, "
      f"α_opt → {tau_opt*bias_g:.4f} rad")

t_cf0 = time.perf_counter()
th_cf = complementary(a_opt)
t_cf = (time.perf_counter() - t_cf0)/N*1e6          # µs/step

# =====================================================================
# WE-2: EKF (상태 = 기울기 θ + 자이로 바이어스 b), 관측 = 가속도계 원신호 2축
# =====================================================================
def run_ekf(th0=None, P0=None, sig_f_eff=1.2, sig_bw=1e-4, n_steps=N):
    x = np.array([th_acc[0] if th0 is None else th0, 0.0])
    P = np.diag([0.1**2, 0.05**2]) if P0 is None else P0.copy()
    Qw = np.diag([(sig_g*dt)**2, (sig_bw*dt)**2])
    Rv = np.eye(2)*sig_f_eff**2
    F = np.array([[1., -dt], [0., 1.]])
    est = np.zeros((n_steps, 2))
    est[0] = x
    for k in range(1, n_steps):
        # 예측: 자이로를 입력으로 (바이어스 추정치를 빼고 적분)
        x = np.array([x[0] + (gyro[k]-x[1])*dt, x[1]])
        P = F @ P @ F.T + Qw
        # 보정: h(θ) = g(−sinθ, cosθ), H = ∂h/∂x  ← 선형화는 현재 추정치에서!
        h = GRAV*np.array([-np.sin(x[0]), np.cos(x[0])])
        H = np.array([[-GRAV*np.cos(x[0]), 0.],
                      [-GRAV*np.sin(x[0]), 0.]])
        S = H @ P @ H.T + Rv
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (fb_meas[:, k] - h)
        P = (np.eye(2) - K @ H) @ P
        est[k] = x
    return est

t_ekf0 = time.perf_counter()
ekf = run_ekf()
t_ekf = (time.perf_counter() - t_ekf0)/N*1e6

print("\n=== WE-2: EKF vs 상보 필터 ===")
print(f"상보 필터(최적 α)  RMSE: {rmse(th_cf, th_true):.4f} rad")
print(f"EKF               RMSE: {rmse(ekf[:,0], th_true):.4f} rad")
print(f"EKF 바이어스 추정: 최종 {ekf[-1,1]:.4f} rad/s (참값 {bias_g}), "
      f"10초 시점 {ekf[int(10/dt),1]:.4f}")
print(f"계산 시간: CF {t_cf:.2f} µs/step, EKF {t_ekf:.2f} µs/step "
      f"(x{t_ekf/t_cf:.0f}, 파이썬 루프 기준)")

# --- 선형화 지점의 위험: 초기 오차 스윕 (자신만만한 P0로) ---
print("\n--- EKF 초기 오차 스윕 (P0 = diag(0.01², 0.01²) 과신 조건) ---")
init_errs = [30, 90, 150, 175]
ekf_init_runs = {}
for e_deg in init_errs:
    est = run_ekf(th0=np.radians(e_deg), P0=np.diag([0.01**2, 0.01**2]))
    err_end = rmse(est[:, 0], th_true, skip=int(50/dt))
    ekf_init_runs[e_deg] = est
    tag = '수렴' if err_end < 0.05 else ('잔류 오차' if err_end < 0.1 else '수렴 실패')
    print(f"초기 오차 {e_deg:>3}° → 마지막 10초 RMSE {err_end:.4f} rad ({tag})")

# =====================================================================
# WE-3: 모멘텀 옵저버 — 2R 팔 (R10의 M, C, g 재사용)
# =====================================================================
m1, m2, l1, l2, lc1, lc2 = 1.0, 1.0, 1.0, 1.0, 0.5, 0.5
I1 = I2 = 1.0/12

def M_mat(q):
    c2 = np.cos(q[1])
    return np.array([[I1+I2+m1*lc1**2+m2*(l1**2+lc2**2+2*l1*lc2*c2),
                      I2+m2*(lc2**2+l1*lc2*c2)],
                     [I2+m2*(lc2**2+l1*lc2*c2), I2+m2*lc2**2]])

def C_mat(q, qd):
    h = m2*l1*lc2*np.sin(q[1])
    return np.array([[-h*qd[1], -h*(qd[0]+qd[1])], [h*qd[0], 0.]])

def g_vec(q):
    c1, c12 = np.cos(q[0]), np.cos(q[0]+q[1])
    return GRAV*np.array([(m1*lc1+m2*l1)*c1 + m2*lc2*c12, m2*lc2*c12])

def jac(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([[-l1*s1-l2*s12, -l2*s12], [l1*c1+l2*c12, l2*c12]])

q_d = np.array([np.pi/2, -np.pi/2])
F_ext = np.array([-20., -10.])           # 충돌 힘 [N]
tau_ext_true_peak = jac(q_d).T @ F_ext
print("\n=== WE-3: 모멘텀 옵저버 (2R) ===")
print(f"자세 q_d = [90°, −90°], J(q_d) =\n{jac(q_d)}")
print(f"충돌 힘 F = {F_ext} N → 참 외력 토크 τ_ext = JᵀF = {tau_ext_true_peak} N·m")

def simulate_2r(K_O, sig_qd=0.005, seed=3, T2=4.0, dt2=1e-3,
                t_hit=2.0, dur_hit=0.15):
    rng2 = np.random.default_rng(seed)
    n2 = int(T2/dt2)
    tt = np.arange(n2)*dt2
    q = q_d + np.array([0.1, -0.1])      # 초기 오프셋 → PD로 복귀
    qd = np.zeros(2)
    Kp = np.diag([60., 30.]); Kd = np.diag([15., 8.])
    KO = np.eye(2)*K_O
    # 옵저버 상태 (측정 기반)
    q_m0 = q + 1e-5*rng2.standard_normal(2)
    qd_m0 = qd + sig_qd*rng2.standard_normal(2)
    integ = np.zeros(2)
    p0 = M_mat(q_m0) @ qd_m0
    log = {k: np.zeros((n2, 2)) for k in ['r', 'tau_ext', 'q']}
    log['naive'] = np.zeros((n2, 2))
    qd_m_prev = qd_m0
    r = np.zeros(2)
    for k in range(n2):
        tau_ext = jac(q).T @ F_ext if t_hit <= tt[k] < t_hit+dur_hit else np.zeros(2)
        tau_cmd = g_vec(q) + Kp @ (q_d - q) - Kd @ qd      # 중력보상 + PD
        # --- 참 동역학 (semi-implicit Euler) ---
        qdd = np.linalg.solve(M_mat(q), tau_cmd + tau_ext - C_mat(q, qd) @ qd - g_vec(q))
        qd = qd + qdd*dt2
        q = q + qd*dt2
        # --- 측정 (엔코더 + 필터링된 속도, 노이즈) ---
        q_m = q + 1e-5*rng2.standard_normal(2)
        qd_m = qd + sig_qd*rng2.standard_normal(2)
        # --- 모멘텀 옵저버 ---
        p = M_mat(q_m) @ qd_m
        integ = integ + (tau_cmd + C_mat(q_m, qd_m).T @ qd_m - g_vec(q_m) + r)*dt2
        r = KO @ (p - integ - p0)
        # --- 순진한 대안: q̈을 수치미분해서 역동역학 ---
        qdd_num = (qd_m - qd_m_prev)/dt2
        log['naive'][k] = (M_mat(q_m) @ qdd_num + C_mat(q_m, qd_m) @ qd_m
                           + g_vec(q_m) - tau_cmd)
        qd_m_prev = qd_m
        log['r'][k] = r
        log['tau_ext'][k] = tau_ext
        log['q'][k] = q
    log['t'] = tt
    return log

KO_list = [10., 50., 200.]
logs = {K: simulate_2r(K) for K in KO_list}
print(f"\n{'K_O':>6} {'노이즈 플로어 σ(r₁) [N·m]':>26} {'피크 r₁ [N·m]':>14} "
      f"{'이론 피크 (1−e^−KΔt)':>20} {'감지 지연 [ms]':>13}")
for K in KO_list:
    lg = logs[K]
    quiet = lg['r'][int(1.0/1e-3):int(1.9/1e-3), 0]     # 정착 후 무접촉 구간
    floor = quiet.std()
    peak = np.abs(lg['r'][:, 0]).max()
    theo = abs(tau_ext_true_peak[0])*(1 - np.exp(-K*0.15))
    thresh = 5*floor
    hit_idx = np.argmax(np.abs(lg['r'][:, 0]) > thresh)
    delay = (lg['t'][hit_idx] - 2.0)*1e3 if hit_idx > 0 else np.nan
    print(f"{K:>6.0f} {floor:>26.4f} {peak:>14.3f} {theo:>20.3f} {delay:>13.1f}")

lg50 = logs[50.]
naive_std = lg50['naive'][int(1.0/1e-3):int(1.9/1e-3), 0].std()
mo_std = lg50['r'][int(1.0/1e-3):int(1.9/1e-3), 0].std()
print(f"\n순진한 역동역학(q̈ 수치미분) 잔차 σ: {naive_std:.1f} N·m "
      f"vs 모멘텀 옵저버(K_O=50): {mo_std:.4f} N·m — {naive_std/mo_std:.0f}배")

# 무노이즈 검증: r이 τ_ext의 1차 저역통과인지
lg_clean = simulate_2r(50., sig_qd=0.0)
k_hit = int(2.075/1e-3)                  # 충돌 중간 시점
r_mid = lg_clean['r'][k_hit, 0]
tau_mid = lg_clean['tau_ext'][k_hit, 0]
pred_mid = tau_mid*(1 - np.exp(-50.*0.075))
print(f"무노이즈 검증(충돌 75ms 시점): r₁ = {r_mid:.3f}, "
      f"1차 지연 예측 τ_ext(1−e^{{−K_O t}}) = {pred_mid:.3f} N·m")

# =====================================================================
# §5: 오도메트리 드리프트 — 성장 법칙
# =====================================================================
print("\n=== §5: 오도메트리 드리프트 ===")
dt_o = 0.1; T_o = 600.0; n_o = int(T_o/dt_o); v = 0.5
t_o = np.arange(1, n_o+1)*dt_o
n_mc = 300
rng3 = np.random.default_rng(5)

def odometry_mc(sig_d=0.0, bias_d=0.0, sig_phi=0.0):
    """직진 주행. 반환: 위치 오차 |Δp|(t)의 MC 평균."""
    err = np.zeros(n_o)
    for _ in range(n_mc):
        dd = v*dt_o*(1 + bias_d) + sig_d*rng3.standard_normal(n_o)
        dphi = sig_phi*rng3.standard_normal(n_o)
        phi = np.cumsum(dphi)
        x = np.cumsum(dd*np.cos(phi)); y = np.cumsum(dd*np.sin(phi))
        err += np.hypot(x - v*t_o, y)
    return err/n_mc

cases = {
    'A 무편향 거리 노이즈 (σ_d=2mm/step)': odometry_mc(sig_d=0.002),
    'B 바퀴 반경 +0.5% 바이어스': odometry_mc(bias_d=0.005),
    'C 방위 노이즈 (σ_φ=0.05°/step)': odometry_mc(sig_phi=np.radians(0.05)),
}
print(f"{'케이스':<38} {'60초 오차':>10} {'600초 오차':>10} {'log-log 기울기':>13}")
slopes = {}
for name, e in cases.items():
    sl = np.polyfit(np.log(t_o[n_o//10:]), np.log(e[n_o//10:]), 1)[0]
    slopes[name] = sl
    print(f"{name:<38} {e[int(60/dt_o)-1]:>9.3f} m {e[-1]:>9.3f} m {sl:>13.2f}")

# =====================================================================
# 그림 1: 한 장 요약 — 주파수 분리 + 시계열
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
f = np.logspace(-3, 1.3, 400)
tau_p = tau_opt
LP = 1/np.sqrt(1 + (2*np.pi*f*tau_p)**2)
HP = (2*np.pi*f*tau_p)/np.sqrt(1 + (2*np.pi*f*tau_p)**2)
ax = axes[0]
ax.semilogx(f, LP, color='tab:blue', lw=2.2, label='저역통과 → 가속도계 (절대 기준, 저주파 신뢰)')
ax.semilogx(f, HP, color='tab:red', lw=2.2, label='고역통과 → 자이로 적분 (드리프트, 고주파 신뢰)')
ax.semilogx(f, np.sqrt(LP**2+HP**2)*0+1, 'k--', lw=1, alpha=0.5)
ax.axvline(1/(2*np.pi*tau_p), color='gray', ls=':', lw=1.5)
ax.annotate(f'교차 주파수 $f_c$ = {1/(2*np.pi*tau_p):.2f} Hz\n(α = {a_opt:.3f}, τ = {tau_p:.1f} s)',
            xy=(1/(2*np.pi*tau_p), 0.72), xytext=(0.004, 0.62), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.text(0.0015, 1.03, 'LP + HP = 1 (전 주파수에서 이득 보존)', fontsize=9, color='k')
ax.fill_betweenx([0, 1.1], 1e-3, 1/(2*np.pi*tau_p), color='tab:blue', alpha=0.06)
ax.fill_betweenx([0, 1.1], 1/(2*np.pi*tau_p), 20, color='tab:red', alpha=0.06)
ax.set_xlabel('주파수 [Hz]'); ax.set_ylabel('이득')
ax.set_title('(a) 상보 필터 = 주파수 분리 융합')
ax.set_ylim(0, 1.15); ax.set_xlim(1.5e-3, 20)
ax.legend(fontsize=8.5, loc='center right'); ax.grid(alpha=0.3)

ax = axes[1]
seg = slice(0, int(30/dt))
ax.plot(t[seg], th_gyro_only[seg], color='tab:red', lw=1.2, label='자이로 적분만 (바이어스 드리프트)')
ax.plot(t[seg], th_acc[seg], color='tab:blue', lw=0.5, alpha=0.45, label='가속도계만 (노이즈+운동 가속 오염)')
ax.plot(t[seg], th_true[seg], 'k', lw=1.6, label='참 기울기')
ax.plot(t[seg], th_cf[seg], color='tab:green', lw=1.1, label=f'상보 필터 (RMSE {rmse(th_cf, th_true)*1e3:.0f} mrad)')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('기울기 θ [rad]')
ax.set_title('(b) 단독 센서는 둘 다 실패, 융합은 성공')
ax.legend(fontsize=8.5, loc='upper left'); ax.grid(alpha=0.3)
fig.suptitle('상보 필터: 두 나쁜 센서를 주파수로 갈라 하나의 좋은 센서로', y=1.00, fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_complementary_summary.png", dpi=140, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 그림 2: α 스윕
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax = axes[0]
ax.semilogx(1-alpha_fine, rmse_fine*1e3, 'o-', ms=3, color='tab:purple')
ax.axvline(1-a_opt, color='tab:green', ls='--', lw=1.2)
ax.annotate(f'최적 α = {a_opt:.4f}\nRMSE = {rmse_fine[i_opt]*1e3:.0f} mrad',
            xy=(1-a_opt, rmse_fine[i_opt]*1e3), xytext=(3e-3, 130), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='tab:green'))
ax.text(4e-4, 210, '← 자이로 쪽으로\n바이어스 누설 τ·b 증가', fontsize=9, color='tab:red', ha='center')
ax.text(0.12, 210, '가속도계 쪽으로 →\n운동 가속 오염 통과', fontsize=9, color='tab:blue', ha='center')
ax.set_xlabel('1 − α (가속도계 가중, 로그축)'); ax.set_ylabel('RMSE [mrad]')
ax.set_title('(a) 컷오프의 트레이드오프: U자 곡선')
ax.grid(alpha=0.3, which='both')

ax = axes[1]
seg = slice(int(20/dt), int(40/dt))
for a, col, nm in [(0.9, 'tab:blue', 'α=0.9 (가속도계 과신)'),
                   (a_opt, 'tab:green', f'α={a_opt:.3f} (최적)'),
                   (0.9999, 'tab:red', 'α=0.9999 (자이로 과신)')]:
    ax.plot(t[seg], (complementary(a)-th_true)[seg]*1e3, color=col, lw=0.9, label=nm)
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('시간 [s]'); ax.set_ylabel('추정 오차 [mrad]')
ax.set_title('(b) 같은 데이터, 세 가지 α의 오차')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_alpha_sweep.png", dpi=140, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 그림 3: EKF vs CF
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax = axes[0]
ax.plot(t, (th_cf-th_true)*1e3, color='tab:green', lw=0.7,
        label=f'상보 필터 (RMSE {rmse(th_cf, th_true)*1e3:.0f} mrad)')
ax.plot(t, (ekf[:, 0]-th_true)*1e3, color='tab:orange', lw=0.7,
        label=f'EKF (RMSE {rmse(ekf[:,0], th_true)*1e3:.0f} mrad)')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('시간 [s]'); ax.set_ylabel('추정 오차 [mrad]')
ax.set_title('(a) 오차 비교: EKF는 바이어스를 배운다')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t, ekf[:, 1]*1e3, color='tab:orange', lw=1.4)
ax.axhline(bias_g*1e3, color='k', ls='--', lw=1.2, label=f'참 바이어스 {bias_g*1e3:.0f} mrad/s')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('바이어스 추정 [mrad/s]')
ax.set_title('(b) 관측하지 않는 상태(바이어스)의 추정')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(init_errs)))
for e_deg, col in zip(init_errs, cols):
    est = ekf_init_runs[e_deg]
    ax.plot(t[:int(20/dt)], np.abs(est[:int(20/dt), 0]-th_true[:int(20/dt)]),
            color=col, lw=1.2, label=f'초기 오차 {e_deg}°')
ax.set_yscale('log')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('|추정 오차| [rad], 로그')
ax.set_title('(c) 선형화 지점의 위험: 초기 오차 스윕')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_ekf_vs_cf.png", dpi=140, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 그림 4: 모멘텀 옵저버
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax = axes[0]
tt = lg50['t']
ax.plot(tt, lg50['tau_ext'][:, 0], 'k--', lw=1.4, label='참 τ_ext,1 (미지)')
ax.plot(tt, lg50['r'][:, 0], color='tab:red', lw=1.0, label='잔차 $r_1$ (K_O=50)')
ax.plot(tt, lg50['r'][:, 1], color='tab:blue', lw=1.0, alpha=0.8, label='잔차 $r_2$')
ax.plot(tt, lg50['tau_ext'][:, 1], 'k:', lw=1.2, label='참 τ_ext,2')
quiet50 = lg50['r'][int(1.0/1e-3):int(1.9/1e-3), 0].std()
ax.axhline(5*quiet50, color='gray', ls='-.', lw=1, label=f'감지 임계 5σ = {5*quiet50:.3f}')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('토크 [N·m]')
ax.set_title('(a) 충돌 잔차 스파이크 (t=2.0~2.15 s 충돌)')
ax.set_xlim(1.5, 3.0); ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

ax = axes[1]
for K, col in zip(KO_list, ['tab:green', 'tab:red', 'tab:purple']):
    ax.plot(logs[K]['t'], logs[K]['r'][:, 0], color=col, lw=1.0,
            label=f'K_O={K:.0f} (τ={1e3/K:.0f} ms)')
ax.plot(tt, lg50['tau_ext'][:, 0], 'k--', lw=1.2, label='참 τ_ext,1')
ax.set_xlim(1.95, 2.45); ax.set_xlabel('시간 [s]'); ax.set_ylabel('$r_1$ [N·m]')
ax.set_title('(b) K_O 스윕: 빠른 감지 vs 노이즈 증폭')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(tt, lg50['naive'][:, 0], color='tab:gray', lw=0.5,
        label=f'역동역학+$\\ddot{{q}}$ 수치미분 (σ={naive_std:.0f} N·m)')
ax.plot(tt, lg50['r'][:, 0], color='tab:red', lw=1.2,
        label=f'모멘텀 옵저버 (σ={mo_std:.3f} N·m)')
ax.plot(tt, lg50['tau_ext'][:, 0], 'k--', lw=1.2, label='참 τ_ext,1')
ax.set_xlim(1.5, 3.0); ax.set_xlabel('시간 [s]'); ax.set_ylabel('τ_ext 추정 [N·m]')
ax.set_title('(c) 왜 모멘텀인가: $\\ddot{q}$ 미분은 노이즈 폭발')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_momentum_observer.png", dpi=140, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 그림 5: 오도메트리 드리프트
# =====================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for (name, e), col in zip(cases.items(), ['tab:blue', 'tab:red', 'tab:purple']):
    ax.loglog(t_o, e, color=col, lw=1.8, label=f'{name} — 기울기 {slopes[name]:.2f}')
for sl, x0, y0, lab in [(0.5, 20, 0.03, r'$\propto \sqrt{t}$'), (1.0, 20, 0.09, r'$\propto t$'),
                        (1.5, 20, 0.0003, r'$\propto t^{1.5}$')]:
    xs = np.array([x0, x0*8])
    ax.loglog(xs, y0*(xs/x0)**sl, 'k:', lw=1)
    ax.text(xs[-1]*1.1, y0*8**sl, lab, fontsize=9)
ax.set_xlabel('시간 [s] (로그)'); ax.set_ylabel('위치 오차 [m] (로그, MC 300회 평균)')
ax.set_title('오도메트리 드리프트의 성장 법칙 — 무엇이 오차를 키우는가')
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_odometry_drift.png", dpi=140, bbox_inches='tight')
plt.close(fig)

print("\n그림 5장 저장 완료:", OUT)
