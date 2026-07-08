# Lec R19 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy/matplotlib 필요)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# ---------- 2링크 파라미터 (R10 Worked Example과 동일: 균일 막대 2개) ----------
m1, m2, l1, l2, lc1, lc2 = 1.0, 1.0, 1.0, 1.0, 0.5, 0.5
I1, I2 = m1*l1**2/12, m2*l2**2/12
grav = 9.81
NOM = (m1, m2, I1, I2)

def M_mat(q, p=NOM):
    _m1, _m2, _I1, _I2 = p
    c2 = np.cos(q[1])
    return np.array([[_I1+_I2+_m1*lc1**2+_m2*(l1**2+lc2**2+2*l1*lc2*c2),
                      _I2+_m2*(lc2**2+l1*lc2*c2)],
                     [_I2+_m2*(lc2**2+l1*lc2*c2), _I2+_m2*lc2**2]])

def C_mat(q, qd, p=NOM):
    h = p[1]*l1*lc2*np.sin(q[1])
    return np.array([[-h*qd[1], -h*(qd[0]+qd[1])], [h*qd[0], 0.0]])

def g_vec(q, p=NOM):
    _m1, _m2 = p[0], p[1]
    c1, c12 = np.cos(q[0]), np.cos(q[0]+q[1])
    return grav*np.array([(_m1*lc1+_m2*l1)*c1 + _m2*lc2*c12, _m2*lc2*c12])

def fk(q):
    return np.array([l1*np.cos(q[0])+l2*np.cos(q[0]+q[1]),
                     l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])])

# ---------- 기준 궤적: 원 (중심 (1.1, 0.3), 반경 0.3, 주기 2 s, 2바퀴) ----------
cx, cy, R, T_lap = 1.1, 0.3, 0.3, 2.0
w = 2*np.pi/T_lap
dt = 1e-3; T_end = 4.0
ts = np.arange(0, T_end+dt, dt)
xs, ys = cx + R*np.cos(w*ts), cy + R*np.sin(w*ts)

def ik2r(x, y):
    c2 = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    return np.array([q1, q2])

q_traj = np.array([ik2r(x, y) for x, y in zip(xs, ys)])   # 목표 관절 위치
qd_traj = np.gradient(q_traj, dt, axis=0)                  # 목표 관절 속도
qdd_traj = np.gradient(qd_traj, dt, axis=0)                # 목표 관절 가속도

Kp, Kd = np.diag([100., 100.]), np.diag([20., 20.])

def simulate(controller, p_ctrl=NOM):
    q, qdv = q_traj[0].copy(), qd_traj[0].copy()
    Q = np.zeros((len(ts), 2))
    for i in range(len(ts)):
        Q[i] = q
        tau = controller(i, q, qdv, p_ctrl)
        def acc(qq, vv):
            return np.linalg.solve(M_mat(qq), tau - C_mat(qq, vv)@vv - g_vec(qq))
        k1q, k1v = qdv, acc(q, qdv)
        k2q, k2v = qdv+dt/2*k1v, acc(q+dt/2*k1q, qdv+dt/2*k1v)
        k3q, k3v = qdv+dt/2*k2v, acc(q+dt/2*k2q, qdv+dt/2*k2v)
        k4q, k4v = qdv+dt*k3v, acc(q+dt*k3q, qdv+dt*k3v)
        q = q + dt/6*(k1q+2*k2q+2*k3q+k4q)
        qdv = qdv + dt/6*(k1v+2*k2v+2*k3v+k4v)
    return Q

def ctrl_pd(i, q, qdv, p):
    return Kp@(q_traj[i]-q) + Kd@(qd_traj[i]-qdv)

def ctrl_gcomp(i, q, qdv, p):
    return g_vec(q, p) + Kp@(q_traj[i]-q) + Kd@(qd_traj[i]-qdv)

def ctrl_ct(i, q, qdv, p):
    e, ed = q_traj[i]-q, qd_traj[i]-qdv
    return M_mat(q, p)@(qdd_traj[i] + Kd@ed + Kp@e) + C_mat(q, qdv, p)@qdv + g_vec(q, p)

def eef_err_mm(Q):
    P = np.array([fk(q) for q in Q])
    return np.linalg.norm(P - np.stack([xs, ys], axis=1), axis=1)*1000

# ---------- 그림 1 (한 장 요약): 세 제어기의 원 추종 ----------
labels = ['독립 관절 PD', '중력보상 + PD', 'computed torque']
colors = ['C3', 'C1', 'C0']
Qs = [simulate(c) for c in (ctrl_pd, ctrl_gcomp, ctrl_ct)]
errs = [eef_err_mm(Q) for Q in Qs]
lap2 = ts >= T_lap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax1.plot(xs, ys, 'k--', lw=2, label='기준 원 궤적', zorder=5)
for Q, lab, c in zip(Qs, labels, colors):
    P = np.array([fk(q) for q in Q])
    rms = np.sqrt(np.mean(eef_err_mm(Q)[lap2]**2))
    ax1.plot(P[:, 0], P[:, 1], c, lw=1.6, alpha=0.9,
             label=f'{lab} (RMS {rms:.3g} mm)')
q0 = q_traj[0]
ax1.plot([0, l1*np.cos(q0[0]), fk(q0)[0]], [0, l1*np.sin(q0[0]), fk(q0)[1]],
         'o-', color='gray', lw=3, ms=6, alpha=0.6)
ax1.plot(0, 0, 'ks', ms=9)
ax1.set_aspect('equal'); ax1.grid(alpha=0.3)
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('(a) 같은 게인($K_p$=100, $K_d$=20), 세 가지 모델 지식')
ax1.legend(fontsize=8.5, loc='upper left')

for err, lab, c in zip(errs, labels, colors):
    ax2.semilogy(ts, np.maximum(err, 1e-4), c, lw=1.6, label=lab)
ax2.axvspan(0, T_lap, color='gray', alpha=0.12)
ax2.text(0.95, 2e-4, '1바퀴째(과도)', fontsize=8.5, ha='center', color='gray')
ax2.set_xlabel('시간 t [s]'); ax2.set_ylabel('EEF 추종 오차 [mm] (log)')
ax2.set_title('(b) 추종 오차 — 세 제어기는 자릿수가 다르다')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig1_three_controllers.png'), dpi=140)
plt.close(fig)

# ---------- 그림 2: 독립 관절 PID의 문제 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 유효 관성·감쇠비의 자세 의존
q2s = np.linspace(-np.pi, np.pi, 300)
M11s = np.array([M_mat(np.array([0., v]))[0, 0] for v in q2s])
zeta = 20/(2*np.sqrt(100*M11s))
axa = ax1
axa.plot(q2s, M11s, 'C0', lw=2, label='$M_{11}(q_2)$ [kg·m²]')
axa.set_xlabel('$q_2$ [rad]'); axa.set_ylabel('유효 관성 $M_{11}$', color='C0')
axa.tick_params(axis='y', labelcolor='C0')
axb = axa.twinx()
axb.plot(q2s, zeta, 'C3', lw=2, label='유효 감쇠비 $\\zeta_{eff}$')
axb.axhline(1.0, color='C3', ls=':', lw=1)
axb.set_ylabel('유효 감쇠비 $\\zeta_{eff}$', color='C3')
axb.tick_params(axis='y', labelcolor='C3')
axa.set_title('(a) 게인은 고정인데 플랜트가 변한다:\n$q_2$에 따라 $\\zeta_{eff}$ 0.61~1.22 (2배)')
axa.grid(alpha=0.3)

# (b) 셋포인트 유지: PD 처짐 / PID 느린 회복 / 중력보상 즉시
def simulate_setpoint(mode, T=4.0, Ki=150.0):
    q, qdv = np.array([0., 0.]), np.zeros(2)
    ei = np.zeros(2)
    tt = np.arange(0, T+dt, dt)
    tip = np.zeros(len(tt))
    for i in range(len(tt)):
        tip[i] = fk(q)[1]
        e, ed = -q, -qdv
        tau = Kp@e + Kd@ed
        if mode == 'pid':
            ei += e*dt; tau += Ki*ei
        elif mode == 'gcomp':
            tau += g_vec(q)
        def acc(qq, vv):
            return np.linalg.solve(M_mat(qq), tau - C_mat(qq, vv)@vv - g_vec(qq))
        k1q, k1v = qdv, acc(q, qdv)
        k2q, k2v = qdv+dt/2*k1v, acc(q+dt/2*k1q, qdv+dt/2*k1v)
        k3q, k3v = qdv+dt/2*k2v, acc(q+dt/2*k2q, qdv+dt/2*k2v)
        k4q, k4v = qdv+dt*k3v, acc(q+dt*k3q, qdv+dt*k3v)
        q = q + dt/6*(k1q+2*k2q+2*k3q+k4q)
        qdv = qdv + dt/6*(k1v+2*k2v+2*k3v+k4v)
    return tt, tip

for mode, lab, c in [('pd', 'PD (처진 채 평형)', 'C3'),
                     ('pid', 'PID ($K_i$=150, 느리게 회복)', 'C2'),
                     ('gcomp', '중력보상 + PD (처지지 않음)', 'C0')]:
    tt, tip = simulate_setpoint(mode)
    ax2.plot(tt, tip*100, c, lw=1.8, label=lab)
ax2.axhline(0, color='k', ls='--', lw=1, label='목표 (수평, y=0)')
ax2.set_xlabel('시간 t [s]'); ax2.set_ylabel('손끝 높이 $y_{ee}$ [cm]')
ax2.set_title('(b) 수평 자세 유지 명령: 중력이 만드는 정상상태 오차\nPD는 손끝이 43 cm 처진 채 평형')
ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5, loc='center right')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig2_pid_problems.png'), dpi=140)
plt.close(fig)

# ---------- 그림 3: 오차 동역학의 극점 배치 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
Kp_s = 100.0
cases = [(10.0, 'C3'), (20.0, 'C0'), (40.0, 'C2')]
th = np.linspace(0, 2*np.pi, 200)
ax1.plot(10*np.cos(th), 10*np.sin(th), 'k:', lw=1, alpha=0.6)
ax1.text(-24, 11.0, '점선 원: $|s|=\\omega_n=\\sqrt{K_p}=10$', fontsize=8.5)
for Kd_s, c in cases:
    poles = np.roots([1, Kd_s, Kp_s])
    zt = Kd_s/(2*np.sqrt(Kp_s))
    ax1.plot(poles.real, poles.imag, 'x', color=c, ms=12, mew=3,
             label=f'$K_d$={Kd_s:.0f} ($\\zeta$={zt:.1f})')
ax1.axvline(0, color='k', lw=1); ax1.axhline(0, color='k', lw=1)
ax1.set_xlim(-42, 8); ax1.set_ylim(-13, 13)
ax1.set_xlabel('Re(s) [1/s]'); ax1.set_ylabel('Im(s) [1/s]')
ax1.set_title('(a) $s^2+K_d s+K_p=0$의 극점 — 게인이 곧 극점')
ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc='lower left')

tt = np.linspace(0, 1.2, 400)
for Kd_s, c in cases:
    zt = Kd_s/(2*np.sqrt(Kp_s)); wn = np.sqrt(Kp_s)
    if zt < 1:
        wd = wn*np.sqrt(1-zt**2)
        e = np.exp(-zt*wn*tt)*(np.cos(wd*tt) + zt*wn/wd*np.sin(wd*tt))
    elif zt == 1:
        e = (1 + wn*tt)*np.exp(-wn*tt)
    else:
        r1, r2 = np.roots([1, Kd_s, Kp_s])
        A = r2/(r2-r1); B = -r1/(r2-r1)
        e = A*np.exp(r1*tt) + B*np.exp(r2*tt)
    ax2.plot(tt, e.real, color=c, lw=2,
             label=f'$K_d$={Kd_s:.0f}: $\\zeta$={zt:.1f}' +
                   (' (임계감쇠)' if zt == 1 else ''))
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_xlabel('시간 t [s]'); ax2.set_ylabel('오차 $e(t)/e(0)$')
ax2.set_title('(b) computed torque 후의 오차 응답 —\n어느 자세·속도에서도 이 곡선 그대로')
ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig3_error_dynamics_poles.png'), dpi=140)
plt.close(fig)

# ---------- 그림 4: 모델 오차 민감도 ----------
alphas = np.linspace(0.7, 1.3, 25)
def pset(a): return (a*m1, a*m2, a*I1, a*I2)
rms_ct, rms_gc = [], []
for a in alphas:
    rms_ct.append(np.sqrt(np.mean(eef_err_mm(simulate(ctrl_ct, pset(a)))[lap2]**2)))
    rms_gc.append(np.sqrt(np.mean(eef_err_mm(simulate(ctrl_gcomp, pset(a)))[lap2]**2)))
rms_pd = np.sqrt(np.mean(errs[0][lap2]**2))

fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.semilogy(alphas, rms_ct, 'C0o-', lw=2, ms=4, label='computed torque')
ax.semilogy(alphas, rms_gc, 'C1s-', lw=2, ms=4, label='중력보상 + PD')
ax.axhline(rms_pd, color='C3', ls='--', lw=2, label=f'독립 관절 PD (모델 불사용, {rms_pd:.0f} mm)')
ax.axvline(1.0, color='gray', ls=':', lw=1)
i12 = np.argmin(np.abs(alphas-1.2))
ax.annotate(f'질량 +20%:\n{rms_ct[i12]:.1f} mm (~270배 악화,\n그래도 중력보상 PD보다 낫다)',
            xy=(1.2, rms_ct[i12]), xytext=(1.02, 1.2), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
i10 = np.argmin(np.abs(alphas-1.0))
ax.annotate(f'완전 모델: {rms_ct[i10]:.3f} mm', xy=(1.0, rms_ct[i10]),
            xytext=(0.73, 0.35), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('제어기 모델의 질량 스케일 $\\alpha$ ($\\hat m_i = \\alpha\\, m_i$, 진실은 $\\alpha$=1)')
ax.set_ylabel('EEF 추종 RMS 오차 [mm] (log)')
ax.set_title('모델 기반의 대가: 모델이 틀린 만큼 성능이 무너진다')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9, loc='lower right')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig4_model_error_sensitivity.png'), dpi=140)
plt.close(fig)

print("figures written")
print(f"fig1 RMS(2nd lap): PD={np.sqrt(np.mean(errs[0][lap2]**2)):.3f}, "
      f"gcomp={np.sqrt(np.mean(errs[1][lap2]**2)):.3f}, "
      f"CT={np.sqrt(np.mean(errs[2][lap2]**2)):.4f} mm")

# ---------- 본문 수치 검증 (그림 없음) ----------
# (1) 본문 §3: 셋포인트 CT의 오차가 임계감쇠 해석해 e0(1+ωn t)e^{-ωn t}와 겹치는가
def ct_setpoint_check(dt2=1e-4, T2=1.2):
    ts2 = np.arange(0, T2+dt2, dt2)
    q_d, e0 = np.array([0.5, 0.8]), np.array([0.1, -0.05])
    q, qdv = q_d - e0, np.zeros(2)
    E = np.zeros((len(ts2), 2))
    for i in range(len(ts2)):
        E[i] = q_d - q
        tau = M_mat(q)@(Kd@(-qdv) + Kp@(q_d-q)) + C_mat(q, qdv)@qdv + g_vec(q)
        def acc(qq, vv):
            return np.linalg.solve(M_mat(qq), tau - C_mat(qq, vv)@vv - g_vec(qq))
        k1q, k1v = qdv, acc(q, qdv)
        k2q, k2v = qdv+dt2/2*k1v, acc(q+dt2/2*k1q, qdv+dt2/2*k1v)
        k3q, k3v = qdv+dt2/2*k2v, acc(q+dt2/2*k2q, qdv+dt2/2*k2v)
        k4q, k4v = qdv+dt2*k3v, acc(q+dt2*k3q, qdv+dt2*k3v)
        q = q + dt2/6*(k1q+2*k2q+2*k3q+k4q)
        qdv = qdv + dt2/6*(k1v+2*k2v+2*k3v+k4v)
    ana = np.outer((1 + 10*ts2)*np.exp(-10*ts2), [1, 1]) * e0
    return np.abs(E - ana).max()

print(f"E3 검증: |E_sim - 해석해|_max = {ct_setpoint_check():.3g} rad")

# (2) 흔한 오해 1·실습 5: 고게인 PD (Kd = 2√Kp 임계감쇠 유지)
for s in (10, 100):
    Kp, Kd = np.diag([100.*s]*2), np.diag([2*np.sqrt(100.*s)]*2)
    r = np.sqrt(np.mean(eef_err_mm(simulate(ctrl_pd))[lap2]**2))
    print(f"고게인 PD Kp={100*s:.0f}: RMS = {r:.2f} mm")
Kp, Kd = np.diag([100., 100.]), np.diag([20., 20.])
