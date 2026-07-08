# Lec R20 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy/matplotlib 필요)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# ---------- 2링크 파라미터 (R10·R19와 동일: 균일 막대 2개) ----------
m1, m2, l1, l2, lc1, lc2 = 1.0, 1.0, 1.0, 1.0, 0.5, 0.5
I1, I2 = m1*l1**2/12, m2*l2**2/12
grav = 9.81

def M_mat(q):
    c2 = np.cos(q[1])
    return np.array([[I1+I2+m1*lc1**2+m2*(l1**2+lc2**2+2*l1*lc2*c2),
                      I2+m2*(lc2**2+l1*lc2*c2)],
                     [I2+m2*(lc2**2+l1*lc2*c2), I2+m2*lc2**2]])

def C_mat(q, qd):
    h = m2*l1*lc2*np.sin(q[1])
    return np.array([[-h*qd[1], -h*(qd[0]+qd[1])], [h*qd[0], 0.0]])

def g_vec(q):
    c1, c12 = np.cos(q[0]), np.cos(q[0]+q[1])
    return grav*np.array([(m1*lc1+m2*l1)*c1 + m2*lc2*c12, m2*lc2*c12])

def fk(q):
    return np.array([l1*np.cos(q[0])+l2*np.cos(q[0]+q[1]),
                     l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])])

def jac(q):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([[-l1*s1 - l2*s12, -l2*s12],
                     [ l1*c1 + l2*c12,  l2*c12]])

def jacdot_qd(q, qd):           # J̇(q,q̇) q̇  (해석식)
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    w1, w12 = qd[0], qd[0]+qd[1]
    Jd = np.array([[-l1*c1*w1 - l2*c12*w12, -l2*c12*w12],
                   [-l1*s1*w1 - l2*s12*w12, -l2*s12*w12]])
    return Jd @ qd

def Lambda(q, damp=0.0):        # 태스크 공간 관성 (E2). damp>0면 DLS판 (R07)
    J = jac(q)
    A = J @ np.linalg.solve(M_mat(q), J.T)
    return np.linalg.inv(A + damp*np.eye(2))

def ik2r(x, y):
    c2 = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    return np.array([q1, q2])

# ---------- 공용 시뮬레이터 (RK4, 제어 zero-order hold, R19와 동일 구조) ----------
def simulate(controller, q0, qd0, T_end, dt=1e-3, tau_max=None, blow=1e6):
    ts = np.arange(0, T_end+dt, dt)
    Q, QD, TAU = np.zeros((len(ts), 2)), np.zeros((len(ts), 2)), np.zeros((len(ts), 2))
    q, qdv = q0.copy(), qd0.copy()
    for i, t in enumerate(ts):
        Q[i], QD[i] = q, qdv
        tau_cmd = controller(t, q, qdv)
        TAU[i] = tau_cmd
        tau = np.clip(tau_cmd, -tau_max, tau_max) if tau_max else tau_cmd
        if not np.all(np.isfinite(tau)) or np.abs(tau).max() > blow or np.abs(qdv).max() > 1e3:
            Q[i:], QD[i:] = q, qdv          # 폭발: 이후 값 고정하고 종료
            TAU[i:] = TAU[i]
            return ts, Q, QD, TAU, i        # i = 폭발 시점 인덱스
        def acc(qq, vv):
            return np.linalg.solve(M_mat(qq), tau - C_mat(qq, vv)@vv - g_vec(qq))
        k1q, k1v = qdv, acc(q, qdv)
        k2q, k2v = qdv+dt/2*k1v, acc(q+dt/2*k1q, qdv+dt/2*k1v)
        k3q, k3v = qdv+dt/2*k2v, acc(q+dt/2*k2q, qdv+dt/2*k2v)
        k4q, k4v = qdv+dt*k3v, acc(q+dt*k3q, qdv+dt*k3v)
        q = q + dt/6*(k1q+2*k2q+2*k3q+k4q)
        qdv = qdv + dt/6*(k1v+2*k2v+2*k3v+k4v)
    return ts, Q, QD, TAU, None

# ---------- 기준 궤적: 원 (R19와 동일: 중심 (1.1,0.3), 반경 0.3) ----------
cx, cy, Rc = 1.1, 0.3, 0.3

def make_circle(T_lap):
    w = 2*np.pi/T_lap
    xd  = lambda t: np.array([cx + Rc*np.cos(w*t), cy + Rc*np.sin(w*t)])
    xdd = lambda t: np.array([-Rc*w*np.sin(w*t),  Rc*w*np.cos(w*t)])
    xddd= lambda t: np.array([-Rc*w*w*np.cos(w*t), -Rc*w*w*np.sin(w*t)])
    return xd, xdd, xddd

# 게인: 작업공간 오차 동역학용 (R19와 같은 극점: s = -10 이중근)
Kp_t, Kd_t = 100.0, 20.0            # [1/s^2], [1/s]  — Λ가 곱해져 단위가 맞는다
KpJ, KdJ = 500.0, 30.0              # Jᵀ 제어용 [N/m], [N·s/m]

def ctrl_opspace(xd, xdd, xddd, damp=0.0):
    def ctrl(t, q, qdv):
        J = jac(q)
        x, xdot = fk(q), J @ qdv
        e, ed = xd(t) - x, xdd(t) - xdot
        a_x = xddd(t) + Kd_t*ed + Kp_t*e
        L = Lambda(q, damp)
        Minv = np.linalg.inv(M_mat(q))
        mu_p = L @ J @ Minv @ (C_mat(q, qdv)@qdv + g_vec(q)) - L @ jacdot_qd(q, qdv)
        F = L @ a_x + mu_p
        return J.T @ F
    return ctrl

def ctrl_jt(xd):
    def ctrl(t, q, qdv):
        J = jac(q)
        e = xd(t) - fk(q)
        xdot = J @ qdv
        return J.T @ (KpJ*e - KdJ*xdot) + g_vec(q)
    return ctrl

# 관절 PD (R19 재현): IK로 만든 관절 궤적을 추종
def make_joint_pd(T_lap, T_end, dt=1e-3):
    ts = np.arange(0, T_end+dt, dt)
    xd, _, _ = make_circle(T_lap)
    q_traj = np.array([ik2r(*xd(t)) for t in ts])
    qd_traj = np.gradient(q_traj, dt, axis=0)
    Kp, Kd = np.diag([100., 100.]), np.diag([20., 20.])
    def ctrl(t, q, qdv):
        i = min(int(round(t/dt)), len(ts)-1)
        return Kp@(q_traj[i]-q) + Kd@(qd_traj[i]-qdv)
    return ctrl, q_traj[0], qd_traj[0]

def eef_err_mm(ts, Q, xd):
    P = np.array([fk(q) for q in Q])
    ref = np.array([xd(t) for t in ts])
    return np.linalg.norm(P - ref, axis=1)*1000

def rms(v): return float(np.sqrt(np.mean(v**2)))

# ---------- 그림 1 (한 장 요약): 관절 PD vs Jᵀ vs op-space ----------
T_lap, T_end = 2.0, 4.0
xd, xdd, xddd = make_circle(T_lap)
q0 = ik2r(*xd(0.0)); qd0 = np.linalg.solve(jac(q0), xdd(0.0))

ctrl_pd, q0pd, qd0pd = make_joint_pd(T_lap, T_end)
runs = {}
runs['joint_pd'] = simulate(ctrl_pd, q0pd, qd0pd, T_end)
runs['jt']       = simulate(ctrl_jt(xd), q0, qd0, T_end)
runs['ops']      = simulate(ctrl_opspace(xd, xdd, xddd), q0, qd0, T_end)

labels = ['관절 PD (R19의 그 제어기)', '$J^\\top$ 제어 (E3)', 'op-space 제어 (E1)']
colors = ['C3', 'C2', 'C0']
lap2 = runs['ops'][0] >= T_lap
RMS = {}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
th = np.linspace(0, 2*np.pi, 200)
ax1.plot(cx+Rc*np.cos(th), cy+Rc*np.sin(th), 'k--', lw=2, label='기준 원 궤적', zorder=5)
for key, lab, c in zip(['joint_pd', 'jt', 'ops'], labels, colors):
    ts, Q, _, _, _ = runs[key]
    err = eef_err_mm(ts, Q, xd)
    RMS[key] = rms(err[lap2])
    P = np.array([fk(q) for q in Q])
    ax1.plot(P[:, 0], P[:, 1], c, lw=1.6, alpha=0.9,
             label=f'{lab}\n(RMS {RMS[key]:.3g} mm)')
ax1.plot([0, l1*np.cos(q0[0]), fk(q0)[0]], [0, l1*np.sin(q0[0]), fk(q0)[1]],
         'o-', color='gray', lw=3, ms=6, alpha=0.6)
ax1.plot(0, 0, 'ks', ms=9)
ax1.set_aspect('equal'); ax1.grid(alpha=0.3)
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('(a) 오차를 어느 공간에서 정의하는가')
ax1.legend(fontsize=8, loc='upper left')
for key, lab, c in zip(['joint_pd', 'jt', 'ops'], labels, colors):
    ts, Q, _, _, _ = runs[key]
    err = eef_err_mm(ts, Q, xd)
    ax2.semilogy(ts, np.maximum(err, 1e-4), c, lw=1.6, label=lab)
ax2.axvspan(0, T_lap, color='gray', alpha=0.12)
ax2.text(0.95, 2e-4, '1바퀴째(과도)', fontsize=8.5, ha='center', color='gray')
ax2.set_xlabel('시간 t [s]'); ax2.set_ylabel('EEF 추종 오차 [mm] (log)')
ax2.set_title('(b) 추종 오차 — $J^\\top$는 정적으론 수렴하지만 동적으론 뒤처진다')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig1_three_controllers.png'), dpi=140)
plt.close(fig)
print(f"fig1 RMS(2바퀴째): 관절PD={RMS['joint_pd']:.1f}, Jᵀ={RMS['jt']:.1f}, "
      f"op-space={RMS['ops']:.4f} mm")

# ---------- 그림 2: 조작성 타원(운동학) vs 태스크 관성 타원(동역학) ----------
def ellipse_pts(A_axes, center, scale):
    # A_axes: (반축벡터1, 반축벡터2) — 이미 스케일된 축
    th = np.linspace(0, 2*np.pi, 100)
    pts = np.outer(np.cos(th), A_axes[0]) + np.outer(np.sin(th), A_axes[1])
    return center[0] + scale*pts[:, 0], center[1] + scale*pts[:, 1]

postures = [np.array([0.9, 2.4]), np.array([0.5, 1.571]), np.array([0.25, 0.9]),
            np.array([1.35, 1.2]), np.array([-0.1, 2.0])]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
for ax, mode in zip(axes, ['manip', 'mass']):
    ax.plot(2*np.cos(th), 2*np.sin(th), 'r--', lw=1, alpha=0.6)
    ax.text(1.15, -1.35, '특이점 경계\n(팔을 다 편 자세)', fontsize=8, color='r')
    for q in postures:
        p = fk(q)
        ax.plot([0, l1*np.cos(q[0]), p[0]], [0, l1*np.sin(q[0]), p[1]],
                'o-', color='gray', lw=2, ms=4, alpha=0.45)
        J = jac(q)
        if mode == 'manip':
            U, S, _ = np.linalg.svd(J)
            axs = (S[0]*U[:, 0], S[1]*U[:, 1])
            xs2, ys2 = ellipse_pts(axs, p, 0.22)
            ax.fill(xs2, ys2, 'C2', alpha=0.35); ax.plot(xs2, ys2, 'C2', lw=1.5)
        else:
            L = Lambda(q)
            lam, V = np.linalg.eigh(L)
            axs = (lam[1]*V[:, 1], lam[0]*V[:, 0])
            xs2, ys2 = ellipse_pts(axs, p, 0.22)
            ax.fill(xs2, ys2, 'C0', alpha=0.35); ax.plot(xs2, ys2, 'C0', lw=1.5)
            ax.text(p[0]+0.06, p[1]+0.12, f'{lam[0]:.2f}~{lam[1]:.2f} kg',
                    fontsize=7.5, color='C0')
    ax.plot(0, 0, 'ks', ms=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlim(-0.9, 2.3); ax.set_ylim(-1.6, 2.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
axes[0].set_title('(a) 조작성 타원 (R06): $J J^\\top$ — 단위 관절속도로\n낼 수 있는 EEF 속도 (운동학만)')
axes[1].set_title('(b) 태스크 관성 타원: $\\Lambda(q)$ 고유값 = 방향별 유효 질량 [kg]\n— 같은 로봇이 방향 따라 다르게 무겁다 (동역학)')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig2_task_inertia_ellipses.png'), dpi=140)
plt.close(fig)

# fig2 관련 수치: 두 타원의 축이 일치하지 않음을 정량화
qx = np.array([0.5, 1.571])
J = jac(qx); U, S, _ = np.linalg.svd(J)
L = Lambda(qx); lam, V = np.linalg.eigh(L)
ang = np.degrees(np.arccos(np.clip(abs(U[:, 1] @ V[:, 1]), 0, 1)))  # 최약 속도축 vs 최대 질량축
print(f"fig2: q={qx}, 조작성 최약축 u_min=({U[0,1]:+.3f},{U[1,1]:+.3f}), "
      f"질량 최대축 v_max=({V[0,1]:+.3f},{V[1,1]:+.3f}), 사이각 {ang:.1f}°")
print(f"      유효 질량 범위(그 자세): {lam[0]:.3f} ~ {lam[1]:.3f} kg (비 {lam[1]/lam[0]:.2f})")
lam_all = []
for q2 in np.linspace(np.radians(10), np.radians(170), 60):
    lam_all.append(np.linalg.eigvalsh(Lambda(np.array([0.0, q2]))))
lam_all = np.array(lam_all)
print(f"      작업공간 내부(q2 10°~170°) 유효 질량 범위: {lam_all.min():.3f} ~ {lam_all.max():.2f} kg")

# ---------- 그림 3: 특이점 접근 — Λ 폭발과 토크 폭발 ----------
q2s = np.radians(np.logspace(np.log10(0.5), np.log10(170), 200))
lam_min = np.array([np.linalg.eigvalsh(Lambda(np.array([0., v])))[0] for v in q2s])
lam_max = np.array([np.linalg.eigvalsh(Lambda(np.array([0., v])))[1] for v in q2s])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
ax1.loglog(np.degrees(q2s), lam_max, 'C0', lw=2, label='$\\lambda_{max}(\\Lambda)$ — 가장 무거운 방향')
ax1.loglog(np.degrees(q2s), lam_min, 'C2', lw=2, label='$\\lambda_{min}(\\Lambda)$ — 가장 가벼운 방향')
for q2deg in [1, 5, 30]:
    lm = np.linalg.eigvalsh(Lambda(np.array([0., np.radians(q2deg)])))[1]
    ax1.plot(q2deg, lm, 'ko', ms=5)
    ax1.annotate(f'{lm:.0f} kg', xy=(q2deg, lm), xytext=(q2deg*1.3, lm*1.5), fontsize=8.5)
ax1.axvline(90, color='gray', ls=':', lw=1)
ax1.set_xlabel('팔꿈치 각 $q_2$ [deg] (0 = 특이점)')
ax1.set_ylabel('$\\Lambda$ 고유값 = 방향별 유효 질량 [kg]')
ax1.set_title('(a) 특이점에 다가가면 유효 질량이 $1/\\sin^2 q_2$로 발산\n(전체 팔 질량은 2 kg뿐인데)')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=9)

# (b) 동적 실험: 목표가 작업공간 경계 바깥으로 나가는 직선
v_line = 0.25
xd_l  = lambda t: np.array([1.5 + v_line*t, 0.0])
xdd_l = lambda t: np.array([v_line, 0.0])
xddd_l= lambda t: np.zeros(2)
q0l = ik2r(*xd_l(0)); q0l = np.array([q0l[0], abs(q0l[1])])
qd0l = np.linalg.solve(jac(q0l), xdd_l(0))
ts_e, Qe, _, TAUe, i_blow = simulate(ctrl_opspace(xd_l, xdd_l, xddd_l), q0l, qd0l, 3.0)
ts_d, Qd, _, TAUd, _ = simulate(ctrl_opspace(xd_l, xdd_l, xddd_l, damp=0.01), q0l, qd0l, 3.0)
tau_e = np.abs(TAUe).max(axis=1)
tau_d = np.abs(TAUd).max(axis=1)
r_of = lambda Q: np.array([np.linalg.norm(fk(q)) for q in Q])
ax2.semilogy(ts_e, np.maximum(tau_e, 1e-2), 'C3', lw=1.8, label='정확한 $\\Lambda$ (E1 그대로)')
ax2.semilogy(ts_d, np.maximum(tau_d, 1e-2), 'C0', lw=1.8, label='감쇠 $\\Lambda_\\delta$ ($\\delta$=0.01, R07의 DLS)')
t_cross = (2.0 - 1.5)/v_line
ax2.axvline(t_cross, color='gray', ls='--', lw=1.2)
ax2.text(t_cross+0.04, 3e3, '목표가 작업공간\n경계(r=2)를 넘는 순간', fontsize=8.5, color='gray')
if i_blow is not None:
    ax2.plot(ts_e[i_blow], tau_e[i_blow], 'C3x', ms=11, mew=3)
    ax2.annotate('수치 폭발', xy=(ts_e[i_blow], tau_e[i_blow]),
                 xytext=(ts_e[i_blow]-0.9, tau_e[i_blow]*0.15), fontsize=9, color='C3',
                 arrowprops=dict(arrowstyle='->', color='C3'))
ax2.set_xlabel('시간 t [s] (목표: 반경 방향 직선, 0.25 m/s)')
ax2.set_ylabel('명령 토크 $\\max_i |\\tau_i|$ [N·m] (log)')
ax2.set_title('(b) 목표가 경계로 갈 때 op-space 토크 명령\n— $\\Lambda$ 폭발이 토크 폭발이 된다')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig3_singularity_explosion.png'), dpi=140)
plt.close(fig)

for q2deg in [90, 30, 5, 1]:
    lm = np.linalg.eigvalsh(Lambda(np.array([0., np.radians(q2deg)])))
    print(f"fig3: q2={q2deg:3d}° → 유효 질량 {lm[0]:.3f} ~ {lm[1]:.1f} kg")
sl = np.polyfit(np.log(q2s[:60]), np.log(lam_max[:60]), 1)[0]
print(f"fig3: 소각 영역 log-log 기울기 = {sl:.3f} (이론: -2)")
print(f"fig3: 정확 Λ 최대 토크 = {tau_e.max():.3g} N·m (폭발 t={ts_e[i_blow] if i_blow is not None else np.nan:.2f}s), "
      f"감쇠 Λ 최대 토크 = {tau_d.max():.1f} N·m")
err_d = np.linalg.norm(np.array([fk(q) for q in Qd]) - np.array([xd_l(t) for t in ts_d]), axis=1)
print(f"fig3: 감쇠판 최종 반경 r={r_of(Qd)[-1]:.4f} (경계 2.0), 최종 목표 이탈 {err_d[-1]*1000:.0f} mm")

# ---------- 그림 4: Jᵀ 제어 — 정적 수렴 vs 동적 대가 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
# (a) 셋포인트: 고정 목표로의 수렴
x_goal = np.array([1.2, 0.8])
xd_s   = lambda t: x_goal
q0s = ik2r(1.6, -0.4)
ts_s, Qs, _, _, _ = simulate(ctrl_jt(xd_s), q0s, np.zeros(2), 4.0)
err_s = eef_err_mm(ts_s, Qs, xd_s)
ax1.semilogy(ts_s, np.maximum(err_s, 1e-5), 'C2', lw=2)
ax1.set_xlabel('시간 t [s]'); ax1.set_ylabel('$\\|x_{goal} - x\\|$ [mm] (log)')
ax1.set_title(f'(a) 셋포인트: $J^\\top$ 제어는 정확히 수렴한다\n(모델은 $J$와 $g$뿐 — 최종 오차 {err_s[-1]:.2g} mm)')
ax1.grid(alpha=0.3, which='both')
# (b) 속도 스윕: 같은 원을 점점 빨리
periods = [16., 8., 4., 2., 1.]
rms_jt, rms_ops = [], []
for Tl in periods:
    xdT, xddT, xdddT = make_circle(Tl)
    q0T = ik2r(*xdT(0)); qd0T = np.linalg.solve(jac(q0T), xddT(0))
    tsT, QT, _, _, _ = simulate(ctrl_jt(xdT), q0T, qd0T, 2*Tl)
    rms_jt.append(rms(eef_err_mm(tsT, QT, xdT)[tsT >= Tl]))
    tsT, QT, _, _, _ = simulate(ctrl_opspace(xdT, xddT, xdddT), q0T, qd0T, 2*Tl)
    rms_ops.append(rms(eef_err_mm(tsT, QT, xdT)[tsT >= Tl]))
speeds = [2*np.pi*Rc/Tl for Tl in periods]
ax2.loglog(speeds, rms_jt, 'C2o-', lw=2, ms=5, label='$J^\\top$ 제어')
ax2.loglog(speeds, rms_ops, 'C0s-', lw=2, ms=5, label='op-space 제어')
for s, r_, Tl in zip(speeds, rms_jt, periods):
    ax2.annotate(f'{r_:.3g}', xy=(s, r_), xytext=(s*0.75, r_*1.7), fontsize=8, color='C2')
ax2.set_xlabel('원 궤적 최대 속도 [m/s]')
ax2.set_ylabel('EEF 추종 RMS [mm] (2바퀴째, log)')
ax2.set_title('(b) 동적 대가: 빨라질수록 $J^\\top$는 뒤처지고\nop-space는 평평하다')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig4_jt_static_vs_dynamic.png'), dpi=140)
plt.close(fig)
print("fig4: Jᵀ 셋포인트 최종 오차 =", f"{err_s[-1]:.3g} mm")
for Tl, s, rj, ro in zip(periods, speeds, rms_jt, rms_ops):
    print(f"fig4: T={Tl:4.1f}s (v_max={s:.3f} m/s): Jᵀ RMS={rj:8.3f} mm, op-space RMS={ro:.4f} mm")

# ---------- 본문 수치 검증 (그림 없음) ----------
print("\n--- WE-1: Λ(0, π/2) 손계산 검증 ---")
qw = np.array([0.0, np.pi/2])
J = jac(qw); M = M_mat(qw)
print("J =\n", J); print("M =\n", M)
A = J @ np.linalg.solve(M, J.T)
print("J M⁻¹ Jᵀ =\n", A)
L = np.linalg.inv(A)
print("Λ =\n", L, "\n고유값(유효 질량):", np.linalg.eigvalsh(L))

print("\n--- E2 검증: 임의 방향 유효 질량 m_u = 1/(uᵀΛ⁻¹u) ---")
qx2 = np.array([0.5, 1.571])
lam2, V2 = np.linalg.eigh(Lambda(qx2))
u = np.array([1.0, 0.0])
print(f"고유값 = {lam2}, m_x = {1/(u @ np.linalg.solve(Lambda(qx2), u)):.4f} kg")

print("\n--- E1 구현 노트 검증: Khatib 꼴 vs resolved-acceleration 꼴 ---")
rng = np.random.default_rng(3)
diffs = []
for _ in range(100):
    q = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(0.2, np.pi-0.2)])
    qdv = rng.uniform(-2, 2, 2)
    a_x = rng.uniform(-5, 5, 2)
    J = jac(q); L = Lambda(q); Minv = np.linalg.inv(M_mat(q))
    F = L @ (a_x - jacdot_qd(q, qdv)) + L @ J @ Minv @ (C_mat(q, qdv)@qdv + g_vec(q))
    tau_khatib = J.T @ F
    tau_ra = M_mat(q) @ np.linalg.solve(J, a_x - jacdot_qd(q, qdv)) \
             + C_mat(q, qdv)@qdv + g_vec(q)
    diffs.append(np.abs(tau_khatib - tau_ra).max())
print(f"랜덤 상태 100개에서 |τ_Khatib - τ_RA|_max = {max(diffs):.3g} N·m")

print("\n--- E1 검증: op-space 오차 동역학이 태스크 공간 임계감쇠 해석해와 겹치는가 ---")
x_goal2 = np.array([1.3, 0.6])
e0 = np.array([0.05, -0.03])
q02 = ik2r(*(x_goal2 - e0))
xd_c  = lambda t: x_goal2
xdd_c = lambda t: np.zeros(2)
xddd_c= lambda t: np.zeros(2)
dt2 = 1e-4
ts2, Q2, _, _, _ = simulate(ctrl_opspace(xd_c, xdd_c, xddd_c), q02, np.zeros(2), 1.2, dt=dt2)
E = np.array([x_goal2 - fk(q) for q in Q2])
ana = np.outer((1 + 10*ts2)*np.exp(-10*ts2), [1, 1]) * e0
print(f"|E_sim - 해석해|_max = {np.abs(E - ana).max():.3g} m")

print("\n--- 흔한 오해 검증: Jᵀ 게인 인상 실험 (T=2s 원) ---")
for kp, kd in [(500., 30.), (5000., 95.)]:
    KpJ, KdJ = kp, kd
    tsT, QT, _, _, _ = simulate(ctrl_jt(xd), q0, qd0, 4.0)
    print(f"Jᵀ Kp={kp:6.0f} N/m: RMS = {rms(eef_err_mm(tsT, QT, xd)[lap2]):.2f} mm")
KpJ, KdJ = 500., 30.

# Jᵀ의 ė 변형 (−Kd ẋ 대신 Kd(ẋ_d − ẋ)) — 본문 각주용
def ctrl_jt_track(xd_f, xdd_f):
    def ctrl(t, q, qdv):
        J = jac(q)
        e = xd_f(t) - fk(q)
        ed = xdd_f(t) - J @ qdv
        return J.T @ (KpJ*e + KdJ*ed) + g_vec(q)
    return ctrl
tsT, QT, _, _, _ = simulate(ctrl_jt_track(xd, xdd), q0, qd0, 4.0)
print(f"Jᵀ(ė 변형, Kp=500): RMS = {rms(eef_err_mm(tsT, QT, xd)[lap2]):.2f} mm")
print("\nfigures written")
