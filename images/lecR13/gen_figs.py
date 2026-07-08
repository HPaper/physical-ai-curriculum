"""Lec R13 그림 생성 스크립트.
fig1: LIP 위상 궤적(saddle)과 스텝의 효과
fig2: capture point 기하와 1-스텝 회복 가능 영역
fig3: compass walker(가장 단순한 보행 모델)의 극한 사이클
fig4: 실습 1·2 — DCM 걸음 계획기와 푸시의 dead-beat 회복
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lecR13"
g = 9.81

# ============================================================
# fig 1 — LIP 위상 궤적과 스텝의 효과
# ============================================================
zc = 0.9
w = np.sqrt(g / zc)
x0, v0 = 0.0, 0.5
xi0 = x0 + v0 / w

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# (a) 위상 평면 (x - p, xdot)
U, V = np.meshgrid(np.linspace(-0.4, 0.4, 24), np.linspace(-1.3, 1.3, 24))
ax1.streamplot(U, V, V, w**2 * U, color='0.75', density=1.1, linewidth=0.8,
               arrowsize=0.9)
uu = np.linspace(-0.4, 0.4, 2)
ax1.plot(uu, w * uu, color='crimson', lw=2.2,
         label=r'발산 고유방향 $\dot{x}=+\omega(x-p)$')
ax1.plot(uu, -w * uu, color='royalblue', lw=2.2,
         label=r'수렴 고유방향 $\dot{x}=-\omega(x-p)$')
ax1.plot(0, 0, 'ko', ms=7, zorder=5)
ax1.annotate('발 위에 정지\n(saddle)', (0, 0), textcoords='offset points',
             xytext=(8, -30), fontsize=9)
# 같은 물리 상태 (x0=0, v0=0.5), 발 위치 3종 → saddle 기준 상대 위치가 달라짐
for p, c, lab in [(xi0, 'royalblue', 'A: $p=\\xi_0$ (capture)'),
                  (0.10, 'crimson', 'B: $p=0.10$ (짧게)'),
                  (0.25, 'darkorange', 'C: $p=0.25$ (길게)')]:
    u_init, v_init = x0 - p, v0
    ax1.plot(u_init, v_init, 'o', color=c, ms=8, zorder=6)
    t = np.linspace(0, 0.9, 300)
    xt = (x0 - p) * np.cosh(w * t) + (v0 / w) * np.sinh(w * t)
    vt = (x0 - p) * w * np.sinh(w * t) + v0 * np.cosh(w * t)
    m = (np.abs(xt) < 0.42) & (np.abs(vt) < 1.35)
    ax1.plot(xt[m], vt[m], color=c, lw=1.8, label=lab)
ax1.set_xlim(-0.42, 0.42); ax1.set_ylim(-1.35, 1.35)
ax1.set_xlabel(r'$x - p$  (CoM$-$발, m)'); ax1.set_ylabel(r'$\dot{x}$ (m/s)')
ax1.set_title('(a) LIP 위상 평면 — 스텝은 상태가 아니라 saddle을 옮긴다')
ax1.legend(fontsize=8, loc='lower left')

# (b) 같은 상태, 발 위치 3종의 시간 응답
t = np.linspace(0, 1.2, 500)
for p, c, lab in [(xi0, 'royalblue', 'A: $p=\\xi_0=0.151$ → 발 위에 정지'),
                  (0.10, 'crimson', 'B: $p=0.10$ → 앞으로 발산'),
                  (0.25, 'darkorange', 'C: $p=0.25$ → 뒤로 넘어짐')]:
    xt = p + (x0 - p) * np.cosh(w * t) + (v0 / w) * np.sinh(w * t)
    ax2.plot(t, xt, color=c, lw=2, label=lab)
    ax2.axhline(p, color=c, ls=':', lw=1)
ax2.set_ylim(-1.2, 1.6)
ax2.set_xlabel('t (s)'); ax2.set_ylabel('CoM 위치 x (m)')
ax2.set_title(r'(b) 같은 초기 상태 $(x_0,\dot{x}_0)=(0,\,0.5)$, 발 위치만 다름')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_lip_phase.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 2 — capture point 기하 + 1-스텝 회복 가능 영역
# ============================================================
zc2 = 0.8
w2 = np.sqrt(g / zc2)
v_push = 0.7
xi_p = v_push / w2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 기하: 지면, 지지발, CoM, 속도 화살표, capture point
ax1.axhline(0, color='k', lw=1.5)
ax1.plot([0], [0], 'k^', ms=12)
ax1.annotate('지지발 $p$', (0, 0), textcoords='offset points', xytext=(-14, -20))
ax1.plot([0, 0.0], [0, zc2], color='0.4', lw=3)          # 다리(현재)
ax1.plot(0, zc2, 'o', ms=18, color='steelblue', zorder=5)  # CoM
ax1.annotate('CoM $(x,\\,z_c)$', (0, zc2), textcoords='offset points',
             xytext=(-75, 6), fontsize=10)
ax1.annotate('', xy=(0.35, zc2), xytext=(0.02, zc2),
             arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2.4))
ax1.annotate(r'$\dot{x}$', (0.19, zc2), textcoords='offset points',
             xytext=(0, 8), color='crimson', fontsize=12)
ax1.plot(xi_p, 0, 'o', ms=11, color='seagreen', zorder=5)
ax1.annotate(r'capture point  $\xi = x + \dot{x}/\omega$', (xi_p, 0),
             textcoords='offset points', xytext=(-30, -26), color='seagreen',
             fontsize=10)
ax1.plot([0, xi_p], [zc2, 0], color='seagreen', ls='--', lw=1.8)  # 스텝할 다리
ax1.annotate('여기를 밟으면\n정지한다', (xi_p * 0.75, zc2 * 0.42),
             color='seagreen', fontsize=9)
ax1.set_xlim(-0.35, 0.65); ax1.set_ylim(-0.16, 1.0)
ax1.set_aspect('equal')
ax1.set_title(f'(a) capture point 기하  ($z_c={zc2}$ m, $\\dot{{x}}={v_push}$ m/s'
              f' → $\\xi={xi_p:.3f}$ m)')
ax1.set_xlabel('x (m)')
ax1.axis('on')

# (b) 1-스텝 회복 가능 영역: v <= omega * l * e^{-omega t_d}
l = np.linspace(0, 0.45, 100)
for t_d, c, lab in [(0.0, 'seagreen', '즉시 스텝 ($t_d=0$)'),
                    (0.15, 'darkorange', '지연 0.15 s'),
                    (0.3, 'crimson', '지연 0.3 s')]:
    vmax = w2 * l * np.exp(-w2 * t_d)
    ax2.plot(l, vmax, color=c, lw=2, label=lab)
ax2.fill_between(l, 0, w2 * l, color='seagreen', alpha=0.10)
ax2.plot([0.35], [w2 * 0.35], 'o', color='seagreen')
ax2.annotate(f'$l_{{max}}$=0.35 m → {w2*0.35:.2f} m/s', (0.35, w2 * 0.35),
             textcoords='offset points', xytext=(-130, 2), fontsize=9)
ax2.plot([0.35], [w2 * 0.35 * np.exp(-w2 * 0.3)], 'o', color='crimson')
ax2.annotate(f'지연 0.3 s → {w2*0.35*np.exp(-w2*0.3):.2f} m/s',
             (0.35, w2 * 0.35 * np.exp(-w2 * 0.3)),
             textcoords='offset points', xytext=(-135, 4), fontsize=9)
ax2.set_xlabel('최대 스텝 길이 $l_{max}$ (m)')
ax2.set_ylabel('1스텝으로 회복 가능한 푸시 속도 (m/s)')
ax2.set_title('(b) 1-스텝 회복 영역 — 지연은 지수적으로 비싸다')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_capture_point.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 3 — compass walker 극한 사이클
# ============================================================
gamma = 0.009


def dyn(s):
    th, phi, thd, phid = s
    thdd = np.sin(th - gamma)
    phidd = thdd + thd**2 * np.sin(phi) - np.cos(th - gamma) * np.sin(phi)
    return np.array([thd, phid, thdd, phidd])


def rk4(s, dt):
    k1 = dyn(s); k2 = dyn(s + dt/2*k1); k3 = dyn(s + dt/2*k2); k4 = dyn(s + dt*k3)
    return s + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def one_step(th0, thd0, dt=2e-4, tmax=15.0):
    """충돌 직후 → 다음 충돌 직후. (성공 시) 궤적과 충돌 전/후 상태 반환."""
    s = np.array([th0, 2*th0, thd0, (1 - np.cos(2*th0)) * thd0])
    t = 0.0
    traj = [s.copy()]
    while t < tmax:
        s_new = rk4(s, dt); t_new = t + dt
        g_old = s[1] - 2*s[0]; g_new = s_new[1] - 2*s_new[0]
        if s_new[0] < -0.05 and g_old < 0 <= g_new:
            a, ta = s, t
            bb, tb = s_new, t_new
            for _ in range(60):
                dtm = (tb - ta) / 2
                sm = rk4(a, dtm); tm = ta + dtm
                if sm[1] - 2*sm[0] < 0:
                    a, ta = sm, tm
                else:
                    bb, tb = sm, tm
            s = bb
            traj.append(s.copy())
            return (-s[0], np.cos(2*s[0]) * s[2]), np.array(traj), (s[0], s[2])
        s, t = s_new, t_new
        traj.append(s.copy())
        if abs(s[0]) > 1.5:
            return None
    return None


th_star, thd_star = 0.200310, -0.199832  # verify_R13.py의 fsolve 결과

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# (a) 위상 궤적: 섭동 시작 → 극한 사이클 수렴
z = (th_star - 0.005, thd_star)
n_steps = 12
cmap = plt.get_cmap('viridis')
for k in range(n_steps):
    r = one_step(*z)
    if r is None:
        break
    z_next, traj, pre = r
    c = cmap(0.15 + 0.7 * k / n_steps)
    ax1.plot(traj[:, 0], traj[:, 2], color=c, lw=1.1, alpha=0.85)
    # 충돌 리셋(점프): (th-, thd-) --> (th+, thd+)
    ax1.plot([pre[0], z_next[0]], [pre[1], z_next[1]], color=c, ls=':',
             lw=0.9, alpha=0.7)
    z = z_next
# 극한 사이클(고정점에서 1스텝) 굵게
r = one_step(th_star, thd_star)
zc_next, traj_c, pre_c = r
ax1.plot(traj_c[:, 0], traj_c[:, 2], color='crimson', lw=2.6, zorder=5,
         label='극한 사이클 (스윙 상)')
ax1.plot([pre_c[0], zc_next[0]], [pre_c[1], zc_next[1]], color='crimson',
         ls='--', lw=1.8, zorder=5, label='heelstrike 리셋 (점프)')
ax1.plot(th_star, thd_star, 'k*', ms=13, zorder=6, label='스트라이드 맵 고정점')
ax1.set_xlabel(r'스탠스각 $\theta$ (rad)')
ax1.set_ylabel(r'$\dot{\theta}$ (무차원)')
ax1.set_title(f'(a) compass walker 위상 궤적 (경사 $\\gamma$={gamma})')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.3)

# (b) 스텝별 오차의 기하급수 감쇠
for d, c in [(0.002, 'royalblue'), (-0.005, 'darkorange')]:
    z = (th_star + d, thd_star)
    errs = [abs(z[0] - th_star)]
    for k in range(10):
        r = one_step(*z)
        if r is None:
            break
        z = r[0]
        errs.append(abs(z[0] - th_star))
    ax2.semilogy(errs, 'o-', color=c, label=f'$\\delta\\theta_0={d:+.3f}$')
kk = np.arange(11)
ax2.semilogy(kk, 5e-3 * 0.589**kk, 'k--', lw=1,
             label=r'$|\lambda|_{max}^k=0.589^k$ (선형 예측)')
ax2.set_xlabel('스텝 수 k'); ax2.set_ylabel(r'$|\theta_k - \theta^*|$')
ax2.set_title('(b) 스트라이드 맵 수렴 — 제어 없이 안정')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_compass_cycle.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 4 — 실습 1·2: DCM 걸음 계획기와 푸시의 dead-beat 회복
# ============================================================
zc4, T4, v_d = 0.9, 0.5, 0.6
w4 = np.sqrt(g / zc4)
L4 = v_d * T4
b4 = L4 / (np.exp(w4 * T4) - 1)
dt4 = 1e-3
n4 = int(T4 / dt4)


def run_planner(push=False, n_steps=14):
    x, xi = 0.0, b4
    ts, xs, xis, ps = [], [], [], []
    t = 0.0
    for k in range(n_steps):
        p = xi - b4                      # 발 위치 규칙 (실습 1)
        for i in range(n4):
            xi += dt4 * w4 * (xi - p)
            x += dt4 * w4 * (xi - x)
            if push and k == 8 and i == 200:
                xi += 0.25 / w4          # 푸시 (실습 2)
            ts.append(t); xs.append(x); xis.append(xi); ps.append(p)
            t += dt4
    return map(np.array, (ts, xs, xis, ps))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, push, title in [
        (ax1, False, '(a) 정속 보행 계획 — $p_k = \\xi - b$, $b = L/(e^{\\omega T}-1)$'),
        (ax2, True, '(b) 8번째 스텝 중 푸시 $\\Delta\\dot{x}=0.25$ m/s → dead-beat 회복 (확대)')]:
    ts, xs, xis, ps = run_planner(push)
    ax.step(ts, ps, where='post', color='0.45', lw=1.6, label='발 위치(ZMP) $p$')
    ax.plot(ts, xs, color='royalblue', lw=2, label='CoM $x$')
    ax.plot(ts, xis, color='crimson', lw=1.6, ls='--', label=r'DCM $\xi$')
    if push:
        tp = 8 * T4 + 200 * dt4
        ax.axvline(tp, color='darkorange', lw=1.6, ls=':')
        ax.annotate('푸시\n($\\xi$가 점프)', (tp, 2.35), color='darkorange',
                    fontsize=9, textcoords='offset points', xytext=(6, 0))
        ax.annotate('다음 스텝이 한 번에 흡수\n($\\xi - p = b$ 복원)', (4.5, 3.15),
                    color='0.25', fontsize=9,
                    xytext=(4.9, 2.35), textcoords='data',
                    arrowprops=dict(arrowstyle='->', color='0.4', lw=1.1))
        ax.set_xlim(3.4, 6.6)
        ax.set_ylim(1.85, 4.15)
    ax.set_xlabel('t (s)')
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3)
ax1.set_ylabel('위치 (m)')
ax1.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_dcm_plan.png", dpi=140)
plt.close(fig)

print("figures written:", OUT)
