# Lec R10 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy/scipy/matplotlib 필요)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# ---------- 2링크 파라미터 (본문 Worked Example과 동일: 균일 막대 2개) ----------
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
lc1, lc2 = 0.5, 0.5
I1, I2 = m1*l1**2/12, m2*l2**2/12
grav = 9.81

def M_mat(q):
    c2 = np.cos(q[1])
    M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
    M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
    M22 = I2 + m2*lc2**2
    return np.array([[M11, M12], [M12, M22]])

def C_mat(q, qd):
    h = m2*l1*lc2*np.sin(q[1])
    return np.array([[-h*qd[1], -h*(qd[0]+qd[1])],
                     [ h*qd[0], 0.0]])

def g_vec(q):
    c1, c12 = np.cos(q[0]), np.cos(q[0]+q[1])
    return grav*np.array([(m1*lc1 + m2*l1)*c1 + m2*lc2*c12,
                          m2*lc2*c12])

def V_pot(q1v, q2v):
    return grav*(m1*lc1*np.sin(q1v) + m2*(l1*np.sin(q1v) + lc2*np.sin(q1v+q2v)))

def kinetic(q, qd):
    return 0.5*qd @ M_mat(q) @ qd

def rhs(t, s):
    q, qd = s[:2], s[2:]
    qdd = np.linalg.solve(M_mat(q), -C_mat(q, qd) @ qd - g_vec(q))
    return np.concatenate([qd, qdd])

# ---------- 그림 1: 무토크 스윙 스냅샷 + 에너지 시계열 ----------
q0 = np.array([1.2, 0.5]); s0 = np.concatenate([q0, np.zeros(2)])
T_end = 4.0
sol = solve_ivp(rhs, [0, T_end], s0, rtol=1e-10, atol=1e-12, dense_output=True)
ts = np.linspace(0, T_end, 600)
S = sol.sol(ts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

snap_ts = np.linspace(0, 1.6, 9)
cmap = plt.cm.viridis
for i, tt in enumerate(snap_ts):
    q = sol.sol(tt)[:2]
    x1, y1 = l1*np.cos(q[0]), l1*np.sin(q[0])
    x2, y2 = x1 + l2*np.cos(q[0]+q[1]), y1 + l2*np.sin(q[0]+q[1])
    c = cmap(i/(len(snap_ts)-1))
    ax1.plot([0, x1, x2], [0, y1, y2], 'o-', color=c, lw=2.2,
             ms=4.5, alpha=0.85, zorder=3)
ax1.plot(0, 0, 'ks', ms=9, zorder=4)
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(0, snap_ts[-1]))
fig.colorbar(sm, ax=ax1, label='시간 t [s]', shrink=0.85)
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2); ax1.set_aspect('equal')
ax1.set_title('(a) 무토크 스윙 (τ=0): 중력만으로 낙하')
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.grid(alpha=0.3)

Ts = np.array([kinetic(s[:2], s[2:]) for s in S.T])
Vs = np.array([V_pot(s[0], s[1]) for s in S.T])
E0 = Ts[0] + Vs[0]
ax2.plot(ts, Ts, label='운동에너지 T', lw=1.8)
ax2.plot(ts, Vs, label='위치에너지 V', lw=1.8)
ax2.plot(ts, Ts+Vs, 'k-', label='총 에너지 E=T+V', lw=2.4)
ax2.set_title(f'(b) 에너지 교환과 보존 (max$|E-E_0|$ ≈ {np.abs(Ts+Vs-E0).max():.1e} J)')
ax2.set_xlabel('시간 t [s]'); ax2.set_ylabel('에너지 [J]')
ax2.legend(loc='center right'); ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig1_swing_energy.png'), dpi=140)
plt.close(fig)

# ---------- 그림 2: M(q) 성분과 조건수 — 자세 의존 메트릭 ----------
q2s = np.linspace(-np.pi, np.pi, 400)
M11s, M12s, M22s, conds = [], [], [], []
for q2v in q2s:
    Mv = M_mat(np.array([0.0, q2v]))
    M11s.append(Mv[0, 0]); M12s.append(Mv[0, 1]); M22s.append(Mv[1, 1])
    conds.append(np.linalg.cond(Mv))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(q2s, M11s, label='$M_{11}$ (관절1이 느끼는 관성)', lw=2)
ax1.plot(q2s, M12s, label='$M_{12}=M_{21}$ (관성 결합)', lw=2)
ax1.plot(q2s, M22s, label='$M_{22}$ (상수!)', lw=2)
ax1.axvline(0, color='gray', ls=':', lw=1)
ax1.annotate('팔을 편 자세\n(관성 최대)', xy=(0, M11s[200]),
             xytext=(0.6, 2.35), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.set_xlabel('$q_2$ [rad]'); ax1.set_ylabel('성분 값 [kg·m²]')
ax1.set_title('(a) 질량행렬 성분 — $q_2$에만 의존')
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.plot(q2s, conds, 'C3', lw=2)
ax2.axvline(0, color='gray', ls=':', lw=1)
ax2.scatter([0, np.pi/2], [conds[200], np.linalg.cond(M_mat(np.array([0, np.pi/2])))],
            color='k', zorder=3, s=25)
ax2.annotate(f'$q_2$=0: cond ≈ {conds[200]:.1f}', xy=(0, conds[200]),
             xytext=(0.35, conds[200]*0.9), fontsize=9)
cpi2 = np.linalg.cond(M_mat(np.array([0, np.pi/2])))
ax2.annotate(f'$q_2$=π/2: cond ≈ {cpi2:.1f}', xy=(np.pi/2, cpi2),
             xytext=(1.15, 12), fontsize=9)
ax2.set_xlabel('$q_2$ [rad]'); ax2.set_ylabel('cond(M)')
ax2.set_title(f'(b) 조건수 — 자세에 따라 {conds[200]/min(conds):.0f}배까지 변한다')
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig2_mass_matrix.png'), dpi=140)
plt.close(fig)

# ---------- 그림 3: 중력 포텐셜 지형과 중력 토크 벡터장 ----------
n = 120
q1g, q2g = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi, np.pi, n))
Vg = V_pot(q1g, q2g)

fig, ax = plt.subplots(figsize=(6.4, 5.2))
pc = ax.pcolormesh(q1g, q2g, Vg, cmap='RdBu_r', shading='auto')
fig.colorbar(pc, ax=ax, label='포텐셜 V(q) [J]')
cs = ax.contour(q1g, q2g, Vg, levels=10, colors='k', linewidths=0.4, alpha=0.5)

nq = 14
q1a, q2a = np.meshgrid(np.linspace(-np.pi, np.pi, nq), np.linspace(-np.pi, np.pi, nq))
U = np.zeros_like(q1a); W = np.zeros_like(q2a)
for i in range(nq):
    for j in range(nq):
        gv = g_vec(np.array([q1a[i, j], q2a[i, j]]))
        U[i, j], W[i, j] = -gv[0], -gv[1]      # −g = −∇V : 내리막 방향
ax.quiver(q1a, q2a, U, W, color='k', alpha=0.75, width=0.0035)
ax.plot(-np.pi/2, 0, 'w*', ms=16, mec='k')
ax.annotate('팔이 아래로 늘어진 안정점\n($q_1$=−π/2, $q_2$=0)', xy=(-np.pi/2, 0),
            xytext=(-2.9, -2.6), fontsize=9, color='k',
            arrowprops=dict(arrowstyle='->'))
ax.set_xlabel('$q_1$ [rad]'); ax.set_ylabel('$q_2$ [rad]')
ax.set_title('중력 지형: V(q) 히트맵과 −g(q)=−∇V 벡터장')

fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig3_gravity_landscape.png'), dpi=140)
plt.close(fig)

print("figures written")
