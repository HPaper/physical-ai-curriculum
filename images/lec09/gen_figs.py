# Lec R09 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# 공통: WE-2의 직육면체 (m=1.2 kg, 0.30 x 0.20 x 0.10 m)
I_diag = np.array([0.005, 0.010, 0.013])   # kg m^2
I = np.diag(I_diag)
I_inv = np.linalg.inv(I)

def euler_eq(t, w):
    return I_inv @ (-np.cross(w, I @ w))

def simulate(w0, T=20.0, n=4001):
    sol = solve_ivp(euler_eq, [0, T], np.asarray(w0, float),
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.01)
    t = np.linspace(0, T, n)
    return t, sol.sol(t)

# ---------------- fig1: 한 장 요약 ----------------
fig = plt.figure(figsize=(11.5, 4.4))

# (a) 기울어진 상자 점구름 + 주축 (관성텐서의 고유분해 = 질량 분포의 PCA)
ax = fig.add_subplot(1, 2, 1, projection='3d')
rng = np.random.default_rng(3)
pts = rng.uniform(-0.5, 0.5, (900, 3)) * np.array([0.30, 0.20, 0.10])
R = Rotation.from_rotvec([0.4, -0.3, 0.7]).as_matrix()
pts = pts @ R.T
ax.scatter(*pts.T, s=2.5, alpha=0.30, color='#607d8b')
w_pt = 1.2 / len(pts)
r2 = np.sum(pts**2, axis=1)
I_cloud = w_pt * (np.sum(r2) * np.eye(3) - pts.T @ pts)
evals, evecs = np.linalg.eigh(I_cloud)
colors = ['#1f77b4', '#d62728', '#2ca02c']
names = ['최소 관성축', '중간축', '최대 관성축']
for k in range(3):
    v = evecs[:, k] * (0.23 - 0.04 * k)
    ax.quiver(0, 0, 0, *v, color=colors[k], linewidth=2.4, arrow_length_ratio=0.18)
    ax.text(*(v * 1.45), f'{names[k]}\nλ={evals[k]:.3f}',
            color=colors[k], fontsize=8.5, ha='center')
ax.set_title('(a) 관성텐서의 고유분해 = 질량 분포의 PCA', fontsize=11)
lim = 0.26
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_box_aspect([1, 1, 1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

# (b) 중간축 회전의 뒤집힘 (Dzhanibekov)
ax2 = fig.add_subplot(1, 2, 2)
t, w = simulate([0.01, 5.0, 0.01])
ax2.plot(t, w[0], color='#1f77b4', lw=1.0, label=r'$\omega_1$ (최소축)')
ax2.plot(t, w[1], color='#d62728', lw=1.6, label=r'$\omega_2$ (중간축)')
ax2.plot(t, w[2], color='#2ca02c', lw=1.0, label=r'$\omega_3$ (최대축)')
for ft in [3.683, 10.134, 16.584]:
    ax2.axvline(ft, color='gray', ls=':', lw=0.8)
ax2.annotate('뒤집힘 (약 6.45 s 간격)', xy=(3.683, 0), xytext=(5.2, 3.2),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax2.set_xlabel('시간 [s]'); ax2.set_ylabel('각속도 [rad/s]')
ax2.set_title('(b) 중간축 회전은 주기적으로 뒤집힌다 (τ=0인데도!)', fontsize=11)
ax2.legend(loc='lower right', fontsize=8.5); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig1_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------------- fig3: 세 주축의 안정성 비교 ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
cases = [(0, '최소 관성축 회전: 안정'),
         (1, '중간축 회전: 불안정 → 뒤집힘'),
         (2, '최대 관성축 회전: 안정')]
for ax, (axis, title) in zip(axes, cases):
    w0 = np.full(3, 0.01); w0[axis] = 5.0
    t, w = simulate(w0)
    for k, (c, lb) in enumerate(zip(['#1f77b4', '#d62728', '#2ca02c'],
                                    [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])):
        ax.plot(t, w[k], color=c, lw=1.1, label=lb)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel('시간 [s]'); ax.grid(alpha=0.3)
axes[0].set_ylabel('각속도 [rad/s]')
axes[2].legend(loc='center right', fontsize=9)
fig.suptitle('같은 크기(0.01 rad/s)의 교란 — 중간축에서만 자란다  (I = diag(0.005, 0.010, 0.013))',
             fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig('fig3_stability.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------------- fig4: 각운동량 방향의 궤적 (몸체좌표) — 안장점 구조 ----------------
# dL/dt|body = -w x L,  w = I^-1 L.  |L|은 몸체좌표에서도 보존 → 단위구 위의 곡선.
def L_eq(t, L):
    return -np.cross(I_inv @ L, L)

fig = plt.figure(figsize=(7.2, 6.6))
ax = fig.add_subplot(111, projection='3d')
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                  color='lightgray', lw=0.3, alpha=0.6)

def trace(L0, T, color, lw=1.0):
    L0 = np.asarray(L0, float); L0 /= np.linalg.norm(L0)
    sol = solve_ivp(L_eq, [0, T], L0 * 0.05, rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.02)
    tt = np.linspace(0, T, 4000)
    L = sol.sol(tt); L /= np.linalg.norm(L, axis=0)
    ax.plot(*L, color=color, lw=lw, alpha=0.9)

for th in np.deg2rad([18, 45, 72]):            # 최소축(x) 주변 폐곡선
    trace([np.cos(th), np.sin(th) * 0.7, np.sin(th) * 0.7], 60, '#1f77b4')
    trace([-np.cos(th), np.sin(th) * 0.7, np.sin(th) * 0.7], 60, '#1f77b4')
for th in np.deg2rad([18, 45, 72]):            # 최대축(z) 주변 폐곡선
    trace([np.sin(th) * 0.7, np.sin(th) * 0.7, np.cos(th)], 60, '#2ca02c')
    trace([np.sin(th) * 0.7, np.sin(th) * 0.7, -np.cos(th)], 60, '#2ca02c')
for s1, s3 in [(1, 1), (-1, -1)]:              # 중간축(y) 근처 — 분리선을 따라 크게 도는 궤적
    trace([0.03 * s1, 1.0, 0.03 * s3], 90, '#d62728', lw=1.5)
    trace([0.03 * s1, -1.0, 0.03 * s3], 90, '#d62728', lw=1.5)
for p, name, c in [([1.45, 0, 0], '최소축 (안정)', '#1f77b4'),
                   ([0, 1.62, 0], '중간축 (안장점)', '#d62728'),
                   ([0, 0, 1.32], '최대축 (안정)', '#2ca02c')]:
    ax.text(*p, name, color=c, fontsize=11, ha='center', weight='bold')
ax.set_title('토크 없는 회전에서 각운동량 방향 $\\hat{L}$의 궤적 (몸체좌표)\n'
             '최소·최대축 주변 = 폐곡선(안정), 중간축 = 안장점(불안정)', fontsize=11)
ax.set_box_aspect([1, 1, 1])
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(elev=18, azim=-52)
fig.tight_layout()
fig.savefig('fig4_polhode.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------------- fig2: 평행축 정리 = 분산 분해 (WE-1의 막대) ----------------
m_rod, L_rod = 2.0, 0.6
I_c_rod = m_rod * L_rod**2 / 12          # 0.06 (COM 축)
d = np.linspace(-0.45, 0.45, 400)
I_d = I_c_rod + m_rod * d**2             # I(d) = I_c + m d^2

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(d, I_d, color='#1f77b4', lw=2.0,
        label=r'$I(d) = I_c + m\,d^2$  (평행축 정리)')
ax.axhline(I_c_rod, color='#2ca02c', ls='--', lw=1.2)
ax.fill_between(d, I_c_rod, I_d, color='#d62728', alpha=0.12)
ax.scatter([0], [I_c_rod], color='#2ca02c', zorder=5)
ax.scatter([L_rod/2], [I_c_rod + m_rod*(L_rod/2)**2], color='#d62728', zorder=5)
ax.annotate('COM 축: 최소 0.06\n("분산" 항만 남음)', xy=(0, I_c_rod),
            xytext=(-0.40, 0.16), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('끝단 축: 0.06 + 2·(0.3)² = 0.24', xy=(0.3, 0.24),
            xytext=(0.02, 0.32), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.text(-0.245, 0.115, '점질량 보정 $m\\,d^2$\n(= "bias" 항, 항상 ≥ 0)',
        color='#d62728', fontsize=9, ha='center')
ax.set_xlabel('축의 COM으로부터의 거리 $d$ [m]')
ax.set_ylabel(r'관성모멘트 $I$ [kg·m$^2$]')
ax.set_title('평행축 정리 (WE-1의 막대, m=2 kg, L=0.6 m): COM 축이 항상 최소', fontsize=11)
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig2_parallel_axis.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('done: fig1_summary.png, fig2_parallel_axis.png, fig3_stability.png, fig4_polhode.png')
