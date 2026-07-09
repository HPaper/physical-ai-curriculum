# Lec R05 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

L1, L2 = 1.0, 0.6

def fk_pts(q):
    """베이스-팔꿈치-EEF 3점."""
    q1, q12 = q[0], q[0] + q[1]
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = p1 + np.array([L2*np.cos(q12), L2*np.sin(q12)])
    return np.array([[0, 0], p1, p2])

def jac(q):
    q1, q12 = q[0], q[0] + q[1]
    return np.array([[-L1*np.sin(q1) - L2*np.sin(q12), -L2*np.sin(q12)],
                     [ L1*np.cos(q1) + L2*np.cos(q12),  L2*np.cos(q12)]])

# ---------------------------------------------------------------- fig 1
# 관절속도 단위원 -> EEF 속도 타원 (자세 3개)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2),
                               gridspec_kw={'width_ratios': [1, 1.5]})

th = np.linspace(0, 2*np.pi, 200)
circle = np.stack([np.cos(th), np.sin(th)])          # 단위원 ||qdot||=1
n_dots = 8
th_d = np.linspace(0, 2*np.pi, n_dots, endpoint=False)
dots = np.stack([np.cos(th_d), np.sin(th_d)])
dot_colors = plt.cm.hsv(th_d / (2*np.pi))

axL.plot(circle[0], circle[1], 'k-', lw=1.5)
axL.scatter(dots[0], dots[1], c=dot_colors, s=60, zorder=3, edgecolors='k', lw=0.5)
axL.axhline(0, color='0.85', lw=0.8); axL.axvline(0, color='0.85', lw=0.8)
axL.set_xlabel(r'$\dot{q}_1$ [rad/s]'); axL.set_ylabel(r'$\dot{q}_2$ [rad/s]')
axL.set_title('관절속도 공간: 단위원 $\\|\\dot{q}\\|=1$')
axL.set_aspect('equal'); axL.set_xlim(-1.6, 1.6); axL.set_ylim(-1.6, 1.6)

postures = [(np.deg2rad([20, 100]),  'tab:blue',   'A'),
            (np.deg2rad([55, 40]),   'tab:green',  'B'),
            (np.deg2rad([95, 10]),   'tab:red',    'C (특이점 근처)')]
scale = 0.28
for q, c, name in postures:
    pts = fk_pts(q)
    J = jac(q)
    axR.plot(pts[:, 0], pts[:, 1], 'o-', color=c, lw=3, ms=5,
             label=f'{name}: q=({np.rad2deg(q[0]):.0f}°, {np.rad2deg(q[1]):.0f}°)')
    ell = pts[2][:, None] + scale * (J @ circle)     # v = J qdot 를 EEF에 그림
    axR.plot(ell[0], ell[1], color=c, lw=1.4, alpha=0.9)
    axR.fill(ell[0], ell[1], color=c, alpha=0.10)

# 자세 A에는 단위원의 색점을 사상해 대응을 보여줌
qA, JA = postures[0][0], jac(postures[0][0])
pA = fk_pts(qA)[2]
mapped = pA[:, None] + scale * (JA @ dots)
axR.scatter(mapped[0], mapped[1], c=dot_colors, s=45, zorder=4, edgecolors='k', lw=0.5)

axR.scatter([0], [0], marker='s', s=70, color='k', zorder=5)
axR.set_title('작업공간: EEF 속도 타원 $v = J(q)\\,\\dot{q}$  (표시 스케일 0.28)')
axR.set_xlabel('x [m]'); axR.set_ylabel('y [m]')
axR.set_aspect('equal'); axR.legend(loc='lower left', fontsize=9)
axR.set_xlim(-1.4, 1.6); axR.set_ylim(-0.4, 2.0)
fig.tight_layout()
fig.savefig('fig1_velocity_ellipse.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# 속도 타원 vs 힘 타원 (쌍대) — worked example 자세 (30°, 60°)
q = np.deg2rad([30.0, 60.0])
J = jac(q)
U, S, Vt = np.linalg.svd(J)
pts = fk_pts(q)
p_ee = pts[2]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
titles = [f'속도 타원: $v=J\\dot{{q}},\\ \\|\\dot{{q}}\\|=1$\n반축 $\\sigma_1={S[0]:.2f},\\ \\sigma_2={S[1]:.2f}$',
          f'힘 타원: $F=J^{{-T}}\\tau,\\ \\|\\tau\\|=1$\n반축 $1/\\sigma_2={1/S[1]:.2f},\\ 1/\\sigma_1={1/S[0]:.2f}$']
for ax, radii, c, ttl in zip(axes, [S, 1/S], ['tab:blue', 'tab:orange'], titles):
    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='0.6', lw=3, ms=5, zorder=1)
    ax.scatter([0], [0], marker='s', s=70, color='k')
    ell = p_ee[:, None] + 0.35 * (U @ np.diag(radii) @ circle)
    ax.plot(ell[0], ell[1], color=c, lw=2)
    ax.fill(ell[0], ell[1], color=c, alpha=0.12)
    for i, r in enumerate(radii):                     # 주축 화살표
        v = 0.35 * r * U[:, i]
        ax.annotate('', xy=p_ee + v, xytext=p_ee,
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel('x [m]'); ax.set_aspect('equal')
axes[0].set_ylabel('y [m]')
axes[0].set_xlim(-1.2, 2.2); axes[0].set_ylim(-0.5, 2.4)
fig.suptitle('같은 자세 q=(30°, 60°)의 쌍대: 빠르게 움직이는 방향일수록 힘은 약하다', y=1.0)
fig.tight_layout()
fig.savefig('fig2_force_duality.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# 해석 자코비안의 함정: ZYX 오일러각의 짐벌락
def B_zyx(al, be):
    sa, ca, sb, cb = np.sin(al), np.cos(al), np.sin(be), np.cos(be)
    return np.array([[0, -sa, ca*cb],
                     [0,  ca, sa*cb],
                     [1,   0,   -sb]])

betas = np.deg2rad(np.linspace(-89.9, 89.9, 400))
worst = np.array([1/np.linalg.svd(B_zyx(0.0, b))[1].min() for b in betas])

fig, ax = plt.subplots(figsize=(8, 4.4))
ax.semilogy(np.rad2deg(betas), worst, lw=2, color='tab:red')
ax.axvline(90, color='k', ls='--', lw=1); ax.axvline(-90, color='k', ls='--', lw=1)
b80 = 1/np.sqrt(1 - np.sin(np.deg2rad(80)))
ax.annotate(f'β=80°에서 약 {b80:.1f}배', xy=(80, b80), xytext=(20, 30),
            arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=11)
ax.set_xlabel('피치각 β [deg]')
ax.set_ylabel(r'$1/\sigma_{\min}(B)$ — 단위 $\omega$에 필요한 $\|\dot{\phi}\|$ (최악 방향)')
ax.set_title('ZYX 오일러각: β→±90°에서 오일러각 속도가 발산한다 (짐벌락)')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig('fig3_gimbal.png', dpi=140, bbox_inches='tight')
plt.close(fig)

print('생성 완료: fig1_velocity_ellipse.png, fig2_force_duality.png, fig3_gimbal.png')
