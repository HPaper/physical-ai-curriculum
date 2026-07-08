"""Lec R04 그림 생성 스크립트.
fig1: 한 장 요약 — 같은 2R FK의 두 방법(링크 변환의 곱 vs PoE)
fig2: UR5(MR Ex 4.5)의 영구성 스크류 축 6개 3D 시각화
fig3: 2R FK 두 방법의 수치 일치 (경로 겹침 + 오차)
fig4: UR5e — 공식 DH 표(standard)에서 추출한 관절축(=PoE 스크류축)과 TCP
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = '/home/hjkim/frontier_ws/vla-study/images/lecR04/'

# ---------- 공용 FK ----------
def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def exp_so3(w, th):
    W = hat(w)
    return np.eye(3) + np.sin(th)*W + (1-np.cos(th))*(W @ W)

def exp_se3(S, th):
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    T = np.eye(4)
    W = hat(w)
    T[:3, :3] = exp_so3(w, th)
    G = np.eye(3)*th + (1-np.cos(th))*W + (th-np.sin(th))*(W @ W)
    T[:3, 3] = G @ v
    return T

def fk_poe(Slist, M, q):
    T = np.eye(4)
    for S, th in zip(Slist, q):
        T = T @ exp_se3(S, th)
    return T @ M

L1, L2 = 1.0, 0.6

def joints_2r(q1, q2):
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = p1 + np.array([L2*np.cos(q1+q2), L2*np.sin(q1+q2)])
    return np.array([[0, 0], p1, p2])

def draw_arm(ax, pts, color, lw=3.5, alpha=1.0, ls='-', label=None):
    ax.plot(pts[:, 0], pts[:, 1], ls, color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', label=label, zorder=3)
    ax.plot(pts[:-1, 0], pts[:-1, 1], 'o', color=color, ms=9, alpha=alpha, zorder=4)
    ax.plot(pts[-1, 0], pts[-1, 1], 's', color=color, ms=8, alpha=alpha, zorder=4)

def draw_frame(ax, origin, angle, size=0.22, label=None, color='k'):
    c, s = np.cos(angle), np.sin(angle)
    ax.annotate('', xy=(origin[0]+size*c, origin[1]+size*s), xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6))
    ax.annotate('', xy=(origin[0]-size*s, origin[1]+size*c), xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6))
    if label:
        ax.text(origin[0]-0.09, origin[1]-0.17, label, fontsize=11, color=color)

# ---------- fig1 ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
q1, q2 = np.pi/2, np.pi/2

# (a) 링크 변환의 곱: 중간 프레임의 사슬
ax = axes[0]
pts = joints_2r(q1, q2)
draw_arm(ax, pts, 'tab:blue')
draw_frame(ax, (0, 0), 0, label='{0}')
draw_frame(ax, pts[1], q1, label='{1}', color='tab:green')
draw_frame(ax, pts[2], q1+q2, label='{2}', color='tab:red')
ax.annotate(r'$T_{01}(q_1)$', xy=(0.42, 0.62), fontsize=13, color='tab:green')
ax.annotate(r'$T_{12}(q_2)$', xy=(-0.15, 1.22), fontsize=13, color='tab:red')
ax.set_title('방법 1: 링크 변환의 곱\n$T_{02}=T_{01}(q_1)\\,T_{12}(q_2)$ — 관절마다 중간 프레임')
ax.set_xlim(-1.3, 1.9); ax.set_ylim(-0.55, 1.75); ax.set_aspect('equal')
ax.grid(alpha=0.3)

# (b) PoE: 고정 스크류축 + 영구성에서 출발
ax = axes[1]
draw_arm(ax, joints_2r(0, 0), '0.65', label='영구성 $M$ ($q=0$)')
draw_arm(ax, joints_2r(0, q2), 'tab:orange', alpha=0.85, ls='--',
         label=r'$e^{[S_2]q_2}M$ (먼 쪽 관절부터)')
draw_arm(ax, joints_2r(q1, q2), 'tab:blue',
         label=r'$e^{[S_1]q_1}e^{[S_2]q_2}M$')
for p, name in [((0, 0), '$S_1$'), ((L1, 0), '$S_2$')]:
    ax.plot(*p, 'o', ms=15, mfc='none', mec='crimson', mew=2, zorder=5)
    ax.plot(*p, '.', ms=4, color='crimson', zorder=5)
    ax.text(p[0]+0.06, p[1]-0.22, name, color='crimson', fontsize=13)
th = np.linspace(0.15, np.pi/2-0.15, 40)
ax.plot(0.55*np.cos(th), 0.55*np.sin(th), color='tab:blue', lw=1)
ax.annotate('', xy=(0.55*np.cos(th[-1]), 0.55*np.sin(th[-1])),
            xytext=(0.55*np.cos(th[-2]), 0.55*np.sin(th[-2])),
            arrowprops=dict(arrowstyle='->', color='tab:blue'))
th2 = np.linspace(0.2, np.pi/2-0.2, 40)
ax.plot(1+0.35*np.cos(th2), 0.35*np.sin(th2), color='tab:orange', lw=1)
ax.annotate('', xy=(1+0.35*np.cos(th2[-1]), 0.35*np.sin(th2[-1])),
            xytext=(1+0.35*np.cos(th2[-2]), 0.35*np.sin(th2[-2])),
            arrowprops=dict(arrowstyle='->', color='tab:orange'))
ax.set_title('방법 2: 지수곱 (PoE)\n프레임은 $\\{0\\}$과 $M$뿐 — 축은 영구성에서 고정(⊙)')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlim(-1.3, 1.9); ax.set_ylim(-0.55, 1.75); ax.set_aspect('equal')
ax.grid(alpha=0.3)
fig.suptitle('같은 FK, 두 가지 문법 — 2R 팔, $q=(90°,90°)$, EEF $=(-0.6,\\ 1.0)$', y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(OUT + 'fig1_two_views_of_fk.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------- fig2: UR5 스크류축 (MR Ex 4.5) ----------
W1, W2, L1u, L2u, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095
# 영구성 스켈레톤 (관절 원점 사슬, 개략)
skel = np.array([
    [0, 0, 0], [0, 0, H1], [L1u, 0, H1], [L1u+L2u, 0, H1],
    [L1u+L2u, W1, H1], [L1u+L2u, W1, H1-H2], [L1u+L2u, W1+W2, H1-H2]])
axes_info = [  # (축 방향 w, 축 위의 점, 라벨)
    ([0, 0, 1], [0, 0, 0], '$S_1$'),
    ([0, 1, 0], [0, 0, H1], '$S_2$'),
    ([0, 1, 0], [L1u, 0, H1], '$S_3$'),
    ([0, 1, 0], [L1u+L2u, 0, H1], '$S_4$'),
    ([0, 0, -1], [L1u+L2u, W1, H1], '$S_5$'),
    ([0, 1, 0], [L1u+L2u, W1, H1-H2], '$S_6$'),
]
fig = plt.figure(figsize=(8.5, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(skel[:, 0], skel[:, 1], skel[:, 2], '-', color='0.4', lw=5, alpha=0.8)
ax.scatter(skel[:-1, 0], skel[:-1, 1], skel[:-1, 2], color='0.25', s=45)
for w, p, name in axes_info:
    w, p = np.array(w, float), np.array(p, float)
    seg = np.array([p - 0.22*w, p + 0.22*w])
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='crimson', lw=2)
    ax.quiver(*p, *(0.24*w), color='crimson', arrow_length_ratio=0.35)
    ax.text(*(p + 0.27*w + np.array([0.015, 0, 0.01])), name, color='crimson', fontsize=12)
# EEF 프레임 M: x̂=(-1,0,0), ŷ=(0,0,1), ẑ=(0,1,0)
pM = skel[-1]
for v, c, name in [((-1, 0, 0), 'tab:blue', '$\\hat{x}_M$'),
                   ((0, 0, 1), 'tab:green', '$\\hat{y}_M$'),
                   ((0, 1, 0), 'tab:orange', '$\\hat{z}_M$')]:
    v = 0.12*np.array(v, float)
    ax.quiver(*pM, *v, color=c, arrow_length_ratio=0.3, lw=2)
    ax.text(*(pM + 1.45*v), name, color=c, fontsize=11)
ax.text(0.03, 0, -0.05, '{0} (베이스)', fontsize=10)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_title('UR5의 영구성과 6개 스크류축 $S_1,\\dots,S_6$ (MR Ch.4 Example 4.5 파라미터)\n'
             '$S_1$: $\\hat{z}$ 둘레 / $S_2,S_3,S_4,S_6$: $\\hat{y}$ 둘레 / $S_5$: $-\\hat{z}$ 둘레', fontsize=11)
ax.set_box_aspect((1.0, 0.45, 0.45))
ax.view_init(elev=22, azim=-62)
fig.tight_layout()
fig.savefig(OUT + 'fig2_ur5_screw_axes.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------- fig3: 두 방법의 수치 일치 ----------
S1v = [0, 0, 1, 0, 0, 0]
S2v = [0, 0, 1, 0, -L1, 0]
M2 = np.eye(4); M2[0, 3] = L1 + L2

def fk_chain(q):
    def T_link(th, l):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 0, l*c], [s, c, 0, l*s], [0, 0, 1, 0], [0, 0, 0, 1]])
    return T_link(q[0], L1) @ T_link(q[1], L2)

t = np.linspace(0, 2*np.pi, 400)
Q = np.stack([t, 0.9*np.sin(2*t) + 0.5], axis=1)
P_poe = np.array([fk_poe([S1v, S2v], M2, q)[:2, 3] for q in Q])
P_chn = np.array([fk_chain(q)[:2, 3] for q in Q])
err = np.array([np.abs(fk_poe([S1v, S2v], M2, q) - fk_chain(q)).max() for q in Q])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.plot(P_chn[:, 0], P_chn[:, 1], '-', color='tab:blue', lw=2.5, label='링크 변환의 곱')
ax.plot(P_poe[::12, 0], P_poe[::12, 1], 'o', color='tab:orange', ms=5, mfc='none',
        mew=1.8, label='PoE (12점마다 표시)')
ax.add_patch(plt.Circle((0, 0), L1+L2, fill=False, ls=':', color='0.6'))
ax.add_patch(plt.Circle((0, 0), abs(L1-L2), fill=False, ls=':', color='0.6'))
ax.plot(0, 0, 'k^', ms=9)
ax.set_title('EEF 경로: $q_1=t,\\ q_2=0.9\\sin 2t+0.5$')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.semilogy(t, np.maximum(err, 1e-18), color='crimson', lw=1.2)
ax.axhline(np.finfo(float).eps, ls='--', color='0.5')
ax.text(0.15, np.finfo(float).eps*1.4, '기계 정밀도 $\\epsilon\\approx 2.2\\times10^{-16}$', fontsize=9, color='0.35')
ax.set_ylim(1e-18, 1e-12)
ax.set_title('두 방법의 원소별 최대 차이')
ax.set_xlabel('t [rad]'); ax.set_ylabel(r'$\max_{ij}|T^{PoE}_{ij}-T^{chain}_{ij}|$')
ax.grid(alpha=0.3, which='both')
fig.suptitle('2R FK: PoE와 링크 변환의 곱은 같은 함수다 (부동소수 오차 수준에서 일치)', y=1.02)
fig.tight_layout()
fig.savefig(OUT + 'fig3_poe_vs_chain.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------- fig4: UR5e DH 표 → PoE 데이터 추출 (본문 §4의 poe_from_dh 시각화) ----------
d_ur5e = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
a_ur5e = [0, -0.425, -0.3922, 0, 0, 0]
al_ur5e = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

def dh_T(th, d, a, al):
    ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[ct, -st*ca, st*sa, a*ct],
                     [st, ct*ca, -ct*sa, a*st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])

T = np.eye(4)
joint_axes, origins = [], []
for i in range(6):
    joint_axes.append((T[:3, 2].copy(), T[:3, 3].copy()))   # (관절축 방향, 축 위의 점)
    T = T @ dh_T(0, d_ur5e[i], a_ur5e[i], al_ur5e[i])
    origins.append(T[:3, 3].copy())
origins = np.array([joint_axes[0][1]] + origins)            # 베이스 원점 + 프레임 원점 사슬
tcp = origins[-1]

fig = plt.figure(figsize=(8.5, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-', color='0.4', lw=5, alpha=0.8)
ax.scatter(origins[:-1, 0], origins[:-1, 1], origins[:-1, 2], color='0.25', s=45)
ax.scatter(*tcp, color='tab:blue', s=70, marker='s')
ax.text(tcp[0]-0.02, tcp[1]-0.05, tcp[2]-0.09,
        f'TCP $M$\n({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})', color='tab:blue', fontsize=10)
for k, (w, p) in enumerate(joint_axes):
    seg = np.array([p - 0.16*w, p + 0.16*w])
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='crimson', lw=2)
    ax.quiver(*p, *(0.18*w), color='crimson', arrow_length_ratio=0.35)
    ax.text(*(p + 0.21*w + np.array([0.012, 0, 0.008])), f'$S_{k+1}$', color='crimson', fontsize=12)
ax.text(0.03, 0, -0.05, '{0} (베이스)', fontsize=10)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_title('UR5e 영구성 — 공식 DH 표(standard)에서 추출한 관절축(=PoE 스크류축)과 TCP\n'
             '$q=0$에서 각 DH 프레임의 $\\hat{z}$축과 원점을 읽으면 $S_i=(\\omega_i, -\\omega_i\\times q_i)$와 $M$이 나온다 (§4 poe_from_dh)',
             fontsize=10.5)
ax.set_box_aspect((1.0, 0.5, 0.5))
ax.view_init(elev=22, azim=-118)
fig.tight_layout()
fig.savefig(OUT + 'fig4_ur5e_dh_to_poe.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('saved:', OUT)
print('max err over path:', err.max())
print('UR5e TCP from DH:', np.round(tcp, 4))
