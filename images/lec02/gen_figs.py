# Lec R02 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
import warnings; warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
Rx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
euler = lambda a, b, g: Rz(a) @ Ry(b) @ Rx(g)
deg = np.deg2rad

C_BLUE, C_RED, C_GREEN, C_GRAY = '#1f77b4', '#d62728', '#2ca02c', '#999999'


def draw_sphere(ax, alpha=0.08):
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                    color='lightsteelblue', alpha=alpha, linewidth=0)
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), 0, color=C_GRAY, lw=0.6, alpha=0.6)  # 적도
    for ph in [0, np.pi/2]:
        ax.plot(np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th),
                color=C_GRAY, lw=0.6, alpha=0.6)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()


# ---------------------------------------------------------------- fig 1
# (a) 회전행렬 평균은 회전이 아니다  (b) 구면 위 lerp vs slerp
fig = plt.figure(figsize=(11, 4.6))

ax = fig.add_subplot(1, 2, 1)
shape = np.array([[0, 0], [1, 0], [1, .28], [.35, .28], [.35, .55],
                  [.75, .55], [.75, .8], [0, .8], [0, 0]]).T - np.array([[.4], [.4]])
rot2 = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
R1, R2 = rot2(deg(60)), rot2(deg(-60))
M = (R1 + R2) / 2
for mat, c, lb, lw in [(np.eye(2), C_GRAY, '원본', 1.2),
                       (R1, C_BLUE, 'R1 (+60°)', 1.2), (R2, C_GREEN, 'R2 (−60°)', 1.2),
                       (M, C_RED, '(R1+R2)/2 — 회전이 아님!', 2.4)]:
    s = mat @ shape
    ax.plot(s[0], s[1], color=c, lw=lw, label=lb)
    ax.fill(s[0], s[1], color=c, alpha=0.10)
ax.set_title('(a) 회전의 "평균"은 회전이 아니다\n(R1+R2)/2: det = %.2f — 도형이 쭈그러든다' % np.linalg.det(M))
ax.legend(loc='lower right', fontsize=8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.15, 1.15)

ax = fig.add_subplot(1, 2, 2, projection='3d')
draw_sphere(ax)
a = np.array([0, 1, 0.]); b = np.array([0, 0, -1.])
ts = np.linspace(0, 1, 60)
Om = np.arccos(a @ b)
slp = np.array([(np.sin((1-t)*Om)*a + np.sin(t*Om)*b)/np.sin(Om) for t in ts])
lrp = np.array([(1-t)*a + t*b for t in ts])
ax.plot(*slp.T, color=C_BLUE, lw=2.5, label='slerp: 구면(다양체) 위를 따라감')
ax.plot(*lrp.T, color=C_RED, lw=2.5, ls='--', label='lerp: 공간을 가로지름 (‖v‖<1)')
for p, name in [(a, 'v_A'), (b, 'v_B')]:
    ax.scatter(*p, color='k', s=25)
    ax.text(*(p*1.18), name, fontsize=10)
mid = lrp[30]
ax.text(mid[0], mid[1]+0.05, mid[2]-0.28, '‖v‖=%.2f' % np.linalg.norm(mid),
        color=C_RED, fontsize=9)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('(b) 다양체 위의 보간\n직선 보간은 다양체를 벗어난다')
ax.view_init(elev=18, azim=-40)
plt.tight_layout()
plt.savefig('fig1_not_vector_space.png', dpi=140, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------- fig 2
# 짐벌락: 세 회전축의 정렬 + 오일러 속도 사상의 특이화
fig = plt.figure(figsize=(12, 3.6))
alpha0 = deg(30)
for k, b_deg in enumerate([0, 60, 90]):
    ax = fig.add_subplot(1, 4, k+1, projection='3d')
    b = deg(b_deg)
    z_ax  = np.array([0, 0, 1.])                   # yaw 축
    yp_ax = Rz(alpha0) @ np.array([0, 1, 0.])      # pitch 축
    xpp_ax = Rz(alpha0) @ Ry(b) @ np.array([1, 0, 0.])  # roll 축
    for v, c, lb in [(z_ax, C_BLUE, 'yaw축 (z)'), (yp_ax, C_GREEN, "pitch축 (y')"),
                     (xpp_ax, C_RED, "roll축 (x'')")]:
        ax.quiver(0, 0, 0, *v, color=c, lw=2.2, arrow_length_ratio=0.12)
        ax.text(*(v*1.25), lb, color=c, fontsize=8)
    ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ang = np.rad2deg(np.arccos(np.clip(xpp_ax @ z_ax, -1, 1)))
    ax.set_title(f'pitch β={b_deg}°\nroll축–yaw축 사이각 {ang:.0f}°', fontsize=9)
    ax.view_init(elev=18, azim=-50)

ax = fig.add_subplot(1, 4, 4)
bs = np.linspace(-180, 180, 721)
sig = []
for b_deg in bs:
    J = np.column_stack([np.array([0, 0, 1.]), Rz(alpha0) @ np.array([0, 1, 0.]),
                         Rz(alpha0) @ Ry(deg(b_deg)) @ np.array([1, 0, 0.])])
    sig.append(np.linalg.svd(J, compute_uv=False)[-1])
ax.plot(bs, sig, color=C_BLUE, lw=1.8)
ax.axvline(90, color=C_RED, ls='--', lw=1); ax.axvline(-90, color=C_RED, ls='--', lw=1)
ax.set_xlabel('pitch β (°)'); ax.set_ylabel('σ_min')
ax.set_title('오일러 속도→각속도 사상의\n최소 특이값 (β=±90°에서 0)', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_gimbal_lock.png', dpi=140, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------- fig 3
# euler-lerp vs nlerp vs slerp: x축 자취와 각속도
qA = Rot.from_matrix(Rz(deg(90))).as_quat()
qB = Rot.from_matrix(Ry(deg(90))).as_quat()
if qA @ qB < 0: qB = -qB
RA, RB = Rz(deg(90)), Ry(deg(90))
R_AB = RA.T @ RB
w_AB = Rot.from_matrix(R_AB).as_rotvec()
eA, eB = np.array([90., 0., 0.]), np.array([0., 90., 0.])
ts = np.linspace(0, 1, 400)

paths = {
    'euler-lerp': lambda t: euler(*deg(eA + t*(eB - eA))),
    'nlerp':      lambda t: Rot.from_quat(((1-t)*qA + t*qB) /
                                          np.linalg.norm((1-t)*qA + t*qB)).as_matrix(),
    'slerp':      lambda t: RA @ Rot.from_rotvec(t*w_AB).as_matrix(),
}
styles = {'euler-lerp': (C_RED, '-'), 'nlerp': (C_GREEN, '--'), 'slerp': (C_BLUE, '-')}

fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
draw_sphere(ax)
for name, path in paths.items():
    tr = np.array([path(t) @ [1, 0, 0.] for t in ts])
    c, ls = styles[name]
    ax.plot(*tr.T, color=c, ls=ls, lw=2.2 if name != 'nlerp' else 1.8, label=name)
for R0, lb in [(RA, 'x축(A)'), (RB, 'x축(B)')]:
    p = R0 @ [1, 0, 0.]
    ax.scatter(*p, color='k', s=25); ax.text(*(p*1.15), lb, fontsize=9)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('(a) 보간 중 몸체 x축의 자취\n(nlerp와 slerp는 같은 경로, 다른 속도)')
ax.view_init(elev=22, azim=35)

ax = fig.add_subplot(1, 2, 2)
dt = ts[1] - ts[0]
for name, path in paths.items():
    Rs = [path(t) for t in ts]
    sp = [np.linalg.norm(Rot.from_matrix(Rs[i].T @ Rs[i+1]).as_rotvec())/dt
          for i in range(len(ts)-1)]
    c, ls = styles[name]
    ax.plot(ts[:-1], np.rad2deg(sp), color=c, ls=ls, lw=2, label=name)
ax.axhline(120, color=C_GRAY, lw=0.8, ls=':')
ax.text(0.01, 121, '측지 거리 120° = 최적 등속', fontsize=8, color=C_GRAY)
ax.set_xlabel('보간 매개변수 t'); ax.set_ylabel('각속도 ‖ω(t)‖ (°/단위시간)')
ax.set_ylim(90, 140); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title('(b) 각속도 프로파일\neuler-lerp: 6% 긴 경로 / nlerp: 속도 요동 / slerp: 등속')
plt.tight_layout()
plt.savefig('fig3_slerp_vs_lerp.png', dpi=140, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------- fig 4
# 회전 표현 4종의 관계도: 파라미터 개수(9→4→3)·제약·특이점을 한 눈에
# 본문 §1 표 "하나의 회전, 네 벌의 옷"과 §5 지도의 시각적 요약.
fig = plt.figure(figsize=(11, 5.0))

# (a) 파라미터 개수 vs 특이점 여부 — 트레이드오프의 뼈대
ax = fig.add_subplot(1, 2, 1)
# (숫자 개수, 제약 개수, 이름, 색, 특이점 여부)
reps = [('회전행렬 R', 9, 6, C_BLUE,  False),
        ('단위 쿼터니언 q', 4, 1, C_GREEN, False),
        ('축-각 ωθ (rotvec)', 3, 0, '#9467bd', True),
        ('오일러각 (α,β,γ)', 3, 0, C_RED,  True)]
ypos = np.arange(len(reps))[::-1]
for y, (name, n, c_con, col, sing) in zip(ypos, reps):
    dof = n - c_con
    ax.barh(y, n, color=col, alpha=0.28, height=0.55, zorder=2)
    ax.barh(y, dof, color=col, alpha=0.95, height=0.55, zorder=3,
            label=None)
    ax.text(-0.25, y, name, ha='right', va='center', fontsize=10)
    tag = '특이점 있음' if sing else '특이점 없음'
    ax.text(n + 0.2, y, f'{n}개 (제약 {c_con} → 자유도 {dof})\n{tag}',
            ha='left', va='center', fontsize=8,
            color=(C_RED if sing else C_GRAY))
ax.axvline(3, color='k', lw=0.8, ls=':', zorder=1)
ax.text(3, len(reps) - 0.35, '자유도 3\n(SO(3)의 차원)', ha='center',
        va='bottom', fontsize=8, color='k')
ax.set_xlim(-3.4, 12.5); ax.set_ylim(-0.6, len(reps) - 0.1)
ax.set_yticks([]); ax.set_xlabel('저장하는 숫자 개수 →')
ax.set_title('(a) 여분의 숫자 = "특이점 없음"의 값\n'
             '진한 막대 = 자유도 3, 옅은 막대 = 여분(제약)', fontsize=9)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', alpha=0.3)

# (b) 변환 관계도 — 어느 표현이 허브인지, 어디서 깨지는지
ax = fig.add_subplot(1, 2, 2)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_axis_off()
nodes = {
    'R':  (5.0, 8.2, '회전행렬 R\n(9, 제약 6)',        C_BLUE),
    'q':  (5.0, 4.6, '단위 쿼터니언 q\n(4, ‖q‖=1)',    C_GREEN),
    'ax': (1.6, 6.4, '축-각 ωθ\n(3, rotvec)',          '#9467bd'),
    'eu': (8.4, 6.4, '오일러각\n(3, ZYX)',            C_RED),
}
for key, (x, y, lab, col) in nodes.items():
    ax.add_patch(plt.Rectangle((x-1.35, y-0.62), 2.7, 1.24, fc=col,
                               ec='k', lw=1.0, alpha=0.85, zorder=3))
    ax.text(x, y, lab, ha='center', va='center', fontsize=8.5,
            color='white', zorder=4, fontweight='bold')


def arrow(a, b, txt, col, rad=0.0, dy=0.0, dx=0.0):
    xa, ya = nodes[a][0], nodes[a][1]
    xb, yb = nodes[b][0], nodes[b][1]
    ax.annotate('', xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6,
                                shrinkA=26, shrinkB=26,
                                connectionstyle=f'arc3,rad={rad}'), zorder=2)
    mx, my = (xa + xb) / 2 + dx, (ya + yb) / 2 + dy
    ax.text(mx, my, txt, ha='center', va='center', fontsize=7.3, color=col,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                      alpha=0.85), zorder=5)


arrow('ax', 'R', 'Rodrigues (E2)', '#9467bd', rad=0.18, dx=-0.55, dy=0.45)
arrow('R', 'ax', 'log — θ=0,π 특이', C_GRAY, rad=0.18, dx=0.55, dy=-0.55)
arrow('ax', 'q', 'q=(cos θ/2, ω sin θ/2)', '#9467bd', rad=-0.15, dx=-0.2)
arrow('q', 'R', '닫힌 공식 (양방향)', C_GREEN, rad=0.0, dx=1.65)
arrow('eu', 'R', 'Rz·Ry·Rx', C_RED, rad=-0.18, dx=0.55, dy=0.45)
arrow('R', 'eu', 'atan2 — β=±90° 특이', C_RED, rad=-0.18, dx=-0.55, dy=-0.55)
ax.text(5.0, 1.7, '허브 = R 또는 q (특이점 없음)\n'
        '오일러각·축-각은 입출력·미소회전에서만',
        ha='center', va='center', fontsize=8, style='italic',
        color='k', bbox=dict(boxstyle='round,pad=0.4', fc='#fff7e6',
                             ec=C_GRAY, lw=0.8))
ax.set_title('(b) 표현 사이의 변환과 특이점\n'
             '붉은 화살표 = 특이점을 지나는 변환', fontsize=9)
plt.tight_layout()
plt.savefig('fig4_representations_map.png', dpi=140, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------- fig 5
# 회전 합성의 비가환성: (x축 90° → z축 90°) ≠ (z축 90° → x축 90°)
# E3의 "q1⊗q2 ≠ q2⊗q1 (외적 항)" — 회전이 벡터공간이 아닌 또 하나의 증거.
def draw_frame(ax, R, origin=(0, 0, 0), scale=1.0, alpha=1.0, lw=2.4,
               labels=False):
    o = np.asarray(origin, float)
    cols = [C_RED, C_GREEN, C_BLUE]
    names = ['x', 'y', 'z']
    for i in range(3):
        v = R[:, i] * scale
        ax.quiver(*o, *v, color=cols[i], lw=lw, arrow_length_ratio=0.16,
                  alpha=alpha)
        if labels:
            ax.text(*(o + v * 1.18), names[i], color=cols[i], fontsize=9)


def draw_book(ax, R, origin=(0, 0, 0), alpha=0.5):
    # 얇은 직육면체(책): 축 순서를 눈으로 읽게 해 주는 강체
    o = np.asarray(origin, float)
    lx, ly, lz = 0.9, 0.65, 0.12
    verts = np.array([[x, y, z] for x in (0, lx) for y in (0, ly)
                      for z in (0, lz)])
    verts = (R @ (verts - np.array([lx/2, ly/2, lz/2])).T).T + o
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection([verts[f] for f in faces], alpha=alpha,
                            facecolor='#f0c060', edgecolor='k', linewidths=0.7)
    ax.add_collection3d(poly)


Rx90, Rz90 = Rx(deg(90)), Rz(deg(90))
# 순서 A: 먼저 x축 90°, 그다음 (월드축 기준) z축 90°  →  R = Rz90 @ Rx90
# 순서 B: 먼저 z축 90°, 그다음 x축 90°               →  R = Rx90 @ Rz90
RA = Rz90 @ Rx90
RB = Rx90 @ Rz90
diff = np.rad2deg(np.linalg.norm(Rot.from_matrix(RA.T @ RB).as_rotvec()))

fig = plt.figure(figsize=(12, 4.4))
seqs = [
    ('순서 A:  x축 90°  →  z축 90°', [np.eye(3), Rx90, RA], 1),
    ('순서 B:  z축 90°  →  x축 90°', [np.eye(3), Rz90, RB], 2),
]
for row, (title, frames, col0) in enumerate(seqs):
    for k, R in enumerate(frames):
        ax = fig.add_subplot(2, 3, row*3 + k + 1, projection='3d')
        draw_frame(ax, np.eye(3), scale=0.55, alpha=0.18, lw=1.0)  # 월드축(옅게)
        draw_book(ax, R)
        draw_frame(ax, R, scale=1.0, labels=True)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        step = ['시작 (I)', ['x축 90° 후', 'z축 90° 후'][row],
                ['+ z축 90° 후', '+ x축 90° 후'][row]][k]
        ax.set_title(step, fontsize=9)
        ax.view_init(elev=22, azim=-60)
    # 행 제목: 첫 서브플롯 왼쪽에
    fig.text(0.008, 0.75 - row*0.47, title, fontsize=10, rotation=90,
             va='center', ha='center', fontweight='bold',
             color=(C_BLUE if row == 0 else C_GREEN))

fig.suptitle('회전 합성은 비가환: 같은 두 회전을 순서만 바꾸면 최종 자세가 다르다  '
             f'(A와 B의 자세 차 = {diff:.0f}°)', fontsize=11, y=1.02)
plt.tight_layout(rect=[0.02, 0, 1, 0.99])
plt.savefig('fig5_noncommutative.png', dpi=140, bbox_inches='tight')
plt.close()

print("figures written")
