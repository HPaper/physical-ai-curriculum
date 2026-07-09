"""Lec R03 그림 생성: 프레임 체인 3D + 표기 실수 오차 증폭."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = os.path.dirname(os.path.abspath(__file__))


def make_T(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T

def inv_T(T):
    R, p = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ p)

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Worked Example의 수치 (본문과 동일)
T_bc = make_T(rot_x(np.pi), np.array([0.4, 0.0, 0.8]))
T_co = make_T(rot_z(np.pi/2), np.array([0.1, 0.2, 0.6]))
T_bo = T_bc @ T_co

# ---------------- Fig 1: 프레임 체인 3D ----------------
fig = plt.figure(figsize=(10, 6.5))
ax = fig.add_subplot(111, projection='3d')

def draw_frame(ax, T, name, scale=0.18, lw=2.2, off=(0, 0, -0.09)):
    o = T[:3, 3]
    cols = ['#d62728', '#2ca02c', '#1f77b4']   # x=빨강, y=초록, z=파랑
    for k in range(3):
        a = T[:3, k] * scale
        ax.quiver(o[0], o[1], o[2], a[0], a[1], a[2],
                  color=cols[k], lw=lw, arrow_length_ratio=0.22)
    ax.text(o[0] + off[0], o[1] + off[1], o[2] + off[2], name,
            fontsize=13, weight='bold')

# 테이블(z=0.2 평면)과 바닥 느낌
xx, yy = np.meshgrid(np.linspace(0.1, 0.9, 2), np.linspace(-0.5, 0.4, 2))
ax.plot_surface(xx, yy, np.full_like(xx, 0.2), alpha=0.12, color='gray')

draw_frame(ax, np.eye(4), '{b} base', off=(-0.32, 0, -0.10))
draw_frame(ax, T_bc, '{c} camera', off=(0.04, 0, 0.07))
draw_frame(ax, T_bo, '{o} object', off=(0.14, -0.10, 0.06))

def dashed(ax, p, q, color, label, frac=0.5, dz=0.03):
    ax.plot(*zip(p, q), ls='--', color=color, lw=1.5)
    m = (1 - frac) * np.asarray(p) + frac * np.asarray(q)
    ax.text(m[0], m[1], m[2] + dz, label, fontsize=12, color=color)

dashed(ax, [0, 0, 0], T_bc[:3, 3], '#9467bd', r'$T_{bc}$ (extrinsic)', 0.30, 0.06)
dashed(ax, T_bc[:3, 3], T_bo[:3, 3], '#ff7f0e', r'$T_{co}$ (detection)', 0.42, 0.03)
dashed(ax, [0, 0, 0], T_bo[:3, 3], '#17becf', r'$T_{bo}=T_{bc}\,T_{co}$', 0.30, -0.14)

ax.set_xlim(-0.15, 0.95); ax.set_ylim(-0.55, 0.45); ax.set_zlim(0, 1.0)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
ax.set_box_aspect((1.1, 1.0, 1.0))
ax.view_init(elev=22, azim=-50)
ax.set_title('좌표계 체인: 아래를 보는 카메라 {c}가 물체 {o}를 검출 → base {b}로 합성\n'
             r'(축 색: x=빨강, y=초록, z=파랑. 카메라는 z축이 테이블을 향한다)',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_frame_chain.png'), dpi=140)
plt.close()

# ---------------- Fig 2: 오차 증폭 ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) 회전 캘리브 오차 → 물체 위치 오차 (지렛대 길이별)
th = np.linspace(0, 5, 100)                       # deg
for d, c in [(0.3, '#1f77b4'), (0.6403, '#ff7f0e'), (1.0, '#d62728')]:
    err = 2 * np.sin(np.deg2rad(th) / 2) * d * 1000
    lbl = f'd = {d:.2f} m' + ('  (WE-1의 카메라-물체 거리)' if abs(d - 0.6403) < 1e-3 else '')
    a1.plot(th, err, color=c, lw=2, label=lbl)
a1.axvline(1.0, color='gray', ls=':', lw=1)
a1.annotate('1°에서 이미 수 mm~1cm', xy=(1.0, 11), xytext=(1.6, 40),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
a1.set_xlabel('카메라 자세(회전) 캘리브레이션 오차 [deg]')
a1.set_ylabel('물체 위치 오차 [mm]')
a1.set_title('(a) 작은 회전 오차 × 지렛대 거리 = 위치 오차')
a1.legend(fontsize=9); a1.grid(alpha=0.3)

# (b) 표기 버그의 오차 (검증 코드의 실제 수치, log 스케일)
cases = ['캘리브 오차 1°', '버그 A\n역변환 방향 뒤집힘\n$T_{cb}$를 $T_{bc}$ 자리에',
         '버그 C\n곱셈 순서 뒤집힘\n$T_{co}T_{bc}$', '버그 B\n역변환을 $[R^T\\!,\\,-p]$로']
vals = [11.04, 800, 1497, 1600]
cols = ['#2ca02c', '#d62728', '#d62728', '#d62728']
bars = a2.bar(range(4), vals, color=cols, alpha=0.85)
a2.set_yscale('log')
a2.set_xticks(range(4)); a2.set_xticklabels(cases, fontsize=8.5)
a2.set_ylabel('물체 위치 오차 [mm] (log)')
a2.set_title('(b) WE-1 체인에서 실수 유형별 오차 (실측)')
for b, v in zip(bars, vals):
    a2.text(b.get_x() + b.get_width() / 2, v * 1.15, f'{v:.0f} mm'
            if v > 100 else f'{v:.1f} mm', ha='center', fontsize=9.5)
a2.set_ylim(1, 6000); a2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_bug_amplification.png'), dpi=140)
plt.close()
print('saved figs to', OUT)
