# Lec R06 그림 생성 스크립트
# 실행: python3 gen_figs.py  (출력: fig1~fig4 PNG, 이 디렉토리에 저장)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = os.path.dirname(os.path.abspath(__file__))

L1, L2 = 1.0, 0.6   # R01과 같은 2R 팔

def J2R(q, l1=L1, l2=L2):
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([[-l1*s1 - l2*s12, -l2*s12],
                     [ l1*c1 + l2*c12,  l2*c12]])

def fk2(q, l1=L1, l2=L2):
    p1 = np.array([l1*np.cos(q[0]), l1*np.sin(q[0])])
    p2 = p1 + np.array([l2*np.cos(q[0]+q[1]), l2*np.sin(q[0]+q[1])])
    return p1, p2

def ik2_elbow_up(x, y, l1=L1, l2=L2):
    r2 = x*x + y*y
    c2 = (r2 - l1*l1 - l2*l2) / (2*l1*l2)
    c2 = np.clip(c2, -1, 1)
    q2 = -np.arccos(c2)                       # elbow-up 가지
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1 + l2*np.cos(q2))
    return np.array([q1, q2])

# ---------------- fig 1: 조작성 히트맵 + 타원 오버레이 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# (a) w = l1 l2 |sin q2| 를 작업공간 (x,y) 위에 그리기 (반경만의 함수)
ax = axes[0]
n = 500
xs = np.linspace(-1.75, 1.75, n)
X, Y = np.meshgrid(xs, xs)
R2 = X**2 + Y**2
c2 = (R2 - L1**2 - L2**2) / (2*L1*L2)
W = np.full_like(X, np.nan)
inside = np.abs(c2) <= 1.0
W[inside] = L1*L2*np.sqrt(1.0 - c2[inside]**2)   # = l1 l2 |sin q2|
im = ax.pcolormesh(X, Y, W, shading='auto', cmap='viridis')
for r in [abs(L1-L2), L1+L2]:
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(r*np.cos(th), r*np.sin(th), 'r--', lw=1.2)
ax.plot(0, 0, 'k^', ms=9)
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('(a) 2R 작업공간의 조작성 w 히트맵\n빨간 점선 = 특이점 (w=0): 완전 신전/완전 접힘')
fig.colorbar(im, ax=ax, label='w = $\\sigma_1\\sigma_2$')

# (b) 자세별 조작성 타원 (elbow-up으로 +x축 위 목표 도달)
ax = axes[1]
targets = [0.55, 0.85, 1.15, 1.45, 1.595]
th = np.linspace(0, 2*np.pi, 100)
circ = np.vstack([np.cos(th), np.sin(th)])     # 단위 관절속도 원
cmap = plt.get_cmap('viridis')
for i, xt in enumerate(targets):
    q = ik2_elbow_up(xt, 0.0)
    p1, p2 = fk2(q)
    col = cmap(0.15 + 0.7*i/(len(targets)-1))
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], '-o', color=col,
            lw=2, ms=4, alpha=0.85)
    E = 0.22 * (J2R(q) @ circ)                 # 타원 = J·(단위원), 표시용 스케일
    ax.plot(p2[0] + E[0], p2[1] + E[1], color=col, lw=1.6)
    w = np.prod(np.linalg.svd(J2R(q), compute_uv=False))
    lab_off = [(-0.13, 0.16), (-0.13, 0.16), (-0.13, 0.16),
               (-0.16, -0.38), (-0.10, -0.52)][i]
    ax.annotate(f'w={w:.2f}', p2 + np.array(lab_off), color=col, fontsize=9)
ax.plot(0, 0, 'k^', ms=9)
ax.axhline(0, color='gray', lw=0.5)
ax.set_aspect('equal'); ax.set_xlim(-0.4, 2.0); ax.set_ylim(-0.85, 1.1)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('(b) 자세별 조작성 타원 (스케일 0.22)\n바깥 경계로 갈수록 타원이 바늘로 붕괴')
fig.tight_layout()
fig.savefig(f'{OUT}/fig1_manipulability_map.png', dpi=140)
plt.close(fig)

# ---------------- fig 2: 특이값 추이 + 관절속도 폭증 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) q2에 따른 특이값 (l1=l2=1, WE-1과 동일 조건)
ax = axes[0]
q2s = np.linspace(0, np.pi, 300)
S1, S2 = [], []
for q2 in q2s:
    s = np.linalg.svd(J2R([0.0, q2], 1.0, 1.0), compute_uv=False)
    S1.append(s[0]); S2.append(s[1])
ax.plot(np.rad2deg(q2s), S1, label='$\\sigma_1$ (긴 반축)', lw=2)
ax.plot(np.rad2deg(q2s), S2, label='$\\sigma_2$ (짧은 반축)', lw=2)
ax.plot(np.rad2deg(q2s), np.array(S1)*np.array(S2), '--', label='w = $\\sigma_1\\sigma_2$ = |sin $q_2$|', lw=1.5)
ax.axvline(0, color='r', ls=':'); ax.axvline(180, color='r', ls=':')
ax.set_xlabel('$q_2$ [deg]'); ax.set_ylabel('특이값')
ax.set_title('(a) 특이값 추이 (2R, $l_1$=$l_2$=1, $q_1$=0)\n$q_2$→0°/180°에서 $\\sigma_2$→0: rank 하락')
ax.legend(); ax.grid(alpha=0.3)

# (b) 특이점 접근 시 요구 관절속도 (radial 방향 1 m/s)
ax = axes[1]
q2s = np.deg2rad(np.linspace(0.2, 90, 400))
lam = 0.1
n_inv, n_dls = [], []
for q2 in q2s:
    J = J2R([0.0, q2], 1.0, 1.0)
    v = np.array([1.0, 0.0])                   # q1=0이면 거의 radial
    n_inv.append(np.linalg.norm(np.linalg.solve(J, v)))
    qd = J.T @ np.linalg.solve(J @ J.T + lam**2*np.eye(2), v)
    n_dls.append(np.linalg.norm(qd))
ax.semilogy(np.rad2deg(q2s), n_inv, label='$\\|J^{-1}v\\|$ (정확한 역행렬)', lw=2)
ax.semilogy(np.rad2deg(q2s), n_dls, label='$\\|J_{DLS}\\,v\\|$ (damping $\\lambda$=0.1)', lw=2)
ax.axhline(1/(2*lam), color='gray', ls='--', lw=1)
ax.annotate('DLS 상한 1/(2$\\lambda$) = 5', (35, 5.6), color='gray', fontsize=9)
ax.set_xlabel('$q_2$ [deg]  (0° = 특이점)'); ax.set_ylabel('$\\|\\dot q\\|$ [rad/s]  (log)')
ax.set_title('(b) radial 1 m/s 요구 시 관절속도\n특이점 접근: 역행렬은 폭증, DLS는 포기하고 유계')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f'{OUT}/fig2_singularity_blowup.png', dpi=140)
plt.close(fig)

# ---------------- fig 3: 3R null-space 자기운동 ----------------
L3 = np.array([0.6, 0.5, 0.4])

def fk3(q):
    a = np.cumsum(q)
    return np.array([np.sum(L3*np.cos(a)), np.sum(L3*np.sin(a))])

def joints3(q):
    a = np.cumsum(q)
    pts = [np.zeros(2)]
    for li, ai in zip(L3, a):
        pts.append(pts[-1] + li*np.array([np.cos(ai), np.sin(ai)]))
    return np.array(pts)

def J3(q):
    a = np.cumsum(q)
    Jm = np.zeros((2, 3))
    for j in range(3):
        Jm[0, j] = -np.sum(L3[j:]*np.sin(a[j:]))
        Jm[1, j] =  np.sum(L3[j:]*np.cos(a[j:]))
    return Jm

def manip3(q):
    Jm = J3(q)
    return np.sqrt(np.linalg.det(Jm @ Jm.T))

def grad_manip3(q, h=1e-6):
    g = np.zeros(3)
    for j in range(3):
        qp, qm = q.copy(), q.copy()
        qp[j] += h; qm[j] -= h
        g[j] = (manip3(qp) - manip3(qm)) / (2*h)
    return g

q = np.array([0.9, -1.5, 1.3])
x_star = fk3(q)
dt, K, steps = 0.01, 5.0, 600
traj_q, traj_w = [q.copy()], [manip3(q)]
for _ in range(steps):
    Jm = J3(q)
    Jp = Jm.T @ np.linalg.inv(Jm @ Jm.T)
    e = x_star - fk3(q)
    qdot = Jp @ (K*e) + (np.eye(3) - Jp @ Jm) @ (2.0*grad_manip3(q))
    q = q + dt*qdot
    traj_q.append(q.copy()); traj_w.append(manip3(q))
traj_q = np.array(traj_q); traj_w = np.array(traj_w)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
cmap = plt.get_cmap('viridis')
for k in np.linspace(0, steps, 9).astype(int):
    pts = joints3(traj_q[k])
    ax.plot(pts[:, 0], pts[:, 1], '-o', color=cmap(k/steps), lw=2, ms=4, alpha=0.8)
ax.plot(*x_star, 'r*', ms=16, zorder=5)
ax.annotate('EEF 고정', x_star + np.array([-0.13, -0.12]), color='r')
ax.plot(0, 0, 'k^', ms=9)
ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('(a) null-space 자기운동: EEF는 그대로, 팔꿈치만 이동\n(색: 시간 진행, 보라→노랑)')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, steps*dt))
fig.colorbar(sm, ax=ax, label='t [s]')

ax = axes[1]
t = np.arange(steps+1)*dt
ax.plot(t, traj_w, lw=2)
ax.set_xlabel('t [s]'); ax.set_ylabel('w(q)')
ax.set_title(f'(b) null-space에서 ∇w 등반: w {traj_w[0]:.3f} → {traj_w[-1]:.3f}\nEEF 이동 없이 조작성만 개선')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_nullspace_motion.png', dpi=140)
plt.close(fig)

# ---------------- fig 4: J = UΣVᵀ 파이프라인 (회전→늘이기→회전) ----------------
# E1을 그대로 시각화: 단위 관절속도 원 → V^T(회전) → Σ(축별 늘이기) → U(회전) = 조작성 타원.
# 건강한 자세 vs 특이점 근처 자세를 비교해, σ_min→0에서 타원이 바늘로 붕괴하고
# 잃은 EEF 방향 u_min / null 관절 방향 v_min 을 U·V에서 직접 읽는다. (l1=l2=1, WE-1 조건)
th = np.linspace(0, 2*np.pi, 200)
circ = np.vstack([np.cos(th), np.sin(th)])          # 단위 관절속도 원 {||q̇||=1}

def _draw_axes(ax, lim):
    ax.axhline(0, color='0.7', lw=0.6, zorder=0)
    ax.axvline(0, color='0.7', lw=0.6, zorder=0)
    ax.set_aspect('equal'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])

def svd_pipeline(fig, gs_row, q, tag, healthy=True):
    J = J2R(q, 1.0, 1.0)
    U, S, Vt = np.linalg.svd(J)
    V = Vt.T
    # 부호 정규화(보기 좋게): 각 특이 방향의 x성분이 음수면 뒤집는다
    for k in range(2):
        if U[0, k] < 0: U[:, k] = -U[:, k]
        if V[0, k] < 0: V[:, k] = -V[:, k]
    col = ['#1f77b4', '#d62728']                     # σ1(파랑), σ2(빨강)
    lim = 2.6

    # 패널 1: 입력 공간 — 단위 원 + V의 두 방향(주축)
    ax = fig.add_subplot(gs_row[0]); _draw_axes(ax, lim)
    ax.plot(circ[0], circ[1], color='0.4', lw=1.8)
    for k in range(2):
        ax.annotate('', xy=V[:, k], xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col[k], lw=2.2))
        ax.text(1.28*V[0, k], 1.28*V[1, k], f'$v_{k+1}$', color=col[k],
                fontsize=11, ha='center', va='center')
    ax.set_title(f'① 관절속도 공간\n단위원 + 주입력방향 $V$', fontsize=10)
    ax.set_xlabel('$\\dot q_1$', fontsize=9); ax.set_ylabel('$\\dot q_2$', fontsize=9)

    # 패널 2: Σ 적용 후(아직 U 회전 전) — 축정렬 타원, 반축 = σ_i
    ax = fig.add_subplot(gs_row[1]); _draw_axes(ax, lim)
    ell = np.diag(S) @ (Vt @ circ)                   # ΣVᵀ·(원): 축정렬 타원
    ax.plot(ell[0], ell[1], color='0.4', lw=1.8)
    for k in range(2):
        e = np.zeros(2); e[k] = S[k]
        ax.annotate('', xy=e, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col[k], lw=2.2))
        ax.text(e[0] + (0.18 if k == 0 else 0.0), e[1] + (0.0 if k == 0 else 0.22),
                f'$\\sigma_{k+1}$={S[k]:.2f}', color=col[k], fontsize=10,
                ha='left' if k == 0 else 'center')
    ax.set_title('② $\\Sigma$: 축별로 $\\sigma_i$배 늘이기', fontsize=10)

    # 패널 3: U 회전 후 = 조작성 타원 (EEF 속도 공간)
    ax = fig.add_subplot(gs_row[2]); _draw_axes(ax, lim)
    manip_ell = J @ circ                             # = UΣVᵀ·(원)
    ax.plot(manip_ell[0], manip_ell[1], color='0.4', lw=1.8)
    ax.fill(manip_ell[0], manip_ell[1], color='0.4', alpha=0.10)
    for k in range(2):
        a = S[k] * U[:, k]
        ax.annotate('', xy=a, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col[k], lw=2.2))
        ax.text(1.14*a[0], 1.14*a[1] + (0.16 if k == 1 else 0.0),
                f'$\\sigma_{k+1}u_{k+1}$', color=col[k], fontsize=10,
                ha='center', va='center')
    if not healthy:
        # 잃어가는 EEF 방향(σ2 u2)을 강조
        ax.annotate('잃어가는\n방향 $u_2$', xy=1.02*S[1]*U[:, 1],
                    xytext=(-1.7, 1.7), color=col[1], fontsize=9,
                    arrowprops=dict(arrowstyle='->', color=col[1], lw=1.3))
    ax.set_title('③ $U$: EEF 속도 공간의 조작성 타원', fontsize=10)
    ax.set_xlabel('$\\dot x$ [m/s]', fontsize=9); ax.set_ylabel('$\\dot y$ [m/s]', fontsize=9)

    q2deg = np.rad2deg(q[1])
    kappa = S[0] / S[1] if S[1] > 1e-12 else np.inf
    ktxt = '∞' if not np.isfinite(kappa) else f'{kappa:.0f}'
    return S, U, V, q2deg, ktxt

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12, 7.4))
outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.42)
poses = [
    (np.array([0.0, np.pi/2]), '건강한 자세  $q_2$=90°', True),
    (np.array([0.0, np.deg2rad(8.0)]), '특이점 근처  $q_2$=8°', False),
]
row_info = []
for r, (q, label, healthy) in enumerate(poses):
    gs_row = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[r], wspace=0.28)
    S, U, V, q2deg, ktxt = svd_pipeline(fig, gs_row, q, label, healthy)
    row_info.append((label, S, ktxt))

# 행 제목(세로 라벨)과 행별 스펙트럼 요약을 배치.
# 스펙트럼 요약은 각 행 패널 ①/②/③ 사이의 세로 중앙 빈틈에 박스로 얹는다(제목과 안 겹침).
label_y = [0.74, 0.30]     # 세로 라벨: 각 행 패널의 세로 중앙
gap_y   = [0.74, 0.30]     # 스펙트럼 박스: 각 행 세로 중앙
for r, (label, S, ktxt) in enumerate(row_info):
    w = S[0]*S[1]
    fig.text(0.012, label_y[r], label, rotation=90, va='center', ha='left',
             fontsize=11, fontweight='bold', color='#333')
    # ①-② 사이 빈틈(x≈0.365)과 ②-③ 사이 빈틈(x≈0.675) 중 왼쪽 빈틈에 배치
    fig.text(0.365, gap_y[r] + 0.135,
             f'$\\sigma$=({S[0]:.2f}, {S[1]:.2f})\n$w$={w:.2f}   $\\kappa$={ktxt}',
             va='center', ha='center', fontsize=9.5, color='#444',
             bbox=dict(boxstyle='round,pad=0.3', fc='#f7f7f7', ec='#bbb', lw=0.8))

fig.suptitle('$J = U\\Sigma V^\\top$: 단위 관절속도 원이 조작성 타원으로 (E1)\n'
             '위=통통한 타원(어디든 잘 감), 아래=$\\sigma_2$가 작아 바늘로 붕괴 ($u_2$ 방향이 사라짐)',
             fontsize=12.5, y=0.99)
fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.06)
fig.savefig(f'{OUT}/fig4_svd_pipeline.png', dpi=140)
plt.close(fig)

print('figures written to', OUT)
