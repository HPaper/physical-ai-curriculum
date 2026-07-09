"""Lec R07 그림 생성 스크립트.
실행: python3 gen_figs.py  (이 디렉토리에서)
출력: fig1_elbow_updown.png, fig2_newton_vs_dls.png, fig3_basin_map.png,
      fig4_dls_lambda_sweep.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

l1, l2 = 1.0, 0.6

def fk(q):
    q1, q2 = q
    return np.array([l1*np.cos(q1) + l2*np.cos(q1+q2),
                     l1*np.sin(q1) + l2*np.sin(q1+q2)])

def jac(q):
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    return np.array([[-l1*s1 - l2*s12, -l2*s12],
                     [ l1*c1 + l2*c12,  l2*c12]])

# ---------------- fig1: elbow-up / elbow-down 두 해 ----------------
target = np.array([0.8, 0.6])
q_down = np.array([0.034116, 1.875489])    # WE-1의 해석해 (elbow-down)
q_up   = np.array([1.252886, -1.875489])   # elbow-up

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
th = np.linspace(0, 2*np.pi, 200)
ax.plot((l1+l2)*np.cos(th), (l1+l2)*np.sin(th), ':', color='gray', lw=1)
ax.plot((l1-l2)*np.cos(th), (l1-l2)*np.sin(th), ':', color='gray', lw=1)
for q, c, lab in [(q_down, 'tab:blue', 'elbow-down  q=(1.95°, +107.46°)'),
                  (q_up, 'tab:red', 'elbow-up    q=(71.79°, −107.46°)')]:
    elbow = np.array([l1*np.cos(q[0]), l1*np.sin(q[0])])
    tip = fk(q)
    ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], 'o-', color=c, lw=3,
            markersize=7, label=lab)
ax.plot(*target, '*', color='black', markersize=16, zorder=5)
ax.annotate('목표 (0.8, 0.6)', target, textcoords='offset points', xytext=(10, -14))
ax.plot(0, 0, 'ks', markersize=9)
ax.set_xlim(-1.0, 1.9); ax.set_ylim(-0.7, 1.7); ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)
ax.set_title('(a) 같은 목표, 두 개의 해')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

ax = axes[1]
ax.fill_between(np.cos(th)*(l1+l2), np.sin(th)*(l1+l2), color='tab:blue', alpha=0.15)
ax.fill_between(np.cos(th)*(l1-l2), np.sin(th)*(l1-l2), color='white')
ax.plot((l1+l2)*np.cos(th), (l1+l2)*np.sin(th), '-', color='tab:red', lw=2)
ax.plot((l1-l2)*np.cos(th), (l1-l2)*np.sin(th), '-', color='tab:red', lw=2)
ax.annotate('내부: 해 2개\n(elbow-up/down)', (0, 1.0), ha='center')
ax.annotate('경계: 해 1개\n(= 특이점, R06)', (0, -1.72), ha='center', color='tab:red')
ax.annotate('바깥: 해 0개', (1.35, 1.45), ha='center')
ax.plot(0, 0, 'ks', markersize=9)
ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1); ax.set_aspect('equal')
ax.set_title('(b) 목표 위치에 따른 해의 개수 (2R, 위치만)')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
fig.tight_layout()
fig.savefig('fig1_elbow_updown.png', dpi=140)
plt.close(fig)

# ---------------- fig2: 특이점 근처 Newton vs DLS ----------------
def run(method, q0, tgt, lam=0.1, iters=50):
    q = np.array(q0, float)
    errs, steps = [], []
    for _ in range(iters):
        e = tgt - fk(q)
        errs.append(np.linalg.norm(e))
        J = jac(q)
        if method == 'newton':
            dq = np.linalg.pinv(J) @ e
        else:
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2*np.eye(2), e)
        steps.append(np.linalg.norm(dq))
        q = q + dq
    errs.append(np.linalg.norm(tgt - fk(q)))
    return np.array(errs), np.array(steps)

tgt = np.array([1.62, 0.0])   # 최대 도달 1.6 바로 바깥
q0 = [0.3, 0.5]
eN, sN = run('newton', q0, tgt)
eD, sD = run('dls', q0, tgt)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.semilogy(eN, 'o-', color='tab:red', markersize=4, label='Newton ($J^+$)')
ax.semilogy(eD, 's-', color='tab:blue', markersize=4, label='DLS (λ=0.1)')
ax.axhline(0.02, color='gray', ls='--', lw=1)
ax.text(0.98, 0.08, '점선: 이론적 최소 잔차 0.02 (목표가 경계 0.02 바깥)',
        transform=ax.transAxes, ha='right', fontsize=9, color='dimgray',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
ax.set_xlabel('반복 k'); ax.set_ylabel('‖e‖ (log)')
ax.set_title('(a) 도달 불가 목표 (1.62, 0): 오차')
ax.legend()
ax = axes[1]
ax.semilogy(sN, 'o-', color='tab:red', markersize=4, label='Newton ($J^+$)')
ax.semilogy(sD, 's-', color='tab:blue', markersize=4, label='DLS (λ=0.1)')
ax.set_xlabel('반복 k'); ax.set_ylabel('‖Δq‖ (log)')
ax.set_title('(b) 스텝 크기: Newton은 경계 특이점에서 폭발')
ax.legend()
fig.tight_layout()
fig.savefig('fig2_newton_vs_dls.png', dpi=140)
plt.close(fig)

# ---------------- fig3: 초기값 → 수렴해 basin 지도 (벡터화 Newton) ----------------
tgt = np.array([0.8, 0.6])
n = 241
g = np.linspace(-np.pi, np.pi, n)
Q1, Q2 = np.meshgrid(g, g)
q1, q2 = Q1.ravel().copy(), Q2.ravel().copy()
alive = np.ones(q1.shape, bool)
for _ in range(60):
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    ex = tgt[0] - (l1*c1 + l2*c12)
    ey = tgt[1] - (l1*s1 + l2*s12)
    J11, J12 = -l1*s1 - l2*s12, -l2*s12
    J21, J22 = l1*c1 + l2*c12, l2*c12
    det = J11*J22 - J12*J21
    ok = alive & (np.abs(det) > 1e-12)
    dq1 = np.where(ok, ( J22*ex - J12*ey)/np.where(ok, det, 1.0), 0.0)
    dq2 = np.where(ok, (-J21*ex + J11*ey)/np.where(ok, det, 1.0), 0.0)
    big = np.hypot(dq1, dq2) > 50            # 발산 가드: 그 점은 동결
    alive &= ~big
    dq1[~alive] = 0.0; dq2[~alive] = 0.0
    q1 += dq1; q2 += dq2
ex = tgt[0] - (l1*np.cos(q1) + l2*np.cos(q1+q2))
ey = tgt[1] - (l1*np.sin(q1) + l2*np.sin(q1+q2))
conv = np.hypot(ex, ey) < 1e-6
basin = np.zeros(q1.shape, int)
basin[conv & (np.sin(q2) > 0)] = 1           # elbow-down
basin[conv & (np.sin(q2) <= 0)] = 2          # elbow-up
basin = basin.reshape(n, n)

fig, ax = plt.subplots(figsize=(7.2, 6))
cmap = matplotlib.colors.ListedColormap(['#404040', '#aecbe8', '#f2b8b5'])
ax.pcolormesh(g, g, basin, cmap=cmap, vmin=0, vmax=2, shading='auto')
ax.plot(0.034116, 1.875489, '*', color='tab:blue', markersize=18,
        markeredgecolor='k', label='해 A: elbow-down (1.95°, 107.46°)')
ax.plot(1.252886, -1.875489, '*', color='tab:red', markersize=18,
        markeredgecolor='k', label='해 B: elbow-up (71.79°, −107.46°)')
ax.set_xlabel('초기값 $q_1(0)$ [rad]'); ax.set_ylabel('초기값 $q_2(0)$ [rad]')
ax.set_title('Newton IK의 수렴 분지(basin): 초기값이 해를 고른다\n(회색 = 미수렴, 목표 (0.8, 0.6))')
ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig('fig3_basin_map.png', dpi=140)
plt.close(fig)

# ---------------- fig4: λ 스윕 — 특이점 통과 안정성 (E3 시각화 + 실습2) ----------------
# (a) E3의 특이값 증폭률: Newton의 1/σ 와 DLS의 σ/(σ²+λ²).
#     최대값이 σ=λ 에서 정확히 1/(2λ) 임을 세 λ에 대해 보인다.
# (b) 실습 2의 경로 추종: 직선 (1.5,0.8)→(1.5,-0.8) 을 warm-start 로 따라가며
#     스텝당 최대 ‖Δq‖. |y|>√(1.6²−1.5²)≈0.557 인 양 끝이 작업공간 바깥 —
#     Newton은 경계를 넘나드는 순간 스텝이 폭발하고, DLS는 λ가 스텝 상한을 통제한다.

def track_step(method, lam, path, iters=20, q_start=(0.5, 1.0)):
    """직선 경로를 warm start 로 추종. 스텝당 최대 ‖Δq‖ 이력을 반환."""
    q = np.array(q_start, float)
    maxstep = []
    for xd in path:
        st = 0.0
        for _ in range(iters):
            e = xd - fk(q); J = jac(q)
            if method == 'newton':
                dq = np.linalg.pinv(J) @ e
            else:
                dq = J.T @ np.linalg.solve(J @ J.T + lam**2*np.eye(2), e)
            st = max(st, np.linalg.norm(dq))
            q = q + dq
        maxstep.append(st)
    return np.array(maxstep)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 특이값 증폭률
ax = axes[0]
sig = np.linspace(1e-3, 1.4, 600)
ax.plot(sig, 1.0/sig, '--', color='tab:red', lw=2, label='Newton  $1/\\sigma$ (상한 없음)')
lam_list = [0.05, 0.1, 0.3]
colors = ['#1a5fb4', '#3584e4', '#99c1f1']
for lam, col in zip(lam_list, colors):
    amp = sig/(sig**2 + lam**2)
    ax.plot(sig, amp, '-', color=col, lw=2.2,
            label=f'DLS λ={lam}:  최대 $1/2\\lambda$={1/(2*lam):.1f}')
    ax.plot(lam, 1/(2*lam), 'o', color=col, markersize=7, markeredgecolor='k', zorder=5)
ax.set_ylim(0, 22)
ax.set_xlabel('특이값 $\\sigma$  [m/rad]')
ax.set_ylabel('증폭률  $\\Delta q$ / $e$')
ax.set_title('(a) E3의 특이값 증폭률: $\\sigma\\!\\to\\!0$ 에서\nNewton은 발산, DLS는 $\\sigma\\!=\\!\\lambda$ 에서 $1/2\\lambda$ 로 포화')
ax.legend(loc='upper right', fontsize=8.5)
ax.grid(alpha=0.25)

# (b) 실습 2 경로 추종: 스텝 크기
ax = axes[1]
ys = np.linspace(0.8, -0.8, 100)
path = [np.array([1.5, y]) for y in ys]
# 바깥은 경로 양 끝(연속 두 구간): |y| > y_cross = √(1.6²−1.5²) ≈ 0.557
y_cross = np.sqrt((l1+l2)**2 - 1.5**2)
ax.axvspan(0.8, y_cross, color='0.85', lw=0)
ax.axvspan(-y_cross, -0.8, color='0.85', lw=0)
ax.text(0.7, 200, '작업공간\n바깥', fontsize=8.5, color='dimgray', ha='center', va='top')

msN = track_step('newton', 0.0, path)
ax.semilogy(ys, msN, 'o-', color='tab:red', markersize=3, lw=1.3, label='Newton ($J^+$)')
for lam, col in zip(lam_list, colors):
    ms = track_step('dls', lam, path)
    ax.semilogy(ys, ms, 's-', color=col, markersize=3, lw=1.6, label=f'DLS λ={lam}')
ax.axvline(y_cross, color='k', ls=':', lw=1); ax.axvline(-y_cross, color='k', ls=':', lw=1)
ax.set_xlim(0.85, -0.85)      # 경로 진행 방향(위→아래)에 맞춰 x 반전
ax.set_xlabel('경로 위치  $y$  (목표 $x=1.5$ 고정)')
ax.set_ylabel('스텝당 최대 ‖Δq‖  [rad] (log)')
ax.set_title('(b) 실습 2 — 경계 두 번 통과: Newton은 폭발,\nλ가 스텝 상한을 통제 (큰 λ = 더 안전, 더 굼뜸)')
ax.legend(loc='lower center', fontsize=8.5, ncol=2)
fig.tight_layout()
fig.savefig('fig4_dls_lambda_sweep.png', dpi=140)
plt.close(fig)

print('figures written')
