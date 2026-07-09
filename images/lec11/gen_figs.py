"""Lec R11 그림 생성: RNEA 2-pass 다이어그램 / 라그랑주 vs RNEA 일치 / O(n) 스케일링.

실행: python3 gen_figs.py  (이 디렉토리에서)
출력: fig1_rnea_two_pass.png, fig2_lagrange_vs_rnea.png, fig3_scaling.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

G_ACC = 9.81

# ---------------- 평면 n-링크 RNEA (본문 WE-2와 동일 구현) ----------------
def rnea_planar(q, qd, qdd, l, m, c, Izz, gravity=G_ACC):
    n = len(q)
    perp = lambda v: np.array([-v[1], v[0]])
    cross2 = lambda a, b: a[0]*b[1] - a[1]*b[0]
    w = np.zeros(n); al = np.zeros(n); a_c = np.zeros((n, 2))
    w_prev, al_prev, a_prev = 0.0, 0.0, np.array([0.0, gravity])
    for i in range(n):
        ci, si = np.cos(q[i]), np.sin(q[i])
        R_T = np.array([[ci, si], [-si, ci]])
        p = np.array([l[i-1], 0.0]) if i > 0 else np.zeros(2)
        w[i] = w_prev + qd[i]; al[i] = al_prev + qdd[i]
        a_o = R_T @ (a_prev + al_prev*perp(p) - w_prev**2 * p)
        cc = np.array([c[i], 0.0])
        a_c[i] = a_o + al[i]*perp(cc) - w[i]**2 * cc
        w_prev, al_prev, a_prev = w[i], al[i], a_o
    tau = np.zeros(n); f_next = np.zeros(2); n_next = 0.0
    for i in range(n-1, -1, -1):
        if i < n-1:
            cj, sj = np.cos(q[i+1]), np.sin(q[i+1])
            f_child = np.array([[cj, -sj], [sj, cj]]) @ f_next
            n_child = n_next + cross2(np.array([l[i], 0.0]), f_child)
        else:
            f_child = np.zeros(2); n_child = 0.0
        cc = np.array([c[i], 0.0])
        f_i = m[i]*a_c[i] + f_child
        tau[i] = Izz[i]*al[i] + cross2(cc, m[i]*a_c[i]) + n_child
        f_next, n_next = f_i, tau[i]
    return tau

def lagrange_2link(q, qd, qdd, l1, l2, m1, m2, g=G_ACC):
    c2, s2 = np.cos(q[1]), np.sin(q[1])
    M = np.array([[(m1+m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*c2, m2*l2**2 + m2*l1*l2*c2],
                  [m2*l2**2 + m2*l1*l2*c2,                   m2*l2**2]])
    Cv = np.array([-m2*l1*l2*s2*(2*qd[0]*qd[1] + qd[1]**2), m2*l1*l2*s2*qd[0]**2])
    gv = np.array([(m1+m2)*g*l1*np.cos(q[0]) + m2*g*l2*np.cos(q[0]+q[1]),
                   m2*g*l2*np.cos(q[0]+q[1])])
    return M @ np.asarray(qdd) + Cv + gv

C_FWD, C_BWD, C_LINK, C_JNT = '#1f77b4', '#d62728', '#555555', '#222222'

# ================= fig1: RNEA 2-pass 다이어그램 =================
def draw_arm(ax, qs, L, direction):
    pts = [np.zeros(2)]; a = 0.0
    for qi, li in zip(qs, L):
        a += qi
        pts.append(pts[-1] + li*np.array([np.cos(a), np.sin(a)]))
    pts = np.array(pts)
    ax.plot([-0.25, 0.25], [0, 0], color=C_LINK, lw=5)
    for x in np.linspace(-0.22, 0.22, 6):
        ax.plot([x, x-0.07], [0, -0.09], color=C_LINK, lw=1.2)
    for i in range(3):
        ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], color=C_LINK, lw=7,
                solid_capstyle='round', zorder=2)
    for i in range(3):
        ax.plot(*pts[i], 'o', ms=13, mfc='white', mec=C_JNT, mew=2, zorder=3)
        ax.plot(*pts[i], 'o', ms=4, color=C_JNT, zorder=4)
    ax.plot(*pts[3], 's', ms=9, color=C_JNT, zorder=3)
    col = C_FWD if direction == 'fwd' else C_BWD
    rng_i = range(3) if direction == 'fwd' else range(2, -1, -1)
    for k, i in enumerate(rng_i):
        p0, p1 = pts[i], pts[i+1]
        mid = 0.5*(p0+p1)
        d = (p1-p0)/np.linalg.norm(p1-p0)
        nvec = np.array([-d[1], d[0]])*0.16
        if direction == 'fwd':
            src, dst = mid - 0.28*d + nvec, mid + 0.28*d + nvec
        else:
            src, dst = mid + 0.28*d + nvec, mid - 0.28*d + nvec
        ax.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.6,
                                    shrinkA=0, shrinkB=0))
        lbl = (r'$\mathcal{V}_%d,\ \dot{\mathcal{V}}_%d$' % (i+1, i+1)) if direction == 'fwd' \
              else (r'$\mathcal{F}_%d,\ \tau_%d$' % (i+1, i+1))
        ax.annotate(lbl, mid + nvec*2.4, ha='center', fontsize=12, color=col)
    return pts

qs, L = [0.72, -0.5, -0.35], [0.95, 0.8, 0.6]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, direction, title, sub in [
        (axes[0], 'fwd', '① 순방향 패스 (기저→말단)',
         '속도·가속도 전파:  $\\mathcal{V}_i,\\dot{\\mathcal{V}}_i$ = f(부모의 값, $\\dot q_i, \\ddot q_i$)'),
        (axes[1], 'bwd', '② 역방향 패스 (말단→기저)',
         '힘·토크 전파:  $\\mathcal{F}_i$ = 자기 관성력 + 자식의 $\\mathcal{F}_{i+1}$,  $\\tau_i = \\mathcal{A}_i^{\\top}\\mathcal{F}_i$')]:
    draw_arm(ax, qs, L, direction)
    ax.set_title(title, fontsize=13,
                 color=C_FWD if direction == 'fwd' else C_BWD, pad=8)
    ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha='center', fontsize=10.5)
    ax.set_xlim(-0.6, 2.4); ax.set_ylim(-0.35, 1.9)
    ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('RNEA: 두 번의 스윕으로 $\\tau = M(q)\\ddot q + h(q,\\dot q)$ 를 $O(n)$에 계산'
             '  —  신경망의 forward/backward pass와 같은 구조', fontsize=13.5, y=1.02)
fig.tight_layout()
fig.savefig('fig1_rnea_two_pass.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('fig1 저장')

# ================= fig2: 라그랑주 vs RNEA 궤적 일치 =================
l1, l2, m1, m2 = 1.0, 0.6, 2.0, 1.0
t = np.linspace(0, 2, 400)
A1, A2, w1, w2 = 0.8, 1.1, 2*np.pi/2, 2*np.pi/1.3
q1 = 0.3 + A1*np.sin(w1*t);  q2 = 0.5 + A2*np.sin(w2*t)
qd1 = A1*w1*np.cos(w1*t);    qd2 = A2*w2*np.cos(w2*t)
qdd1 = -A1*w1**2*np.sin(w1*t); qdd2 = -A2*w2**2*np.sin(w2*t)
tau_L = np.array([lagrange_2link([q1[k], q2[k]], [qd1[k], qd2[k]], [qdd1[k], qdd2[k]],
                                 l1, l2, m1, m2) for k in range(len(t))])
tau_R = np.array([rnea_planar([q1[k], q2[k]], [qd1[k], qd2[k]], [qdd1[k], qdd2[k]],
                              [l1, l2], [m1, m2], [l1, l2], [0, 0]) for k in range(len(t))])
err = np.abs(tau_L - tau_R).max(axis=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                               gridspec_kw={'height_ratios': [2.4, 1]})
ax1.plot(t, tau_L[:, 0], color=C_FWD, lw=2, label=r'$\tau_1$ 라그랑주 (R10 폐형식)')
ax1.plot(t, tau_L[:, 1], color='#2ca02c', lw=2, label=r'$\tau_2$ 라그랑주 (R10 폐형식)')
ax1.plot(t[::16], tau_R[::16, 0], 'o', color=C_BWD, ms=5, mfc='none', mew=1.6,
         label=r'$\tau_1$ RNEA (2-pass 수치)')
ax1.plot(t[::16], tau_R[::16, 1], 's', color='#9467bd', ms=5, mfc='none', mew=1.6,
         label=r'$\tau_2$ RNEA (2-pass 수치)')
ax1.set_ylabel('토크 [N·m]'); ax1.legend(ncol=2, fontsize=9.5); ax1.grid(alpha=0.3)
ax1.set_title('같은 방정식, 다른 계산 절차 — 2링크 궤적 위에서 라그랑주 폐형식 vs 평면 RNEA')
ax2.semilogy(t, np.maximum(err, 1e-17), color=C_LINK, lw=1.5)
ax2.set_ylabel(r'$\max_i |\Delta\tau_i|$'); ax2.set_xlabel('시간 [s]')
ax2.set_title('두 방법의 차이: 부동소수점 반올림 수준', fontsize=10.5)
ax2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('fig2_lagrange_vs_rnea.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('fig2 저장, 궤적상 최대 오차 =', err.max())

# ================= fig3: O(n) 스케일링 =================
import mujoco

def time_fn(fn, reps):
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        best = min(best, (time.perf_counter() - t0)/reps)
    return best

def make_chain_xml(n, l=0.3, m=1.0):
    I = m*l**2/12
    body = ''
    for i in range(n):
        pos = '0 0 0' if i == 0 else f'{l} 0 0'
        body += (f'<body pos="{pos}"><joint type="hinge" axis="0 0 1"/>'
                 f'<inertial pos="{l/2} 0 0" mass="{m}" diaginertia="1e-9 {I} {I}"/>')
    return ('<mujoco><option gravity="0 -9.81 0"/><worldbody>'
            + body + '</body>'*n + '</worldbody></mujoco>')

ns = [2, 4, 8, 16, 32, 64, 128, 256]
rng = np.random.default_rng(0)
t_rnea, t_fd, t_mj = [], [], []
print('n     RNEA(py)      FD=M조립+solve(py)   MuJoCo mj_inverse(C)')
for n in ns:
    ln = [0.3]*n; mn = [1.0]*n; cn = [0.15]*n; In = [1.0*0.3**2/12]*n
    qn = rng.uniform(-1, 1, n); qdn = rng.uniform(-1, 1, n); qddn = rng.uniform(-1, 1, n)
    reps = max(3, 1000//n)
    tr = time_fn(lambda: rnea_planar(qn, qdn, qddn, ln, mn, cn, In), reps)
    def fd():
        h = rnea_planar(qn, qdn, np.zeros(n), ln, mn, cn, In)
        M = np.column_stack([rnea_planar(qn, np.zeros(n), e, ln, mn, cn, In, gravity=0.0)
                             for e in np.eye(n)])
        np.linalg.solve(M, -h)
    tf = time_fn(fd, max(3, reps//10))
    mm = mujoco.MjModel.from_xml_string(make_chain_xml(n))
    dd = mujoco.MjData(mm)
    dd.qpos[:] = qn; dd.qvel[:] = qdn; dd.qacc[:] = qddn
    mujoco.mj_forward(mm, dd)
    tm = time_fn(lambda: mujoco.mj_inverse(mm, dd), max(20, 3000//n))
    t_rnea.append(tr); t_fd.append(tf); t_mj.append(tm)
    print(f'{n:4d}  {tr*1e6:9.1f} us  {tf*1e6:15.1f} us  {tm*1e6:15.1f} us')

fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.loglog(ns, np.array(t_rnea)*1e6, 'o-', color=C_FWD, lw=2, label='RNEA (이 강의의 순수 Python)')
ax.loglog(ns, np.array(t_fd)*1e6, 's-', color=C_BWD, lw=2,
          label='순동역학: $M$ 조립($n{+}1$회 ID) + solve')
ax.loglog(ns, np.array(t_mj)*1e6, '^-', color='#2ca02c', lw=2, label='MuJoCo mj_inverse (C 구현)')
ref = np.array(ns, float)
ax.loglog(ns, ref/ns[0]*t_rnea[0]*1e6, '--', color=C_FWD, alpha=0.45, label=r'기울기 1 ($O(n)$ 기준선)')
ax.loglog(ns, (ref/ns[0])**2*t_fd[0]*1e6, '--', color=C_BWD, alpha=0.45, label=r'기울기 2 ($O(n^2)$ 기준선)')
ax.set_xlabel('관절 수 $n$'); ax.set_ylabel('1회 호출 시간 [µs]')
ax.set_title('역동역학은 $O(n)$ — 관절이 2배면 시간도 2배 (log-log에서 기울기 1)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9.5)
fig.tight_layout()
fig.savefig('fig3_scaling.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('fig3 저장')
