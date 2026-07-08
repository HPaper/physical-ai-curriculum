# Lec R24 그림 생성 스크립트 — 전신 제어(WBC)
# 실행: python3 gen_figs.py  (이 디렉토리에서)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
C1, C2, C3, C4 = '#0072B2', '#D55E00', '#009E73', '#CC79A7'

# ============================================================
# 공통: 3R 평면 팔
# ============================================================
L = np.array([0.5, 0.4, 0.3])
target = np.array([0.6, 0.4])
q_pref = np.array([1.2, -0.9, 0.5])   # 선호 자세 (팔꿈치 반대로 꺾인 쪽) — 목표와 충돌

def fk(q, upto=3):
    p, a = np.zeros(2), 0.0
    pts = [p.copy()]
    for li, qi in zip(L[:upto], q[:upto]):
        a += qi
        p = p + li * np.array([np.cos(a), np.sin(a)])
        pts.append(p.copy())
    return p, np.array(pts)

def jac(q):
    J = np.zeros((2, 3))
    eps = 1e-7
    f0, _ = fk(q)
    for j in range(3):
        dq = q.copy(); dq[j] += eps
        J[:, j] = (fk(dq)[0] - f0) / eps
    return J

def hierarchy_run(q0, k1=4.0, k2=4.0, dt=0.01, T=6.0):
    """엄격 위계: qdot = J+ (k1 e1) + N k2 (q_pref - q)"""
    q = q0.copy()
    inv_max = 0.0
    for _ in range(int(T/dt)):
        e1 = target - fk(q)[0]
        J = jac(q)
        Jp = np.linalg.pinv(J)
        N = np.eye(3) - Jp @ J
        sec = N @ (k2 * (q_pref - q))
        inv_max = max(inv_max, np.linalg.norm(J @ N @ (k2*(q_pref - q))))
        q = q + dt * (Jp @ (k1 * e1) + sec)
    return q, inv_max

def weighted_run(q0, rho, k1=4.0, k2=4.0, dt=0.01, T=12.0):
    """가중 QP: qdot = argmin ||J qdot - k1 e1||^2 + rho ||qdot - k2(q_pref-q)||^2"""
    q = q0.copy()
    for _ in range(int(T/dt)):
        e1 = target - fk(q)[0]
        J = jac(q)
        A = J.T @ J + rho * np.eye(3)
        b = J.T @ (k1 * e1) + rho * k2 * (q_pref - q)
        q = q + dt * np.linalg.solve(A, b)
    return q

q0 = np.array([0.3, 0.3, 0.3])

# --- WE-1 수치 ---
print('=== WE-1: 위계 ===')
print('충돌 크기 |fk(q_pref)-target| =', round(np.linalg.norm(fk(q_pref)[0] - target), 4))
q_h, inv_max = hierarchy_run(q0)
eef_h = np.linalg.norm(fk(q_h)[0] - target)
print('위계: EEF 오차 =', f'{eef_h:.2e}', ' 자세 거리 =', round(np.linalg.norm(q_h - q_pref), 4))
print('위계 불변량 max||J N z|| =', f'{inv_max:.2e}')
# 1차 태스크만 (자세 무시)
q_ik, _ = hierarchy_run(q0, k2=0.0)
print('IK만:  EEF 오차 =', f'{np.linalg.norm(fk(q_ik)[0]-target):.2e}',
      ' 자세 거리 =', round(np.linalg.norm(q_ik - q_pref), 4))

# --- WE-2 수치: 가중 스윕 ---
print('=== WE-2: 가중 vs 위계 ===')
rhos = np.logspace(-4, 1, 26)
eef_w, post_w = [], []
for rho in rhos:
    qw = weighted_run(q0, rho)
    eef_w.append(np.linalg.norm(fk(qw)[0] - target))
    post_w.append(np.linalg.norm(qw - q_pref))
eef_w, post_w = np.array(eef_w), np.array(post_w)
for r in [0.01, 0.1, 1.0]:
    i = np.argmin(np.abs(rhos - r))
    print(f'rho={rhos[i]:.3g}: EEF 오차 {eef_w[i]*1000:.2f} mm, 자세 거리 {post_w[i]:.4f}')
# 기울기 (log-log)
sl = np.polyfit(np.log10(rhos[:12]), np.log10(eef_w[:12]), 1)[0]
print('log-log 기울기 (작은 rho 구간):', round(sl, 3))

# KKT 동치성 검증: min||qd-z||^2 s.t. J qd = v1  ==  J+ v1 + N z
J = jac(q0); z = 4.0*(q_pref - q0); v1 = 4.0*(target - fk(q0)[0])
KKT = np.block([[np.eye(3), J.T], [J, np.zeros((2, 2))]])
sol = np.linalg.solve(KKT, np.concatenate([z, v1]))[:3]
proj = np.linalg.pinv(J) @ v1 + (np.eye(3) - np.linalg.pinv(J) @ J) @ z
print('KKT해 vs 사영식 차이:', f'{np.linalg.norm(sol - proj):.2e}')

# --- E1 재귀 3태스크: null space 소진 검증 (본문 "재귀 한 단 더") ---
print('=== E1 재귀: 3태스크 null-space 소진 ===')
q = q0.copy()
J1r = jac(q)                       # 1순위: EEF 위치 (2차원)
J2r = np.array([[0.0, 1.0, 0.0]])  # 2순위: 팔꿈치 관절 속도 (1차원)
v1r = np.array([0.10, -0.05]); v2r = np.array([0.3])
z3r = np.array([1.0, 1.0, 1.0])    # 3순위 희망 (3차원)
qd1 = np.linalg.pinv(J1r) @ v1r
N1r = np.eye(3) - np.linalg.pinv(J1r) @ J1r
qd2 = qd1 + np.linalg.pinv(J2r @ N1r) @ (v2r - J2r @ qd1)
Jbr = np.vstack([J1r, J2r])
N2r = np.eye(3) - np.linalg.pinv(Jbr) @ Jbr
qd3 = qd2 + N2r @ z3r
print('1순위 잔차', f'{np.linalg.norm(J1r @ qd3 - v1r):.2e}',
      ' 2순위 잔차', f'{np.linalg.norm(J2r @ qd3 - v2r):.2e}')
print('||N2|| =', f'{np.linalg.norm(N2r):.2e}', ' → 3순위 몫 =', np.round(N2r @ z3r, 12))

# ============================================================
# WE-3: 서 있는 로봇의 지면반력 배분
# ============================================================
print('=== WE-3: 지면반력 배분 ===')
mass, g0, h = 40.0, 9.81, 0.9
mg = mass * g0
x1, x2 = -0.15, 0.15   # 뒤/앞 발
mu = 0.6

def grf_qp(Fp):
    """min ||f||^2 s.t. 힘·모멘트 평형, 마찰원뿔. f=(fx1,fz1,fx2,fz2)"""
    cons = [
        {'type': 'eq', 'fun': lambda f: f[0] + f[2] + Fp},           # sum fx = -Fp
        {'type': 'eq', 'fun': lambda f: f[1] + f[3] - mg},           # sum fz = mg
        {'type': 'eq', 'fun': lambda f: x1*f[1] + x2*f[3] + h*(f[0]+f[2])},  # 모멘트
        {'type': 'ineq', 'fun': lambda f: mu*f[1] - abs(f[0])},
        {'type': 'ineq', 'fun': lambda f: mu*f[3] - abs(f[2])},
        {'type': 'ineq', 'fun': lambda f: f[1]},
        {'type': 'ineq', 'fun': lambda f: f[3]},
    ]
    r = minimize(lambda f: f @ f, x0=[0, mg/2, 0, mg/2], constraints=cons,
                 method='SLSQP', options={'maxiter': 300, 'ftol': 1e-10})
    return r.x if r.success else None

for Fp in [0.0, 50.0, 55.0]:
    f = grf_qp(Fp)
    print(f'F_push={Fp:5.1f}: fx1={f[0]:7.2f} fz1={f[1]:7.2f} | fx2={f[2]:7.2f} fz2={f[3]:7.2f}'
          f'  뒤발 원뿔여유 {mu*f[1]-abs(f[0]):.2f}')
# 손계산 대조 (F=50): fz1=196.2-150=46.2, fz2=346.2, fx 균등 -25
# 실행 가능 경계: LP feasibility (선형이므로 linprog가 강건) + 이분법
from scipy.optimize import linprog
def grf_feasible(Fp):
    # 변수 f=(fx1,fz1,fx2,fz2). 등식 3개, 원뿔 |fx_i| <= mu fz_i -> 부등식 4개
    A_eq = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [h, x1, h, x2]])
    b_eq = np.array([-Fp, mg, 0.0])
    A_ub = np.array([[1, -mu, 0, 0], [-1, -mu, 0, 0],
                     [0, 0, 1, -mu], [0, 0, -1, -mu]])
    r = linprog(np.zeros(4), A_ub=A_ub, b_ub=np.zeros(4), A_eq=A_eq, b_eq=b_eq,
                bounds=[(None, None), (0, None), (None, None), (0, None)])
    return r.status == 0
lo, hi = 0.0, 200.0
for _ in range(50):
    mid = 0.5*(lo+hi)
    lo, hi = (mid, hi) if grf_feasible(mid) else (lo, mid)
print('실행 가능 경계 (이분법):', round(lo, 3), 'N  | 해석해 mg*x2/h =', round(mg*x2/h, 3), 'N')
# 원뿔 활성화 경계 해석해: 균등 분배 Fp/2 = mu*(mg/2 - 3Fp) -> Fp(0.5+3mu) = mu*mg/2
F_cone_analytic = (mu*mg/2)/(0.5 + 3*mu)
print('원뿔 활성화 경계 해석해:', round(F_cone_analytic, 2), 'N')

# ============================================================
# 그림 1: 한 장 요약 — 위계(팔) + 접촉 분배(다리)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
_, pts_pref = fk(q_pref)
_, pts_h = fk(q_h)
q_w1 = weighted_run(q0, 1.0)
_, pts_w = fk(q_w1)
ax.plot(pts_pref[:, 0], pts_pref[:, 1], 'o--', color='0.6', lw=2, ms=5, label='선호 자세 $q_{pref}$ (2순위)')
ax.plot(pts_w[:, 0], pts_w[:, 1], 'o-', color=C2, lw=2.5, ms=6, label=r'가중 QP ($\rho=1$): 둘 다 타협')
ax.plot(pts_h[:, 0], pts_h[:, 1], 'o-', color=C1, lw=3, ms=6, label='엄격 위계: EEF 오차 0')
ax.plot(*target, 'X', color=C3, ms=16, mew=2, label='EEF 목표 (1순위)', zorder=5)
ax.plot(0, 0, 'ks', ms=9)
ax.annotate(f'가중: 목표를 {np.linalg.norm(fk(q_w1)[0]-target)*1000:.0f} mm 놓침',
            xy=fk(q_w1)[0], xytext=(0.72, 0.12), color=C2, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C2))
ax.set_xlim(-0.15, 1.25); ax.set_ylim(-0.12, 0.95)
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='upper left')
ax.set_title('(a) 태스크 위계 — 1순위(EEF)는 침해되지 않는다')

ax = axes[1]
# 서 있는 로봇 + 마찰원뿔 + 반력
f50 = grf_qp(50.0)
ax.plot([x1, x2], [0, 0], 'k', lw=1)
ax.fill_between([-0.5, 0.5], [-0.06, -0.06], [0, 0], color='0.85', zorder=0)
# 몸: 두 다리 + 몸통
for xf in (x1, x2):
    ax.plot([xf, 0], [0, h], color='0.55', lw=3, zorder=2)
ax.plot([0], [h], 'o', color='k', ms=14, zorder=4)
ax.annotate('CoM', (0, h), textcoords='offset points', xytext=(10, 6))
ax.annotate('', xy=(0.28, h), xytext=(0.02, h),
            arrowprops=dict(arrowstyle='-|>', color=C4, lw=3))
ax.text(0.09, h+0.05, '$F_{push}=50$ N', color=C4, fontsize=10)
sc = 0.0016
for xf, fx, fz, nm in [(x1, f50[0], f50[1], '뒤발'), (x2, f50[2], f50[3], '앞발')]:
    th = np.arctan(mu)
    for s in (-1, 1):
        ax.plot([xf, xf + 0.34*np.sin(th)*s], [0, 0.34*np.cos(th)], '--', color=C3, lw=1.2)
    ax.annotate('', xy=(xf + fx*sc, fz*sc), xytext=(xf, 0),
                arrowprops=dict(arrowstyle='-|>', color=C1, lw=2.5))
    ax.text(xf + fx*sc - 0.1, fz*sc + 0.03, f'{nm}\n({fx:.0f}, {fz:.0f}) N',
            fontsize=8.5, color=C1)
cop = 0.9*50/mg
ax.plot([cop], [0], 'v', color=C2, ms=11, zorder=5)
ax.text(cop-0.02, -0.15, f'CoP = +{cop*100:.1f} cm\n(한계 +15 cm)', color=C2, fontsize=9)
ax.set_xlim(-0.5, 0.62); ax.set_ylim(-0.22, 1.15)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(b) 접촉력 배분 — 마찰원뿔 안에서 미는 힘 버티기')
fig.suptitle('WBC = 태스크들의 우선순위 사회: 위계가 팔을, 원뿔 제약이 발을 지배한다', y=1.02)
fig.tight_layout()
fig.savefig('fig1_wbc_overview.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('fig1 저장')

# ============================================================
# 그림 2: 우선순위 vs 가중치
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.loglog(rhos, eef_w*1000, 'o-', color=C2, ms=4, label='가중 QP: 1순위 오차')
ax.axhline(eef_h*1000, color=C1, lw=2, ls='--',
           label=f'엄격 위계: {eef_h*1000:.1e} mm (수치 0)')
ax.set_xlabel(r'가중비 $\rho = w_2/w_1$ (2순위 가중 ↑)')
ax.set_ylabel('EEF 오차 [mm]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=9)
ax.set_title('(a) 가중치는 아무리 줄여도 1순위를 침해한다')
ax = axes[1]
ax.plot(eef_w*1000, post_w, 'o-', color=C2, ms=4, label=r'가중 QP ($\rho$ 스윕)')
ax.plot(eef_h*1000, np.linalg.norm(q_h - q_pref), '*', color=C1, ms=18,
        label='엄격 위계')
ax.plot(np.linalg.norm(fk(q_ik)[0]-target)*1000, np.linalg.norm(q_ik - q_pref),
        's', color='0.4', ms=10, label='IK만 (2순위 없음)')
ax.annotate(r'$\rho\to 0$', (eef_w[0]*1000, post_w[0]), textcoords='offset points', xytext=(8, 8))
ax.annotate(r'$\rho=10$', (eef_w[-1]*1000, post_w[-1]), textcoords='offset points', xytext=(8, -12))
ax.set_xlabel('EEF 오차 [mm] (1순위)')
ax.set_ylabel(r'자세 거리 $\|q-q_{pref}\|$ [rad] (2순위)')
ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title('(b) 트레이드오프 곡선 위 어디에 설 것인가')
fig.tight_layout()
fig.savefig('fig2_priority_vs_weight.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('fig2 저장')

# ============================================================
# 그림 3: 지면반력 배분 — 밀기 스윕
# ============================================================
Fs2 = np.linspace(0, 68, 200)
fz1a = mg/2 - 3*Fs2          # 뒤발 (해석해)
fz2a = mg/2 + 3*Fs2
fx_even = Fs2/2              # 균등 분배 시 |fx| (각 발)
cone1 = mu*np.maximum(fz1a, 0)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
fcolors = {0.0: '0.55', 50.0: C1, 62.0: C2}
for Fp, col in fcolors.items():
    f = grf_qp(Fp)
    for xf, fx, fz in [(x1, f[0], f[1]), (x2, f[2], f[3])]:
        ax.annotate('', xy=(xf + fx*sc, fz*sc), xytext=(xf, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2))
    ax.plot([], [], color=col, lw=2.2, label=f'$F_{{push}}$ = {Fp:.0f} N')
th = np.arctan(mu)
for xf in (x1, x2):
    for s in (-1, 1):
        ax.plot([xf, xf + 0.65*np.sin(th)*s], [0, 0.65*np.cos(th)], '--', color=C3, lw=1.3)
ax.fill_between([-0.5, 0.5], [-0.05, -0.05], [0, 0], color='0.85')
ax.text(x1, -0.12, '뒤발', ha='center'); ax.text(x2, -0.12, '앞발', ha='center')
ax.text(x1+0.02, 0.62, '마찰원뿔 ($\\mu$=0.6)', color=C3, fontsize=9)
ax.set_xlim(-0.55, 0.75); ax.set_ylim(-0.16, 0.75)
ax.set_aspect('equal'); ax.axis('off'); ax.legend(fontsize=9, loc='upper right')
ax.set_title('(a) 미는 힘이 커질수록: 앞발로 하중 이동, 원뿔이 눕는 걸 허락하는 만큼만')
ax = axes[1]
ax.plot(Fs2, fz2a, color=C1, lw=2, label='앞발 $f_{z2}$')
ax.plot(Fs2, fz1a, color=C2, lw=2, label='뒤발 $f_{z1}$')
ax.plot(Fs2, fx_even, color='0.3', lw=1.8, ls=':', label='균등 분배 시 각 발 $|f_x| = F_{push}/2$')
ax.plot(Fs2, cone1, color=C3, lw=1.8, ls='--', label=r'뒤발 마찰 한계 $\mu f_{z1}$')
F_cone = 117.72/2.3
F_tip = mg*x2/h
ax.axvline(F_cone, color=C3, lw=1); ax.axvline(F_tip, color='k', lw=1)
ax.text(F_cone-1, 240, f'원뿔 활성\n{F_cone:.1f} N', ha='right', color=C3, fontsize=9)
ax.text(F_tip-1, 300, f'뒤발 뜸(CoP=앞발)\n{F_tip:.1f} N → 스텝 필요', ha='right', fontsize=9)
ax.axvspan(F_tip, 68, color='0.9')
ax.set_xlabel('$F_{push}$ [N]'); ax.set_ylabel('힘 [N]')
ax.set_xlim(0, 68); ax.set_ylim(0, 420)
ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc='upper left')
ax.set_title('(b) 두 개의 벽: 마찰원뿔(51.2 N), 지지다각형(65.4 N)')
fig.tight_layout()
fig.savefig('fig3_grf_distribution.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('fig3 저장')

# ============================================================
# 그림 4: null-space 사영의 기하 (2R 손계산 예제)
# ============================================================
fig, ax = plt.subplots(figsize=(5.6, 5.2))
j = np.array([-1.0, -1.0])           # 1차 태스크 행벡터 (x방향 EEF 속도)
z = np.array([1.0, 0.0])             # 2순위 희망
N = np.eye(2) - np.outer(j, j)/ (j @ j)
Nz = N @ z
t = np.linspace(-1.2, 1.2, 2)
ax.plot(t*1/np.sqrt(2), -t*1/np.sqrt(2), color=C1, lw=2.5,
        label=r'null space of $J_1$: $J_1\dot q = 0$')
ax.annotate('', xy=z, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='0.4', lw=2.5))
ax.text(*(z + [0.02, 0.05]), r'2순위 희망 $z=(1,0)$', color='0.3', fontsize=10)
ax.annotate('', xy=Nz, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=C3, lw=3))
ax.text(Nz[0]+0.04, Nz[1]-0.06, r'채택 $N_1 z = (\frac{1}{2}, -\frac{1}{2})$', color=C3, fontsize=10)
ax.plot([z[0], Nz[0]], [z[1], Nz[1]], ':', color=C2, lw=2)
ax.annotate('', xy=(z+Nz)/2 + np.array([0.1, 0.1])*0, xytext=(z+Nz)/2,
            arrowprops=dict(arrowstyle='-', color=C2))
ax.text(0.82, -0.28, '기각: 1순위를\n침해하는 성분', color=C2, fontsize=9, ha='center')
ax.annotate('', xy=np.array([0.55, 0.55]), xytext=np.array([-0.25, -0.25]),
            arrowprops=dict(arrowstyle='<|-|>', color='0.75', lw=1.5))
ax.text(0.33, 0.62, r'row space of $J_1$ = $\pm J_1^\top$'+'\n(태스크를 움직이는 방향)', color='0.5', fontsize=9)
ax.set_xlabel(r'$\dot q_1$'); ax.set_ylabel(r'$\dot q_2$')
ax.set_xlim(-1.1, 1.35); ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower left')
ax.set_title('2R 손계산 예제: 사영 = "1순위에 안 보이는 성분만 통과"\n(딥러닝의 gradient surgery와 같은 기하)')
fig.tight_layout()
fig.savefig('fig4_nullspace_geometry.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('fig4 저장')
