"""Lec R23 그림 생성 스크립트 — MPC.
실행: python3 gen_figs.py  (이 디렉토리에서)
본문에 인용된 수치도 함께 출력한다 (수치 재현성 각주 참조).
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
from scipy.linalg import expm, solve_discrete_are, block_diag
from scipy.optimize import minimize, lsq_linear

np.set_printoptions(precision=4, suppress=True)
C_LQR, C_MPC, C_REF = '#4477aa', '#cc3311', '#555555'

# ============================================================
# 공통: 카트폴 (R18과 동일 파라미터) + 50 Hz 이산화
# ============================================================
M, m, l, g = 1.0, 0.1, 0.5, 9.81
A = np.array([[0,0,1,0],[0,0,0,1],[0,-m*g/M,0,0],[0,(M+m)*g/(M*l),0,0]], float)
B = np.array([[0.],[0.],[1/M],[-1/(M*l)]])
dt = 0.02                                   # 제어 주기 50 Hz
aug = np.zeros((5,5)); aug[:4,:4] = A; aug[:4,4:] = B
Md = expm(aug*dt); Ad, Bd = Md[:4,:4], Md[:4,4:]
Q = np.diag([1., 10., 0.1, 0.1]); R = np.array([[0.1]])
P = solve_discrete_are(Ad, Bd, Q, R)
Kd = np.linalg.solve(R + Bd.T@P@Bd, Bd.T@P@Ad)
print("K_dlqr =", Kd.ravel())

def f_nl(s, F):
    x, th, xd, thd = s
    sn, cs = np.sin(th), np.cos(th); den = M + m*sn**2
    return np.array([xd, thd, (F + m*sn*(l*thd**2 - g*cs))/den,
                     (-F*cs - m*l*thd**2*sn*cs + (M+m)*g*sn)/(l*den)])

def rk4(s, F, h):
    k1=f_nl(s,F); k2=f_nl(s+h/2*k1,F); k3=f_nl(s+h/2*k2,F); k4=f_nl(s+h*k3,F)
    return s + h/6*(k1+2*k2+2*k3+k4)

def simulate(ctrl, th0_deg, umax, T=8.0, nsub=10, record_plan=None):
    s = np.array([0., np.radians(th0_deg), 0., 0.]); h = dt/nsub
    S, U, Ts, plans = [s.copy()], [], [], {}
    for k in range(int(T/dt)):
        t0 = time.perf_counter()
        u = float(np.clip(ctrl(s), -umax, umax))
        Ts.append(time.perf_counter() - t0)
        if record_plan is not None and k in record_plan:
            plans[k] = ctrl.last_plan.copy()
        U.append(u)
        for _ in range(nsub): s = rk4(s, u, h)
        S.append(s.copy())
        if abs(s[1]) > np.pi/2:
            return False, np.array(S), np.array(U), np.array(Ts), plans
    ok = abs(s[1]) < 0.01 and abs(s[3]) < 0.05
    return ok, np.array(S), np.array(U), np.array(Ts), plans

def batch(Ad, Bd, N):
    n, mm = Bd.shape
    Sx = np.zeros((n*N, n)); Su = np.zeros((n*N, mm*N))
    for k in range(N):
        Sx[n*k:n*(k+1)] = np.linalg.matrix_power(Ad, k+1)
        for j in range(k+1):
            Su[n*k:n*(k+1), mm*j:mm*(j+1)] = np.linalg.matrix_power(Ad, k-j)@Bd
    return Sx, Su

def make_mpc(N, umax, xmax=None):
    Sx, Su = batch(Ad, Bd, N)
    Qbar = block_diag(*([Q]*(N-1) + [P])); Rbar = np.kron(np.eye(N), R)
    H = Su.T@Qbar@Su + Rbar; H = 0.5*(H + H.T); F = Su.T@Qbar@Sx
    rows = np.arange(N)*4
    Gpos, Spos = Su[rows, :], Sx[rows, :]
    prev = {'U': np.zeros(N)}
    def ctrl(s):
        fv = F@s
        fun = lambda U: U@H@U/2 + fv@U
        jac = lambda U: H@U + fv
        cons = []
        if xmax is not None:
            c0 = Spos@s
            cons = [{'type':'ineq','fun':lambda U: xmax-(c0+Gpos@U),'jac':lambda U:-Gpos},
                    {'type':'ineq','fun':lambda U: xmax+(c0+Gpos@U),'jac':lambda U: Gpos}]
        res = minimize(fun, np.r_[prev['U'][1:], 0.], jac=jac, method='SLSQP',
                       bounds=[(-umax, umax)]*N, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-9})
        prev['U'] = res.x
        ctrl.last_plan = Spos@s + Gpos@res.x        # 계획된 카트 위치 (fig2용)
        return res.x[0]
    return ctrl

umax, xmax, th0 = 15., 0.5, 20.
cost_of = lambda S, U: sum(S[i]@Q@S[i] + U[i]**2*R[0,0] for i in range(len(U)))*dt

# ============================================================
# fig1: 한 장 요약 — 포화 LQR vs 제약 MPC (레일 ±0.5 m)
# ============================================================
okL, SL, UL, _, _ = simulate(lambda s: -(Kd@s)[0], th0, umax)
mpc25 = make_mpc(25, umax, xmax)
okM, SM, UM, TsM, plans = simulate(mpc25, th0, umax, record_plan={0, 10, 25, 50, 100})
print(f"\n[fig1] th0={th0}°, 레일 ±{xmax} m, umax={umax} N")
print(f"  포화 LQR: {'성공' if okL else '실패'}, max|x|={np.max(np.abs(SL[:,0])):.3f} m "
      f"(침범 {np.max(np.abs(SL[:,0]))-xmax:+.3f} m), peak|u|={np.max(np.abs(UL)):.2f} N, cost={cost_of(SL,UL):.3f}")
print(f"  MPC N=25: {'성공' if okM else '실패'}, max|x|={np.max(np.abs(SM[:,0])):.4f} m, "
      f"peak|u|={np.max(np.abs(UM)):.2f} N, cost={cost_of(SM,UM):.3f}, "
      f"solve avg={np.mean(TsM)*1e3:.2f} ms / max={np.max(TsM)*1e3:.2f} ms")

tL = np.arange(len(SL))*dt; tM = np.arange(len(SM))*dt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].axhspan(xmax, 0.9, color='k', alpha=0.12)
ax[0].axhspan(-0.9, -xmax, color='k', alpha=0.12)
ax[0].axhline(xmax, color='k', lw=1.2); ax[0].axhline(-xmax, color='k', lw=1.2)
ax[0].plot(tL, SL[:,0], color=C_LQR, ls='--', lw=2, label='포화 LQR (제약을 모름)')
ax[0].plot(tM, SM[:,0], color=C_MPC, lw=2, label='제약 MPC (N=25)')
ax[0].annotate('레일 끝 = 하드웨어 파손', xy=(1.35, 0.63), xytext=(2.6, 0.72),
               arrowprops=dict(arrowstyle='->'), fontsize=9)
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('카트 위치 x [m]'); ax[0].set_ylim(-0.35, 0.9)
ax[0].set_title('(a) 카트 위치와 레일 제약 ±0.5 m'); ax[0].legend(fontsize=8, loc='lower right')
ax[1].plot(tL, np.degrees(SL[:,1]), color=C_LQR, ls='--', lw=2)
ax[1].plot(tM, np.degrees(SM[:,1]), color=C_MPC, lw=2)
ax[1].axhline(0, color='k', lw=0.5)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel(r'막대 각도 $\theta$ [deg]')
ax[1].set_title(r'(b) 둘 다 막대는 세운다 ($\theta_0=20°$)')
ax[2].plot(np.arange(len(UL))*dt, UL, color=C_LQR, ls='--', lw=1.5)
ax[2].plot(np.arange(len(UM))*dt, UM, color=C_MPC, lw=1.5)
ax[2].axhline(umax, color='k', lw=1); ax[2].axhline(-umax, color='k', lw=1)
ax[2].set_xlabel('t [s]'); ax[2].set_ylabel('입력 u [N]')
ax[2].set_title(r'(c) 입력 (|u| ≤ 15 N)')
for a in ax: a.grid(alpha=0.3)
fig.suptitle('같은 비용·같은 힘 한계 — 차이는 "레일을 아는가": MPC는 0.500 m에서 정확히 멈춘다', y=1.02)
fig.tight_layout(); fig.savefig('fig1_mpc_vs_lqr.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ============================================================
# fig2: receding horizon — 시점별 계획과 실행 궤적
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].axhline(xmax, color='k', lw=1.2)
ax[0].axhspan(xmax, 0.62, color='k', alpha=0.12)
cols = plt.cm.viridis(np.linspace(0, 0.85, len(plans)))
for c, (k, plan) in zip(cols, sorted(plans.items())):
    tp = (k + 1 + np.arange(len(plan)))*dt
    ax[0].plot(tp, plan, ls='--', lw=1.4, color=c, label=f't={k*dt:.1f} s의 계획')
    ax[0].plot(k*dt, SM[k, 0], 'o', ms=5, color=c)
ax[0].plot(tM, SM[:,0], 'k', lw=2, label='실제 실행 궤적')
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('카트 위치 x [m]'); ax[0].set_xlim(0, 3.5)
ax[0].set_title('(a) N=25스텝(0.5 s)을 계획하고 1스텝만 실행 — 매번 다시')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
# t=0의 입력 계획 vs 실제 실행된 입력
mpc_tmp = make_mpc(25, umax, xmax)
s0 = np.array([0., np.radians(th0), 0., 0.])
_ = mpc_tmp(s0)
Sx25, Su25 = batch(Ad, Bd, 25)
# t=0 계획의 입력열을 다시 구한다
Qbar25 = block_diag(*([Q]*24 + [P])); Rbar25 = np.kron(np.eye(25), R)
H25 = Su25.T@Qbar25@Su25 + Rbar25; F25 = Su25.T@Qbar25@Sx25
rows25 = np.arange(25)*4
res0 = minimize(lambda U: U@H25@U/2 + (F25@s0)@U, np.zeros(25),
                jac=lambda U: H25@U + F25@s0, method='SLSQP',
                bounds=[(-umax, umax)]*25,
                constraints=[{'type':'ineq','fun':lambda U: xmax-(Sx25[rows25]@s0+Su25[rows25]@U),'jac':lambda U:-Su25[rows25]},
                             {'type':'ineq','fun':lambda U: xmax+(Sx25[rows25]@s0+Su25[rows25]@U),'jac':lambda U: Su25[rows25]}],
                options={'maxiter':200,'ftol':1e-9})
ax[1].step(np.arange(25)*dt, res0.x, where='post', color=C_REF, ls='--', lw=1.8, label='t=0에서 세운 입력 계획 25개')
ax[1].step(np.arange(25)*dt, UM[:25], where='post', color=C_MPC, lw=1.8, label='실제 실행된 입력 (매번 첫 개)')
ax[1].axhline(umax, color='k', lw=1); ax[1].axhline(-umax, color='k', lw=1)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('u [N]')
ax[1].set_title('(b) 계획은 참고일 뿐 — 실행되는 것은 항상 "다시 푼 첫 수"')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig('fig2_receding_horizon.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ============================================================
# fig3: N 스윕 — 성능(근시안)과 계산 시간(예산)
# ============================================================
print("\n[fig3] N 스윕 @ th0=20°, 레일 0.5 m")
Ns = [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 65, 80]
res_sweep = []
for N in Ns:
    ok, S2, U2, Ts2, _ = simulate(make_mpc(N, umax, xmax), th0, umax)
    ok2 = ok and np.max(np.abs(S2[:,0])) <= xmax + 1e-3
    res_sweep.append((N, ok2, cost_of(S2, U2), np.mean(Ts2)*1e3, np.max(Ts2)*1e3))
    print(f"  N={N:3d} (지평 {N*dt:.2f} s): {'성공' if ok2 else '실패'} cost={res_sweep[-1][2]:8.3f} "
          f"solve avg={res_sweep[-1][3]:6.2f} ms / max={res_sweep[-1][4]:6.2f} ms")

arr = np.array([(r[0], r[2], r[3], r[4]) for r in res_sweep])
oks = np.array([r[1] for r in res_sweep])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(arr[oks,0], arr[oks,1], 'o-', color='#228833', label='성공 (막대 세움 + 레일 준수)')
ax[0].plot(arr[~oks,0], arr[~oks,1], 'x', ms=9, mew=2.5, color=C_MPC, label='실패 (낙하)')
ax[0].axvspan(0, 22.5, color=C_MPC, alpha=0.07)
ax[0].annotate('근시안 영역:\n레일이 "보일" 때는 이미 늦다', xy=(11.5, 4.4), fontsize=9, color=C_MPC, ha='center')
ax[0].set_xlabel('호라이즌 N (스텝, 1스텝 = 20 ms)'); ax[0].set_ylabel('폐루프 비용')
ax[0].set_title(r'(a) 성능: N ≤ 20(0.4 s)은 낙하, N ≥ 25(0.5 s)부터 성공')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].semilogy(arr[:,0], arr[:,2], 'o-', color=C_LQR, label='평균 해 시간')
ax[1].semilogy(arr[:,0], arr[:,3], 's--', color=C_MPC, label='최악 해 시간')
ax[1].axhline(20, color='k', lw=1.5)
ax[1].annotate('제어 주기 20 ms = 데드라인', xy=(2, 13.5), fontsize=9)
ax[1].set_xlabel('호라이즌 N'); ax[1].set_ylabel('QP 해 시간 [ms] (SLSQP)')
ax[1].set_title('(b) 계산: N이 크면 최악 해 시간이 주기를 넘는다')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('fig3_horizon_sweep.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ============================================================
# fig4: LIP 보행 MPC — ZMP 박스 제약 + 밀침 흡수
# ============================================================
z0 = 0.8; w = np.sqrt(g/z0); Tl = 0.1; Nl = 16
ch, sh = np.cosh(w*Tl), np.sinh(w*Tl)
Al = np.array([[ch, sh/w],[w*sh, ch]]); Bl = np.array([[1-ch],[-w*sh]])
step_T, stride, d = 0.8, 0.25, 0.05
p_ref_at = lambda k: stride*min(int(k*Tl/step_T), 7)
Ktot = int(6.4/Tl)
Sxl = np.zeros((2*Nl, 2)); Sul = np.zeros((2*Nl, Nl))
for k in range(Nl):
    Sxl[2*k:2*k+2] = np.linalg.matrix_power(Al, k+1)
    for j in range(k+1):
        Sul[2*k:2*k+2, j:j+1] = np.linalg.matrix_power(Al, k-j)@Bl
xi_row = Sxl[2*Nl-2] + Sxl[2*Nl-1]/w
xi_su  = Sul[2*Nl-2] + Sul[2*Nl-1]/w

def lip_run(push_t=None, push_dv=0.0):
    x = np.array([0., 0.]); out = []
    for k in range(Ktot):
        pref = np.array([p_ref_at(k+1+j) for j in range(Nl)])
        A_ls = np.vstack([np.eye(Nl), 10.*xi_su])
        b_ls = np.concatenate([pref, [10.*(pref[-1] - xi_row@x)]])
        sol = lsq_linear(A_ls, b_ls, bounds=(pref-d, pref+d), method='bvls')
        p0 = sol.x[0]
        x = Al@x + Bl.ravel()*p0
        if push_t is not None and abs(k*Tl - push_t) < Tl/2: x[1] += push_dv
        out.append((x[0], x[1], p0, x[0] + x[1]/w))
    return np.array(out)

out = lip_run(push_t=2.0, push_dv=0.15)
tl = (np.arange(Ktot)+1)*Tl
prefs = np.array([p_ref_at(k+1) for k in range(Ktot)])
print(f"\n[fig4] LIP: omega={w:.4f}, 밀침 +0.15 m/s @2.0 s → 최종 DCM 오차 "
      f"{abs(out[-1,3]-prefs[-1]):.4f} m, ZMP 박스 준수: {np.max(np.abs(out[:,2]-prefs)) <= d+1e-9}")
out_norm = lip_run()
print(f"  정상 보행: ZMP 최대 편차 {np.max(np.abs(out_norm[:,2]-prefs)):.4f} m (박스 ±{d}), "
      f"평균 전진 속도 {out_norm[-1,0]/6.4:.3f} m/s")
dvs = np.arange(0.05, 0.40, 0.01)
errs = np.array([abs(lip_run(push_t=2.0, push_dv=dv)[-1,3]-prefs[-1]) for dv in dvs])
cliff = dvs[np.argmax(errs > 0.01)]
print(f"  밀침 흡수 한계 (측정): {cliff:.2f} m/s / 이론 ω·d = {w*d:.3f} m/s")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].fill_between(tl, prefs-d, prefs+d, step='pre', color='k', alpha=0.15, label='지지 다각형 (ZMP 박스 ±5 cm)')
ax[0].step(tl, out[:,2], where='pre', color=C_MPC, lw=1.6, label='ZMP $p$ (QP의 해)')
ax[0].plot(tl, out[:,0], 'k', lw=2, label='CoM $c$')
ax[0].plot(tl, out[:,3], color=C_LQR, ls='--', lw=1.6, label=r'DCM $\xi = c + \dot c/\omega$')
ax[0].annotate('밀침 +0.15 m/s', xy=(2.0, out[np.argmin(abs(tl-2.05)), 3]), xytext=(2.6, 0.15),
               arrowprops=dict(arrowstyle='->'), fontsize=9)
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('전진 방향 위치 [m]')
ax[0].set_title('(a) LIP 보행 MPC: ZMP를 박스 안에서 움직여 밀침을 흡수')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.3)
ax[1].semilogy(dvs, np.maximum(errs, 1e-6), 'o-', color=C_MPC)
ax[1].axvline(w*d, color='k', ls='--', lw=1.5)
ax[1].annotate(r'$\omega d$ = 0.175 m/s' + '\n(R13 capture point 한계)', xy=(w*d, 1e-3),
               xytext=(0.19, 3e-4), fontsize=9, arrowprops=dict(arrowstyle='->'))
ax[1].set_xlabel('밀침 크기 [m/s]'); ax[1].set_ylabel('최종 DCM 오차 [m]')
ax[1].set_title('(b) 발을 못 옮기면 ZMP 재계획의 한계는 정확히 ω·d')
ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('fig4_lip_walking.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ============================================================
# fig5: 마찰 원뿔의 선형화 (R12 회수)
# ============================================================
mu = 0.6
fig = plt.figure(figsize=(10, 3.8))
ax3 = fig.add_subplot(1, 2, 1, projection='3d')
th_ = np.linspace(0, 2*np.pi, 60); z_ = np.linspace(0, 1, 12)
TH, Z = np.meshgrid(th_, z_)
ax3.plot_surface(mu*Z*np.cos(TH), mu*Z*np.sin(TH), Z, alpha=0.25, color=C_LQR)
for sx in [-1, 1]:
    for sy in [-1, 1]:
        ax3.plot([0, sx*mu/np.sqrt(2)], [0, sy*mu/np.sqrt(2)], [0, 1], color=C_MPC, lw=2)
sq = mu/np.sqrt(2)
ax3.plot([sq, sq, -sq, -sq, sq], [sq, -sq, -sq, sq, sq], [1]*5, color=C_MPC, lw=2)
ax3.set_xlabel('$f_x$'); ax3.set_ylabel('$f_y$'); ax3.set_zlabel('$f_z$')
ax3.set_title('(a) 마찰 원뿔(파랑)과 내접 피라미드(빨강)')
ax2 = fig.add_subplot(1, 2, 2)
ax2.add_patch(plt.Circle((0, 0), mu, fill=False, color=C_LQR, lw=2))
ax2.add_patch(plt.Rectangle((-sq, -sq), 2*sq, 2*sq, fill=False, color=C_MPC, lw=2))
ax2.add_patch(plt.Rectangle((-mu, -mu), 2*mu, 2*mu, fill=False, color='#228833', lw=2, ls='--'))
ax2.annotate('원뿔 $\\|f_t\\| \\leq \\mu f_z$ (2차)', xy=(0.03, 0.68), fontsize=9, color=C_LQR)
ax2.annotate('내접 피라미드\n(보수적, 안전)', xy=(0., 0.), fontsize=9, color=C_MPC, ha='center', va='center')
ax2.annotate('외접 박스 $|f_x|,|f_y| \\leq \\mu f_z$ (Cheetah 3: 모서리에서 √2배 낙관)', xy=(0.0, -0.82),
             fontsize=9, color='#228833', ha='center')
ax2.set_xlim(-1.05, 1.05); ax2.set_ylim(-1.05, 1.05); ax2.set_aspect('equal')
ax2.set_xlabel('$f_x/f_z$'); ax2.set_ylabel('$f_y/f_z$')
ax2.set_title('(b) 단면 ($f_z$=1, μ=0.6): 2차 원뿔 → 선형 부등식 4개')
ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('fig5_friction_pyramid.png', dpi=140, bbox_inches='tight'); plt.close(fig)

# ============================================================
# WE-3 표: 회복 가능 최대 초기각 이진 탐색 (본문 §WE-3의 표)
# 주의: MPC 시뮬 × 이진 탐색이라 몇 분 걸린다
# ============================================================
def boundary(make_ctrl, umax_, xmax_=None, lo=5., hi=90., tol=0.05):
    """성공하는 최대 초기각(도)을 이진 탐색. xmax_ 지정 시 레일 준수도 요구."""
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        ok, S2, _, _, _ = simulate(make_ctrl(), mid, umax_)
        if ok and (xmax_ is None or np.max(np.abs(S2[:,0])) <= xmax_ + 1e-3):
            lo = mid
        else:
            hi = mid
    return lo

print("\n[WE-3 표] 회복 가능 최대 초기각 (이진 탐색, 허용오차 0.05°)")
lqr = lambda: (lambda s: -(Kd@s)[0])
print(f"  힘 무제한      LQR {boundary(lqr, 1e9):5.1f}°   (무제약 MPC = LQR — E2)")
print(f"  |u| ≤ 15 N    LQR {boundary(lqr, umax):5.1f}° / MPC N=25 {boundary(lambda: make_mpc(25, umax), umax):5.1f}°")
print(f"  + 레일 ±0.5 m LQR {boundary(lqr, umax, xmax):5.1f}° / MPC N=25 {boundary(lambda: make_mpc(25, umax, xmax), umax, xmax):5.1f}°")

print("\n그림 5장 생성 완료.")
