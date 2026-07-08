"""Lec R18 그림 생성 스크립트.
fig1: 한 장 요약 — LQR 안정화(제어)와 KF 융합(추정), 같은 Riccati의 두 얼굴
fig2: Q/R 스윕 — 응답 속도 vs 입력 크기의 트레이드오프 (보상 성형)
fig3: 1-DoF 칼만 필터 — 위치/속도 추정과 이득 수렴
fig4: LQR basin 지도 — 초기각 × 힘 한계 평면에서 안정화 성공/실패
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lecR18"

# ---------------- 공통: 카트폴 모델 (본문 WE-2와 동일) ----------------
M, m, l, g = 1.0, 0.1, 0.5, 9.81
A = np.array([[0,0,1,0],[0,0,0,1],[0,-m*g/M,0,0],[0,(M+m)*g/(M*l),0,0]], float)
B = np.array([[0],[0],[1/M],[-1/(M*l)]])

def make_K(Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)

Q0 = np.diag([1., 10., 0.1, 0.1]); R0 = np.array([[0.1]])
K0 = make_K(Q0, R0)

def cartpole(t, s, K, umax):
    x, th, xd, thd = s
    F = np.clip(-(K @ s)[0], -umax, umax)
    sin, cos = np.sin(th), np.cos(th)
    den = M + m*sin**2
    xdd = (F + m*sin*(l*thd**2 - g*cos)) / den
    thdd = (-F*cos - m*l*thd**2*sin*cos + (M+m)*g*sin) / (l*den)
    return [xd, thd, xdd, thdd]

def fallen(t, s, K, umax):
    return abs(s[1]) - np.pi/2
fallen.terminal = True

def sim(deg, K=K0, umax=1e9, T=6.0, n=1201):
    sol = solve_ivp(cartpole, [0, T], [0, np.radians(deg), 0, 0],
                    args=(K, umax), rtol=1e-9, atol=1e-11,
                    dense_output=True, events=fallen)
    tt = np.linspace(0, min(T, sol.t[-1]), n)
    ss = sol.sol(tt)
    uu = np.clip(-(K @ ss).ravel(), -umax, umax)
    return tt, ss, uu

# ---------------- 공통: 1-DoF KF (본문 WE-3와 동일, seed 0) ----------------
def run_kf():
    rng = np.random.default_rng(0)
    dt, T = 0.01, 8.0
    t = np.arange(0, T, dt)
    p_true = 0.3*np.sin(1.5*t) + 0.1*np.sin(4.3*t)
    v_true = 0.45*np.cos(1.5*t) + 0.43*np.cos(4.3*t)
    a_true = -0.675*np.sin(1.5*t) - 1.849*np.sin(4.3*t)
    sig_a, sig_enc = 0.4, 0.005
    a_meas = a_true + sig_a*rng.standard_normal(t.size)
    y_enc  = p_true + sig_enc*rng.standard_normal(t.size)
    F = np.array([[1, dt], [0, 1]]); G = np.array([[0.5*dt**2], [dt]])
    H = np.array([[1., 0.]])
    Qw = G @ G.T * sig_a**2; Rv = np.array([[sig_enc**2]])
    xh = np.zeros((2,1)); P = np.eye(2)*1e-2
    est = np.zeros((t.size, 2)); K1 = np.zeros(t.size); P11 = np.zeros(t.size)
    for k in range(t.size):
        xh = F @ xh + G * a_meas[k]
        P  = F @ P @ F.T + Qw
        K  = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rv)
        xh = xh + K * (y_enc[k] - H @ xh)
        P  = (np.eye(2) - K @ H) @ P
        est[k] = xh.ravel(); K1[k] = K[0,0]; P11[k] = P[0,0]
    return t, p_true, v_true, y_enc, est, K1, P11

t_kf, p_true, v_true, y_enc, est, K1, P11 = run_kf()

# ============================================================
# fig 1 — 한 장 요약
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

for deg, c in [(20, 'tab:blue'), (40, 'tab:green'), (60, 'tab:orange')]:
    tt, ss, _ = sim(deg)
    ax1.plot(tt, np.degrees(ss[1]), color=c, lw=2, label=f"$\\theta_0$={deg}° → 안정화")
tt, ss, _ = sim(65)
ax1.plot(tt, np.degrees(ss[1]), color='tab:red', lw=2, ls='--',
         label="$\\theta_0$=65° → 실패 (경계 63.3°)")
ax1.axhline(0, color='k', lw=0.6)
ax1.axhline(90, color='0.6', lw=0.8, ls=':'); ax1.text(3.6, 92, "수평 = 낙하", fontsize=9, color='0.4')
ax1.set_xlabel("t [s]"); ax1.set_ylabel("막대 각도 θ [deg]")
ax1.set_title("(a) 최적 제어: LQR $u=-Kx$ — 하나의 게인으로 안정화")
ax1.legend(fontsize=9); ax1.set_ylim(-30, 100); ax1.set_xlim(0, 6)
ax1.grid(alpha=0.3)

sl = slice(200, 400)   # 2~4초 확대
ax2.plot(t_kf[sl], y_enc[sl]*1e3, '.', ms=3, color='0.6', label="엔코더 관측 (σ=5mm)")
ax2.plot(t_kf[sl], p_true[sl]*1e3, 'k-', lw=2, label="참값")
ax2.plot(t_kf[sl], est[sl,0]*1e3, '-', color='tab:red', lw=1.5, label="KF 추정 (RMSE 2.1mm)")
band = 2*np.sqrt(P11[sl])*1e3
ax2.fill_between(t_kf[sl], est[sl,0]*1e3-band, est[sl,0]*1e3+band,
                 color='tab:red', alpha=0.15, label="±2σ (공분산)")
ax2.set_xlabel("t [s]"); ax2.set_ylabel("위치 [mm]")
ax2.set_title("(b) 최적 추정: 칼만 필터 — 모델 예측 × 관측의 신뢰도 가중 융합")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle("제어와 추정은 같은 수학이다 — 둘 다 Riccati 방정식의 해 (LQR ↔ KF 쌍대성)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_lqr_kf_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================
# fig 2 — Q/R 스윕
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
Rs = [0.01, 0.1, 1.0, 10.0]
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(Rs)))
stats = []
for Rval, c in zip(Rs, colors):
    K = make_K(Q0, np.array([[Rval]]))
    tt, ss, uu = sim(20, K=K, T=15.0, n=6001)
    axes[0].plot(tt, np.degrees(ss[1]), color=c, lw=1.8, label=f"R={Rval:g}")
    axes[1].plot(tt, uu, color=c, lw=1.8, label=f"R={Rval:g}")
    th = np.abs(ss[1]); bandv = 0.02*np.radians(20)
    idx = np.where(th > bandv)[0]
    ts = tt[idx[-1]+1] if len(idx) and idx[-1] < len(tt)-1 else 0.0
    stats.append((Rval, np.abs(uu).max(), ts, np.trapz(uu**2, tt)))
axes[0].set_xlim(0, 8); axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("θ [deg]")
axes[0].set_title("(a) 각도 응답 — R↑ = 입력이 비싸짐 = 느긋한 제어")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3); axes[0].axhline(0, color='k', lw=0.6)
axes[1].set_xlim(0, 4); axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("입력 힘 u [N]")
axes[1].set_title("(b) 입력 이력 — R↓ = 피크 힘 폭증")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].axhline(0, color='k', lw=0.6)
st = np.array(stats)
axes[2].plot(st[:,3], st[:,2], 'o-', color='tab:purple', lw=1.5)
for Rval, pk, ts, en in stats:
    dy = -22 if Rval == 10.0 else -2
    axes[2].annotate(f"R={Rval:g}\n피크 {pk:.1f}N", (en, ts), textcoords="offset points",
                     xytext=(8, dy), fontsize=9)
axes[2].set_xlabel(r"제어 에너지 $\int u^2 dt$"); axes[2].set_ylabel("2% 정착 시간 [s]")
axes[2].set_title("(c) 트레이드오프 곡선 — 공짜 점심은 없다")
axes[2].grid(alpha=0.3); axes[2].set_xlim(5, 30)
fig.suptitle("Q/R 가중 스윕 ($\\theta_0$=20°): 비용 행렬 = RL의 보상 성형과 같은 설계 언어", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_qr_sweep.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================
# fig 3 — 칼만 필터 상세
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
sl = slice(100, 500)
axes[0].plot(t_kf[sl], y_enc[sl]*1e3, '.', ms=2.5, color='0.65', label="엔코더 (RMSE 4.9mm)")
axes[0].plot(t_kf[sl], p_true[sl]*1e3, 'k-', lw=2, label="참값")
axes[0].plot(t_kf[sl], est[sl,0]*1e3, '-', color='tab:red', lw=1.3, label="KF (RMSE 2.1mm)")
axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("위치 [mm]")
axes[0].set_title("(a) 위치: 관측 노이즈의 절반 이하로")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

v_fd = np.diff(y_enc, prepend=y_enc[0])/0.01
axes[1].plot(t_kf[sl], v_fd[sl], '-', color='0.75', lw=0.7, label="엔코더 유한차분 (RMSE 0.69 m/s)")
axes[1].plot(t_kf[sl], v_true[sl], 'k-', lw=2, label="참 속도")
axes[1].plot(t_kf[sl], est[sl,1], '-', color='tab:red', lw=1.5, label="KF (RMSE 0.07 m/s)")
axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("속도 [m/s]")
axes[1].set_title("(b) 속도: 직접 재지 않는 상태의 추정 — 10배 개선")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylim(-2.5, 2.5)

axes[2].plot(t_kf[:150], K1[:150], color='tab:blue', lw=2, label="칼만 이득 $K_1$")
ax2b = axes[2].twinx()
ax2b.plot(t_kf[:150], np.sqrt(P11[:150])*1e3, color='tab:orange', lw=2, ls='--',
          label=r"$\sqrt{P_{11}}$ [mm]")
axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("이득 $K_1$", color='tab:blue')
ax2b.set_ylabel("위치 표준편차 [mm]", color='tab:orange')
axes[2].set_title("(c) 이득·공분산이 정상값으로 수렴 ($K_1\\rightarrow$0.119)")
axes[2].grid(alpha=0.3)
h1, l1 = axes[2].get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
axes[2].legend(h1+h2, l1+l2, fontsize=9, loc='center right')
fig.suptitle("1-DoF 위치 추정: 노이즈 엔코더 + IMU 가속도의 융합 (WE-3, 100Hz)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_kalman.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================
# fig 4 — basin 지도 (θ0 × umax), 벡터화 RK4
# ============================================================
deg_grid = np.linspace(2, 80, 40)
umax_grid = np.linspace(2, 40, 39)
D, U = np.meshgrid(deg_grid, umax_grid)
S = np.zeros(D.shape + (4,))
S[..., 1] = np.radians(D)
alive = np.ones(D.shape, bool)

def f_vec(S, U):
    x, th, xd, thd = S[...,0], S[...,1], S[...,2], S[...,3]
    F = np.clip(-(K0[0,0]*x + K0[0,1]*th + K0[0,2]*xd + K0[0,3]*thd), -U, U)
    sin, cos = np.sin(th), np.cos(th)
    den = M + m*sin**2
    xdd = (F + m*sin*(l*thd**2 - g*cos)) / den
    thdd = (-F*cos - m*l*thd**2*sin*cos + (M+m)*g*sin) / (l*den)
    return np.stack([xd, thd, xdd, thdd], axis=-1)

dt = 0.002
for step in range(int(8.0/dt)):
    k1 = f_vec(S, U); k2 = f_vec(S + 0.5*dt*k1, U)
    k3 = f_vec(S + 0.5*dt*k2, U); k4 = f_vec(S + dt*k3, U)
    S = S + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    alive &= np.abs(S[...,1]) < np.pi/2
    S[~alive] = 0.0   # 죽은 셀은 동결 (발산 방지)

success = alive & (np.abs(S[...,1]) < 0.01) & (np.abs(S[...,3]) < 0.01)

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.pcolormesh(D, U, success, cmap=matplotlib.colors.ListedColormap(['#f2c7c2', '#bfe3c8']),
              shading='auto')
ax.axvline(63.3, color='tab:red', lw=1.5, ls='--')
ax.text(64, 30, "포화 없음 한계 63.3°\n(비선형성 자체의 벽)", fontsize=10, color='tab:red')
ax.plot([47.7], [15], 'k*', ms=15)
ax.annotate("실습 기준점: umax=15N → 47.7°", (47.7, 15), xytext=(12, 24),
            textcoords='data', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='k'))
ax.set_xlabel("초기 기울기 $\\theta_0$ [deg]"); ax.set_ylabel("힘 한계 $u_{max}$ [N]")
ax.set_title("LQR basin 지도: 어디까지 선형 게인이 통하는가\n(초록=안정화, 붉음=낙하 — 선형화는 국소 면허다)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_basin_map.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("figs done")
