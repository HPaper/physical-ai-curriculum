# Lec R29 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy/matplotlib 필요)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# A. WE-1: 정지거리와 충돌 에너지 — 속도 제한의 물리  (fig2)
# =====================================================================
print("=== A. 정지거리·충돌 에너지 (WE-1) ===")

def d_stop(v, t_d, a):
    return v*t_d + v**2/(2*a)

t_d, a = 0.10, 5.0
for v in (1.0, 0.25):
    print(f"v={v:.2f} m/s: d_stop = {v*t_d:.4f} + {v**2/(2*a):.5f} = {d_stop(v, t_d, a)*1000:.2f} mm")
print(f"속도 4배 감소 → 정지거리 {d_stop(1.0,t_d,a)/d_stop(0.25,t_d,a):.2f}배 감소")

m_R, m_H = 10.0, 0.6                      # 가상의 예: 로봇 유효질량 10 kg, 인체 부위 0.6 kg
mu = 1.0/(1.0/m_R + 1.0/m_H)
for v in (1.0, 0.25):
    print(f"v={v:.2f} m/s: mu={mu:.3f} kg, E = {0.5*mu*v**2*1000:.1f} mJ")
print(f"에너지 비 = {(1.0/0.25)**2:.0f}배")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
vv = np.linspace(0, 2.0, 400)
for (td_i, a_i, c, ls) in [(0.05, 10.0, 'tab:green', '-'), (0.10, 5.0, 'tab:blue', '-'),
                           (0.20, 2.0, 'tab:red', '-')]:
    ax[0].plot(vv, d_stop(vv, td_i, a_i)*1000, c, ls=ls,
               label=f"$t_d$={td_i*1000:.0f} ms, $a$={a_i:.0f} m/s²")
ax[0].plot(vv, vv*0.10*1000, 'k:', lw=1, label='반응 지연 성분 $v\\,t_d$ (선형)')
ax[0].scatter([1.0, 0.25], [d_stop(1.0, t_d, a)*1000, d_stop(0.25, t_d, a)*1000],
              zorder=5, color='tab:blue')
ax[0].annotate('200 mm', (1.0, 200), xytext=(1.05, 260), fontsize=9,
               arrowprops=dict(arrowstyle='->', lw=0.8))
ax[0].annotate('31 mm', (0.25, 31), xytext=(0.32, 130), fontsize=9,
               arrowprops=dict(arrowstyle='->', lw=0.8))
ax[0].set_xlabel('속도 $v$ [m/s]'); ax[0].set_ylabel('정지거리 [mm]')
ax[0].set_title('(a) 정지거리 $d = v\\,t_d + v^2/2a$ — 고속에서 2차항 지배')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(vv, 0.5*mu*vv**2*1000, 'tab:purple')
ax[1].scatter([1.0, 0.25], [0.5*mu*1.0**2*1000, 0.5*mu*0.25**2*1000], color='tab:purple', zorder=5)
ax[1].annotate('283 mJ', (1.0, 283), xytext=(0.6, 330), fontsize=9,
               arrowprops=dict(arrowstyle='->', lw=0.8))
ax[1].annotate('17.7 mJ (16배↓)', (0.25, 17.7), xytext=(0.3, 150), fontsize=9,
               arrowprops=dict(arrowstyle='->', lw=0.8))
ax[1].set_xlabel('상대 속도 $v$ [m/s]'); ax[1].set_ylabel('충돌 에너지 [mJ]')
ax[1].set_title('(b) 전달 에너지 $\\frac{1}{2}\\mu v^2$ ($\\mu$=유효 환산질량 %.2f kg)' % mu)
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_stopping_distance.png', dpi=140)
plt.close(fig)

# =====================================================================
# B. WE-2: 1-DoF CBF-QP 해석해 — 명목 제어가 벽으로 돌진 (fig1, 한 장 요약)
# =====================================================================
print("\n=== B. 1-DoF CBF 필터 (WE-2) ===")
x_w, alpha, u_nom_c = 1.0, 2.0, 1.0
dt, T = 1e-3, 2.0
ts = np.arange(0, T+dt, dt)

def run_1dof(filtered, alpha=alpha):
    x, xs_, us_ = 0.0, [], []
    for t in ts:
        u = u_nom_c
        if filtered:
            u = min(u, alpha*(x_w - x))       # CBF-QP 해석해 (E3)
        xs_.append(x); us_.append(u)
        x = x + dt*u
    return np.array(xs_), np.array(us_)

xn, un = run_1dof(False)
xf, uf = run_1dof(True)
i15 = int(1.5/dt)
x_analytic = x_w - 0.5*np.exp(-alpha*(1.5 - 0.5))
print(f"개입 시점(해석): x*=1-u_nom/α={1-u_nom_c/alpha}, t*={0.5}s")
print(f"x_filtered(1.5) 시뮬 = {xf[i15]:.5f}, 해석해 = {x_analytic:.5f}, 차 = {abs(xf[i15]-x_analytic):.2e}")
print(f"min h (필터) = {np.min(x_w - xf):.4f}, h(2.0) = {x_w - xf[-1]:.4f} (해석 {0.5*np.exp(-alpha*1.5):.4f})")
print(f"명목: 벽 통과 시각 = {ts[np.argmax(xn >= x_w)]:.3f} s, x(2.0) = {xn[-1]:.3f} (침범 {xn[-1]-x_w:.2f} m)")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
ax[0].axhspan(x_w, 2.1, color='tab:red', alpha=0.12)
ax[0].axhline(x_w, color='tab:red', lw=1.2)
ax[0].text(0.06, 1.35, '금지 영역 $h<0$', color='tab:red', fontsize=9)
ax[0].plot(ts, xn, 'tab:red', ls='--', label='명목 (필터 없음)')
ax[0].plot(ts, xf, 'tab:blue', label='CBF 필터 적용')
ax[0].axvline(0.5, color='gray', lw=0.8, ls=':')
ax[0].text(0.52, 0.15, '개입 시작\n$x^*=0.5$', fontsize=8, color='gray')
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('x [m]'); ax[0].set_ylim(0, 2.05)
ax[0].set_title('(a) 위치: 침범 vs 점근 정지'); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

ax[1].plot(ts, un, 'tab:red', ls='--', label='$u_{nom}$')
ax[1].plot(ts, uf, 'tab:blue', label='$u = \\min(u_{nom},\\, \\alpha h)$')
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('u [m/s]')
ax[1].set_title('(b) 명령: 경계에 다가갈수록 감속'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

ax[2].plot(ts, x_w - xn, 'tab:red', ls='--', label='명목')
ax[2].plot(ts, x_w - xf, 'tab:blue', label='필터')
ax[2].axhline(0, color='k', lw=0.8)
ax[2].fill_between(ts, -1.05, 0, color='tab:red', alpha=0.12)
ax[2].set_xlabel('t [s]'); ax[2].set_ylabel('$h(x) = x_w - x$')
ax[2].set_ylim(-1.05, 1.05)
ax[2].set_title('(c) 배리어 $h$: 필터는 $h \\geq 0$ 유지'); ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_cbf_intervention.png', dpi=140)
plt.close(fig)

# =====================================================================
# C. WE-3: 이중적분기 — 클리핑(반응적) vs CBF(예측적)  (fig3)
# =====================================================================
print("\n=== C. 클리핑 vs CBF (WE-3) ===")
u_max, x_w = 2.0, 1.0
x_t = 1.5                                  # 명목 목표(벽 너머)
dt, T = 1e-3, 6.0
ts3 = np.arange(0, T+dt, dt)

def h_brake(x, v):
    return (x_w - x) - max(v, 0.0)**2/(2*u_max)

def simulate(mode, alpha=5.0, v_cap=0.5):
    x, v = 0.0, 0.0
    X, V, H = [], [], []
    hit_v = None
    for t in ts3:
        if mode in ('nominal', 'cbf'):
            u = np.clip(25*(x_t - x) - 10*v, -u_max, u_max)
        if mode == 'clip':                 # 캐스케이드: 속도 명령 포화
            v_cmd = np.clip(5*(x_t - x), -v_cap, v_cap)
            u = np.clip(20*(v_cmd - v), -u_max, u_max)
        if mode == 'cbf' and v > 1e-9:
            u = min(u, u_max*(alpha*h_brake(x, v)/v - 1.0))
            u = max(u, -u_max)
        X.append(x); V.append(v); H.append(h_brake(x, v))
        x += dt*v; v += dt*u
        if hit_v is None and x >= x_w:
            hit_v = v
    return np.array(X), np.array(V), np.array(H), hit_v

res = {}
for mode in ('nominal', 'clip', 'cbf'):
    X, V, H, hit = simulate(mode)
    res[mode] = (X, V, H, hit)
    margin = x_w - X.max()
    print(f"{mode:8s}: 충돌속도 = {hit if hit is not None else 0:.3f} m/s, "
          f"최소 벽 여유 = {margin*1000:8.2f} mm, min h = {H.min():.5f}")
hitn = res['nominal'][3]; hitc = res['clip'][3]
print(f"에너지 비 (명목/클립) = {(hitn/hitc)**2:.1f}배")
print(f"명목 침투 깊이 예측 v^2/2u_max = {hitn**2/(2*u_max)*1000:.1f} mm (실측 {-(x_w-res['nominal'][0].max())*1000:.1f} mm)")
print(f"클립이 안전하려면 필요한 사전 제동 개시 거리 = v_cap^2/2u_max = {0.5**2/(2*u_max)*1000:.1f} mm")

# 이산화 잔여(-3 mm) 대책: 배리어에 마진 δ를 넣는다
x_w_eff = x_w - 0.005
def h_brake_m(x, v): return (x_w_eff - x) - max(v, 0.0)**2/(2*u_max)
x, v = 0.0, 0.0; xmax = 0.0
for t in ts3:
    u = np.clip(25*(x_t - x) - 10*v, -u_max, u_max)
    if v > 1e-9:
        u = max(min(u, u_max*(5.0*h_brake_m(x, v)/v - 1.0)), -u_max)
    x += dt*v; v += dt*u; xmax = max(xmax, x)
print(f"마진 δ=5 mm 적용 CBF: 실제 벽까지 최소 여유 = {(x_w-xmax)*1000:+.2f} mm")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
xx = np.linspace(-0.1, 1.0, 200)
ax[0].plot(xx, np.sqrt(2*u_max*(x_w - xx)), 'k--', lw=1.2,
           label='제동 포물선 $v=\\sqrt{2 u_{max}(x_w-x)}$')
ax[0].axvspan(x_w, 1.35, color='tab:red', alpha=0.12)
ax[0].axvline(x_w, color='tab:red', lw=1.2)
labels = {'nominal': ('명목 PD (필터 없음)', 'tab:red', '--'),
          'clip': ('속도 클리핑 $v \\leq 0.5$', 'tab:orange', '-'),
          'cbf': ('CBF-QP (α=5)', 'tab:blue', '-')}
for mode, (lab, c, ls) in labels.items():
    X, V, H, hit = res[mode]
    ax[0].plot(X, V, c, ls=ls, label=lab, lw=1.6)
ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('v [m/s]'); ax[0].set_xlim(-0.05, 1.35)
ax[0].set_title('(a) 위상 평면 — 안전 집합 = 제동 포물선 아래')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.3)

for mode, (lab, c, ls) in labels.items():
    X, V, H, hit = res[mode]
    ax[1].plot(ts3, X, c, ls=ls, label=lab, lw=1.6)
ax[1].axhspan(x_w, 1.6, color='tab:red', alpha=0.12)
ax[1].axhline(x_w, color='tab:red', lw=1.2)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('x [m]'); ax[1].set_ylim(0, 1.6)
ax[1].set_title('(b) 시간 응답 — CBF만 벽 앞에서 멈춘다')
ax[1].legend(fontsize=8, loc='lower right'); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_clip_vs_cbf.png', dpi=140)
plt.close(fig)

# =====================================================================
# D. α 스윕 — 보수성 트레이드오프 + 이산화 주의  (fig4)
# =====================================================================
print("\n=== D. α 스윕 (보수성) ===")
dt4, T4 = 1e-3, 20.0
ts4 = np.arange(0, T4+dt4, dt4)
x_ref = 0.7 + 0.4*np.sin(0.8*ts4)          # 봉우리 1.1 > 벽 1.0
xd_ref = 0.4*0.8*np.cos(0.8*ts4)

def track(alpha, dt=dt4, ts=ts4, xr=x_ref, xdr=xd_ref):
    x, X = 0.7, []
    for i in range(len(ts)):
        u = xdr[i] + 5*(xr[i] - x)
        if alpha is not None:
            u = min(u, alpha*(1.0 - x))
        X.append(x); x += dt*u
    return np.array(X)

alphas = [0.5, 2.0, 10.0]
trajs = {a: track(a) for a in alphas}
xr_best = np.minimum(x_ref, 1.0)           # 달성 가능한 최선의 안전 참조
for a in alphas:
    X = trajs[a]
    rms = np.sqrt(np.mean((X - xr_best)**2))*1000
    print(f"α={a:5.1f}: RMS(최선 안전 참조 대비) = {rms:7.2f} mm, min h = {(1.0-X).min():.3e} m")

a_grid = np.logspace(-1, 2, 25)
rms_g = [np.sqrt(np.mean((track(a) - xr_best)**2))*1000 for a in a_grid]
minh_g = [(1.0 - track(a)).min()*1000 for a in a_grid]

# 이산화 주의: 같은 α라도 dt가 크면 보증이 샌다
for (a_c, dt_c) in [(200.0, 1e-3), (200.0, 2e-2)]:
    ts_c = np.arange(0, T4+dt_c, dt_c)
    xr_c = 0.7 + 0.4*np.sin(0.8*ts_c); xdr_c = 0.4*0.8*np.cos(0.8*ts_c)
    X = track(a_c, dt_c, ts_c, xr_c, xdr_c)
    print(f"이산화: α={a_c:.0f}, dt={dt_c*1000:.0f} ms (α·dt={a_c*dt_c:.1f}) → min h = {(1.0-X).min()*1000:+.4f} mm")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ts4, x_ref, 'k:', lw=1.2, label='참조 (봉우리 1.1)')
ax[0].axhspan(1.0, 1.15, color='tab:red', alpha=0.12); ax[0].axhline(1.0, color='tab:red', lw=1.2)
for a, c in zip(alphas, ('tab:green', 'tab:blue', 'tab:purple')):
    ax[0].plot(ts4, trajs[a], c, label=f'α={a:g}')
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('x [m]')
ax[0].set_title('(a) α가 작을수록 일찍·멀리서 개입 (보수적)')
ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)

ax2 = ax[1].twinx()
l1, = ax[1].plot(a_grid, rms_g, 'tab:blue', marker='o', ms=3, label='추종 RMS (좌)')
l2, = ax2.plot(a_grid, minh_g, 'tab:red', marker='s', ms=3, label='최소 여유 $h_{min}$ (우)')
ax[1].set_xscale('log'); ax[1].set_xlabel('α [1/s]')
ax[1].set_ylabel('최선 안전 참조 대비 RMS [mm]', color='tab:blue')
ax2.set_ylabel('벽까지 최소 여유 [mm]', color='tab:red')
ax[1].set_title('(b) 성능↔여유 트레이드오프')
ax[1].legend(handles=[l1, l2], fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_alpha_sweep.png', dpi=140)
plt.close(fig)

# =====================================================================
# E. WE-4: 무작위(나쁜) 정책 + 필터 — 안전 집합 불변성  (fig5)
# =====================================================================
print("\n=== E. 무작위 정책 + 필터 (WE-4) ===")
u_max, alpha5 = 2.0, 5.0
dt5, T5, hold = 1e-3, 10.0, 0.05
n_ep = 200
rng = np.random.default_rng(0)

def h1(x, v): return (1.0 - x) - max(v, 0.0)**2/(2*u_max)
def h2(x, v): return (1.0 + x) - max(-v, 0.0)**2/(2*u_max)

def episode(seed, filtered):
    r = np.random.default_rng(seed)
    x, v = 0.0, 0.0
    n = int(T5/dt5)
    X = np.empty(n); n_int = 0
    u = 0.0
    for i in range(n):
        if i % int(hold/dt5) == 0:
            u_nom = r.uniform(-u_max, u_max)
        u = u_nom
        if filtered:
            if v > 1e-9:
                u = min(u, u_max*(alpha5*h1(x, v)/v - 1.0))
            elif v < -1e-9:
                u = max(u, u_max*(1.0 - alpha5*h2(x, v)/(-v)))
            u = float(np.clip(u, -u_max, u_max))
            if u != u_nom:
                n_int += 1
        X[i] = x
        x += dt5*v; v += dt5*u
    return X, n_int/n

viol_unf, viol_fil, margins, int_frac = 0, 0, [], []
samples_unf, samples_fil = [], []
for s in range(n_ep):
    Xu, _ = episode(s, False)
    Xf, fr = episode(s, True)
    if np.abs(Xu).max() > 1.0: viol_unf += 1
    if np.abs(Xf).max() > 1.0: viol_fil += 1
    margins.append(1.0 - np.abs(Xf).max())
    int_frac.append(fr)
    if s < 15:
        samples_unf.append(Xu); samples_fil.append(Xf)
print(f"무필터: {n_ep}회 중 {viol_unf}회 침범 ({100*viol_unf/n_ep:.1f}%)")
print(f"필터  : {n_ep}회 중 {viol_fil}회 침범, 최소 여유 = {min(margins)*1000:.2f} mm")
print(f"개입 비율(스텝 기준) 평균 = {100*np.mean(int_frac):.1f}%")

ts5 = np.arange(0, T5, dt5)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for X in samples_unf:
    ax[0].plot(ts5, X, lw=0.7, alpha=0.7)
for X in samples_fil:
    ax[1].plot(ts5, X, lw=0.7, alpha=0.7)
for a_ in ax:
    a_.axhspan(1.0, 4.0, color='tab:red', alpha=0.12)
    a_.axhspan(-4.0, -1.0, color='tab:red', alpha=0.12)
    a_.axhline(1.0, color='tab:red', lw=1.0); a_.axhline(-1.0, color='tab:red', lw=1.0)
    a_.set_xlabel('t [s]'); a_.grid(alpha=0.3)
ax[0].set_ylabel('x [m]'); ax[0].set_ylim(-3.5, 3.5)
ax[0].set_title(f'(a) 무작위 정책, 필터 없음 — {100*viol_unf/n_ep:.0f}% 침범')
ax[1].set_title(f'(b) 같은 정책 + CBF 필터 — 침범 {viol_fil}회 (200 에피소드)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_random_policy.png', dpi=140)
plt.close(fig)

print("\n그림 5장 생성 완료:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
