# Lec R21 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy/matplotlib 필요)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# ================= 공통 설정: 1-DoF 접촉 문제 =================
# 질량 m 로봇(태스크 공간 겉보기 관성), x_w 에 강성 k_e 벽.
# 명령: x_d 를 0 → 0.15 m 로 0.1 m/s 램프 (벽 뒤 5 cm 를 목표로 '밀어 넣는' 명령)
m      = 2.0        # kg
x_w    = 0.10       # 벽 위치 [m]
ke_nom = 1.0e4      # 명목 환경 강성 [N/m]
x_goal = 0.15       # 명령 목표 (벽 뒤 5 cm)
v_cmd  = 0.10       # 접근 속도 [m/s]

def xd_of(t):
    xd = np.minimum(x_goal, v_cmd * t)
    vd = np.where(v_cmd * t < x_goal, v_cmd, 0.0)
    return xd, vd

def f_ss(K, ke=ke_nom):
    """정상 접촉력: 직렬 스프링 K·ke/(K+ke) × (x_goal - x_w)"""
    return K * ke / (K + ke) * (x_goal - x_w)

# ---------------- 임피던스 로봇 (토크원): 벡터화 시뮬 ----------------
def sim_impedance(K, B, ke=ke_nom, T=6.0, dt=1e-4, tau_lag=0.0, record=False):
    """m ẍ = u + F_env,  u = K(x_d-x) + B(ẋ_d-ẋ)  (tau_lag>0 이면 1차 지연 액추에이터)
    K, B: 스칼라 또는 같은 shape 의 배열(벡터화 스윕)."""
    K = np.atleast_1d(np.asarray(K, float)); B = np.broadcast_to(np.asarray(B, float), K.shape).copy()
    n = K.size
    x = np.zeros(n); v = np.zeros(n); u = np.zeros(n)
    steps = int(T / dt)
    Fss = f_ss(K, ke)
    F_peak = np.zeros(n); bounce = np.zeros(n, int); touched = np.zeros(n, bool)
    in_c_prev = np.zeros(n, bool); t_settle = np.zeros(n)
    hist = [] if record else None
    for i in range(steps):
        t = i * dt
        xd, vd = xd_of(t)
        Fc = ke * np.clip(x - x_w, 0.0, None)          # 벽 → 로봇: -Fc
        u_cmd = K * (xd - x) + B * (vd - v)
        if tau_lag > 0:
            u += dt / tau_lag * (u_cmd - u)            # 액추에이터 1차 지연
        else:
            u = u_cmd
        v += dt * (u - Fc) / m
        x += v * dt
        in_c = Fc > 0
        touched |= in_c
        bounce += (in_c_prev & ~in_c).astype(int)      # 접촉 이탈 횟수
        in_c_prev = in_c
        F_peak = np.maximum(F_peak, Fc)
        off = np.abs(Fc - Fss) > 0.05 * Fss            # 5% 정착 판정
        t_settle = np.where(off, t, t_settle)
        if record:
            hist.append((t, x.copy(), Fc.copy()))
    if record:
        ts = np.array([h[0] for h in hist])
        xs = np.array([h[1] for h in hist]).squeeze()
        fs = np.array([h[2] for h in hist]).squeeze()
        return ts, xs, fs
    return F_peak, bounce, t_settle, Fss

# ---------------- 어드미턴스 로봇 (강성 위치 로봇 + F/T): 벡터화 ----------------
def sim_admittance(K, B, ke, Md=2.0, tau_c=5e-3, T=8.0, dt=1e-4, record=False):
    """측정 F → 어드미턴스 필터가 x_c 생성 → 내부 위치루프(1차 지연 tau_c)가 추종.
    Md ẍ_c + B(ẋ_c-ẋ_d) + K(x_c-x_d) = -F_meas"""
    K = np.atleast_1d(np.asarray(K, float))
    B = np.broadcast_to(np.asarray(B, float), K.shape).copy()
    ke = np.broadcast_to(np.asarray(ke, float), K.shape).copy()
    n = K.size
    x = np.zeros(n); xc = np.zeros(n); vc = np.zeros(n)
    steps = int(T / dt)
    hist = [] if record else None
    F_last = []  # 마지막 2 s 의 힘 (안정 판정용)
    for i in range(steps):
        t = i * dt
        xd, vd = xd_of(t)
        F = ke * np.clip(x - x_w, 0.0, None)           # F/T 측정치
        ac = (-F - B * (vc - vd) - K * (xc - xd)) / Md
        vc += ac * dt
        xc += vc * dt
        x += dt * (xc - x) / tau_c                     # 내부 위치루프 (1차)
        if record:
            hist.append((t, x.copy(), F.copy()))
        if t > T - 2.0:
            F_last.append(F.copy())
    if record:
        ts = np.array([h[0] for h in hist])
        xs = np.array([h[1] for h in hist]).squeeze()
        fs = np.array([h[2] for h in hist]).squeeze()
        return ts, xs, fs
    F_last = np.array(F_last)
    Fss = K * ke / (K + ke) * (x_goal - x_w)
    osc = F_last.max(axis=0) - F_last.min(axis=0)      # 말미 진동 폭
    unstable = osc > 0.2 * Fss
    return unstable, osc, Fss

def adm_ke_max(B, K=200.0, Md=2.0, tau=5e-3):
    """Routh–Hurwitz 경계: (τs+1)(Md s²+Bs+K)+ke=0 이 안정일 조건"""
    return (Md + tau * B) * (B + tau * K) / (tau * Md) - K

print("=" * 60)
print("[WE-1] 손계산 검증")
K_d, B_d = 200.0, 40.0
ks = K_d * ke_nom / (K_d + ke_nom)
print(f"직렬 강성 k_s = {ks:.2f} N/m,  F_ss = {ks*0.05:.3f} N")
Kp_pos = 2.0e4
ksp = Kp_pos * ke_nom / (Kp_pos + ke_nom)
print(f"위치 로봇(Kp=2e4): k_s = {ksp:.1f} N/m,  F_ss = {ksp*0.05:.1f} N")
zeta_c = B_d / (2 * np.sqrt(m * (K_d + ke_nom)))
wn_c = np.sqrt((K_d + ke_nom) / m)
zeta_free = B_d / (2 * np.sqrt(m * K_d))
print(f"자유공간 ζ = {zeta_free:.2f} → 접촉 ζ = {zeta_c:.3f}, ω_n(접촉) = {wn_c:.1f} rad/s")
Fp, bc, tst, Fss = sim_impedance(K_d, B_d)
print(f"임피던스(K=200,B=40): F_peak = {Fp[0]:.2f} N (F_ss 대비 {Fp[0]/Fss[0]:.2f}배), "
      f"바운스 {bc[0]}회, 5% 정착 {tst[0]:.2f} s")
print(f"Franka setCartesianImpedance 병진 최대 3000 N/m: F_ss = {f_ss(3000.):.1f} N (본문 §4·실습 5)")
# 관절 강성 → 태스크 강성 합동 변환 (본문 §4): R20 WE-1 의 자세 q=(0, π/2)
Jq = np.array([[-1.0, -1.0], [1.0, 0.0]])
K_th = np.diag([3000.0, 2000.0])
Kx = np.linalg.inv(Jq).T @ K_th @ np.linalg.inv(Jq)
print(f"K_x = J^-T K_θ J^-1 =\n{Kx}\n고유값 = {np.linalg.eigvalsh(Kx)}")

# ================= 그림 2: 같은 명령, 두 로봇 =================
ts_i, xs_i, fs_i = sim_impedance(K_d, B_d, record=True, T=4.0)
ts_p, xs_p, fs_p = sim_impedance(Kp_pos, 2*np.sqrt(m*Kp_pos), record=True, T=4.0)  # '위치 제어' = 초고강성 임피던스
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(ts_p, xd_of(ts_p)[0] * 1e3, 'k--', lw=1.2, label='명령 $x_d$ (벽 뒤 5 cm)')
ax[0].plot(ts_p, xs_p * 1e3, color='crimson', label=f'위치 제어 ($K_p$=20000 N/m)')
ax[0].plot(ts_i, xs_i * 1e3, color='seagreen', label='임피던스 ($K_d$=200, $B_d$=40)')
ax[0].axhline(x_w * 1e3, color='gray', lw=2)
ax[0].text(2.4, x_w * 1e3 + 4, '벽 ($k_e$ = 10⁴ N/m)', color='gray')
ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('x [mm]'); ax[0].set_title('(a) 위치: 명령은 같다')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[1].semilogy(ts_p, np.maximum(fs_p, 1e-2), color='crimson', label='위치 제어')
ax[1].semilogy(ts_i, np.maximum(fs_i, 1e-2), color='seagreen', label='임피던스')
ax[1].axhline(fs_p[-1], color='crimson', ls=':', lw=1)
ax[1].axhline(fs_i[-1], color='seagreen', ls=':', lw=1)
ax[1].annotate(f'{fs_p[-1]:.0f} N', (3.0, fs_p[-1] * 1.25), color='crimson')
ax[1].annotate(f'{fs_i[-1]:.1f} N', (3.0, fs_i[-1] * 1.3), color='seagreen')
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('접촉력 [N] (log)')
ax[1].set_title('(b) 힘: 34배 차이')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which='both')
fig.suptitle('같은 명령("벽 뒤 5 cm로 가라"), 두 로봇 — 강성이 곧 접촉력이다', y=1.02)
fig.tight_layout(); fig.savefig('fig2_position_vs_impedance.png', dpi=140, bbox_inches='tight')
print(f"\n[그림 2] 정상 접촉력: 위치 {fs_p[-1]:.1f} N vs 임피던스 {fs_i[-1]:.2f} N "
      f"(비 {fs_p[-1]/fs_i[-1]:.1f}), 위치 F_peak = {fs_p.max():.0f} N")

# ================= 그림 1 (한 장 요약): K·B 스윕 지도 =================
nK, nB = 60, 60
Ks = np.logspace(np.log10(50), np.log10(2000), nK)
Bs = np.logspace(np.log10(5), np.log10(400), nB)
KK, BB = np.meshgrid(Ks, Bs, indexing='ij')
Fp, bc, tst, Fss = sim_impedance(KK.ravel(), BB.ravel())
ratio = (Fp / Fss).reshape(nK, nB)
bounce = bc.reshape(nK, nB)
tset = tst.reshape(nK, nB)

Fpk = Fp.reshape(nK, nB)

picks = [(1200.0, 60.0, 'A', 'crimson'), (200.0, 8.0, 'B', 'darkorange'), (200.0, 80.0, 'C', 'seagreen')]
fig = plt.figure(figsize=(12, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.35)
axm = fig.add_subplot(gs[:, 0])
pc = axm.pcolormesh(Bs, Ks, Fpk, norm=matplotlib.colors.LogNorm(vmin=Fpk.min(), vmax=Fpk.max()),
                    cmap='inferno', shading='auto')
cb = fig.colorbar(pc, ax=axm); cb.set_label('접촉력 피크 $F_{peak}$ [N]')
cs = axm.contour(Bs, Ks, bounce, levels=[0.5], colors='cyan', linewidths=2)
try:
    axm.clabel(cs, fmt={0.5: '바운스 경계'}, fontsize=9, manual=[(25, 400)])
except Exception:
    axm.clabel(cs, fmt={0.5: '바운스 경계'}, fontsize=9)
cs2 = axm.contour(Bs, Ks, tset, levels=[1.8, 2.5], colors='white', linewidths=1, linestyles=':')
# 접촉 임계감쇠선 B = 2√(m(K+ke))
axm.plot(2 * np.sqrt(m * (Ks + ke_nom)), Ks, 'w--', lw=1.2)
axm.annotate('접촉 임계감쇠\n$B=2\\sqrt{m(K_d+k_e)}$', xy=(287, 900), xytext=(95, 1350),
             color='w', fontsize=8, arrowprops=dict(arrowstyle='->', color='w', lw=0.8))
for K0, B0, name, c in picks:
    axm.plot(B0, K0, 'o', color=c, mec='w', ms=9)
    axm.annotate(name, (B0, K0), textcoords='offset points', xytext=(7, 4), color='w', fontsize=11)
axm.plot(40, 200, '+', color='w', ms=10, mew=2)
axm.annotate('WE-1', (40, 200), textcoords='offset points', xytext=(5, 5), color='w', fontsize=8)
axm.set_xscale('log'); axm.set_yscale('log')
axm.set_xlabel('$B_d$ [N·s/m]'); axm.set_ylabel('$K_d$ [N/m]')
axm.set_title(f'벽 접촉 응답 지도 (m=2 kg, $k_e$=10⁴ N/m, 접근 0.1 m/s)')
ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
for K0, B0, name, c in picks:
    t, xx, ff = sim_impedance(K0, B0, record=True, T=4.0)
    ax1.plot(t, ff, color=c, label=f'{name}: K={K0:.0f}, B={B0:.0f}')
    ax2.plot(t, xx * 1e3, color=c)
ax1.set_ylabel('접촉력 [N]'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_title('세 지점의 시간 응답')
ax2.axhline(x_w * 1e3, color='gray', lw=1.5)
ax2.set_ylabel('x [mm]'); ax2.set_xlabel('t [s]'); ax2.grid(alpha=0.3)
fig.savefig('fig1_kb_sweep_map.png', dpi=140, bbox_inches='tight')
for K0, B0, name, c in picks:
    Fp1, bc1, ts1, Fs1 = sim_impedance(K0, B0)
    print(f"[그림 1] {name}(K={K0:.0f},B={B0:.0f}): F_peak {Fp1[0]:.1f}N (F_ss {Fs1[0]:.1f}N, "
          f"{Fp1[0]/Fs1[0]:.2f}배), 바운스 {bc1[0]}, 정착 {ts1[0]:.2f}s")

# ================= 그림 3: 어드미턴스 안정 경계 =================
print("\n[WE-3] 어드미턴스 vs 임피던스 안정성")
print(f"해석 경계 (K=200, Md=2, τ=5ms): B=40 → k_e,max = {adm_ke_max(40):.0f} N/m")
# (a) 경계 곡선 + 시뮬 분류
B_list = np.array([15, 25, 40, 70, 120, 200.])
ke_list = np.logspace(np.log10(1500), np.log10(6e4), 14)
BB2, KE2 = np.meshgrid(B_list, ke_list, indexing='ij')
unst, osc, _ = sim_admittance(np.full(BB2.size, 200.0), BB2.ravel(), KE2.ravel())
unst = unst.reshape(BB2.shape)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Bfine = np.linspace(10, 250, 200)
ax[0].plot(Bfine, adm_ke_max(Bfine), 'k-', lw=2, label='해석 경계 (Routh–Hurwitz)')
for i, b in enumerate(B_list):
    for j, k in enumerate(ke_list):
        ax[0].plot(b, k, 'x' if unst[i, j] else 'o',
                   color='crimson' if unst[i, j] else 'seagreen', ms=5, alpha=0.8)
ax[0].plot([], [], 'o', color='seagreen', label='시뮬: 안정')
ax[0].plot([], [], 'x', color='crimson', label='시뮬: 채터/발산')
ax[0].set_yscale('log'); ax[0].set_xlabel('$B_d$ [N·s/m]'); ax[0].set_ylabel('$k_e$ [N/m]')
ax[0].set_title('(a) 어드미턴스 안정 경계 ($K_d$=200, $τ_c$=5 ms)')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which='both')
# (b) 시간 응답: 경계 아래/위 + 임피던스
t1, x1, f1 = sim_admittance(200., 40., 4000., record=True, T=6.0)
t2, x2, f2 = sim_admittance(200., 40., 12000., record=True, T=6.0)
t3, x3, f3 = sim_impedance(200., 40., ke=1e5, tau_lag=5e-3, record=True, T=6.0)
ax[1].plot(t1, f1, color='seagreen', label='어드미턴스, $k_e$=4000 (안정)')
ax[1].plot(t2, f2, color='crimson', alpha=0.85, label='어드미턴스, $k_e$=12000 (채터)')
ax[1].plot(t3, f3, color='navy', lw=1.2, label='임피던스, $k_e$=100000 (같은 5 ms 지연)')
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('접촉력 [N]')
ax[1].set_title('(b) 경계 양쪽의 거동'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
ax[1].set_ylim(-2, 60)
fig.tight_layout(); fig.savefig('fig3_admittance_stability.png', dpi=140, bbox_inches='tight')
# 경계 수치 확인 (B=40): 시뮬에서 안정→불안정 전환 구간
un_row, _, _ = sim_admittance(np.full(20, 200.0), np.full(20, 40.0), np.linspace(7000, 11000, 20))
kes = np.linspace(7000, 11000, 20)
idx = np.argmax(un_row)
print(f"시뮬 전환(B=40): {kes[idx-1]:.0f} ~ {kes[idx]:.0f} N/m (해석 8820)")
# 임피던스 쌍대 한계: B > τK → K_max = B/τ = 8000
for Kt in (6000., 10000.):
    Fp3, bc3, _, _ = sim_impedance(Kt, 40., ke=1e4, tau_lag=5e-3, T=6.0)
    print(f"임피던스+지연 5ms, K_d={Kt:.0f}: F_peak={Fp3[0]:.1f} N, 바운스 {bc3[0]}회 "
          f"({'불안정' if bc3[0] > 20 else '안정'}) — 해석 한계 K_max = B/τ = 8000")
print("임피던스+지연 결합 3차식 τm·s³ + m·s² + (B+τk_e)s + (K+k_e) 의 최대 실부 (K=200, B=40):")
for ke_t in (1e4, 1e5, 1e6):
    r = np.roots([5e-3 * m, m, 40 + 5e-3 * ke_t, 200 + ke_t])
    print(f"  k_e={ke_t:.0e}: max Re(s) = {r.real.max():+.2f}  ({'안정' if r.real.max() < 0 else '불안정'})")
# 어드미턴스 결합 3차식과 대조
print("어드미턴스 결합 3차식 (τs+1)(Md s²+Bs+K)+k_e 의 최대 실부 (같은 파라미터):")
for ke_t in (4e3, 1e4, 1e5):
    poly = np.polymul([5e-3, 1.0], [2.0, 40.0, 200.0]); poly[-1] += ke_t
    r = np.roots(poly)
    print(f"  k_e={ke_t:.0e}: max Re(s) = {r.real.max():+.2f}  ({'안정' if r.real.max() < 0 else '불안정'})")

# ================= 그림 4: peg-in-hole 미니 (2-DoF) =================
print("\n[WE-4] peg-in-hole")
mm = 1e-3
c_hole, c_cham, d_cham = 0.5 * mm, 2.5 * mm, 2.0 * mm    # 구멍 반폭, 챔퍼 바깥 반폭, 챔퍼 깊이 (45°)
mu, k_c = 0.3, 1.0e5
m_peg = 0.5
b_c = 2 * 0.4 * np.sqrt(k_c * m_peg)
v_down = 0.01                                             # 하강 속도 10 mm/s

def faces():
    F = []
    # (A, Bpt, n) — 오른쪽/왼쪽 대칭. n 은 재료 밖(자유 공간) 방향.
    for s in (+1, -1):
        F.append((np.array([s * c_cham, 0]), np.array([s * 0.20, 0]), np.array([0, 1.0])))          # 윗면
        n = np.array([-s, 1.0]) / np.sqrt(2)
        F.append((np.array([s * c_cham, 0]), np.array([s * c_hole, -d_cham]), n))                   # 챔퍼
        F.append((np.array([s * c_hole, -d_cham]), np.array([s * c_hole, -0.05]), np.array([-s * 1.0, 0.0])))  # 벽
    return F
FACES = faces()
TOL = 0.1 * mm

def contact_force(p, v):
    """가장 얕게 파고든 면 하나에 페널티+마찰. 면 뒤(재료 쪽)이고 세그먼트 범위 안일 때만."""
    best = None
    for A, Bp, n in FACES:
        depth = -np.dot(p - A, n)                 # 면 평면 뒤로 파고든 깊이
        if depth <= 0:
            continue
        t_vec = Bp - A; L = np.linalg.norm(t_vec); t_hat = t_vec / L
        s_raw = np.dot(p - A, t_hat)
        if s_raw < -TOL or s_raw > L + TOL:       # 세그먼트 범위 밖 → 이 면 아님
            continue
        if best is None or depth < best[0]:
            best = (depth, n)
    if best is None:
        return np.zeros(2)
    pen, n = best
    pen_rate = -np.dot(v, n)
    N = max(k_c * pen + b_c * pen_rate, 0.0)
    t_hat = np.array([-n[1], n[0]])
    vt = np.dot(v, t_hat)
    Ft = -mu * N * np.tanh(vt / 1e-3)
    return N * n + Ft * t_hat

def sim_peg(Kx, Kz, x_off=1.5 * mm, T=2.6, dt=1e-5, F_stop=15.0):
    K = np.array([Kx, Kz]); B = 2 * np.sqrt(m_peg * K)
    p = np.array([x_off, 5 * mm]); v = np.zeros(2)
    traj, forces, ts = [], [], []
    stopped = None
    for i in range(int(T / dt)):
        t = i * dt
        z_d = max(5 * mm - v_down * t, -15 * mm)
        pd = np.array([x_off, z_d])
        vd = np.array([0, -v_down if z_d > -15 * mm else 0.0])
        Fc = contact_force(p, v)
        if F_stop is not None and stopped is None and np.linalg.norm(Fc) > F_stop:
            stopped = t                       # 충돌 반사(reflex): 그 자리에서 정지
        if stopped is not None:
            pd, vd = p.copy(), np.zeros(2)    # 명령 동결
        u = K * (pd - p) + B * (vd - v)
        v += dt * (u + Fc) / m_peg
        p += v * dt
        if i % 100 == 0:
            traj.append(p.copy()); forces.append(np.linalg.norm(Fc)); ts.append(t)
    return np.array(ts), np.array(traj), np.array(forces), stopped

ts_s, tr_s, fo_s, stop_s = sim_peg(20000, 20000)             # 강성 위치 제어
ts_c, tr_c, fo_c, stop_c = sim_peg(100, 1500)                # 컴플라이언트
ts_n, tr_n, fo_n, _ = sim_peg(20000, 20000, F_stop=None)     # 반사 껐다면?
print(f"위치 제어(K=2e4): 최종 깊이 {tr_s[-1,1]/mm:+.2f} mm, F_peak {fo_s.max():.1f} N, "
      f"반사 정지 t={f'{stop_s:.2f} s' if stop_s else '없음'}")
print(f"컴플라이언스(Kx=100, Kz=1500): 최종 깊이 {tr_c[-1,1]/mm:+.2f} mm, F_peak {fo_c.max():.2f} N, "
      f"반사 정지 {stop_c}")
print(f"위치 제어+반사 OFF: 최종 깊이 {tr_n[-1,1]/mm:+.2f} mm, F_peak {fo_n.max():.1f} N, "
      f"말미 접촉력 {fo_n[-1]:.1f} N")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
# 기하
xs_geo = np.linspace(-6 * mm, 6 * mm, 400)
def surf(x):
    ax_ = abs(x)
    if ax_ >= c_cham: return 0.0
    if ax_ >= c_hole: return -(c_cham - ax_)
    return -15 * mm
zs_geo = np.array([surf(x) for x in xs_geo])
ax[0].fill_between(xs_geo / mm, zs_geo / mm, -6, color='0.85', zorder=0)
ax[0].plot(tr_n[:, 0] / mm, tr_n[:, 1] / mm, color='crimson', lw=1, ls=':', alpha=0.7,
           label=f'위치 제어, 반사 OFF (벽에 {fo_n[-1]:.0f} N)')
ax[0].plot(xs_geo / mm, zs_geo / mm, color='0.4', lw=1)
ax[0].plot(tr_s[:, 0] / mm, tr_s[:, 1] / mm, color='crimson', lw=2, label='위치 제어 (K=20000)')
if stop_s:
    ax[0].plot(tr_s[-1, 0] / mm, tr_s[-1, 1] / mm, 'X', color='crimson', ms=12,
               label=f'반사 정지 ({fo_s.max():.0f} N)')
ax[0].plot(tr_c[:, 0] / mm, tr_c[:, 1] / mm, color='seagreen', lw=2, label='임피던스 ($K_x$=100)')
ax[0].set_xlim(-4, 5); ax[0].set_ylim(-6, 5.5)
ax[0].set_xlabel('x [mm]'); ax[0].set_ylabel('z [mm]')
ax[0].set_title(f'(a) 팁 궤적 (오프셋 1.5 mm, 틈새 0.5 mm, μ=0.3)')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].set_aspect('equal')
ax[1].plot(ts_s, fo_s, color='crimson', label='위치 제어')
ax[1].plot(ts_c, fo_c, color='seagreen', label='임피던스')
ax[1].axhline(15, color='k', ls='--', lw=1)
ax[1].text(0.05, 14.1, '반사 임계 15 N (Franka의 setCollisionBehavior 스타일)', fontsize=8)
ax[1].annotate(f'임피던스 peak {fo_c.max():.1f} N', xy=(0.62, fo_c.max()), xytext=(1.0, 4.0),
               color='seagreen', fontsize=9, arrowprops=dict(arrowstyle='->', color='seagreen'))
ax[1].annotate('반사 정지\n(명령 동결)', xy=(stop_s, 15), xytext=(1.1, 11),
               color='crimson', fontsize=9, arrowprops=dict(arrowstyle='->', color='crimson'))
ax[1].set_ylim(-0.5, 16.5)
ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('|접촉력| [N]')
ax[1].set_title('(b) 접촉력 — 삽입의 성패는 힘의 방향과 크기'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig('fig4_peg_in_hole.png', dpi=140, bbox_inches='tight')
print("그림 4 저장 완료")
