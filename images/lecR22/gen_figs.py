"""Lec R22 그림 생성 스크립트.
실행: python3 gen_figs.py  (이 디렉토리에서)
fig1: wiping 하이브리드 vs 위치 제어 시계열 (한 장 요약)
fig2: 자연/인공 구속 기하 다이어그램 (wiping / peg-in-hole)
fig3: 힘 루프 게인-강성 안정성 지도 + 강성 스윕 과도응답
fig4: F/T 센서 노이즈 vs 전류 기반 추정 바이어스
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

C = dict(hyb='#1565c0', pos1='#e65100', pos2='#8e24aa', tgt='#455a64',
         ft='#2e7d32', cur='#c62828', ok='#a5d6a7', bad='#ef9a9a')

# ================================================================ 공통 시뮬레이터
m   = 2.0      # EEF 유효 질량 [kg]
dt  = 1e-3
k_e = 5000.0   # 환경 강성 [N/m]
b_e = 10.0
mu  = 0.3
W   = 3.0      # 미지 툴 하중 [N]
f_d = 5.0
kp_x, kd_x = 400.0, 2*np.sqrt(400.0*m)
K_fp, K_fi = 1.0, 200.0
K_dz = 2*np.sqrt(m*k_e*(1+K_fp))
T = 4.0

def xd_of_t(t):
    if t < 0.5: return 0.0, 0.0
    if t < 3.5: return 0.1*(t-0.5), 0.1
    return 0.3, 0.0

def contact_force(z, zdot, ke):
    d = -z
    if d <= 0: return 0.0
    return max(ke*d + b_e*(-zdot), 0.0)

def sim_hybrid(ke=k_e, use_integral=True, sensor=None, seed=0):
    rng = np.random.default_rng(seed)
    n = int(T/dt)
    x = np.array([0.0, 0.005]); v = np.zeros(2)
    I_f, in_contact, f_lp = 0.0, False, 0.0
    alpha = dt/(dt + 1/(2*np.pi*50))
    log = np.zeros((n, 6))
    for k in range(n):
        t = k*dt
        f_true = contact_force(x[1], v[1], ke)
        if sensor is None:
            f_meas = f_true
        elif sensor[0] == 'ft':
            f_lp += alpha*((f_true + rng.normal(0, sensor[1])) - f_lp)
            f_meas = f_lp
        else:
            _, ge, bias, sg = sensor
            f_meas = (1+ge)*f_true + bias + rng.normal(0, sg)
        xd, vd = xd_of_t(t)
        ux = kp_x*(xd - x[0]) + kd_x*(vd - v[0])
        if not in_contact:
            uz = 60.0*(-0.05 - v[1])
            if f_meas > 0.5: in_contact, I_f = True, 0.0
        if in_contact:
            e_f = f_d - f_meas
            if use_integral: I_f += e_f*dt
            uz = -(f_d + K_fp*e_f + K_fi*I_f) - K_dz*v[1]
        fric = -mu*f_true*np.tanh(v[0]/1e-3)
        v += np.array([(ux+fric)/m, (uz - W + f_true)/m])*dt
        x += v*dt
        log[k] = [t, x[0], x[1], f_true, f_meas, xd]
    return log

def sim_position(z_err):
    kp_z = 2e4; kd_z = 2*np.sqrt(kp_z*m)
    z_d = z_err - f_d/k_e
    n = int(T/dt)
    x = np.array([0.0, 0.005]); v = np.zeros(2)
    log = np.zeros((n, 4))
    for k in range(n):
        t = k*dt
        f_true = contact_force(x[1], v[1], k_e)
        xd, vd = xd_of_t(t)
        ux = kp_x*(xd - x[0]) + kd_x*(vd - v[0])
        uz = kp_z*(z_d - x[1]) - kd_z*v[1]
        fric = -mu*f_true*np.tanh(v[0]/1e-3)
        v += np.array([(ux+fric)/m, (uz - W + f_true)/m])*dt
        x += v*dt
        log[k] = [t, x[1], f_true, xd]
    return log

# ================================================================ fig 1
hyb = sim_hybrid()
p0, p1, p2 = sim_position(0.0), sim_position(1e-3), sim_position(-2e-3)

fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.45, wspace=0.28)
ax = fig.add_subplot(gs[:, 0])
ax.plot(hyb[:,0], hyb[:,3], color=C['hyb'], lw=1.8, label='하이브리드 (법선=힘 제어)')
ax.plot(p1[:,0], p1[:,2], color=C['pos1'], lw=1.4, ls='--', label='위치 제어, 표면 추정 +1 mm → 0.6 N')
ax.plot(p2[:,0], p2[:,2], color=C['pos2'], lw=1.4, ls='-.', label='위치 제어, 표면 추정 −2 mm → 12.6 N')
ax.axhline(f_d, color=C['tgt'], lw=1, ls=':', label='목표 5 N')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('법선 접촉력 $f_n$ [N]')
ax.set_title('(a) 같은 wiping, 법선 방향을 무엇으로 제어하는가')
ax.set_ylim(-0.5, 14); ax.legend(fontsize=8.5, loc='upper right'); ax.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(hyb[:,0], hyb[:,5]*100, color=C['tgt'], lw=1.2, ls=':', label='목표 $x_d$')
ax2.plot(hyb[:,0], hyb[:,1]*100, color=C['hyb'], lw=1.5, label='실제 $x$')
ax2.set_ylabel('접선 위치 [cm]'); ax2.set_title('(b) 접선 = 위치 추종', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(hyb[:,0], hyb[:,2]*1000, color=C['hyb'], lw=1.5)
ax3.axhline(0, color='k', lw=0.8)
ax3.fill_between([0, T], [-2, -2], [0, 0], color='0.85')
ax3.set_xlabel('시간 [s]'); ax3.set_ylabel('$z$ [mm]')
ax3.set_title('(c) 법선: 접근 → 접촉 → 1 mm 침투 유지', fontsize=10)
ax3.set_ylim(-2, 5.5); ax3.grid(alpha=0.3)
fig.savefig('fig1_wiping_hybrid.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ================================================================ fig 2: 구속 기하
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for a in axes: a.set_aspect('equal'); a.axis('off')

ax = axes[0]
ax.set_title('(a) wiping: 평면 접촉', fontsize=11)
ax.add_patch(plt.Rectangle((-1.6, -0.55), 3.2, 0.55, color='0.82'))
ax.plot([-1.6, 1.6], [0, 0], color='k', lw=1.5)
ax.add_patch(plt.Circle((0, 0.12), 0.13, color=C['hyb']))
ax.plot([0, -0.5], [0.12, 0.9], color='0.4', lw=5)  # 툴 자루
ax.annotate('', xy=(0, -0.42), xytext=(0, 0.05),
            arrowprops=dict(arrowstyle='-|>', color=C['cur'], lw=2.2))
ax.text(0.08, -0.38, '법선: 힘 지정\n$f_n = f_d$ (인공)\n$v_n = 0$ (자연)', color=C['cur'], fontsize=9.5)
ax.annotate('', xy=(1.15, 0.24), xytext=(0.18, 0.24),
            arrowprops=dict(arrowstyle='-|>', color=C['ft'], lw=2.2))
ax.text(0.42, 0.34, '접선: 운동 지정\n$v_t = v_d$ (인공)\n$f_t$는 마찰이 결정 (자연)', color=C['ft'], fontsize=9.5)
ax.text(-1.5, 0.75, '구속 좌표계는 접촉 기하에\n붙는다 (표면이 기울면 같이 기운다)', fontsize=9, color='0.35')
ax.set_xlim(-1.7, 1.9); ax.set_ylim(-0.7, 1.15)

ax = axes[1]
ax.set_title('(b) peg-in-hole: 원통 접촉', fontsize=11)
ax.add_patch(plt.Rectangle((-1.3, -1.0), 1.02, 1.0, color='0.82'))
ax.add_patch(plt.Rectangle((0.28, -1.0), 1.02, 1.0, color='0.82'))
ax.add_patch(plt.Rectangle((-0.2, -0.75), 0.4, 1.45, color=C['hyb'], alpha=0.85))
ax.annotate('', xy=(0, -0.95), xytext=(0, -0.05),
            arrowprops=dict(arrowstyle='-|>', color=C['ft'], lw=2.2))
ax.text(0.06, -0.62, '축방향: 운동 지정\n$v_z = v_d$ (삽입 진행)', color=C['ft'], fontsize=9.5)
ax.annotate('', xy=(0.62, 0.35), xytext=(-0.62, 0.35),
            arrowprops=dict(arrowstyle='<|-|>', color=C['cur'], lw=2.2))
ax.text(0.60, 0.42, '횡방향: 힘 지정\n$f_x = f_y = 0$\n(구멍 벽에 눌리지 않기)', color=C['cur'], fontsize=9.5)
ax.annotate('', xy=(0.42, 0.86), xytext=(-0.42, 0.98),
            arrowprops=dict(arrowstyle='<|-|>', color=C['cur'], lw=1.6,
                            connectionstyle='arc3,rad=0.35'))
ax.text(-1.25, 1.02, '기울기 모멘트: $m_x = m_y = 0$ (힘 지정)', color=C['cur'], fontsize=9.5)
ax.set_xlim(-1.4, 2.1); ax.set_ylim(-1.1, 1.25)
fig.suptitle('접촉 기하가 방향별 목표를 결정한다 — 각 방향은 위치 또는 힘, 둘 중 하나만 지정할 수 있다', y=1.02, fontsize=11)
fig.savefig('fig2_constraints.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ================================================================ fig 3: 안정성 지도 + 강성 스윕
# (a) 이산 어드미턴스 힘 루프 + 측정 지연 1샘플: e_{k+1} = e_k - a e_{k-1}, a = k_e K_f dt
kes = np.logspace(2, 6, 120)
kfs = np.logspace(-4, 0, 120)
KE, KF = np.meshgrid(kes, kfs)
A = KE*KF*dt
# 특성근 |z| 최대 (z^2 - z + a = 0)
disc = 1 - 4*A + 0j
r1 = np.abs((1 + np.sqrt(disc))/2)
r2 = np.abs((1 - np.sqrt(disc))/2)
rho = np.maximum(r1, r2)
settle = np.where(rho < 1, np.ceil(3/np.maximum(-np.log(rho), 1e-12)), np.nan)  # 스텝 수

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
pc = ax.pcolormesh(KE, KF, np.log10(np.clip(settle, 1, 1e5)), cmap='viridis_r', shading='auto')
ax.contourf(KE, KF, np.where(np.isnan(settle), 1.0, np.nan), colors=[C['bad']])
ax.plot(kes, 1/(kes*dt), color='k', lw=2, label='안정 한계 $K_f = 1/(k_e\\,\\Delta t)$ (지연 1샘플)')
ax.plot(kes, 2/(kes*dt), color='k', lw=1.2, ls='--', label='지연 없음: $2/(k_e\\,\\Delta t)$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('환경 강성 $k_e$ [N/m]'); ax.set_ylabel('힘 게인 $K_f$ [m/(N·s)]')
ax.set_title('(a) 힘 루프 안정성 지도 — 루프 게인 = $k_e K_f \\Delta t$')
cb = fig.colorbar(pc, ax=ax); cb.set_label('log$_{10}$(정착 스텝 수)')
ax.text(2e5, 3e-1, '불안정', fontsize=11, color='#7f0000')
ax.legend(fontsize=8, loc='lower left')

ax = axes[1]
for ke_, col in [(5e3, C['hyb']), (5e4, C['pos1']), (5e5, C['pos2'])]:
    lg = sim_hybrid(ke=ke_)
    zeta = K_dz/(2*np.sqrt(m*ke_*(1+K_fp)))
    ax.plot(lg[:,0], lg[:,3], color=col, lw=1.3,
            label=f'$k_e$={ke_:.0e} N/m (ζ={zeta:.2f}), 피크 {lg[:,3].max():.1f} N')
ax.axhline(f_d, color=C['tgt'], lw=1, ls=':')
ax.set_xlim(0.0, 0.6); ax.set_ylim(0, 56)
ax.set_xlabel('시간 [s]'); ax.set_ylabel('접촉력 [N]')
ax.set_title('(b) 같은 게인($k_e$=5000용 ζ=1 설계), 다른 강성')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.savefig('fig3_stiffness_gain_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ================================================================ fig 4: 센서 트레이드오프
ft = sim_hybrid(sensor=('ft', 0.25), seed=3)
cu = sim_hybrid(sensor=('cur', 0.05, 1.2, 0.05), seed=3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={'width_ratios': [1.7, 1]})
ax = axes[0]
ax.plot(ft[:,0], ft[:,3], color=C['ft'], lw=0.9, alpha=0.9,
        label='F/T 센서 (노이즈 σ=0.25 N): 떨리지만 평균은 5 N')
ax.plot(cu[:,0], cu[:,3], color=C['cur'], lw=1.6,
        label='전류 추정 (5% 게인오차+1.2 N 바이어스): 매끈하지만 3.62 N')
ax.axhline(f_d, color=C['tgt'], lw=1, ls=':', label='목표 5 N')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('실제 접촉력 [N]')
ax.set_title('(a) 실제로 가해진 힘 (측정값이 아니라!)')
ax.set_xlim(0, 4); ax.set_ylim(0, 8); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

ax = axes[1]
mk_ft, mk_cu = ft[:,0] >= 2.0, cu[:,0] >= 2.0
e_ft, e_cu = ft[mk_ft,3]-f_d, cu[mk_cu,3]-f_d
xpos = np.arange(2)
ax.bar(xpos-0.18, [abs(e_ft.mean()), abs(e_cu.mean())], 0.36, color=C['tgt'], label='|평균 오차| (바이어스)')
ax.bar(xpos+0.18, [e_ft.std(), e_cu.std()], 0.36, color='#90a4ae', label='표준편차 (노이즈)')
ax.set_xticks(xpos, ['F/T 센서', '전류 추정'])
ax.set_ylabel('힘 오차 [N]'); ax.set_title('(b) 노이즈는 평균되지만\n바이어스는 피드백이 못 본다')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis='y')
for i, (b, s) in enumerate([(abs(e_ft.mean()), e_ft.std()), (abs(e_cu.mean()), e_cu.std())]):
    ax.text(i-0.18, b+0.03, f'{b:.3f}', ha='center', fontsize=8.5)
    ax.text(i+0.18, s+0.03, f'{s:.3f}', ha='center', fontsize=8.5)
fig.savefig('fig4_sensor_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("생성 완료: fig1_wiping_hybrid.png, fig2_constraints.png, fig3_stiffness_gain_map.png, fig4_sensor_tradeoff.png")

# ================================================================ 본문 인용 수치 검증 출력
print("\n--- 본문 수치 검증 ---")
mask = hyb[:, 0] >= 1.0
print(f"[WE-2] 힘 오차 max = {np.abs(hyb[mask,3]-f_d).max()*1000:.1f} mN, "
      f"x RMS = {np.sqrt(np.mean((hyb[mask,1]-hyb[mask,5])**2))*1000:.2f} mm, "
      f"평균 침투 = {-hyb[mask,2].mean()*1000:.3f} mm")
wm = (hyb[:,0] >= 1.5) & (hyb[:,0] <= 3.0)
print(f"[WE-2] wipe 중 평균 접선 오차 = {np.mean(hyb[wm,5]-hyb[wm,1])*1000:.2f} mm (예측 3.75 mm)")
noI = sim_hybrid(use_integral=False)
print(f"[WE-2] 적분 없음 정상 힘 = {noI[noI[:,0]>=2.0,3].mean():.3f} N (예측 6.5 N)")
print(f"[WE-1] 직렬 강성 = {2e4*k_e/(2e4+k_e):.0f} N/m; 위치 제어 정상 힘: "
      f"0mm={p0[p0[:,0]>=2.0,2].mean():.2f} N, +1mm={p1[p1[:,0]>=2.0,2].mean():.2f} N, "
      f"-2mm={p2[p2[:,0]>=2.0,2].mean():.2f} N")
print("[WE-3a] 강성 스윕 (게인 고정):")
for ke_ in [500, 5000, 5e4, 5e5]:
    lg = sim_hybrid(ke=ke_)
    fc = np.argmax(lg[:,3] > 0.5)
    chat = np.mean(lg[fc:,3] < 1e-9)
    zeta = K_dz/(2*np.sqrt(m*ke_*(1+K_fp)))
    print(f"  k_e={ke_:8.0f}: zeta={zeta:5.2f}, 피크={lg[:,3].max():6.2f} N, 접촉이탈={chat*100:4.1f}%")
print("[WE-3b] 센서 트레이드오프 (t>=2s):")
for name, lg in [("F/T ", ft), ("전류", cu)]:
    e = lg[lg[:,0]>=2.0,3] - f_d
    print(f"  {name}: 평균 오차={e.mean():+.3f} N, 표준편차={e.std():.3f} N")
print(f"[WE-3b] 전류 정상 힘 예측 = {(f_d-1.2)/1.05:.3f} N")
