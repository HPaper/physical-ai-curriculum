"""Lec R15 그림 생성 스크립트.
fig1: (a) 감속의 기본 교환 — 토크-속도 곡선의 환전 (b) 숨은 가격표 — 반사 관성 n²J_m
fig2: 부하 가속도 vs 감속비 — 최적점 n* = sqrt(J_L/J_m) (임피던스 매칭)
fig3: 충돌 모델 — (a) 피크 접촉력 vs n (b) QDD vs 하모닉 충격 에너지 분배
fig4: 감속기 유형별 트레이드오프 레이더 (정성 점수)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lec15"

# 공통 파라미터 (본문 WE와 동일)
tau_stall = 0.5      # 모터 스톨 토크 [Nm]
w_nl = 500.0         # 무부하 속도 [rad/s]
J_m = 1e-5           # 소형 BLDC 로터 관성 [kg m^2]
J_L = 0.5            # 팔 링크+페이로드 관성 [kg m^2]

# ============================================================
# fig 1 — (a) 기본 교환 (b) 반사 관성
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# (a) 같은 모터의 토크-속도 직선이 n에 따라 환전됨 (log-log)
P_pk = tau_stall * w_nl / 4          # 선형 T-ω 곡선의 피크 동력
for n, c in [(1, '0.55'), (10, 'royalblue'), (100, 'crimson')]:
    w = np.linspace(w_nl/n * 1e-3, w_nl/n * 0.999, 400)
    tau = n * tau_stall * (1 - n*w/w_nl)
    ax1.loglog(w, tau, color=c, lw=2.2, label=f'n = {n}')
w = np.logspace(-1, np.log10(w_nl*1.2), 200)
ax1.loglog(w, P_pk/w, 'k--', lw=1.0, alpha=0.6)
ax1.annotate('일정 동력선 $\\tau\\omega = P_{peak}$\n(감속기는 이 선 위에서만 환전)',
             (0.5, P_pk/0.5), fontsize=9, xytext=(0.15, 8),
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.set_xlim(0.1, 700); ax1.set_ylim(0.005, 200)
ax1.set_xlabel('출력 속도 $\\omega_{out}$ (rad/s)')
ax1.set_ylabel('출력 토크 $\\tau_{out}$ (Nm)')
ax1.set_title('(a) 기본 교환: 속도를 팔아 토크를 산다 ($\\eta=1$ 가정)')
ax1.legend(fontsize=9); ax1.grid(True, which='both', alpha=0.25)

# (b) 반사 관성 n^2 J_m (log-log) + 실제 감속비 마커
n = np.logspace(0, 2.8, 200)
ax2.loglog(n, n**2 * J_m, color='darkslateblue', lw=2.2,
           label=r'$J_{ref} = n^2 J_m$  ($J_m = 10^{-5}$)')
ax2.axhline(J_L, color='0.35', ls='--', lw=1.2)
ax2.text(1.3, J_L*0.45, r'부하 관성 $J_L = 0.5$ (점선)', fontsize=9, color='0.25')
n_star = np.sqrt(J_L/J_m)
ax2.plot(n_star, J_L, '*', color='goldenrod', ms=16, zorder=6,
         label=f'$n^*=\\sqrt{{J_L/J_m}}={n_star:.0f}$ (임피던스 매칭)')
for nn, name, c in [(8, 'QDD\n(저감속 유성)', 'royalblue'),
                    (51, '싸이클로이드\nDYD', 'darkorange'),
                    (100, '하모닉', 'crimson'),
                    (345, '버스서보\nSTS3215', 'seagreen')]:
    ax2.plot(nn, nn**2*J_m, 'o', color=c, ms=8, zorder=5)
    ax2.annotate(name, (nn, nn**2*J_m), textcoords='offset points',
                 xytext=(6, -24), fontsize=8, color=c, ha='left')
ax2.set_xlabel('감속비 $n$'); ax2.set_ylabel('$J_{ref}$ (kg·m²)')
ax2.set_title('(b) 숨은 가격표: 로터 관성은 $n^2$배로 보인다')
ax2.legend(fontsize=8, loc='upper left'); ax2.grid(True, which='both', alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_exchange_reflected.png", dpi=150)
plt.close(fig)

# ============================================================
# fig 2 — 부하 가속도 vs 감속비 (최적점)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
n = np.logspace(0, 3.2, 500)
for JL, c, ls in [(0.5, 'darkslateblue', '-'), (0.05, 'seagreen', '--')]:
    a = n*tau_stall/(JL + n**2*J_m)
    ns = np.sqrt(JL/J_m); amax = tau_stall/(2*np.sqrt(J_m*JL))
    ax.semilogx(n, a, color=c, ls=ls, lw=2.2,
                label=f'$J_L={JL}$: $n^*={ns:.0f}$, $a_{{max}}={amax:.0f}$ rad/s²')
    ax.plot(ns, amax, 'o', color=c, ms=9, zorder=5)
    ax.axvline(ns, color=c, ls=':', lw=0.9, alpha=0.6)
# J_L=0.5 곡선 위 실제 감속비 지점
for nn, name, c, off in [(8, 'QDD n=8', 'royalblue', (8, 10)),
                         (100, '하모닉 n=100', 'crimson', (-12, -32))]:
    a_pt = nn*tau_stall/(0.5 + nn**2*J_m)
    ax.plot(nn, a_pt, 's', color=c, ms=8, zorder=6)
    ax.annotate(f'{name}\n{a_pt:.0f} rad/s²', (nn, a_pt),
                textcoords='offset points', xytext=off, fontsize=8.5, color=c)
a_srv = 345*tau_stall/(0.5 + 345**2*J_m)
ax.plot(345, a_srv, 's', color='seagreen', ms=8, zorder=6)
ax.annotate(f'서보 n=345\n{a_srv:.0f} rad/s²', (345, a_srv), xytext=(430, 12),
            textcoords='data', fontsize=8.5, color='seagreen',
            arrowprops=dict(arrowstyle='->', color='seagreen', lw=0.9))
ax.text(2.2, 88, '토크 부족 구간\n($J_L$ 지배: $a \\approx n\\tau_m/J_L$)',
        fontsize=9, color='0.3')
ax.text(1500, 175, '관성 지배 구간\n($a \\approx \\tau_m/(nJ_m)$)',
        fontsize=9, color='0.3', ha='right')
ax.set_xlabel('감속비 $n$'); ax.set_ylabel('부하 각가속도 $\\ddot{q}$ (rad/s²)')
ax.set_title('같은 모터($\\tau_m=0.5$ Nm, $J_m=10^{-5}$)로 낼 수 있는 부하 가속도 — 최적 감속비는 $\\sqrt{J_L/J_m}$')
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, which='both', alpha=0.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_accel_vs_ratio.png", dpi=150)
plt.close(fig)

# ============================================================
# fig 3 — 충돌: 피크 접촉력과 충격 에너지 분배 (WE-3와 동일 파라미터)
# ============================================================
J_leg, r, v, k = 0.02, 0.3, 1.0, 2e4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

n = np.logspace(0, 2.7, 300)
for Jm_i, c, lab in [(1e-5, 'crimson', '소구경 고속 로터 ($J_m=10^{-5}$)'),
                     (1.2e-4, 'royalblue', '대구경 토크 로터 ($J_m=1.2\\times10^{-4}$)')]:
    m_eff = (J_leg + n**2*Jm_i)/r**2
    ax1.semilogx(n, v*np.sqrt(k*m_eff), color=c, lw=2.2, label=lab)
F_bare = v*np.sqrt(k*J_leg/r**2)
ax1.axhline(F_bare, color='0.4', ls='--', lw=1.0)
ax1.text(150, F_bare + 60, f'링크만 무장착: {F_bare:.0f} N (점선)', fontsize=8.5, color='0.3')
for nn, Jm_i, c, name, off in [(8, 1.2e-4, 'royalblue', 'QDD (n=8)', (0, 16)),
                               (100, 1e-5, 'crimson', '하모닉 (n=100)', (-4, 20))]:
    F = v*np.sqrt(k*(J_leg + nn**2*Jm_i)/r**2)
    ax1.plot(nn, F, 'o', color=c, ms=10, zorder=6)
    ax1.annotate(f'{name}: {F:.0f} N', (nn, F), textcoords='offset points',
                 xytext=off, fontsize=9, color=c, ha='center')
ax1.set_xlabel('감속비 $n$'); ax1.set_ylabel('피크 접촉력 $F_{peak}$ (N)')
ax1.set_title('(a) 착지 충격 $F_{peak} = v\\sqrt{k\\,m_{eff}}$ — 반사 관성이 세게 때린다')
ax1.legend(fontsize=8.5, loc='upper left'); ax1.grid(True, which='both', alpha=0.25)

# (b) 충격 운동에너지의 분배 (링크 vs 로터) — 기어를 지나야 하는 에너지
cases = [('QDD\n$n=8$', 8, 1.2e-4), ('하모닉\n$n=100$', 100, 1e-5)]
names = [c[0] for c in cases]
link_sh, rot_sh, taus = [], [], []
for _, nn, Jm_i in cases:
    Jref = nn**2*Jm_i; Jtot = J_leg + Jref
    link_sh.append(J_leg/Jtot*100); rot_sh.append(Jref/Jtot*100)
    F = v*np.sqrt(k*Jtot/r**2)
    taus.append(Jref * F*r/Jtot)     # 피크에서 기어를 통과하는 토크
x = np.arange(2)
ax2.bar(x, link_sh, 0.5, color='0.75', label='링크 몫 (기어 안 지남)')
ax2.bar(x, rot_sh, 0.5, bottom=link_sh, color='indianred',
        label='로터 몫 (기어 이빨을 지나 감속돼야 함)')
for i in range(2):
    ax2.text(i, link_sh[i] + rot_sh[i]/2, f'{rot_sh[i]:.0f}%\n(기어 통과 토크\n{taus[i]:.1f} Nm)',
             ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax2.set_xticks(x, names); ax2.set_ylabel('충격 운동에너지 분배 (%)')
ax2.set_ylim(0, 105)
ax2.set_title('(b) 같은 충돌, 에너지의 행선지 — R16(QDD 철학) 예고')
ax2.legend(fontsize=8.5, loc='lower right')
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_impact_inertia.png", dpi=150)
plt.close(fig)

# ============================================================
# fig 4 — 감속기 유형 트레이드오프 레이더 (정성 점수 1~5)
# ============================================================
axes_lab = ['토크 밀도\n(고감속)', '정밀\n(저백래시)', '비틀림 강성',
            '역구동성', '충격 내성', '효율']
data = {
    '하모닉':            ([5, 5, 3, 1, 2, 3], 'crimson'),
    '싸이클로이드(DYD)': ([4, 4, 4, 2, 4, 3], 'darkorange'),
    'QDD(저감속 유성)':  ([2, 3, 4, 5, 5, 4], 'royalblue'),
    '다단 스퍼(버스서보)': ([5, 1, 2, 1, 2, 2], 'seagreen'),
    '벨트/텐던':         ([2, 2, 1, 4, 5, 4], 'mediumorchid'),
}
K = len(axes_lab)
ang = np.linspace(0, 2*np.pi, K, endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw=dict(polar=True))
for name, (vals, c) in data.items():
    vv = vals + vals[:1]
    ax.plot(ang, vv, color=c, lw=2.0, label=name)
    ax.fill(ang, vv, color=c, alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_lab, fontsize=9.5)
ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8)
ax.set_ylim(0, 5.2)
ax.set_title('감속기·전동 유형별 트레이드오프 (정성 점수, 상위 25강 표 기반)\n'
             '— 전 축 만점은 없다: 어떤 축을 포기할지가 로봇의 정체성', fontsize=10.5, pad=22)
ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.12), fontsize=8.5)
fig.savefig(f"{OUT}/fig4_transmission_radar.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("saved: fig1_exchange_reflected, fig2_accel_vs_ratio, fig3_impact_inertia, fig4_transmission_radar")
