"""Lec 49 그림 생성 + 본문 수치 검증 스크립트 (CPU only, GPU 불필요).

fig1: 감속기 유형별 반사 관성 J_ref = n^2 J_m (로그 막대) — "정책이 보는 하드웨어"
fig2: 시각 토큰 예산 히트맵 (카메라 수 x 해상도) — tokens = (res/patch)^2 * n_cam
fig3: 토크-속도 곡선 — 전압 한계선 + 열 한계선(i^2 R), 그 아래가 연속 작동 영역

본문(lec49)이 인용하는 모든 수치는 이 스크립트의 stdout 과 일치한다.
파라미터 규약은 15강(gen_figs.py)과 동일하게 맞춰 두 강의의 J_ref 값이 교차검증되도록 했다.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lec49"


def banner(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


# ============================================================
# 수식 1 — 반사 관성 J_ref = n^2 J_m (감속기 유형별)
# 15강과 동일 규약: 소구경 로터 J_m=1e-5, QDD 대구경 로터 J_m=1.2e-4
# ============================================================
banner("수식1: 반사 관성 J_reflected = n^2 * J_m (감속기 유형별)")

# (이름, n, J_m[kg m^2], 역구동성 계열)
GEARS = [
    ("QDD 유성",       8,   1.2e-4),   # 저감속 + 대구경 로터
    ("싸이클로이드 DYD", 51,  1.0e-5),  # DYD-14-051
    ("하모닉",         100,  1.0e-5),
    ("서보 STS3215",   345,  1.0e-5),
]
J_L = 0.5          # 어깨 관절 링크+페이로드 관성 [kg m^2] (15강과 동일)
tau_f_motor = 0.01 # 모터축 마찰(쿨롱) [Nm] — 역구동 문턱 하한 계산용

print(f"{'유형':<16s}{'n':>5s}{'J_m':>10s}{'J_ref=n^2 J_m':>16s}"
      f"{'로터몫%':>9s}{'역구동문턱[Nm]':>14s}")
jref_tbl = {}
for name, n, J_m in GEARS:
    J_ref = n**2 * J_m
    jref_tbl[name] = J_ref
    share = J_ref / (J_L + J_ref) * 100
    tau_bd = n * tau_f_motor          # 반사된 모터 마찰 = 역구동 문턱의 하한
    print(f"{name:<16s}{n:>5d}{J_m:>10.1e}{J_ref:>16.5f}"
          f"{share:>9.1f}{tau_bd:>14.2f}")

# 정책 관점 해석용 비율
print(f"\n하모닉 J_ref / QDD J_ref = "
      f"{jref_tbl['하모닉']/jref_tbl['QDD 유성']:.1f}x  "
      f"(같은 외력에도 로터가 {jref_tbl['하모닉']/jref_tbl['QDD 유성']:.1f}배 무겁게 응답)")
print(f"서보 J_ref / QDD J_ref  = "
      f"{jref_tbl['서보 STS3215']/jref_tbl['QDD 유성']:.1f}x")


# ============================================================
# 수식 2 — 시각 토큰 예산 tokens = (res/patch)^2 * n_cam
# ViT patch=14 (SigLIP/DINOv2 계열), 카메라 수 x 해상도
# ============================================================
banner("수식2: 시각 토큰 예산 tokens = (res/patch)^2 * n_cam  (patch=14)")

PATCH = 14
RES = [128, 224, 256, 384, 512]      # 정사각 입력 변 길이 [px]
NCAM = [1, 2, 3]

def tok_per_img(res, patch=PATCH):
    return (res // patch) ** 2

hdr = "res\\ncam"
print(f"{hdr:>10s}" + "".join(f"{c:>10d}" for c in NCAM)
      + f"{'/img':>10s}")
budget = np.zeros((len(RES), len(NCAM)), dtype=int)
for i, res in enumerate(RES):
    per = tok_per_img(res)
    row = ""
    for j, c in enumerate(NCAM):
        budget[i, j] = per * c
        row += f"{budget[i, j]:>10d}"
    print(f"{res:>10d}" + row + f"{per:>10d}")

# self-attention 비용은 토큰 수의 제곱 -> 지연 대리 지표
banner("수식2 보조: prefill self-attention 비용 ~ tokens^2 (상대 지연)")
base = tok_per_img(224) * 1           # 1캠 224 = 기준
print(f"기준(1캠 224px): {base} 토큰")
for (res, c) in [(224, 1), (256, 2), (384, 3), (512, 3)]:
    t = tok_per_img(res) * c
    print(f"  {c}캠 x {res}px: {t:>4d} 토큰  -> attention 상대비용 "
          f"{(t/base)**2:6.1f}x  (토큰 {t/base:.1f}x)")


# ============================================================
# 수식 3 — 토크-속도 곡선: 전압 한계 + 열 한계(i^2 R)
# BLDC/PMSM: tau = Kt * i,  전압한계 tau_max(w) = Kt/R*(V - Ke*w)
# 열 한계(연속): i_rms 로 정한 tau_cont = Kt * i_cont (속도 무관 수평선)
# ============================================================
banner("수식3: 토크-속도 곡선 (전압 한계선 + 열 한계 i^2 R)")

# 대표 소형 관절 모터 파라미터 (14강 규약과 정합, 예시값)
Kt = 0.10     # 토크 상수 [Nm/A]
Ke = 0.10     # 역기전력 상수 [V/(rad/s)] (SI에서 Kt=Ke)
R = 0.5       # 상 저항 [ohm]
V = 24.0      # 버스 전압 [V]
i_peak = 12.0 # 순간(피크) 전류 한계 [A]
i_cont = 5.0  # 연속(열 한계) 전류 [A]  <- i^2 R 발열이 정상상태 방열과 균형

w_stall = V / Ke                       # 이론상 무부하 속도 상한(전압선이 tau=0 되는 곳)
tau_stall_volt = Kt / R * V            # w=0 에서 전압선이 허용하는 최대 토크
tau_peak = Kt * i_peak                 # 피크 전류가 정한 토크 천장
tau_cont = Kt * i_cont                 # 연속(열) 토크 천장
P_diss_cont = i_cont**2 * R            # 연속 운전 시 권선 발열 [W]
P_diss_peak = i_peak**2 * R            # 피크 시 발열 [W]

print(f"Kt={Kt} Nm/A, R={R} ohm, V={V} V, i_peak={i_peak} A, i_cont={i_cont} A")
print(f"무부하 속도 상한 w = V/Ke = {w_stall:.1f} rad/s "
      f"({w_stall*60/2/np.pi:.0f} rpm)")
print(f"전압선 스톨 토크 Kt/R*V = {tau_stall_volt:.2f} Nm")
print(f"피크 토크 천장 Kt*i_peak = {tau_peak:.2f} Nm  (발열 i^2R = {P_diss_peak:.0f} W)")
print(f"연속 토크 천장 Kt*i_cont = {tau_cont:.2f} Nm  (발열 i^2R = {P_diss_cont:.0f} W)")

# 전압선이 tau_cont 를 만나는 속도 = 연속 정격을 유지할 수 있는 최고 속도
# tau_cont = Kt/R*(V - Ke*w)  ->  w = (V - tau_cont*R/Kt)/Ke
w_knee_cont = (V - tau_cont * R / Kt) / Ke
w_knee_peak = (V - tau_peak * R / Kt) / Ke
print(f"연속 토크를 유지 가능한 최고 속도(전압선 교점) = {w_knee_cont:.1f} rad/s")
print(f"피크 토크를 유지 가능한 최고 속도(전압선 교점) = {w_knee_peak:.1f} rad/s")

# 이 모터에 n=100 하모닉을 붙이면 관절 연속 정격 (수식1/15강과 연결)
n_h = 100
print(f"\nn={n_h} 하모닉 부착 시 관절 연속 토크 = {tau_cont*n_h:.1f} Nm, "
      f"관절 최고속도 = {w_stall/n_h:.2f} rad/s ({w_stall/n_h*60/2/np.pi:.1f} rpm)")


# ============================================================
# fig 1 — 감속기 유형별 반사 관성 (로그 막대)
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 4.8))
names = [g[0] for g in GEARS]
jref_vals = [jref_tbl[n] for n in names]
colors = ['#2a9d8f', '#e9c46a', '#e76f51', '#8d99ae']
bars = ax.bar(names, jref_vals, color=colors, edgecolor='black', lw=0.8)
ax.axhline(J_L, color='gray', ls='--', lw=1.4)
ax.text(3.42, J_L*1.08, f'링크+페이로드 관성 $J_L$={J_L}', ha='right',
        fontsize=9, color='dimgray')
ax.set_yscale('log')
ax.set_ylabel('출력축에서 본 반사 관성 $J_{ref}=n^2 J_m$  (kg·m²)')
ax.set_title('감속기 유형별 반사 관성 — "정책이 미는 것은 로터까지"')
for b, v, (nm, n, jm) in zip(bars, jref_vals, GEARS):
    share = v/(J_L+v)*100
    ax.text(b.get_x()+b.get_width()/2, v*1.15,
            f'{v:.4f}\n(n={n}, 로터몫 {share:.0f}%)',
            ha='center', va='bottom', fontsize=8.3)
ax.set_ylim(3e-3, 3.0)
ax.margins(x=0.06)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_reflected_inertia.png", dpi=130)
plt.close()


# ============================================================
# fig 2 — 시각 토큰 예산 히트맵 (카메라 수 x 해상도)
# ============================================================
fig, ax = plt.subplots(figsize=(7.6, 5.0))
im = ax.imshow(budget, cmap='YlOrRd', aspect='auto',
               norm=matplotlib.colors.LogNorm(vmin=budget.min(),
                                               vmax=budget.max()))
ax.set_xticks(range(len(NCAM))); ax.set_xticklabels([f'{c}캠' for c in NCAM])
ax.set_yticks(range(len(RES))); ax.set_yticklabels([f'{r}px' for r in RES])
ax.set_xlabel('카메라 수'); ax.set_ylabel('입력 해상도 (정사각 변)')
ax.set_title('시각 토큰 예산  tokens = (res/14)² × 카메라수\n숫자↑ = prefill 지연↑ (attention ∝ tokens²)')
for i in range(len(RES)):
    for j in range(len(NCAM)):
        v = budget[i, j]
        ax.text(j, i, f'{v}', ha='center', va='center',
                color='black' if v < 1500 else 'white', fontsize=10, fontweight='bold')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('시각 토큰 수 (로그)')
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_token_budget.png", dpi=130)
plt.close()


# ============================================================
# fig 3 — 토크-속도 곡선 (전압 한계 + 열 한계 i^2 R)
# ============================================================
fig, ax = plt.subplots(figsize=(8.4, 5.0))
w = np.linspace(0, w_stall, 400)
tau_volt = Kt / R * (V - Ke * w)          # 전압 한계선(우하향 직선)
tau_volt_clip = np.clip(tau_volt, 0, None)

# 전압선 (모터가 낼 수 있는 순간 토크의 이론 상한)
ax.plot(w, tau_volt_clip, color='#264653', lw=2.4, label='전압 한계선  $\\tau=\\frac{K_t}{R}(V-K_e\\omega)$')
# 피크 전류 천장 (수평)
ax.axhline(tau_peak, color='#e76f51', lw=2.0, ls='-',
           label=f'피크 전류 천장 $K_t i_{{peak}}$ = {tau_peak:.1f} Nm')
# 열(연속) 한계 (수평)
ax.axhline(tau_cont, color='#e9c46a', lw=2.2, ls='--',
           label=f'열 한계(연속) $K_t i_{{cont}}$ = {tau_cont:.1f} Nm  (i²R={P_diss_cont:.0f}W)')

# 연속 작동 영역 = 전압선 아래 AND 열 한계 아래
env_cont = np.minimum(tau_volt_clip, tau_cont)
ax.fill_between(w, 0, env_cont, color='#2a9d8f', alpha=0.25,
                label='연속 안전 작동 영역')
# 피크(단시간) 영역 = 열 한계 위 ~ min(전압선, 피크천장)
env_peak = np.minimum(tau_volt_clip, tau_peak)
ax.fill_between(w, tau_cont, env_peak, where=(env_peak > tau_cont),
                color='#e76f51', alpha=0.18, label='단시간 피크 영역(발열 누적)')

ax.plot([w_knee_cont], [tau_cont], 'o', color='black', ms=6)
ax.annotate(f'연속 정격 무릎\n({w_knee_cont:.0f} rad/s, {tau_cont:.1f} Nm)',
            (w_knee_cont, tau_cont), xytext=(w_knee_cont*0.62, tau_cont*1.9),
            fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.set_xlabel('모터축 각속도 $\\omega$ (rad/s)')
ax.set_ylabel('모터축 토크 $\\tau$ (Nm)')
ax.set_title('토크-속도 곡선 — 전압이 속도를, i²R 발열이 지속 토크를 가둔다')
ax.set_xlim(0, w_stall); ax.set_ylim(0, tau_stall_volt*1.05)
ax.legend(fontsize=8.2, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_torque_speed.png", dpi=130)
plt.close()

banner("생성 완료")
for f in ["fig1_reflected_inertia.png", "fig2_token_budget.png", "fig3_torque_speed.png"]:
    print(f"  {OUT}/{f}")
