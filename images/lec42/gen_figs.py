# Lec 42 그림 생성 스크립트 (RT-1/RT-2/OXE — 행동 이산화·양자화 오차·vocabulary 점유)
# 실행: python3 gen_figs.py  (numpy/matplotlib 필요, CPU만)
# 실제 모델 다운로드/GPU 없음 — RT-1/RT-2식 256빈 이산화를 numpy로 재현하는 토이.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

rng = np.random.default_rng(42)

# ============================================================
# 0. 균일 이산화 / 역이산화 (RT-1/RT-2식: 차원별 [lo,hi]를 N빈 균일 분할)
# ============================================================
def discretize(x, lo, hi, N):
    """연속값 x를 [lo,hi] 범위 N빈 중 정수 토큰(0..N-1)으로. 빈 중앙으로 역양자화."""
    xc = np.clip(x, lo, hi)
    # 빈 폭 Δ = (hi-lo)/N. 토큰 = floor((x-lo)/Δ), 마지막 경계 보정.
    idx = np.floor((xc - lo) / (hi - lo) * N).astype(int)
    idx = np.clip(idx, 0, N - 1)
    return idx

def dequantize(idx, lo, hi, N):
    """토큰 → 빈 중앙값 (mid-rise 복원)."""
    delta = (hi - lo) / N
    return lo + (idx + 0.5) * delta

# ============================================================
# 1. 합성 7차원 ΔEEF 궤적 (RT-2 행동공간: Δx,Δy,Δz,Δroll,Δpitch,Δyaw,grip)
# ============================================================
Tsteps = 400
tg = np.linspace(0, 2 * np.pi, Tsteps)
# 각 차원: 서로 다른 주파수·진폭의 매끄러운 델타 + 소량 잡음. 그리퍼는 계단.
traj = np.zeros((Tsteps, 7))
traj[:, 0] = 0.030 * np.sin(tg)                    # Δx  [m]
traj[:, 1] = 0.020 * np.sin(2 * tg + 0.5)          # Δy  [m]
traj[:, 2] = 0.015 * np.cos(tg) * np.exp(-tg / 8)  # Δz  [m]
traj[:, 3] = 0.10 * np.sin(1.5 * tg)               # Δroll  [rad]
traj[:, 4] = 0.08 * np.cos(tg + 1.0)               # Δpitch [rad]
traj[:, 5] = 0.12 * np.sin(0.7 * tg)               # Δyaw   [rad]
traj[:, 6] = (np.sin(3 * tg) > 0).astype(float)    # grip 0/1
traj[:, :6] += 0.002 * rng.standard_normal((Tsteps, 6))

# 차원별 양자화 범위 = RT-2식 분위수 클리핑의 단순판 (여기선 min/max 여유 5%)
lo = traj.min(0) - 0.05 * (traj.max(0) - traj.min(0)) - 1e-9
hi = traj.max(0) + 0.05 * (traj.max(0) - traj.min(0)) + 1e-9
DIMNAMES = ['Δx', 'Δy', 'Δz', 'Δroll', 'Δpitch', 'Δyaw', 'grip']

# ---- 256빈으로 왕복 후 per-dim RMS 오차 vs 이론 Δ/√12 ----
N = 256
per_dim_rms = np.zeros(7)
per_dim_theory = np.zeros(7)
for d in range(7):
    idx = discretize(traj[:, d], lo[d], hi[d], N)
    rec = dequantize(idx, lo[d], hi[d], N)
    per_dim_rms[d] = np.sqrt(np.mean((rec - traj[:, d]) ** 2))
    delta = (hi[d] - lo[d]) / N
    per_dim_theory[d] = delta / np.sqrt(12)

print("=== [본문 WE] 256빈 per-dim 양자화 RMS vs 이론 Δ/√12 ===")
for d in range(7):
    tag = ' (계단신호 — 이론 부적용)' if DIMNAMES[d] == 'grip' else ''
    print(f"  {DIMNAMES[d]:7s}: 실측 RMS={per_dim_rms[d]:.3e}  "
          f"이론 Δ/√12={per_dim_theory[d]:.3e}  비율={per_dim_rms[d]/per_dim_theory[d]:.3f}{tag}")
# 연속 6차원만: 균일 양자화 이론이 적용되는 대상. grip(0/1 계단)은 제외.
ratio_mean = np.mean(per_dim_rms[:6] / per_dim_theory[:6])
print(f"  연속 6차원 평균 실측/이론 비율 = {ratio_mean:.4f} (grip 제외)")

# 6개 연속 차원 스택 RMS(전체 요소)
cont = traj[:, :6]
idx6 = np.stack([discretize(cont[:, d], lo[d], hi[d], N) for d in range(6)], 1)
rec6 = np.stack([dequantize(idx6[:, d], lo[d], hi[d], N) for d in range(6)], 1)
overall_rms = np.sqrt(np.mean((rec6 - cont) ** 2))
print(f"  6개 연속차원 전체 RMS = {overall_rms:.3e}")

# ============================================================
# 그림 1: 이산화 계단 재구성 — 대표 차원(Δx) 원본 vs 256빈 복원 + 오차
# ============================================================
d0 = 0  # Δx
idx0 = discretize(traj[:, d0], lo[d0], hi[d0], N)
rec0 = dequantize(idx0, lo[d0], hi[d0], N)
delta0 = (hi[d0] - lo[d0]) / N

# 시각적으로 계단이 보이도록 앞부분 60스텝 + 낮은 빈수(N=16)도 함께
Nvis = 16
idx0v = discretize(traj[:, d0], lo[d0], hi[d0], Nvis)
rec0v = dequantize(idx0v, lo[d0], hi[d0], Nvis)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.5))
seg = slice(0, 120)
axA.plot(np.arange(Tsteps)[seg], traj[seg, d0] * 1000, 'k-', lw=2, label='원본 연속 Δx')
axA.step(np.arange(Tsteps)[seg], rec0v[seg] * 1000, 'C1', lw=1.5, where='mid',
         label=f'{Nvis}빈 복원 (Δ={ (hi[d0]-lo[d0])/Nvis*1000:.2f} mm)')
axA.step(np.arange(Tsteps)[seg], rec0[seg] * 1000, 'C0', lw=1.5, where='mid',
         label=f'256빈 복원 (Δ={delta0*1000:.3f} mm)')
# 빈 경계 눈금 (16빈)
for b in range(Nvis + 1):
    axA.axhline((lo[d0] + b * (hi[d0]-lo[d0])/Nvis) * 1000, color='gray', lw=0.4, alpha=0.3)
axA.set_xlabel('타임스텝'); axA.set_ylabel('Δx [mm]')
axA.set_title('(a) 이산화 계단 재구성 — 빈이 성길수록 계단이 커진다')
axA.legend(fontsize=8.5, loc='upper right'); axA.grid(alpha=0.25)

# 오차 히스토그램: 256빈일 때 (rec-orig)이 [-Δ/2, Δ/2] 균등에 가까움
err0 = rec0 - traj[:, d0]
axB.hist(err0 / delta0, bins=30, density=True, color='C0', alpha=0.75,
         label='256빈 오차 분포 (Δ 단위)')
axB.axhline(1.0, color='k', ls='--', lw=1.5, label='이론: [-Δ/2,Δ/2] 균등분포 (밀도=1/Δ)')
axB.axvline(-0.5, color='gray', ls=':', lw=1); axB.axvline(0.5, color='gray', ls=':', lw=1)
axB.set_xlabel('양자화 오차 / Δ'); axB.set_ylabel('확률밀도 (Δ 단위)')
axB.set_title(f'(b) 오차는 균등분포 → RMS=Δ/√12\n실측/이론 비율 평균 {ratio_mean:.3f}')
axB.legend(fontsize=8.5); axB.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig1_discretization_staircase.png'), dpi=140)
plt.close(fig)

# ============================================================
# 그림 2: 빈수 2^k (k=4..10) 스윕 — 실측 오차곡선 vs 이론선 Δ/√12
# ============================================================
ks = np.arange(4, 11)      # 16 .. 1024
Ns = 2 ** ks
sweep_rms = np.zeros(len(Ns))     # 6개 연속차원 전체 RMS(정규화)
sweep_theory = np.zeros(len(Ns))
# 정규화: 각 차원을 자기 범위로 나눈 뒤 합쳐 단위 통일 (dimensionless)
rng_span = (hi[:6] - lo[:6])
for j, Nk in enumerate(Ns):
    errs = []
    ths = []
    for d in range(6):
        idx = discretize(cont[:, d], lo[d], hi[d], Nk)
        rec = dequantize(idx, lo[d], hi[d], Nk)
        errs.append(((rec - cont[:, d]) / rng_span[d]))       # 범위 정규화
        ths.append(((hi[d]-lo[d])/Nk) / np.sqrt(12) / rng_span[d])
    sweep_rms[j] = np.sqrt(np.mean(np.concatenate(errs) ** 2))
    sweep_theory[j] = np.mean(ths)   # 모든 차원 동일 = 1/(N√12)

print("\n=== [그림2] 빈수 스윕 (정규화 RMS) ===")
for j, Nk in enumerate(Ns):
    print(f"  N={Nk:5d} (2^{ks[j]}): 실측={sweep_rms[j]:.3e}  이론 1/(N√12)={sweep_theory[j]:.3e}")

fig, ax = plt.subplots(figsize=(7.8, 5.0))
ax.loglog(Ns, sweep_rms, 'C0o-', lw=2, ms=7, label='실측 RMS (범위 정규화)')
ax.loglog(Ns, sweep_theory, 'k--', lw=2, label=r'이론선 $\Delta/\sqrt{12}=1/(N\sqrt{12})$')
ax.axvline(256, color='C3', ls=':', lw=1.8)
i256 = np.argmin(np.abs(Ns - 256))
ax.annotate(f'RT-1/RT-2: N=256\n정규화 RMS≈{sweep_rms[i256]:.2e}\n(≈범위의 {sweep_rms[i256]*100:.3f}%)',
            xy=(256, sweep_rms[i256]), xytext=(300, sweep_rms[i256]*6),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='C3'))
ax.set_xlabel('빈 수 $N=2^k$'); ax.set_ylabel('양자화 RMS (범위 정규화, log)')
ax.set_title('빈수 vs 양자화 오차 — 실측이 이론선 $1/(N\\sqrt{12})$을 정확히 따른다\n'
             '빈 2배 → 오차 1/2 (기울기 −1)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig2_bins_vs_error.png'), dpi=140)
plt.close(fig)

# ============================================================
# 그림 3: 256 행동토큰의 vocabulary 점유 — "최저빈도 256토큰 덮어쓰기"
# ============================================================
# RT-2식: 기존 VLM vocabulary(예: PaLI/PaLM 계열)의 일부 토큰을 행동 빈 256개로 재지정.
vocabs = {'PaLI-X류 (~256k)': 256000,
          'PaLM류 (~256k)': 256000,
          'Llama류 (32k)': 32000,
          'SentencePiece 32k': 32000,
          '작은 토크나이저 (8k)': 8000}
action_tokens = 256

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# (a) 점유율 막대
names = list(vocabs.keys())
occ = [action_tokens / v * 100 for v in vocabs.values()]
ycol = ['C0', 'C0', 'C1', 'C1', 'C3']
bars = axA.barh(names, occ, color=ycol, alpha=0.85)
for b, o, v in zip(bars, occ, vocabs.values()):
    axA.text(o + max(occ)*0.01, b.get_y()+b.get_height()/2,
             f'{o:.3f}%  ({action_tokens}/{v//1000}k)', va='center', fontsize=9)
axA.set_xlabel('행동 토큰이 차지하는 vocabulary 비율 [%]')
axA.set_title('(a) 256 행동토큰의 vocabulary 점유율\n큰 어휘일수록 "언어 능력 손실"이 작다')
axA.grid(alpha=0.25, axis='x')

# (b) 개념도: 256k 어휘 중 최저빈도 256개를 행동으로 덮어쓰기
V = 256000
axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis('off')
# 큰 사각형 = 전체 어휘. 오른쪽 끝에 실제 비율(0.1%)의 얇은 띠 = 256 행동토큰.
x0, w0 = 0.05, 0.90
axB.add_patch(plt.Rectangle((x0, 0.42), w0, 0.34, facecolor='C0', alpha=0.30, edgecolor='k'))
frac = action_tokens / V                          # = 0.001 (실제 비율)
band = w0 * frac                                  # 실제 폭(거의 선처럼 얇다)
axB.add_patch(plt.Rectangle((x0 + w0 - band, 0.42), max(band, 0.004), 0.34,
                            facecolor='C3', alpha=1.0, edgecolor='C3'))
axB.text(0.5, 0.81, '전체 vocabulary  V ≈ 256,000 토큰', ha='center', fontsize=11)
axB.text(0.5, 0.59, '자연어 토큰 (그대로 보존)', ha='center', fontsize=10, color='#0c3a63')
# 실제 비율의 띠는 거의 선이라 확대 콜아웃으로 설명
axB.annotate('실제 비율의 띠 (선처럼 얇음)',
             xy=(x0 + w0, 0.59), xytext=(0.40, 0.30), fontsize=9, color='C3',
             ha='center', arrowprops=dict(arrowstyle='->', color='C3'))
axB.text(0.5, 0.16, '최저빈도 256개 토큰을 행동 빈으로 재지정\n'
                    '→ 어휘의 0.1%만 소모, 언어 능력은 거의 보존',
         ha='center', fontsize=9.5, color='C3')
axB.text(0.5, 0.93, '(b) "행동은 또 하나의 언어" — 어휘 재지정 개념도 (실제 비율)',
         ha='center', fontsize=11, weight='bold')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig3_vocab_occupancy.png'), dpi=140)
plt.close(fig)

print("\n=== [그림3] vocabulary 점유율 ===")
for n, v in vocabs.items():
    print(f"  {n:22s}: {action_tokens}/{v} = {action_tokens/v*100:.4f}%")

# ============================================================
# 그림 4: "왜 256인가" — 정밀도(E1)와 학습 난이도(E2)의 상충
# ============================================================
Ns_t = 2 ** np.arange(4, 11)          # 16..1024
# 정밀도: 범위 정규화 분해능 = 1/(N√12) (작을수록 좋다) → "정밀도 점수" = -log
prec = 1.0 / (Ns_t * np.sqrt(12))     # 정규화 양자화 RMS
# 학습/예산 비용: 한 스텝 균등 CE 상한(연속6+grip) ∝ 6 ln N + ln2, vocab 점유 ∝ N
ce_cost = 6 * np.log(Ns_t) + np.log(2)          # nats
occ_cost = Ns_t / 256000.0 * 100                # 256k 어휘 점유율 %

fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
l1 = ax1.loglog(Ns_t, prec, 'C0o-', lw=2, ms=6,
                label='양자화 RMS (정규화) — 작을수록 정밀')
ax1.set_xlabel('빈 수 $N=2^k$'); ax1.set_ylabel('양자화 RMS (정규화, log)', color='C0')
ax1.tick_params(axis='y', labelcolor='C0'); ax1.grid(alpha=0.3, which='both')
ax2 = ax1.twinx()
l2 = ax2.plot(Ns_t, ce_cost, 'C3s--', lw=2, ms=6,
              label='균등 CE 상한 $6\\ln N+\\ln2$ [nats] — 학습 난이도')
ax2.set_ylabel('한 스텝 균등 cross-entropy 상한 [nats]', color='C3')
ax2.tick_params(axis='y', labelcolor='C3')
ax1.axvline(256, color='gray', ls=':', lw=1.8)
ax1.text(256*1.1, prec[0]*0.6, 'RT-1/RT-2\nN=256', color='gray', fontsize=9)
lines = l1 + l2
ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=9, loc='center left')
ax1.set_title('왜 256인가 — 정밀도(↑ N)와 학습 난이도·어휘예산(↓ N)의 상충\n'
              '두 곡선이 반대로 움직인다 → 256은 타협점')
fig.tight_layout()
fig.savefig(__file__.replace('gen_figs.py', 'fig4_precision_vs_difficulty.png'), dpi=140)
plt.close(fig)
print("\n=== [그림4] 정밀도 vs 학습난이도 상충 ===")
for j, Nk in enumerate(Ns_t):
    print(f"  N={Nk:5d}: 정규화RMS={prec[j]:.3e}  균등CE상한={ce_cost[j]:.2f} nats  "
          f"256k점유={occ_cost[j]:.3f}%")

# ============================================================
# [핵심 수식 2 검증] autoregressive 토큰 cross-entropy 토이
# 7토큰 행동열의 CE = -Σ log p(정답빈). 완벽 예측이면 0, 균등 예측이면 log N.
# ============================================================
print("\n=== [수식2] AR 행동토큰 cross-entropy 토이 ===")
gt = idx6[0]  # 첫 스텝의 6개 정답 빈 + 그리퍼(0/1)
gt_full = np.concatenate([gt, [int(traj[0, 6])]])
# 균등분포 예측의 CE (연속6은 N=256, 그리퍼는 2빈)
ce_uniform = 6 * np.log(256) + 1 * np.log(2)
# "이웃빈에 폭넓게 퍼진" 예측 (연속6은 정답빈 확률 0.6, 그리퍼는 0.95)
p_correct = 0.6
ce_confident = -(6 * np.log(p_correct) + 1 * np.log(0.95))
print(f"  균등예측 CE(1스텝,7토큰) = {ce_uniform:.4f} nats "
      f"(= 6·ln256 + ln2 = {6*np.log(256):.4f}+{np.log(2):.4f})")
print(f"  자신있는 예측 CE        = {ce_confident:.4f} nats")
# 연속6차원의 토큰당 perplexity: 균등이면 정확히 빈 수 256.
print(f"  연속차원 토큰당 perplexity(균등) = exp(ln256) = {np.exp(np.log(256)):.1f} "
      f"(= 빈 수 N, '256개 중 무작위' 감각)")
print(f"  자신있는 예측의 연속차원 토큰당 perplexity = exp(-ln0.6) = {np.exp(-np.log(0.6)):.2f}")

# ============================================================
# [핵심 수식 3 검증] cross-embodiment 향상 조건 토이 (RT-1-X ~50%)
# 소규모 도메인: 자기데이터 n_self 작음 → 혼합데이터가 편향 감수하고 분산 크게 줄임.
# 간단 bias-variance 모형으로 "이득 조건"을 수치로.
# ============================================================
print("\n=== [수식3] cross-embodiment 이득 토이 (bias-variance) ===")
# 전문가(자기데이터만): 오차 ~ 분산 c_v / n_self  (bias=0)
# 혼합(cross-embodiment): 오차 ~ 분산 c_v / (n_self + n_other) + bias^2 (도메인갭)
c_v = 1.0
bias2 = 0.0125    # 다른 로봇 데이터의 도메인 갭이 남기는 편향^2
n_other = 80.0    # 다른 로봇 데이터 규모(정규화)
for n_self in [5.0, 20.0, 100.0, 500.0]:
    err_spec = c_v / n_self
    err_mix = c_v / (n_self + n_other) + bias2
    improve = (err_spec - err_mix) / err_spec * 100
    tag = '이득' if improve > 0 else '손해'
    print(f"  n_self={n_self:6.0f}: 전문가={err_spec:.4f} 혼합={err_mix:.4f} "
          f"→ {improve:+.1f}% ({tag})")
# 이득 조건: c_v/n_self - c_v/(n_self+n_other) > bias2  (분산감소 > 편향증가)
# 소규모(n_self=20)에서 ~50%대 이득이 나오도록 파라미터 선택됨.
n_demo = 20.0
imp_demo = (c_v/n_demo - (c_v/(n_demo+n_other)+bias2)) / (c_v/n_demo) * 100
print(f"  → 소규모 데이터 도메인(n_self={n_demo:.0f})에서 이득 ≈ {imp_demo:.1f}% "
      f"(RT-1-X의 ~50%와 정성적으로 일치)")
# 손익분기 n_self* : 분산감소량 = bias2 인 지점 (수치 탐색)
grid = np.linspace(1, 400, 40000)
gain = (c_v/grid) - (c_v/(grid+n_other) + bias2)
n_star = grid[np.argmin(np.abs(gain))]
print(f"  손익분기 n_self* ≈ {n_star:.0f} — 이보다 자기데이터가 많으면 혼합이 손해로 전환")

print("\nfigures written: fig1_discretization_staircase.png, "
      "fig2_bins_vs_error.png, fig3_vocab_occupancy.png, "
      "fig4_precision_vs_difficulty.png")
