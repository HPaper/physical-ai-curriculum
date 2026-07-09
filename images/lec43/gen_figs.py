# Lec 43 그림 생성 스크립트 — Octo / OpenVLA / OpenVLA-OFT
# 실행: python3 gen_figs.py   (numpy / matplotlib 만 사용, CPU)
# 개념 재현 목적의 토이. 실제 모델 다운로드/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

RNG = np.random.default_rng(43)
OUT = __file__.replace('gen_figs.py', '')

# =====================================================================
# 실험 1 — 이산화 빈: min-max vs q01-q99 (이상치가 빈을 잡아먹는 문제)
#   OpenVLA(RT-2 계승)는 차원당 256빈. 빈 경계를 어떻게 잡느냐로
#   "유효 분해능(effective resolution)"이 갈린다.
# =====================================================================
N_BINS = 256

# 행동 한 차원의 분포: 대부분 [-1, 1]에 몰려 있고, 드물게 큰 이상치(센서 튐,
# 리셋 점프, 텔레오퍼레이션 실수)가 섞인다.
n = 100_000
core = RNG.normal(0.0, 0.35, size=int(n * 0.98))          # 실제 조작이 사는 곳
core = np.clip(core, -1.0, 1.0)
outliers = RNG.uniform(-12.0, 12.0, size=int(n * 0.02))   # 2% 이상치
actions = np.concatenate([core, outliers])

lo_mm, hi_mm = actions.min(), actions.max()               # min-max 경계
lo_q, hi_q = np.quantile(actions, [0.01, 0.99])           # q01-q99 경계

def discretize(x, lo, hi, nb=N_BINS):
    xc = np.clip(x, lo, hi)
    idx = np.floor((xc - lo) / (hi - lo) * nb).astype(int)
    return np.clip(idx, 0, nb - 1)

bins_mm = discretize(actions, lo_mm, hi_mm)
bins_q = discretize(actions, lo_q, hi_q)

# 코어(실제 조작) 영역이 실제로 몇 개의 빈을 쓰는가 = 유효 분해능
core_mask = (actions >= -1.0) & (actions <= 1.0)
eff_mm = len(np.unique(bins_mm[core_mask]))
eff_q = len(np.unique(bins_q[core_mask]))
# 한 빈이 덮는 물리적 폭 [단위/빈]
width_mm = (hi_mm - lo_mm) / N_BINS
width_q = (hi_q - lo_q) / N_BINS

print("=== 실험 1: 빈 경계와 유효 분해능 ===")
print(f"min-max 경계 : [{lo_mm:.3f}, {hi_mm:.3f}], 빈 폭 {width_mm:.4f}/빈")
print(f"q01-q99 경계 : [{lo_q:.3f}, {hi_q:.3f}], 빈 폭 {width_q:.4f}/빈")
print(f"코어 영역[-1,1]이 쓰는 빈 수: min-max {eff_mm}/256, q01-q99 {eff_q}/256")
print(f"유효 분해능 비: {eff_q / eff_mm:.1f}배")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
edges_view = np.linspace(-2.5, 2.5, 120)
axA.hist(actions, bins=edges_view, color='C7', alpha=0.55, label='행동 표본(코어)')
for k in range(0, N_BINS + 1, 8):
    axA.axvline(lo_mm + k * width_mm, color='C3', lw=0.4, alpha=0.5)
axA.axvline(lo_mm, color='C3', lw=1.6, label=f'min-max 빈 경계(간격 {width_mm:.3f})')
axA.set_xlim(-2.5, 2.5)
axA.set_title(f'(a) min-max 빈: 이상치가 척도를 늘려\n코어가 {eff_mm}/256빈만 사용')
axA.set_xlabel('행동값 (한 차원)'); axA.set_ylabel('표본 수')
axA.legend(fontsize=8.5, loc='upper right')

axB.hist(actions, bins=edges_view, color='C7', alpha=0.55, label='행동 표본(코어)')
for k in range(0, N_BINS + 1, 8):
    axB.axvline(lo_q + k * width_q, color='C0', lw=0.4, alpha=0.5)
axB.axvline(lo_q, color='C0', lw=1.6, label=f'q01-q99 빈 경계(간격 {width_q:.3f})')
axB.axvline(hi_q, color='C0', lw=1.6)
axB.set_xlim(-2.5, 2.5)
axB.set_title(f'(b) q01-q99 빈: 이상치를 잘라내\n코어가 {eff_q}/256빈 사용 (~{eff_q/eff_mm:.0f}배 분해능)')
axB.set_xlabel('행동값 (한 차원)'); axB.set_ylabel('표본 수')
axB.legend(fontsize=8.5, loc='upper right')
fig.tight_layout()
fig.savefig(OUT + 'fig1_quantile_bins.png', dpi=140)
plt.close(fig)

# =====================================================================
# 실험 2 — L1 회귀의 수렴: 단봉 vs 다봉
#   같은 조건 z에서 여러 행동이 관측될 때(다봉), L1 회귀는 하나의 값만
#   낼 수 있다. L1 최소해 = 조건부 중앙값(median). 단봉이면 그 중앙값이
#   곧 정답, 다봉이면 중앙값은 어느 모드도 아닌 값에 앉는다.
# =====================================================================
def fit_constant_L1(samples, iters=4000, lr=0.05):
    """상수 예측값을 L1 손실로 경사하강 → 중앙값으로 수렴 (subgradient=sign)."""
    y = 0.0
    for _ in range(iters):
        grad = np.sign(y - samples).mean()   # d/dy |y - s| = sign(y - s)
        y -= lr * grad
    return y

def fit_constant_L2(samples):
    return samples.mean()                    # L2 최소해 = 평균

# 단봉: 조건 z에서 시연이 하나의 행동 주변에 몰림
uni = RNG.normal(0.40, 0.05, size=5000)
# 다봉: 같은 z에서 "왼쪽으로 우회" vs "오른쪽으로 우회" 두 시연 모드
left = RNG.normal(-0.6, 0.05, size=2500)
right = RNG.normal(0.6, 0.05, size=2500)
multi = np.concatenate([left, right])

y_uni = fit_constant_L1(uni)
y_multi = fit_constant_L1(multi)
med_uni, med_multi = np.median(uni), np.median(multi)

# 수렴 궤적(그림용)
def trace_L1(samples, iters=400, lr=0.05):
    y, hist = 0.0, []
    for _ in range(iters):
        hist.append(y)
        y -= lr * np.sign(y - samples).mean()
    return np.array(hist)

tr_uni = trace_L1(uni)
tr_multi = trace_L1(multi)

print("\n=== 실험 2: L1 회귀 수렴 ===")
print(f"단봉: L1 수렴값 {y_uni:.4f}  (중앙값 {med_uni:.4f}, 참 모드 0.40) -> 정확")
print(f"다봉: L1 수렴값 {y_multi:.4f} (중앙값 {med_multi:.4f}) -> 두 모드(-0.6/+0.6) 사이 중앙값")
print(f"다봉에서 어느 모드로부터의 거리: |{y_multi:.3f}-(±0.6)| = {abs(abs(y_multi)-0.6):.3f} 만큼 떨어짐")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 분포 + 수렴값
axA.hist(uni, bins=60, density=True, color='C0', alpha=0.5, label='단봉 시연')
axA.hist(multi, bins=60, density=True, color='C1', alpha=0.5, label='다봉 시연(2모드)')
axA.axvline(y_uni, color='C0', lw=2.2, ls='-', label=f'단봉 L1해 {y_uni:.2f} (=모드)')
axA.axvline(y_multi, color='C3', lw=2.2, ls='--', label=f'다봉 L1해 {y_multi:.2f} (모드 사이!)')
axA.axvline(-0.6, color='gray', lw=1, ls=':'); axA.axvline(0.6, color='gray', lw=1, ls=':')
axA.set_title('(a) L1 회귀 = 조건부 중앙값\n단봉이면 정확, 다봉이면 어느 모드도 아님')
axA.set_xlabel('행동값'); axA.set_ylabel('밀도'); axA.legend(fontsize=8)
# (b) 수렴 궤적
it = np.arange(len(tr_uni))
axB.plot(it, tr_uni, 'C0', lw=2, label='단봉 → 0.40 근방')
axB.plot(it, tr_multi, 'C3', lw=2, label='다봉 → ~0.0 (중앙값)')
axB.axhline(0.40, color='C0', ls=':', lw=1)
axB.axhline(0.0, color='C3', ls=':', lw=1)
axB.axhspan(-0.65, -0.55, color='C1', alpha=0.15)
axB.axhspan(0.55, 0.65, color='C1', alpha=0.15, label='다봉의 두 실제 모드')
axB.set_title('(b) L1 경사하강 수렴(subgradient=sign)')
axB.set_xlabel('반복'); axB.set_ylabel('예측 상수값'); axB.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + 'fig2_l1_convergence.png', dpi=140)
plt.close(fig)

# =====================================================================
# 실험 3 — 병렬(1-shot) 디코딩 vs 순차(AR) 디코딩 처리량
#   OFT의 핵심: 청크 길이 H, 차원 D의 행동을 순차 AR로 뽑으면 H*D번의
#   forward가 필요하다. 병렬이면 1번. 벽시계 시간으로 비 재현.
#   실제 트랜스포머 대신 "한 번의 forward = 고정 지연"을 numpy 행렬곱으로
#   대리(proxy)한다 — 순차 루프의 오버헤드까지 포함해 측정.
# =====================================================================
D = 7                       # 행동 차원 (x,y,z,roll,pitch,yaw,grip)
# 실제 VLA 디코딩에서 한 번의 forward는 '거의 고정 지연'이다: 수십억 파라미터를
# 메모리(HBM)에서 읽어 여러 층을 깊이 방향으로 통과하는 비용이 지배적이라,
# 토큰을 1개 넣든 200개 넣든 한 번의 forward 벽시계 시간은 대체로 일정하다
# (배치가 가중치 로드를 분할상환하는, memory-bandwidth-bound 영역).
# → 핵심 비용은 '몇 번 forward 하는가'이지 '몇 토큰인가'가 아니다.
# 이 고정 지연을, 토큰 수와 무관한 고정 크기 행렬곱으로 대리한다.
d_model, d_ff = 256, 1024
# 한 forward가 '읽어야 하는 큰 가중치'를 흉내: 고정 비용 커널
Wa = (RNG.standard_normal((d_model, d_ff)) / np.sqrt(d_model)).astype(np.float32)
Wb = (RNG.standard_normal((d_ff, d_model)) / np.sqrt(d_ff)).astype(np.float32)
_FIX = (RNG.standard_normal((64, d_model)) * 0.1).astype(np.float32)  # 고정 배치

def one_forward(tokens):
    """한 번의 forward = 큰 가중치를 통과시키는 고정 비용 커널.
    비용이 토큰 수에 (거의) 무관 → 실제 weight-bound 디코드의 대리."""
    h = np.tanh(_FIX @ Wa)           # 토큰 수와 무관한 고정 작업(가중치 로드/적용)
    _ = np.tanh(h @ Wb)
    return tokens                    # 실제로는 다음 토큰을 반환

def decode_autoregressive(H):
    """청크 길이 H·차원 D를 토큰 하나씩 H*D번 순차 forward.
    다음 토큰이 이전 토큰에 의존 → 병렬 불가, forward를 H*D회 호출."""
    ctx = (RNG.standard_normal((1, d_model)) * 0.1).astype(np.float32)
    for _ in range(H * D):
        ctx = one_forward(ctx)       # 순차 의존: forward 한 번당 토큰 하나
    return ctx

def decode_parallel(H):
    """H*D 토큰을 한 번의 forward로 동시 생성 (forward 1회 호출)."""
    batch = (RNG.standard_normal((H * D, d_model)) * 0.1).astype(np.float32)
    return one_forward(batch)        # 1회 호출로 전부

def timed(fn, H, reps=15):
    fn(H)                            # warmup
    best = np.inf                    # 최소 시간(스케줄링 외란에 강건)
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(H)
        best = min(best, time.perf_counter() - t0)
    return best

chunks = [1, 2, 4, 8, 16, 32]
t_ar, t_par, speedup = [], [], []
for H in chunks:
    ta = timed(decode_autoregressive, H)
    tp = timed(decode_parallel, H)
    t_ar.append(ta * 1e3); t_par.append(tp * 1e3); speedup.append(ta / tp)

print("\n=== 실험 3: 순차(AR) vs 병렬 디코딩 (D=7) ===")
for H, ta, tp, s in zip(chunks, t_ar, t_par, speedup):
    print(f"H={H:2d} ({H*D:3d}토큰): AR {ta:7.2f} ms | 병렬 {tp:6.3f} ms | {s:6.1f}배")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
x = np.arange(len(chunks))
axA.plot(x, t_ar, 'C3o-', lw=2, ms=6, label='순차 AR (H·D회 forward)')
axA.plot(x, t_par, 'C0s-', lw=2, ms=6, label='병렬 1-shot (1회 forward)')
axA.set_yscale('log')
axA.set_xticks(x); axA.set_xticklabels([f'{H}\n({H*D}토큰)' for H in chunks])
axA.set_xlabel('청크 길이 H (괄호=총 토큰 H·D, D=7)')
axA.set_ylabel('디코딩 벽시계 시간 [ms] (log)')
axA.set_title('(a) 순차 vs 병렬 디코딩 시간')
axA.grid(alpha=0.3, which='both'); axA.legend(fontsize=9)

axB.bar(x, speedup, color='C2', alpha=0.8)
for xi, s in zip(x, speedup):
    axB.text(xi, s * 1.02, f'{s:.0f}×', ha='center', fontsize=9)
axB.set_xticks(x); axB.set_xticklabels([str(H) for H in chunks])
axB.set_xlabel('청크 길이 H')
axB.set_ylabel('병렬/순차 속도 향상 (배)')
axB.set_title(f'(b) 처리량 향상 — H가 길수록 벌어진다\n(H=32에서 {speedup[-1]:.0f}배)')
axB.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig3_decoding_throughput.png', dpi=140)
plt.close(fig)

# =====================================================================
# 실험 4 — 이산화의 왕복(round-trip) 재구성 오차
#   연속 행동을 빈으로 이산화한 뒤 빈 중심으로 되돌리면 양자화 오차가 남는다.
#   그 오차의 상한은 빈 폭의 절반 Δ/2. 경계 설계가 곧 도달 가능한 정밀도다.
#   여기서는 코어 영역의 행동만 얼마나 정확히 복원되는지 min-max vs 분위수 비교.
# =====================================================================
def dequantize(idx, lo, hi, nb=N_BINS):
    return lo + (idx + 0.5) * (hi - lo) / nb    # 빈 중심으로 복원

core_vals = actions[core_mask]                   # 코어 영역 실제 행동
rec_mm = dequantize(discretize(core_vals, lo_mm, hi_mm), lo_mm, hi_mm)
rec_q = dequantize(discretize(core_vals, lo_q, hi_q), lo_q, hi_q)
err_mm = np.abs(rec_mm - core_vals)
err_q = np.abs(rec_q - core_vals)
mae_mm, mae_q = err_mm.mean(), err_q.mean()

print("\n=== 실험 4: 왕복 재구성 오차 (코어 영역) ===")
print(f"min-max : 평균 절대오차 {mae_mm:.4f}  (상한 Δ/2 = {width_mm/2:.4f})")
print(f"q01-q99 : 평균 절대오차 {mae_q:.4f}  (상한 Δ/2 = {width_q/2:.4f})")
print(f"재구성 오차 감소: {mae_mm / mae_q:.1f}배")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 원값 대 복원값 (샘플)
sel = RNG.choice(len(core_vals), 400, replace=False)
axA.plot([-1, 1], [-1, 1], 'k--', lw=1, alpha=0.6, label='이상적 복원 (y=x)')
axA.plot(core_vals[sel], rec_mm[sel], 'C3.', ms=4, alpha=0.5,
         label=f'min-max (MAE {mae_mm:.3f})')
axA.plot(core_vals[sel], rec_q[sel], 'C0.', ms=4, alpha=0.5,
         label=f'q01-q99 (MAE {mae_q:.4f})')
axA.set_xlim(-1.05, 1.05); axA.set_ylim(-1.05, 1.05); axA.set_aspect('equal')
axA.set_xlabel('원래 행동값'); axA.set_ylabel('이산화→복원 값')
axA.set_title('(a) 왕복 재구성: min-max는 계단처럼 뭉개진다')
axA.legend(fontsize=8.5, loc='upper left'); axA.grid(alpha=0.3)
# (b) 오차 히스토그램
axB.hist(err_mm, bins=50, color='C3', alpha=0.55, label=f'min-max (평균 {mae_mm:.3f})')
axB.hist(err_q, bins=50, color='C0', alpha=0.7, label=f'q01-q99 (평균 {mae_q:.4f})')
axB.axvline(width_mm/2, color='C3', ls=':', lw=1.5, label=f'min-max 상한 Δ/2={width_mm/2:.3f}')
axB.axvline(width_q/2, color='C0', ls=':', lw=1.5, label=f'q01-q99 상한 Δ/2={width_q/2:.3f}')
axB.set_xlabel('절대 재구성 오차'); axB.set_ylabel('표본 수')
axB.set_title(f'(b) 오차 분포 — 분위수 빈이 ~{mae_mm/mae_q:.0f}배 정밀')
axB.legend(fontsize=8); axB.set_xlim(0, width_mm*0.7)
fig.tight_layout()
fig.savefig(OUT + 'fig4_reconstruction_error.png', dpi=140)
plt.close(fig)

print("\nfigures written: fig1_quantile_bins.png, fig2_l1_convergence.png, "
      "fig3_decoding_throughput.png, fig4_reconstruction_error.png")
