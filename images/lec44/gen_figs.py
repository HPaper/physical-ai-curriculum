# Lec 44 그림 생성 스크립트 — π0 / π0-FAST
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만)
# 개념을 numpy/scipy 토이로 재현한다 — 실제 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

rng = np.random.default_rng(0)
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 합성 dexterous 궤적: H=50 스텝, D=7차원 (양팔 손끝의 대역제한 운동 흉내)
#   대역제한 신호 = 저주파 성분의 합 + 소량 잡음. DCT가 희소해지는 성질을 재현.
# ============================================================================
H, D = 50, 7
t = np.arange(H)
def make_traj(seed=1):
    r = np.random.default_rng(seed)
    traj = np.zeros((H, D))
    for d in range(D):
        # 각 차원 = 2~3개의 낮은 주파수 사인 성분 (대역제한) + 작은 고주파 잡음
        n_modes = r.integers(2, 4)
        for _ in range(n_modes):
            f = r.uniform(0.3, 2.0)                # 낮은 주파수 (저주파 집중)
            amp = r.uniform(0.4, 1.0) / (f**1.3)   # 1/f^1.3 스펙트럼: 저주파에 에너지
            ph = r.uniform(0, 2*np.pi)
            traj[:, d] += amp*np.sin(2*np.pi*f*t/H + ph)
        traj[:, d] += 0.01*r.standard_normal(H)    # 소량 고주파 잡음 (매끄러운 dexterous 운동)
    return traj

traj = make_traj(1)

# ---------- 차원별 DCT (type-II, orthonormal) ----------
# scipy dct(x, norm='ortho')는 정규직교 → 파세발: ||x||^2 == ||X||^2
coefs = dct(traj, type=2, norm='ortho', axis=0)     # (H, D) 계수

# 파세발 검증 (에너지 보존)
energy_time = np.sum(traj**2)
energy_freq = np.sum(coefs**2)
parseval_err = abs(energy_time - energy_freq)

# 저주파 에너지 집중: 전체 계수 중 하위(저주파) k개가 담는 에너지 비율
def energy_fraction_lowfreq(k):
    # 차원 무관 전체 에너지에서 각 차원 상위 k 저주파 계수가 담는 비율
    e_total = np.sum(coefs**2)
    e_low = np.sum(coefs[:k, :]**2)
    return e_low / e_total

frac_top5 = energy_fraction_lowfreq(5)      # 첫 5개 저주파 계수의 에너지 비율
frac_top10 = energy_fraction_lowfreq(10)

# ============================================================================
# 그림 1: 궤적과 DCT 스펙트럼 (저주파 집중)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 시간영역 궤적 (3개 차원만 표시)
for d in range(3):
    ax1.plot(t, traj[:, d], lw=1.8, label=f'차원 {d+1}')
ax1.set_xlabel('타임스텝 (H=50)'); ax1.set_ylabel('관절/EEF 값')
ax1.set_title('(a) 합성 dexterous 궤적 (대역제한 신호)')
ax1.grid(alpha=0.3); ax1.legend(fontsize=9)

# (b) DCT 계수 크기 — 저주파(왼쪽)에 에너지 집중
mean_mag = np.mean(np.abs(coefs), axis=1)   # 차원 평균 계수 크기
ax2.bar(np.arange(H), mean_mag, color='C0', width=0.9)
ax2.axvspan(-0.5, 4.5, color='C3', alpha=0.15)
ax2.text(2, mean_mag.max()*0.85,
         f'저주파 5개\n= 전체 에너지 {frac_top5*100:.1f}%',
         fontsize=9, ha='center', color='C3')
ax2.set_xlabel('DCT 계수 인덱스 (저주파 → 고주파)')
ax2.set_ylabel('|계수| 평균 (차원 평균)')
ax2.set_title('(b) DCT 스펙트럼 — 에너지가 저주파에 몰린다\n(궤적은 대역제한 → 주파수영역에서 희소)')
ax2.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig1_dct_spectrum.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 2: 압축률 vs 재구성오차 — DCT(상위 k계수) vs naive(스텝별 양자화)
# ============================================================================
# 여러 궤적에 대해 평균을 내어 곡선을 안정화
N_TRAJ = 40
trajs = [make_traj(s+10) for s in range(N_TRAJ)]
coefs_list = [dct(x, type=2, norm='ortho', axis=0) for x in trajs]

def nrmse(x, xhat):
    # 정규화 RMSE: RMSE / (신호 표준편차) — 스케일 무관 재구성 오차
    rmse = np.sqrt(np.mean((x - xhat)**2))
    return rmse / np.std(x)

# --- DCT 방식: 차원별 상위 k 저주파 계수만 유지 (희소화) ---
# 유지 계수 수 k당 "저장해야 하는 스칼라 수" = k * D (naive와 공정 비교)
ks = np.arange(1, H+1)
dct_scalars = ks * D                         # DCT 방식 저장 스칼라 수
dct_err = []
for k in ks:
    errs = []
    for x, X in zip(trajs, coefs_list):
        Xk = X.copy(); Xk[k:, :] = 0.0        # 상위 k 저주파만
        xhat = idct(Xk, type=2, norm='ortho', axis=0)
        errs.append(nrmse(x, xhat))
    dct_err.append(np.mean(errs))
dct_err = np.array(dct_err)

# --- naive 방식: 모든 스텝×차원을 b비트로 균일 양자화 (스텝별 유지) ---
# 저장 스칼라 수는 항상 H*D (모든 스텝 유지) — 압축은 비트수로만.
def quantize_bits(x, b):
    lo, hi = x.min(), x.max()
    levels = 2**b - 1
    q = np.round((x - lo)/(hi - lo)*levels)
    return lo + q/levels*(hi - lo)

naive_bits = np.arange(1, 9)
naive_err = []
for b in naive_bits:
    errs = [nrmse(x, quantize_bits(x, b)) for x in trajs]
    naive_err.append(np.mean(errs))
naive_err = np.array(naive_err)
# naive의 "스칼라 수"는 항상 H*D; 압축은 오직 정밀도(비트) 축소로만 → 오차가 큼

# --- 토큰 수 비교: 같은 재구성 오차(예: NRMSE ~ 0.1)를 내는 데 필요한 표현 크기 ---
TARGET = 0.07
# DCT: 목표 오차 이하가 되는 최소 k
k_needed = ks[np.argmax(dct_err <= TARGET)] if np.any(dct_err <= TARGET) else H
dct_scalars_at_target = int(k_needed * D)
# naive: 스텝별 유지가 원칙 → H*D 스칼라 (오차는 비트로만 조절, 스텝 수는 안 줆)
naive_scalars = int(H * D)
token_ratio = naive_scalars / dct_scalars_at_target

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 압축률(저장 스칼라 수) vs 재구성 오차
ax1.semilogy(dct_scalars, dct_err, 'C0o-', lw=2, ms=4,
             label='DCT (상위 k 저주파 계수)')
# naive: 스텝을 못 줄인다 → 항상 x=350에 갇힘. 비트만 줄여도 왼쪽으로 못 감.
ax1.semilogy([naive_scalars]*len(naive_bits), naive_err, 'C3^', ms=7,
             label='naive 스텝별 양자화 (1~8비트)')
ax1.axhline(TARGET, color='gray', ls=':', lw=1.2, label=f'목표 오차 NRMSE={TARGET}')
ax1.axvline(naive_scalars, color='C3', ls='--', lw=1.5, alpha=0.6)
ax1.plot(dct_scalars_at_target, TARGET, 'C0*', ms=16, zorder=5)
ax1.annotate(f'DCT: {dct_scalars_at_target} 스칼라로\n같은 오차 달성',
             xy=(dct_scalars_at_target, TARGET), xytext=(70, 0.012),
             fontsize=9, color='C0', arrowprops=dict(arrowstyle='->', color='C0'))
ax1.annotate('naive는 스텝을\n못 줄인다 (x=350 고정)',
             xy=(naive_scalars, naive_err[3]), xytext=(175, 0.35),
             fontsize=8.5, ha='center', color='C3',
             arrowprops=dict(arrowstyle='->', color='C3'))
ax1.set_xlabel('저장 스칼라 수 (= 토큰 예산 대리 지표)')
ax1.set_ylabel('재구성 오차 NRMSE (log)')
ax1.set_ylim(3e-3, 1.0)   # 관심 구간(0.01~1)만 — k=H의 완전복원 꼬리는 잘라 가독성 확보
ax1.set_title(f'(a) 같은 오차, 표현 크기 ~{token_ratio:.0f}배 차이\n(DCT 희소화 vs naive 스텝별 유지)')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=8, loc='lower left')

# (b) DCT 복원 예시: 상위 k=5 계수만으로 복원 (10배 압축)
x0, X0 = trajs[0], coefs_list[0]
K_DEMO = 5
X5 = X0.copy(); X5[K_DEMO:, :] = 0.0
xr = idct(X5, type=2, norm='ortho', axis=0)
for d in range(2):
    ax2.plot(t, x0[:, d], 'C0' if d == 0 else 'C2', lw=1.8,
             label=f'원본 차원 {d+1}')
    ax2.plot(t, xr[:, d], '--', color='C0' if d == 0 else 'C2', lw=1.8,
             label=f'복원(k={K_DEMO}) 차원 {d+1}')
ax2.set_xlabel('타임스텝'); ax2.set_ylabel('값')
ax2.set_title(f'(b) 상위 {K_DEMO}개 저주파 계수만으로 복원 ({H*D//(K_DEMO*D)}배 압축)\n'
              f'(스텝당 저장량 {H*D}→{K_DEMO*D} 스칼라, NRMSE={nrmse(x0, xr):.3f})')
ax2.grid(alpha=0.3); ax2.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(OUT + 'fig2_compression_vs_error.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: flow matching 1D 토이 — 노이즈 → 데이터 (직선보간 + Euler 적분)
#   40강 회수: x_t = (1-t) x0 + t x1, 목표 속도장 v = x1 - x0.
#   여기서는 데이터 분포를 두 봉우리(다봉) 가우시안 혼합으로 두고,
#   해석적 조건부 속도장을 이용해 Euler N스텝으로 노이즈를 데이터로 옮긴다.
# ============================================================================
# 데이터 분포 p1: 두 델타에 가까운 좁은 가우시안 혼합 (다봉성 = 액션의 특징)
modes = np.array([-2.0, 2.0])
mode_w = np.array([0.5, 0.5])
sig1 = 0.15

# 조건부 확률경로 (Gaussian probability path, MIT 6.S184 스타일):
#   p_t(x) = sum_k w_k N(x; t*mu_k, ((1-t)+t*sig1)^2 )   [x0 ~ N(0,1)]
# 주변 속도장 v_t(x) = E[ x1 - x0 | x_t = x ] (사후 가중 평균)로 근사.
def marginal_velocity(x, tt):
    # x0 ~ N(0,1), x1 ~ mixture(modes). 조건부: x_t = (1-t)x0 + t x1.
    # 각 모드 k에 대해 x_t ~ N( t*mu_k, (1-t)^2 + t^2 sig1^2 ).
    var = (1-tt)**2 + (tt**2)*sig1**2
    resp = np.zeros((len(x), len(modes)))
    for k, mu in enumerate(modes):
        resp[:, k] = mode_w[k]*np.exp(-0.5*(x - tt*mu)**2/var)/np.sqrt(2*np.pi*var)
    resp /= resp.sum(axis=1, keepdims=True) + 1e-12
    # 각 모드 조건에서 E[x1-x0 | x_t]:
    #   x1 | (x_t, mode k) 의 사후평균, x0 = (x_t - t x1)/(1-t)
    v = np.zeros(len(x))
    for k, mu in enumerate(modes):
        # x1 ~ N(mu, sig1^2), x_t = (1-t)x0 + t x1, x0~N(0,1)
        # 사후 E[x1 | x_t, k]: 선형가우시안
        a = tt; b = (1-tt)              # x_t = a*x1 + b*x0
        # prior x1~N(mu,sig1^2), obs x_t = a x1 + b x0, x0~N(0,1) → noise var b^2
        post_var = 1.0/(1.0/sig1**2 + a**2/b**2)
        post_mean = post_var*(mu/sig1**2 + a*x/b**2)
        E_x1 = post_mean
        E_x0 = (x - tt*E_x1)/(1-tt + 1e-9)
        v += resp[:, k]*(E_x1 - E_x0)
    return v

# 여러 노이즈 샘플을 Euler N=10 스텝으로 적분
N_STEPS = 10
n_samp = 400
x0s = rng.standard_normal(n_samp)
dt_fm = 1.0/N_STEPS
paths = np.zeros((N_STEPS+1, n_samp))
x = x0s.copy()
paths[0] = x
for i in range(N_STEPS):
    tt = i*dt_fm
    x = x + dt_fm*marginal_velocity(x, tt)
    paths[i+1] = x
x_final = paths[-1]

# 최종 샘플이 두 모드 근처로 갔는지: 각 모드에 배정 후 평균/표준편차
assign = np.argmin(np.abs(x_final[:, None] - modes[None, :]), axis=1)
recovered_means = np.array([x_final[assign == k].mean() for k in range(len(modes))])
mode_err = np.max(np.abs(recovered_means - modes))
frac_mode0 = np.mean(assign == 0)

# Euler 스텝 수에 따른 최종 분포 품질 (모드 위치 오차)
def run_euler(n_steps):
    dt2 = 1.0/n_steps
    xx = rng2.standard_normal(2000)
    for i in range(n_steps):
        xx = xx + dt2*marginal_velocity(xx, i*dt2)
    asg = np.argmin(np.abs(xx[:, None] - modes[None, :]), axis=1)
    rm = np.array([xx[asg == k].mean() for k in range(len(modes))])
    return np.max(np.abs(rm - modes))
rng2 = np.random.default_rng(7)
step_counts = [1, 2, 3, 5, 10, 20, 50]
step_errs = [run_euler(n) for n in step_counts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 궤적: 노이즈(t=0) → 데이터 두 봉우리(t=1)
tgrid = np.linspace(0, 1, N_STEPS+1)
for j in range(0, n_samp, 4):
    c = 'C0' if assign[j] == 0 else 'C1'
    ax1.plot(tgrid, paths[:, j], color=c, lw=0.5, alpha=0.35)
for mu in modes:
    ax1.axhline(mu, color='k', ls=':', lw=1)
ax1.text(1.01, modes[0], '모드 A', fontsize=9, va='center')
ax1.text(1.01, modes[1], '모드 B', fontsize=9, va='center')
ax1.set_xlabel('flow 시간 t (0=노이즈, 1=데이터)')
ax1.set_ylabel('행동 값 x')
ax1.set_title(f'(a) flow matching {N_STEPS}-스텝 Euler 적분\n노이즈 N(0,1) → 다봉 데이터 (직선보간 속도장)')
ax1.grid(alpha=0.3); ax1.set_xlim(0, 1.15)

# (b) Euler 스텝 수 vs 모드 위치 오차
ax2.semilogx(step_counts, step_errs, 'C2o-', lw=2, ms=6)
ax2.axhline(0, color='k', ls='--', lw=1)
for n, e in zip(step_counts, step_errs):
    if n in (1, 10):
        ax2.annotate(f'{n}스텝\n오차 {e:.3f}', xy=(n, e),
                     xytext=(n, e+0.08), fontsize=8.5, ha='center')
ax2.set_xlabel('Euler 적분 스텝 수 (log)')
ax2.set_ylabel('복원된 모드 위치 오차 [최대 |Δ|]')
ax2.set_title('(b) 스텝 수 ↔ 궤적 품질 트레이드오프\n(π0: ~10스텝이면 충분 — ODE 솔버 감각)')
ax2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(OUT + 'fig3_flow_matching.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 4: attention 공유 action expert (E3) — 마스크 구조 + 블록별 가중치
#   VLM 토큰 = 인과 마스크(하삼각), 액션 토큰 = 양방향(마스크 없음).
#   attention은 공유(한 행렬), 가중치는 블록별로 다름(색으로 표시).
# ============================================================================
M_TOK, H_TOK = 6, 8          # VLM 토큰 6개(축약) + 액션 토큰 8개(H 축약)
n = M_TOK + H_TOK
# 허용 마스크: 1=볼 수 있음, 0=가림
mask = np.ones((n, n))
# VLM 구간(0..M-1): 인과(자기 이하만)
for i in range(M_TOK):
    for j in range(n):
        if j > i:
            mask[i, j] = 0
# 액션 구간(M..n-1): VLM 전부 + 액션 전부 양방향 (마스크 없음) → 이미 1
# (액션은 미래 액션도 봄: 궤적은 문장이 아니다)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                gridspec_kw={'width_ratios': [1.15, 1]})
# (a) attention 마스크 히트맵
ax1.imshow(mask, cmap='Greys', vmin=0, vmax=1.6, aspect='equal')
ax1.axhline(M_TOK-0.5, color='C0', lw=2); ax1.axvline(M_TOK-0.5, color='C0', lw=2)
ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
labs = [f'V{i+1}' for i in range(M_TOK)] + [f'a{i+1}' for i in range(H_TOK)]
ax1.set_xticklabels(labs, fontsize=8); ax1.set_yticklabels(labs, fontsize=8)
ax1.set_ylabel('보는 주체 (query)  |  V=VLM · a=액션')
ax1.set_xlabel('보이는 대상 (key)  |  V=VLM · a=액션')
ax1.text(0.9, 3.3, 'VLM:\n인과\n(하삼각)', fontsize=8, color='w')
ax1.text(M_TOK+2.0, M_TOK+3.3, '액션:\n양방향\n(전부 봄)', fontsize=8, color='w')
ax1.set_title('(a) 공유 attention 마스크 (파란선=VLM/액션 경계)\n검음=볼 수 있음 / 흰색=가림', pad=12)

# (b) 블록별 가중치 = 별도 파라미터 (개념도)
ax2.axis('off')
ax2.text(0.5, 0.95, 'E3: 별도 가중치 · 같은 시퀀스 · 액션 양방향',
         ha='center', fontsize=10, weight='bold', transform=ax2.transAxes)
# VLM 블록
ax2.add_patch(plt.Rectangle((0.08, 0.55), 0.36, 0.28, fc='#e1eefb', ec='C0', lw=2,
                             transform=ax2.transAxes))
ax2.text(0.26, 0.69, 'VLM 가중치\n$W^{VLM}$\nSigLIP+Gemma\n(웹 지식)',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes)
# 액션 블록
ax2.add_patch(plt.Rectangle((0.56, 0.55), 0.36, 0.28, fc='#f7dede', ec='C3', lw=2,
                             transform=ax2.transAxes))
ax2.text(0.74, 0.69, 'Action expert\n$W^{act}$ (~300M)\nflow $v_\\theta$\n(별도 가중치)',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes)
# 공유 attention 바
ax2.add_patch(plt.Rectangle((0.08, 0.30), 0.84, 0.13, fc='#efe4d8', ec='#9a6a3a', lw=2,
                             transform=ax2.transAxes))
ax2.text(0.5, 0.365, '공유 attention (한 시퀀스에서 서로를 봄)',
         ha='center', va='center', fontsize=9, transform=ax2.transAxes)
ax2.annotate('', xy=(0.26, 0.44), xytext=(0.26, 0.54), transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='<->', color='gray'))
ax2.annotate('', xy=(0.74, 0.44), xytext=(0.74, 0.54), transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='<->', color='gray'))
ax2.text(0.5, 0.16, 'RT-2/OpenVLA: 한 뇌(같은 $W$, 이산 AR)\n'
                    'CogACT(47강): 완전 분리 두 모듈\n'
                    'π0: 그 중간 — $W$는 둘, attention은 하나',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes,
         bbox=dict(boxstyle='round', fc='white', ec='gray'))
fig.tight_layout()
fig.savefig(OUT + 'fig4_attention_expert.png', dpi=140)
plt.close(fig)

# 마스크 검증용 수치
vlm_causal_ok = all(mask[i, j] == 0 for i in range(M_TOK) for j in range(n) if j > i)
act_bidir_ok = all(mask[i, j] == 1 for i in range(M_TOK, n) for j in range(n))

# ============================================================================
# 본문 인용 수치 출력 (그림 없음 포함) — 본문과 일치해야 함
# ============================================================================
print("=== 그림 파일 4개 생성 완료 ===")
print(f"[E3 마스크] VLM 인과 마스크 OK={vlm_causal_ok}, 액션 양방향 OK={act_bidir_ok} "
      f"(액션은 미래 토큰도 봄)")
print(f"[파세발] 시간영역 에너지 {energy_time:.6f} vs 주파수영역 {energy_freq:.6f} "
      f"→ 차이 {parseval_err:.2e}")
print(f"[저주파 집중] 저주파 5계수 = 전체 에너지 {frac_top5*100:.1f}% / "
      f"10계수 = {frac_top10*100:.1f}%")
print(f"[압축] 목표 NRMSE={TARGET}: DCT k={k_needed} → {dct_scalars_at_target} 스칼라, "
      f"naive 스텝별 유지 {naive_scalars} 스칼라 → 표현 크기 {token_ratio:.1f}배 절감")
print(f"[DCT k=5 복원] traj0 NRMSE = {nrmse(x0, xr):.4f} "
      f"(저장량 {H*D}→{5*D} 스칼라, {H*D/(5*D):.1f}배)")
print(f"[flow matching] {N_STEPS}스텝 Euler 후 복원 모드평균 = "
      f"{recovered_means[0]:.4f}, {recovered_means[1]:.4f} "
      f"(목표 {modes[0]:.1f}, {modes[1]:.1f}), 최대오차 {mode_err:.4f}, "
      f"모드A 비율 {frac_mode0*100:.0f}%")
print("[Euler 스텝 vs 모드오차]", {n: round(e, 4) for n, e in zip(step_counts, step_errs)})

# WE 손계산 검증용: 8점 S자 상승 궤적 (관절이 목표로 이동하는 매끄러운 운동)
we = np.array([0.0, 0.2, 0.5, 0.9, 1.4, 1.7, 1.9, 2.0])
we_dct = dct(we, type=2, norm='ortho')
print(f"[WE 손계산] x={list(we)} 의 DCT(ortho) = {np.round(we_dct, 3).tolist()}")
print(f"[WE 파세발] ||x||^2={np.sum(we**2):.4f} == ||X||^2={np.sum(we_dct**2):.4f}")
print(f"[WE 저주파 집중] 상위 2계수 에너지 비율 = "
      f"{np.sum(we_dct[:2]**2)/np.sum(we_dct**2)*100:.2f}%")
# 상위 2계수만 유지한 복원 (8→2, 4배)
we_k2 = we_dct.copy(); we_k2[2:] = 0
we_rec = idct(we_k2, type=2, norm='ortho')
print(f"[WE k=2 복원] {np.round(we_rec, 3).tolist()}, NRMSE={nrmse(we, we_rec):.4f} "
      f"(저장 8→2 스칼라, 4배)")
