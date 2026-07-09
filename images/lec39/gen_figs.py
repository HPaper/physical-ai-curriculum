# Lec 39 그림 생성 스크립트 — Diffusion Policy
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델/GPU 없음.
#   핵심: 행동 다봉성 → MSE 붕괴 vs diffusion 복원, forward/reverse 과정,
#         DDPM vs DDIM 스텝수-품질(수십 스텝 — 40강 flow ~10스텝 대비), receding horizon.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공통: 1D 다봉 데이터 분포 (같은 관측에서 왼쪽/오른쪽 두 옳은 행동)
#   40강 flow 토이와 동일한 두 모드 {-2, +2} — 계보 정합.
# ============================================================================
MODES = np.array([-2.0, 2.0])
MODE_W = np.array([0.5, 0.5])
SIG_DATA = 0.15                     # 각 모드의 유한 폭

def sample_data(n, rng):
    which = rng.choice(len(MODES), size=n, p=MODE_W)
    return MODES[which] + SIG_DATA * rng.standard_normal(n)

# ============================================================================
# DDPM 스케줄 (Ho et al. 2020) — 최소 구현
#   forward:  q(x_t | x_0) = N( sqrt(abar_t) x_0, (1-abar_t) I )
#   여기서는 "참 스코어/노이즈"를 데이터 분포로부터 해석적으로 계산(학습 대신).
#   x_t ~ sum_k w_k N( sqrt(abar) mu_k, abar*sig^2 + (1-abar) )
# ============================================================================
def make_schedule(T, beta1=1e-4, beta2=0.02):
    betas = np.linspace(beta1, beta2, T)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return betas, alphas, abar

def true_eps(x, t_idx, abar):
    """참 노이즈 예측 eps_theta*(x,t) = -sqrt(1-abar) * score(x_t).
    데이터가 가우시안 혼합이므로 x_t의 주변분포도 혼합 → score 해석적."""
    ab = abar[t_idx]
    var = ab * SIG_DATA**2 + (1.0 - ab)          # x_t의 각 성분 분산
    mu_t = np.sqrt(ab) * MODES                    # x_t의 각 성분 평균
    # 책임도(responsibility)
    resp = np.stack([MODE_W[k] * np.exp(-0.5*(x - mu_t[k])**2/var) / np.sqrt(2*np.pi*var)
                     for k in range(len(MODES))], axis=1)
    resp /= resp.sum(axis=1, keepdims=True) + 1e-30
    # score = d/dx log p_t(x) = sum_k resp_k * (mu_t_k - x)/var
    score = np.sum(resp * (mu_t[None, :] - x[:, None]) / var, axis=1)
    eps = -np.sqrt(1.0 - ab) * score              # eps = -sqrt(1-abar) * score
    return eps

def ddpm_sample(T, n, rng, sched=None):
    """확률적 역과정 (조상 샘플링). T 스텝 모두 사용."""
    betas, alphas, abar = sched if sched else make_schedule(T)
    x = rng.standard_normal(n)                     # x_T ~ N(0, I)
    for t in range(T-1, -1, -1):
        eps = true_eps(x, t, abar)
        coef = (1 - alphas[t]) / np.sqrt(1 - abar[t])
        mean = (x - coef * eps) / np.sqrt(alphas[t])
        if t > 0:
            sigma = np.sqrt(betas[t])
            x = mean + sigma * rng.standard_normal(n)
        else:
            x = mean
    return x

def ddim_sample(n_steps, n, rng, T=1000, eta=0.0):
    """결정적 역과정 (DDIM, eta=0). T개 중 n_steps개만 골라 건너뛴다."""
    betas, alphas, abar = make_schedule(T)
    ts = np.linspace(0, T-1, n_steps).round().astype(int)   # 균등 서브샘플
    ts = np.unique(ts)
    x = rng.standard_normal(n)                     # x_T
    for i in range(len(ts)-1, -1, -1):
        t = ts[i]
        ab_t = abar[t]
        eps = true_eps(x, t, abar)
        x0_pred = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)   # x0 추정
        if i > 0:
            t_prev = ts[i-1]
            ab_prev = abar[t_prev]
            # DDIM 갱신 (eta=0 → 결정적): x_{prev} = sqrt(abar_prev) x0 + sqrt(1-abar_prev) eps
            x = np.sqrt(ab_prev) * x0_pred + np.sqrt(1 - ab_prev) * eps
        else:
            x = x0_pred
    return x

# 모드 복원 품질 지표: 최종 샘플을 두 모드에 배정 후 위치 오차
def mode_error(x):
    asg = np.argmin(np.abs(x[:, None] - MODES[None, :]), axis=1)
    rm = np.array([x[asg == k].mean() if np.any(asg == k) else np.nan for k in range(len(MODES))])
    return np.nanmax(np.abs(rm - MODES)), rm, asg

# 두 모드 모두 채워졌는지 (mode coverage): 각 모드 근처 표본 비율
def mode_coverage(x):
    asg = np.argmin(np.abs(x[:, None] - MODES[None, :]), axis=1)
    return np.array([np.mean(asg == k) for k in range(len(MODES))])

# ============================================================================
# 그림 1: 행동 다봉성 — MSE 회귀는 가운데로 붕괴, diffusion은 두 모드 복원
# ============================================================================
rng = np.random.default_rng(0)
N = 4000
data = sample_data(N, rng)

# (a) MSE 회귀: 조건부 평균 = 두 모드의 평균 = 0 (벽으로 돌진)
#     같은 관측 o에 대한 최적 MSE 예측은 E[a|o] = 0.
mse_pred = data.mean()                              # 단일 관측 조건이므로 전체 평균 = 0 부근

# (b) diffusion 샘플 (DDPM T=200)
T_FULL = 200
x_ddpm = ddpm_sample(T_FULL, N, np.random.default_rng(1))
cov_ddpm = mode_coverage(x_ddpm)
err_ddpm, rm_ddpm, _ = mode_error(x_ddpm)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
bins = np.linspace(-3.5, 3.5, 80)
ax1.hist(data, bins=bins, color='C7', alpha=0.55, density=True, label='데이터 $p(a\\,|\\,o)$ (다봉)')
ax1.axvline(mse_pred, color='C3', lw=2.5, label=f'MSE 회귀 예측 = E[a|o] = {mse_pred:.2f}')
for mu in MODES:
    ax1.axvline(mu, color='k', ls=':', lw=1)
ax1.text(0.0, 1.15, '평균은\n두 모드 사이\n(벽으로 돌진)', color='C3', fontsize=9,
         ha='center', va='top')
ax1.set_xlabel('행동 값 a'); ax1.set_ylabel('밀도')
ax1.set_title('(a) 왜 가우시안/MSE가 실패하는가\n조건부 평균 = 모드 사이 = 금지된 행동')
ax1.legend(fontsize=8.5, loc='upper right'); ax1.grid(alpha=0.3)

ax2.hist(data, bins=bins, color='C7', alpha=0.45, density=True, label='데이터 (다봉)')
ax2.hist(x_ddpm, bins=bins, color='C0', alpha=0.55, density=True,
         label=f'diffusion 샘플 (DDPM, T={T_FULL})')
for mu in MODES:
    ax2.axvline(mu, color='k', ls=':', lw=1)
ax2.text(-2.0, 1.1, f'모드 A\n{cov_ddpm[0]*100:.0f}%', ha='center', fontsize=9, color='C0')
ax2.text(2.0, 1.1, f'모드 B\n{cov_ddpm[1]*100:.0f}%', ha='center', fontsize=9, color='C0')
ax2.set_xlabel('행동 값 a'); ax2.set_ylabel('밀도')
ax2.set_title(f'(b) diffusion은 두 모드를 모두 살린다\n위치오차 {err_ddpm:.3f} · 커버리지 {cov_ddpm[0]*100:.0f}/{cov_ddpm[1]*100:.0f}%')
ax2.legend(fontsize=8.5, loc='upper right'); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + 'fig1_mode_averaging.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 2: forward(노이즈 주입) / reverse(denoise) 과정
# ============================================================================
T2 = 200
betas2, alphas2, abar2 = make_schedule(T2)
rng2 = np.random.default_rng(3)
n_show = 1500
x0 = sample_data(n_show, rng2)

# forward: x_t = sqrt(abar) x0 + sqrt(1-abar) eps
snap_t = [0, 20, 60, 120, 199]
fwd = {}
noise = rng2.standard_normal(n_show)
for t in snap_t:
    fwd[t] = np.sqrt(abar2[t]) * x0 + np.sqrt(1 - abar2[t]) * noise

# reverse: DDPM에서 스냅샷 저장
rng3 = np.random.default_rng(4)
x = rng3.standard_normal(n_show)
rev = {T2-1: x.copy()}
save_rev = {199: T2-1, 120: 120, 60: 60, 20: 20, 0: 0}
for t in range(T2-1, -1, -1):
    eps = true_eps(x, t, abar2)
    coef = (1 - alphas2[t]) / np.sqrt(1 - abar2[t])
    mean = (x - coef * eps) / np.sqrt(alphas2[t])
    if t > 0:
        x = mean + np.sqrt(betas2[t]) * rng3.standard_normal(n_show)
    else:
        x = mean
    if t in save_rev:
        rev[t] = x.copy()

fig, axes = plt.subplots(2, len(snap_t), figsize=(13, 5.2), sharex=True, sharey='row')
bins2 = np.linspace(-3.5, 3.5, 60)
for j, t in enumerate(snap_t):
    axes[0, j].hist(fwd[t], bins=bins2, color='C1', alpha=0.7, density=True)
    axes[0, j].set_title(f't={t}\n$\\bar\\alpha$={abar2[t]:.3f}', fontsize=9)
    axes[0, j].grid(alpha=0.3)
for j, t in enumerate(snap_t):
    axes[1, j].hist(rev[t], bins=bins2, color='C0', alpha=0.7, density=True)
    axes[1, j].grid(alpha=0.3)
axes[0, 0].set_ylabel('forward →\n(노이즈 주입)\n밀도', fontsize=9)
axes[1, 0].set_ylabel('← reverse\n(배운 denoiser)\n밀도', fontsize=9)
for j in range(len(snap_t)):
    axes[1, j].set_xlabel('a')
fig.suptitle('그림 2: forward q(x_t|x_0)는 데이터를 노이즈로 뭉갠다(위, 왼→오) · '
             'reverse는 배운 denoiser로 되돌린다(아래, 오→왼)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT + 'fig2_forward_reverse.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: DDPM(확률·다스텝) vs DDIM(결정·적은 스텝) — 스텝수 vs 품질
#   diffusion은 수십 스텝 필요 (40강 flow ~10스텝 대비의 수치 근거).
# ============================================================================
step_list = [2, 5, 10, 20, 50, 100, 200]
n_eval = 6000

# DDIM (결정적): T=1000에서 서브샘플
ddim_errs = []
for ns in step_list:
    xe = ddim_sample(ns, n_eval, np.random.default_rng(100 + ns), T=1000, eta=0.0)
    err, _, _ = mode_error(xe)
    ddim_errs.append(err)

# DDPM (확률적): 스케줄 길이 = 스텝 수 (T=step)
ddpm_errs = []
for ns in step_list:
    xe = ddpm_sample(ns, n_eval, np.random.default_rng(200 + ns))
    err, _, _ = mode_error(xe)
    ddpm_errs.append(err)

# 40강 flow 참조값(직선보간 토이, lec44 WE-2와 동일 수치): 스텝별 오차
flow_steps = [1, 2, 3, 5, 10, 20, 50]
flow_errs = [2.0, 0.55, 0.15, 0.07, 0.04, 0.02, 0.015]

# 임계(0.05) 도달에 필요한 최소 스텝
def min_steps(steps, errs, thr=0.05):
    for s, e in zip(steps, errs):
        if e <= thr:
            return s
    return None
ddim_min = min_steps(step_list, ddim_errs)
ddpm_min = min_steps(step_list, ddpm_errs)
flow_min = min_steps(flow_steps, flow_errs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax1.loglog(step_list, ddim_errs, 'C0o-', lw=2, ms=6, label='DDIM (결정적, eta=0)')
ax1.loglog(step_list, ddpm_errs, 'C3s--', lw=2, ms=6, label='DDPM (확률적)')
ax1.axhline(0.05, color='gray', ls=':', lw=1.2, label='품질 임계 (오차 0.05)')
ax1.set_xlabel('역과정 스텝 수 (log)')
ax1.set_ylabel('모드 위치 오차 (log)')
ax1.set_title('(a) diffusion: 스텝수 ↔ 품질\nDDIM은 적은 스텝에서 DDPM보다 낫다')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=8.5)

# (b) DDPM(diffusion 기본 샘플러) vs flow(40강) 스텝 예산 비교
#   Diffusion Policy의 기본 = DDPM 조상 샘플링. flow(40강)는 결정적 ODE.
ax2.loglog(step_list, ddpm_errs, 'C3s-', lw=2, ms=6, label='DDPM (diffusion 기본, 이 강의)')
ax2.loglog(flow_steps, flow_errs, 'C2^--', lw=2, ms=6, label='flow matching (40강 · lec44 WE-2)')
ax2.axhline(0.05, color='gray', ls=':', lw=1.2, label='품질 임계 (오차 0.05)')
ax2.axvline(flow_min, color='C2', ls=':', lw=1, alpha=0.7)
ax2.axvline(ddpm_min, color='C3', ls=':', lw=1, alpha=0.7)
ax2.set_xlabel('적분/역과정 스텝 수 (log)')
ax2.set_ylabel('모드 위치 오차 (log)')
ax2.set_title(f'(b) DDPM vs flow 스텝 예산\nflow ~{flow_min}스텝 vs DDPM ~{ddpm_min}스텝 (같은 임계)')
ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + 'fig3_ddpm_vs_ddim.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 4: receding horizon 실행 (생성 H, 실행 h<H, 재관측 반복)
#   23강 MPC receding horizon과 동일 구조의 확률적 버전.
# ============================================================================
# 참 궤적(따라야 할 목표) + 관측마다 diffusion이 청크를 생성(여기선 잡음 섞인 예측으로 흉내)
rng4 = np.random.default_rng(5)
Tsim = 40
true_traj = 1.2*np.sin(2*np.pi*np.arange(Tsim+16)/28.0)   # 부드러운 목표
H_chunk = 16     # 생성 지평선
h_exec = 8       # 실행 스텝

fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.plot(np.arange(len(true_traj)), true_traj, 'k-', lw=1.4, alpha=0.5, label='목표(참) 궤적')
executed = []
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
step = 0
ci = 0
replans = []
while step < Tsim:
    # 관측 시점 step에서 청크 생성: 미래 H스텝 예측(예측 오차 = 미래로 갈수록 커짐)
    idx = np.arange(step, step + H_chunk)
    pred = true_traj[idx] + rng4.normal(0, 0.04*(1 + np.arange(H_chunk)*0.12))
    ax.plot(idx, pred, color=colors[ci % len(colors)], lw=1.0, ls='--', alpha=0.55,
            label='생성 청크 (H=16)' if ci == 0 else None)
    ax.plot(idx[0], pred[0], 'o', color=colors[ci % len(colors)], ms=4)
    # 실행: 앞 h스텝만
    ex_idx = idx[:h_exec]
    ax.plot(ex_idx, pred[:h_exec], color=colors[ci % len(colors)], lw=3.0, alpha=0.9,
            label='실행 (h=8)' if ci == 0 else None)
    replans.append(step)
    executed.extend(list(zip(ex_idx, pred[:h_exec])))
    step += h_exec
    ci += 1

for r in replans:
    ax.axvline(r, color='gray', ls=':', lw=0.8, alpha=0.6)
ax.text(replans[1], ax.get_ylim()[1]*0.92, '재관측·재계획', fontsize=8.5, color='gray',
        rotation=90, va='top')
ax.set_xlabel('타임스텝'); ax.set_ylabel('행동 / 상태')
ax.set_title('그림 4: receding horizon — 청크 H=16 생성, 앞 h=8만 실행 후 재관측·재계획\n'
             '(23강 MPC의 "N계획·1실행"의 확률적·청크 버전)')
ax.set_xlim(0, Tsim + 2); ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + 'fig4_receding_horizon.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문과 일치해야 함
# ============================================================================
print("=== 그림 4개 생성 완료 ===")
print(f"[다봉/MSE] 데이터 두 모드 {MODES.tolist()}, MSE 회귀 예측 E[a|o] = {mse_pred:.4f} "
      f"(모드 사이 0 부근 — 금지된 행동)")
print(f"[diffusion 복원] DDPM T={T_FULL}: 커버리지 {cov_ddpm[0]*100:.0f}/{cov_ddpm[1]*100:.0f}%, "
      f"위치오차 {err_ddpm:.4f}, 모드평균 {rm_ddpm[0]:.4f}/{rm_ddpm[1]:.4f}")
print("[DDIM 스텝 vs 오차]", {s: round(e, 4) for s, e in zip(step_list, ddim_errs)})
print("[DDPM 스텝 vs 오차]", {s: round(e, 4) for s, e in zip(step_list, ddpm_errs)})
print(f"[임계 0.05 도달 스텝] DDIM={ddim_min}, DDPM={ddpm_min}, flow(40강)={flow_min}")
print(f"[forward abar] t=0:{abar2[0]:.4f}, t=20:{abar2[20]:.4f}, t=60:{abar2[60]:.4f}, "
      f"t=120:{abar2[120]:.4f}, t=199:{abar2[199]:.4f}")
print(f"[receding horizon] H={H_chunk} 생성, h={h_exec} 실행, 재계획 시점={replans}")
