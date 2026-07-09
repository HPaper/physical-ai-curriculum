# Lec 40 그림 생성 스크립트 — Flow matching (직선보간 + ODE)
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델/GPU 없음.
# 40강은 flow matching의 "원류" 강의 — 여기 수치는 44강(π0)과 정합해야 한다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공통: 1D 다봉 데이터 분포와 직선보간 속도장 (사후평균)
#   데이터 x1 ~ mixture( {-2,+2}, sig1 ),  노이즈 x0 ~ N(0,1)
#   직선보간 x_t = (1-t)x0 + t x1,  목표 속도 u_t = x1 - x0
#   주변 속도장 v_t(x) = E[x1 - x0 | x_t = x]  (조건부 속도의 사후평균)
# ============================================================================
modes = np.array([-2.0, 2.0])
mode_w = np.array([0.5, 0.5])
sig1 = 0.15                                   # 데이터 모드의 유한 폭

def marginal_velocity(x, t):
    """v_t(x) = E[x1 - x0 | x_t = x]  (선형가우시안 사후평균의 혼합 가중)."""
    var = (1 - t)**2 + (t**2) * sig1**2
    resp = np.stack([mode_w[k] * np.exp(-0.5*(x - t*mu)**2/var)/np.sqrt(2*np.pi*var)
                     for k, mu in enumerate(modes)], axis=1)
    resp /= resp.sum(1, keepdims=True) + 1e-12
    v = np.zeros_like(x)
    a, b = t, (1 - t)
    for k, mu in enumerate(modes):
        post_var = 1.0/(1.0/sig1**2 + a**2/b**2)      # 사후 분산
        E_x1 = post_var*(mu/sig1**2 + a*x/b**2)        # E[x1 | x_t, mode k]
        E_x0 = (x - t*E_x1)/(1 - t + 1e-9)             # x0 = (x_t - t x1)/(1-t)
        v += resp[:, k]*(E_x1 - E_x0)
    return v

# ============================================================================
# 그림 1: 직선보간 속도장 (위상공간 벡터장)
#   (a) 개별 조건부 직선 x_t=(1-t)x0+t x1 (여러 (x0,x1) 쌍)
#   (b) 주변 속도장 v_t(x)의 (t, x) 평면 벡터장 — 노이즈에서 두 모드로 갈라짐
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 조건부 직선 몇 개 (노이즈 한 점 -> 데이터 한 점, 가장 곧은 길)
rng_c = np.random.default_rng(3)
tt = np.linspace(0, 1, 50)
for _ in range(9):
    x0 = rng_c.standard_normal()
    x1 = modes[rng_c.integers(0, 2)] + sig1*rng_c.standard_normal()
    xt = (1 - tt)*x0 + tt*x1
    c = 'C0' if x1 < 0 else 'C1'
    ax1.plot(tt, xt, color=c, lw=1.4, alpha=0.8)
    ax1.plot(0, x0, 'ko', ms=3); ax1.plot(1, x1, 's', color=c, ms=5)
for mu in modes:
    ax1.axhline(mu, color='gray', ls=':', lw=1)
ax1.set_xlabel('flow 시간 t  (0=노이즈, 1=데이터)')
ax1.set_ylabel('행동 값 x')
ax1.set_title('(a) 조건부 직선보간 $x_t=(1{-}t)x_0+t\\,x_1$\n'
              '기울기 = 조건부 속도 $u_t = x_1-x_0$ (상수)')
ax1.grid(alpha=0.3)

# (b) 주변 속도장 벡터장 (t, x) 평면
T = np.linspace(0.02, 0.98, 16)
X = np.linspace(-3.2, 3.2, 18)
TT, XX = np.meshgrid(T, X)
U = np.ones_like(TT)                              # dt 방향 성분(시간은 항상 +1)
V = np.zeros_like(XX)
for i in range(TT.shape[0]):
    V[i, :] = marginal_velocity(XX[i, :], None) if False else \
              np.array([marginal_velocity(np.array([XX[i, j]]), TT[i, j])[0]
                        for j in range(TT.shape[1])])
# 화살표 정규화(시각화용): 방향만 보이도록 스케일
mag = np.sqrt(U**2 + V**2)
ax2.quiver(TT, XX, U/mag, V/mag, mag, cmap='viridis', pivot='mid',
           scale=26, width=0.004, alpha=0.9)
for mu in modes:
    ax2.axhline(mu, color='C3', ls='--', lw=1.2)
ax2.text(1.0, modes[0], '모드 A', color='C3', fontsize=9, va='center', ha='right')
ax2.text(1.0, modes[1], '모드 B', color='C3', fontsize=9, va='center', ha='right')
ax2.set_xlabel('flow 시간 t')
ax2.set_ylabel('행동 값 x')
ax2.set_title('(b) 주변 속도장 $v_t(x)=E[x_1{-}x_0\\mid x_t{=}x]$\n'
              '위상공간 벡터장 — 노이즈가 두 모드로 갈라진다')
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + 'fig1_velocity_field.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 2: Euler 적분 궤적 (노이즈 N(0,1) -> 두 모드)
#   10스텝 Euler로 400 샘플을 민다. 궤적이 두 모드로 갈라짐 = 다봉성 표현.
# ============================================================================
N_STEPS = 10
n_samp = 400
rng = np.random.default_rng(0)
x0s = rng.standard_normal(n_samp)
dt = 1.0/N_STEPS
paths = np.zeros((N_STEPS+1, n_samp)); x = x0s.copy(); paths[0] = x
for i in range(N_STEPS):
    x = x + dt*marginal_velocity(x, i*dt); paths[i+1] = x
x_final = paths[-1]
assign = np.argmin(np.abs(x_final[:, None] - modes[None, :]), axis=1)
recovered = np.array([x_final[assign == k].mean() for k in range(2)])
mode_err = np.max(np.abs(recovered - modes))
frac0 = np.mean(assign == 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                gridspec_kw={'width_ratios': [1.35, 1]})
tg = np.linspace(0, 1, N_STEPS+1)
for j in range(0, n_samp, 3):
    c = 'C0' if assign[j] == 0 else 'C1'
    ax1.plot(tg, paths[:, j], color=c, lw=0.5, alpha=0.35)
for mu in modes:
    ax1.axhline(mu, color='k', ls=':', lw=1)
ax1.text(1.01, modes[0], '모드 A', fontsize=9, va='center')
ax1.text(1.01, modes[1], '모드 B', fontsize=9, va='center')
ax1.set_xlabel('flow 시간 t (0=노이즈, 1=데이터)'); ax1.set_ylabel('행동 값 x')
ax1.set_title(f'(a) Euler {N_STEPS}스텝 적분: 노이즈 N(0,1) → 다봉 데이터\n'
              f'복원 모드 {recovered[0]:.3f}, {recovered[1]:.3f} (목표 −2, +2)')
ax1.grid(alpha=0.3); ax1.set_xlim(0, 1.12)

# (b) 시작/끝 히스토그램
ax2.hist(x0s, bins=30, orientation='horizontal', color='gray', alpha=0.5,
         density=True, label='t=0: 노이즈 N(0,1)')
ax2.hist(x_final, bins=40, orientation='horizontal', color='C2', alpha=0.7,
         density=True, label='t=1: 복원 데이터')
for mu in modes:
    ax2.axhline(mu, color='k', ls=':', lw=1)
ax2.set_xlabel('밀도'); ax2.set_ylabel('행동 값 x')
ax2.set_title(f'(b) 분포 변형 (최대오차 {mode_err:.3f})\n단봉 노이즈 → 이봉 데이터')
ax2.legend(fontsize=8, loc='upper right'); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + 'fig2_euler_trajectories.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: 스텝 수 vs 오차 — flow (직선 ODE) vs diffusion (DDPM 확률 역방)
#   같은 다봉 타깃. flow는 ~10스텝에서 포화, diffusion은 수십 스텝 필요.
# ============================================================================
def flow_sample(n_steps, seed=7, n=2000):
    xx = np.random.default_rng(seed).standard_normal(n)
    for i in range(n_steps):
        xx = xx + (1.0/n_steps)*marginal_velocity(xx, i/n_steps)
    return xx

def make_schedule(Tn):
    betas = np.linspace(1e-4, 0.02, Tn)
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)

def diff_score(x, abar_t):
    s = np.sqrt(abar_t); var = (s*sig1)**2 + (1 - abar_t)
    comp = np.stack([mode_w[k]*np.exp(-0.5*(x - s*mu)**2/var)/np.sqrt(2*np.pi*var)
                     for k, mu in enumerate(modes)], axis=1)
    p = comp.sum(1); g = np.zeros_like(x)
    for k, mu in enumerate(modes):
        g += comp[:, k]*(-(x - s*mu)/var)
    return g/(p + 1e-12)

def ddpm_sample(Tn, seed=7, n=2000):
    betas, alphas, abar = make_schedule(Tn)
    rng_d = np.random.default_rng(seed)
    x = rng_d.standard_normal(n)
    for t in range(Tn-1, -1, -1):
        score = diff_score(x, abar[t])
        mean = (x + betas[t]*score)/np.sqrt(alphas[t])
        x = mean + (np.sqrt(betas[t])*rng_d.standard_normal(n) if t > 0 else 0.0)
    return x

def quality(xx):
    a = np.argmin(np.abs(xx[:, None] - modes[None, :]), axis=1)
    rm = np.array([xx[a == k].mean() if np.any(a == k) else 1e9 for k in range(2)])
    return np.max(np.abs(rm - modes))

flow_steps = [1, 2, 3, 5, 8, 10, 20, 50]
flow_errs = [quality(flow_sample(n)) for n in flow_steps]
diff_steps = [5, 10, 20, 40, 80, 160]
diff_errs = [quality(ddpm_sample(Tn)) for Tn in diff_steps]

# 같은 품질(<=0.05) 최소 스텝
def min_steps(fn, cand):
    for c in cand:
        if quality(fn(c)) <= 0.05:
            return c
    return None
flow_min = min_steps(flow_sample, [1,2,3,4,5,6,7,8,9,10,12,15,20])
diff_min = min_steps(ddpm_sample, [5,10,15,20,30,40,60,80,120,160,240])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax1.loglog(flow_steps, flow_errs, 'C2o-', lw=2, ms=6, label='flow (직선 ODE, Euler)')
ax1.loglog(diff_steps, diff_errs, 'C3s--', lw=2, ms=6, label='diffusion (DDPM 확률 역방)')
ax1.axhline(0.05, color='gray', ls=':', lw=1.2, label='목표 오차 0.05')
ax1.set_xlabel('추론 스텝 수 (log)'); ax1.set_ylabel('복원 모드 위치 오차 (log)')
ax1.set_title('(a) 스텝 수 ↔ 품질: flow vs diffusion\n같은 다봉 타깃, 같은 오차 지표')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=8.5)

# (b) 막대: 같은 오차(<=0.05) 도달 스텝 수
ax2.bar(['flow\n(직선 ODE)', 'diffusion\n(DDPM)'], [flow_min, diff_min],
        color=['C2', 'C3'], width=0.55)
for i, (lab, v) in enumerate(zip(['flow', 'diff'], [flow_min, diff_min])):
    ax2.text(i, v + 1, f'{v}스텝', ha='center', fontsize=11, weight='bold')
ax2.set_ylabel('오차 ≤ 0.05 도달에 필요한 스텝 수')
ax2.set_title(f'(b) 같은 품질까지 스텝 수 — {diff_min/flow_min:.0f}배 차이\n'
              'flow가 로봇 폐루프(고주파)에 유리한 이유')
ax2.set_ylim(0, diff_min*1.25); ax2.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig3_steps_vs_error.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 4: 액션 헤드로서의 flow (π0/GR00T/SmolVLA 공통 도식)
#   VLM 컨텍스트 + 로봇 상태 + 노이즈 -> flow head (Euler N스텝) -> 액션 청크 H
# ============================================================================
fig, ax = plt.subplots(figsize=(11.5, 4.8))
ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 7)

def box(x, y, w, h, text, fc, ec, fs=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.8))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

# 입력들
box(0.3, 5.2, 2.4, 1.1, 'VLM 컨텍스트\n(이미지·언어 토큰)', '#e1eefb', 'C0')
box(0.3, 3.6, 2.4, 1.1, '로봇 상태\n$s$', '#e1eefb', 'C0')
box(0.3, 2.0, 2.4, 1.1, '노이즈 $x_0\\sim N(0,I)$', '#efe4d8', '#9a6a3a')

# flow head
box(4.0, 2.6, 3.4, 3.0,
    'flow head  $v_\\theta(x,t\\,|\\,s, \\mathrm{ctx})$\n\n'
    '직선보간 속도장 회귀\n$\\mathcal{L}=\\|v_\\theta-(x_1{-}x_0)\\|^2$\n\n'
    'Euler N≈10 스텝 적분\n$x_{k+1}=x_k+\\frac{1}{N} v_\\theta$',
    '#f7dede', 'C3', fs=9)

# 출력
box(8.7, 3.1, 3.0, 2.0,
    '액션 청크 $a_{t:t+H}$\n(H스텝 · 한 번에)\n→ 제어 스택(50강)',
    '#e6f4e6', 'C2', fs=9)

# 화살표
for (x0a, y0a) in [(2.7, 5.75), (2.7, 4.15), (2.7, 2.55)]:
    ax.annotate('', xy=(4.0, 4.1), xytext=(x0a, y0a),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.annotate('', xy=(8.7, 4.1), xytext=(7.4, 4.1),
            arrowprops=dict(arrowstyle='->', color='k', lw=2))
ax.text(8.05, 4.35, 'Euler\nN스텝', ha='center', fontsize=8, color='C3')

# 하단: 공통 채택 모델
ax.text(6.0, 1.2, 'π0 (44강): H=50, ≤18D, ~10스텝, 최대 50Hz   |   '
                  'GR00T (46강): DiT flow head, H=16→40   |   '
                  'SmolVLA (47강): ~100M flow expert',
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round', fc='white', ec='gray'))
ax.text(6.0, 6.6, 'flow를 "액션 헤드"로: VLM은 이해, flow head는 생성 — '
                  'π0/GR00T/SmolVLA의 공통 선택',
        ha='center', fontsize=10.5, weight='bold')
fig.tight_layout()
fig.savefig(OUT + 'fig4_flow_action_head.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문/캡션과 정확히 일치해야 함
# ============================================================================
print("=== 그림 4개 생성 완료 ===")
print(f"[Euler 10스텝] 복원 모드평균 = {recovered[0]:.4f}, {recovered[1]:.4f} "
      f"(목표 -2, 2), 최대오차 {mode_err:.4f}, 모드A 비율 {frac0*100:.0f}%")
# flow 스텝별 오차 (그림3 (a) flow 곡선; WE-1은 별도 시드7 순차 소비로 44강과 정합)
print("[flow 스텝별 오차 (재시드)]", {n: round(e, 3) for n, e in zip(flow_steps, flow_errs)})
print("[diff 스텝별 오차]", {n: round(e, 3) for n, e in zip(diff_steps, diff_errs)})
print(f"[같은 오차 0.05 도달] flow {flow_min}스텝 vs diffusion {diff_min}스텝 "
      f"-> {diff_min/flow_min:.0f}배")

# WE-1 (본문 코드 블록과 동일): 순차 시드7 소비 -> 44강 WE-2와 동일 수치
rng2 = np.random.default_rng(7)
def run_euler_seq(n):
    xx = rng2.standard_normal(2000)
    for i in range(n):
        xx = xx + (1.0/n)*marginal_velocity(xx, i/n)
    a = np.argmin(np.abs(xx[:, None] - modes[None, :]), axis=1)
    rm = np.array([xx[a == k].mean() for k in range(2)])
    return np.max(np.abs(rm - modes))
we1_errs = {n: round(run_euler_seq(n), 3) for n in [1, 2, 3, 5, 10, 20, 50]}
print("[WE-1 순차시드 스텝별 오차 (44강 WE-2 정합)]", we1_errs)
