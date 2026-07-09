# Lec 41 그림 생성 스크립트 — 강화학습 압축 코스 (MDP·정책경사·advantage·오프라인 RL)
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공용: 3상태 체인 MDP  (0=시작, 1=중간, 2=목표[흡수])
#   행동 a=0(왼/머무름), a=1(오/전진). 목표 도달 시 보상, 그 외 스텝비용.
#   가치반복이 벨만 최적방정식을 반복 적용해 V*로 수렴하는지 본다.
# ============================================================================
nS, nA = 3, 2
gamma = 0.9

# 전이확률 P[s, a, s'] : 전진(a=1)은 0.8로 오른쪽, 0.2로 제자리(미끄러짐)
P = np.zeros((nS, nA, nS))
# 상태 0
P[0, 0] = [1.0, 0.0, 0.0]          # a=0: 제자리
P[0, 1] = [0.2, 0.8, 0.0]          # a=1: 0.8로 전진
# 상태 1
P[1, 0] = [1.0, 0.0, 0.0]          # a=0: 왼쪽으로 되돌아감
P[1, 1] = [0.0, 0.2, 0.8]          # a=1: 0.8로 목표 도달
# 상태 2 (흡수: 어떤 행동이든 제자리)
P[2, 0] = [0.0, 0.0, 1.0]
P[2, 1] = [0.0, 0.0, 1.0]

# 보상 R[s, a]: 목표(2)로 들어가는 전이에 +1, 그 외 스텝비용 -0.04, 흡수상태는 0
R = np.full((nS, nA), -0.04)
R[1, 1] = 0.8 * 1.0 + 0.2 * (-0.04)   # 목표 도달 기대보상
R[0, 1] = -0.04
R[2, :] = 0.0                          # 목표 흡수: 이후 보상 없음

def value_iteration(P, R, gamma, n_iter=60, V0=None):
    V = np.zeros(nS) if V0 is None else V0.copy()
    hist = [V.copy()]
    for _ in range(n_iter):
        Q = R + gamma * np.einsum('sap,p->sa', P, V)   # 벨만 백업
        V = Q.max(axis=1)                              # 최적 백업(max_a)
        hist.append(V.copy())
    pi = (R + gamma * np.einsum('sap,p->sa', P, V)).argmax(axis=1)
    return V, pi, np.array(hist)

Vstar, pi_star, Vhist = value_iteration(P, R, gamma)

# 수렴 지표: 매 반복의 ||V_k - V*||_inf (선형 수렴, 비율 ~ gamma)
conv = np.max(np.abs(Vhist - Vstar[None, :]), axis=1)
# 관측된 수축률 (연속 오차 비)
ratios = conv[5:15] / conv[4:14]
observed_contraction = np.mean(ratios[np.isfinite(ratios) & (conv[4:14] > 1e-9)])

# ============================================================================
# 그림 2: 가치반복 수렴 곡선 + 선형2차 특수케이스(LQR 해석해와 일치)
# ============================================================================
# --- (b)용 선형2차 MDP: x_{k+1}=a x + b u + noise, 비용 q x^2 + r u^2 ---
#   이산시간 LQR: 가치함수 V(x)=Px^2, P는 이산 대수 Riccati 방정식의 해.
#   여기서 "가치반복(리카티 반복)"이 solve_discrete_are 해석해로 수렴함을 보인다.
a_d, b_d = 1.0, 1.0
q, r = 1.0, 0.5
A_ = np.array([[a_d]]); B_ = np.array([[b_d]])
Q_ = np.array([[q]]);   R_ = np.array([[r]])
P_are = solve_discrete_are(A_, B_, Q_, R_)[0, 0]     # 해석해 (cost-to-go 곡률)

# 리카티 반복(=선형2차에서의 가치반복): P_{k+1} = q + a^2 P - (a b P)^2/(r + b^2 P)
def riccati_iter(n_iter=40):
    Pk = 0.0
    hist = [Pk]
    for _ in range(n_iter):
        Pk = q + a_d**2 * Pk - (a_d * b_d * Pk)**2 / (r + b_d**2 * Pk)
        hist.append(Pk)
    return np.array(hist)
Phist = riccati_iter()
ricc_conv = np.abs(Phist - P_are)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 표 형태 MDP 가치반복: 오차 ||V_k - V*||_inf 수렴 + γ^k 상계 포락선
its = np.arange(len(conv))
ax1.semilogy(its, conv + 1e-16, 'C0o-', ms=4, lw=1.8,
             label='가치반복 오차 $\\|V_k-V^*\\|_\\infty$')
env = conv[1] * gamma**(its - 1)      # γ^k 상계 포락선(수축사상 이론값)
ax1.semilogy(its[1:], env[1:], 'k--', lw=1.4, label=f'$\\gamma^k$ 상계 (γ={gamma})')
ax1.set_xlabel('가치반복 횟수 k')
ax1.set_ylabel('$\\|V_k - V^*\\|_\\infty$ (log)')
ax1.set_xlim(0, 30); ax1.set_ylim(1e-14, 3)
ax1.set_title('(a) 3상태 체인 MDP — 벨만 백업은 γ-수축사상\n'
              f'V*={np.round(Vstar,3).tolist()}, π*=(전진,전진,·)')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=8.5, loc='upper right')

# (b) 선형2차 특수케이스: 리카티 반복이 LQR 해석해 P로 수렴
axb = ax2
axb.semilogy(np.arange(len(ricc_conv)), ricc_conv + 1e-16, 'C3o-', ms=4, lw=1.8,
             label='리카티 반복 오차 $|P_k - P^*|$')
axb.axhline(1e-12, color='gray', ls=':', lw=1)
axb.set_xlabel('반복 횟수 k')
axb.set_ylabel('$|P_k - P^*|$ (log)')
axb.set_xlim(0, 30)
axb.set_title('(b) 선형2차 MDP = LQR — 같은 벨만, 해석해 존재\n'
              f'가치반복→ $V(x)=P^*x^2$, $P^*$={P_are:.4f} (ARE 해)')
axb.grid(alpha=0.3, which='both'); axb.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + 'fig2_value_iteration.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 1: MDP 에이전트-환경 폐루프 (0강 회수) — 개념도
# ============================================================================
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 6)

# 에이전트(정책) 박스 — 연산(파랑)
ax.add_patch(plt.Rectangle((0.6, 3.4), 3.4, 1.9, fc='#e1eefb', ec='#2c6fb0', lw=2.2))
ax.text(2.3, 4.75, '에이전트 (정책 $\\pi_\\theta$)', ha='center', fontsize=11, weight='bold', color='#0c3a63')
ax.text(2.3, 4.15, '상태 $s$ → 행동 $a$\n(37–40강: BC/생성 정책)', ha='center', fontsize=8.5, color='#0c3a63')

# 환경(플랜트) 박스 — 물리(갈색)
ax.add_patch(plt.Rectangle((6.0, 3.4), 3.4, 1.9, fc='#efe4d8', ec='#9a6a3a', lw=2.2))
ax.text(7.7, 4.75, '환경 (MDP 동역학)', ha='center', fontsize=11, weight='bold', color='#4a2f14')
ax.text(7.7, 4.15, '전이 $P(s\'|s,a)$\n보상 $r(s,a)$', ha='center', fontsize=8.5, color='#4a2f14')

# 위쪽 화살표: 행동 a (에이전트 → 환경)
ax.annotate('', xy=(6.0, 4.7), xytext=(4.0, 4.7),
            arrowprops=dict(arrowstyle='-|>', color='#2c6fb0', lw=2.2))
ax.text(5.0, 5.05, '행동 $a_t$', ha='center', fontsize=10, color='#2c6fb0')

# 아래쪽 화살표: 상태 s' + 보상 r (환경 → 에이전트)
ax.annotate('', xy=(4.0, 3.95), xytext=(6.0, 3.95),
            arrowprops=dict(arrowstyle='-|>', color='#9a6a3a', lw=2.2))
ax.text(5.0, 3.55, '상태 $s_{t+1}$,  보상 $r_t$', ha='center', fontsize=10, color='#9a6a3a')

# 목적함수 박스 (아래)
ax.add_patch(plt.Rectangle((2.0, 0.7), 6.0, 1.4, fc='#f3eede', ec='#8a7a4a', lw=1.8))
ax.text(5.0, 1.62, '목표: 누적 할인보상 최대화', ha='center', fontsize=10, weight='bold', color='#3a3320')
ax.text(5.0, 1.02, r'$J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_t \gamma^t r_t\right]$'
                   '     (18강 cost-to-go의 부호 반전판)',
        ha='center', fontsize=10.5, color='#3a3320')

# 폐루프 강조 곡선 화살표
ax.annotate('', xy=(2.3, 3.4), xytext=(2.3, 2.1),
            arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.4, ls='--',
                            connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(7.7, 2.1), xytext=(7.7, 3.4),
            arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.4, ls='--'))

ax.set_title('그림 1. MDP = 에이전트–환경 폐루프 (0강의 감지→판단→제어→…→감지 루프의 학습판)',
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT + 'fig1_mdp_loop.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: baseline(=V)의 경사 추정 분산 감소  (REINFORCE with baseline)
#   2행동 밴딧: 보상 r(a) 확률적. 정책경사 추정치의 분산을 baseline 유/무로 비교.
# ============================================================================
rng = np.random.default_rng(0)

# 밴딧: 행동 0/1, 참 기대보상 mu=[1.0, 1.2] (둘 다 큰 상수 오프셋)
#   baseline 없이 g = (r) * d_logpi,  baseline 있으면 g=(r-b)*d_logpi, b=E[r]
mu = np.array([1.0, 1.2])
noise_sd = 0.3
# softmax 정책, 로짓 z (theta). d/dz logpi(a) for a taken:
#   grad_z_k logpi(a) = 1{a=k} - pi_k
z = np.array([0.0, 0.2])
pi = np.exp(z) / np.exp(z).sum()

def sample_grad(use_baseline, b_val):
    a = rng.choice(2, p=pi)
    r = mu[a] + noise_sd * rng.standard_normal()
    dlog = -pi.copy()
    dlog[a] += 1.0                       # ∇_z logπ(a)
    adv = (r - b_val) if use_baseline else r
    return adv * dlog                    # 정책경사 몬테카를로 추정치 (2벡터)

N_SAMP = 4000
b_opt = float((pi * mu).sum())           # baseline = E[r] = V (기대보상)
g_nobase = np.array([sample_grad(False, 0.0) for _ in range(N_SAMP)])
# 동일 시드 흐름을 위해 rng 재설정(공정 비교: 같은 표본 경로)
rng = np.random.default_rng(0)
g_base = np.array([sample_grad(True, b_opt) for _ in range(N_SAMP)])

# 각 성분(로짓 0에 대한 경사)의 분산 비교 — 기댓값은 같고 분산만 다름
var_nobase = g_nobase[:, 0].var()
var_base = g_base[:, 0].var()
mean_nobase = g_nobase[:, 0].mean()
mean_base = g_base[:, 0].mean()
var_ratio = var_nobase / var_base

# baseline b를 스윕하며 분산이 b=V에서 최소 부근인지 (실제 최적 b는 가중식이나 V가 좋은 근사)
bs = np.linspace(-0.5, 2.5, 41)
def grad_variance_for_b(b_val, n=6000):
    r = np.random.default_rng(1)
    gs = []
    for _ in range(n):
        a = r.choice(2, p=pi)
        rew = mu[a] + noise_sd * r.standard_normal()
        dlog = -pi.copy(); dlog[a] += 1.0
        gs.append((rew - b_val) * dlog[0])
    return np.var(gs)
var_curve = np.array([grad_variance_for_b(b) for b in bs])
b_argmin = bs[np.argmin(var_curve)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 경사 추정치 히스토그램 (baseline 유/무) — 같은 평균, 다른 폭
ax1.hist(g_nobase[:, 0], bins=45, alpha=0.55, color='C3',
         label=f'baseline 없음 (분산 {var_nobase:.3f})', density=True)
ax1.hist(g_base[:, 0], bins=45, alpha=0.6, color='C0',
         label=f'baseline b=V (분산 {var_base:.3f})', density=True)
ax1.axvline(mean_nobase, color='C3', ls='--', lw=1.2)
ax1.axvline(mean_base, color='C0', ls='--', lw=1.2)
ax1.set_xlabel('정책경사 추정치 $\\hat g_0$ (로짓 0 성분)')
ax1.set_ylabel('밀도')
ax1.set_title(f'(a) baseline이 분산을 {var_ratio:.1f}배 줄인다\n'
              f'(평균은 거의 동일: {mean_nobase:+.3f} vs {mean_base:+.3f} — 불편 유지)')
ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5)

# (b) baseline 값에 따른 경사 분산 — b=V 부근에서 최소
ax2.plot(bs, var_curve, 'C2-', lw=2)
ax2.axvline(b_opt, color='k', ls=':', lw=1.5, label=f'$b=V=\\mathbb{{E}}[r]$={b_opt:.3f}')
ax2.plot(b_argmin, var_curve.min(), 'C2*', ms=15,
         label=f'수치 최소 위치 b≈{b_argmin:.2f}')
ax2.set_xlabel('baseline 값 $b$')
ax2.set_ylabel('정책경사 분산 $\\mathrm{Var}[\\hat g_0]$')
ax2.set_title('(b) 분산은 $b=V$ 부근에서 최소\n'
              'advantage $A=r-V$ = "상대 이득"이 최적 근처')
ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT + 'fig3_baseline_variance.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 4: 온라인 vs 오프라인 RL — 분포이동과 가치 과대평가
#   오프라인 데이터가 상태공간의 일부만 덮을 때, 순진한 Q학습이 미방문 행동을
#   과대평가하는 것을 1D 토이로 시각화. 보수적 방법(데이터 근처로 제약)의 필요성.
# ============================================================================
# 행동축 위 참 Q값(볼록한 단봉) — 데이터는 왼쪽 일부만 덮음
a_grid = np.linspace(-3, 3, 300)
Q_true = -0.5 * (a_grid - 0.4)**2 + 1.0            # 참 Q(a): a=0.4에서 최대

# 오프라인 데이터: 행동이 [-2, 0.2] 구간에만 분포 (behavior 정책의 지지)
data_mask = (a_grid >= -2.0) & (a_grid <= 0.2)

# 순진한 함수근사 Q_hat: 데이터 구간은 잘 맞지만, 데이터 밖으로 '외삽 오차'가 커짐
#   미방문 영역(오른쪽)에서 참값보다 높게 튀는(과대평가) 다항 외삽을 흉내낸다.
#   순진한 오프라인 Q학습은 max_a Q̂ 를 백업에 쓰는데, 미방문 a에서 Q̂이 크게 튀면
#   그 과대평가가 부트스트랩으로 누적된다 — 4차 외삽으로 그 발산을 흉내낸다.
rng = np.random.default_rng(3)
a_data = a_grid[data_mask]
Q_data = Q_true[data_mask] + 0.05 * rng.standard_normal(data_mask.sum())
# 4차 다항 적합 → 데이터 밖(오른쪽)에서 과대평가로 발산
coef = np.polyfit(a_data, Q_data, 4)
Q_hat = np.polyval(coef, a_grid)

# 과대평가 지점: 데이터 밖(오른쪽)에서 Q_hat이 참 최대값을 초과하는 argmax
a_true_best = a_grid[np.argmax(Q_true)]
a_hat_best = a_grid[np.argmax(Q_hat)]
overest = Q_hat.max() - Q_true.max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 분포이동·과대평가
ax1.plot(a_grid, Q_true, 'k-', lw=2, label='참 $Q(s,a)$')
ax1.plot(a_grid, Q_hat, 'C3--', lw=2, label='순진한 오프라인 $\\hat Q$ (외삽)')
ax1.axvspan(-2.0, 0.2, color='C0', alpha=0.15, label='데이터 지지 (behavior 분포)')
ax1.axvline(a_true_best, color='k', ls=':', lw=1.2)
ax1.axvline(a_hat_best, color='C3', ls=':', lw=1.2)
ax1.annotate(f'과대평가\n+{overest:.2f}', xy=(a_hat_best, Q_hat.max()),
             xytext=(1.3, Q_true.max()+0.2), fontsize=9, color='C3',
             arrowprops=dict(arrowstyle='->', color='C3'))
ax1.set_xlabel('행동 $a$'); ax1.set_ylabel('$Q(s,a)$')
ax1.set_ylim(-2.0, Q_hat.max()+0.6)
ax1.set_title('(a) 오프라인 RL의 함정 — 미방문 행동 과대평가\n'
              'argmax가 데이터 밖으로 샘 → 보수적 제약 필요')
ax1.grid(alpha=0.3); ax1.legend(fontsize=8, loc='lower center')

# (b) 온라인 vs 오프라인 데이터 흐름 개념도
ax2.axis('off')
ax2.set_xlim(0, 10); ax2.set_ylim(0, 6)
ax2.text(5, 5.6, '온라인 RL: 정책이 스스로 탐색 (로봇에 위험·고비용)',
         ha='center', fontsize=9.5, weight='bold', color='#2c6fb0')
# 온라인 루프
ax2.add_patch(plt.Rectangle((0.8, 3.7), 2.6, 1.1, fc='#e1eefb', ec='#2c6fb0', lw=1.8))
ax2.text(2.1, 4.25, '정책 $\\pi_\\theta$', ha='center', fontsize=9)
ax2.add_patch(plt.Rectangle((6.0, 3.7), 2.8, 1.1, fc='#efe4d8', ec='#9a6a3a', lw=1.8))
ax2.text(7.4, 4.25, '환경 (실제 로봇)', ha='center', fontsize=9)
ax2.annotate('', xy=(6.0, 4.5), xytext=(3.4, 4.5),
             arrowprops=dict(arrowstyle='-|>', color='#2c6fb0', lw=1.8))
ax2.text(4.7, 4.72, '탐색 행동', ha='center', fontsize=8, color='#2c6fb0')
ax2.annotate('', xy=(3.4, 3.95), xytext=(6.0, 3.95),
             arrowprops=dict(arrowstyle='-|>', color='#9a6a3a', lw=1.8))
ax2.text(4.7, 3.55, '새 경험 (온라인)', ha='center', fontsize=8, color='#9a6a3a')

ax2.text(5, 2.75, '오프라인 RL: 고정 로그 데이터로만 (45강 RECAP)',
         ha='center', fontsize=9.5, weight='bold', color='#c0392b')
ax2.add_patch(plt.Rectangle((0.8, 0.9), 3.0, 1.1, fc='#f7dede', ec='#c0392b', lw=1.8))
ax2.text(2.3, 1.45, '고정 데이터셋 $\\mathcal{D}$\n(과거 teleop·경험)', ha='center', fontsize=8.5)
ax2.add_patch(plt.Rectangle((6.0, 0.9), 2.8, 1.1, fc='#e1eefb', ec='#2c6fb0', lw=1.8))
ax2.text(7.4, 1.45, '정책 학습\n(상호작용 없음)', ha='center', fontsize=8.5)
ax2.annotate('', xy=(6.0, 1.45), xytext=(3.8, 1.45),
             arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=1.8))
ax2.text(4.9, 1.68, '읽기만', ha='center', fontsize=8, color='#c0392b')
ax2.set_title('(b) 온라인은 탐색 루프가 닫힘 / 오프라인은 열려 있음\n'
              '로봇은 대개 오프라인·사후학습 (안전·샘플효율)', fontsize=10)
fig.tight_layout()
fig.savefig(OUT + 'fig4_online_vs_offline.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문과 정확히 일치해야 함
# ============================================================================
print("=== 그림 파일 4개 생성 완료 ===")
print(f"[VI] V* = {np.round(Vstar, 4).tolist()},  π* = {pi_star.tolist()} (0=머무름,1=전진)")
print(f"[VI 수렴] 관측 수축률 ≈ {observed_contraction:.4f} (이론 γ={gamma})")
print(f"[VI 오차] k=10 오차 {conv[10]:.2e}, k=20 오차 {conv[20]:.2e}")
print(f"[LQR] 이산 ARE 해 P* = {P_are:.6f}, 리카티반복 k=20 오차 {ricc_conv[20]:.2e}")
print(f"[LQR] 최적게인 K = R^-1... u=-Kx, K={(b_d*P_are)/(r+b_d**2*P_are):.6f}")
print(f"[baseline] 분산: 없음 {var_nobase:.4f} vs b=V {var_base:.4f} → {var_ratio:.2f}배 감소")
print(f"[baseline] 평균(불편): 없음 {mean_nobase:+.4f} vs b=V {mean_base:+.4f} (거의 동일)")
print(f"[baseline] b=V={b_opt:.4f}, 분산최소 수치위치 b≈{b_argmin:.3f}")
print(f"[offline] 참 최적행동 a*={a_true_best:.3f}, 순진 Q̂ argmax={a_hat_best:.3f} (데이터 밖으로 샘)")
print(f"[offline] Q 과대평가량 = +{overest:.4f}")
