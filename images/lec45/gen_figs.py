# Lec 45 그림 생성 스크립트 — π 패밀리 II (π0.5 / KI / RECAP / π0.7)
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델 다운로드/GPU 없음.
#   fig2: Knowledge Insulation — stop-gradient 유무로 공유 백본 파라미터 이동/표현 왜곡
#   fig3: advantage-weighted regression(AWR) — 온도 β별 보상모드 집중 (BC 대비)
#   fig4: 계층 추론 — 같은 백본이 고수준 서브골로 조건화되면 저수준 flow 청크 분포가 갈린다
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# fig2. Knowledge Insulation — 공유 백본 2층 토이
# ----------------------------------------------------------------------------
# 구조(0강 E1의 함수 합성을 최소로 재현):
#   입력 x → [백본 W1,W2] → 표현 h(2차원) → 두 헤드가 h를 공유
#     · 언어/FAST 헤드(작고 '의미'를 담음, 얕은 gradient)
#     · flow expert 헤드(크고 초기 noisy, gradient가 큼)
# 딜레마: expert의 큰 gradient가 백본을 흔들면 백본이 담은 '의미 표현'이 왜곡된다.
# KI = expert→백본 경로에 stop-gradient. 백본은 언어/FAST 손실로만 학습.
#
# 우리는 실제 backprop을 numpy로 손수 구현해 두 조건을 '동일 초기값·동일 데이터'로 비교한다.
# ============================================================================
rng = np.random.default_rng(0)

N, Din, Dh = 256, 4, 2           # 표본 256, 입력 4차원, 표현(백본 출력) 2차원
X = rng.standard_normal((N, Din))

# '의미 있는' 표현 방향: 백본이 담아야 할 목표 표현 H* (언어/FAST 헤드가 요구)
#   웹에서 배운 의미 = 이 방향으로 h를 정렬하는 것.
W_star = rng.standard_normal((Din, Dh))
H_star = X @ W_star                          # 백본이 담아야 할 이상적 표현

# flow expert가 요구하는 회귀 타깃: '의미와 무관한' 방향의 액션 + 큰 noise.
#   핵심 장치 — expert 타깃은 X의 '다른 방향'(직교에 가까운 성분)에서 나온다.
#   그래서 expert gradient는 백본을 의미방향(H*)에서 '떼어내려' 한다.
#   차단하지 않으면 백본이 이 방향으로 끌려가 웹 지식(H* 정렬)이 손상된다.
W_act = rng.standard_normal((Din, 3)) * 2.0
Y_act = X @ W_act + 3.0*rng.standard_normal((N, 3))        # X의 다른 방향 + noisy → 큰 gradient

def init_params():
    r = np.random.default_rng(42)            # 두 조건 동일 초기값
    W1 = r.standard_normal((Din, Dh)) * 0.3
    W2 = r.standard_normal((Dh, Dh)) * 0.3
    Wlang = r.standard_normal((Dh, Dh)) * 0.3
    Wexp = r.standard_normal((Dh, 3)) * 0.3
    return W1, W2, Wlang, Wexp

def backbone(x, W1, W2):
    z1 = x @ W1
    a1 = np.tanh(z1)                          # 은닉 활성
    h = a1 @ W2                               # 표현 h
    return z1, a1, h

var_Hstar = np.mean(H_star**2)               # H* 분산(중심0) — 정규화 기준

def semantic_fit(W1, W2):
    # 표현 h에서 H*를 '즉석 최소제곱'으로 얼마나 복원할 수 있나 (R² 형태, 0~1).
    #   = 표현이 웹 의미를 담고 있는가의 헤드-무관 척도.
    _, _, h = backbone(X, W1, W2)
    B, *_ = np.linalg.lstsq(h, H_star, rcond=None)   # h→H* 최적 선형사상
    resid = H_star - h @ B
    return 1.0 - np.mean(resid**2)/var_Hstar

def train(insulate, steps=400, lr=0.05, lam_lang=1.0, lam_exp=1.0):
    W1, W2, Wlang, Wexp = init_params()
    W1_0, W2_0 = W1.copy(), W2.copy()
    hist = {'lang': [], 'exp': [], 'drift': [], 'align': []}
    for _ in range(steps):
        z1, a1, h = backbone(X, W1, W2)
        # --- 언어/FAST 헤드: 표현을 H_star에 정렬 (의미 보존 손실) ---
        pred_lang = h @ Wlang
        r_lang = pred_lang - H_star
        L_lang = np.mean(r_lang**2)
        # --- flow expert 헤드: 의미와 다른 방향의 noisy 액션 회귀 (큰 gradient) ---
        pred_exp = h @ Wexp
        r_exp = pred_exp - Y_act
        L_exp = np.mean(r_exp**2)

        # 헤드 gradient
        gWlang = 2/N * h.T @ r_lang
        gWexp = 2/N * h.T @ r_exp
        # 표현 h로 흘러드는 gradient (헤드→백본)
        dh_lang = 2/N * (r_lang @ Wlang.T)
        dh_exp = 2/N * (r_exp @ Wexp.T)
        # KI: expert→백본 경로 차단 (stop-gradient). 백본은 lang gradient만 받는다.
        dh_back = lam_lang*dh_lang + (0.0 if insulate else lam_exp)*dh_exp

        # 백본 backprop (h = tanh(x W1) W2)
        gW2 = a1.T @ dh_back
        da1 = dh_back @ W2.T
        dz1 = da1 * (1 - a1**2)
        gW1 = X.T @ dz1

        # 파라미터 갱신 (헤드는 둘 다 자기 손실로 갱신 — expert 헤드는 계속 학습됨)
        W1 -= lr*gW1
        W2 -= lr*gW2
        Wlang -= lr*lam_lang*gWlang
        Wexp -= lr*lam_exp*gWexp

        # 기록: 백본 파라미터 이동량, 표현-의미 정렬(복원 R²)
        drift = np.sqrt(np.sum((W1-W1_0)**2) + np.sum((W2-W2_0)**2))
        align = semantic_fit(W1, W2)
        hist['lang'].append(L_lang); hist['exp'].append(L_exp)
        hist['drift'].append(drift); hist['align'].append(align)
    return {k: np.array(v) for k, v in hist.items()}, (W1, W2)

hist_ki, _ = train(insulate=True)
hist_no, _ = train(insulate=False)

drift_ki = hist_ki['drift'][-1]
drift_no = hist_no['drift'][-1]
align_ki = hist_ki['align'][-1]
align_no = hist_no['align'][-1]
lang_ki = hist_ki['lang'][-1]
lang_no = hist_no['lang'][-1]
drift_ratio = drift_no / drift_ki

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
steps = np.arange(len(hist_ki['drift']))
# (a) 백본 파라미터 이동량
ax1.plot(steps, hist_no['drift'], 'C3-', lw=2,
         label=f'gradient 차단 없음 (expert가 백본을 흔듦)\n최종 이동 {drift_no:.2f}')
ax1.plot(steps, hist_ki['drift'], 'C0-', lw=2,
         label=f'KI: stop-gradient (백본은 FAST만)\n최종 이동 {drift_ki:.2f}')
ax1.set_xlabel('학습 스텝'); ax1.set_ylabel('백본 파라미터 이동 ‖ΔW1, ΔW2‖')
ax1.set_title(f'(a) 공유 백본이 초기값에서 얼마나 밀려났나\n'
              f'expert gradient가 백본을 {drift_ratio:.1f}배 더 밀어낸다')
ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5, loc='upper left')

# (b) 표현-의미 정렬 (웹 지식 보존 대리 지표)
ax2.plot(steps, hist_no['align'], 'C3-', lw=2,
         label=f'차단 없음: 표현 왜곡\n최종 정렬 {align_no:.3f}')
ax2.plot(steps, hist_ki['align'], 'C0-', lw=2,
         label=f'KI: 의미 표현 보존\n최종 정렬 {align_ki:.3f}')
ax2.axhline(1.0, color='gray', ls=':', lw=1, label='완전 정렬 (의미 그대로)')
ax2.set_xlabel('학습 스텝'); ax2.set_ylabel('표현 h와 의미방향 H* 정렬 (평균 코사인)')
ax2.set_title('(b) "웹에서 배운 의미"가 유지되는가\n'
              'expert gradient는 표현을 액션 회귀 쪽으로 끌어 왜곡')
ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5, loc='lower left')
fig.tight_layout()
fig.savefig(OUT + 'fig2_knowledge_insulation.png', dpi=140)
plt.close(fig)

# ============================================================================
# fig3. Advantage-Weighted Regression — 온도 β별 보상모드 집중
# ----------------------------------------------------------------------------
# 다봉 액션 데이터(0강 fig2의 정량판):
#   시연 데이터 = 두 모드(A: 저성능, B: 고성능)의 혼합. 순수 BC는 둘을 '평균'해
#   가운데(어느 모드도 아닌 실패 지점)로 수렴하려 한다(mode averaging).
#   RECAP의 advantage 조건화 = AWR류 가중치 exp(A/β)로 '좋은 행동'에 무게를 몰아준다.
#   β→∞: 균일 가중(BC와 동일). β→0: 최고 advantage 행동만 (argmax, greedy).
# 여기서는 가중 평균/가중 분포로 "분포가 보상 있는 모드로 이동"을 정량화한다.
# ============================================================================
rng2 = np.random.default_rng(3)
# 1D 액션 토이: 두 모드
muA, muB = -1.5, 1.5           # A=나쁜 모드, B=좋은 모드
sig = 0.35
nA, nB = 600, 600
aA = muA + sig*rng2.standard_normal(nA)
aB = muB + sig*rng2.standard_normal(nB)
actions = np.concatenate([aA, aB])
# advantage: 좋은 모드(B)가 높음. critic(41강)이 매긴 점수라고 생각.
#   A(a) = 행동이 좋은 모드에 가까울수록 큰 값. 여기선 위치의 선형함수로 단순화.
adv = actions.copy()           # advantage ∝ 액션 위치 (B쪽이 +, A쪽이 −)
adv = (adv - adv.mean())       # 중심화 (baseline 뺀 advantage, 41강)

def awr_weights(beta):
    # exp(A/β) 가중치, 정규화. 수치 안정 위해 max 빼기.
    w = np.exp((adv - adv.max())/beta)
    return w / w.sum()

def weighted_stats(beta):
    w = awr_weights(beta)
    mean = np.sum(w*actions)                          # 가중 평균 = 정책이 향하는 행동
    # 좋은 모드(B, a>0)에 실린 확률질량
    massB = np.sum(w[actions > 0])
    return mean, massB

betas = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
means = np.array([weighted_stats(b)[0] for b in betas])
massBs = np.array([weighted_stats(b)[1] for b in betas])

# 순수 BC = 균일 가중 (β→∞ 극한). 여기선 단순 표본평균.
bc_mean = actions.mean()
bc_massB = np.mean(actions > 0)

# 대표 온도들의 정량치 (본문 인용)
beta_lo, beta_mid, beta_hi = 0.5, 2.0, 100.0
mean_lo, massB_lo = weighted_stats(beta_lo)
mean_mid, massB_mid = weighted_stats(beta_mid)
mean_hi, massB_hi = weighted_stats(beta_hi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 가중 분포 히스토그램: BC vs AWR(작은 β)
bins = np.linspace(-3, 3, 60)
ax1.hist(actions, bins=bins, weights=np.ones_like(actions)/len(actions),
         color='0.75', label=f'시연 데이터 (BC 목표)\n좋은모드 질량 {bc_massB:.2f}·평균 {bc_mean:+.2f}',
         alpha=0.9)
w_mid = awr_weights(beta_mid)
ax1.hist(actions, bins=bins, weights=w_mid, color='C0', alpha=0.6,
         label=f'AWR β={beta_mid} 가중분포\n좋은모드 질량 {massB_mid:.2f}·평균 {mean_mid:+.2f}')
w_lo = awr_weights(beta_lo)
ax1.hist(actions, bins=bins, weights=w_lo, color='C3', alpha=0.55,
         label=f'AWR β={beta_lo} (더 공격적)\n좋은모드 질량 {massB_lo:.2f}·평균 {mean_lo:+.2f}')
ax1.axvline(muA, color='k', ls=':', lw=1); ax1.axvline(muB, color='k', ls=':', lw=1)
ax1.text(muA, ax1.get_ylim()[1]*0.9, '모드 A\n(나쁨)', ha='center', fontsize=8.5)
ax1.text(muB, ax1.get_ylim()[1]*0.9, '모드 B\n(좋음)', ha='center', fontsize=8.5)
ax1.set_xlabel('액션 값 (advantage ∝ 위치)'); ax1.set_ylabel('확률질량')
ax1.set_title('(a) advantage 가중이 분포를 좋은 모드로 옮긴다\n'
              'BC(회색)는 두 모드 균등 → AWR은 B로 질량 이동')
ax1.legend(fontsize=7.8, loc='upper left')
ax1.grid(alpha=0.3, axis='y')

# (b) β 스윕: 좋은 모드 질량 & 가중 평균
axb = ax2.twinx()
l1 = ax2.semilogx(betas, massBs, 'C0o-', lw=2, ms=6, label='좋은 모드(B) 질량')
ax2.axhline(bc_massB, color='C0', ls='--', lw=1.2, alpha=0.6)
ax2.text(betas[-1], bc_massB+0.02, f'BC 극한 {bc_massB:.2f}', fontsize=8,
         ha='right', color='C0')
l2 = axb.semilogx(betas, means, 'C3s--', lw=2, ms=6, label='가중 평균 액션')
axb.axhline(muB, color='k', ls=':', lw=1); axb.text(0.22, muB-0.18, 'B 위치', fontsize=8)
ax2.set_xlabel('온도 β (log) — 작을수록 공격적'); ax2.set_ylabel('좋은 모드 질량', color='C0')
axb.set_ylabel('가중 평균 액션', color='C3')
ax2.set_ylim(0.4, 1.02); axb.set_ylim(-0.2, 1.8)
ax2.set_title('(b) 온도 β 스윕: β↓ = greedy(argmax), β↑ = BC\n'
              'β가 "얼마나 좋은 행동만 고를까"의 손잡이')
lines = l1 + l2
ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=8.5, loc='center right')
ax2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(OUT + 'fig3_awr_beta_sweep.png', dpi=140)
plt.close(fig)

# ============================================================================
# fig4. 계층 추론 — 같은 백본, 서브골 조건화가 저수준 flow 분포를 가른다
# ----------------------------------------------------------------------------
# π0.5: 같은 가중치가 (1) 고수준 서브골 텍스트를 예측하고 (2) 그 서브골을 조건으로
#   저수준 flow 액션 청크를 낸다. 서브골이 바뀌면 같은 관측에서도 액션 분포가 갈린다.
# 토이: 같은 상태 s에서 서브골 g∈{집기, 놓기}에 따라 flow가 다른 모드로 수렴.
#   40강/44강 flow 토이 재사용 — 조건 g가 목표 분포(모드 위치)를 결정.
# ============================================================================
rng3 = np.random.default_rng(11)
# 두 서브골이 요구하는 액션 모드 (예: 그리퍼 개폐 방향이 반대)
goal_modes = {'집기 (grasp)': -1.2, '놓기 (release)': +1.2}
sig_g = 0.18

def marginal_velocity_1mode(x, tt, mu):
    # 단일 가우시안 모드로의 조건부 flow 속도장 (44강 fig3 골격의 1모드판)
    a = tt; b = (1-tt)
    post_var = 1.0/(1.0/sig_g**2 + a**2/(b**2+1e-9))
    E_x1 = post_var*(mu/sig_g**2 + a*x/(b**2+1e-9))
    E_x0 = (x - tt*E_x1)/(1-tt+1e-9)
    return E_x1 - E_x0

N_STEPS = 10
n_samp = 300
paths = {}
finals = {}
for g, mu in goal_modes.items():
    x = rng3.standard_normal(n_samp)          # 같은 노이즈 시드 계열
    p = np.zeros((N_STEPS+1, n_samp)); p[0] = x
    for i in range(N_STEPS):
        tt = i/N_STEPS
        x = x + (1.0/N_STEPS)*marginal_velocity_1mode(x, tt, mu)
        p[i+1] = x
    paths[g] = p; finals[g] = x

# 검증 수치: 각 서브골 조건에서 최종 액션 평균이 목표 모드로 갔는가
recov = {g: finals[g].mean() for g in goal_modes}
sep = abs(recov['놓기 (release)'] - recov['집기 (grasp)'])   # 두 조건 분리도

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
tgrid = np.linspace(0, 1, N_STEPS+1)
colors = {'집기 (grasp)': 'C0', '놓기 (release)': 'C1'}
for g in goal_modes:
    for j in range(0, n_samp, 3):
        ax1.plot(tgrid, paths[g][:, j], color=colors[g], lw=0.4, alpha=0.3)
    ax1.plot([], [], color=colors[g], lw=2,
             label=f'서브골="{g}" → 액션 {recov[g]:+.2f}')
for mu in goal_modes.values():
    ax1.axhline(mu, color='k', ls=':', lw=1)
ax1.set_xlabel('flow 시간 t (0=노이즈, 1=액션)'); ax1.set_ylabel('저수준 액션 값')
ax1.set_title(f'(a) 같은 백본·같은 상태, 서브골로 조건화\n'
              f'고수준 텍스트가 저수준 flow 분포를 가른다 (분리 {sep:.2f})')
ax1.legend(fontsize=8.5, loc='center left'); ax1.grid(alpha=0.3); ax1.set_xlim(0, 1)

# (b) 계층 구조 개념도 (한 모델 안의 계층 vs 두 모델)
ax2.axis('off')
ax2.text(0.5, 0.96, 'π0.5: 한 가중치 안의 계층 (43강 ECoT의 구조 승격)',
         ha='center', fontsize=10, weight='bold', transform=ax2.transAxes)
# 백본 박스
ax2.add_patch(plt.Rectangle((0.28, 0.62), 0.44, 0.16, fc='#e1eefb', ec='C0', lw=2,
                            transform=ax2.transAxes))
ax2.text(0.5, 0.70, '공유 백본 (같은 가중치)\nSigLIP + Gemma', ha='center', va='center',
         fontsize=9, transform=ax2.transAxes)
# 고수준 출력
ax2.add_patch(plt.Rectangle((0.06, 0.36), 0.40, 0.15, fc='#f3eede', ec='#8a7a4a', lw=2,
                            transform=ax2.transAxes))
ax2.text(0.26, 0.435, '① 고수준: 서브골 텍스트\n(AR 예측) "베개를 집어라"',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes)
# 저수준 출력
ax2.add_patch(plt.Rectangle((0.54, 0.36), 0.40, 0.15, fc='#f7dede', ec='C3', lw=2,
                            transform=ax2.transAxes))
ax2.text(0.74, 0.435, '② 저수준: flow 액션 청크\n(서브골로 조건화)',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes)
ax2.annotate('', xy=(0.26, 0.51), xytext=(0.40, 0.62), transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax2.annotate('', xy=(0.74, 0.51), xytext=(0.60, 0.62), transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax2.annotate('', xy=(0.60, 0.435), xytext=(0.46, 0.435), transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
ax2.text(0.53, 0.40, '조건', ha='center', fontsize=8, color='C3', transform=ax2.transAxes)
ax2.text(0.5, 0.18, '48강 상용 진영(Helix, Gemini)은 같은 계층을 모델 두 개로.\n'
                    'π0.5는 그 계층을 가중치 하나 안에 접었다.',
         ha='center', va='center', fontsize=8.5, transform=ax2.transAxes,
         bbox=dict(boxstyle='round', fc='white', ec='gray'))
fig.tight_layout()
fig.savefig(OUT + 'fig4_hierarchical_inference.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문과 반드시 일치
# ============================================================================
print("=== lec45 그림 3개 생성 완료 (fig2/fig3/fig4) ===")
print(f"[KI 백본이동] 차단없음 {drift_no:.3f} vs KI {drift_ki:.3f} "
      f"→ expert gradient가 백본을 {drift_ratio:.2f}배 더 밀어냄")
print(f"[KI 표현정렬] 차단없음 {align_no:.4f} vs KI {align_ki:.4f} "
      f"(1.0=의미 그대로; 차단없음이 표현을 왜곡)")
print(f"[KI 언어손실] 차단없음 {lang_no:.4f} vs KI {lang_ki:.4f} "
      f"(백본이 의미 손실을 얼마나 낮추나)")
print(f"[AWR β 스윕] betas={betas.tolist()}")
print(f"   좋은모드 질량 = {[round(m,3) for m in massBs]}")
print(f"   가중 평균    = {[round(m,3) for m in means]}")
print(f"[AWR 대표치] β={beta_lo}: 질량 {massB_lo:.3f}·평균 {mean_lo:+.3f} | "
      f"β={beta_mid}: 질량 {massB_mid:.3f}·평균 {mean_mid:+.3f} | "
      f"β={beta_hi}: 질량 {massB_hi:.3f}·평균 {mean_hi:+.3f}")
print(f"[AWR BC극한] 균일가중 좋은모드 질량 {bc_massB:.3f}·평균 {bc_mean:+.3f} "
      f"(β→∞와 일치해야: β={beta_hi} 질량 {massB_hi:.3f})")
print(f"[계층추론] 서브골 조건 최종 액션: 집기 {recov['집기 (grasp)']:+.3f}, "
      f"놓기 {recov['놓기 (release)']:+.3f} (목표 {goal_modes['집기 (grasp)']:+.1f}/"
      f"{goal_modes['놓기 (release)']:+.1f}), 분리도 {sep:.3f}")

# ----------------------------------------------------------------------------
# WE 손계산 검증: AWR 가중치를 4개 행동으로 손계산 (본문 WE와 일치)
# ----------------------------------------------------------------------------
print("\n--- WE 손계산 검증 (AWR 4행동 토이) ---")
A_we = np.array([-2.0, 0.0, 1.0, 2.0])     # 4개 행동의 advantage
a_we = np.array([-1.0, 0.2, 0.8, 1.5])     # 대응 액션 값
for beta in [0.5, 1.0, 5.0]:
    w = np.exp((A_we - A_we.max())/beta); w /= w.sum()
    mean = np.sum(w*a_we)
    print(f"  β={beta}: 가중치={np.round(w,4).tolist()} → 가중평균 액션={mean:+.4f}")
# β=1.0 손계산 상세 (본문에 노출)
beta = 1.0
raw = np.exp(A_we - A_we.max())            # exp(A - Amax)
Z = raw.sum()
w1 = raw / Z
print(f"  [β=1 손계산] exp(A-Amax)={np.round(raw,4).tolist()}, Z={Z:.4f}, "
      f"w={np.round(w1,4).tolist()}, 가중평균={np.sum(w1*a_we):+.4f}")
