# Lec 54 그림 생성 스크립트 — 학습된 world model (맛보기)
# 실행: python3 gen_figs.py  (이 디렉토리에서)
# 순수 numpy/scipy/matplotlib. 결정적(시드 고정). numpy 1.26 / scipy 1.15 / matplotlib 3.5.
#
# 이 토이는 "학습된 전이함수 f_theta"의 개념을 CPU numpy로 재현한다.
# 실제 V-JEPA 2 / Cosmos / DreamGen 같은 대형 모델이 아니다 — 개념 재현용.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = os.path.dirname(os.path.abspath(__file__))
C = dict(true='tab:green', wm='tab:red', onestep='tab:blue',
         latent='tab:blue', pixel='tab:red', data='tab:orange')

# ============================================================
# 공통 참 시스템 (0강의 물리 f): 감쇠 진자형 2차 선형계 + 행동
#   s_{t+1} = A s_t + B a_t     (참 전이함수, 우리가 "모른다"고 가정)
#   world model은 이 A,B 를 데이터로 추정한다 → 시스템 식별(60강)의 축소판.
# ============================================================
rng = np.random.default_rng(0)

dt = 0.1
# 참 동역학: 감쇠 스프링 (위치, 속도), 행동=힘
A_true = np.array([[1.0, dt],
                   [-2.0*dt, 1.0 - 0.30*dt]])   # 고유값 |lambda|<1 안정
B_true = np.array([[0.0],
                   [dt]])
ns, na = 2, 1

def true_step(s, a):
    return A_true @ s + B_true @ a

def rollout_true(s0, actions):
    s = s0.copy(); traj = [s.copy()]
    for a in actions:
        s = true_step(s, a); traj.append(s.copy())
    return np.array(traj)

# ------------------------------------------------------------
# world model 학습: 관측 (s_t, a_t, s_{t+1}) 로 A_hat, B_hat 를 최소제곱 추정.
#   유한 데이터 + 관측 노이즈 때문에 f_theta 는 참 f 와 약간 다르다.
#   이 미세한 1스텝 오차가 롤아웃에서 누적된다(37강 compounding error).
# ------------------------------------------------------------
def collect_data(n, noise_std, seed):
    r = np.random.default_rng(seed)
    S = r.standard_normal((n, ns))               # 상태 표본
    Acs = r.standard_normal((n, na))             # 행동 표본
    Snext = np.stack([true_step(S[i], Acs[i]) for i in range(n)])
    Snext = Snext + noise_std * r.standard_normal(Snext.shape)   # 관측 노이즈
    return S, Acs, Snext

def fit_wm(S, Acs, Snext):
    # [A_hat B_hat] = Snext^T  (X^+),  X=[s;a]
    X = np.hstack([S, Acs])                       # (n, ns+na)
    theta, *_ = np.linalg.lstsq(X, Snext, rcond=None)   # (ns+na, ns)
    theta = theta.T                               # (ns, ns+na)
    return theta[:, :ns], theta[:, ns:]           # A_hat, B_hat

def wm_step(Ah, Bh, s, a):
    return Ah @ s + Bh @ a

def rollout_wm(Ah, Bh, s0, actions):
    s = s0.copy(); traj = [s.copy()]
    for a in actions:
        s = wm_step(Ah, Bh, s, a); traj.append(s.copy())
    return np.array(traj)

# 표준 데이터셋으로 하나 학습(본문 WE-1 기준값)
N_DATA, NOISE = 200, 0.02
S, Acs, Snext = collect_data(N_DATA, NOISE, seed=1)
A_hat, B_hat = fit_wm(S, Acs, Snext)
onestep_param_err = np.linalg.norm(np.hstack([A_hat, B_hat]) - np.hstack([A_true, B_true]))
print("== WE-1: 학습된 f_theta ==")
print(f"  파라미터 오차 ||[A_hat B_hat]-[A B]||_F = {onestep_param_err:.5f}  (n={N_DATA}, noise={NOISE})")
print("  A_hat =", np.array2string(A_hat, precision=4))
print("  A_true=", np.array2string(A_true, precision=4))

# ------------------------------------------------------------
# 1스텝 정확 vs N스텝 드리프트: 같은 행동열을 참/WM 에 롤아웃
# ------------------------------------------------------------
H = 40
a_seq = 0.6*np.sin(0.25*np.arange(H))[:, None]    # 결정적 행동열
s0 = np.array([1.0, 0.0])
tr_true = rollout_true(s0, a_seq)
tr_wm = rollout_wm(A_hat, B_hat, s0, a_seq)

# teacher-forced 1스텝 예측(매 스텝 참 상태에서 한 스텝만 WM 예측) — 오차가 작고 누적 안 됨
onestep_err = []
for t in range(H):
    pred = wm_step(A_hat, B_hat, tr_true[t], a_seq[t])
    onestep_err.append(np.linalg.norm(pred - tr_true[t+1]))
onestep_err = np.array(onestep_err)

rollout_err = np.linalg.norm(tr_wm - tr_true, axis=1)   # free-run 롤아웃 오차(누적)

print(f"  1스텝(teacher-forced) 평균 오차 = {onestep_err.mean():.5f}")
print(f"  롤아웃 오차: H=1 {rollout_err[1]:.5f} | H=10 {rollout_err[10]:.5f} | "
      f"H=20 {rollout_err[20]:.5f} | H=40 {rollout_err[40]:.5f}")
print(f"  누적 배율(H=40 롤아웃 / 1스텝평균) = {rollout_err[40]/onestep_err.mean():.1f}x")

# ------------------------------------------------------------
# 여러 데이터 크기에서 롤아웃 오차 vs 지평선 (그림 2)
# ------------------------------------------------------------
def horizon_curve(n_data, noise, seed):
    Sd, Ad, Snx = collect_data(n_data, noise, seed=seed)
    Ah, Bh = fit_wm(Sd, Ad, Snx)
    trw = rollout_wm(Ah, Bh, s0, a_seq)
    return np.linalg.norm(trw - tr_true, axis=1)

curves = {n: horizon_curve(n, NOISE, seed=1) for n in (50, 200, 1000)}
print("== 그림2: 데이터 크기별 롤아웃 오차(H=40) ==")
for n, c in curves.items():
    print(f"  n={n:5d}: err(H=40) = {c[40]:.4f}")

# ============================================================
# WE-2 (a): latent 예측 vs pixel(고차원) 예측 안정성
#   같은 잠재 2D 동역학. pixel = 랜덤 선형 사상으로 D차원에 올림 + 픽셀노이즈.
#   latent 모델: 2D 잠재에서 직접 f_theta 학습·롤아웃.
#   pixel 모델: D차원 관측공간에서 직접 선형 dynamics 학습·롤아웃(고차원·저신호대잡음).
# ============================================================
D_PIX = 64
rr = np.random.default_rng(7)
W_dec = rr.standard_normal((D_PIX, ns)) / np.sqrt(ns)   # 잠재→픽셀 디코더(고정)
# 픽셀공간 참 전이는 존재하지 않지만(비가역 사상), 픽셀 모델은 그래도 D×D 선형을 맞추려 한다.

def make_pixel_dataset(n, latent_noise, pixel_noise, seed):
    r = np.random.default_rng(seed)
    Sd = r.standard_normal((n, ns))
    Ad = r.standard_normal((n, na))
    Snx = np.stack([true_step(Sd[i], Ad[i]) for i in range(n)])
    Snx += latent_noise * r.standard_normal(Snx.shape)
    # 픽셀 관측
    Xp = Sd @ W_dec.T + pixel_noise * r.standard_normal((n, D_PIX))
    Xnp = Snx @ W_dec.T + pixel_noise * r.standard_normal((n, D_PIX))
    return Sd, Ad, Snx, Xp, Xnp

def fit_pixel_wm(Xp, Ad, Xnp):
    Xin = np.hstack([Xp, Ad])
    theta, *_ = np.linalg.lstsq(Xin, Xnp, rcond=None)   # (D+na, D)
    theta = theta.T
    return theta[:, :D_PIX], theta[:, D_PIX:]           # A_pix (D×D), B_pix (D×na)

# 데이터는 넉넉하지 않고(N2), 픽셀 관측엔 노이즈가 있다 — 현실적 설정.
# latent 모델은 2×3 만 추정하지만, pixel 모델은 64×65 를 추정해야 한다.
# → pixel 모델은 (D-2)개의 "노이즈 방향"에서 허구의 동역학을 학습하고,
#    그 스펙트럼 반경이 1 근처/이상이면 긴 롤아웃에서 픽셀 노이즈를 증폭한다.
N2, LAT_NOISE, PIX_NOISE = 300, 0.02, 0.15
Sd, Ad, Snx, Xp, Xnp = make_pixel_dataset(N2, LAT_NOISE, PIX_NOISE, seed=3)
# latent 모델
A_lat, B_lat = fit_wm(Sd, Ad, Snx)
# pixel 모델
A_pix, B_pix = fit_pixel_wm(Xp, Ad, Xnp)
rho_lat = np.abs(np.linalg.eigvals(A_lat)).max()
rho_pix = np.abs(np.linalg.eigvals(A_pix)).max()
print("== WE-2a: 스펙트럼 반경(안정성 지표) ==")
print(f"  latent A_hat: rho = {rho_lat:.4f} (참 {np.abs(np.linalg.eigvals(A_true)).max():.4f})")
print(f"  pixel  A_hat: rho = {rho_pix:.4f}  ({'발산 위험 >1' if rho_pix>1 else '유계'})")

# 롤아웃 비교: 참 잠재 궤적을 픽셀로 올려 기준으로.
tr_true_lat = rollout_true(s0, a_seq)            # (H+1, 2)
tr_true_pix = tr_true_lat @ W_dec.T              # (H+1, D)

tr_lat = rollout_wm(A_lat, B_lat, s0, a_seq)     # latent 롤아웃
tr_lat_pix = tr_lat @ W_dec.T                    # 디코더로 픽셀에 투영해 공정 비교

x0_pix = s0 @ W_dec.T
tr_pix = [x0_pix.copy()]
xp = x0_pix.copy()
for a in a_seq:
    xp = A_pix @ xp + (B_pix @ a)
    tr_pix.append(xp.copy())
tr_pix = np.array(tr_pix)

# 정규화 오차(픽셀 노름 기준)
def rel_pix_err(pred_pix):
    return np.linalg.norm(pred_pix - tr_true_pix, axis=1) / (np.linalg.norm(tr_true_pix, axis=1) + 1e-9)

err_lat = rel_pix_err(tr_lat_pix)
err_pix = rel_pix_err(tr_pix)
print("== WE-2a: latent vs pixel 롤아웃 (상대 픽셀오차) ==")
print(f"  latent: H=1 {err_lat[1]:.4f} | H=10 {err_lat[10]:.4f} | H=40 {err_lat[40]:.4f}")
print(f"  pixel : H=1 {err_pix[1]:.4f} | H=10 {err_pix[10]:.4f} | H=40 {err_pix[40]:.4f}")
print(f"  H=40 비율 pixel/latent = {err_pix[40]/err_lat[40]:.1f}x")

# ============================================================
# WE-2 (b): world model 롤아웃으로 정책 평가 — 상상 환경의 편향
#   비용 J = sum ||s_t||^2 을 참 환경 vs WM(상상) 환경에서 각각 평가.
#   WM 오차가 지평선에 따라 누적 → 상상 롤아웃의 J 추정이 편향된다.
# ============================================================
def eval_policy_cost(Ah, Bh, s0, actions, use_true=False):
    s = s0.copy(); J = 0.0; costs=[]
    for a in actions:
        s = true_step(s, a) if use_true else wm_step(Ah, Bh, s, a)
        c = float(s @ s)
        J += c; costs.append(c)
    return J, np.array(costs)

# 두 후보 행동열(정책) 비교
a_A = 0.6*np.sin(0.25*np.arange(H))[:, None]
a_B = 0.6*np.sin(0.25*np.arange(H) + 1.0)[:, None] * 0.7
J_true_A, cA_t = eval_policy_cost(None, None, s0, a_A, use_true=True)
J_true_B, cB_t = eval_policy_cost(None, None, s0, a_B, use_true=True)
J_wm_A, cA_w = eval_policy_cost(A_hat, B_hat, s0, a_A)
J_wm_B, cB_w = eval_policy_cost(A_hat, B_hat, s0, a_B)
print("== WE-2b: 상상 환경 정책 평가 편향 ==")
print(f"  참 환경  J(A)={J_true_A:.3f}  J(B)={J_true_B:.3f}  → 우수: {'A' if J_true_A<J_true_B else 'B'}")
print(f"  WM 환경  J(A)={J_wm_A:.3f}  J(B)={J_wm_B:.3f}  → 우수: {'A' if J_wm_A<J_wm_B else 'B'}")
print(f"  J(A) 상대편향 = {abs(J_wm_A-J_true_A)/J_true_A*100:.1f}%  J(B) 상대편향 = {abs(J_wm_B-J_true_B)/J_true_B*100:.1f}%")

# ============================================================
# fig1 — world model 롤아웃(상상 궤적): 참 vs WM vs 1스텝
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
tt = np.arange(H+1)
ax[0].plot(tt, tr_true[:, 0], C['true'], lw=2.2, label='참 환경 f(s,a) (0강)')
ax[0].plot(tt, tr_wm[:, 0], C['wm'], lw=1.6, ls='--', label='WM 상상 롤아웃 f_θ (free-run)')
# 1스텝(teacher-forced) 점들: 참 상태에서 한 스텝
os_pred = np.array([wm_step(A_hat, B_hat, tr_true[t], a_seq[t])[0] for t in range(H)])
ax[0].plot(tt[1:], os_pred, C['onestep'], marker='.', ls='none', ms=6,
           label='WM 1스텝 예측 (teacher-forced)')
ax[0].set_xlabel('스텝 t'); ax[0].set_ylabel('위치 s1')
ax[0].set_title('(a) 상상 롤아웃 — 1스텝은 정확, free-run은 서서히 이탈')
ax[0].legend(fontsize=8, loc='upper right'); ax[0].grid(alpha=0.3)

# 위상평면
ax[1].plot(tr_true[:, 0], tr_true[:, 1], C['true'], lw=2.2, label='참 궤적')
ax[1].plot(tr_wm[:, 0], tr_wm[:, 1], C['wm'], lw=1.6, ls='--', label='WM 상상 궤적')
ax[1].plot(s0[0], s0[1], 'ko', ms=7); ax[1].text(s0[0]+0.03, s0[1], "s0", fontsize=10)
ax[1].set_xlabel('위치 s1'); ax[1].set_ylabel('속도 s2')
ax[1].set_title('(b) 위상평면 — 상상이 참 궤적을 얼마나 따라가는가')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_wm_rollout.png', dpi=140); plt.close(fig)

# ============================================================
# fig2 — 롤아웃 오차 누적 vs 지평선 (37강 compounding error 회수)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tt, rollout_err, C['wm'], lw=2, marker='o', ms=3,
           label='free-run 롤아웃 오차 (누적)')
ax[0].axhline(onestep_err.mean(), color=C['onestep'], ls='--',
              label=f'1스텝 평균 오차 ≈ {onestep_err.mean():.4f} (누적 안 됨)')
ax[0].set_xlabel('예측 지평선 H (스텝)'); ax[0].set_ylabel('상태공간 오차 ||est-s||')
ax[0].set_title('(a) 오차 누적 — compounding error (37강)')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.3)

for n, c in curves.items():
    ax[1].semilogy(tt, np.maximum(c, 1e-6), lw=1.8, marker='.', ms=3,
                   label=f'학습 데이터 n={n}')
ax[1].set_xlabel('예측 지평선 H (스텝)'); ax[1].set_ylabel('롤아웃 오차 ||est-s|| (로그)')
ax[1].set_title('(b) 데이터 많을수록 f_θ 정확 → 오차 누적 완만')
ax[1].legend(fontsize=8, loc='lower right'); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_compounding.png', dpi=140); plt.close(fig)

# ============================================================
# fig3 — latent vs pixel 예측 안정성
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(tt, err_lat, C['latent'], lw=2, marker='o', ms=3,
           label='latent 예측 (2D 잠재에서 롤아웃)')
ax[0].plot(tt, err_pix, C['pixel'], lw=2, ls='--', marker='s', ms=3,
           label=f'pixel 예측 ({D_PIX}D 관측공간에서 롤아웃)')
ax[0].set_xlabel('예측 지평선 H'); ax[0].set_ylabel('상대 픽셀 오차')
ax[0].set_title(f'(a) latent vs pixel 롤아웃 안정성 (H=40서 {err_pix[40]/err_lat[40]:.1f}배 차)')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.3)

# 자유도(추정할 파라미터 수) 막대
ax[1].bar(['latent\n(2D)', 'pixel\n(64D)'],
          [ns*(ns+na), D_PIX*(D_PIX+na)],
          color=[C['latent'], C['pixel']])
ax[1].set_yscale('log'); ax[1].set_ylabel('추정할 파라미터 수 (로그)')
for i, v in enumerate([ns*(ns+na), D_PIX*(D_PIX+na)]):
    ax[1].text(i, v*1.15, f'{v}', ha='center', fontsize=10)
ax[1].set_title('(b) 왜 latent가 안정한가 — 추정 부담 %d배' % (D_PIX*(D_PIX+na)//(ns*(ns+na))))
ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_latent_vs_pixel.png', dpi=140); plt.close(fig)

# ============================================================
# fig4 — world model 3가지 쓰임 (데이터 / 플래닝 / 정책의 환경)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
# (a) 정책 평가 편향 막대
labels = ['정책 A', '정책 B']
x = np.arange(2); wbar = 0.36
ax[0].bar(x - wbar/2, [J_true_A, J_true_B], wbar, color=C['true'], label='참 환경 J')
ax[0].bar(x + wbar/2, [J_wm_A, J_wm_B], wbar, color=C['wm'], label='WM 상상 J (편향)')
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_ylabel('누적 비용 J = Σ‖s‖²')
ax[0].set_title('(a) 상상 환경의 정책 평가 — 순위는 유지, 값은 편향')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y')

# (b) 개념 다이어그램: 3가지 쓰임 (텍스트 박스)
ax[1].axis('off')
boxes = [
    ("① 데이터 생성", "WM이 상상 롤아웃을 뽑아\n정책 학습 데이터로 (53강)\nDreamGen 'neural trajectory'", 0.72, 'tab:orange'),
    ("② 플래닝", "행동열을 상상 롤아웃해\n목표까지 비용 최소화 (CEM)\nV-JEPA 2-AC · Dreamer", 0.42, 'tab:blue'),
    ("③ 정책의 환경", "실제 환경 대신 WM 안에서\n정책을 학습·평가\nHa&Schmidhuber 'dream'", 0.12, 'tab:green'),
]
for title, body, y, col in boxes:
    ax[1].add_patch(plt.Rectangle((0.03, y-0.02), 0.94, 0.24,
                    transform=ax[1].transAxes, facecolor=col, alpha=0.13,
                    edgecolor=col, lw=1.5))
    ax[1].text(0.06, y+0.17, title, transform=ax[1].transAxes,
               fontsize=11, fontweight='bold', color=col)
    ax[1].text(0.06, y+0.01, body, transform=ax[1].transAxes, fontsize=8.5, va='bottom')
ax[1].set_title('(b) 학습된 world model의 세 가지 쓰임')
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_three_uses.png', dpi=140); plt.close(fig)

print("figs saved:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
