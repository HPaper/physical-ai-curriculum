# Lec 26 그림 생성 스크립트 — 신경망 = 함수 근사기
# 실행: cd images/lec26 && python3 gen_figs.py   (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음. 결정론적(시드 고정).
import os
# BLAS 스레드 폭주로 인한 과도한 지연 방지(작은 행렬 반복). 결과에는 무영향.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = os.path.dirname(os.path.abspath(__file__))

C = {'data': '#333333', 'w2': '#1f77b4', 'w8': '#2ca02c', 'w32': '#ff7f0e',
     'w64': '#d62728', 'small': '#1f77b4', 'good': '#2ca02c', 'large': '#d62728',
     'tgt': '#333333', 'pred': '#d62728', 'delta': '#d62728', 'fwd': '#1f77b4'}


# ============================================================================
# 그림 1. ReLU MLP의 piecewise-linear 1D 함수 근사 — 은닉폭↑ 따라 근사 향상
#   1-은닉층 ReLU 특징(랜덤 고정) + 출력가중치는 최소자승으로 결정.
#   => "MLP = 학습 가능한 basis + 최소자승"이라는 로봇공학 번역을 그대로 재현.
#   universal approximation(Cybenko/Hornik)의 직관: 폭↑ → 접힘(knot)↑ → 오차↓.
# ============================================================================
def target(x):
    return np.sin(3 * x) + 0.5 * np.sin(7 * x) + 0.3 * x

def relu_fit(xs, ys, width, seed=1):
    r = np.random.default_rng(seed)
    w = r.uniform(-3, 3, width)           # 각 뉴런의 기울기 (접힘 방향)
    b = r.uniform(-3, 3, width)           # 각 뉴런의 절편 (접힘 위치 = knot)
    H = np.maximum(0.0, xs[:, None] * w[None, :] + b[None, :])   # ReLU 특징
    H = np.concatenate([H, np.ones((len(xs), 1))], axis=1)        # 바이어스 열
    coef, *_ = np.linalg.lstsq(H, ys, rcond=None)                # 최소자승 출력층
    yhat = H @ coef
    rmse = float(np.sqrt(np.mean((yhat - ys) ** 2)))
    return yhat, rmse

xs = np.linspace(-2, 2, 400)
ys = target(xs)
widths = [2, 4, 8, 16, 32, 64]
fits = {w: relu_fit(xs, ys, w, seed=1) for w in widths}
rmses = [fits[w][1] for w in widths]
# 콘솔 출력(본문 수치와 대조): width=2 RMSE=0.7696 ... width=64 RMSE=0.0776
print("[fig1] ReLU MLP 1D 근사 RMSE:")
for w in widths:
    print(f"  width={w:3d}  RMSE={fits[w][1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(xs, ys, color=C['data'], lw=2.5, label='목표 함수', zorder=5)
for w, col in [(2, C['w2']), (8, C['w8']), (32, C['w32'])]:
    ax[0].plot(xs, fits[w][0], color=col, lw=1.6, alpha=0.9,
               label=f'ReLU MLP 폭={w} (RMSE {fits[w][1]:.2f})')
ax[0].set_title('(a) piecewise-linear 근사: 폭이 커질수록 목표에 밀착', fontsize=11)
ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].legend(fontsize=8, loc='upper left')
ax[0].grid(alpha=0.3)

ax[1].plot(widths, rmses, 'o-', color=C['w64'], lw=2)
for w, e in zip(widths, rmses):
    ax[1].annotate(f'{e:.3f}', (w, e), textcoords='offset points',
                   xytext=(6, 6), fontsize=8)
ax[1].set_xscale('log', base=2); ax[1].set_yscale('log')
ax[1].set_xticks(widths); ax[1].set_xticklabels(widths)
ax[1].set_title('(b) 은닉폭 ↑ → 근사 오차 ↓ (universal approximation)', fontsize=11)
ax[1].set_xlabel('은닉 뉴런 수 (= piecewise-linear 조각 수)')
ax[1].set_ylabel('RMSE (log)')
ax[1].grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_relu_approx.png", dpi=130)
plt.close()


# ============================================================================
# 그림 2. 2D 손실지형 위 경사하강 궤적 — η 소/적정/과대 (수렴/느림/발산)
#   L(w)=1/2 w^T A w,  A=diag(1,20).  GD: w <- w - eta * A w.
#   수렴 조건 eta < 2/lambda_max = 2/20 = 0.1  (17강 게인↔안정성과 같은 구조).
# ============================================================================
A = np.array([1.0, 20.0])                  # lambda_min=1, lambda_max=20
eta_crit = 2.0 / A.max()                    # = 0.1
w0 = np.array([9.0, 1.5])
def gd(eta, steps=40):
    w = w0.copy(); traj = [w.copy()]
    for _ in range(steps):
        g = A * w                           # ∇L
        w = w - eta * g                     # θ ← θ − η∇L
        traj.append(w.copy())
    return np.array(traj)

cases = [("η=0.02 (작음: 느린 수렴)", 0.02, C['small']),
         ("η=0.09 (적정: 빠른 수렴)", 0.09, C['good']),
         ("η=0.104 (과대: 발산)", 0.104, C['large'])]
print(f"[fig2] eta_crit = 2/lambda_max = {eta_crit:.3f}")
for name, eta, _ in cases:
    tr = gd(eta)
    print(f"  {name:26s} |w_final|={np.linalg.norm(tr[-1]):.4e}")

g1 = np.linspace(-10, 10, 200); g2 = np.linspace(-4, 4, 200)
G1, G2 = np.meshgrid(g1, g2)
Z = 0.5 * (A[0] * G1**2 + A[1] * G2**2)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
for k, (name, eta, col) in enumerate(cases):
    tr = gd(eta)
    ax[k].contour(G1, G2, Z, levels=np.linspace(1, 200, 14), colors='#bbbbbb', linewidths=0.7)
    ax[k].plot(tr[:, 0], tr[:, 1], 'o-', color=col, ms=3.5, lw=1.3)
    ax[k].plot(0, 0, '*', color='#333333', ms=13)
    ax[k].plot(w0[0], w0[1], 's', color=col, ms=7)
    ax[k].set_title(name, fontsize=10.5)
    ax[k].set_xlabel(r'$w_1$ (곡률 작음)')
    if k == 0:
        ax[k].set_ylabel(r'$w_2$ (곡률 큼)')
    ax[k].set_xlim(-11, 11); ax[k].set_ylim(-4, 4)
plt.suptitle(r'경사하강 궤적: 발산 경계 $\eta_{crit}=2/\lambda_{max}=0.1$ (17강 게인↔안정성)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_gd_landscape.png", dpi=130, bbox_inches='tight')
plt.close()


# ============================================================================
# 그림 3. 역전파 = 자코비안 연쇄 (층별 δ 역전파 도식)
#   forward: a0 -> z1 -> a1 -> ... -> zL -> yhat  (아핀 ∘ 비선형)
#   backward: δ_L -> δ_{L-1} -> ... 를 자코비안 곱으로.
#   5강(속도 자코비안 연쇄), 11강(RNEA forward/backward 쌍)과 같은 구조.
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axis('off')
Ln = 4                       # 층 수(가시화용): x, h1, h2, y
xpos = np.linspace(0.08, 0.92, Ln)
y_fwd, y_bwd = 0.72, 0.28
names_f = ['x', r'$a_1=\sigma(z_1)$', r'$a_2=\sigma(z_2)$', r'$\hat y=z_3$']
names_b = [r'$\delta_1$', r'$\delta_2$', r'$\delta_3$', r'$\partial L/\partial \hat y$']
# forward row
for i, xp in enumerate(xpos):
    ax.scatter([xp], [y_fwd], s=1500, color=C['fwd'], alpha=0.18, zorder=1)
    ax.text(xp, y_fwd, names_f[i], ha='center', va='center', fontsize=11)
    if i < Ln - 1:
        ax.annotate('', xy=(xpos[i+1]-0.05, y_fwd), xytext=(xp+0.05, y_fwd),
                    arrowprops=dict(arrowstyle='-|>', color=C['fwd'], lw=2))
        ax.text((xp+xpos[i+1])/2, y_fwd+0.07, r'$W_{%d}$' % (i+1),
                ha='center', fontsize=10, color=C['fwd'])
ax.text(0.5, 0.90, 'forward (아핀→비선형 합성): 한 번의 전방 계산', ha='center',
        fontsize=11, color=C['fwd'])
# backward row
for i, xp in enumerate(xpos):
    ax.scatter([xp], [y_bwd], s=1500, color=C['delta'], alpha=0.18, zorder=1)
    ax.text(xp, y_bwd, names_b[i], ha='center', va='center', fontsize=11)
    if i < Ln - 1:
        ax.annotate('', xy=(xpos[i]+0.05, y_bwd), xytext=(xpos[i+1]-0.05, y_bwd),
                    arrowprops=dict(arrowstyle='-|>', color=C['delta'], lw=2))
        ax.text((xp+xpos[i+1])/2, y_bwd-0.09,
                r'$\times W_{%d}^{\top}\,\odot\,\sigma^{\prime}$' % (i+1),
                ha='center', fontsize=9, color=C['delta'])
ax.text(0.5, 0.10, r'backward (연쇄법칙 역방향 누적): $\delta_l=(W_{l+1}^{\top}\delta_{l+1})\odot\sigma^{\prime}(z_l)$',
        ha='center', fontsize=11, color=C['delta'])
# vertical link showing gradient wrt W
for i in range(Ln - 1):
    xm = (xpos[i] + xpos[i+1]) / 2
    ax.annotate('', xy=(xm, y_bwd+0.06), xytext=(xm, y_fwd-0.06),
                arrowprops=dict(arrowstyle='-', color='#999999', lw=1, ls=':'))
ax.text(0.5, 0.50, r'$\partial L/\partial W_l = \delta_l\,a_{l-1}^{\top}$  (forward의 $a$ × backward의 $\delta$)',
        ha='center', fontsize=10.5, color='#555555',
        bbox=dict(boxstyle='round', fc='#f4f4f4', ec='#cccccc'))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('역전파 = 자코비안 연쇄 (5강 속도 자코비안 · 11강 RNEA와 같은 forward/backward 쌍)',
             fontsize=11.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_backprop_chain.png", dpi=130)
plt.close()


# ============================================================================
# 그림 4. WE-2의 2링크 IK 근사 결과 — 목표 vs 예측 산점 + 위치오차
#   from-scratch numpy MLP (2->32->32->2, tanh). elbow-down 분지만 학습.
#   예측 관절각을 FK로 되돌려 위치 잔차를 잰다. 다해성 관찰(한 분지 선택).
# ============================================================================
L1, L2 = 1.0, 1.0
def fk(th1, th2):
    x = L1*np.cos(th1) + L2*np.cos(th1+th2)
    y = L1*np.sin(th1) + L2*np.sin(th1+th2)
    return np.stack([x, y], axis=-1)

rng = np.random.default_rng(0)
Ntr = 1200
tr1 = rng.uniform(0.0, np.pi/2, Ntr)
tr2 = rng.uniform(0.3, np.pi-0.3, Ntr)       # elbow-down 분지만 (θ2>0)
Ytr = np.stack([tr1, tr2], axis=1)
Xtr = fk(tr1, tr2)
Xm, Xs = Xtr.mean(0), Xtr.std(0)
Xtrn = (Xtr - Xm) / Xs

def init(sizes, seed=1):
    r = np.random.default_rng(seed); Ws = []; bs = []
    for i in range(len(sizes)-1):
        W = r.standard_normal((sizes[i+1], sizes[i])) * np.sqrt(2.0/sizes[i]) * 0.7
        Ws.append(W); bs.append(np.zeros(sizes[i+1]))
    return Ws, bs

def forward(Ws, bs, X):
    a = X; caches = [a]; zs = []
    for l in range(len(Ws)):
        z = a @ Ws[l].T + bs[l]; zs.append(z)
        a = np.tanh(z) if l < len(Ws)-1 else z
        caches.append(a)
    return a, zs, caches

Ws, bs = init([2, 32, 32, 2], seed=1)
n = Xtrn.shape[0]; EP = 2500; lr = 0.1
for ep in range(EP):
    yhat, zs, caches = forward(Ws, bs, Xtrn)
    delta = (yhat - Ytr) / n
    for l in reversed(range(len(Ws))):
        gW = delta.T @ caches[l]; gb = delta.sum(0)
        if l > 0:
            delta = (delta @ Ws[l]) * (1 - np.tanh(zs[l-1])**2)
        Ws[l] -= lr*gW; bs[l] -= lr*gb
train_loss = 0.5*np.mean(np.sum((yhat - Ytr)**2, axis=1))

rng2 = np.random.default_rng(7)
te1 = rng2.uniform(0.0, np.pi/2, 800)
te2 = rng2.uniform(0.3, np.pi-0.3, 800)
Xte = fk(te1, te2)
Xten = (Xte - Xm) / Xs
pred, _, _ = forward(Ws, bs, Xten)
Xpred = fk(pred[:, 0], pred[:, 1])
pos_err = np.linalg.norm(Xpred - Xte, axis=1)
print(f"[fig4] train_loss={train_loss:.6f}  pos_err mean={pos_err.mean():.5f} "
      f"median={np.median(pos_err):.5f} p95={np.percentile(pos_err,95):.5f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].scatter(Xte[:, 0], Xte[:, 1], s=10, color=C['tgt'], alpha=0.5, label='목표 (x,y)')
ax[0].scatter(Xpred[:, 0], Xpred[:, 1], s=10, color=C['pred'], alpha=0.5,
              marker='x', label='FK(예측 관절각)')
ax[0].set_aspect('equal')
ax[0].set_title(f'(a) 목표 vs 예측 EEF 위치 (평균 잔차 {pos_err.mean():.3f} m)', fontsize=10.5)
ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

ax[1].hist(pos_err, bins=35, color=C['pred'], alpha=0.8)
ax[1].axvline(pos_err.mean(), color='#333333', lw=1.6, ls='--',
              label=f'평균 {pos_err.mean():.3f} m')
ax[1].axvline(np.percentile(pos_err, 95), color='#666666', lw=1.2, ls=':',
              label=f'p95 {np.percentile(pos_err,95):.3f} m')
ax[1].set_title('(b) 위치 잔차 분포 (외삽·다해성 한계)', fontsize=10.5)
ax[1].set_xlabel('위치 오차 [m]'); ax[1].set_ylabel('빈도')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_ik_approx.png", dpi=130)
plt.close()

print("saved: fig1_relu_approx.png fig2_gd_landscape.png fig3_backprop_chain.png fig4_ik_approx.png")
