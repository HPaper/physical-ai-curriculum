# Lec 27 그림 생성 스크립트 — 학습 파이프라인 해부
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만, 결정론적)
# 개념(과적합·미니배치 분산·편향-분산·학습률)을 순수 numpy 토이로 재현한다.
# 실제 대형 모델/GPU 없음. 모든 난수는 default_rng로 시드 고정.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'figure.dpi': 120, 'savefig.bbox': 'tight'})
OUT = __file__.replace('gen_figs.py', '')

# 색
C_TR = '#2c6fb0'   # train (파랑)
C_VA = '#c0392b'   # val   (빨강)
C_HL = '#1a8a4a'   # 강조  (초록)
C_GY = '#888888'

# ============================================================================
# 공통 토이: 1D 다항 회귀. 참 함수 f(x)=sin(1.6x), 관측 y=f(x)+noise.
#   - 데이터량 N, 모델 용량(다항 차수 p), 정칙화 λ, 학습률을 바꿔가며
#     과적합/편향-분산/최적화를 재현한다.
# ============================================================================
def f_true(x):
    return np.sin(1.6 * x)

def make_data(n, noise=0.25, seed=0, xlo=-2.5, xhi=2.5):
    r = np.random.default_rng(seed)
    x = r.uniform(xlo, xhi, n)
    y = f_true(x) + noise * r.standard_normal(n)
    return x, y

def poly_features(x, p):
    # [1, x, x^2, ..., x^p]
    X = np.vander(x, p + 1, increasing=True)
    return X

def standardize(X):
    # 상수열(0번)은 그대로 두고 1..p 열만 표준화 (조건수 개선, 절편 유지)
    mu = X.mean(0); sd = X.std(0)
    mu[0] = 0.0; sd[0] = 1.0            # 절편 열 보존
    sd[sd == 0] = 1.0
    return mu, sd

def ridge_fit(X, y, lam):
    # (X^T X + lam I) w = X^T y  — 릿지 = L2 = DLS 댐핑 λ (7강 회수)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)

def mse(y, yp):
    return float(np.mean((y - yp) ** 2))

# ----------------------------------------------------------------------------
# FIG 1: train vs val 손실 U자 곡선 + 조기중단 지점
#   경사하강(full-batch)으로 고용량 모델을 오래 학습 → train은 계속 내려가고
#   val은 어느 순간부터 다시 오른다. 최소 val 에폭 = 조기중단 지점.
# ----------------------------------------------------------------------------
def fig1():
    p = 15                      # 고용량 (15차 다항)
    ntr = 18                    # 소량 데이터 → 과적합 유도
    xtr, ytr = make_data(ntr, noise=0.22, seed=4)
    xva, yva = make_data(400, noise=0.22, seed=99)
    Xtr = poly_features(xtr, p)
    Xva = poly_features(xva, p)
    # 열 표준화 (경사하강 안정화)
    mu, sd = standardize(Xtr)
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd

    w = np.zeros(p + 1)
    lr = 0.03
    epochs = 60000
    tr_hist, va_hist = [], []
    for e in range(epochs):
        grad = (2.0 / ntr) * Xtr_n.T @ (Xtr_n @ w - ytr)   # full-batch gradient
        w = w - lr * grad
        tr_hist.append(mse(ytr, Xtr_n @ w))
        va_hist.append(mse(yva, Xva_n @ w))
    tr_hist, va_hist = np.array(tr_hist), np.array(va_hist)
    e_star = int(np.argmin(va_hist))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ep = np.arange(1, epochs + 1)
    ax.plot(ep, tr_hist, color=C_TR, lw=2, label='훈련 손실 (train)')
    ax.plot(ep, va_hist, color=C_VA, lw=2, label='검증 손실 (val)')
    ax.axvline(e_star, color=C_HL, ls='--', lw=1.8)
    ax.scatter([e_star], [va_hist[e_star]], color=C_HL, zorder=5, s=55)
    ax.annotate(f'조기중단 지점\n에폭 {e_star}, val={va_hist[e_star]:.3f}',
                xy=(e_star, va_hist[e_star]), xytext=(e_star + 550, va_hist[e_star] + 0.12),
                color=C_HL, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=C_HL))
    ax.set_xscale('log')
    ax.set_xlabel('에폭 (log 스케일)')
    ax.set_ylabel('MSE 손실')
    ax.set_title('과적합의 U자: train은 내려가고 val은 되오른다')
    ax.legend(loc='upper center')
    ax.grid(alpha=0.3)
    fig.savefig(OUT + 'fig1_overfit_ucurve.png')
    plt.close(fig)
    print(f"[fig1] e*={e_star}  train@e*={tr_hist[e_star]:.4f}  val@e*={va_hist[e_star]:.4f}  "
          f"train@end={tr_hist[-1]:.4f}  val@end={va_hist[-1]:.4f}")
    return e_star, tr_hist[e_star], va_hist[e_star], tr_hist[-1], va_hist[-1]

# ----------------------------------------------------------------------------
# FIG 2: 미니배치 기울기 분산 vs 배치크기 (1/B 스케일, 로그-로그)
#   고정 파라미터 w0에서, 배치 B개를 무작위로 뽑아 기울기를 여러 번 계산 →
#   그 기울기 표본의 분산(성분 평균)이 1/B로 준다. 편향 ~ 0.
# ----------------------------------------------------------------------------
def fig2():
    p = 3
    N = 4096
    x, y = make_data(N, noise=0.5, seed=7)
    X = poly_features(x, p)
    mu, sd = standardize(X)
    Xn = (X - mu) / sd
    w0 = ridge_fit(Xn, y, 1e-3) + 0.3    # 최적 근처의 임의 지점 (기울기 0 아님)

    full_grad = (2.0 / N) * Xn.T @ (Xn @ w0 - y)   # 참 기울기 (편향 기준)

    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    reps = 6000
    r = np.random.default_rng(42)
    var_meas, bias_meas = [], []
    for B in batches:
        gs = np.empty((reps, p + 1))
        for t in range(reps):
            idx = r.integers(0, N, B)          # 복원추출 (iid)
            Xb, yb = Xn[idx], y[idx]
            gs[t] = (2.0 / B) * Xb.T @ (Xb @ w0 - yb)
        gbar = gs.mean(0)
        var_meas.append(float(gs.var(0).mean()))        # 성분별 분산의 평균
        bias_meas.append(float(np.linalg.norm(gbar - full_grad)))
    var_meas = np.array(var_meas)

    # 1/B 기준선: B=1의 분산을 앵커로
    ref = var_meas[0] / np.array(batches)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(batches, var_meas, 'o-', color=C_TR, lw=2, ms=7, label='측정된 기울기 분산')
    ax.loglog(batches, ref, '--', color=C_GY, lw=1.8, label=r'$\propto 1/B$ 기준선')
    ax.set_xlabel('배치 크기 B (log)')
    ax.set_ylabel('기울기 표본 분산 (성분 평균, log)')
    ax.set_title(r'미니배치 기울기 분산 $\propto 1/B$ · 편향 ≈ 0')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.savefig(OUT + 'fig2_minibatch_variance.png')
    plt.close(fig)
    print("[fig2] B, var, bias:")
    for B, v, b in zip(batches, var_meas, bias_meas):
        print(f"   B={B:4d}  var={v:.5e}  bias={b:.2e}")
    # 분산 비율이 1/B에 얼마나 붙는지
    ratio = var_meas[0] / var_meas
    print("   var(B=1)/var(B) vs B (완전 1/B면 같아야):", np.round(ratio, 2))
    return batches, var_meas, bias_meas

# ----------------------------------------------------------------------------
# FIG 3: 모델 용량 vs 오차 — 편향-분산 트레이드오프 (U자)
#   차수 p를 1..14로 키우며, 여러 훈련셋 재추출로 test MSE를 편향²/분산/노이즈로 분해.
# ----------------------------------------------------------------------------
def fig3():
    degs = list(range(0, 13))
    ntr = 30
    noise = 0.3
    ntrials = 300
    # 고정 테스트 격자 (편향/분산 분해용)
    xte = np.linspace(-2.5, 2.5, 200)
    fte = f_true(xte)

    bias2, var, test_err = [], [], []
    for p in degs:
        preds = np.empty((ntrials, xte.size))
        errs = []
        for t in range(ntrials):
            xtr, ytr = make_data(ntr, noise=noise, seed=1000 + t)
            Xtr = poly_features(xtr, p)
            mu, sd = standardize(Xtr)
            Xtr_n = (Xtr - mu) / sd
            w = ridge_fit(Xtr_n, ytr, 1e-6)     # 거의 무정칙 → 용량 효과가 드러남
            Xte_n = (poly_features(xte, p) - mu) / sd
            yp = Xte_n @ w
            preds[t] = yp
            errs.append(np.mean((yp - fte) ** 2))   # 노이즈 없는 참값 대비 (test err 대리)
        mean_pred = preds.mean(0)
        b2 = float(np.mean((mean_pred - fte) ** 2))
        vr = float(np.mean(preds.var(0)))
        bias2.append(b2); var.append(vr); test_err.append(float(np.mean(errs)))
    bias2, var, test_err = np.array(bias2), np.array(var), np.array(test_err)
    p_star = degs[int(np.argmin(test_err))]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(degs, bias2, 's-', color=C_TR, lw=2, ms=5, label='편향²')
    ax.plot(degs, var, '^-', color=C_VA, lw=2, ms=5, label='분산')
    ax.plot(degs, test_err, 'o-', color='#333333', lw=2.4, ms=6, label='기대 오차 (≈편향²+분산)')
    ax.axvline(p_star, color=C_HL, ls='--', lw=1.6)
    ax.annotate(f'최적 용량\n차수 p={p_star}', xy=(p_star, test_err[p_star]),
                xytext=(p_star + 1.2, test_err[p_star] + 0.12), color=C_HL, fontsize=10,
                arrowprops=dict(arrowstyle='->', color=C_HL))
    ax.set_xlabel('모델 용량 (다항 차수 p)')
    ax.set_ylabel('오차 (test 격자, 참값 대비 MSE)')
    ax.set_title('편향-분산 트레이드오프: 용량↑ → 편향↓·분산↑')
    ax.set_ylim(0, min(1.2, test_err.max() * 1.1))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT + 'fig3_bias_variance.png')
    plt.close(fig)
    print(f"[fig3] p*={p_star}  test_err(min)={test_err[p_star]:.4f}")
    for p, b, v, te in zip(degs, bias2, var, test_err):
        print(f"   p={p:2d}  bias2={b:.4f}  var={v:.4f}  test={te:.4f}")
    return degs, bias2, var, test_err, p_star

# ----------------------------------------------------------------------------
# FIG 4: 학습률 스윕 (소/적정/과대) — 수렴속도·발산
#   같은 이차형 손실에서 lr를 바꿔 손실 궤적. lr_crit=2/L (L=최대 곡률).
# ----------------------------------------------------------------------------
def fig4():
    p = 4
    N = 400
    x, y = make_data(N, noise=0.3, seed=3)
    X = poly_features(x, p)
    mu, sd = standardize(X)
    Xn = (X - mu) / sd
    # 손실 L(w)=1/N||Xn w - y||^2, Hessian=2/N Xn^T Xn, 최대고유값=L(립시츠)
    H = (2.0 / N) * (Xn.T @ Xn)
    L = float(np.linalg.eigvalsh(H).max())
    lr_crit = 2.0 / L
    wstar = ridge_fit(Xn, y, 1e-10)     # 최소제곱해 (수치안정용 미세 릿지)
    Lstar = mse(y, Xn @ wstar)

    lrs = {'과소 (lr=0.05·lr_crit)': 0.05 * lr_crit,
           '적정 (lr=0.6·lr_crit)': 0.6 * lr_crit,
           '과대 (lr=1.02·lr_crit)': 1.02 * lr_crit}
    colors = {'과소 (lr=0.05·lr_crit)': C_TR, '적정 (lr=0.6·lr_crit)': C_HL,
              '과대 (lr=1.02·lr_crit)': C_VA}
    steps = 60
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    finals = {}
    for name, lr in lrs.items():
        w = np.zeros(p + 1)
        hist = []
        for s in range(steps):
            g = (2.0 / N) * Xn.T @ (Xn @ w - y)
            w = w - lr * g
            hist.append(mse(y, Xn @ w) - Lstar + 1e-9)   # 최적 손실 대비 초과분
        hist = np.array(hist)
        finals[name] = hist[-1]
        ax.semilogy(np.arange(1, steps + 1), hist, color=colors[name], lw=2, label=name)
    ax.axhline(1e-9, color=C_GY, ls=':', lw=1)
    ax.set_xlabel('경사하강 스텝')
    ax.set_ylabel(r'손실 초과분 $L(w)-L^*$ (log)')
    ax.set_title(r'학습률 스윕: $lr_{crit}=2/L$ (L=최대 곡률)')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.savefig(OUT + 'fig4_lr_sweep.png')
    plt.close(fig)
    print(f"[fig4] L(max curvature)={L:.4f}  lr_crit=2/L={lr_crit:.4f}")
    for name, fv in finals.items():
        print(f"   {name}: final (L-L*)={fv:.3e}")
    return L, lr_crit, finals

if __name__ == '__main__':
    print("=== Lec27 그림 생성 ===")
    fig1()
    fig2()
    fig3()
    fig4()
    print("=== 완료 ===")
