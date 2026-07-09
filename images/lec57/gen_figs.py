"""
Lec 57 — 벤치마크와 평가의 함정: 그림 생성 + 수치 확정 스크립트.

순수 numpy / scipy / matplotlib (CPU, 결정론 시드). sklearn 미사용.
본문·캡션·Worked Example의 모든 수치는 이 스크립트의 실행 출력과 일치한다.

그림:
  fig1_ci_vs_N.png       이항 성공률 신뢰구간 폭 vs N (Wilson/CP/Wald, 1/sqrt(N))
  fig2_rank_noise.png    포화 벤치마크 순위 노이즈 (반복 평가 top-1 분포)
  fig3_libero_plus.png   LIBERO -> LIBERO-Plus 교란 시 성능 붕괴 (막대)
  fig4_AB_overlap.png    모델 A vs B 차이의 CI 겹침과 유의성 (N=20/50/200)

실행: python3 gen_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

OUT = __file__.rsplit("/", 1)[0]
SEED = 42

# ---------------------------------------------------------------------------
# 공통: 이항 비율 신뢰구간 (Wald / Wilson / Clopper-Pearson)
# ---------------------------------------------------------------------------
def wald(k, n, z=1.96):
    p = k / n
    h = z * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

C = {  # 색 팔레트
    "wald": "#d62728", "wilson": "#1f77b4", "cp": "#2ca02c",
    "sat": "#8c564b", "libero": "#1f77b4", "plus": "#d62728",
    "A": "#ff7f0e", "B": "#1f77b4", "grid": "#cccccc",
}


# ===========================================================================
# 그림 1 — CI 폭 vs N (1/sqrt(N) 스케일링)
# ===========================================================================
def fig1_ci_vs_N():
    Ns = np.array([5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500])
    phat = 0.7
    w_wald, w_wilson, w_cp = [], [], []
    for N in Ns:
        k = round(phat * N)
        a, b = wald(k, N); w_wald.append(b - a)
        a, b = wilson(k, N); w_wilson.append(b - a)
        a, b = clopper_pearson(k, N); w_cp.append(b - a)
    w_wald, w_wilson, w_cp = map(np.array, (w_wald, w_wilson, w_cp))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # (a) 폭 vs N
    ax[0].plot(Ns, w_cp, "-o", color=C["cp"], ms=4, label="Clopper-Pearson (정확)")
    ax[0].plot(Ns, w_wilson, "-s", color=C["wilson"], ms=4, label="Wilson score")
    ax[0].plot(Ns, w_wald, "-^", color=C["wald"], ms=4, label="Wald (정규 근사)")
    # 1/sqrt(N) 기준선 (Wald N=20 폭에 정규화)
    ref = w_wald[Ns == 20][0] * np.sqrt(20 / Ns)
    ax[0].plot(Ns, ref, "--", color="gray", lw=1, label=r"$\propto 1/\sqrt{N}$ 기준선")
    for xv, lbl in [(10, "N=10"), (20, "N=20"), (50, "N=50")]:
        ax[0].axvline(xv, color=C["grid"], lw=0.8, ls=":")
    ax[0].set_xlabel("롤아웃 수 N (log)"); ax[0].set_ylabel("95% CI 폭 (성공률)")
    ax[0].set_xscale("log"); ax[0].set_title(r"(a) CI 폭은 $1/\sqrt{N}$로만 줄어든다  ($\hat p=0.7$)")
    ax[0].legend(fontsize=8, loc="upper right"); ax[0].grid(alpha=0.3)
    ax[0].annotate("N=20: 폭 ~0.37\n(±19%p)", xy=(20, w_wilson[Ns == 20][0]),
                   xytext=(45, 0.52), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="k", lw=0.8))

    # (b) 에러바로 본 phat=0.7 추정 (Wilson) at several N
    Nshow = [10, 20, 50, 100, 200]
    xs = np.arange(len(Nshow))
    for i, N in enumerate(Nshow):
        k = round(0.7 * N); lo, hi = wilson(k, N)
        ax[1].errorbar(i, 0.7, yerr=[[0.7 - lo], [hi - 0.7]], fmt="o",
                       color=C["wilson"], capsize=5, ms=6)
        ax[1].text(i, hi + 0.015, f"±{(hi-lo)/2*100:.0f}%p", ha="center", fontsize=8)
    ax[1].axhline(0.7, color="gray", ls="--", lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"N={n}" for n in Nshow])
    ax[1].set_ylabel(r"성공률 추정 $\hat p$ (95% Wilson CI)")
    ax[1].set_ylim(0.3, 1.0); ax[1].set_title(r"(b) 같은 $\hat p=0.7$, N만 다르게")
    ax[1].grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_ci_vs_N.png"); plt.close(fig)

    print("\n[FIG1] phat=0.7, 95% CI 폭:")
    for N in (10, 20, 50, 100):
        k = round(0.7 * N)
        print(f"  N={N:4d}: Wilson {wilson(k,N)[1]-wilson(k,N)[0]:.3f}, "
              f"CP {clopper_pearson(k,N)[1]-clopper_pearson(k,N)[0]:.3f}, "
              f"Wald {wald(k,N)[1]-wald(k,N)[0]:.3f}")
    return w_wilson, Ns


# ===========================================================================
# 그림 2 — 포화 벤치마크 순위 노이즈
# ===========================================================================
def fig2_rank_noise():
    rng = np.random.default_rng(SEED)
    names = list("ABCDE")
    true_sat = np.array([0.94, 0.95, 0.96, 0.95, 0.97])     # 천장에 몰림 (포화)
    true_spread = np.array([0.30, 0.45, 0.60, 0.50, 0.72])  # 교란이 벌려놓음
    N, reps = 20, 20000

    def top1(true_p):
        cnt = np.zeros(len(true_p)); correct = 0; tb = int(np.argmax(true_p))
        for _ in range(reps):
            obs = rng.binomial(N, true_p) / N
            m = obs.max(); w = rng.choice(np.where(obs == m)[0])
            cnt[w] += 1
            if w == tb: correct += 1
        return cnt / reps, correct / reps

    def two_run_disagree(true_p):
        d = 0
        for _ in range(reps):
            o1 = rng.binomial(N, true_p) / N; o2 = rng.binomial(N, true_p) / N
            w1 = rng.choice(np.where(o1 == o1.max())[0])
            w2 = rng.choice(np.where(o2 == o2.max())[0])
            d += (w1 != w2)
        return d / reps

    t_sat, c_sat = top1(true_sat)
    t_spr, c_spr = top1(true_spread)
    dis_sat = two_run_disagree(true_sat)
    dis_spr = two_run_disagree(true_spread)

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    x = np.arange(len(names))
    for a, tvals, freq, ttl, dis, cbest in [
        (ax[0], true_sat, t_sat, "(a) 포화 벤치마크 (참 성공률 0.94~0.97)", dis_sat, c_sat),
        (ax[1], true_spread, t_spr, "(b) 교란 후 (참 성공률 0.30~0.72)", dis_spr, c_spr)]:
        bars = a.bar(x, freq, color=C["sat"], alpha=0.55, edgecolor="k", lw=0.6)
        best = int(np.argmax(tvals))
        bars[best].set_color(C["plus"]); bars[best].set_alpha(0.9)
        a.axhline(0.2, color="gray", ls=":", lw=1)
        a.text(len(names) - 1, 0.21, "무작위(1/5)", fontsize=7.5, ha="right", color="gray")
        for i, (tv, fv) in enumerate(zip(tvals, freq)):
            a.text(i, fv + 0.012, f"{fv*100:.0f}%", ha="center", fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels([f"모델{n}\n(참 {tv:.2f})" for n, tv in zip(names, tvals)],
                          fontsize=8.5)
        a.set_title(ttl, fontsize=10)
        a.set_ylim(0, 0.85)
        a.text(0.03, 0.78, f"참 1등이 1등으로 뽑힐 확률 {cbest*100:.0f}%\n"
                           f"두 재실행이 1등 불일치 {dis*100:.0f}%",
               transform=a.transAxes, fontsize=8,
               bbox=dict(boxstyle="round", fc="#fff4d6", ec="gray", lw=0.6))
    ax[0].set_ylabel("반복 평가에서 '1등'으로 뽑힌 빈도")
    fig.suptitle("N=20 롤아웃 · 5모델 · 각 20000회 반복 평가 (빨강=참 성공률 1등)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_rank_noise.png"); plt.close(fig)

    print("\n[FIG2] 순위 노이즈 (N=20):")
    print(f"  포화 top-1 빈도: {dict(zip(names, np.round(t_sat,3)))}")
    print(f"    참1등(E,0.97) 1등확률 {c_sat*100:.1f}%, 두 재실행 불일치 {dis_sat*100:.1f}%")
    print(f"  교란 top-1 빈도: {dict(zip(names, np.round(t_spr,3)))}")
    print(f"    참1등(E,0.72) 1등확률 {c_spr*100:.1f}%, 두 재실행 불일치 {dis_spr*100:.1f}%")


# ===========================================================================
# 그림 3 — LIBERO -> LIBERO-Plus 붕괴 (교란별)
# ===========================================================================
def fig3_libero_plus():
    # 대표 서사 수치: clean ~95% -> 교란별 하락 (LIBERO-Plus 보고 경향의 도식화;
    # 정확한 per-factor 수치는 논문마다 다르므로 '개념 도식'으로 명시).
    factors = ["clean\n(원본)", "조명", "배경\n질감", "센서\n노이즈",
               "언어\n지시", "물체\n배치", "초기\n상태", "카메라\n시점"]
    perf = np.array([95, 82, 70, 74, 90, 58, 40, 28])   # 도식적 하락 프로파일
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    cols = [C["libero"]] + [C["plus"]] * (len(factors) - 1)
    bars = ax.bar(factors, perf, color=cols, alpha=0.8, edgecolor="k", lw=0.6)
    ax.axhline(95, color=C["libero"], ls="--", lw=1)
    ax.axhline(30, color="gray", ls=":", lw=1)
    ax.text(len(factors) - 0.4, 96, "원본 성능 ~95%", color=C["libero"], fontsize=8, ha="right")
    for b, v in zip(bars, perf):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}", ha="center", fontsize=8.5)
    # 붕괴 화살표
    ax.annotate("", xy=(7, 30), xytext=(0, 92),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls="--"))
    ax.text(3.4, 60, "교란 축을 바꿀수록 붕괴\n(95% → 30% 아래)",
            fontsize=9, color="#555", rotation=-18)
    ax.set_ylabel("성공률 (%)"); ax.set_ylim(0, 105)
    ax.set_title("LIBERO (clean) → LIBERO-Plus (교란 축별) — 도식적 붕괴 프로파일")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_libero_plus.png"); plt.close(fig)
    print("\n[FIG3] LIBERO-Plus 도식 프로파일:", dict(zip([f.replace(chr(10),' ') for f in factors], perf)))


# ===========================================================================
# 그림 4 — A vs B 차이의 CI 겹침·유의성 (N=20/50/200)
# ===========================================================================
def fig4_AB_overlap():
    pA, pB = 0.70, 0.80
    Ns = [20, 50, 200]
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 4.2), sharey=True)
    for j, N in enumerate(Ns):
        kA, kB = round(pA * N), round(pB * N)
        loA, hiA = wilson(kA, N); loB, hiB = wilson(kB, N)
        # Fisher exact 두-비율 검정
        _, pval = stats.fisher_exact([[kB, N - kB], [kA, N - kA]])
        a = ax[j]
        a.errorbar(0, pA, yerr=[[pA - loA], [hiA - pA]], fmt="o", color=C["A"],
                   capsize=6, ms=8, label="모델 A")
        a.errorbar(1, pB, yerr=[[pB - loB], [hiB - pB]], fmt="s", color=C["B"],
                   capsize=6, ms=8, label="모델 B")
        # 겹침 음영
        ov_lo, ov_hi = max(loA, loB), min(hiA, hiB)
        if ov_hi > ov_lo:
            a.axhspan(ov_lo, ov_hi, xmin=0.1, xmax=0.9, color="gray", alpha=0.18)
        sig = "유의 (p<0.05)" if pval < 0.05 else "유의하지 않음"
        a.set_title(f"N={N}  (Fisher p={pval:.3f})\n{sig}",
                    fontsize=9.5, color=(C["cp"] if pval < 0.05 else C["plus"]))
        a.set_xticks([0, 1]); a.set_xticklabels(["A (70%)", "B (80%)"])
        a.set_ylim(0.3, 1.02); a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=8, loc="lower left"); a.set_ylabel("성공률 (95% Wilson CI)")
        print(f"[FIG4] N={N}: A[{loA:.3f},{hiA:.3f}] B[{loB:.3f},{hiB:.3f}] "
              f"겹침={'예' if ov_hi>ov_lo else '아니오'} Fisher p={pval:.4f}")
    fig.suptitle("A(70%) vs B(80%): N이 작으면 CI가 겹쳐 '더 낫다'를 말할 수 없다", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_AB_overlap.png"); plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("Lec 57 그림 생성 + 수치 확정")
    print("=" * 60)
    fig1_ci_vs_N()
    fig2_rank_noise()
    fig3_libero_plus()
    fig4_AB_overlap()
    print("\n완료: fig1_ci_vs_N.png fig2_rank_noise.png fig3_libero_plus.png fig4_AB_overlap.png")
