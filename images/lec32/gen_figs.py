# Lec 32 그림 생성 스크립트 — LLM의 탄생 (사전학습·스케일링·창발·ICL)
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만, 결정론적)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')
BLUE, ORANGE, GREEN, RED, GRAY = '#2c6fb0', '#d9822b', '#2e8b57', '#c0392b', '#888888'

# ============================================================================
# FIG 1: next-token 예측 + perplexity 도식 (WE-1 재현)
#   (a) 'the' 다음 토큰 분포 (bigram, add-1)  (b) perplexity = exp(CE) 막대
# ============================================================================
corpus = ("the cat sat on the mat . the dog sat on the log . "
          "the cat ran to the dog . the dog ran to the cat . ")
tokens = corpus.split()
vocab = sorted(set(tokens)); V = len(vocab)
tok2id = {t: i for i, t in enumerate(vocab)}
ids = np.array([tok2id[t] for t in tokens])
uni = np.bincount(ids, minlength=V).astype(float); uni_p = uni/uni.sum()
big = np.ones((V, V))
for a, b in zip(ids[:-1], ids[1:]): big[a, b] += 1
big_p = big/big.sum(axis=1, keepdims=True)
def ppl_big(seq):
    lp = sum(np.log(big_p[a, b]) for a, b in zip(seq[:-1], seq[1:]))
    return np.exp(-lp/(len(seq)-1))
def ppl_uni(seq):
    return np.exp(-np.sum(np.log(uni_p[seq]))/len(seq))
pb, pu = ppl_big(ids), ppl_uni(ids)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
a = tok2id['the']; p = big_p[a]; order = np.argsort(-p)
labels = [vocab[j] for j in order]; vals = p[order]
bars = ax1.bar(range(V), vals, color=[BLUE if v > 0.09 else GRAY for v in vals])
ax1.set_xticks(range(V)); ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.set_ylabel('P(다음 토큰 | "the")'); ax1.set_title('(a) next-token 분포 — "the" 다음\n예측=분포에서 생성 (검색 아님)')
ax1.axhline(1/V, color=RED, ls='--', lw=1, label=f'균일 1/V={1/V:.2f}')
ax1.annotate('cat / dog = 0.22', xy=(0.5, 0.222), xytext=(3.2, 0.17),
             fontsize=9, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
ax1.legend(fontsize=8)

models = ['균일\n(V=10)', '유니그램\n(빈도)', '바이그램\n(문맥1)']
ppls = [V, pu, pb]
cols = [RED, GRAY, BLUE]
bars2 = ax2.bar(models, ppls, color=cols)
for b, v in zip(bars2, ppls):
    ax2.text(b.get_x()+b.get_width()/2, v+0.15, f'{v:.2f}', ha='center', fontsize=10)
ax2.set_ylabel('perplexity = exp(cross-entropy)')
ax2.set_title('(b) perplexity = 평균 분기수\n문맥을 볼수록 헷갈림이 준다')
ax2.set_ylim(0, 11)
fig.tight_layout(); fig.savefig(OUT+'fig1_next_token_ppl.png', dpi=120); plt.close(fig)
print(f"fig1: bigram ppl={pb:.4f}, unigram ppl={pu:.4f}, V={V}")

# ============================================================================
# FIG 2: 스케일링 법칙 로그-로그 직선 (WE-2 재현) + 피팅 기울기 alpha
# ============================================================================
rng = np.random.default_rng(0)
def target(x): return np.sin(3*x)+0.5*np.sin(7*x)+0.3*np.cos(13*x)
Xtr = np.linspace(-1, 1, 400).reshape(-1, 1); ytr = target(Xtr).ravel()
Xte = rng.uniform(-1, 1, 2000).reshape(-1, 1); yte = target(Xte).ravel()
def rff(n, seed=1, gamma=6.0, ridge=1e-6):
    r = np.random.default_rng(seed)
    W = r.normal(0, np.sqrt(2*gamma), (1, n)); b = r.uniform(0, 2*np.pi, n)
    Ptr = np.cos(Xtr@W+b)*np.sqrt(2.0/n); Pte = np.cos(Xte@W+b)*np.sqrt(2.0/n)
    A = Ptr.T@Ptr+ridge*np.eye(n); w = np.linalg.solve(A, Ptr.T@ytr)
    return np.mean((Pte@w-yte)**2)
feats = np.array([2, 4, 8, 16, 32, 64, 128, 256])
losses = np.array([np.mean([rff(n, seed=s) for s in range(8)]) for n in feats])
logN, logL = np.log(feats.astype(float)), np.log(losses)
A = np.vstack([logN, np.ones_like(logN)]).T
slope, icpt = np.linalg.lstsq(A, logL, rcond=None)[0]; alpha = -slope
r2 = 1-np.sum((logL-A@np.array([slope, icpt]))**2)/np.sum((logL-logL.mean())**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.loglog(feats, losses, 'o-', color=BLUE, ms=7, label='토이 측정 손실')
fitline = np.exp(icpt)*feats.astype(float)**slope
ax1.loglog(feats, fitline, '--', color=RED, lw=1.8,
           label=f'거듭제곱 피팅\n$L\\propto N^{{-{alpha:.2f}}}$  ($R^2$={r2:.2f})')
ax1.set_xlabel('모델 용량 N (특징 수, 로그축)'); ax1.set_ylabel('테스트 손실 (로그축)')
ax1.set_title('(a) 스케일링 법칙: 로그-로그 직선\n용량↑ → 손실이 거듭제곱으로 감소')
ax1.grid(True, which='both', alpha=0.3); ax1.legend(fontsize=9)

# (b) 같은 데이터 선형축 — "직선이 아니다" 대비
ax2.plot(feats, losses, 'o-', color=BLUE, ms=7)
ax2.set_xlabel('모델 용량 N (선형축)'); ax2.set_ylabel('테스트 손실 (선형축)')
ax2.set_title('(b) 선형축에서 보면 그냥 수확체감\n예측력은 (a)의 로그-로그에서만 나온다')
ax2.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(OUT+'fig2_scaling_law.png', dpi=120); plt.close(fig)
print(f"fig2: alpha={alpha:.4f}, R2={r2:.4f}")

# ============================================================================
# FIG 3: perplexity vs 모델크기 + Chinchilla compute-optimal (D≈20N)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
# (a) perplexity = exp(loss)이므로 손실 거듭제곱 감소 -> ppl도 매끄럽게 감소
Ncont = np.logspace(np.log10(2), np.log10(256), 50)
Lcont = np.exp(icpt)*Ncont**slope
ppl_cont = np.exp(Lcont)          # perplexity = exp(cross-entropy)
ax1.semilogx(Ncont, ppl_cont, '-', color=GREEN, lw=2)
ax1.semilogx(feats, np.exp(losses), 'o', color=BLUE, ms=6)
ax1.set_xlabel('모델 용량 N (로그축)'); ax1.set_ylabel('perplexity = exp(손실)')
ax1.set_title('(a) perplexity vs 모델크기\n손실이 내려가면 분기수도 내려간다')
ax1.grid(True, which='both', alpha=0.3)
ax1.axhline(1.0, color=GRAY, ls=':', lw=1); ax1.text(3, 1.03, 'ppl=1 (완벽예측)', fontsize=8, color=GRAY)

# (b) Chinchilla: 고정 연산 C=6ND에서 손실 최소가 되는 (N,D) 균형, D≈20N
#   토이 손실모형 L(N,D)= a/N^p + b/D^q + e  (Hoffmann 형태), C=6ND 고정선
a_, b_, p_, q_, e_ = 400.0, 400.0, 0.34, 0.29, 0.0
def Lmod(N, D): return a_/N**p_ + b_/D**q_ + e_
Cbudget = 6*np.array([1e17, 1e18, 1e19])   # 세 연산 예산
fracs = np.linspace(0.02, 0.98, 200)
opt_N, opt_D = [], []
for C in Cbudget:
    ND = C/6.0
    # N = sqrt(ND)*10^s 스캔 -> D=ND/N
    Ns = np.sqrt(ND)*10**np.linspace(-1.5, 1.5, 400)
    Ds = ND/Ns
    Ls = Lmod(Ns, Ds)
    k = np.argmin(Ls); opt_N.append(Ns[k]); opt_D.append(Ds[k])
opt_N, opt_D = np.array(opt_N), np.array(opt_D)
ax2.loglog(opt_N, opt_D, 'o-', color=ORANGE, ms=8, label='compute-optimal (N*,D*)')
# 참조선 D=20N
nn = np.array([opt_N.min()*0.5, opt_N.max()*2])
ax2.loglog(nn, 20*nn, '--', color=GRAY, lw=1.5, label='D = 20·N (약 20토큰/파라미터)')
ratio = opt_D/opt_N
ax2.set_xlabel('최적 파라미터 N* (로그축)'); ax2.set_ylabel('최적 데이터 D* 토큰 (로그축)')
ax2.set_title(f'(b) Chinchilla compute-optimal\n토이 최적비 D*/N* ≈ {ratio.mean():.0f} (실제 ≈20)')
ax2.legend(fontsize=8); ax2.grid(True, which='both', alpha=0.3)
fig.tight_layout(); fig.savefig(OUT+'fig3_ppl_chinchilla.png', dpi=120); plt.close(fig)
print(f"fig3: toy D/N ratio = {ratio.mean():.2f} (target ~20)")

# ============================================================================
# FIG 4: 창발 vs 신기루 (Schaeffer) — 매끄러운 향상, 지표 선택이 만드는 착시
# ============================================================================
Nsz = np.logspace(2, 7, 60)
z = (np.log10(Nsz)-4.0)/0.9
per_tok = 0.05 + 0.94/(1+np.exp(-z))     # 매끄러운 per-token accuracy
L = 10
exact = per_tok**L                        # 비선형 지표 (전부 맞아야) -> 급점프처럼

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.semilogx(Nsz, per_tok, '-', color=BLUE, lw=2.2)
ax1.set_xlabel('모델 크기 N (로그축)'); ax1.set_ylabel('per-token 정확도 (연속 지표)')
ax1.set_title('(a) 연속 지표: 매끄럽게 향상\n어떤 급점프도 없다'); ax1.set_ylim(0, 1.02)
ax1.grid(True, which='both', alpha=0.3)
ax2.semilogx(Nsz, exact, '-', color=RED, lw=2.2)
ax2.set_xlabel('모델 크기 N (로그축)'); ax2.set_ylabel('exact-match 정확도 (L=10 전부 맞기)')
ax2.set_title('(b) 같은 모델, 비선형 지표\n→ "창발"처럼 급점프 (약 1.5 decade)')
ax2.set_ylim(0, 1.02); ax2.grid(True, which='both', alpha=0.3)
# 표시: 급점프 구간
def cross(y, t):
    i = np.argmax(y >= t); return Nsz[i] if np.any(y >= t) else np.nan
lo, hi = cross(exact, 0.05), cross(exact, 0.5)
ax2.axvspan(lo, hi, color=ORANGE, alpha=0.15)
ax2.text(lo*1.1, 0.6, f'"점프"\n{np.log10(hi/lo):.1f} decade', fontsize=9, color=ORANGE)
fig.tight_layout(); fig.savefig(OUT+'fig4_emergence_mirage.png', dpi=120); plt.close(fig)
print(f"fig4: jump 0.05->0.5 spans {np.log10(hi/lo):.2f} decades")

print("ALL FIGS DONE")
