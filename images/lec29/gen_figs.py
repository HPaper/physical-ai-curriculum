# Lec 29 그림 생성 스크립트 — 토큰과 임베딩
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만, 결정론적)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')
rng = np.random.default_rng(0)

C_IO   = '#f3eede'; C_IOE  = '#8a7a4a'
C_SW   = '#e1eefb'; C_SWE  = '#2c6fb0'
C_TASK = '#e8f5e9'; C_TASKE= '#2e7d32'
C_BR   = '#f3e5f5'; C_BRE  = '#7b1fa2'
C_HOT  = '#c0392b'


# ============================================================================
# 공용: 아주 작은 BPE (바이트/문자 단위 병합) — E2·WE-1 재현
# ============================================================================
def bpe_train(words, num_merges):
    """words: {tuple(symbols): freq}. 가장 빈번한 인접쌍을 num_merges회 병합."""
    vocab = {tuple(w): f for w, f in words.items()}
    merges = []
    sizes = [count_tokens(vocab)]
    for _ in range(num_merges):
        pairs = {}
        for syms, f in vocab.items():
            for a, b in zip(syms[:-1], syms[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + f
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], -ord_key(p)))  # 동점이면 사전순 안정
        merges.append((best, pairs[best]))
        vocab = merge_vocab(vocab, best)
        sizes.append(count_tokens(vocab))
    return merges, sizes, vocab


def ord_key(p):
    return sum(ord(c) for c in (p[0] + p[1]))


def merge_vocab(vocab, pair):
    a, b = pair
    new = {}
    for syms, f in vocab.items():
        out, i = [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(syms[i]); i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + f
    return new


def count_tokens(vocab):
    return sum(len(syms) * f for syms, f in vocab.items())


# ============================================================================
# 그림 1: one-hot @ E = 룩업 (E1)
# ============================================================================
def fig1_lookup():
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.axis('off')
    V, d = 6, 4
    tokens = ['<pad>', 'the', 'cat', 'sat', 'on', 'mat']
    sel = 2  # 'cat'
    Emat = np.round(rng.standard_normal((V, d)) * 0.6, 2)

    def ry(i):
        return V - i  # 행 i의 하단 y (i=0 맨위)

    # one-hot 벡터 (세로)
    x0 = 0.5
    ax.text(x0 + 0.4, V + 1.15, 'one-hot  o  (1×V)', ha='center', fontsize=11, color=C_HOT)
    for i in range(V):
        val = 1 if i == sel else 0
        fc = C_HOT if val else 'white'
        tc = 'white' if val else '#555'
        ax.add_patch(Rectangle((x0, ry(i)), 0.8, 0.9, fc=fc, ec='#888'))
        ax.text(x0 + 0.4, ry(i) + 0.45, str(val), ha='center', va='center', color=tc, fontsize=11)
        ax.text(x0 - 0.15, ry(i) + 0.45, tokens[i], ha='right', va='center', fontsize=9, color='#333')

    # 임베딩 테이블 E (V×d)
    xE = 3.6
    ax.text(xE + d * 0.5, V + 1.15, '임베딩 테이블  E  (V×d, 학습됨)', ha='center', fontsize=11, color=C_SWE)
    for i in range(V):
        row_hot = (i == sel)
        for j in range(d):
            fc = '#fdecea' if row_hot else C_SW
            ec = C_HOT if row_hot else C_SWE
            ax.add_patch(Rectangle((xE + j, ry(i)), 0.95, 0.9, fc=fc, ec=ec, lw=1.6 if row_hot else 0.7))
            ax.text(xE + j + 0.47, ry(i) + 0.45, f'{Emat[i, j]:.2f}', ha='center', va='center',
                    fontsize=8.5, color='#0c3a63', fontweight='bold' if row_hot else 'normal')

    # 결과 임베딩 벡터 e = E[cat]  (선택된 행과 같은 높이)
    yr = ry(sel)
    xr = 9.1
    ax.text(xr + d * 0.5, V + 1.15, 'e = E[cat]  (1×d)', ha='center', fontsize=11, color=C_TASKE)
    for j in range(d):
        ax.add_patch(Rectangle((xr + j, yr), 0.95, 0.9, fc=C_TASK, ec=C_TASKE, lw=1.4))
        ax.text(xr + j + 0.47, yr + 0.45, f'{Emat[sel, j]:.2f}', ha='center', va='center', fontsize=8.5, color='#14421a')

    ax.add_patch(FancyArrowPatch((xE + d + 0.15, yr + 0.45), (xr - 0.15, yr + 0.45),
                                 arrowstyle='-|>', mutation_scale=18, color='#444', lw=1.6))
    ax.text((xE + d + xr) / 2, yr + 1.0, '행 선택\n(역전파로 학습)', ha='center', fontsize=9, color='#444')
    ax.add_patch(FancyArrowPatch((x0 + 0.85, yr + 0.45), (xE - 0.1, yr + 0.45),
                                 arrowstyle='-|>', mutation_scale=16, color=C_HOT, lw=1.4))
    ax.text((x0 + xE) / 2 + 0.2, yr - 0.1, 'o @ E', ha='center', fontsize=9.5, color=C_HOT)

    ax.set_xlim(-0.3, 14.0); ax.set_ylim(0.4, V + 1.7)
    ax.set_title('그림 1 · 이산 토큰 → 벡터: one-hot 곱은 임베딩 테이블의 행 선택 (E1)', fontsize=12.5)
    fig.tight_layout(); fig.savefig(OUT + 'fig1_embedding_lookup.png', dpi=130); plt.close(fig)


# ============================================================================
# 그림 2: BPE 병합 과정 — 병합 단계별 토큰 수 (E2·WE-1)
# ============================================================================
def fig2_bpe_merges():
    # 작은 코퍼스: 고전 BPE 예시(low/lower/newest/widest)에 '_'로 단어끝 표시.
    # '_'를 한 문자로 취급해 </w> 같은 다중문자 경계 토큰의 노이즈를 없앤다.
    corpus = {
        'low_': 5, 'lower_': 2, 'newest_': 6, 'widest_': 3, 'slow_': 4,
    }
    words = {tuple(w): f for w, f in corpus.items()}
    merges, sizes, final_vocab = bpe_train(words, 8)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    steps = np.arange(len(sizes))
    ax.plot(steps, sizes, 'o-', color=C_SWE, lw=2.2, ms=8)
    for s, v in zip(steps, sizes):
        ax.annotate(str(v), (s, v), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9, color='#0c3a63')
    ax.axhline(sizes[0], ls='--', color=C_HOT, lw=1.2)
    ax.text(0.1, sizes[0] + 0.6, f'문자 단위 시작 = {sizes[0]} 토큰', color=C_HOT, fontsize=9)
    ax.set_xlabel('병합 횟수'); ax.set_ylabel('코퍼스 총 토큰 수 (빈도 가중)')
    ax.set_title('(a) 병합할수록 시퀀스가 짧아진다')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axis('off')
    ax.text(0.02, 0.96, '학습된 병합 규칙 (빈도 순)', fontsize=11, color=C_SWE, transform=ax.transAxes)
    y = 0.86
    for k, ((a, b), cnt) in enumerate(merges):
        rule = f"{k+1}. ('{a}','{b}') -> '{a + b}'"
        # ASCII 규칙은 monospace, 한국어 '빈도'는 별도 텍스트(폰트 분리)로 글리프 경고 방지
        ax.text(0.04, y, rule, fontsize=9.6, family='monospace', transform=ax.transAxes, color='#333')
        ax.text(0.66, y, f'빈도 {cnt}', fontsize=9.4, transform=ax.transAxes, color='#777')
        y -= 0.095
    ax.text(0.02, 0.02, "OOV 없음: 새 단어도 문자·기존 병합으로 항상 분해 ('_'=단어끝)",
            fontsize=9, color=C_TASKE, transform=ax.transAxes)
    fig.suptitle('그림 2 · BPE = 빈도 기반 병합 (E2 · WE-1)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT + 'fig2_bpe_merges.png', dpi=130); plt.close(fig)
    return merges, sizes


# ============================================================================
# 그림 3: 2D 임베딩 공간 — 유추(analogy)와 군집 (E1·E3·WE-2)
# ============================================================================
def build_toy_embeddings():
    """의미 축을 손으로 설계한 8단어 2D 임베딩. gender축(x), royalty축(y)."""
    # (gender: 남 +1 / 여 -1,  royalty: 왕족 +1 / 평민 0)
    words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 'boy', 'girl']
    base = {
        'king':    (1.0, 1.0), 'queen':   (-1.0, 1.0),
        'man':     (1.0, 0.0), 'woman':   (-1.0, 0.0),
        'prince':  (1.0, 0.7), 'princess':(-1.0, 0.7),
        'boy':     (1.0, -0.6), 'girl':   (-1.0, -0.6),
    }
    E = np.array([base[w] for w in words], float)
    E += rng.standard_normal(E.shape) * 0.05  # 학습 잡음 흉내
    return words, E


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fig3_analogy():
    words, E = build_toy_embeddings()
    idx = {w: i for i, w in enumerate(words)}
    # king - man + woman ≈ ?
    q = E[idx['king']] - E[idx['man']] + E[idx['woman']]
    sims = np.array([cosine(q, E[i]) for i in range(len(words))])
    order = np.argsort(-sims)
    # 자기 자신(king,man,woman)은 유추 관례상 제외한 최근접
    exclude = {idx['king'], idx['man'], idx['woman']}
    best = next(i for i in order if i not in exclude)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    ax = axes[0]
    ax.scatter(E[:, 0], E[:, 1], s=80, color=C_SWE, zorder=3)
    for i, w in enumerate(words):
        ax.annotate(w, (E[i, 0], E[i, 1]), textcoords='offset points', xytext=(6, 5), fontsize=10)
    # 유추 평행사변형: man→king 벡터를 woman에 더함
    ax.add_patch(FancyArrowPatch(E[idx['man']], E[idx['king']], arrowstyle='-|>',
                                 mutation_scale=15, color=C_HOT, lw=1.8))
    ax.add_patch(FancyArrowPatch(E[idx['woman']], q, arrowstyle='-|>',
                                 mutation_scale=15, color=C_TASKE, lw=1.8, ls='--'))
    ax.scatter([q[0]], [q[1]], marker='*', s=260, color=C_HOT, zorder=4, edgecolor='k')
    ax.annotate('king−man+woman', (q[0], q[1]), textcoords='offset points', xytext=(8, -14),
                fontsize=9.5, color=C_HOT)
    ax.axhline(0, color='#ccc', lw=0.7); ax.axvline(0, color='#ccc', lw=0.7)
    ax.set_xlabel('축 1  (성별: 남 → 여)'); ax.set_ylabel('축 2  (지위: 왕족 →)')
    ax.set_title(f'(a) 벡터 산술: 최근접 = {words[best]!r}  (cos={sims[best]:.3f})')
    ax.grid(alpha=0.25)

    ax = axes[1]
    order_full = np.argsort(-sims)
    bars = [words[i] for i in order_full]
    vals = [sims[i] for i in order_full]
    cols = [C_HOT if i == best else (C_IOE if i in exclude else C_SWE) for i in order_full]
    ax.barh(range(len(bars)), vals, color=cols)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels(bars)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
    ax.set_xlabel('질의 벡터와의 코사인 유사도')
    ax.set_title('(b) 코사인 순위 (회색=질의 성분, 제외)')
    ax.set_xlim(min(vals) - 0.15, 1.05)
    fig.suptitle('그림 3 · 임베딩 공간에서 의미는 기하가 된다 (E1·E3·WE-2)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT + 'fig3_embedding_analogy.png', dpi=130); plt.close(fig)
    return words, E, best, sims


# ============================================================================
# 그림 4: 텍스트 토큰 vs 행동 토큰 (RT-2 · FAST) — 복선
# ============================================================================
def fig4_text_vs_action():
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.4))

    # (상) 텍스트 파이프라인
    ax = axes[0]; ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 2)
    stages = ['"pick the cup"\n(문자열)', 'BPE\n토크나이저', '토큰 ID\n[318, 262, 6508]',
              '임베딩 E\n(V×d 룩업)', '벡터 시퀀스\n→ Transformer']
    cols = [C_IO, C_SW, C_SW, C_SW, C_TASK]; ecs = [C_IOE, C_SWE, C_SWE, C_SWE, C_TASKE]
    x = 0.3
    for s, c, e in zip(stages, cols, ecs):
        ax.add_patch(Rectangle((x, 0.55), 2.0, 0.9, fc=c, ec=e, lw=1.4))
        ax.text(x + 1.0, 1.0, s, ha='center', va='center', fontsize=9)
        if x > 0.4:
            ax.add_patch(FancyArrowPatch((x - 0.25, 1.0), (x - 0.02, 1.0), arrowstyle='-|>',
                                         mutation_scale=13, color='#555', lw=1.3))
        x += 2.35
    ax.text(0.3, 1.75, '텍스트 (29강)', fontsize=11.5, color=C_SWE, fontweight='bold')

    # (하) 행동 파이프라인 (RT-2 / FAST)
    ax = axes[1]; ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 2)
    stages = ['액션 궤적\nΔx,Δy,…,grip\n(연속 실수)', 'FAST:\nDCT→양자화', '이산 코드\n[142, 88, 3]',
              '동일 임베딩 E\n(어휘에 편입)', '벡터 시퀀스\n→ 같은 Transformer']
    cols = [C_IO, C_BR, C_BR, C_SW, C_TASK]; ecs = [C_IOE, C_BRE, C_BRE, C_SWE, C_TASKE]
    x = 0.3
    for s, c, e in zip(stages, cols, ecs):
        ax.add_patch(Rectangle((x, 0.55), 2.0, 0.9, fc=c, ec=e, lw=1.4))
        ax.text(x + 1.0, 1.0, s, ha='center', va='center', fontsize=9)
        if x > 0.4:
            ax.add_patch(FancyArrowPatch((x - 0.25, 1.0), (x - 0.02, 1.0), arrowstyle='-|>',
                                         mutation_scale=13, color='#555', lw=1.3))
        x += 2.35
    ax.text(0.3, 1.75, '로봇 행동 (RT-2 42강 · FAST 44강)  —  같은 토크나이저·임베딩 문법',
            fontsize=11.5, color=C_BRE, fontweight='bold')

    fig.suptitle('그림 4 · 언어의 토크나이저 기법이 궤적에 그대로 적용된다 (복선)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT + 'fig4_text_vs_action_tokens.png', dpi=130); plt.close(fig)


# ============================================================================
if __name__ == '__main__':
    fig1_lookup()
    merges, sizes = fig2_bpe_merges()
    words, E, best, sims = fig3_analogy()
    fig4_text_vs_action()
    print('=== fig2 BPE ===')
    print('token sizes per merge:', sizes)
    print('merges:', [(''.join(p), c) for p, c in merges])
    print('=== fig3 analogy ===')
    print('best =', words[best], 'cos =', round(sims[best], 4))
    print('all sims:', {w: round(float(s), 4) for w, s in zip(words, sims)})
    print('DONE: fig1_embedding_lookup, fig2_bpe_merges, fig3_embedding_analogy, fig4_text_vs_action_tokens')
