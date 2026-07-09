"""
Lec 35 — CLIP에서 SigLIP으로: 개념 재현용 CPU numpy 토이 그림.
matplotlib Agg / 결정론적 시드. 실제 대형 모델·GPU 없음.
그림:
  fig1_contrastive_space.png   대조 임베딩 공간(같은 쌍 당기고 다른 쌍 밀기)
  fig2_similarity_matrix.png   유사도 행렬 히트맵(대각 positive) + 온도 τ 효과
  fig3_zeroshot_pipeline.png   zero-shot 분류 파이프라인 + 최근접 정확도
  fig4_softmax_vs_sigmoid.png  CLIP softmax vs SigLIP sigmoid 손실의 배치크기 의존
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.rsplit('/', 1)[0]
BLUE, RED, GREEN, GRAY = '#2c6fb0', '#c0392b', '#2e7d32', '#888888'
rng = np.random.default_rng(0)


def l2norm(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# fig1: contrastive embedding space — matched pairs pulled together,
#       mismatched pushed apart, on the unit circle (shared space).
# ----------------------------------------------------------------------
def fig1():
    # 3 concepts placed on the unit circle; image & text of same concept nearby.
    ang = np.array([0.35, 2.3, 4.2])          # concept anchor angles (rad)
    names = ['개(dog)', '자동차(car)', '컵(cup)']
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))

    # (a) BEFORE: random — image & text scattered
    ax[0].add_patch(plt.Circle((0, 0), 1, fill=False, color=GRAY, lw=1, ls='--'))
    rng2 = np.random.default_rng(3)
    for k in range(3):
        ia = rng2.uniform(0, 2*np.pi); ta = rng2.uniform(0, 2*np.pi)
        pi_, pt = (np.cos(ia), np.sin(ia)), (np.cos(ta), np.sin(ta))
        ax[0].scatter(*pi_, s=140, marker='o', color=BLUE, zorder=3, edgecolor='k', lw=0.6)
        ax[0].scatter(*pt, s=140, marker='^', color=RED, zorder=3, edgecolor='k', lw=0.6)
        ax[0].plot([pi_[0], pt[0]], [pi_[1], pt[1]], color=GRAY, lw=1.0, ls=':')
    ax[0].set_title('(a) 학습 전 — 이미지·텍스트가 흩어져 있다', fontsize=11)

    # (b) AFTER: aligned — matched image/text co-located, concepts separated
    ax[1].add_patch(plt.Circle((0, 0), 1, fill=False, color=GRAY, lw=1, ls='--'))
    for k, a in enumerate(ang):
        c = [BLUE, GREEN, '#8e44ad'][k]
        ia = a + rng.uniform(-0.08, 0.08); ta = a + rng.uniform(-0.08, 0.08)
        pi_, pt = (np.cos(ia), np.sin(ia)), (np.cos(ta), np.sin(ta))
        ax[1].scatter(*pi_, s=150, marker='o', color=c, zorder=3, edgecolor='k', lw=0.6)
        ax[1].scatter(*pt, s=150, marker='^', color=c, zorder=3, edgecolor='k', lw=0.6)
        # short green "pull" arrow between matched pair
        ax[1].annotate('', xy=pt, xytext=pi_,
                       arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.6))
        ax[1].text(1.18*np.cos(a), 1.18*np.sin(a), names[k], fontsize=10,
                   ha='center', va='center')
    # red "push" arcs between concepts
    for (u, v) in [(0, 1), (1, 2), (0, 2)]:
        mu = (ang[u]+ang[v])/2
        ax[1].annotate('', xy=(0.55*np.cos(ang[v]), 0.55*np.sin(ang[v])),
                       xytext=(0.55*np.cos(ang[u]), 0.55*np.sin(ang[u])),
                       arrowprops=dict(arrowstyle='<->', color=RED, lw=0.9, ls='--', alpha=0.5))
    ax[1].set_title('(b) 대조학습 후 — 맞는 쌍은 당기고(초록), 다른 개념은 밀린다(빨강)', fontsize=11)

    for a in ax:
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_aspect('equal')
        a.set_xticks([]); a.set_yticks([])
    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY, markeredgecolor='k', markersize=11, label='이미지 임베딩 f'),
           Line2D([0], [0], marker='^', color='w', markerfacecolor=GRAY, markeredgecolor='k', markersize=11, label='텍스트 임베딩 g')]
    ax[1].legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9)
    fig.suptitle('대조학습 = 공유 임베딩 공간에서 맞는 (이미지,텍스트)는 가깝게, 틀린 쌍은 멀게 (메트릭 러닝)',
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/fig1_contrastive_space.png', dpi=120)
    plt.close(fig)
    print('fig1 saved')


# ----------------------------------------------------------------------
# fig2: similarity matrix heatmap (diagonal = positives) + temperature effect
#       reuses the WE-1 3x3 matrix and shows softmax sharpening as tau shrinks.
# ----------------------------------------------------------------------
def fig2():
    S = np.array([[0.9, 0.2, 0.1],
                  [0.3, 0.8, 0.2],
                  [0.1, 0.3, 0.7]])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))

    # (a) raw similarity matrix
    im = ax[0].imshow(S, cmap='viridis', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax[0].text(j, i, f'{S[i,j]:.1f}', ha='center', va='center',
                       color='white' if S[i, j] < 0.55 else 'black', fontsize=11)
    ax[0].add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec='none'))
    for d in range(3):
        ax[0].add_patch(plt.Rectangle((d-0.5, d-0.5), 1, 1, fill=False, ec=RED, lw=2.5))
    ax[0].set_title('(a) 유사도 행렬 $S=f\\,g^{\\top}/\\tau$\n대각선(빨강)=맞는 쌍(positive)', fontsize=10.5)
    ax[0].set_xlabel('텍스트 j'); ax[0].set_ylabel('이미지 i')
    ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    # (b),(c) row-0 softmax at two temperatures
    for k, tau in enumerate([1.0, 0.1]):
        axk = ax[k+1]
        for i in range(3):
            L = S[i] / tau
            p = np.exp(L - L.max()); p /= p.sum()
            axk.bar(np.arange(3) + i*0.0, p, width=0.6, alpha=0.0)  # spacing placeholder
        # show all 3 rows as grouped bars
        w = 0.25
        for i in range(3):
            L = S[i] / tau
            p = np.exp(L - L.max()); p /= p.sum()
            axk.bar(np.arange(3) + (i-1)*w, p, width=w,
                    label=f'행 {i} (정답 {i})',
                    color=[BLUE, GREEN, '#8e44ad'][i], edgecolor='k', lw=0.4)
        axk.axhline(1/3, color=GRAY, ls='--', lw=1, label='균등(1/3)')
        axk.set_ylim(0, 1.05); axk.set_xticks(range(3))
        axk.set_xlabel('텍스트 클래스 j')
        axk.set_title(f'({"bc"[k]}) 행별 softmax, τ={tau}\n'
                      + ('밋밋함 (p_00=0.51)' if tau == 1.0 else '날카로움 (p_00=1.00)'),
                      fontsize=10.5)
        if k == 0:
            axk.set_ylabel('softmax 확률')
            axk.legend(fontsize=7.5, loc='upper right')
    fig.suptitle('온도 τ는 softmax 날카로움을 지배한다 — τ↓ 이면 대각선(positive)에 확률이 집중된다',
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/fig2_similarity_matrix.png', dpi=120)
    plt.close(fig)
    print('fig2 saved')


# ----------------------------------------------------------------------
# fig3: zero-shot classification pipeline + nearest-text accuracy
# ----------------------------------------------------------------------
def fig3():
    D, C, N = 6, 4, 25
    rng3 = np.random.default_rng(0)
    protos = l2norm(rng3.standard_normal((C, D)))
    imgs, labels = [], []
    for c in range(C):
        x = protos[c] + 0.35*rng3.standard_normal((N, D))
        imgs.append(x); labels += [c]*N
    imgs = l2norm(np.vstack(imgs)); labels = np.array(labels)
    sims = imgs @ protos.T
    pred = sims.argmax(1)
    acc = (pred == labels).mean()

    fig = plt.figure(figsize=(12, 4.6))
    # (a) pipeline schematic (text boxes + arrows)
    axp = fig.add_axes([0.02, 0.08, 0.5, 0.84]); axp.axis('off')
    axp.set_xlim(0, 10); axp.set_ylim(0, 10)
    def box(x, y, w, h, txt, col):
        axp.add_patch(plt.Rectangle((x, y), w, h, fc=col, ec='k', lw=1.2, alpha=0.85))
        axp.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=9)
    box(0.3, 6.6, 3.0, 1.6, '테스트 이미지', '#dfeaf6')
    box(0.3, 1.4, 3.0, 3.6, '클래스 이름들\n"a photo of a\n{dog/car/cup/...}"', '#f6 e9 df'.replace(' ', ''))
    box(4.2, 6.6, 2.4, 1.6, '이미지\n인코더 f', '#e1eefb')
    box(4.2, 2.6, 2.4, 1.6, '텍스트\n인코더 g', '#fbeee1')
    box(7.4, 4.6, 2.3, 1.6, '공유 공간\n코사인 비교', '#e8f5e9')
    axp.annotate('', xy=(4.2, 7.4), xytext=(3.3, 7.4), arrowprops=dict(arrowstyle='->', lw=1.4))
    axp.annotate('', xy=(4.2, 3.4), xytext=(3.3, 3.2), arrowprops=dict(arrowstyle='->', lw=1.4))
    axp.annotate('', xy=(7.4, 5.7), xytext=(6.6, 7.4), arrowprops=dict(arrowstyle='->', lw=1.4, color=BLUE))
    axp.annotate('', xy=(7.4, 5.2), xytext=(6.6, 3.4), arrowprops=dict(arrowstyle='->', lw=1.4, color=RED))
    axp.text(8.55, 3.9, 'argmax_c\ncos(f,g_c)', ha='center', fontsize=8.5, color=GREEN)
    axp.text(5.0, 9.4, '(a) zero-shot 파이프라인 — 학습된 분류기 없이 클래스 이름을 인코딩해 비교',
             ha='center', fontsize=10.5)

    # (b) accuracy bar + a confusion-ish scatter of max-cos
    axb = fig.add_axes([0.60, 0.15, 0.36, 0.72])
    maxcos = sims.max(1)
    correct = pred == labels
    axb.scatter(labels[correct] + rng3.uniform(-0.15, 0.15, correct.sum()),
                maxcos[correct], s=22, color=GREEN, label='정답', zorder=3)
    axb.scatter(labels[~correct] + rng3.uniform(-0.15, 0.15, (~correct).sum()),
                maxcos[~correct], s=40, color=RED, marker='x', label='오답', zorder=3)
    axb.axhline(1/C, color=GRAY, ls=':', lw=1)
    axb.set_xticks(range(C)); axb.set_xlabel('참 클래스')
    axb.set_ylabel('최근접 텍스트와의 코사인')
    axb.set_title(f'(b) 최근접-텍스트 분류 정확도 = {acc:.2f}\n(4클래스, 100 이미지, 우연 0.25)',
                  fontsize=10.5)
    axb.legend(fontsize=9, loc='lower right')
    fig.suptitle('zero-shot = 개방어휘 인식: 고정 분류기가 아니라 클래스 이름 텍스트를 그때그때 인코딩',
                 fontsize=12, y=1.03)
    fig.savefig(f'{OUT}/fig3_zeroshot_pipeline.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'fig3 saved (acc={acc:.3f})')
    return acc


# ----------------------------------------------------------------------
# fig4: CLIP softmax vs SigLIP sigmoid — loss dependence on batch size
# ----------------------------------------------------------------------
def fig4():
    D, M = 6, 64
    rng4 = np.random.default_rng(1)          # WE-2 Part B와 동일 시드·순서
    z = l2norm(rng4.standard_normal((M, D)))
    img = l2norm(z + 0.15*rng4.standard_normal((M, D)))
    txt = l2norm(z + 0.15*rng4.standard_normal((M, D)))
    tau, bias = 0.07, -10.0

    def clip_loss(I, T):
        L = (I @ T.T) / tau
        def ce(L):
            m = L.max(1, keepdims=True)
            logZ = m[:, 0] + np.log(np.exp(L - m).sum(1))
            return (-(np.diag(L) - logZ)).mean()
        return 0.5*(ce(L) + ce(L.T))

    def siglip_loss(I, T):
        B = I.shape[0]
        Z = (I @ T.T) / tau + bias
        Y = -np.ones((B, B)); np.fill_diagonal(Y, 1.0)
        return -np.mean(np.log(1/(1+np.exp(-Y*Z))))

    Bs = [2, 4, 8, 16, 32, 64]
    cl = [clip_loss(img[:B], txt[:B]) for B in Bs]
    sl = [siglip_loss(img[:B], txt[:B]) for B in Bs]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax[0].plot(Bs, cl, 'o-', color=BLUE, lw=2, label='CLIP softmax (배치 전역 정규화)')
    ax[0].plot(Bs, sl, 's-', color=RED, lw=2, label='SigLIP sigmoid (쌍별 독립 이진)')
    ax[0].set_xscale('log', base=2); ax[0].set_xticks(Bs); ax[0].set_xticklabels(Bs)
    ax[0].set_xlabel('배치 크기 B'); ax[0].set_ylabel('손실')
    ax[0].set_title('(a) 손실의 배치크기 의존\nCLIP은 B와 함께 커지고, SigLIP은 평평', fontsize=10.5)
    ax[0].legend(fontsize=8.5, loc='upper left'); ax[0].grid(alpha=0.3)

    # (b) schematic: softmax normalizes across a row (coupled), sigmoid per-cell (independent)
    axg = ax[1]; axg.axis('off'); axg.set_xlim(0, 10); axg.set_ylim(0, 10)
    # CLIP: one row, arrow spanning it
    for j in range(4):
        c = GREEN if j == 0 else '#dfeaf6'
        axg.add_patch(plt.Rectangle((0.5+j*1.05, 7.2), 1.0, 1.0, fc=c, ec='k'))
    axg.annotate('', xy=(4.7, 7.7), xytext=(0.4, 7.7),
                 arrowprops=dict(arrowstyle='<->', color=BLUE, lw=2))
    axg.text(2.4, 8.6, 'CLIP: 행 전체에 softmax → B−1 음성과 경쟁 (배치에 결합)',
             fontsize=9, color=BLUE)
    axg.text(0.5, 6.7, 'positive는 하나, 나머지는 음성 — 배치가 크면 문제가 어려워짐',
             fontsize=8, color=GRAY)
    # SigLIP: 4x4 grid, each cell independent
    for i in range(4):
        for j in range(4):
            c = GREEN if i == j else '#f9e6e6'
            axg.add_patch(plt.Rectangle((0.5+j*1.05, 1.0+i*1.05), 1.0, 1.0, fc=c, ec='k'))
            sign = '+' if i == j else '−'
            axg.text(1.0+j*1.05, 1.5+i*1.05, sign, ha='center', va='center',
                     fontsize=9, color='k')
    axg.text(0.5, 5.5, 'SigLIP: 각 칸이 독립 이진(+1/−1) → 배치 전역 정규화 없음',
             fontsize=9, color=RED)
    fig.suptitle('SigLIP=sigmoid 손실: 각 (이미지,텍스트) 쌍을 독립 이진분류 → 거대 배치 불필요 (PaliGemma 백본, 36·44강)',
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/fig4_softmax_vs_sigmoid.png', dpi=120)
    plt.close(fig)
    print('fig4 saved  CLIP', np.round(cl, 4).tolist(), ' SigLIP', np.round(sl, 4).tolist())


if __name__ == '__main__':
    fig1()
    fig2()
    acc = fig3()
    fig4()
    print('all figures done, zero-shot acc =', round(acc, 3))
