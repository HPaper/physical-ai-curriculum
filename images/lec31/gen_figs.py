# Lec 31 그림 생성 스크립트 — Transformer 완성 (residual/LN/PE/mask/KV캐시)
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')
rng = np.random.default_rng(0)

# ============================================================================
# fig1: Transformer 블록 다이어그램 (pre-LN, residual, attention, FFN)
# ============================================================================
def fig1_block():
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis('off')
    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                    fc=fc, ec=ec, lw=1.6))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10.5)
    def arrow(x0,y0,x1,y1,c='#333',lw=1.8,ls='-'):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',
                    mutation_scale=14, lw=lw, color=c, ls=ls))
    xc = 5.0
    # 입력
    box(xc-1.7, 0.4, 3.4, 0.8, "입력 x  (임베딩 + PE)", '#f3eede', '#8a7a4a')
    # residual stream 수직선 (오른쪽 굵은 화살표)
    ax.plot([xc+2.6, xc+2.6], [1.0, 12.6], color='#c0392b', lw=3.0, zorder=0)
    ax.text(xc+2.85, 6.8, 'residual\nstream\n(기울기\n고속도로)', color='#c0392b',
            fontsize=9, va='center', rotation=0)
    # --- sublayer 1: LN -> Attn -> +
    arrow(xc, 1.2, xc, 1.9)
    box(xc-1.4, 1.9, 2.8, 0.75, "LayerNorm", '#e8f5e9', '#2e7d32')
    arrow(xc, 2.65, xc, 3.35)
    box(xc-2.0, 3.35, 4.0, 1.0, "Multi-Head Self-Attn\n(causal mask)", '#e1eefb', '#2c6fb0')
    arrow(xc, 4.35, xc, 5.0)
    # add node
    ax.add_patch(plt.Circle((xc, 5.4), 0.35, fc='white', ec='#c0392b', lw=2))
    ax.text(xc, 5.4, '＋', ha='center', va='center', fontsize=13, color='#c0392b')
    # skip 연결
    ax.plot([xc-2.6, xc-2.6], [1.0, 5.4], color='#c0392b', lw=2.2, ls='--', zorder=0)
    arrow(xc-2.6, 1.0, xc-2.6, 1.0)  # dummy
    ax.plot([xc-1.7, xc-2.6],[1.0,1.0], color='#c0392b', lw=2.2, ls='--')
    arrow(xc-2.6, 5.4, xc-0.35, 5.4, c='#c0392b', lw=2.2)
    arrow(xc, 5.75, xc, 6.3)
    ax.text(xc-2.9, 3.3, 'skip', color='#c0392b', fontsize=9, rotation=90, va='center')
    # --- sublayer 2: LN -> FFN -> +
    box(xc-1.4, 6.3, 2.8, 0.75, "LayerNorm", '#e8f5e9', '#2e7d32')
    arrow(xc, 7.05, xc, 7.7)
    box(xc-2.0, 7.7, 4.0, 1.1, "FFN\n$W_2\\,\\sigma(W_1 x)$", '#fde7e1', '#c0562c')
    arrow(xc, 8.8, xc, 9.5)
    ax.add_patch(plt.Circle((xc, 9.9), 0.35, fc='white', ec='#c0392b', lw=2))
    ax.text(xc, 9.9, '＋', ha='center', va='center', fontsize=13, color='#c0392b')
    ax.plot([xc-2.6, xc-2.6],[5.75,9.9], color='#c0392b', lw=2.2, ls='--', zorder=0)
    arrow(xc-2.6, 9.9, xc-0.35, 9.9, c='#c0392b', lw=2.2)
    arrow(xc, 10.25, xc, 11.0)
    # 출력
    box(xc-1.7, 11.0, 3.4, 0.8, "출력 x'  (다음 블록으로)", '#f3eede', '#8a7a4a')
    ax.text(xc, 13.3, "Transformer 블록 (pre-LN)\nx = x + Attn(LN(x)) ;  x = x + FFN(LN(x))",
            ha='center', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(OUT+'fig1_block_diagram.png', dpi=130, bbox_inches='tight'); plt.close()

# ============================================================================
# fig2: sinusoidal PE 히트맵 + 두 위치 내적의 상대거리 의존
# ============================================================================
def pe(pos, d):
    v = np.zeros(d); i = np.arange(d//2); denom = 10000.0 ** (2*i/d)
    v[0::2] = np.sin(pos/denom); v[1::2] = np.cos(pos/denom); return v

def fig2_pe():
    d = 64; P = 60
    M = np.stack([pe(p, d) for p in range(P)])   # (P, d)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(M.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax[0].set_xlabel('위치 pos'); ax[0].set_ylabel('차원 i (낮을수록 저주파)')
    ax[0].set_title('(a) sinusoidal PE 히트맵\n각 차원 = 다른 주파수의 시계')
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # (b) 내적 vs delta (base 여러개 겹침 -> 상대거리 의존)
    deltas = np.arange(0, 30)
    for base in [5, 15, 30]:
        dots = [pe(base, d)@pe(base+dl, d) for dl in deltas]
        ax[1].plot(deltas, dots, marker='o', ms=3, label=f'base={base}')
    ax[1].set_xlabel('상대거리 Δ = |pos_a − pos_b|')
    ax[1].set_ylabel('PE(pos_a)·PE(pos_b)')
    ax[1].set_title('(b) 두 위치 벡터의 내적\nbase가 달라도 Δ만의 함수 (곡선 겹침)')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT+'fig2_positional_encoding.png', dpi=130, bbox_inches='tight'); plt.close()

# ============================================================================
# fig3: causal mask 하삼각 + 그 결과 attention 가중치
# ============================================================================
def softmax_rows(S):
    S = S - S.max(-1, keepdims=True); e = np.exp(S); return e/e.sum(-1, keepdims=True)

def fig3_mask():
    r = np.random.default_rng(0)
    T, d = 8, 16
    X = r.standard_normal((T, d))
    scores = X@X.T/np.sqrt(d)
    mask = np.triu(np.ones((T,T)), 1).astype(bool)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    # (a) mask 자체
    Mvis = np.where(mask, -np.inf, 0.0)
    ax[0].imshow(np.where(mask, 0, 1), cmap='Greys_r', vmin=0, vmax=1)
    ax[0].set_title('(a) causal mask M\n하삼각 = 볼 수 있음(흰), 미래 = −∞(검)')
    ax[0].set_xlabel('키 위치 j (본다)'); ax[0].set_ylabel('쿼리 위치 i')
    for i in range(T):
        for j in range(T):
            ax[0].text(j,i,'−∞' if mask[i,j] else '0', ha='center',va='center',
                       fontsize=7, color='#c0392b' if mask[i,j] else '#2e7d32')
    # (b) full attention
    Af = softmax_rows(scores)
    im1 = ax[1].imshow(Af, cmap='viridis', vmin=0, vmax=1)
    ax[1].set_title('(b) 마스크 없는 attention\n미래도 봄 (상삼각 ≠ 0)')
    ax[1].set_xlabel('키 j'); ax[1].set_ylabel('쿼리 i'); fig.colorbar(im1, ax=ax[1], fraction=0.046)
    # (c) causal attention
    sm = scores.copy(); sm[mask]=-np.inf
    Ac = softmax_rows(sm)
    im2 = ax[2].imshow(Ac, cmap='viridis', vmin=0, vmax=1)
    ax[2].set_title('(c) causal attention\n하삼각만 (미래 가중치 = 0)')
    ax[2].set_xlabel('키 j'); ax[2].set_ylabel('쿼리 i'); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    plt.tight_layout(); plt.savefig(OUT+'fig3_causal_mask.png', dpi=130, bbox_inches='tight'); plt.close()

# ============================================================================
# fig4: KV 캐시 증분 도식 + 스텝당 비용 O(n^2) vs O(n)
# ============================================================================
def fig4_kvcache():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    # (a) 도식: 스텝마다 새 토큰의 K,V만 append
    axk = ax[0]; axk.set_xlim(0,7); axk.set_ylim(0,6); axk.axis('off')
    axk.set_title('(a) KV 캐시: 이미 계산한 K,V 재사용\n새 토큰은 자기 K,V만 추가')
    for step in range(4):
        y = 4.6 - step*1.25
        for j in range(step+1):
            fc = '#c9e3f7' if j < step else '#f6c9a8'
            axk.add_patch(FancyBboxPatch((0.6+j*0.85, y), 0.75, 0.55,
                boxstyle="round,pad=0.03", fc=fc, ec='#555', lw=1))
            axk.text(0.6+j*0.85+0.37, y+0.27, f'K{j}\nV{j}', ha='center', va='center', fontsize=7)
        axk.text(0.2, y+0.27, f't={step}', ha='right', va='center', fontsize=9)
        axk.text(0.6+(step+1)*0.85+0.1, y+0.27, '← 새로 계산' , ha='left', va='center',
                 fontsize=8, color='#c0562c')
    axk.text(3.5, 0.2, '파랑=캐시(재사용)  주황=이번 스텝 새 항목', ha='center', fontsize=8.5)
    # (b) 비용 곡선
    n = np.arange(1, 33)
    full = n*(n+1)/2       # 매 스텝 전체 재계산 누적
    cache = n              # 스텝당 O(n) (K·V 1개 + 어텐션 n)
    ax[1].plot(n, full, 'o-', ms=4, lw=3.2, color='#c0392b', zorder=2,
               label='재계산: K·V 프로젝션 누적 O(n²)')
    ax[1].plot(n, np.cumsum(np.ones_like(n)*1), 's-', ms=3, color='#2c6fb0', zorder=4,
               label='KV 캐시: K·V 프로젝션 O(n) (없애는 비용)')
    # 어텐션 내적 총합도 O(n²)=재계산 곡선과 동일한 값 → 빨강 위에 초록 파선을
    # 겹쳐 그려 "둘 다 O(n²)로 일치"를 보인다(초록 파선 사이로 빨강이 비친다).
    ax[1].plot(n, np.cumsum(n), '--', lw=1.6, color='#2e7d32', zorder=3,
               label='어텐션 내적 총합 O(n²) (캐시해도 불가피)')
    ax[1].set_xlabel('생성한 토큰 수 n'); ax[1].set_ylabel('누적 연산(프록시)')
    ax[1].set_title('(b) KV 캐시가 없애는 비용\nK,V 재계산 O(n²)→O(n), 값은 동일')
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT+'fig4_kv_cache.png', dpi=130, bbox_inches='tight'); plt.close()

fig1_block(); fig2_pe(); fig3_mask(); fig4_kvcache()
print("saved:", *[f for f in __import__('os').listdir(OUT) if f.endswith('.png')])
