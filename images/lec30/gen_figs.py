# Lec 30 그림 생성 스크립트 — Attention 해부 (Q/K/V, self-attention, multi-head)
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만, 결정론적)
# 개념을 numpy 토이로 재현한다 — 실제 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')
rng = np.random.default_rng(42)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# 공통 토이: WE-2와 동일한 5토큰 self-attention (본문 수치와 일치해야 함)
#   X (5x8) → Q,K,V = X W_{Q,K,V} → softmax(QK^T/sqrt(d)) → weights → @V
# ============================================================================
T, DM = 5, 8
X = rng.standard_normal((T, DM))
Wq = rng.standard_normal((DM, DM)) / np.sqrt(DM)
Wk = rng.standard_normal((DM, DM)) / np.sqrt(DM)
Wv = rng.standard_normal((DM, DM)) / np.sqrt(DM)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
S = Q @ K.T / np.sqrt(DM)
W_full = softmax(S, axis=1)                      # (5,5) 전체(양방향) attention

mask = np.triu(np.ones((T, T)), 1).astype(bool)  # 상삼각 = 미래 = 가림
Sc = S.copy(); Sc[mask] = -np.inf
W_causal = softmax(Sc, axis=1)                    # causal(인과) attention

tok_labels = ['로봇이', '빨간', '블록을', '집어', '올린다']  # 5토큰 예시 문장


# ============================================================================
# 그림 1: Q/K/V attention 도식 — 내용기반 주소지정 파이프라인
# ============================================================================
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 7)


def box(x, y, w, h, text, fc, ec, fs=9.5, tc='black'):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc)


def arrow(x0, y0, x1, y1, c='gray', lw=1.6, style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=14, color=c, lw=lw))


# 입력 X
box(0.3, 3.0, 1.5, 1.0, '입력 X\n(토큰 임베딩)\n29강', '#f3eede', '#8a7a4a', 9)
# 세 사영
box(2.6, 5.1, 1.6, 0.9, 'Q = X·W_Q\n(질의: 무엇을 찾나)', '#e1eefb', '#2c6fb0', 8.5)
box(2.6, 3.05, 1.6, 0.9, 'K = X·W_K\n(키: 무엇을 가졌나)', '#e1eefb', '#2c6fb0', 8.5)
box(2.6, 1.0, 1.6, 0.9, 'V = X·W_V\n(값: 실제 내용)', '#e1eefb', '#2c6fb0', 8.5)
arrow(1.8, 3.7, 2.6, 5.5); arrow(1.8, 3.5, 2.6, 3.5); arrow(1.8, 3.3, 2.6, 1.45)
# 점수
box(5.0, 4.1, 1.9, 0.9, '$QK^{\\top}$\n(질의·키 유사도)', '#fff3d9', '#c8922a', 8.5)
arrow(4.2, 5.5, 5.0, 4.9); arrow(4.2, 3.5, 5.0, 4.35)
# 스케일 + softmax
box(7.3, 4.1, 2.0, 0.9, '$\\div\\sqrt{d_k}$ → softmax\n(행별 가중치, 합=1)', '#e8f5e9', '#2e7d32', 8.5)
arrow(6.9, 4.55, 7.3, 4.55)
# 가중합
box(9.7, 2.55, 2.0, 1.9, '가중합\n$\\Sigma\\, w_i V_i$\n= 문맥 벡터', '#f3e5f5', '#7b1fa2', 9.5)
arrow(9.3, 4.4, 9.9, 4.45)                     # weights -> combine
arrow(4.2, 1.45, 10.4, 2.5, c='#7b1fa2', lw=1.4)  # V -> combine
ax.text(6.6, 1.75, 'V는 값 자체로 흘러들어 가중평균된다',
        fontsize=8.5, color='#7b1fa2', ha='center')
# 상태 의존적 게인 스케줄링 비유 배너
ax.text(6.0, 6.6, 'Attention = 내용기반 주소지정 = 상태 의존적 게인 스케줄링 (17강)',
        fontsize=11, ha='center', weight='bold', color='#333')
ax.text(6.0, 0.35,
        '질의·키 유사도가 "지금 상황"을, softmax 가중치가 "어느 정보를 얼마나 섞을지"를 정한다',
        fontsize=9, ha='center', color='#555')
fig.tight_layout()
fig.savefig(OUT + 'fig1_qkv_schematic.png', dpi=140)
plt.close(fig)


# ============================================================================
# 그림 2: attention 가중치 히트맵 — 전체(양방향) vs causal(인과) 마스크
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9),
                               gridspec_kw={'wspace': 0.35})


def heat(ax, Wm, title, note):
    im = ax.imshow(Wm, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(tok_labels, fontsize=8.5, rotation=30, ha='right')
    ax.set_yticklabels(tok_labels, fontsize=8.5)
    ax.set_xlabel('보이는 대상 (key: 어디서 정보를 끌어오나)', fontsize=9)
    ax.set_ylabel('보는 주체 (query)', fontsize=9)
    for i in range(T):
        for j in range(T):
            v = Wm[i, j]
            if v > 1e-6:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7.5, color='white' if v < 0.5 else 'black')
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.42, note, transform=ax.transAxes, ha='center',
            fontsize=8.5, color='#444')
    return im


im = heat(ax1, W_full, '(a) 전체 self-attention (양방향)',
          '각 행의 합=1. 모든 토큰이 모든 토큰을 본다\n(순서 정보 없음 → 31강 PE 필요)')
heat(ax2, W_causal, '(b) causal mask (인과, 31강 예고)',
     '미래(상삼각)를 가림 → 하삼각만 남음\nLLM 생성이 왼→오로 도는 이유')
fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.02, label='attention 가중치')
fig.savefig(OUT + 'fig2_attention_heatmap.png', dpi=140, bbox_inches='tight')
plt.close(fig)


# ============================================================================
# 그림 3: multi-head — d를 h개로 쪼개 병렬 사영 → concat → W_O
# ============================================================================
H_HEADS = 2
DH = DM // H_HEADS
head_W = []
for i in range(H_HEADS):
    Wqi = rng.standard_normal((DM, DH)) / np.sqrt(DM)
    Wki = rng.standard_normal((DM, DH)) / np.sqrt(DM)
    Wvi = rng.standard_normal((DM, DH)) / np.sqrt(DM)
    Qi, Ki = X @ Wqi, X @ Wki
    Wi = softmax(Qi @ Ki.T / np.sqrt(DH), axis=1)
    head_W.append(Wi)
head_div = np.abs(head_W[0] - head_W[1]).mean()

fig = plt.figure(figsize=(11.5, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.35)
# 상단: 두 헤드의 attention 패턴이 다름
for i in range(H_HEADS):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(head_W[i], cmap='magma', vmin=0, vmax=1, aspect='equal')
    ax.set_title(f'head {i+1}\n($d_k$={DH} 부분공간)', fontsize=9)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(range(1, T + 1), fontsize=7)
    ax.set_yticklabels(range(1, T + 1), fontsize=7)
    if i == 0:
        ax.set_ylabel('query', fontsize=8)
# 상단 오른쪽: 개념 설명
axc = fig.add_subplot(gs[0, 2]); axc.axis('off')
axc.text(0.5, 0.85, '왜 여러 헤드?', ha='center', fontsize=10, weight='bold',
         transform=axc.transAxes)
axc.text(0.02, 0.62,
         '• 각 헤드가 서로 다른 관계를\n  포착 (문법·의미·거리…)\n'
         f'• 두 헤드 가중치 평균 차이\n  = {head_div:.3f} (다른 패턴)\n'
         '• d를 h로 쪼갤 뿐 총 연산량\n  은 단일 헤드와 비슷',
         va='top', fontsize=8.5, transform=axc.transAxes)
# 하단: split → concat → W_O 파이프라인 도식
axp = fig.add_subplot(gs[1, :]); axp.axis('off')
axp.set_xlim(0, 12); axp.set_ylim(0, 3)


def pbox(x, y, w, h, t, fc, ec, fs=8.5):
    axp.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.6))
    axp.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs)


def parrow(x0, y0, x1, y1):
    axp.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                  mutation_scale=12, color='gray', lw=1.5))


pbox(0.2, 1.0, 1.4, 1.0, 'X\n($d_m$=8)', '#f3eede', '#8a7a4a')
pbox(2.3, 1.9, 1.7, 0.85, 'head1: Q,K,V\n($d_k$=4)', '#e1eefb', '#2c6fb0')
pbox(2.3, 0.35, 1.7, 0.85, 'head2: Q,K,V\n($d_k$=4)', '#fde6e6', '#c0392b')
parrow(1.6, 1.7, 2.3, 2.3); parrow(1.6, 1.3, 2.3, 0.75)
pbox(4.7, 1.9, 1.6, 0.85, '$attn_1$ → $z_1$\n(5×4)', '#e8f5e9', '#2e7d32')
pbox(4.7, 0.35, 1.6, 0.85, '$attn_2$ → $z_2$\n(5×4)', '#e8f5e9', '#2e7d32')
parrow(4.0, 2.3, 4.7, 2.3); parrow(4.0, 0.75, 4.7, 0.75)
pbox(7.0, 1.0, 1.7, 1.0, 'concat\n$[z_1 ; z_2]$\n(5×8)', '#fff3d9', '#c8922a')
parrow(6.3, 2.3, 7.0, 1.7); parrow(6.3, 0.75, 7.0, 1.3)
pbox(9.3, 1.0, 1.7, 1.0, '·W_O\n= MultiHead\n(5×8)', '#f3e5f5', '#7b1fa2')
parrow(8.7, 1.5, 9.3, 1.5)
axp.text(6.0, 2.75, 'multi-head = 여러 부분공간에서 병렬 attention → 이어붙여 섞기',
         ha='center', fontsize=9.5, weight='bold', color='#333')
fig.savefig(OUT + 'fig3_multihead.png', dpi=140, bbox_inches='tight')
plt.close(fig)


# ============================================================================
# 그림 4: √dₖ 스케일 효과 — 내적 분산 안정화와 softmax 날카로움
#   (a) 내적 std가 sqrt(d)로 자라고, /sqrt(d)가 이를 1로 되돌림
#   (b) 스케일 유무에 따른 softmax 엔트로피(날카로움) — 게인 스케줄링 비유
# ============================================================================
dims = np.array([2, 4, 8, 16, 32, 64, 128, 256])
raw_std, scaled_std = [], []
for d in dims:
    a = rng.standard_normal((4000, d))
    b = rng.standard_normal((4000, d))
    dot = np.sum(a * b, axis=1)
    raw_std.append(dot.std())
    scaled_std.append((dot / np.sqrt(d)).std())
raw_std = np.array(raw_std); scaled_std = np.array(scaled_std)

# softmax 엔트로피: 한 질의가 5개 키를 볼 때, d가 커지면 unscaled softmax가 붕괴
d_demo = 64
n_keys = 5
q_demo = rng.standard_normal(d_demo)
K_demo = rng.standard_normal((n_keys, d_demo))
logit = K_demo @ q_demo
p_unscaled = softmax(logit)
p_scaled = softmax(logit / np.sqrt(d_demo))
ent_unscaled = -np.sum(p_unscaled * np.log(p_unscaled + 1e-12))
ent_scaled = -np.sum(p_scaled * np.log(p_scaled + 1e-12))
ent_uniform = np.log(n_keys)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 내적 분산
ax1.loglog(dims, raw_std, 'C3o-', lw=2, ms=5, label='스케일 없음: std($Q^{\\top}K$)')
ax1.loglog(dims, np.sqrt(dims), 'k--', lw=1.3, label='이론 √d')
ax1.loglog(dims, scaled_std, 'C0s-', lw=2, ms=5, label='÷√d: std ≈ 1 (안정)')
ax1.axhline(1.0, color='C0', ls=':', lw=1)
ax1.set_xlabel('키/질의 차원 $d_k$ (log)')
ax1.set_ylabel('내적 표준편차 (log)')
ax1.set_title('(a) 내적 분산은 $d_k$로 자란다 → $\\sqrt{d_k}$가 되돌린다\n(스케일이 없으면 큰 차원에서 logit이 폭주)')
ax1.grid(alpha=0.3, which='both'); ax1.legend(fontsize=8.5)

# (b) softmax 날카로움
xk = np.arange(n_keys)
w = 0.38
ax2.bar(xk - w / 2, p_unscaled, w, color='C3', alpha=0.85,
        label=f'스케일 없음 (엔트로피 {ent_unscaled:.2f})')
ax2.bar(xk + w / 2, p_scaled, w, color='C0', alpha=0.85,
        label=f'÷√d (엔트로피 {ent_scaled:.2f})')
ax2.axhline(1 / n_keys, color='gray', ls=':', lw=1.2,
            label=f'균등(최대 엔트로피 {ent_uniform:.2f})')
ax2.set_xticks(xk); ax2.set_xticklabels([f'키{i+1}' for i in xk], fontsize=8.5)
ax2.set_ylabel('softmax 가중치')
ax2.set_title('(b) $d_k$=64에서 스케일 없으면 한 키로 쏠린다(argmax화)\n$\\div\\sqrt{d}$가 부드러운 혼합(게인 스케줄링)을 지킨다')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig4_scale_effect.png', dpi=140)
plt.close(fig)


# ============================================================================
# 본문 인용 수치 출력 — 본문/캡션과 반드시 일치
# ============================================================================
print("=== 그림 파일 4개 생성 완료 ===")
print(f"[WE-2 전체 attn 행합] {np.round(W_full.sum(1), 4).tolist()} (모두 1)")
print(f"[WE-2 causal 상삼각 0?] {np.allclose(W_causal[mask], 0.0)}, "
      f"행합 {np.round(W_causal.sum(1), 4).tolist()}")
print(f"[WE-2 causal 토큰4 행] {np.round(W_causal[3], 4).tolist()} "
      f"(미래 토큰5=0)")
print(f"[multi-head] 두 헤드 attention 가중치 평균 차이 = {head_div:.4f}")
print(f"[내적 분산] d=256: 스케일없음 std={raw_std[-1]:.3f} (√256=16), "
      f"÷√d std={scaled_std[-1]:.3f}")
print(f"[softmax 엔트로피] d=64, 5키: 스케일없음={ent_unscaled:.3f}, "
      f"÷√d={ent_scaled:.3f}, 균등최대={ent_uniform:.3f}")
print(f"[softmax 가중치] 스케일없음 최대={p_unscaled.max():.3f}, "
      f"÷√d 최대={p_scaled.max():.3f}")
