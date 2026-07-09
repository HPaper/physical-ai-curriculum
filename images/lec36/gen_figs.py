# Lec 36 그림 생성 스크립트 — VLM 조립 (LLaVA 템플릿)
# 실행: cd images/lec36 && python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# 팔레트 (0강·28강·30강 계열과 통일)
C_IO   = '#f3eede'; C_IO_E   = '#8a7a4a'
C_SW   = '#e1eefb'; C_SW_E   = '#2c6fb0'
C_OP   = '#e8f5e9'; C_OP_E   = '#2e7d32'
C_ACT  = '#f3e5f5'; C_ACT_E  = '#7b1fa2'
C_PROJ = '#fde8d0'; C_PROJ_E = '#c76a1a'


def box(ax, xy, w, h, text, fc, ec, fs=10, lw=1.6, style='round,pad=0.02,rounding_size=0.03'):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color='#1a1a1a', zorder=5)


def arrow(ax, p0, p1, text='', color='#444', lw=1.8, rad=0.0, fs=8, dashed=False):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=14, lw=lw,
                        color=color, connectionstyle=f'arc3,rad={rad}',
                        linestyle='--' if dashed else '-', zorder=3)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
        ax.text(mx, my + 0.02, text, ha='center', va='bottom', fontsize=fs, color=color)


# ============================================================================
# 그림 1: encoder + projector + LLM 조립도 (= VLA 백본의 뼈대)
# ============================================================================
def fig1_assembly():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis('off')

    # 입력
    box(ax, (0.15, 3.35), 1.5, 1.0, '이미지\nH×W×3', C_IO, C_IO_E, fs=10)
    box(ax, (0.15, 0.85), 1.5, 1.0, '언어 지시\n"빨간 컵을 집어"', C_IO, C_IO_E, fs=9.5)

    # 비전 인코더
    box(ax, (2.1, 3.15), 1.85, 1.4, '비전 인코더\n(ViT, 34강)\nSigLIP/DINOv2\n35·28강', C_SW, C_SW_E, fs=9)
    # projector (급소, 강조색)
    box(ax, (4.45, 3.25), 1.5, 1.2, 'projector\n(작은 MLP)\n어댑터', C_PROJ, C_PROJ_E, fs=9.5, lw=2.4)

    # 텍스트 임베딩
    box(ax, (2.1, 0.75), 1.85, 1.2, '텍스트\n임베딩·토크나이저\n(29강)', C_SW, C_SW_E, fs=9)

    # 합쳐진 시퀀스
    box(ax, (6.35, 1.95), 1.55, 1.4, '한 시퀀스\n[t_v ; t_text]\n이미지+텍스트\n토큰', C_OP, C_OP_E, fs=9)

    # LLM
    box(ax, (8.35, 1.75), 1.7, 1.8, 'LLM\n(Transformer, 31강)\nself-attention\n으로 혼합', C_SW, C_SW_E, fs=9)

    # 출력 / 액션 헤드 (점선 — VLA 확장)
    box(ax, (8.55, 4.05), 1.35, 0.85, '텍스트 답변\n(VLM)', C_IO, C_IO_E, fs=9)
    box(ax, (8.55, 0.15), 1.35, 1.15, '액션 헤드/전문가\n(VLA · 42·44강)\n행동 청크', C_ACT, C_ACT_E, fs=8.5, lw=2.0)

    # 화살표
    arrow(ax, (1.65, 3.85), (2.1, 3.85))                            # img → enc
    arrow(ax, (3.95, 3.85), (4.45, 3.85), 'h_v (1152차원)', fs=7.5)   # enc → proj
    arrow(ax, (5.95, 3.85), (6.9, 3.35), 't_v (LLM d)', fs=7.5, rad=-0.2)  # proj → seq
    arrow(ax, (1.65, 1.35), (2.1, 1.35))                            # txt → embed
    arrow(ax, (3.95, 1.35), (6.9, 1.95), 't_text', fs=7.5, rad=0.15)  # embed → seq
    arrow(ax, (7.9, 2.65), (8.35, 2.65))                           # seq → LLM
    arrow(ax, (9.2, 3.55), (9.2, 4.05), '', rad=0.0)                # LLM → text
    arrow(ax, (9.2, 1.75), (9.2, 1.30), '', dashed=True, color=C_ACT_E)  # LLM → action (점선)

    ax.text(5.5, 5.0, 'E1:  h_v = Enc(img)   →   t_v = Proj(h_v)   →   y = LLM( [t_v ; t_text] )',
            ha='center', fontsize=11, color='#333', style='italic')
    ax.text(9.22, 0.02, '점선 = VLA 확장 (Part 10)', ha='center', fontsize=7.5, color=C_ACT_E)
    plt.tight_layout()
    fig.savefig(OUT + 'fig1_vlm_assembly.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 그림 2: LLaVA 2단계 학습 레시피 (무엇을 얼리고 무엇을 학습하나)
# ============================================================================
def fig2_two_stage():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    stages = [
        ('1단계: projector 정렬\n(feature alignment)',
         [('비전 인코더', 'frozen'), ('projector', 'train'), ('LLM', 'frozen')],
         '이미지-캡션 쌍으로\n"이미지 토큰↔단어" 사상만 학습\n(다리 놓기 · 인터페이스 계약 맞춤)'),
        ('2단계: 시각 instruction tuning\n(visual SFT · 33강)',
         [('비전 인코더', 'frozen'), ('projector', 'train'), ('LLM', 'train')],
         '지시-대화 데이터로\nprojector + LLM 함께 미세조정\n(대화·지시 따르기 학습)'),
    ]
    for ax, (title, comps, caption) in zip(axes, stages):
        ax.set_xlim(0, 4); ax.set_ylim(0, 5); ax.axis('off')
        ax.text(2, 4.75, title, ha='center', va='center', fontsize=11, weight='bold', color='#222')
        y0 = 3.4
        for name, state in comps:
            if state == 'frozen':
                fc, ec, tag = '#dfe6ee', '#7a8aa0', '[ 얼림 · frozen ]'
            else:
                fc, ec, tag = C_PROJ if name == 'projector' else C_SW, \
                              C_PROJ_E if name == 'projector' else C_SW_E, '[ 학습 · train ]'
            box(ax, (0.6, y0), 2.8, 0.8, f'{name}   —   {tag}', fc, ec, fs=10,
                lw=2.4 if (name == 'projector' and state == 'train') else 1.6)
            y0 -= 1.0
        ax.text(2, 0.55, caption, ha='center', va='center', fontsize=8.8, color='#444')
    fig.suptitle('LLaVA 2단계 레시피 — E2: 먼저 다리만, 그다음 지시 학습 (사전학습 인코더 재사용 = 부분 캘리브레이션)',
                 fontsize=10.5, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT + 'fig2_llava_two_stage.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 그림 3: 이미지 토큰 + 텍스트 토큰 한 시퀀스 self-attention 히트맵
#   projector를 거친 이미지 토큰이 LLM 안에서 텍스트 토큰과 섞이는 것을 보인다.
#   30강 self-attention 재사용. prefix-LM 마스크(이미지·프롬프트=양방향, 생성=인과).
# ============================================================================
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fig3_mixed_attention():
    rng = np.random.default_rng(36)
    n_img, n_txt = 4, 4           # 이미지 토큰 4 + 텍스트 토큰 4 = 시퀀스 8
    T = n_img + n_txt
    d = 16
    # 이미지 토큰과 텍스트 토큰이 각기 다른 통계를 갖되, projector가 이미지를 LLM 공간에 맞춤
    X = rng.standard_normal((T, d))
    Wq = rng.standard_normal((d, d)) / np.sqrt(d)
    Wk = rng.standard_normal((d, d)) / np.sqrt(d)
    Q, K = X @ Wq, X @ Wk
    S = Q @ K.T / np.sqrt(d)

    # prefix-LM 마스크 (PaliGemma식): [이미지 + 프롬프트]는 하나의 prefix로 양방향,
    # 생성되는 suffix 토큰(여기선 마지막 2개 TXT3·TXT4)만 인과(자기·과거만).
    n_prefix = n_img + 2          # 이미지4 + 프롬프트2 = prefix 6, 생성 suffix 2
    M = np.zeros((T, T))
    for i in range(n_prefix, T):          # suffix 위치는 미래를 못 본다
        M[i, i+1:] = -np.inf
    A_prefix = softmax(S + M)

    # 비교용: 순수 causal (모든 토큰 인과) — prefix-LM과의 차이를 보이기 위함
    Mc = np.where(np.triu(np.ones((T, T)), 1), -np.inf, 0.0)
    A_causal = softmax(S + Mc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ([f'IMG{i+1}' for i in range(n_img)]
              + ['PMT1', 'PMT2', 'GEN1', 'GEN2'])   # 프롬프트2 + 생성2
    for ax, A, ttl in [(axes[0], A_prefix, '(a) prefix-LM: prefix(IMG+PMT)=양방향, GEN만 인과'),
                       (axes[1], A_causal, '(b) 순수 causal: 전부 왼→오')]:
        im = ax.imshow(A, cmap='magma', vmin=0, vmax=A.max())
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('키 (누구를 보는가)', fontsize=9)
        ax.set_ylabel('질의 (누가 보는가)', fontsize=9)
        ax.set_title(ttl, fontsize=9.5)
        # 이미지/텍스트 경계선
        ax.axhline(n_img - 0.5, color='#00e5ff', lw=1.5)
        ax.axvline(n_img - 0.5, color='#00e5ff', lw=1.5)
        for i in range(T):
            for j in range(T):
                if A[i, j] > 0.01:
                    ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                            fontsize=6.2, color='w' if A[i, j] < 0.4 else 'k')
    # 텍스트 토큰이 이미지 토큰에 준 주의의 합 (섞임의 증거)
    txt_to_img = A_prefix[n_img:, :n_img].sum(axis=1).mean()
    fig.suptitle(f'이미지·텍스트 한 시퀀스 self-attention (30강 재사용) — '
                 f'텍스트 토큰이 이미지 토큰에 준 평균 주의 = {txt_to_img:.2f}',
                 fontsize=10, y=1.0)
    plt.tight_layout()
    fig.savefig(OUT + 'fig3_mixed_attention.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return txt_to_img


# ============================================================================
# 그림 4: VLM → VLA 매핑표 (같은 조립도, 다른 부품)
# ============================================================================
def fig4_vlm_to_vla():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')

    cols = ['VLM 백본', '비전 인코더', 'LLM', '이미지 토큰수*', '→ VLA (액션 헤드)']
    rows = [
        ['LLaVA-1.5', 'CLIP ViT-L/336', 'Vicuna 7B', '576', '(원형 · VLA 아님)'],
        ['PaliGemma 3B', 'SigLIP-So400m', 'Gemma 2B', '256', 'π0 (44강) · flow expert'],
        ['Eagle-2', 'SigLIP-2', 'SmolLM2', '64', 'GR00T (46강) · DiT flow'],
        ['SmolVLM', 'SigLIP(shape-opt)', 'SmolLM2', '~64**', 'SmolVLA (47강) · flow'],
    ]
    x_edges = [0.2, 2.5, 5.0, 7.2, 9.0, 11.8]
    y_top = 7.2
    rh = 1.25
    # 헤더
    for c, name in enumerate(cols):
        xc = (x_edges[c] + x_edges[c+1]) / 2
        box(ax, (x_edges[c], y_top), x_edges[c+1]-x_edges[c]-0.05, 0.8,
            name, '#333', '#111', fs=9)
        ax.texts[-1].set_color('white')
    # 행
    row_fc = [C_IO, C_SW, C_OP, C_ACT]
    for r, row in enumerate(rows):
        y = y_top - (r+1)*rh
        for c, val in enumerate(row):
            fc = row_fc[r] if c == 0 else '#fbfbfb'
            box(ax, (x_edges[c], y), x_edges[c+1]-x_edges[c]-0.05, rh-0.12,
                val, fc, '#999', fs=8.5)
    ax.text(6, 0.35,
            '*224px · patch14 기준 프레임당 이미지 토큰수 (해상도↑ → 토큰수↑ = 대역폭 트레이드오프).  '
            '**SmolVLM은 pixel-shuffle로 토큰 압축.',
            ha='center', fontsize=7.5, color='#555')
    fig.suptitle('E3: 같은 조립도(encoder+projector+LLM), 다른 부품 — "VLM을 알면 VLA 구조의 대부분을 안다"',
                 fontsize=10.5, y=0.99)
    plt.tight_layout()
    fig.savefig(OUT + 'fig4_vlm_to_vla_map.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig1_assembly()
    fig2_two_stage()
    t2i = fig3_mixed_attention()
    fig4_vlm_to_vla()
    print('그림 4개 저장 완료.')
    print(f'그림3 텍스트→이미지 평균 주의 = {t2i:.4f}')
