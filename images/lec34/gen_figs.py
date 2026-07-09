# Lec 34 그림 생성 스크립트 — ViT: 이미지를 패치 토큰으로
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 ViT/DINOv2/CLIP 모델·GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

rng = np.random.default_rng(0)
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공통: 토이 "이미지" 하나 — 밝은 사각형 물체 + 대각선 엣지 (28강 톤 유지)
#   224x224가 아니라 그림용 축소판. 패치 구조를 눈으로 보기 위한 것.
# ============================================================================
def toy_image(S=48):
    img = np.zeros((S, S))
    yy, xx = np.mgrid[0:S, 0:S]
    # 밝은 사각형 물체
    img[int(S*0.25):int(S*0.6), int(S*0.30):int(S*0.70)] = 0.8
    # 대각선 밝은 띠 (엣지 구조)
    img += 0.5*np.exp(-((yy - xx*0.7 - S*0.15)**2)/(2*(S*0.06)**2))
    # 원형 밝은 반점
    img += 0.6*np.exp(-((yy-S*0.72)**2 + (xx-S*0.30)**2)/(2*(S*0.10)**2))
    img = np.clip(img, 0, 1)
    return img

# ============================================================================
# 그림 1: 이미지 → 패치 → 토큰 시퀀스 (patchify 도식)
#   S×S 이미지를 P×P 패치로 자르고, 각 패치를 선형사영해 토큰 벡터로.
#   토큰 수 N = (S/P)^2. "패치 = 단어"의 시각화.
# ============================================================================
S, P = 48, 16                 # 이미지 48x48, 패치 16x16 → grid 3x3 = 9 패치
G = S // P                    # grid 한 변 = 3
N = G * G                     # 토큰 수 = 9
img = toy_image(S)

fig = plt.figure(figsize=(12.5, 4.6))
# (a) 원본 이미지 + 패치 격자
ax1 = fig.add_axes([0.04, 0.12, 0.26, 0.76])
ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
for i in range(1, G):
    ax1.axhline(i*P-0.5, color='C1', lw=2)
    ax1.axvline(i*P-0.5, color='C1', lw=2)
# 패치 번호
for gy in range(G):
    for gx in range(G):
        ax1.text(gx*P + P/2, gy*P + P/2, f'{gy*G+gx+1}',
                 color='C3', fontsize=11, weight='bold', ha='center', va='center')
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title(f'(a) 이미지 {S}×{S}\n→ {P}×{P} 패치 {G}×{G}={N}개', fontsize=10)

# (b) 하나의 패치를 펼침 (flatten) → 선형사영
ax2 = fig.add_axes([0.36, 0.12, 0.22, 0.76])
ax2.axis('off')
patch = img[0:P, 0:P]                      # 패치 1
# 작은 패치 썸네일
axp = fig.add_axes([0.355, 0.62, 0.09, 0.24])
axp.imshow(patch, cmap='gray', vmin=0, vmax=1)
axp.set_xticks([]); axp.set_yticks([])
axp.set_title('패치 1', fontsize=8.5)
ax2.annotate('flatten\n$P^2\\cdot C$', xy=(0.5, 0.50), xytext=(0.5, 0.58),
             ha='center', fontsize=9, transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax2.text(0.5, 0.44,
         '$x_p \\in \\mathbb{R}^{P^2 C}=\\mathbb{R}^{768}$\n(16·16·3)',
         ha='center', va='center', fontsize=9, transform=ax2.transAxes,
         bbox=dict(boxstyle='round', fc='#e1eefb', ec='C0'))
ax2.annotate('선형사영 $E$', xy=(0.5, 0.20), xytext=(0.5, 0.30),
             ha='center', fontsize=9, transform=ax2.transAxes,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax2.text(0.5, 0.09,
         '토큰 $E x_p \\in \\mathbb{R}^{D}$',
         ha='center', va='center', fontsize=9.5, transform=ax2.transAxes,
         bbox=dict(boxstyle='round', fc='#e8f5e9', ec='C2'))

# (c) 토큰 시퀀스: [CLS] + N 패치토큰 + PE
ax3 = fig.add_axes([0.62, 0.12, 0.35, 0.76])
ax3.axis('off')
ax3.set_xlim(0, 11); ax3.set_ylim(0, 10)
# CLS 토큰
ax3.add_patch(Rectangle((0.3, 4), 0.9, 1.3, fc='#f3e5f5', ec='C4', lw=1.8))
ax3.text(0.75, 4.65, '[CLS]', ha='center', va='center', fontsize=8, weight='bold')
# 패치 토큰들
for k in range(N):
    x0 = 1.5 + k*0.95
    ax3.add_patch(Rectangle((x0, 4), 0.85, 1.3, fc='#e8f5e9', ec='C2', lw=1.4))
    ax3.text(x0+0.42, 4.65, f'{k+1}', ha='center', va='center', fontsize=8)
# PE 더하기
for k in range(N+1):
    x0 = 0.3 + k*0.95
    ax3.add_patch(Rectangle((x0, 2.4), 0.85, 1.0, fc='#fff3e0', ec='C1', lw=1.1))
ax3.text(5.5, 2.9, '+ positional encoding (31강)', ha='center', fontsize=8.5, color='C1')
ax3.text(5.5, 6.8, '토큰 시퀀스 (길이 $N{+}1$) → Transformer', ha='center',
         fontsize=10, weight='bold')
ax3.text(5.5, 6.0, '패치 = 단어, 이미지 = 문장', ha='center', fontsize=9, color='C3')
ax3.annotate('', xy=(5.5, 4.0), xytext=(5.5, 3.5),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax3.text(5.5, 1.6, 'CLS = 분류용 요약 · 패치토큰 = 조밀 특징', ha='center',
         fontsize=8, color='#555')
ax3.set_title('(c) $[\\mathrm{CLS};\\ \\mathrm{patches}] + \\mathrm{PE}$', fontsize=10)

fig.savefig(OUT + 'fig1_patchify_tokens.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 그림 2: 해상도 ↔ 토큰 수 ↔ attention FLOPs (N^2 스케일)
#   N = (H/P)(W/P) ∝ H^2.  attention 연산 ∝ N^2 ∝ H^4.
# ============================================================================
P_fix = 16
resolutions = np.array([112, 224, 336, 448, 672, 896])
N_tokens = (resolutions // P_fix)**2                     # 토큰 수
attn_cost = N_tokens.astype(float)**2                    # attention ∝ N^2 (self-attn 점수행렬)

# 224 기준 정규화
base_idx = np.where(resolutions == 224)[0][0]
tok_ratio = N_tokens / N_tokens[base_idx]
cost_ratio = attn_cost / attn_cost[base_idx]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 해상도 vs 토큰 수 (2배 해상도 → 4배 토큰)
axA.plot(resolutions, N_tokens, 'C0o-', lw=2, ms=6, label='토큰 수 $N=(H/P)^2$')
for r, n in zip(resolutions, N_tokens):
    axA.annotate(f'{n}', xy=(r, n), xytext=(0, 8), textcoords='offset points',
                 fontsize=8, ha='center', color='C0')
axA.axvline(224, color='gray', ls=':', lw=1)
axA.annotate('224²(=14×14=196)\n→ 448²(=28×28=784)\n해상도 2배 → 토큰 4배',
             xy=(448, 784), xytext=(300, 2600), fontsize=8.5, color='C3',
             arrowprops=dict(arrowstyle='->', color='C3'))
axA.set_xlabel('입력 해상도 $H=W$ (픽셀, $P{=}16$ 고정)')
axA.set_ylabel('토큰 수 $N$')
axA.set_title('(a) 해상도 ↔ 토큰 수\n$N \\propto H^2$ (2배 해상도 → 4배 토큰)')
axA.grid(alpha=0.3); axA.legend(fontsize=9)

# (b) 토큰 수 vs attention 비용 (N^2) — log-log
axB.loglog(N_tokens, cost_ratio, 'C3s-', lw=2, ms=6,
           label='attention 점수행렬 $\\propto N^2$')
axB.loglog(N_tokens, tok_ratio, 'C0o--', lw=1.6, ms=5,
           label='토큰 수 $\\propto N$ (선형 참조)')
# 기준점
axB.plot(N_tokens[base_idx], 1.0, 'k*', ms=15, zorder=5)
axB.annotate('224² 기준\n(N=196)', xy=(N_tokens[base_idx], 1.0),
             xytext=(60, 0.15), fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color='k'))
# 448 강조: 토큰 4배, attention 16배
i448 = np.where(resolutions == 448)[0][0]
axB.annotate(f'448²: 토큰 {tok_ratio[i448]:.0f}배,\nattention {cost_ratio[i448]:.0f}배',
             xy=(N_tokens[i448], cost_ratio[i448]), xytext=(300, 3),
             fontsize=8.5, color='C3',
             arrowprops=dict(arrowstyle='->', color='C3'))
axB.set_xlabel('토큰 수 $N$ (log)')
axB.set_ylabel('224² 대비 비율 (log)')
axB.set_title('(b) attention 연산 $O(N^2)$\n해상도 2배 = 토큰 4배 = attention 16배')
axB.grid(alpha=0.3, which='both'); axB.legend(fontsize=8.5, loc='upper left')
fig.tight_layout()
fig.savefig(OUT + 'fig2_resolution_scaling.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: patchify = stride-P·kernel-P conv 동치 (28강 회수)
#   비겹침 패치 임베딩 = 커널크기 P·스트라이드 P 합성곱. 수치로 오차 0 검증.
# ============================================================================
Cimg = 3
P3 = 4                      # 작은 패치로 명확히
S3 = 12                    # 12x12 이미지 → 3x3 grid
G3 = S3 // P3
img3 = rng.standard_normal((Cimg, S3, S3))            # (C,H,W) 랜덤 이미지
D3 = 5                     # 임베딩 차원
# 선형사영 행렬 E: (D, P^2*C)  — 패치 벡터 -> D차원 토큰
E = rng.standard_normal((D3, P3*P3*Cimg)) * 0.3
bias = rng.standard_normal(D3) * 0.1

# --- 방법 A: patchify + 선형사영 ---
tokens_A = np.zeros((G3*G3, D3))
idx = 0
for gy in range(G3):
    for gx in range(G3):
        patch = img3[:, gy*P3:(gy+1)*P3, gx*P3:(gx+1)*P3]   # (C,P,P)
        xp = patch.reshape(-1)                               # flatten (P^2*C,) : C-major
        tokens_A[idx] = E @ xp + bias
        idx += 1

# --- 방법 B: stride-P, kernel-P conv ---
# conv 커널 W_conv: (D, C, P, P). E의 각 행을 (C,P,P)로 reshape하면 커널.
W_conv = E.reshape(D3, Cimg, P3, P3)                        # E는 C-major로 폈으므로 (C,P,P)
tokens_B = np.zeros((G3*G3, D3))
idx = 0
for gy in range(G3):
    for gx in range(G3):
        region = img3[:, gy*P3:(gy+1)*P3, gx*P3:(gx+1)*P3]  # (C,P,P)
        for d in range(D3):
            tokens_B[idx, d] = np.sum(W_conv[d]*region) + bias[d]   # conv = 원소곱 합
        idx += 1

patchify_conv_err = np.max(np.abs(tokens_A - tokens_B))

fig, (axC, axD) = plt.subplots(1, 2, figsize=(11.5, 4.4))
# (a) 개념도: 패치 선형사영 vs conv
axC.axis('off')
axC.set_xlim(0, 10); axC.set_ylim(0, 10)
axC.text(5, 9.4, '두 계산은 같은 함수다', ha='center', fontsize=11, weight='bold')
# 위: patchify path
axC.add_patch(Rectangle((0.5, 6.3), 1.6, 1.6, fc='#eeeeee', ec='k'))
axC.text(1.3, 7.1, 'P×P\n패치', ha='center', va='center', fontsize=8.5)
axC.annotate('flatten', xy=(3.0, 7.1), xytext=(2.2, 7.1),
             fontsize=8, va='center', arrowprops=dict(arrowstyle='->'))
axC.add_patch(Rectangle((3.1, 6.5), 0.5, 1.2, fc='#e1eefb', ec='C0'))
axC.text(3.35, 7.1, '$x_p$', ha='center', va='center', fontsize=9)
axC.annotate('$E\\,x_p$', xy=(6.2, 7.1), xytext=(3.9, 7.1),
             fontsize=9, va='center', arrowprops=dict(arrowstyle='->'))
axC.add_patch(Rectangle((6.3, 6.5), 0.5, 1.2, fc='#e8f5e9', ec='C2'))
axC.text(6.55, 7.1, '토큰', ha='center', va='center', fontsize=7.5)
axC.text(0.5, 8.3, '① patchify + 선형사영', fontsize=9.5, color='C0', weight='bold')
# 아래: conv path
axC.add_patch(Rectangle((0.5, 2.3), 1.6, 1.6, fc='#eeeeee', ec='k'))
axC.text(1.3, 3.1, '입력\n(C,H,W)', ha='center', va='center', fontsize=8.5)
axC.annotate('conv\nk=P, s=P', xy=(6.2, 3.1), xytext=(2.4, 3.1),
             fontsize=8.5, va='center', ha='left',
             arrowprops=dict(arrowstyle='->'))
axC.add_patch(Rectangle((6.3, 2.5), 0.5, 1.2, fc='#e8f5e9', ec='C2'))
axC.text(6.55, 3.1, '토큰', ha='center', va='center', fontsize=7.5)
axC.text(0.5, 4.3, '② stride-P·kernel-P 합성곱', fontsize=9.5, color='C3', weight='bold')
axC.text(5, 0.9, f'수치 검증: 두 토큰의 최대 차 = {patchify_conv_err:.1e}\n(같은 커널 $W=E$, 비겹침 stride ⇒ 동일)',
         ha='center', fontsize=9, color='#333',
         bbox=dict(boxstyle='round', fc='#fffde7', ec='C1'))

# (b) 수치: 두 방법 토큰 산점도 (완전 일치 → 대각선)
axD.scatter(tokens_A.ravel(), tokens_B.ravel(), s=28, color='C2',
            edgecolor='k', lw=0.4, zorder=3)
lim = [tokens_A.min()-0.3, tokens_A.max()+0.3]
axD.plot(lim, lim, 'k--', lw=1, alpha=0.6, label='$y=x$ (완전 일치)')
axD.set_xlabel('patchify + 선형사영 토큰값')
axD.set_ylabel('stride-P conv 토큰값')
axD.set_title(f'(b) 두 방법의 토큰값이 정확히 일치\n(최대 차 {patchify_conv_err:.1e} — 부동소수 오차 수준)')
axD.grid(alpha=0.3); axD.legend(fontsize=9); axD.set_aspect('equal')
axD.set_xlim(lim); axD.set_ylim(lim)
fig.tight_layout()
fig.savefig(OUT + 'fig3_patchify_conv.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 4: DINOv2(기하·조밀대응) vs CLIP류(의미·언어정렬) 개념 대비
#   목적이 특징을 만든다. numpy 토이로 "무엇을 배우느냐"의 차이를 도식화.
#   왼쪽: 두 뷰 사이 조밀 대응(같은 부위끼리 매칭) — DINOv2/기하.
#   오른쪽: 이미지-텍스트 정렬(의미 클러스터) — CLIP/의미.
# ============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.8))

# --- 왼쪽: 조밀 대응 (기하) ---
# 같은 물체의 두 뷰(왼쪽 이미지, 오른쪽으로 이동+살짝 변형). 패치 특징이
# "같은 부위끼리" 가장 유사 → 대응선. 특징 매칭/광류/비주얼 서보잉의 언어.
axL.set_xlim(0, 10); axL.set_ylim(0, 8); axL.axis('off')
axL.set_title('DINOv2류 (자기지도) — 조밀 대응·기하\n(특징 매칭·광류·비주얼 서보잉의 언어)',
              fontsize=10)
# 두 뷰의 특징점 (같은 물체, 시점 변화)
pts_view1 = np.array([[1.5, 5.5], [2.8, 4.3], [1.9, 3.0], [3.2, 6.1], [2.3, 5.0]])
shift = np.array([5.0, -0.4])
# 시점 변화: 이동 + 약한 회전
theta = 0.18
Rm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
pts_view2 = (pts_view1 - pts_view1.mean(0)) @ Rm.T + pts_view1.mean(0) + shift
cols = ['C0', 'C1', 'C2', 'C3', 'C4']
for i in range(len(pts_view1)):
    axL.plot(*pts_view1[i], 'o', color=cols[i], ms=11, mec='k')
    axL.plot(*pts_view2[i], 'o', color=cols[i], ms=11, mec='k')
    # 대응선 (같은 부위끼리)
    axL.plot([pts_view1[i, 0], pts_view2[i, 0]],
             [pts_view1[i, 1], pts_view2[i, 1]], '-', color=cols[i], lw=1.3, alpha=0.6)
axL.text(2.3, 7.2, '뷰 1', ha='center', fontsize=9, weight='bold')
axL.text(7.3, 7.2, '뷰 2 (시점 변화)', ha='center', fontsize=9, weight='bold')
axL.text(5.0, 1.3, '같은 부위 = 같은 색\n패치 특징의 코사인 최대 → 대응(match)\n"어디가 어디에 대응하나"',
         ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round', fc='#e1eefb', ec='C0'))

# --- 오른쪽: 의미 클러스터 (언어 정렬) ---
# 여러 이미지 임베딩이 텍스트 프롬프트 임베딩 주변에 의미별로 모임.
# 개방어휘 인식(zero-shot): 텍스트 프롬프트를 분류기로.
axR.set_xlim(-3.5, 3.5); axR.set_ylim(-3.2, 3.2); axR.axis('off')
axR.set_title('CLIP류 (언어감독) — 의미·언어 정렬\n(개방어휘 인식 = zero-shot 분류)',
              fontsize=10)
# 세 의미 클러스터 (텍스트 앵커 + 이미지들)
anchors = {'"컵"': np.array([-1.8, 1.4]),
           '"로봇 팔"': np.array([1.9, 1.1]),
           '"식탁"': np.array([0.1, -1.9])}
acol = {'"컵"': 'C0', '"로봇 팔"': 'C3', '"식탁"': 'C2'}
r_img = np.random.default_rng(5)
for name, anc in anchors.items():
    # 텍스트 앵커 (별)
    axR.plot(*anc, '*', color=acol[name], ms=22, mec='k', zorder=5)
    axR.text(anc[0], anc[1]+0.42, name, ha='center', fontsize=9.5,
             weight='bold', color=acol[name])
    # 이미지 임베딩 (앵커 주변)
    imgs = anc + r_img.standard_normal((7, 2))*0.42
    axR.scatter(imgs[:, 0], imgs[:, 1], s=45, color=acol[name],
                alpha=0.55, edgecolor='k', lw=0.4, zorder=3)
axR.text(0.0, 2.9, '★ = 텍스트 프롬프트 임베딩,  ● = 이미지 임베딩',
         ha='center', fontsize=8.5)
axR.text(0.0, -2.85, '이미지·텍스트가 같은 공간에 정렬(대조학습, 35강)\n'
                     '새 라벨을 텍스트로 주면 분류 → zero-shot',
         ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round', fc='#ffebee', ec='C3'))
fig.tight_layout()
fig.savefig(OUT + 'fig4_dino_vs_clip.png', dpi=140)
plt.close(fig)

# ============================================================================
# WE-2 그림 자료용: 2D positional encoding이 인접 패치를 가깝게 (WE-2 대안)
#   & self-attention이 유사 패치를 묶는 히트맵.
#   여기서는 본문 WE-2 코드와 동일한 계산을 재현해 그림 5로 저장.
# ============================================================================
# 4x4 패치 그리드. 각 패치에 "특징 벡터"(색+위치) 부여.
# 유사 패치(같은 물체 조각)끼리 self-attention 가중치가 커지는지 히트맵.
Gp = 4
n_patch = Gp*Gp
# 각 패치 특징: 토이 이미지를 4x4 grid로 평균 풀 → 밝기 특징 + 2D PE
img5 = toy_image(48)
Pp = 48 // Gp
feat = np.zeros((n_patch, 4))
for gy in range(Gp):
    for gx in range(Gp):
        block = img5[gy*Pp:(gy+1)*Pp, gx*Pp:(gx+1)*Pp]
        k = gy*Gp+gx
        feat[k, 0] = block.mean()          # 평균 밝기
        feat[k, 1] = block.std()           # 질감(대비)
        feat[k, 2] = gx / (Gp-1)           # 2D PE (x)
        feat[k, 3] = gy / (Gp-1)           # 2D PE (y)

# self-attention 가중치 (내용 유사도 기반): softmax(F F^T / sqrt(d))
Fmat = feat.copy()
# 밝기·질감에 가중치를 크게 (내용), PE는 약하게 → 유사 물체조각끼리 묶임
w = np.array([3.0, 3.0, 0.6, 0.6])
Fw = Fmat * w
scores = Fw @ Fw.T / np.sqrt(Fw.shape[1])
scores -= scores.max(axis=1, keepdims=True)
A = np.exp(scores); A /= A.sum(axis=1, keepdims=True)

fig, (axP, axQ) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                gridspec_kw={'width_ratios': [1, 1.15]})
# (a) 패치 그리드 (밝기)
bright_grid = feat[:, 0].reshape(Gp, Gp)
axP.imshow(bright_grid, cmap='gray', vmin=0, vmax=1)
for gy in range(Gp):
    for gx in range(Gp):
        axP.text(gx, gy, f'{gy*Gp+gx+1}', color='C1', fontsize=10,
                 ha='center', va='center', weight='bold')
axP.set_xticks([]); axP.set_yticks([])
axP.set_title(f'(a) {Gp}×{Gp} 패치 그리드 (평균 밝기)\n패치 = 토큰', fontsize=10)

# (b) self-attention 가중치 히트맵
im = axQ.imshow(A, cmap='viridis', vmin=0)
axQ.set_xticks(range(n_patch)); axQ.set_yticks(range(n_patch))
axQ.set_xticklabels(range(1, n_patch+1), fontsize=6.5)
axQ.set_yticklabels(range(1, n_patch+1), fontsize=6.5)
axQ.set_xlabel('보이는 패치 (key)')
axQ.set_ylabel('보는 패치 (query)')
axQ.set_title('(b) self-attention 가중치 (행 합=1)\n유사 패치(밝은 물체조각)끼리 큰 가중치', fontsize=10)
fig.colorbar(im, ax=axQ, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT + 'fig5_patch_attention.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문과 정확히 일치해야 함
# ============================================================================
# 224/16 = 14 → 196 토큰; 448/16=28 → 784 토큰
n196 = (224 // 16)**2
n784 = (448 // 16)**2
print("=== 그림 파일 5개 생성 완료 ===")
print(f"[E1 토큰수] 224²/P16 = {224//16}×{224//16} = {n196} 토큰; "
      f"448²/P16 = {448//16}×{448//16} = {n784} 토큰 (토큰 {n784//n196}배)")
print(f"[E2 attention] N 4배 → attention 점수행렬 N² {(n784/n196)**2:.0f}배")
print(f"[그림3 patchify=conv] 두 방법 토큰 최대 차 = {patchify_conv_err:.2e} (동치)")
# self-attention 히트맵: 가장 밝은 물체 패치들끼리 attention이 큰지 검증
bright_patches = np.argsort(feat[:, 0])[-4:]      # 가장 밝은 4개 패치
avg_attn_bright = A[np.ix_(bright_patches, bright_patches)].mean()
avg_attn_all = A.mean()
print(f"[그림5 attention] 밝은 패치 4개 상호 attention 평균 {avg_attn_bright:.3f} "
      f"vs 전체 평균 {avg_attn_all:.3f} (유사 패치끼리 큰 가중치)")
# WE-2 대응(그림4): 조밀 대응 정확도 — 뷰1 각 점의 최근접이 같은 색인지
# (개념 검증: 시점변화 후에도 대응이 유지되는지)
print(f"[그림1] N={N} 토큰 (48²/P16, grid {G}×{G})")
