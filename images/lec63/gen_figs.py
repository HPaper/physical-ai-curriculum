# Lec 63 그림 생성 스크립트 — 프론티어 지도: 4 물결 · latent action · System 2/1/0 경계 · VLA→WAM
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만. 모델 다운로드/GPU 없음)
#
# 이 스크립트가 하는 일:
#   (WE-1) latent action 토이 — 라벨 없는 (o_t, o_{t+1}) 쌍에서 latent action을 역추정
#          (inverse dynamics)하고, 그 latent가 "참 행동"에 따라 군집됨을 수치로 보인다.
#          라벨 없이 행동 구조를 회복한다는 LAPA/GO-1 아이디어의 최소 재현.
#   fig1: 4 물결 프론티어 지도 (2024~2026 타임라인 + 물결 배치)
#   fig2: latent action 파이프라인 (라벨없는 영상 → latent → 소량 실제행동 정렬) + 군집 산점도
#   fig3: System 2/1/0 경계 비교 (GR00T 2단 vs Helix 3단, 현재 폐루프는 번호 바깥)
#   fig4: VLA→WAM 수렴 / 효율 프론티어 (파라미터 vs 실측 신호)
# 본문이 인용하는 모든 정량 수치는 이 스크립트 출력이다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from scipy.linalg import lstsq

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# WE-1: latent action 토이 (inverse dynamics, 라벨 없는 사전학습)
# ----------------------------------------------------------------------------
# 설정: 잠재 상태 s in R^4 (예: 물체 2D 위치 + 그리퍼 2D 위치).
#   "참 행동" a in R^2가 있고(사람은 이 라벨을 모른다), 관측은 s를 고차원으로 올린
#   o = W s + noise 로 주어진다(영상 프레임의 축소판). 데이터는 (o_t, o_{t+1}) 쌍뿐 —
#   행동 라벨 a는 전혀 주어지지 않는다.
#   latent action z를 "두 연속 관측에서 무엇을 했는지"의 역추정으로 배운다:
#       z = g(o_t, o_{t+1})  ≈  선형 inverse dynamics로 (o_{t+1}-o_t)를 저차원으로 사영.
#   핵심 검증: 이렇게 라벨 없이 얻은 z가 참 행동 a에 따라 군집되는가?
#   → 군집되면 "라벨 없는 영상에서 행동 구조를 회복"한 것(LAPA/GO-1의 정량 근거).
# ============================================================================
print("=" * 70)
print("[WE-1] latent action 토이 — 라벨 없는 (o_t,o_{t+1})에서 행동 구조 회복")
print("=" * 70)

rng = np.random.default_rng(0)
Dz_true, Do = 4, 128         # 참 잠재 상태 차원, 관측(영상 축소) 차원(고차원)
K = 4                        # 참 행동 프리미티브 수 (예: 상/하/좌/우 밀기)
n_per = 300                  # 프리미티브당 전이 수
dt = 1.0

# 참 동역학: s_{t+1} = s_t + B a + process noise.  B는 행동을 상태변화로 옮기는 사상.
B = np.array([[1.0, 0.0],    # 물체 x는 행동 x에 반응
              [0.0, 1.0],    # 물체 y는 행동 y에 반응
              [0.6, 0.0],    # 그리퍼 x도 (약하게) 따라감
              [0.0, 0.6]])
# 4개 행동 프리미티브(사람 라벨 — 모델은 못 봄): 네 방향 단위 밀기
prims = np.array([[+1.0, 0.0], [-1.0, 0.0], [0.0, +1.0], [0.0, -1.0]])

# 관측 인코더(고정): o = s @ W^T + noise.  W는 잠재→관측 상승 사상.
W = rng.standard_normal((Do, Dz_true)) / np.sqrt(Dz_true)

S_t, S_tp1, A_true, lab = [], [], [], []
for k in range(K):
    s = rng.standard_normal((n_per, Dz_true)) * 0.8            # 다양한 시작 상태
    a = prims[k] + 0.12 * rng.standard_normal((n_per, 2))      # 프리미티브 + 소량 변이
    ds = a @ B.T + 0.05 * rng.standard_normal((n_per, Dz_true))  # 상태 변화
    S_t.append(s); S_tp1.append(s + ds); A_true.append(a); lab += [k] * n_per
S_t = np.vstack(S_t); S_tp1 = np.vstack(S_tp1)
A_true = np.vstack(A_true); lab = np.array(lab)

# 관측(영상 축소판) — 행동 라벨은 여기 어디에도 없다. 관측 노이즈가 크다(영상의 조명·질감).
obs_noise = 0.6
O_t = S_t @ W.T + obs_noise * rng.standard_normal((S_t.shape[0], Do))
O_tp1 = S_tp1 @ W.T + obs_noise * rng.standard_normal((S_t.shape[0], Do))

# --- latent action 학습 (VQ-VAE류를 선형으로 근사한 최소판) ---
# 1) inverse dynamics: 관측 변화 Δo = o_{t+1}-o_t 를 저차원 latent z로 사영.
#    Δo의 주성분(PCA) 상위 Dz_lat개 = "무엇이 바뀌었는가"의 압축 = latent action z.
#    (라벨 없이, 오직 관측 쌍만으로 — inverse dynamics의 비지도 판)
Dz_lat = 2
dO = O_tp1 - O_t
dO_c = dO - dO.mean(0, keepdims=True)
U, Sv, Vt = np.linalg.svd(dO_c, full_matrices=False)
P = Vt[:Dz_lat]                       # 관측변화 → latent 사영 (Do → Dz_lat)
Z = dO_c @ P.T                        # latent action z = g(o_t, o_{t+1})
print(f"관측 차원 Do={Do}, 참 행동 차원=2, latent z 차원={Dz_lat}")
print(f"Δo의 상위 {Dz_lat} 주성분이 담는 분산 비율: {(Sv[:Dz_lat]**2).sum()/(Sv**2).sum():.4f}")

# 2) 검증 A: latent z가 참 행동 프리미티브에 따라 군집되는가?
#    각 프리미티브 군집의 latent 중심 간 거리 vs 군집 내 산포로 분리도(silhouette류)를 잰다.
centers = np.array([Z[lab == k].mean(0) for k in range(K)])
within = np.mean([np.linalg.norm(Z[lab == k] - centers[k], axis=1).mean() for k in range(K)])
# 가장 가까운 다른 중심까지 거리
between = np.mean([min(np.linalg.norm(centers[k] - centers[j])
                       for j in range(K) if j != k) for k in range(K)])
sep_ratio = between / within
print(f"latent 군집: 군집내 산포={within:.4f}, 최근접 군집간 거리={between:.4f}, "
      f"분리도(간/내)={sep_ratio:.2f}")

# 최근접-중심 분류 정확도(라벨은 평가에만 사용 — 학습엔 안 씀)
pred = np.array([np.argmin(np.linalg.norm(centers - z, axis=1)) for z in Z])
# 군집 인덱스는 임의 순열일 수 있으므로 라벨↔군집 최적 매칭(작은 K라 완전탐색)
from itertools import permutations
best_acc = max((pred_perm := np.array([perm[p] for p in pred]),
                (pred_perm == lab).mean())[1] for perm in permutations(range(K)))
print(f"latent 최근접중심 분류 정확도(순열 정렬 후): {best_acc:.4f}  "
      f"(라벨 없이 4개 행동을 회복)")

# 3) 검증 B: 소량의 "실제 행동" 라벨로 latent→행동 디코더 정렬 (LAPA의 마지막 단계).
#    핵심: latent 사전학습(위 PCA)은 라벨 없는 전체 데이터에서 이미 끝났다.
#    이제 아주 적은 라벨(30개)만으로 저차원 z(2D)→행동(2D)을 맞추면 된다.
#    대조군은 라벨 없는 사전학습 없이, 같은 30개 라벨로 고차원 Δo(128D)→행동을 직접 회귀 —
#    128D를 30개로 맞추니 과소결정(underdetermined)이라 무너진다. 이 격차가 latent의 가치.
n_all = Z.shape[0]
idx = rng.permutation(n_all)
n_lab = 12                                       # 라벨 12개(전체의 ~1%)만
tr, te = idx[:n_lab], idx[n_lab:]
Zc = np.hstack([Z, np.ones((n_all, 1))])         # bias 항 (3D)
D, *_ = lstsq(Zc[tr], A_true[tr])                # latent→행동 디코더 (30 라벨, 3D 회귀)
A_hat = Zc @ D
r2_align = 1 - ((A_true[te] - A_hat[te]) ** 2).sum() / ((A_true[te] - A_true[te].mean(0)) ** 2).sum()
print(f"소량({n_lab}개) 실제행동 라벨로 latent(2D)→행동 정렬: 테스트 R²={r2_align:.4f}")

# 대조군: latent 없이 같은 30 라벨로 관측변화 Δo(128D) → 행동 직접 회귀(과소결정)
dOc = np.hstack([dO, np.ones((n_all, 1))])       # 129D
Dpix, *_ = lstsq(dOc[tr], A_true[tr])
A_hat_pix = dOc @ Dpix
r2_pix = 1 - ((A_true[te] - A_hat_pix[te]) ** 2).sum() / ((A_true[te] - A_true[te].mean(0)) ** 2).sum()
print(f"[대조] latent 없이 고차원 Δo(128D)→행동 직접회귀({n_lab} 라벨): 테스트 R²={r2_pix:.4f}")
print(f">>> latent 사전학습이 소량 라벨 효율을 회복: R² {r2_align:.3f} vs {r2_pix:.3f}")

# ============================================================================
# fig1: 4 물결 프론티어 지도 — 2024~2026 타임라인에 물결별 대표 사건 배치
# ============================================================================
# 각 물결: (이름, 색, [(날짜라벨, x위치[개월], 대표모델)])  x = 2024.06 기준 개월 오프셋
waves = {
    '① RL post-training': ('#c0392b', [
        ('2025.11', 17, 'π*0.6 / RECAP'), ('2026.03', 21, 'RL Tokens')]),
    '② latent action':    ('#2c6fb0', [
        ('2024.10', 4, 'LAPA'), ('2025.03', 9, 'GO-1 (ViLLA)')]),
    '③ world model 수렴':  ('#3a9a5a', [
        ('2025.01', 7, 'Cosmos'), ('2025.06', 12, 'V-JEPA 2'),
        ('2026.03', 21, 'GR00T N2 / DreamZero')]),
    '④ 효율화':            ('#8a5cb0', [
        ('2025.02', 8, 'OpenVLA-OFT'), ('2025.06', 12, 'SmolVLA')]),
}
fig, ax = plt.subplots(figsize=(12.4, 5.8))
wave_names = list(waves.keys())
for wi, wname in enumerate(wave_names):
    col, events = waves[wname]
    y = len(wave_names) - wi
    ax.plot([2, 23], [y, y], color=col, lw=2.0, alpha=0.35, zorder=1)
    ax.text(1.4, y, wname, ha='right', va='center', fontsize=10.5,
            color=col, weight='bold')
    for (dlabel, x, model) in events:
        ax.scatter(x, y, s=150, color=col, edgecolor='k', lw=0.9, zorder=4)
        ax.annotate(f"{model}\n{dlabel}", (x, y), xytext=(x, y + 0.24),
                    ha='center', fontsize=7.6, va='bottom')
# 배경: 공통 뿌리(π0 템플릿)과 화살표
ax.axvline(4, color='#888', ls=':', lw=1.0)
ax.text(4, 0.3, 'π0 (2024.10)\nVLM+flow 템플릿', ha='center', fontsize=7.4, color='#555')
ax.set_xlim(0, 23.5)
ax.set_ylim(0.0, len(wave_names) + 1.0)
xt = [4, 7, 10, 13, 16, 19, 22]
ax.set_xticks(xt)
ax.set_xticklabels(['2024.10', '2025.01', '2025.04', '2025.07',
                    '2025.10', '2026.01', '2026.04'], fontsize=8.5)
ax.set_yticks([])
ax.set_xlabel('시간')
ax.set_title('VLA 프론티어의 4 물결 (2024~2026) — 하나의 템플릿(π0) 위에서 병렬로 흐른다',
             fontsize=12)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + 'fig1_four_waves.png', dpi=140)
plt.close(fig)

# ============================================================================
# fig2: latent action 파이프라인 + 군집 산점도(WE-1 결과)
# ============================================================================
fig, (axP, axS) = plt.subplots(1, 2, figsize=(12.6, 5.2),
                               gridspec_kw={'width_ratios': [1.15, 1.0]})

# (a) 파이프라인 다이어그램
axP.set_xlim(0, 10); axP.set_ylim(0, 10); axP.axis('off')
axP.set_title('(a) latent action 사전학습 파이프라인 (LAPA/GO-1)', fontsize=11)
boxes = [
    (5.0, 8.7, '라벨 없는 영상\n(o_t, o_{t+1}) 쌍 — 행동 라벨 없음', '#e8eef7'),
    (5.0, 6.4, 'inverse dynamics 인코더\nz = g(o_t, o_{t+1})  (VQ-VAE류)', '#cfe0f2'),
    (5.0, 4.1, 'latent VLA 사전학습\nπ(z | o, 지시) — 인터넷 규모 영상', '#cfe0f2'),
    (5.0, 1.8, '소량 실제 행동으로 정렬\nz → a 디코더 (라벨 12개, ~1%)', '#d6efd9'),
]
for (x, y, txt, col) in boxes:
    axP.add_patch(Rectangle((x - 3.6, y - 0.7), 7.2, 1.4, facecolor=col,
                            edgecolor='k', lw=1.2))
    axP.text(x, y, txt, ha='center', va='center', fontsize=8.6)
for y0, y1 in [(8.0, 7.1), (5.7, 4.8), (3.4, 2.5)]:
    axP.annotate('', xy=(5.0, y1), xytext=(5.0, y0),
                 arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.8))
axP.text(6.9, 5.55, 'embodiment\n무관 · 라벨 0',
         fontsize=7.4, color='#2c6fb0', ha='left')
axP.text(6.9, 2.95, 'embodiment\n특화 · 소량',
         fontsize=7.4, color='#2a8a4a', ha='left')

# (b) latent z 산점도 — 참 행동 프리미티브로 색칠 (라벨 없이 얻은 z가 군집)
prim_names = ['밀기 +x', '밀기 -x', '밀기 +y', '밀기 -y']
prim_cols = ['#c0392b', '#e08a1e', '#2c6fb0', '#3a9a5a']
for k in range(K):
    axS.scatter(Z[lab == k, 0], Z[lab == k, 1], s=10, alpha=0.5,
                color=prim_cols[k], label=prim_names[k])
    axS.scatter(*centers[k], s=220, marker='*', color=prim_cols[k],
                edgecolor='k', lw=1.2, zorder=5)
axS.set_xlabel('latent $z_1$'); axS.set_ylabel('latent $z_2$')
axS.set_title(f'(b) 라벨 없이 얻은 latent z가 참 행동으로 군집\n분리도(간/내)={sep_ratio:.2f} · 순열정렬 분류정확도={best_acc:.2f}',
              fontsize=10)
axS.legend(fontsize=8, loc='best', framealpha=0.95)
axS.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + 'fig2_latent_action.png', dpi=140)
plt.close(fig)

# ============================================================================
# fig3: System 2/1/0 경계 비교 — GR00T(2단) vs Helix(3단), 폐루프는 번호 바깥
#   lec48 fig4와 정합하되, "감지→구동→로봇→환경 폐루프가 번호 바깥"을 명시적으로 그림.
# ============================================================================
fig, (axG, axH) = plt.subplots(1, 2, figsize=(12.6, 6.4))

def draw_stack(ax, title, blocks, boundary_after):
    n = len(blocks)
    ax.set_xlim(0, 10); ax.set_ylim(0, n * 1.8 + 2.2)
    ax.axis('off'); ax.set_title(title, fontsize=12, pad=10)
    for i, (lab_, hz, sysnum, col) in enumerate(blocks):
        y = (n - 1 - i) * 1.8 + 1.7
        ax.add_patch(Rectangle((1.3, y), 7.4, 1.35, facecolor=col,
                               edgecolor='k', lw=1.3, alpha=0.92))
        tag = f"[{sysnum}]" if sysnum else "[번호 밖]"
        ax.text(5.0, y + 0.86, lab_, ha='center', va='center',
                fontsize=9.6, color='white' if sysnum else '#333', weight='bold')
        ax.text(5.0, y + 0.34, f"{hz}   {tag}", ha='center', va='center',
                fontsize=8.0, color='white' if sysnum else '#333')
        if i < n - 1:
            ax.annotate('', xy=(5.0, y - 0.02), xytext=(5.0, y + 0.06),
                        arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.5))
    yb = (n - 1 - boundary_after) * 1.8 + 1.42
    ax.plot([0.5, 9.5], [yb, yb], color='#c0392b', ls='--', lw=2.3)
    ax.text(9.45, yb + 0.12, 'System 번호 여기서 끝', ha='right', va='bottom',
            fontsize=8.8, color='#c0392b', weight='bold')
    # 폐루프(감지→구동→로봇→환경)는 항상 번호 바깥 — 맨 아래 회색 배너
    ax.add_patch(Rectangle((0.5, 0.15), 9.0, 1.0, facecolor='#eeeeee',
                           edgecolor='#999', lw=1.0, ls='--'))
    ax.text(5.0, 0.65, '감지 → 구동 → 로봇 → 환경 (폐루프 · 물리) — 어느 번호에도 안 들어감',
            ha='center', va='center', fontsize=8.2, color='#666', style='italic')

draw_stack(axG, 'GR00T N1 — 2단 (번호는 S1에서 끝)',
           [('System 2  VLM', '~10 Hz · 이해·계획', 'S2', '#2c6fb0'),
            ('System 1  DiT', '~120 Hz · 행동 a 생성', 'S1', '#e08a1e'),
            ('저수준 관절 제어기', '~1 kHz · 서보(번호 밖)', None, '#c9c9c9'),
            ('전류루프 (FOC)', '~20 kHz · 펌웨어·물리', None, '#c9c9c9')],
           boundary_after=1)
draw_stack(axH, 'Figure Helix 02 — 3단 (S0까지 번호 안)',
           [('System 2  추론(VLM)', '7~9 Hz · 이해·추론', 'S2', '#2c6fb0'),
            ('System 1  트랜스포머', '200 Hz · 전신 관절목표', 'S1', '#e08a1e'),
            ('System 0  전신제어기', '1 kHz · 학습형(옛 C++ 대체)', 'S0', '#3a9a5a'),
            ('전류루프 (FOC)', '~20 kHz · 펌웨어·물리', None, '#c9c9c9')],
           boundary_after=2)
fig.suptitle('System 2/1/0은 표준이 아니다 — 같은 1kHz 전신 서보를 GR00T는 "번호 밖", Helix는 "System 0"으로 부른다',
             fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT + 'fig3_system_boundary.png', dpi=140)
plt.close(fig)

# ============================================================================
# fig4: (a) VLA→WAM 수렴 신호(RoboArena),  (b) 효율 프론티어(파라미터 vs 성능)
#   실측/회사발표 수치만 사용. 코드는 배치·정규화만 담당(수치를 만들지 않음).
# ============================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.0))

# (a) RoboArena Elo — DreamZero(WAM) vs π0.5(VLA)  [NVIDIA 발표, 2026.04]
ra_models = ['π0.5\n(VLA)', 'DreamZero\n(WAM, GR00T N2)']
ra_scores = [1622, 1750]
ra_cols = ['#2c6fb0', '#3a9a5a']
bars = axA.bar(ra_models, ra_scores, color=ra_cols, edgecolor='k', lw=0.8, width=0.55)
axA.set_ylim(1500, 1820)
axA.set_ylabel('RoboArena Elo (실세계, 높을수록↑)')
axA.set_title('(a) VLA→WAM 수렴 신호 — DreamZero가 π0.5를 앞섬\n(NVIDIA 발표 2026.04; +128 Elo)', fontsize=10)
for b, s in zip(bars, ra_scores):
    axA.text(b.get_x() + b.get_width() / 2, s + 8, str(s),
             ha='center', fontsize=10, weight='bold')
axA.text(0.5, 1780, '"pretrained to imagine,\nfine-tuned to act"', ha='center',
         fontsize=8, color='#3a9a5a', style='italic',
         transform=axA.get_xaxis_transform() if False else axA.transData)
axA.grid(alpha=0.25, axis='y')

# (b) 효율 프론티어: 파라미터(log) vs 대표 성공률/성능 (각 실측·회사발표)
#   점: (파라미터M, 성능%, 이름, 색).  성능 축은 서로 다른 벤치라 "동일 잣대 아님" 경고.
eff = [
    ('OpenVLA', 7000, 76.5, '#888888'),
    ('OpenVLA-OFT', 7000, 97.1, '#8a5cb0'),
    ('SmolVLA', 450, 87.0, '#2c6fb0'),       # LIBERO 평균대(회사발표 근사, 라벨용)
]
for (nm, p, perf, c) in eff:
    axB.scatter(p, perf, s=140, color=c, edgecolor='k', lw=0.8, zorder=4)
    axB.annotate(nm, (p, perf), xytext=(p * 1.05, perf + 0.6),
                 fontsize=8.4)
# OpenVLA → OFT 화살표(같은 크기, 디코딩만 바꿔 성능↑)
axB.annotate('', xy=(7000, 96.3), xytext=(7000, 77.4),
             arrowprops=dict(arrowstyle='-|>', color='#8a5cb0', lw=1.6))
axB.text(7700, 87, '디코딩만 교체\n(+20.6%p, 26x 처리량)', fontsize=7.6,
         color='#8a5cb0', va='center')
axB.set_xscale('log')
axB.set_xlabel('파라미터 수 [M] (log)')
axB.set_ylabel('대표 성공률 [%] (벤치 상이 — 절대비교 아님)')
axB.set_title('(b) 효율 프론티어 — 크기가 아니라 디코딩·데이터가 성능을 옮긴다', fontsize=10)
axB.set_xlim(200, 20000)
axB.set_ylim(70, 102)
axB.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig(OUT + 'fig4_wam_efficiency.png', dpi=140)
plt.close(fig)

print("\nfigures written: fig1_four_waves, fig2_latent_action, "
      "fig3_system_boundary, fig4_wam_efficiency")
