# Lec 65 그림 생성 스크립트 — 캡스톤: 두 논문(GR00T N1 · π*0.6/RECAP)의 프레임워크 분석
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만. 모델 다운로드/GPU 없음)
#
# 이 강의의 Worked Example은 "두 실제 논문에 64강 프레임워크를 채운 완성 분석"이다.
# 코드가 하는 일은 두 갈래:
#   (A) 분석 결과(6축 판정·novelty 좌표·주장 강도 할인)를 표/레이더/지도로 시각화한다.
#       — 축별 점수·좌표는 웹 확인한 사실(참고문헌)에서 도출한 서수적 판정이다(임의 수치 아님).
#   (B) numpy 토이 1개: latent action(VQ-VAE식 이산 잠재행동)이 라벨 없는 영상에서
#       유사행동(pseudo-action)을 어떻게 만드는지 — GR00T N1의 데이터 피라미드 2층(3M 인간영상)이
#       LAPA(2410.11758)의 그 아이디어를 흡수한 지점 — 을 결정론적으로 재현한다.
# 본문이 인용하는 모든 정량 수치는 이 스크립트의 실행 출력이다.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 0. 64강 6축 지도 — 두 논문의 축별 "기여 강도" 서수 판정 (0=관습적/이미 아는 것 … 3=핵심 새로움)
#    판정 근거는 각 논문의 웹 확인 사실(참고문헌). 이것은 "성능 점수"가 아니라
#    "이 논문의 새로움이 이 축에 얼마나 실려 있나"의 서수 라벨이다.
# ============================================================================
AXES6 = ['① 액션 디코딩', '② 아키텍처', '③ 학습 레시피',
         '④ 데이터', '⑤ 평가', '⑥ 효율']

# 서수 판정 (근거는 본문 WE-1/WE-2 표와 동일):
#   GR00T N1: 아키텍처(dual-system 오픈화)·데이터(피라미드+잠재행동)에 새로움이 실림.
#             액션 디코딩(DiT flow)·학습 레시피(BC)·평가(표준 sim 벤치)·효율은 관습적.
#   π*0.6:    학습 레시피(RECAP=오프라인 RL+보정+advantage 조건화)에 새로움이 집중.
#             데이터(이종 경험 통합)도 다소. 아키텍처·디코딩은 π0.6 계승(관습).
#             평가는 실전 신뢰성(18h 무중단)이라는 새 축을 세움.
GROOT = np.array([1, 3, 1, 3, 1, 1], dtype=float)   # ②④에 집중
PISTAR = np.array([1, 1, 3, 2, 2, 1], dtype=float)  # ③에 집중, ④⑤ 다소

print("=" * 70)
print("[프레임워크 1] 64강 6축 novelty 판정 (0=관습 … 3=핵심 새로움)")
print("=" * 70)
for i, ax_name in enumerate(AXES6):
    print(f"  {ax_name:<14s}: GR00T N1 = {GROOT[i]:.0f}   π*0.6 = {PISTAR[i]:.0f}")
# 두 논문의 novelty가 서로 다른 축에 있음을 정량으로: 축별 차이의 argmax
diff = PISTAR - GROOT
print(f"\n>>> novelty 최대 분리 축: "
      f"GR00T 우위={AXES6[int(np.argmax(GROOT - PISTAR))]}, "
      f"π*0.6 우위={AXES6[int(np.argmax(diff))]}")
# 축 겹침도(내적 정규화) — 두 기여가 얼마나 '다른 자리'에 있나
cos = float(GROOT @ PISTAR / (np.linalg.norm(GROOT) * np.linalg.norm(PISTAR)))
print(f">>> 두 논문 6축 벡터 코사인 유사도 = {cos:.3f} "
      f"(1이면 같은 축, 0이면 완전 직교) → 서로 다른 축에 새로움")

# ---------------------------------------------------------------------------
# fig1: 두 논문 6축 레이더 겹침
# ---------------------------------------------------------------------------
def radar(ax, values, color, label, fill_alpha):
    n = len(AXES6)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    v = np.concatenate([values, values[:1]])
    ax.plot(ang, v, color=color, lw=2.4, label=label, zorder=4)
    ax.fill(ang, v, color=color, alpha=fill_alpha, zorder=2)

fig = plt.figure(figsize=(8.4, 7.2))
ax = fig.add_subplot(111, polar=True)
n = len(AXES6)
ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
radar(ax, GROOT, '#2c6fb0', 'GR00T N1 (2503.14734) — ②아키텍처·④데이터', 0.16)
radar(ax, PISTAR, '#c0392b', 'π*0.6 / RECAP (2511.14759) — ③학습 레시피', 0.16)
ax.set_xticks(ang)
ax.set_xticklabels(AXES6, fontsize=10.5)
ax.set_ylim(0, 3)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['0 관습', '1', '2', '3 핵심 새로움'], fontsize=8.5, color='#555')
ax.set_title('두 논문의 novelty는 서로 다른 축에 있다\n'
             f'(6축 벡터 코사인 유사도 {cos:.2f} — 겹침이 적다)',
             fontsize=12.5, pad=24)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=9.6, framealpha=0.96)
ax.grid(alpha=0.35)
fig.tight_layout()
fig.savefig(OUT + 'fig1_sixaxis_radar.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# [프레임워크 2] 0강 정책 3축(아키텍처 ⊥ 학습목적 ⊥ 행동표현) 위 novelty 좌표
#   두 논문을 3축 값으로 위치시키고, "새로움이 어느 축의 이동인가"를 표시한다.
#   축 값은 순서형 라벨을 좌표로: 아키텍처(단일=0 … dual-system=2 … 2모델=3),
#   학습목적(BC=0, BC+오프라인RL=2, ...), 행동표현(이산=0, flow/diffusion=2).
# ============================================================================
print("\n" + "=" * 70)
print("[프레임워크 2] 0강 3축 좌표 (아키텍처 / 학습목적 / 행동표현)")
print("=" * 70)
# (아키텍처, 학습목적, 행동표현)  — 0강 표와 동일 어휘
POINTS = {
    'OpenVLA (43강)':      (1.0, 0.0, 0.0, '#9a9a9a'),   # 단일 VLA / BC / 이산 AR
    'π0 (44강)':           (1.5, 0.0, 2.0, '#9a9a9a'),   # VLA+expert / BC / flow
    'GR00T N1 (분석 A)':   (2.0, 0.0, 2.0, '#2c6fb0'),   # dual-system / BC / diffusion
    'π*0.6/RECAP (분석 B)':(1.5, 2.0, 2.0, '#c0392b'),   # VLA / BC+오프라인RL / flow
}
for k, (a, l, r, _) in POINTS.items():
    print(f"  {k:<22s}: 아키텍처={a:.1f}  학습목적={l:.1f}  행동표현={r:.1f}")
print(">>> GR00T의 이동 = 아키텍처 축(단일→dual-system, 오픈)")
print(">>> π*0.6의 이동 = 학습목적 축(BC→BC+오프라인RL) — 행동표현·아키텍처는 π0 계승")

# ---------------------------------------------------------------------------
# fig2: 3축 지도 위 novelty 위치 (3D 산점 — 두 논문의 '이동 화살표'를 강조)
# ---------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(9.6, 7.4))
ax = fig.add_subplot(111, projection='3d')
for k, (a, l, r, c) in POINTS.items():
    ax.scatter(a, l, r, s=160, color=c, edgecolor='k', lw=0.8, depthshade=False)
    ax.text(a + 0.05, l + 0.05, r + 0.08, k, fontsize=8.6)
# 이동 화살표: π0 → GR00T (아키텍처 축), π0 → π*0.6 (학습목적 축)
p0 = np.array(POINTS['π0 (44강)'][:3])
pg = np.array(POINTS['GR00T N1 (분석 A)'][:3])
pp = np.array(POINTS['π*0.6/RECAP (분석 B)'][:3])
for start, end, col in [(p0, pg, '#2c6fb0'), (p0, pp, '#c0392b')]:
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            color=col, lw=2.2, ls='--', zorder=1)
ax.text(2.0, 1.05, 2.15, 'B: 학습목적 축 이동\n(BC→+오프라인RL)',
        fontsize=8.4, color='#c0392b')
ax.text(2.05, -0.35, 1.5, 'A: 아키텍처 축 이동\n(단일→dual-system·오픈)',
        fontsize=8.4, color='#2c6fb0')
ax.set_xlabel('축1 아키텍처\n(단일→dual→2모델)', fontsize=9)
ax.set_ylabel('축2 학습목적\n(BC→+RL)', fontsize=9)
ax.set_zlabel('축3 행동표현\n(이산→flow)', fontsize=9)
ax.set_xlim(0.5, 2.5); ax.set_ylim(-0.2, 2.4); ax.set_zlim(-0.2, 2.4)
ax.set_xticks([1, 1.5, 2]); ax.set_yticks([0, 1, 2]); ax.set_zticks([0, 1, 2])
ax.view_init(elev=18, azim=-58)
ax.set_title('novelty 위치잡기 — 두 논문은 0강 3축의 서로 다른 축을 따라 이동한다\n'
             '("새로운 점 = 결과 수치"가 아니라 "어느 축의 이동인가")', fontsize=11.5, pad=6)
fig.tight_layout()
fig.savefig(OUT + 'fig2_novelty_3axis.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# [프레임워크 3] 주장 강도 할인 — 평가 신뢰성 가중
#   64강 체크리스트의 정량화: 원 주장 강도 c 를 재현성/N·CI/공개/실패공개로 할인.
#   할인계수 d = ∏(1 - penalty_i).  두 논문에 실제로 채워 넣는다.
# ============================================================================
print("\n" + "=" * 70)
print("[프레임워크 3] 주장 강도 할인 d = ∏(1 - penalty)  (평가 신뢰성 가중)")
print("=" * 70)
# 체크 항목별 penalty (0=문제없음 … 큰 값=크게 할인). 서수 판정, 근거는 WE 표.
CHECKS = ['가중치 공개', 'N·신뢰구간 보고', '실패 사례 공개', '베이스라인 통제', '독립 재현']
# GR00T N1: 오픈 모델·코드·표준 sim 벤치(N·baseline 통제 양호) → 할인 작음
GROOT_PEN = np.array([0.00, 0.10, 0.15, 0.05, 0.05])
# π*0.6: 가중치 미공개·실기 데모 중심(N·CI 약함, 실패 릴 제한, 독립 재현 불가) → 할인 큼
PISTAR_PEN = np.array([0.40, 0.25, 0.20, 0.10, 0.30])

def discount(pen):
    return float(np.prod(1.0 - pen))

dg, dp = discount(GROOT_PEN), discount(PISTAR_PEN)
print("  항목                : GR00T penalty / π*0.6 penalty")
for i, c in enumerate(CHECKS):
    print(f"  {c:<18s}: {GROOT_PEN[i]:.2f} / {PISTAR_PEN[i]:.2f}")
print(f"\n>>> 할인계수 d:  GR00T N1 = {dg:.3f}   π*0.6 = {dp:.3f}")
print(f">>> 같은 '핵심 주장 강도 1.0'이라도 신뢰가중 후: "
      f"GR00T {dg:.2f} vs π*0.6 {dp:.2f} "
      f"(π*0.6 주장은 재현·검증 관점에서 {dg/dp:.1f}배 더 크게 할인)")

# ---------------------------------------------------------------------------
# fig3: 정보 파이프라인 도식 + 주장 강도 할인 막대
# ---------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.6),
                               gridspec_kw={'width_ratios': [1.5, 1.0]})

# (좌) 정보 파이프라인: 소스 → 필터(6축·체크리스트) → 판정 → 자기 지식 지도 갱신
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
axL.set_title('(a) 정보 파이프라인 — 새 논문을 "새로운 점만" 통과시키는 여과기', fontsize=11.5)

def box(ax, x, y, w, h, text, fc, fs=8.6, tc='#0c3a63'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                 facecolor=fc, edgecolor='k', lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, wrap=True)

def arrow(ax, x0, y0, x1, y1, col='#555'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=15, color=col, lw=1.6))

# 소스 열
box(axL, 0.2, 8.2, 3.0, 1.3, 'arXiv cs.RO RSS\n· HF Daily Papers', '#e1eefb')
box(axL, 0.2, 6.4, 3.0, 1.3, '서베이 (Anatomy)\n· alphaXiv 토론', '#e1eefb')
box(axL, 0.2, 4.6, 3.0, 1.3, '공식 블로그·모델카드\n(PI·NVIDIA·Figure)', '#e1eefb')
box(axL, 0.2, 2.8, 3.0, 1.3, '학회 CoRL·RSS\n· 뉴스레터 알림', '#e1eefb')
# 여과기
box(axL, 4.2, 4.6, 2.6, 3.2,
    '여과기\n64강 6축 지도\n+ 체크리스트 10항\n+ 3축 층위 진단', '#f3eede', 9.4, '#3a3320')
# 판정
box(axL, 7.4, 6.4, 2.4, 1.3, '"새로운 점"\n(축의 이동)', '#d8f0d8', 8.8, '#1a5a3a')
box(axL, 7.4, 4.6, 2.4, 1.3, '주장 강도 할인\n(신뢰가중 d)', '#f7dede', 8.8, '#7a1f16')
box(axL, 7.4, 2.8, 2.4, 1.3, '커리큘럼 지도\n갱신(회고)', '#f3eede', 8.8, '#3a3320')
for y in [8.85, 7.05, 5.25, 3.45]:
    arrow(axL, 3.2, y, 4.2, 6.2)
for (yt) in [7.05, 5.25, 3.45]:
    arrow(axL, 6.8, 6.2, 7.4, yt)

# (우) 주장 강도 할인 막대: 원 강도 1.0 → 신뢰가중 후
axR.set_title('(b) 주장 강도 할인 — 같은 "1.0"도 검증 가능성으로 깎인다', fontsize=11)
labels = ['GR00T N1\n(오픈·표준벤치)', 'π*0.6\n(미공개·실기 데모)']
raw = [1.0, 1.0]
disc = [dg, dp]
xb = np.arange(2)
axR.bar(xb - 0.19, raw, 0.36, color='#c9c9c9', edgecolor='k', lw=0.6, label='원 주장 강도')
axR.bar(xb + 0.19, disc, 0.36, color=['#2c6fb0', '#c0392b'], edgecolor='k', lw=0.6,
        label='신뢰가중 후 $d=\\prod(1-p_i)$')
for i, (rv, dv) in enumerate(zip(raw, disc)):
    axR.text(i - 0.19, rv + 0.02, f'{rv:.2f}', ha='center', fontsize=8.5)
    axR.text(i + 0.19, dv + 0.02, f'{dv:.2f}', ha='center', fontsize=8.5, weight='bold')
axR.set_xticks(xb); axR.set_xticklabels(labels, fontsize=9)
axR.set_ylim(0, 1.15); axR.set_ylabel('주장 강도 (상대)')
axR.legend(fontsize=8.2, loc='upper right')
axR.grid(alpha=0.25, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig3_pipeline_discount.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# [프레임워크 4] 커리큘럼 회고 지도 — 각 논문의 기여를 "어느 강의가 미리 준비시켰나"
# ============================================================================
# ---------------------------------------------------------------------------
# fig4: 회고 지도 (bipartite: 강의 번호 → 두 논문의 기여 요소)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.4, 7.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis('off')
ax.set_title('두 논문의 각 기여를 "미리 준비시킨" 커리큘럼 강의 — 새로 배울 것은 얇은 위층뿐',
             fontsize=12.5, pad=8)

# 왼쪽: 커리큘럼 강의 (준비시킨 개념)
LECS = [
    ('0강 · 정책 3축·닫힌 루프', 10.8),
    ('40·44강 · flow/diffusion 헤드', 9.4),
    ('46·48강 · dual-system·System 라벨', 8.0),
    ('37·41강 · compounding err·오프라인 RL', 6.6),
    ('45강 · RECAP·advantage 조건화', 5.2),
    ('53·54강 · 합성데이터·world model·잠재행동', 3.8),
    ('55·57강 · 데이터·평가 신뢰성(N·CI)', 2.4),
    ('43·47강 · 효율·디코딩', 1.0),
]
# 오른쪽: 두 논문의 기여 요소 (A=GR00T, B=π*0.6)
CONTRIB = [
    ('A: dual-system(S2 VLM/S1 DiT)', 10.4, '#2c6fb0'),
    ('A: 데이터 피라미드 3층', 9.0, '#2c6fb0'),
    ('A: 잠재행동 codebook(라벨없는 영상)', 7.6, '#2c6fb0'),
    ('A: DiT flow 액션 헤드', 6.2, '#2c6fb0'),
    ('B: RECAP 오프라인 RL 사후훈련', 4.8, '#c0392b'),
    ('B: advantage 조건화 토큰', 3.4, '#c0392b'),
    ('B: 실전 신뢰성(18h 무중단) 평가', 2.0, '#c0392b'),
]
for name, y in LECS:
    ax.add_patch(FancyBboxPatch((0.2, y - 0.35), 4.7, 0.72, boxstyle='round,pad=0.05',
                 facecolor='#f3eede', edgecolor='#8a7a4a', lw=1.0))
    ax.text(2.55, y, name, ha='center', va='center', fontsize=8.5, color='#3a3320')
for name, y, c in CONTRIB:
    ax.add_patch(FancyBboxPatch((7.1, y - 0.33), 4.7, 0.68, boxstyle='round,pad=0.05',
                 facecolor='#e1eefb' if c == '#2c6fb0' else '#f7dede',
                 edgecolor=c, lw=1.1))
    ax.text(9.45, y, name, ha='center', va='center', fontsize=8.3, color=c)

# 연결(어느 강의가 어느 기여를 준비시켰나)
LY = {n: y for n, y in LECS}
CY = {n: y for n, y, _ in CONTRIB}
EDGES = [
    ('46·48강 · dual-system·System 라벨', 'A: dual-system(S2 VLM/S1 DiT)'),
    ('40·44강 · flow/diffusion 헤드', 'A: DiT flow 액션 헤드'),
    ('53·54강 · 합성데이터·world model·잠재행동', 'A: 데이터 피라미드 3층'),
    ('53·54강 · 합성데이터·world model·잠재행동', 'A: 잠재행동 codebook(라벨없는 영상)'),
    ('55·57강 · 데이터·평가 신뢰성(N·CI)', 'A: 데이터 피라미드 3층'),
    ('37·41강 · compounding err·오프라인 RL', 'B: RECAP 오프라인 RL 사후훈련'),
    ('45강 · RECAP·advantage 조건화', 'B: RECAP 오프라인 RL 사후훈련'),
    ('45강 · RECAP·advantage 조건화', 'B: advantage 조건화 토큰'),
    ('55·57강 · 데이터·평가 신뢰성(N·CI)', 'B: 실전 신뢰성(18h 무중단) 평가'),
    ('43·47강 · 효율·디코딩', 'A: DiT flow 액션 헤드'),
    ('0강 · 정책 3축·닫힌 루프', 'A: dual-system(S2 VLM/S1 DiT)'),
    ('0강 · 정책 3축·닫힌 루프', 'B: RECAP 오프라인 RL 사후훈련'),
]
for lname, cname in EDGES:
    y0, y1 = LY[lname], CY[cname]
    col = '#c0392b' if cname.startswith('B') else '#2c6fb0'
    ax.add_patch(FancyArrowPatch((4.95, y0), (7.05, y1), arrowstyle='-',
                 connectionstyle='arc3,rad=0.06', color=col, lw=1.0, alpha=0.5))
ax.text(2.55, 11.6, '커리큘럼이 준비시킨 것 (이미 아는 것)', ha='center',
        fontsize=10, weight='bold', color='#8a7a4a')
ax.text(9.45, 11.6, '두 논문의 기여 (A=GR00T N1 · B=π*0.6)', ha='center',
        fontsize=10, weight='bold', color='#333')
fig.tight_layout()
fig.savefig(OUT + 'fig5_curriculum_retro.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# [numpy 토이] latent action — 라벨 없는 영상에서 유사행동(pseudo-action) 뽑기
#   GR00T N1 데이터 피라미드 2층(3M 인간영상)이 LAPA(2410.11758)의 아이디어를 흡수한 지점.
#   최소판: 연속 잠재행동 = z_{t+1}-z_t (프레임차) 를 K개 프로토타입으로 VQ(양자화).
#   그리고 IDM(역동역학) 회귀가 이 이산 잠재를 실제 행동으로 복원할 수 있는지 확인.
# ============================================================================
print("\n" + "=" * 70)
print("[numpy 토이] latent action: 라벨 없는 영상 → 이산 잠재행동 → IDM으로 실행동 복원")
print("=" * 70)
rng = np.random.default_rng(0)
# 참 시스템: z_{t+1} = z_t + B a_t  (2D 잠재, 4가지 '진짜' 행동을 반복 수행)
Dz, Da, K = 2, 2, 4
B_true = np.array([[0.9, 0.1], [-0.2, 0.8]])
true_actions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # 4가지 원형 행동
# 궤적 생성: 각 스텝에서 4행동 중 하나 무작위 선택 (라벨은 '가려짐' — 영상만 관측)
N = 2000
idx = rng.integers(0, 4, size=N)
A = true_actions[idx] + 0.03 * rng.standard_normal((N, Da))    # 실제 행동(우리는 못 봄)
Z = np.zeros((N + 1, Dz)); Z[0] = rng.standard_normal(Dz)
for t in range(N):
    Z[t + 1] = Z[t] + A[t] @ B_true.T
delta = Z[1:] - Z[:-1]      # 프레임차 = 관측 가능한 '변화'(영상에서 읽는 것)

# --- (1) 잠재행동 codebook 학습: delta 를 K개 프로토타입으로 k-means(VQ, k-means++ 초기화) ---
def kmeans(x, k, iters=80, seed=1):
    r = np.random.default_rng(seed)
    # k-means++ 초기화 (안정적 수렴)
    c = [x[r.integers(len(x))]]
    for _ in range(k - 1):
        d2 = np.min(((x[:, None, :] - np.array(c)[None, :, :]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum()
        c.append(x[r.choice(len(x), p=p)])
    c = np.array(c)
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return c, lab

codebook, code = kmeans(delta, K)                 # 이산 잠재행동 = 코드 인덱스
# 학습된 이산 잠재가 참 행동 클러스터를 복원했는가 — 코드↔참행동 정합(순열 무시한 순도)
purity = 0.0
for j in range(K):
    if (code == j).any():
        maj = np.bincount(idx[code == j], minlength=4).max()
        purity += maj
purity /= N
print(f"  학습된 잠재행동 codebook (K={K}) 순도(참행동 복원율) = {purity:.3f}")

# --- (2) IDM(역동역학): (z_t, z_{t+1}) → 이산 잠재행동, 그리고 잠재→실행동 선형복원 ---
# 잠재 코드 원핫으로 실제 행동을 회귀(최소제곱). '이산 잠재가 실행동을 담는가'의 검증.
onehot = np.eye(K)[code]
W, *_ = np.linalg.lstsq(onehot, A, rcond=None)    # 코드→실행동 사상
A_hat = onehot @ W
recon = 1 - ((A - A_hat) ** 2).mean() / A.var()   # R²
print(f"  이산 잠재행동 → 실제 행동 복원 R² = {recon:.3f} "
      f"(1이면 잠재가 행동을 완전히 담음)")
print(">>> 함의: 라벨 없는 영상(프레임차만)에서도 이산 잠재행동을 만들면, "
      "그것이 실제 행동의 대리(pseudo-action)로 쓸 만하다 —")
print(">>> GR00T N1이 3M 인간영상을 '또 하나의 embodiment'로 사전학습에 넣은 원리(LAPA 계보).")

# ---------------------------------------------------------------------------
# fig(추가): 잠재행동 토이 — delta 산점 + codebook, 그리고 순도/R²
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.9))
colors = ['#2c6fb0', '#c0392b', '#3a9a5a', '#e08a1e']
for j in range(K):
    m = code == j
    a1.scatter(delta[m, 0], delta[m, 1], s=8, color=colors[j], alpha=0.35)
a1.scatter(codebook[:, 0], codebook[:, 1], s=220, marker='*',
           color='k', edgecolor='w', lw=1.2, zorder=5, label='학습된 codebook')
a1.set_title('(a) 라벨 없는 영상의 프레임차 $\\Delta z$ →\n이산 잠재행동 codebook (K=4) 자동 발견')
a1.set_xlabel('$\\Delta z_1$'); a1.set_ylabel('$\\Delta z_2$')
a1.legend(fontsize=8.5); a1.grid(alpha=0.25); a1.set_aspect('equal')

a2.bar([0, 1], [purity, recon], color=['#2c6fb0', '#3a9a5a'], edgecolor='k', lw=0.6)
a2.set_xticks([0, 1])
a2.set_xticklabels(['codebook 순도\n(참행동 복원율)', '잠재→실행동\n복원 $R^2$'], fontsize=9)
a2.set_ylim(0, 1.1)
for i, v in enumerate([purity, recon]):
    a2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, weight='bold')
a2.set_title('(b) 이산 잠재행동이 실제 행동을 담는가 —\n'
             '라벨 없는 데이터를 학습 신호로 쓰는 근거')
a2.grid(alpha=0.25, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig4_latent_action_toy.png', dpi=140, bbox_inches='tight')
plt.close(fig)

print("\nfigures written: fig1_sixaxis_radar, fig2_novelty_3axis, "
      "fig3_pipeline_discount, fig4_latent_action_toy, fig5_curriculum_retro")
