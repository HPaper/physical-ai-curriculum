# Lec 64 그림 생성 스크립트 — 논문 읽기 프레임워크: 6축 지도·체크리스트·층위 진단·평가 신뢰성
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만. 모델 다운로드/GPU 없음)
#
# 이 스크립트는 "프레임워크를 시각화"한다. 6축 좌표·체크리스트·3축 직교 진단은
# 논문에 채운 판정값(범주형)이고, E3의 평가 신뢰성(Clopper-Pearson 이항 CI)만
# 순수 numpy/scipy 계산이다. 본문이 인용하는 CI·유의성 수치는 이 스크립트 출력이다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from scipy import stats

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# 색: 파랑=구조/축, 주황=경계/주의, 초록=통과/유효, 빨강=위반/오류, 회색=중립
C_AX, C_WARN, C_OK, C_BAD, C_NEU = '#2c6fb0', '#e08a1e', '#2e9e5b', '#c0392b', '#8a8a8a'

# ============================================================================
# 6축 지도 정의 — 각 축 = 논문에 던지는 하나의 질문 (E1)
#   축 이름 · 던지는 질문 · 관련 강의
# ============================================================================
AXES = [
    ('① 액션 디코딩', '이산/연속/flow/diffusion?', '44'),
    ('② 아키텍처',   '백본·dual-system?',        '36·46'),
    ('③ 학습 레시피', 'BC/RL/co-training?',       '45'),
    ('④ 데이터',     '규모·출처·embodiment?',    '55'),
    ('⑤ 평가',       '벤치마크·N·CI?',           '57'),
    ('⑥ 효율',       '파라미터·주기·디코딩?',     '43·47'),
]

# ============================================================================
# 세 논문의 6축 novelty 점수 (0=관례 따름, 3=핵심 신규 기여).
#   이 점수는 "어느 축이 새로운가"의 판정값(범주형)이지 성능 점수가 아니다.
#   근거: OpenVLA-OFT[2502.19645] / pi0-FAST[2501.09747] / pi0.7[2604.15483]
# ============================================================================
PAPERS = {
    'OpenVLA-OFT': [3, 0, 1, 0, 1, 3],   # 디코딩(병렬+L1 연속) & 효율(26x)이 핵심
    'π0-FAST':     [3, 0, 0, 0, 1, 1],   # 액션 디코딩(DCT 토크나이저)이 거의 전부
    'π0.7':        [0, 1, 3, 1, 1, 0],   # 학습 레시피(다양 컨텍스트 조건화 스티어링)
}
PAPER_COLORS = {'OpenVLA-OFT': C_AX, 'π0-FAST': C_WARN, 'π0.7': C_OK}

print("=" * 70)
print("[E1] 6축 novelty 분해 — 새로움은 대개 1~2축에 몰린다 (희소성)")
print("=" * 70)
for name, vec in PAPERS.items():
    v = np.array(vec)
    n_hi = int((v >= 2).sum())        # 핵심 기여(2 이상) 축 개수
    frac = v.sum() / (3 * len(v))     # 전체 축 대비 novelty 총량
    top = [AXES[i][0] for i in np.argsort(-v)[:2]]
    print(f"  {name:12s} 벡터={vec}  핵심축={n_hi}개  "
          f"novelty밀도={frac:.2f}  주축={top}")

# ---------------------------------------------------------------------------
# FIG 1 — 6축 레이더: 세 논문의 novelty 좌표를 한 판에
# ---------------------------------------------------------------------------
def radar():
    fig, axes_grid = plt.subplots(1, 3, figsize=(13.5, 4.8),
                                  subplot_kw=dict(polar=True))
    K = len(AXES)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    short = [a[0] for a in AXES]
    for ax, (name, vec) in zip(axes_grid, PAPERS.items()):
        vals = np.array(vec + vec[:1], dtype=float)
        col = PAPER_COLORS[name]
        ax.plot(angles_closed, vals, color=col, lw=2.2)
        ax.fill(angles_closed, vals, color=col, alpha=0.22)
        ax.set_xticks(angles)
        ax.set_xticklabels(short, fontsize=8.5)
        ax.set_ylim(0, 3)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['1', '2', '3'], fontsize=7, color=C_NEU)
        n_hi = int((np.array(vec) >= 2).sum())
        ax.set_title(f'{name}\n핵심 기여 축 {n_hi}개',
                     fontsize=11, color=col, pad=16, fontweight='bold')
        ax.grid(color='#cccccc', lw=0.6)
    fig.suptitle('그림 1 · 6축 novelty 레이더 — 새로움은 1~2축에 집중된다 '
                 '(0=관례, 3=핵심 신규)',
                 fontsize=12.5, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT + 'fig1_six_axis_radar.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("saved fig1_six_axis_radar.png")

# ---------------------------------------------------------------------------
# FIG 2 — 비판적 읽기 체크리스트 10항목 × 세 논문 판정 히트맵
#   판정: 2=통과/명시, 1=부분/유보, 0=미흡/누락  (공개 자료 기준)
# ---------------------------------------------------------------------------
CHECKLIST = [
    'C1 평가 N·CI를 보고했나 (57강)',
    'C2 baseline이 공정한가 (동일 조건)',
    'C3 코드·가중치를 공개했나',
    'C4 sim vs real을 구분했나',
    'C5 체리피킹 아닌가 (실패 릴)',
    'C6 ablation으로 기여를 분리했나',
    'C7 데이터 출처·규모를 명시했나',
    'C8 실패 사례를 보고했나',
    'C9 embodiment 일반화를 통제했나',
    'C10 새로움이 어느 축인가 (6축)',
]
# 판정 근거는 각 논문의 공개 정도(오픈 코드/가중치·통계 방법론·회사 비공개 등).
CHECK_SCORES = {
    'OpenVLA-OFT': [2, 2, 2, 2, 1, 2, 2, 1, 1, 2],
    'π0-FAST':     [1, 2, 2, 1, 1, 2, 2, 1, 2, 2],
    'π0.7':        [1, 1, 0, 1, 1, 1, 1, 1, 2, 2],   # 회사 발표·가중치 미공개
}

def checklist_heatmap():
    names = list(CHECK_SCORES.keys())
    M = np.array([CHECK_SCORES[n] for n in names]).T   # 10 x 3
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    cmap = {0: C_BAD, 1: C_WARN, 2: C_OK}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            ax.add_patch(Rectangle((j, M.shape[0]-1-i), 1, 1,
                         facecolor=cmap[val], edgecolor='white', lw=2, alpha=0.9))
            mark = {0: 'X', 1: '△', 2: '○'}[val]
            ax.text(j+0.5, M.shape[0]-1-i+0.5, mark, ha='center', va='center',
                    fontsize=15, color='white', fontweight='bold')
    ax.set_xlim(0, M.shape[1]); ax.set_ylim(0, M.shape[0])
    ax.set_xticks(np.arange(M.shape[1]) + 0.5)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(M.shape[0]) + 0.5)
    ax.set_yticklabels(CHECKLIST[::-1], fontsize=9.2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 범례
    handles = [Rectangle((0, 0), 1, 1, facecolor=cmap[v]) for v in (2, 1, 0)]
    ax.legend(handles, ['○ 통과/명시', '△ 부분/유보', 'X 미흡/누락'],
              loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9.5,
              frameon=False)
    ax.set_title('그림 2 · 비판적 읽기 체크리스트 10항목 — 세 논문 판정\n'
                 '(가중치·코드 공개 축에서 회사 발표 논문[π0.7]이 낮다)',
                 fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(OUT + 'fig2_checklist.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("saved fig2_checklist.png")

# ---------------------------------------------------------------------------
# FIG 3 — 층위 진단: 0강 3축 직교 → novelty를 정확한 축에 놓기
#   "VLA vs RL" 범주 오류를 3축(아키텍처⊥학습목적⊥행동표현)으로 가른다
# ---------------------------------------------------------------------------
def layer_diagnosis():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # 왼쪽: 3축 직교 개념 — 각 축의 값 슬롯
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis('off')
    axL.set_title('(a) 정책 3축은 직교 — 한 축의 값이 다른 축을 강제하지 않는다',
                  fontsize=11, fontweight='bold')
    rows = [
        ('축1 아키텍처 (함수족)',   ['MLP', 'Transformer', 'VLA=VLM백본', 'dual-system'], C_AX),
        ('축2 학습 목적 (신호)',    ['BC 모방', 'RL 보상', 'SSL', 'co-training'], C_WARN),
        ('축3 행동 표현 (분포)',    ['가우시안', '이산 토큰', 'flow', 'diffusion'], C_OK),
    ]
    y0 = 8.2
    for r, (label, slots, col) in enumerate(rows):
        y = y0 - r * 2.6
        axL.text(0.2, y + 0.75, label, fontsize=10.5, fontweight='bold', color=col)
        for s, name in enumerate(slots):
            x = 0.3 + s * 2.4
            box = FancyBboxPatch((x, y - 0.35), 2.15, 0.95,
                                 boxstyle="round,pad=0.04", linewidth=1.4,
                                 edgecolor=col, facecolor=col, alpha=0.14)
            axL.add_patch(box)
            axL.text(x + 1.07, y + 0.12, name, ha='center', va='center', fontsize=8.8)
    axL.text(5.0, 0.5,
             '"VLA vs RL" = 축1 값 vs 축2 값 → 범주 오류 (SUV vs 디젤)',
             ha='center', fontsize=9.6, color=C_BAD, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#fbeaea', edgecolor=C_BAD))

    # 오른쪽: 문장 타입 판별 — 유효 조합 vs 범주 오류
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis('off')
    axR.set_title('(b) 논문 문장을 3축으로 타입 판별 → novelty 위치 확정',
                  fontsize=11, fontweight='bold')
    claims = [
        ('"VLA를 RL로 post-train"', '축1×축2 조합 (독립)', True,  '유효 — π*0.6이 실물 예시'),
        ('"이산 vs flow 디코딩"',   '축3 안의 값 비교',    True,  '유효 — 같은 축 내 비교'),
        ('"VLA vs RL 중 택1"',      '축1 값 vs 축2 값',    False, '범주 오류 — 비교 불가'),
        ('"diffusion = RL의 일종"', '축3을 축2로 오분류',  False, '타입 오류 — 대개 BC'),
    ]
    yy = 8.4
    for i, (claim, kind, ok, verdict) in enumerate(claims):
        y = yy - i * 2.0
        col = C_OK if ok else C_BAD
        mark = '○' if ok else 'X'
        axR.add_patch(FancyBboxPatch((0.3, y - 0.55), 9.4, 1.5,
                      boxstyle="round,pad=0.05", linewidth=1.5,
                      edgecolor=col, facecolor=col, alpha=0.10))
        axR.text(0.6, y + 0.5, f'{mark} {claim}', fontsize=10, fontweight='bold', color=col)
        axR.text(0.9, y - 0.05, f'{kind}  →  {verdict}', fontsize=8.9, color='#333')
    fig.suptitle('그림 3 · 층위 진단 (0강 회수) — 3축 직교로 범주 오류를 걸러 '
                 'novelty를 정확한 축에 놓는다',
                 fontsize=12.5, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT + 'fig3_layer_diagnosis.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("saved fig3_layer_diagnosis.png")

# ---------------------------------------------------------------------------
# FIG 4 — 채워진 논문 분석 예시표 (WE-1): OpenVLA-OFT를 6축으로
# ---------------------------------------------------------------------------
def filled_table():
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.axis('off')
    ax.set_xlim(0, 12); ax.set_ylim(0, 11)
    ax.set_title('그림 4 · 채워진 6축 분석 (WE-1) — OpenVLA-OFT [arXiv:2502.19645]',
                 fontsize=12.5, fontweight='bold', pad=10)
    header = ['6축', '이 논문의 값', 'novelty', '판정 근거']
    xcol = [0.2, 2.4, 6.9, 8.4]
    wcol = [2.2, 4.5, 1.5, 3.6]
    # 헤더
    for h, x, w in zip(header, xcol, wcol):
        ax.add_patch(Rectangle((x, 9.6), w, 0.9, facecolor=C_AX, alpha=0.85))
        ax.text(x + 0.12, 10.05, h, fontsize=10, color='white', fontweight='bold',
                va='center')
    rows = [
        ('① 액션 디코딩', '이산 AR → 병렬 디코딩 + 연속 L1 회귀', 3, '핵심 신규 (7패스→1패스)'),
        ('② 아키텍처',   'OpenVLA 7B 백본 그대로',            0, '관례 (재사용)'),
        ('③ 학습 레시피', 'BC (L1 회귀 목적)',                 1, '부분 (목적 교체)'),
        ('④ 데이터',     'LIBERO·기존 데이터',                0, '관례'),
        ('⑤ 평가',       'LIBERO 76.5→97.1%, N 보고',        1, '벤치마크 위주'),
        ('⑥ 효율',       '26x 액션생성·3x 지연 (~5→>100Hz)',  3, '핵심 신규'),
    ]
    y = 9.4
    for name, val, nov, why in rows:
        y -= 1.42
        novcol = {0: C_NEU, 1: C_WARN, 3: C_BAD}[nov]
        # 각 셀
        ax.text(xcol[0] + 0.12, y + 0.45, name, fontsize=9.6, fontweight='bold', va='center')
        ax.text(xcol[1] + 0.12, y + 0.45, val, fontsize=9.0, va='center')
        ax.add_patch(Rectangle((xcol[2], y), wcol[2], 0.9, facecolor=novcol, alpha=0.22,
                     edgecolor=novcol, lw=1.3))
        star = {0: '·', 1: '△', 3: '★★★'}[nov]
        ax.text(xcol[2] + wcol[2]/2, y + 0.45, star, ha='center', va='center',
                fontsize=11, color=novcol, fontweight='bold')
        ax.text(xcol[3] + 0.12, y + 0.45, why, fontsize=8.6, va='center', color='#333')
        ax.plot([0.2, 12], [y - 0.05, y - 0.05], color='#e0e0e0', lw=0.6)
    ax.text(6.1, 0.35,
            '판정: novelty는 ①·⑥ 두 축에 집중 (★★★). '
            '"디코딩만 바꿔 26x"가 이 논문의 한 줄 요약 — 나머지 4축은 관례.',
            ha='center', fontsize=9.7, color=C_BAD, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fbeaea', edgecolor=C_BAD))
    fig.tight_layout()
    fig.savefig(OUT + 'fig5_filled_table.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("saved fig5_filled_table.png")

# ============================================================================
# [E3] 평가 신뢰성 가중 — 이항 CI(Clopper-Pearson)와 유의성으로 주장 할인
#   TRI LBM[2507.05331]은 실기 태스크당 N=50 롤아웃을 블라인드·랜덤화로 평가하고
#   성공률 불확실성을 베타 균등 사전 위의 베이지안 사후분포(바이올린)로 그린다.
#   여기 Clopper-Pearson 계산은 그 균등 베타 사후[Beta(k+1,n-k+1)]와 수치적으로
#   거의 겹치는 프리퀀티스트 대응물이다. (N=50→CI 폭 ~18.5%p는 아래 계산값.)
# ============================================================================
def clopper_pearson(k, n, alpha=0.05):
    """이항 비율의 정확(Clopper-Pearson) 신뢰구간 — 순수 scipy.stats.beta."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha/2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha/2, k + 1, n - k)
    return lo, hi

print("\n" + "=" * 70)
print("[E3] 평가 신뢰성 — 이항 CI(Clopper-Pearson) 폭 vs 표본수 N")
print("=" * 70)
p_hat = 0.90                       # 관측 성공률 90%로 고정
for n in (10, 20, 50, 100, 300, 1000):
    k = round(p_hat * n)
    lo, hi = clopper_pearson(k, n)
    print(f"  N={n:5d}  성공 {k:4d}/{n:<5d}  p̂={k/n:.2f}  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  폭={hi-lo:.3f} ({(hi-lo)*100:.1f}%p)")

# 두 정책 비교의 유의성 (2-proportion z-test), N=50 vs N=300
print("\n  두 정책 A(성공률 90%) vs B(성공률 78%) 비교 — N별 유의성:")
def two_prop_z(p1, p2, n1, n2):
    x1, x2 = round(p1*n1), round(p2*n2)
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p*(1-p)*(1/n1 + 1/n2))
    z = (x1/n1 - x2/n2) / se
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, pval
for n in (50, 100, 300):
    z, pval = two_prop_z(0.90, 0.78, n, n)
    sig = '유의(p<0.05)' if pval < 0.05 else '유의하지 않음'
    print(f"    N={n:4d}/arm:  z={z:5.2f}  p={pval:.4f}  → {sig}")

def eval_reliability():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    # (a) CI 폭 vs N
    Ns = np.array([10, 20, 50, 100, 200, 300, 500, 1000])
    los, his = [], []
    for n in Ns:
        k = round(p_hat * n)
        lo, hi = clopper_pearson(k, n)
        los.append(lo); his.append(hi)
    los, his = np.array(los), np.array(his)
    axL.fill_between(Ns, los*100, his*100, color=C_AX, alpha=0.18,
                     label='95% Clopper-Pearson CI')
    axL.plot(Ns, [p_hat*100]*len(Ns), '--', color=C_AX, lw=1.6, label='관측 성공률 90%')
    axL.axvline(50, color=C_WARN, lw=1.4, ls=':')
    k50 = round(p_hat*50); lo50, hi50 = clopper_pearson(k50, 50)
    axL.annotate(f'N=50: CI 폭 {int(round((hi50-lo50)*100))}%p\n(TRI LBM 실기 표본수)',
                 xy=(50, lo50*100), xytext=(120, 55),
                 fontsize=9, color=C_WARN,
                 arrowprops=dict(arrowstyle='->', color=C_WARN))
    axL.set_xscale('log')
    axL.set_xlabel('롤아웃 표본수 N (log)', fontsize=10)
    axL.set_ylabel('성공률 [%]', fontsize=10)
    axL.set_ylim(40, 102)
    axL.set_title('(a) N이 작으면 CI가 넓어 주장이 약하다', fontsize=11, fontweight='bold')
    axL.legend(fontsize=8.8, loc='lower right'); axL.grid(alpha=0.3)

    # (b) 두 정책 차이가 N에 따라 유의해지는가
    Ns2 = np.array([20, 30, 50, 75, 100, 150, 200, 300, 500])
    pvals = [two_prop_z(0.90, 0.78, n, n)[1] for n in Ns2]
    axR.plot(Ns2, pvals, 'o-', color=C_OK, lw=2, markersize=5)
    axR.axhline(0.05, color=C_BAD, ls='--', lw=1.5, label='α=0.05 문턱')
    # 유의해지는 최소 N
    n_sig = next(n for n in Ns2 if two_prop_z(0.90, 0.78, n, n)[1] < 0.05)
    axR.axvline(n_sig, color=C_WARN, ls=':', lw=1.4)
    axR.annotate(f'N≈{n_sig}/arm부터\n유의', xy=(n_sig, 0.05), xytext=(n_sig+40, 0.22),
                 fontsize=9, color=C_WARN,
                 arrowprops=dict(arrowstyle='->', color=C_WARN))
    axR.set_xlabel('arm당 표본수 N', fontsize=10)
    axR.set_ylabel('two-proportion p-value', fontsize=10)
    axR.set_title('(b) 90% vs 78% 차이 — N이 커야 유의해진다', fontsize=11, fontweight='bold')
    axR.set_ylim(0, 0.5); axR.legend(fontsize=8.8); axR.grid(alpha=0.3)
    fig.suptitle('그림 6 · 평가 신뢰성 가중 (E3) — 성공률 숫자를 N·CI·유의성으로 할인',
                 fontsize=12.5, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT + 'fig6_eval_reliability.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print("saved fig6_eval_reliability.png")

if __name__ == '__main__':
    radar()
    checklist_heatmap()
    layer_diagnosis()
    filled_table()
    eval_reliability()
    print("\n모든 그림 생성 완료.")
