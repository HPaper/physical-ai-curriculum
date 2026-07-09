"""Lec 25 그림 생성 스크립트 — "딥러닝, 왜 로봇에 필요한가".
실행: cd images/lec25 && python3 gen_figs.py
본문·Worked Example의 수치와 동일한 numpy 토이를 사용한다(재현성 각주 참조).
순수 numpy/matplotlib. 결정론적 시드(default_rng(0)).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = "."

C = {
    'rule': '#d62728', 'learn': '#1f77b4', 'truth': '#333333',
    'ok': '#2ca02c', 'no': '#d62728', 'grid': '#cccccc',
    'sw': '#e1eefb', 'swE': '#2c6fb0', 'ph': '#efe4d8', 'phE': '#9a6a3a',
    'hl': '#ffe08a', 'hlE': '#c79a2a', 'io': '#f3eede', 'ioE': '#8a7a4a',
}


# ============================================================
# 그림 1 — 룰베이스 가능/불능 4사분면
#   축: 모델 가용성 f(x) (x) × 태스크 명세성 setpoint (y)
# ============================================================
def fig1_quadrant():
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # 4사분면 배경
    ax.axhspan(0.5, 1.0, 0.5, 1.0, color=C['ok'], alpha=0.10)   # 우상: 둘 다 O
    ax.axhspan(0.0, 0.5, 0.0, 0.5, color=C['no'], alpha=0.12)   # 좌하: 둘 다 X
    ax.axhspan(0.5, 1.0, 0.0, 0.5, color='#ffd27f', alpha=0.18)
    ax.axhspan(0.0, 0.5, 0.5, 1.0, color='#ffd27f', alpha=0.18)

    ax.axhline(0.5, color='#555', lw=1.2)
    ax.axvline(0.5, color='#555', lw=1.2)

    # 사분면 라벨
    ax.text(0.75, 0.93, "룰베이스가 훌륭한 영역", ha='center', fontsize=12.5,
            color=C['ok'], fontweight='bold')
    ax.text(0.75, 0.88, "해석적 f + 손설계 C + 셋포인트", ha='center', fontsize=9.5, color='#2a2a2a')
    ax.text(0.25, 0.07, "룰베이스가 막히는 영역", ha='center', fontsize=12.5,
            color=C['no'], fontweight='bold')
    ax.text(0.25, 0.02, "→ 데이터로 π를 학습", ha='center', fontsize=9.5, color='#2a2a2a')

    # 예시 시스템 배치 (x=모델 가용성, y=태스크 명세성)
    pts = [
        # 우상: 둘 다 높음 — 고전 제어의 홈그라운드
        (0.90, 0.93, "핀-홀\n삽입", C['ok']),
        (0.86, 0.80, "용접\n(정형 경로)", C['ok']),
        (0.78, 0.72, "픽앤플레이스\n(정위치)", C['ok']),
        (0.72, 0.90, "2링크\nIK(7강)", C['ok']),
        # 우하: 모델은 있으나 목표가 자연어
        (0.80, 0.30, "\"식탁을\n치워라\"", '#b8860b'),
        # 좌상: 셋포인트는 있으나 모델(픽셀→상태 f)이 없음
        (0.22, 0.82, "원시 픽셀\n기반 파지", '#b8860b'),
        # 좌하: 둘 다 없음 — 빨래 개기
        (0.20, 0.22, "빨래 개기\n(π0, 49강)", C['no']),
        (0.32, 0.34, "가변형물체\n조작", C['no']),
        (0.14, 0.40, "접촉多\n어셈블리(12강)", C['no']),
    ]
    for x, y, lab, col in pts:
        ax.scatter([x], [y], s=170, color=col, edgecolor='white', lw=1.6, zorder=5)
        ax.text(x, y - 0.055, lab, ha='center', va='top', fontsize=8.4,
                color='#1a1a1a', zorder=6, linespacing=1.0)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0.12, 0.88]); ax.set_xticklabels(["모델 없음\n(픽셀→상태 f 부재·변형체)", "모델 있음\n(해석적 f, M(q))"], fontsize=9.5)
    ax.set_yticks([0.12, 0.88]); ax.set_yticklabels(["명세 불가\n(\"깔끔하게 개라\")", "명세 가능\n(셋포인트·좌표)"], fontsize=9.5, rotation=90, va='center')
    ax.set_xlabel("모델 가용성:  이미지→상태의 해석식이 있는가", fontsize=11)
    ax.set_ylabel("태스크 명세성:  목표가 셋포인트로 적히는가", fontsize=11)
    ax.set_title("룰베이스가 성립하는 두 조건 — 하나만 걸려도 학습이 필요해진다", fontsize=12.5, pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_rulebase_quadrant.png", dpi=130)
    plt.close(fig)
    print("saved fig1_rulebase_quadrant.png")


# ============================================================
# 그림 2 — 차원의 저주: N = K^d (로그스케일)
# ============================================================
def fig2_curse():
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    d = np.arange(1, 45)
    for K, col, lab in [(2, '#1f77b4', 'K=2 (이진 픽셀)'),
                        (10, '#ff7f0e', 'K=10'),
                        (256, '#d62728', 'K=256 (8bit)')]:
        log10N = d * np.log10(K)
        ax.plot(d, log10N, color=col, lw=2.2, label=lab, marker='o', ms=3)

    # 참조선: 텔레옵 궤적 10^4, 우주 원자수 10^80
    ax.axhline(4, color='#2ca02c', ls='--', lw=1.6)
    ax.text(1.2, 5.2, "텔레옵 궤적 $10^4$개 (유효 표본)", color='#2ca02c', fontsize=10)
    ax.axhline(80, color='#7f7f7f', ls=':', lw=1.6)
    ax.text(1.2, 82, "관측가능 우주의 원자 수 $\\sim 10^{80}$", color='#555', fontsize=9.5)

    # 200x200 이진 이미지 마커: d=40000 은 축 밖 → 화살표로 표기
    ax.annotate("$200{\\times}200$ 이진 이미지: $d{=}4{\\times}10^4$\n→ $N{=}2^{40000}{\\approx}10^{12041}$ (축 밖)",
                xy=(44, 44*np.log10(2)), xytext=(20, 60),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4),
                fontsize=9.5, color='#1f77b4',
                bbox=dict(boxstyle='round', fc='white', ec='#1f77b4', alpha=0.9))

    ax.set_xlabel("관측 차원 $d$ (격자로 다루는 변수 개수)", fontsize=11)
    ax.set_ylabel("필요한 칸 수 $\\log_{10} N,\\; N=K^d$", fontsize=11)
    ax.set_title("차원의 저주 — 룩업 테이블의 칸 수는 차원에 지수적", fontsize=12.5)
    ax.set_ylim(0, 100)
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_curse_of_dim.png", dpi=130)
    plt.close(fig)
    print("saved fig2_curse_of_dim.png")


# ============================================================
# 그림 3 — WE-2: 룰 vs 학습 일반화 (분포이동/보간)
# ============================================================
def fig3_rule_vs_learn():
    rng = np.random.default_rng(0)
    def gt(x): return np.sin(1.4 * x) + 0.15 * x

    xl = rng.uniform(-2.0, -0.5, 30)
    xr = rng.uniform(0.5, 2.0, 30)
    x_tr = np.sort(np.concatenate([xl, xr]))
    y_tr = gt(x_tr) + 0.02 * rng.standard_normal(x_tr.size)

    ml, bl = np.linalg.lstsq(np.stack([xl, np.ones_like(xl)], 1), gt(xl), rcond=None)[0]
    mr, br = np.linalg.lstsq(np.stack([xr, np.ones_like(xr)], 1), gt(xr), rcond=None)[0]
    def rule(x):
        x = np.asarray(x, float); return np.where(x < 0.0, ml * x + bl, mr * x + br)

    centers = np.linspace(-2.0, 2.0, 12); gamma = 1.5
    def rbf(x):
        x = np.asarray(x, float).reshape(-1, 1)
        return np.exp(-gamma * (x - centers.reshape(1, -1))**2)
    Phi = rbf(x_tr); lam = 1e-2
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(12), Phi.T @ y_tr)
    def learned(x): return (rbf(x) @ w).ravel()

    xs = np.linspace(-2.0, 2.0, 600)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    # 훈련 데이터가 없는 gap 음영
    ax.axvspan(-0.4, 0.4, color='#ffd27f', alpha=0.35, label='분포이동: 훈련 데이터 없는 구간')

    ax.plot(xs, gt(xs), color=C['truth'], lw=2.4, label='정답 함수 $a^*(x)$')
    # 룰: gap 양쪽을 각각 그려 불연속 점프를 노출
    xm = xs[xs < 0]; xp = xs[xs >= 0]
    ax.plot(xm, rule(xm), color=C['rule'], lw=2.2, ls='--')
    ax.plot(xp, rule(xp), color=C['rule'], lw=2.2, ls='--', label='룰베이스 (임계 스위치)')
    # 점프 표시
    ax.plot([0, 0], [rule(np.array([-1e-6]))[0], rule(np.array([1e-6]))[0]],
            color=C['rule'], lw=1.4, ls=':', alpha=0.8)
    ax.annotate("불연속 점프 2.243\n(정답은 0)", xy=(0.0, 0.0), xytext=(0.55, -1.4),
                arrowprops=dict(arrowstyle='->', color=C['rule'], lw=1.3),
                fontsize=9.2, color=C['rule'])

    ax.plot(xs, learned(xs), color=C['learn'], lw=2.4, label='학습 함수 (RBF 회귀, 보간)')
    ax.scatter(x_tr, y_tr, s=22, color='#555', alpha=0.55, zorder=5, label='훈련 표본 (양쪽 군집)')

    ax.set_xlabel("입력 특징 $x$  (예: 천 가장자리 밝기)", fontsize=11)
    ax.set_ylabel("행동 $a$  (예: 미는 정도)", fontsize=11)
    ax.set_title("분포이동에서 룰은 무너지고(RMSE 0.82), 학습은 보간한다(0.013)", fontsize=12)
    ax.legend(loc='upper left', fontsize=8.8, ncol=1)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-2.2, 2.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_rule_vs_learn.png", dpi=130)
    plt.close(fig)
    print("saved fig3_rule_vs_learn.png")


# ============================================================
# 그림 4 — 0강 지도 위, "AI 파트가 채우는 블록" 하이라이트
# ============================================================
def _box(ax, x, y, w, h, text, fc, ec, fs=9.5, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                         fc=fc, ec=ec, lw=1.8)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color='#1a1a1a', fontweight='bold' if bold else 'normal',
            linespacing=1.05)


def _arrow(ax, x0, y0, x1, y1, col='#555', lw=1.6, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=13, color=col, lw=lw, ls=ls,
                 shrinkA=2, shrinkB=2))


def fig4_map_highlight():
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

    # 연산(파랑) 블록 — 지각·판단은 AI가 채운다(하이라이트)
    _box(ax, 0.3, 4.2, 2.2, 1.2, "감지·지각 φ\n센서→특징 z", C['hl'], C['hlE'], bold=True)
    _box(ax, 2.9, 4.2, 2.2, 1.2, "판단·정책 π\n z→행동 a", C['hl'], C['hlE'], bold=True)
    ax.text(2.7, 5.62, "★ AI 파트(25~48강)가 훈련된 함수로 채우는 블록",
            ha='center', fontsize=10.5, color=C['hlE'], fontweight='bold')

    # 제어(파랑, 그대로 고전제어)
    _box(ax, 5.5, 4.2, 2.2, 1.2, "제어 C\n a→τ (17~24강)", C['sw'], C['swE'])
    # 물리(갈색) 블록 — 남는다
    _box(ax, 0.3, 1.4, 2.2, 1.2, "구동기\n전류→토크", C['ph'], C['phE'])
    _box(ax, 2.9, 1.4, 2.2, 1.2, "로봇 몸체\n토크→운동", C['ph'], C['phE'])
    _box(ax, 5.5, 1.4, 2.2, 1.2, "환경\n(실제/시뮬)", C['ph'], C['phE'])
    ax.text(4.0, 0.75, "물리 — 학습이 대체하지 않는다(0강: 층을 없앤 게 아니라 옮겼다)",
            ha='center', fontsize=10, color=C['phE'])

    # 신호 흐름
    _arrow(ax, 2.5, 4.8, 2.9, 4.8)                 # φ->π
    _arrow(ax, 5.1, 4.8, 5.5, 4.8)                 # π->C
    _arrow(ax, 6.6, 4.2, 3.9, 2.6, col=C['swE'])   # C->로봇(τ)
    _arrow(ax, 2.5, 2.0, 2.9, 2.0)                 # 구동->몸체
    _arrow(ax, 5.1, 2.0, 5.5, 2.0)                 # 몸체->환경
    # 폐루프: 환경 -> 지각
    ax.add_patch(FancyArrowPatch((6.6, 2.6), (1.4, 4.2), arrowstyle='-|>',
                 mutation_scale=13, color='#888', lw=1.4, ls=(0, (4, 3)),
                 connectionstyle="arc3,rad=-0.25", shrinkA=2, shrinkB=2))
    ax.text(8.05, 3.5, "관측 o\n(카메라·엔코더)", fontsize=8.6, color='#666', ha='left')

    # 룰베이스 대비 메모
    ax.text(9.9, 5.3, "룰베이스: 노란 블록을\n손코드로 채움\n(φ=해석식, π=셋포인트+IK)",
            ha='right', va='top', fontsize=8.8, color='#8a5a00',
            bbox=dict(boxstyle='round', fc='#fff3d6', ec=C['hlE'], alpha=0.95))

    ax.set_title("패러다임 전환 — 같은 지도, 노란 두 블록만 손코드→학습 함수로 교체",
                 fontsize=12.5, pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_map_highlight.png", dpi=130)
    plt.close(fig)
    print("saved fig4_map_highlight.png")


if __name__ == "__main__":
    fig1_quadrant()
    fig2_curse()
    fig3_rule_vs_learn()
    fig4_map_highlight()
    print("ALL FIGURES DONE")
