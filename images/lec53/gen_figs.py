"""
Lec 53 그림 생성 — 합성 데이터와 도메인 랜덤화.
순수 numpy/scipy/matplotlib, 결정론적 시드. CPU 개념 토이.
실행: cd images/lec53 && python3 gen_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 110

C_ORIG = '#d62728'    # 원 데모
C_NEW = '#1f77b4'     # 증식된 데모
C_OK = '#2ca02c'
C_BAD = '#d62728'
C_GRID = '#cccccc'


# ---------------------------------------------------------------------------
# SE(3) 유틸 (3강 회수) — 2D 강체변환 T = [[R, t],[0,1]]
# ---------------------------------------------------------------------------
def se2(theta, tx, ty):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def apply(T, pts):
    """pts: (N,2) -> (N,2). 동차좌표로 변환."""
    h = np.c_[pts, np.ones(len(pts))]
    return (T @ h.T).T[:, :2]


# ===========================================================================
# 그림 1: MimicGen식 SE(3) 데모 증식 (1 -> N 포즈)
# ===========================================================================
def fig1_mimicgen_augment():
    rng = np.random.default_rng(0)
    # 원 데모: 물체 프레임에서 정의된 접근-파지 궤적 (물체 원점 기준)
    # object-centric segment: 물체 좌표계에서 위로 접근 후 내려와 파지
    t = np.linspace(0, 1, 40)
    # 물체 프레임 기준 궤적: 위쪽에서 들어와 원점(파지점)으로
    traj_obj = np.c_[0.35 * np.cos(1.5 * np.pi * t) * (1 - t),
                     0.30 * (1 - t) + 0.02]
    grasp_obj = np.array([[0.0, 0.02]])  # 파지 목표점(물체 프레임)

    # 원 데모의 물체 포즈 T_old
    T_old = se2(0.0, 0.0, 0.0)
    # 새 물체 포즈들 T_new (N개) — 위치+회전 랜덤
    N = 6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    # (a) 원 데모 하나
    ax = axes[0]
    world_orig = apply(T_old, traj_obj)
    ax.plot(world_orig[:, 0], world_orig[:, 1], '-', color=C_ORIG, lw=2.4,
            label='원 데모 궤적 (1개)')
    ax.plot(world_orig[0, 0], world_orig[0, 1], 'o', color=C_ORIG, ms=8)
    gp = apply(T_old, grasp_obj)[0]
    ax.plot(gp[0], gp[1], '*', color='k', ms=16, label='물체(파지점)')
    ax.add_patch(plt.Rectangle((-0.08, -0.06), 0.16, 0.10, fc='#888', ec='k', alpha=0.5))
    ax.set_title('(a) 사람 시연 1개\n물체 프레임의 object-centric segment', fontsize=11)
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.15, 0.45)
    ax.set_aspect('equal'); ax.grid(True, color=C_GRID, alpha=0.6)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    # (b) N개 새 포즈로 증식
    ax = axes[1]
    ax.plot(world_orig[:, 0], world_orig[:, 1], '--', color=C_ORIG, lw=1.6,
            alpha=0.7, label='원 데모')
    poses = []
    for i in range(N):
        th = rng.uniform(-0.9, 0.9)
        tx = rng.uniform(-0.42, 0.42)
        ty = rng.uniform(-0.02, 0.10)
        T_new = se2(th, tx, ty)
        # a'_t = T_new @ T_old^{-1} @ a_t  (물체 프레임 재생)
        T_rel = T_new @ np.linalg.inv(T_old)
        wtraj = apply(T_rel, world_orig)
        wgrasp = apply(T_rel, gp.reshape(1, 2))[0]
        ax.plot(wtraj[:, 0], wtraj[:, 1], '-', color=C_NEW, lw=1.5, alpha=0.85)
        ax.plot(wgrasp[0], wgrasp[1], '*', color='k', ms=12)
        ax.add_patch(plt.Rectangle((wgrasp[0] - 0.06, wgrasp[1] - 0.05),
                                   0.12, 0.08, fc='#888', ec='k', alpha=0.35))
        poses.append((th, tx, ty))
    ax.plot([], [], '-', color=C_NEW, lw=1.6, label=f'증식된 데모 ({N}개)')
    ax.set_title('(b) SE(3) 변환으로 증식\n' r"$a'_t = T_{new}\,T_{old}^{-1}\,a_t$",
                 fontsize=11)
    ax.set_xlim(-0.65, 0.65); ax.set_ylim(-0.15, 0.45)
    ax.set_aspect('equal'); ax.grid(True, color=C_GRID, alpha=0.6)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('x [m]')

    fig.suptitle('MimicGen식 데모 증식: 물체 프레임 궤적을 새 포즈로 강체변환·재생 (3강 SE(3) 회수)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig('fig1_mimicgen_augment.png', bbox_inches='tight')
    plt.close(fig)
    print('[fig1] saved.', f'포즈들(theta,tx,ty)={np.round(poses,3).tolist()}')


# ===========================================================================
# 그림 2: 도메인 랜덤화 커버리지 (폭 vs 커버 확률)
# ===========================================================================
def fig2_dr_coverage():
    # 실제 파라미터가 sim 명목값(mu0)에서 delta_real 만큼 벗어나 있다.
    # 랜덤화: param ~ Uniform(mu0 - w, mu0 + w).  '커버'는 실제값이 이 구간 안에 드는 것.
    # 실제값 자체가 불확실(우리가 모르므로) -> 실제값이 N(mu0, s_real) 분포라 가정.
    # 커버 확률 = P(|param_real - mu0| <= w) = erf(w / (sqrt(2) s_real))
    from scipy.special import erf
    s_real = 0.20   # 실제 파라미터의 sim 명목값 대비 표준편차 (예: 마찰계수 상대오차)
    w = np.linspace(0, 0.9, 200)
    cover = erf(w / (np.sqrt(2) * s_real))

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(w, cover, '-', color=C_NEW, lw=2.6)
    # 특정 폭에서의 커버 확률 표시
    for wi, col in [(0.20, '#ff7f0e'), (0.40, '#2ca02c'), (0.60, '#9467bd')]:
        ci = erf(wi / (np.sqrt(2) * s_real))
        ax.plot([wi, wi], [0, ci], ':', color=col, lw=1.5)
        ax.plot([0, wi], [ci, ci], ':', color=col, lw=1.5)
        ax.plot(wi, ci, 'o', color=col, ms=8)
        ax.annotate(f'w={wi:.1f}σ_r={wi/s_real:.0f}σ\ncover={ci:.3f}',
                    (wi, ci), textcoords='offset points', xytext=(8, -28),
                    fontsize=8.5, color=col)
    ax.axhline(0.997, color='gray', ls='--', lw=1, alpha=0.7)
    ax.text(0.02, 0.965, '99.7% (≈3σ_r)', fontsize=8.5, color='gray')
    ax.set_xlabel('랜덤화 폭 w  (파라미터 명목값 대비)')
    ax.set_ylabel('실제 파라미터 커버 확률')
    ax.set_title('도메인 랜덤화 커버리지: 폭을 넓힐수록 실제가 분포 안에 든다\n'
                 r'($\sigma_r=0.20$; cover $= \mathrm{erf}(w/\sqrt{2}\sigma_r)$)',
                 fontsize=11)
    ax.set_ylim(0, 1.03); ax.set_xlim(0, 0.9)
    ax.grid(True, color=C_GRID, alpha=0.6)
    fig.tight_layout()
    fig.savefig('fig2_dr_coverage.png', bbox_inches='tight')
    plt.close(fig)
    print('[fig2] saved.',
          f'cover(w=0.2)={erf(0.2/(np.sqrt(2)*s_real)):.4f}, '
          f'cover(w=0.4)={erf(0.4/(np.sqrt(2)*s_real)):.4f}, '
          f'cover(w=0.6)={erf(0.6/(np.sqrt(2)*s_real)):.4f}')


# ===========================================================================
# 그림 3: 강건성-학습난이도 트레이드오프
# ===========================================================================
def fig3_robustness_tradeoff():
    """
    폭 w가 커지면:
      - 커버 확률 ↑ -> 실제에서 성공 가능성 ↑ (강건성)
      - 그러나 정책이 넓은 분포를 동시에 맞춰야 함 -> 최적 성능 ↓ (학습난이도)
    토이 모델:
      real_success(w) = cover(w) * peak(w)
        cover(w) = erf(w/(sqrt2 s_real))           # 실제가 분포 안에 들 확률
        peak(w)  = 1/(1 + (w/w_cap)^2)             # 넓힐수록 정책 용량 한계로 개별 성능 저하
    """
    from scipy.special import erf
    s_real = 0.20
    w_cap = 0.45   # 정책 용량 스케일 (넓어질수록 개별 파라미터 성능 저하)
    w = np.linspace(0.01, 0.9, 300)
    cover = erf(w / (np.sqrt(2) * s_real))
    peak = 1.0 / (1.0 + (w / w_cap) ** 2)
    real_succ = cover * peak

    w_star = w[np.argmax(real_succ)]
    s_star = real_succ.max()

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.plot(w, cover, '--', color='#2ca02c', lw=2, label='커버 확률 (강건성↑)')
    ax.plot(w, peak, '--', color='#d62728', lw=2, label='분포 내 개별 성능 (학습난이도↑로 저하)')
    ax.plot(w, real_succ, '-', color=C_NEW, lw=3, label='실제 성공률 ≈ 커버 × 개별성능')
    ax.plot(w_star, s_star, '*', color='k', ms=18, zorder=5)
    ax.annotate(f'최적 폭 w*={w_star:.2f}\n성공률={s_star:.3f}',
                (w_star, s_star), textcoords='offset points', xytext=(12, -6),
                fontsize=10, fontweight='bold')
    ax.axvspan(0, w_star * 0.5, alpha=0.06, color='red')
    ax.axvspan(w_star * 1.6, 0.9, alpha=0.06, color='red')
    ax.text(0.03, 0.10, '너무 좁음\n(실제 미커버)', fontsize=8.5, color='#a33')
    ax.text(0.72, 0.10, '너무 넓음\n(학습 과부하)', fontsize=8.5, color='#a33')
    ax.set_xlabel('랜덤화 폭 w')
    ax.set_ylabel('확률 / 성공률')
    ax.set_title('강건성 ↔ 학습난이도 트레이드오프: DR은 "많을수록 좋다"가 아니다',
                 fontsize=11.5)
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 0.9)
    ax.legend(fontsize=9, loc='upper center')
    ax.grid(True, color=C_GRID, alpha=0.6)
    fig.tight_layout()
    fig.savefig('fig3_robustness_tradeoff.png', bbox_inches='tight')
    plt.close(fig)
    print('[fig3] saved.', f'w*={w_star:.4f}, success*={s_star:.4f}')
    return w_star, s_star


# ===========================================================================
# 그림 4: neural trajectory 생성 파이프라인 (DreamGen/Cosmos)
# ===========================================================================
def fig4_neural_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.axis('off')

    boxes = [
        (0.02, "단일/소수\n실기 데이터\n(예: pick&place\n1개 태스크)", '#e8f0fe'),
        (0.215, "World Model\n(Cosmos WFM)\n이미지+지시\n→ 영상 생성", '#fde8e8'),
        (0.41, "'dream' 영상\n(픽셀만,\n행동 라벨 없음)", '#fff4e0'),
        (0.605, "IDM / latent\naction model\n영상 → 유사행동\n라벨링", '#e8f8e8'),
        (0.80, "neural trajectory\n(영상+유사행동)\n→ 정책 학습", '#e8f0fe'),
    ]
    bw, bh, by = 0.16, 0.42, 0.30
    centers = []
    for x, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x, by), bw, bh, fc=col, ec='#333', lw=1.4,
                                   transform=ax.transAxes))
        ax.text(x + bw / 2, by + bh / 2, txt, ha='center', va='center',
                fontsize=9.3, transform=ax.transAxes)
        centers.append(x + bw)

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + bw
        x1 = boxes[i + 1][0]
        ax.annotate('', xy=(x1, by + bh / 2), xytext=(x0, by + bh / 2),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.8))

    ax.text(0.5, 0.90, 'neural trajectory 생성: 실제 롤아웃 없이 world model로 데이터를 "꿈꾼다"',
            ha='center', fontsize=12, transform=ax.transAxes, fontweight='bold')
    ax.text(0.315, 0.14, '행동조건부 생성', ha='center', fontsize=8.5,
            color='#a33', transform=ax.transAxes, style='italic')
    ax.text(0.70, 0.14, '역동역학/잠재행동 라벨링', ha='center', fontsize=8.5,
            color='#a33', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.03,
            '⚠ 생성모델이므로 물리 위반 가능 (54강) · 여전히 sim2real/real 갭 존재',
            ha='center', fontsize=8.8, color='#666', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig('fig4_neural_pipeline.png', bbox_inches='tight')
    plt.close(fig)
    print('[fig4] saved.')


if __name__ == '__main__':
    fig1_mimicgen_augment()
    fig2_dr_coverage()
    fig3_robustness_tradeoff()
    fig4_neural_pipeline()
    print('모든 그림 생성 완료.')
