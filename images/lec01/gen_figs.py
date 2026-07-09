# Lec 01. 로봇 해부학 — 그림 생성 스크립트
# 실행: 이 디렉토리에서  python3 gen_figs.py
# 주의: fig1_workspace_redundancy.png 는 기존 자산 — 이 스크립트는 fig2~fig5 만 생성한다.
import warnings; warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
np.random.seed(1)

C_BLUE, C_RED, C_GREEN, C_GRAY = '#1f77b4', '#d62728', '#2ca02c', '#999999'
C_ORANGE, C_PURPLE = '#ff7f0e', '#9467bd'


# ============================================================
# fig2: 직렬(serial) vs 병렬(parallel) 구조 — 구조와 DoF 비교
# 본문 §5의 "직렬 6R" / WE-1의 "평면 5절 병렬 팔" 시각화
# ============================================================
def fig2_serial_vs_parallel():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # ---- (a) 직렬 개루프 사슬: 지면→L1→L2→L3, R관절 3개 ----
    def fk_chain(L, q):
        p, a, pts = np.zeros(2), 0.0, [np.zeros(2)]
        for li, qi in zip(L, q):
            a += qi
            p = p + li * np.array([np.cos(a), np.sin(a)])
            pts.append(p.copy())
        return np.array(pts)

    L = [1.0, 0.8, 0.6]
    pts = fk_chain(L, [0.5, 0.6, -0.7])
    a1.plot(pts[:, 0], pts[:, 1], '-', color=C_BLUE, lw=6, solid_capstyle='round', zorder=2)
    # 관절(회전) 표시
    for i, p in enumerate(pts[:-1]):
        a1.add_patch(Circle(p, 0.075, fc='white', ec=C_RED, lw=2.2, zorder=4))
        a1.text(p[0], p[1] - 0.22, f'R{i+1}', ha='center', color=C_RED, fontsize=11, fontweight='bold')
    # 지면
    a1.plot([-0.35, 0.35], [0, 0], color='k', lw=2)
    for x0 in np.linspace(-0.32, 0.28, 6):
        a1.plot([x0, x0 - 0.12], [0, -0.14], color='k', lw=1)
    # EEF
    a1.scatter(*pts[-1], marker='*', s=360, color=C_GREEN, zorder=5, edgecolor='k')
    a1.text(pts[-1][0] + 0.06, pts[-1][1] + 0.05, 'EEF', color=C_GREEN, fontsize=11, fontweight='bold')
    a1.set_title('(a) 직렬 구조 (개루프 사슬)\n'
                 r'$N{=}4,\ J{=}3,\ f_i{=}1,\ m{=}3$' + '\n'
                 r'dof $= 3(4{-}1{-}3)+3 = \mathbf{3}$ = 관절 합', fontsize=11)
    a1.set_xlim(-0.7, 2.6); a1.set_ylim(-0.6, 2.2); a1.set_aspect('equal'); a1.axis('off')
    a1.text(-0.6, 2.05, '지면 제외 링크마다 자유를 주고\n관절마다 구속을 깎음 → 겹칠 일 없음',
            fontsize=9, color=C_GRAY, va='top')

    # ---- (b) 병렬 폐루프: 평면 5절 링키지 (2모터 병렬 팔) ----
    # 두 고정 베이스에서 각각 2링크가 올라와 공통 커플러점에서 만남
    base1, base2 = np.array([-0.6, 0.0]), np.array([0.6, 0.0])
    # 커플러 점(플랫폼)
    plat = np.array([0.0, 1.15])
    # 각 다리의 중간 관절
    mid1 = base1 + np.array([0.15, 0.85])
    mid2 = base2 + np.array([-0.15, 0.85])
    legs = [(base1, mid1, plat), (base2, mid2, plat)]
    for (b, mdl, top) in legs:
        seg = np.array([b, mdl, top])
        a2.plot(seg[:, 0], seg[:, 1], '-', color=C_BLUE, lw=6, solid_capstyle='round', zorder=2)
    # 관절 5개: base1, mid1, plat, mid2, base2 (plat은 두 다리 공유 → 5개 관절)
    joint_pts = [base1, mid1, plat, mid2, base2]
    for p in joint_pts:
        a2.add_patch(Circle(p, 0.07, fc='white', ec=C_RED, lw=2.2, zorder=4))
    # 플랫폼(엔드이펙터) 강조
    a2.scatter(*plat, marker='*', s=360, color=C_GREEN, zorder=5, edgecolor='k')
    a2.text(plat[0] + 0.06, plat[1] + 0.06, '플랫폼(EEF)', color=C_GREEN, fontsize=10, fontweight='bold')
    # 지면 표시
    for b in (base1, base2):
        a2.plot([b[0] - 0.2, b[0] + 0.2], [0, 0], color='k', lw=2)
        for x0 in np.linspace(b[0] - 0.18, b[0] + 0.12, 4):
            a2.plot([x0, x0 - 0.08], [0, -0.1], color='k', lw=1)
    # 폐루프임을 나타내는 음영
    loop = np.array([base1, mid1, plat, mid2, base2])
    a2.fill(loop[:, 0], loop[:, 1], color=C_ORANGE, alpha=0.08, zorder=1)
    a2.set_title('(b) 병렬 구조 (폐루프 5절)\n'
                 r'$N{=}5,\ J{=}5,\ f_i{=}1,\ m{=}3$' + '\n'
                 r'dof $= 3(5{-}1{-}5)+5 = \mathbf{2}$', fontsize=11)
    a2.set_xlim(-1.15, 1.15); a2.set_ylim(-0.35, 1.6); a2.set_aspect('equal'); a2.axis('off')
    a2.text(-1.1, 1.5, '루프가 닫히면 구속이 서로 얽힘\n→ dof < 관절 합 (여기선 5→2)',
            fontsize=9, color=C_GRAY, va='top')

    fig.suptitle('직렬 vs 병렬: Grübler 공식으로 세는 구조별 자유도  (본문 §5 · WE-1)',
                 fontsize=12.5, y=1.005)
    plt.tight_layout()
    plt.savefig('fig2_serial_vs_parallel.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================
# fig3: Grübler 공식 DoF 계산 도해 (막대 분해)
# dof = m(N-1) - sum(m - f_i) 를 "자유 - 구속" 막대로 시각화
# 본문 E2 · WE-1
# ============================================================
def fig3_grubler_breakdown():
    cases = [
        ('평면 4절\n(four-bar)', 3, 4, [1, 1, 1, 1], 1),
        ('평면 5절\n(병렬 팔)', 3, 5, [1] * 5, 2),
        ('공간 직렬 6R\n(UR5e급)', 6, 7, [1] * 6, 6),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax, (name, m, N, fs, expect) in zip(axes, cases):
        J = len(fs)
        free = m * (N - 1)                    # 지면 제외 링크의 총 자유
        constr = sum(m - fi for fi in fs)     # 관절들이 깎는 총 구속
        dof = free - constr
        assert dof == expect, (name, dof, expect)

        # 자유 막대(위로) vs 구속 막대(자유 위에서 아래로 깎임)
        ax.bar(0, free, width=0.55, color=C_BLUE, alpha=0.85, label='자유  $m(N{-}1)$')
        ax.bar(0, -constr, width=0.55, bottom=free, color=C_RED, alpha=0.85,
               label=r'구속  $\sum(m{-}f_i)$')
        # 남는 dof 표시
        ax.plot([-0.45, 0.45], [dof, dof], color=C_GREEN, lw=2.5, zorder=5)
        ax.annotate(f'dof = {dof}', xy=(0, dof), xytext=(0.62, dof),
                    fontsize=12, fontweight='bold', color=C_GREEN, va='center')
        ax.text(0, free + 0.35, f'{free}', ha='center', color=C_BLUE, fontsize=10, fontweight='bold')
        ax.text(0, free - constr / 2, f'−{constr}', ha='center', color='white',
                fontsize=11, fontweight='bold')
        ax.set_title(f'{name}\n' + rf'$m{{=}}{m},\ N{{=}}{N},\ J{{=}}{J}$', fontsize=11)
        ax.set_xlim(-0.9, 1.5); ax.set_ylim(min(0, dof) - 0.5, free + 1.2)
        ax.set_xticks([]); ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='k', lw=0.8)
    axes[0].set_ylabel('자유도 셈 (단위: 스칼라 구속)')
    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    fig.suptitle(r'Grübler 도해:  dof $= m(N{-}1) - \sum_i (m - f_i)$ '
                 '— 자유를 주고(파랑) 구속을 깎아(빨강) 남는 것(초록)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('fig3_grubler_breakdown.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================
# fig4: 관절공간 → 작업공간 사상 (2R 팔)
# 왼쪽: C-space(토러스 격자, q1-q2 평면) / 오른쪽: 작업공간(도달 고리)
# 대응하는 격자선을 색으로 잇는다 → 사상 f 가 공간을 접고 늘림
# 본문 E3 · WE-2(a)
# ============================================================
def fig4_joint_to_task():
    l1, l2 = 1.0, 0.6

    def fk(q1, q2):
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        return x, y

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # 배경: 작업공간(고리) — q 균일 표본
    q1s = np.random.uniform(-np.pi, np.pi, 40000)
    q2s = np.random.uniform(-np.pi, np.pi, 40000)
    xs, ys = fk(q1s, q2s)
    a2.scatter(xs, ys, s=0.4, color=C_GRAY, alpha=0.25, zorder=1)

    # 격자선 몇 개를 골라 두 공간에서 같은 색으로
    q1_lines = np.linspace(-np.pi, np.pi, 9)
    cmap = plt.cm.viridis
    tt = np.linspace(-np.pi, np.pi, 200)
    for k, q1v in enumerate(q1_lines):
        c = cmap(k / (len(q1_lines) - 1))
        # C-space: q1=const 인 수직선
        a1.plot([q1v, q1v], [-np.pi, np.pi], color=c, lw=2)
        # 작업공간: 그 선의 상(q2 변화 → 원호)
        xg, yg = fk(np.full_like(tt, q1v), tt)
        a2.plot(xg, yg, color=c, lw=1.8, zorder=3)

    # 이론 내·외경 원
    for R, ls, lab in [(l1 + l2, '-', f'외경 $l_1{{+}}l_2={l1+l2}$'),
                       (abs(l1 - l2), '--', f'내경 $|l_1{{-}}l_2|={abs(l1-l2):.1f}$')]:
        th = np.linspace(0, 2 * np.pi, 200)
        a2.plot(R * np.cos(th), R * np.sin(th), ls, color='k', lw=1.2, alpha=0.7, label=lab)

    a1.set_title('(a) 관절공간 $\\mathcal{C}=T^2$\n$q_1$-const 선들 (색 = 대응)', fontsize=11)
    a1.set_xlabel('$q_1$ (rad)'); a1.set_ylabel('$q_2$ (rad)')
    a1.set_xlim(-np.pi, np.pi); a1.set_ylim(-np.pi, np.pi); a1.set_aspect('equal')
    a1.set_xticks([-np.pi, 0, np.pi]); a1.set_xticklabels(['$-\\pi$', '0', '$\\pi$'])
    a1.set_yticks([-np.pi, 0, np.pi]); a1.set_yticklabels(['$-\\pi$', '0', '$\\pi$'])
    a1.grid(alpha=0.3)

    a2.set_title('(b) 작업공간 = 사상 $f$의 상(image)\n'
                 r'$x=f(q)$ — 격자가 접히고 늘어남', fontsize=11)
    a2.set_xlabel('$x$'); a2.set_ylabel('$y$')
    a2.set_aspect('equal'); a2.legend(loc='lower right', fontsize=8.5)
    a2.set_xlim(-1.8, 1.8); a2.set_ylim(-1.8, 1.8); a2.grid(alpha=0.3)

    # 화살표(사상 f)
    fig.text(0.505, 0.5, r'$f$', fontsize=16, ha='center', va='center')
    fig.text(0.505, 0.44, '→', fontsize=22, ha='center', va='center')

    fig.suptitle('관절공간 → 작업공간 사상: 균일한 격자가 도달 고리로 왜곡된다 (본문 E3 · WE-2a)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('fig4_joint_to_task.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================
# fig5: 여유자유도 & 그리퍼 DoF 개념
# (a) 3R 팔: 같은 EEF 위치(t=2)를 이루는 관절해 1차원 족(redundancy = n-t = 1)
# (b) DoF 예산 막대: 태스크 차원 vs 관절 수 (UR5e/Franka/그리퍼)
# 본문 E3 · §5 표
# ============================================================
def fig5_redundancy_and_gripper():
    L = [0.6, 0.5, 0.4]
    target = np.array([0.8, 0.5])

    def fk(q):
        p, a, pts = np.zeros(2), 0.0, [np.zeros(2)]
        for li, qi in zip(L, q):
            a += qi
            p = p + li * np.array([np.cos(a), np.sin(a)])
            pts.append(p.copy())
        return np.array(pts)

    def ik(q):
        q = q.copy()
        for _ in range(400):
            e = target - fk(q)[-1]
            if np.linalg.norm(e) < 1e-7:
                return q
            Jc = np.zeros((2, 3)); eps = 1e-6
            f0 = fk(q)[-1]
            for j in range(3):
                dq = q.copy(); dq[j] += eps
                Jc[:, j] = (fk(dq)[-1] - f0) / eps
            q = q + Jc.T @ np.linalg.solve(Jc @ Jc.T + 1e-6 * np.eye(2), e)
        return q

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # ---- (a) 여유자유도: 여러 초기값에서 IK → 같은 EEF, 다른 팔 자세 ----
    sols = []
    for _ in range(9):
        q = ik(np.random.uniform(-np.pi, np.pi, 3))
        if q is not None and np.linalg.norm(target - fk(q)[-1]) < 1e-5:
            sols.append(q)
    cmap = plt.cm.cool
    for k, q in enumerate(sols):
        pts = fk(q)
        c = cmap(k / max(1, len(sols) - 1))
        a1.plot(pts[:, 0], pts[:, 1], '-o', color=c, lw=2, ms=4, alpha=0.85, zorder=2)
    a1.scatter(0, 0, s=120, color='k', zorder=5)
    a1.text(-0.02, -0.12, '베이스', ha='center', fontsize=9)
    a1.scatter(*target, marker='*', s=420, color=C_RED, zorder=6, edgecolor='k')
    a1.text(target[0] + 0.03, target[1] + 0.04, '공통 목표\n(EEF 위치)', color=C_RED, fontsize=10, fontweight='bold')
    a1.set_title('(a) 여유자유도 $n{-}t = 3{-}2 = 1$\n'
                 '같은 목표를 이루는 관절해의 1차원 족', fontsize=11)
    a1.set_aspect('equal'); a1.grid(alpha=0.3)
    a1.set_xlim(-0.4, 1.4); a1.set_ylim(-0.5, 1.2)
    a1.set_xlabel('$x$'); a1.set_ylabel('$y$')

    # ---- (b) DoF 예산 막대: 관절 수 vs 태스크 차원, 여유 = 차이 ----
    robots = ['UR5e', 'Franka\nFR3', '인간 팔', '2지\n그리퍼']
    n_joints = [6, 7, 7, 1]
    task_dim = [6, 6, 6, 1]   # 그리퍼는 개폐 1차원 태스크로 봄
    redund = [n - t for n, t in zip(n_joints, task_dim)]
    x = np.arange(len(robots))
    a2.bar(x, task_dim, width=0.55, color=C_BLUE, alpha=0.85, label='태스크 차원 $t$')
    a2.bar(x, redund, width=0.55, bottom=task_dim, color=C_GREEN, alpha=0.9,
           label='여유자유도 $n{-}t$')
    for xi, (n, t, r) in zip(x, zip(n_joints, task_dim, redund)):
        a2.text(xi, n + 0.15, f'n={n}', ha='center', fontsize=10, fontweight='bold')
        if r > 0:
            a2.text(xi, t + r / 2, f'+{r}', ha='center', color='white', fontsize=11, fontweight='bold')
    a2.set_xticks(x); a2.set_xticklabels(robots, fontsize=10)
    a2.set_ylabel('자유도')
    a2.set_ylim(0, 8.2); a2.grid(axis='y', alpha=0.3)
    a2.legend(loc='upper right', fontsize=9)
    a2.set_title('(b) DoF 예산: 관절 수 = 태스크 + 여유\n'
                 'Franka의 +1이 특이점·장애물 회피 예산', fontsize=11)

    fig.suptitle('여유자유도와 DoF 예산: 관절이 태스크보다 많으면 "쓸 수 있는 여분"이 생긴다 (본문 E3 · §5)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig('fig5_redundancy_and_gripper.png', dpi=140, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    fig2_serial_vs_parallel()
    fig3_grubler_breakdown()
    fig4_joint_to_task()
    fig5_redundancy_and_gripper()
    print('figures written: fig2, fig3, fig4, fig5')
