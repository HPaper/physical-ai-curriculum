# Lec 37 그림 생성 스크립트 — 모방학습이 무너지는 방식
# 실행: cd images/lec37 && python3 gen_figs.py   (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 정책/로봇/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

C_EXP = '#2c6fb0'    # 전문가 (파랑)
C_BC  = '#c0392b'    # BC (빨강)
C_DAG = '#27ae60'    # DAgger (초록)
C_T2  = '#8a7a4a'    # 이론 T^2 (황토)

# ============================================================================
# 공통 토이: 매 스텝 확률 eps로 '실수' → 전문가 미방문 상태로 이탈.
#   BC     : 이탈하면 회복 불가(rho=0)      → 개루프 드리프트가 지평선 끝까지 쌓임
#   DAgger : 이탈해도 확률 rho로 복귀        → 준폐루프
# ============================================================================
def rollout(T, eps, rho, seed, drift=1.0):
    """반환: 스텝별 추종 편차 x(t) 배열. off 상태에서 개루프 드리프트 누적."""
    r = np.random.default_rng(seed)
    x = 0.0; off = False; xs = []
    for t in range(T):
        if not off:
            if r.random() < eps: off = True
        else:
            if r.random() < rho: off = False
        x += drift if off else (-0.6*x)   # off: 개루프 드리프트 / on: 원점으로 복원
        xs.append(x)
    return np.array(xs)

# ---------------------------------------------------------------------------
# 그림 1: 전문가 vs BC 궤적 드리프트 (한 번의 롤아웃)
# ---------------------------------------------------------------------------
def fig1_drift():
    T = 120
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    # 전문가: 편차 0 유지 (원점 위)
    ax.plot(np.zeros(T), color=C_EXP, lw=2.6, label='전문가 π*  (편차 ≈ 0)')
    # BC 여러 롤아웃 (얇게) + 대표 하나 (굵게)
    for s in range(1, 9):
        xs = rollout(T, eps=0.02, rho=0.0, seed=40+s)
        ax.plot(xs, color=C_BC, lw=0.9, alpha=0.28)
    xs0 = rollout(T, eps=0.02, rho=0.0, seed=43)
    ax.plot(xs0, color=C_BC, lw=2.4, label='BC 정책  (한 번 이탈하면 회복 불가)')
    # 이탈 시점 표시
    on = np.where(np.abs(np.diff(xs0)) > 0.5)[0]
    if len(on):
        t0 = on[0]
        ax.axvline(t0, color='#555', ls=':', lw=1.2)
        ax.annotate('첫 실수 → 분포 이탈', xy=(t0, xs0[t0+3]), xytext=(t0+10, 42),
                    fontsize=10.5, color='#333',
                    arrowprops=dict(arrowstyle='->', color='#555'))
    ax.set_xlabel('시간 스텝  t'); ax.set_ylabel('전문가 궤적 대비 추종 편차  x(t)')
    ax.set_title('BC 정책의 개루프 드리프트 — 작은 실수가 회복 불가로 누적', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.25); ax.axhline(0, color='#999', lw=0.8)
    fig.tight_layout(); fig.savefig(OUT+'fig1_bc_drift.png', dpi=130); plt.close(fig)
    print('saved fig1_bc_drift.png')

# ---------------------------------------------------------------------------
# 그림 2: 오차 vs 지평선 T — BC(≈T^2) vs DAgger(≈T), 로그-로그
# ---------------------------------------------------------------------------
def cost(T, eps, rho, seed):
    r = np.random.default_rng(seed)
    off = False; c = 0
    for t in range(T):
        if not off:
            if r.random() < eps: off = True
        else:
            if r.random() < rho: off = False
        c += 1 if off else 0
    return c

def sweep(eps, rho, Ts, tag, N=6000):
    return np.array([np.mean([cost(T, eps, rho, seed=tag*1_000_003 + T*131 + k*99991)
                              for k in range(N)]) for T in Ts])

def fig2_scaling():
    eps = 0.0015
    Ts  = np.array([10, 20, 40, 80, 160, 320])
    J_bc  = sweep(eps, 0.0, Ts, tag=1)
    J_dag = sweep(eps, 0.5, Ts, tag=2)
    s_bc  = np.polyfit(np.log(Ts), np.log(J_bc), 1)[0]
    s_dag = np.polyfit(np.log(Ts), np.log(J_dag), 1)[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(Ts, J_bc, 'o-', color=C_BC, lw=2.2, ms=6,
              label=f'BC  (기울기 {s_bc:.2f} ≈ 2 → O(εT²))')
    ax.loglog(Ts, J_dag, 's-', color=C_DAG, lw=2.2, ms=6,
              label=f'DAgger  (기울기 {s_dag:.2f} ≈ 1 → O(εT))')
    # 기준선: T^2, T (정규화하여 표시)
    ref2 = J_bc[0]*(Ts/Ts[0])**2
    ref1 = J_dag[0]*(Ts/Ts[0])**1
    ax.loglog(Ts, ref2, '--', color=C_T2, lw=1.3, alpha=0.8, label='기울기 2 기준선 (∝T²)')
    ax.loglog(Ts, ref1, ':', color='#555', lw=1.3, alpha=0.8, label='기울기 1 기준선 (∝T)')
    ax.set_xlabel('지평선 길이  T  (로그)'); ax.set_ylabel('누적 비용 J  (로그)')
    ax.set_title('compounding error의 지평선 스케일링 — 왜 BC가 긴 태스크에서 무너지나', fontsize=11.5)
    ax.legend(loc='upper left', fontsize=9.5); ax.grid(alpha=0.25, which='both')
    fig.tight_layout(); fig.savefig(OUT+'fig2_error_vs_horizon.png', dpi=130); plt.close(fig)
    print(f'saved fig2_error_vs_horizon.png  (BC slope {s_bc:.3f}, DAgger slope {s_dag:.3f})')

# ---------------------------------------------------------------------------
# 그림 3: DAgger 데이터 집계 루프 (도식)
# ---------------------------------------------------------------------------
def fig3_dagger_loop():
    fig, ax = plt.subplots(figsize=(7.8, 4.4)); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    boxes = [
        (1.4, 4.6, '① 현재 정책 π(i)\n로봇에서 롤아웃', '#e1eefb', '#2c6fb0'),
        (5.0, 4.6, '② 방문 상태 {o~π(i)}\n수집  (on-policy)', '#e8f6ee', '#27ae60'),
        (8.3, 4.6, '③ 전문가 재라벨\na* = expert(o)', '#f3eede', '#8a7a4a'),
        (5.0, 1.4, '④ 데이터 집계\nD ← D ∪ {(o~π, a*)}', '#fbe9e7', '#c0392b'),
        (1.4, 1.4, '⑤ D로 재학습\n→ π(i+1)', '#eee6f5', '#7d5ba6'),
    ]
    cxy = {}
    for i, (x, y, t, fc, ec) in enumerate(boxes):
        b = FancyBboxPatch((x-1.0, y-0.55), 2.0, 1.1, boxstyle='round,pad=0.06',
                           fc=fc, ec=ec, lw=1.8)
        ax.add_patch(b); ax.text(x, y, t, ha='center', va='center', fontsize=9.6)
        cxy[i] = (x, y)
    def arrow(a, b, rad=0.0):
        (x0, y0), (x1, y1) = cxy[a], cxy[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                     mutation_scale=16, lw=1.6, color='#444',
                     shrinkA=42, shrinkB=42))
    arrow(0, 1); arrow(1, 2); arrow(2, 3, rad=-0.25); arrow(3, 4); arrow(4, 0, rad=-0.2)
    ax.text(5.0, 5.75, 'DAgger: 훈련 분포를 테스트(정책 방문) 분포에 맞추는 반복',
            ha='center', fontsize=12, weight='bold')
    ax.text(5.0, 0.25, '핵심은 데이터 "양"이 아니라 라벨을 붙이는 "상태 분포" — on-policy 상태에서의 전문가 라벨',
            ha='center', fontsize=9.5, color='#555', style='italic')
    fig.tight_layout(); fig.savefig(OUT+'fig3_dagger_loop.png', dpi=130); plt.close(fig)
    print('saved fig3_dagger_loop.png')

# ---------------------------------------------------------------------------
# 그림 4: covariate shift — 훈련(전문가) 분포 vs 배포(정책 방문) 분포 이동
# ---------------------------------------------------------------------------
def fig4_covariate_shift():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = np.linspace(-6, 12, 700)
    def gauss(mu, sd): return np.exp(-0.5*((xs-mu)/sd)**2)
    # 훈련 분포 d_{π*}: 전문가 방문 상태 — 좁고 원점 근처
    train = gauss(0.0, 0.8)
    # 배포 분포 d_{π̂}: 정책이 실제로 방문 — 이동 + 확산 (드리프트로 꼬리가 오른쪽으로)
    test = 0.55*gauss(0.3, 1.1) + 0.45*gauss(6.5, 2.6)
    ax.fill_between(xs, train, color=C_EXP, alpha=0.35, label='훈련 분포  $d_{\\pi^*}$  (전문가 방문 상태)')
    ax.plot(xs, train, color=C_EXP, lw=2)
    ax.fill_between(xs, test, color=C_BC, alpha=0.30, label='배포 분포  $d_{\\hat\\pi}$  (정책이 실제 방문)')
    ax.plot(xs, test, color=C_BC, lw=2)
    # 데이터 없는 영역 음영
    ax.axvspan(3.0, 12, color='#999', alpha=0.10)
    ax.text(9.3, 0.98, '전문가가 안 가본 영역\n= 훈련 데이터 없음\n→ 정책이 회복 못 함',
            fontsize=9.4, color='#555', ha='center')
    ax.annotate('', xy=(6.3, 0.30), xytext=(0.4, 0.30),
                arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.8))
    ax.text(3.3, 0.35, 'covariate shift\n(정책이 자기 상태분포를 만든다)',
            ha='center', fontsize=9.8, color='#333')
    ax.set_xlabel('상태 특징  z  (추종 편차의 축소판)'); ax.set_ylabel('방문 밀도 (정규화)')
    ax.set_title('covariate shift — iid 가정이 깨지는 자리: 정책이 자기 분포를 만든다', fontsize=11.5)
    ax.legend(loc='upper left', fontsize=9.3); ax.set_yticks([]); ax.set_ylim(0, 1.18)
    ax.grid(alpha=0.2, axis='x')
    fig.tight_layout(); fig.savefig(OUT+'fig4_covariate_shift.png', dpi=130); plt.close(fig)
    print('saved fig4_covariate_shift.png')

if __name__ == '__main__':
    fig1_drift()
    fig2_scaling()
    fig3_dagger_loop()
    fig4_covariate_shift()
    print('done.')
