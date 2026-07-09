# Lec 51 그림 생성 스크립트 — 시뮬레이터 지형도
# 실행: cd images/lec51 && python3 gen_figs.py
# 모든 실험은 결정적이다(시드 고정). 순수 numpy/matplotlib.
# numpy 1.26 / matplotlib 3.5 기준.
#
# 주의: 이 스크립트의 수치(처리량·학습시간·정확도)는 개념 재현용 CPU 토이다.
#       시뮬레이터의 실측 스펙(FPS·FLOPS)은 본문 [n] 인용을 따른다.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'figure.dpi': 120, 'savefig.dpi': 120})

OUT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# fig1: 시뮬레이터 지형도 (축: 물리충실도 x 병렬성/처리량)
#   위치는 정성적 배치(본문 비교표의 정성 평가) — 수치 주장 아님.
# ============================================================
def fig_landscape():
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    # (물리·접촉 충실도 x, 병렬 처리량 y) 정성 좌표 (0~10)
    sims = {
        'MuJoCo / robosuite':   (8.5, 2.0, 'tab:blue',   '단일 CPU 표준\n볼록 접촉·모델기반'),
        'MuJoCo MJX':           (8.0, 6.5, 'tab:cyan',   'JAX GPU 벡터화\n미분가능'),
        'Isaac Lab (PhysX)':    (6.5, 8.5, 'tab:green',  'GPU 대규모 병렬\nRTX 렌더 통합'),
        'ManiSkill3 (SAPIEN)':  (6.0, 9.3, 'tab:olive',  'state+렌더 GPU\n최고 처리량'),
        'Genesis':              (5.0, 9.6, 'tab:red',    '멀티솔버·Python\n초고속(회사 발표)'),
        'RoboCasa+MimicGen':    (7.5, 3.5, 'tab:purple', 'MuJoCo 위 씬·데이터\n주방 120·객체 2500+'),
    }
    for name,(x,y,c,note) in sims.items():
        ax.scatter([x],[y], s=380, c=c, edgecolors='k', linewidths=1.4, zorder=3, alpha=0.9)
        dy = 0.55 if name!='ManiSkill3 (SAPIEN)' else -0.95
        va = 'bottom' if dy>0 else 'top'
        ax.annotate(name, (x,y), (x, y+dy), ha='center', va=va,
                    fontsize=10.5, fontweight='bold', zorder=4)
        ax.annotate(note, (x,y), (x, y+dy+(0.62 if dy>0 else -0.62)), ha='center',
                    va=va, fontsize=7.6, color='#333', zorder=4)
    # 축·화살표
    ax.set_xlim(3.5, 10); ax.set_ylim(0.5, 11)
    ax.set_xlabel('접촉·물리 충실도  →  (모델기반·정밀 접촉)', fontsize=11)
    ax.set_ylabel('병렬 처리량  →  (수천 env·GPU)', fontsize=11)
    ax.set_title('그림 1. 로봇 학습용 시뮬레이터 지형도\n'
                 '(정성 배치 — 좌하: 정밀·단일 / 우상: 대량병렬. 절대 좌표는 무의미)',
                 fontsize=11.5)
    # 대각 트레이드오프 안내선
    ax.plot([4,9.5],[10.5,1.5], ls='--', c='gray', alpha=0.5, lw=1.2)
    ax.text(5.0, 1.6, '충실도↔처리량\n트레이드오프(경향)', fontsize=8.5, color='gray',
            rotation=-32, ha='left', va='bottom')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_landscape.png', bbox_inches='tight')
    plt.close(fig)
    print('fig1_landscape.png 저장')

# ============================================================
# fig2: 병렬 처리량 vs 학습 벽시계 시간 (WE-1)
# ============================================================
def fig_throughput():
    N = 1e9   # 필요 표본 (환경 스텝)
    # (이름, aggregate steps/s) — 토이 대표값. 실측 스펙은 본문 인용.
    rows = [
        ('단일 CPU\nMuJoCo (1 env)',   1_000,      'tab:blue'),
        ('MuJoCo MJX\n(수천 env)',     200_000,    'tab:cyan'),
        ('Isaac Lab\n(수천 env)',      100_000,    'tab:green'),
        ('ManiSkill3\n(state, GPU)',   1_000_000,  'tab:olive'),
    ]
    names  = [r[0] for r in rows]
    tps    = np.array([r[1] for r in rows], float)
    cols   = [r[2] for r in rows]
    hours  = N/tps/3600.0

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # (a) 처리량 막대 (log)
    y = np.arange(len(rows))
    axA.barh(y, tps, color=cols, edgecolor='k', alpha=0.9)
    axA.set_yticks(y); axA.set_yticklabels(names, fontsize=9)
    axA.set_xscale('log'); axA.set_xlabel('총 처리량 [환경 스텝/s]  (log)', fontsize=10)
    axA.set_title('(a) 처리량 = envs × steps/s', fontsize=11)
    for yi,tp in zip(y,tps):
        axA.text(tp*1.3, yi, f'{tp:,.0f}', va='center', fontsize=8.5)
    axA.set_xlim(5e2, 5e6); axA.grid(axis='x', alpha=0.3)

    # (b) 학습 벽시계 시간
    axB.barh(y, hours, color=cols, edgecolor='k', alpha=0.9)
    axB.set_yticks(y); axB.set_yticklabels([]); axB.invert_yaxis(); axA.invert_yaxis()
    axB.set_xscale('log'); axB.set_xlabel('학습 벽시계 시간 [시간]  (log)', fontsize=10)
    axB.set_title(f'(b) 벽시계 = 필요표본({N:.0e}) / 처리량', fontsize=11)
    for yi,hr in zip(y,hours):
        lab = f'{hr:.0f} h' if hr>=1 else f'{hr*60:.0f} min'
        if hr>=24: lab = f'{hr/24:.1f} 일'
        axB.text(hr*1.3, yi, lab, va='center', fontsize=8.5)
    axB.set_xlim(1e-1, 1e3); axB.grid(axis='x', alpha=0.3)
    axB.axhline(-0.5, color='none')

    fig.suptitle('그림 2. 병렬 처리량이 RL 학습 벽시계 시간을 가른다 (WE-1, 개념 토이)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_throughput.png', bbox_inches='tight')
    plt.close(fig)
    print('fig2_throughput.png 저장')
    return hours

# ============================================================
# fig3: sim2real 분포이동 + DR 회복 (WE-2, 지름길 학습)
# ============================================================
def true_label(x2):
    return (x2 > 0.5).astype(int)         # 참 규칙: 마찰(x2)만 결정

def sim_data(n, rng):
    x2 = rng.uniform(0.1, 0.9, n)
    y = true_label(x2)
    x1 = np.where(y==1, 0.75, 0.25) + rng.normal(0,0.04,n)   # 색이 라벨과 공변(지름길)
    return np.stack([x1,x2],1), y

def real_data(n, rng):
    x2 = rng.uniform(0.1, 0.9, n)
    x1 = rng.uniform(0.1, 0.9, n)          # 색 무작위 (공변 깨짐)
    return np.stack([x1,x2],1), true_label(x2)

def dr_data(n, rng):
    x2 = rng.uniform(0.1, 0.9, n)
    x1 = rng.uniform(0.1, 0.9, n)          # DR: 색 랜덤화
    return np.stack([x1,x2],1), true_label(x2)

def fit_logreg(X, y, lr=0.3, iters=6000):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1/(1+np.exp(-(X@w+b)))
        w -= lr*(X.T@(p-y)/len(y)); b -= lr*np.mean(p-y)
    return w, b

def acc(w,b,X,y):
    return np.mean(((1/(1+np.exp(-(X@w+b))))>0.5).astype(int)==y)

def fig_sim2real():
    rng = np.random.default_rng(0)
    Xs, ys = sim_data(4000, rng)
    Xr, yr = real_data(4000, rng)
    Xd, yd = dr_data(8000, rng)
    ws, bs = fit_logreg(Xs, ys)
    wd, bd = fit_logreg(Xd, yd)

    a_ss = acc(ws,bs,Xs,ys); a_sr = acc(ws,bs,Xr,yr)
    a_ds = acc(wd,bd,Xs,ys); a_dr = acc(wd,bd,Xr,yr)
    frac_s = abs(ws[0])/(abs(ws[0])+abs(ws[1]))
    frac_d = abs(wd[0])/(abs(wd[0])+abs(wd[1]))
    print(f'  [WE-2] 시뮬학습: sim={a_ss:.3f} real={a_sr:.3f} 색비중={frac_s:.3f}')
    print(f'  [WE-2] DR학습:  sim={a_ds:.3f} real={a_dr:.3f} 색비중={frac_d:.3f}')

    def draw(ax, X, y, w, b, title):
        ax.scatter(X[y==0,0], X[y==0,1], s=6, c='tab:red',  alpha=0.35, label='실패')
        ax.scatter(X[y==1,0], X[y==1,1], s=6, c='tab:blue', alpha=0.35, label='성공')
        # 결정경계 w·x+b=0
        xs = np.linspace(0.05,0.95,50)
        if abs(w[1])>1e-6:
            ys_ = -(w[0]*xs+b)/w[1]
            ax.plot(xs, ys_, 'k-', lw=2, label='학습 경계')
        ax.axhline(0.5, ls='--', c='green', lw=1.6, label='참 경계 (마찰=0.5)')
        ax.set_xlim(0,1); ax.set_ylim(0.05,0.95)
        ax.set_xlabel('x1 = 물체 색 (혼동 특징)', fontsize=9)
        ax.set_ylabel('x2 = 마찰 (참 원인)', fontsize=9)
        ax.set_title(title, fontsize=10)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    draw(axes[0], Xs, ys, ws, bs,
         f'(a) 시뮬 학습·시뮬 평가\n색이 라벨과 공변 → 지름길\nacc={a_ss:.2f}')
    draw(axes[1], Xr, yr, ws, bs,
         f'(b) 같은 정책·실제 분포\n공변 깨짐 → 경계 어긋남\nacc={a_sr:.2f} (갭!)')
    draw(axes[2], Xr, yr, wd, bd,
         f'(c) DR 학습·실제 분포\n색 랜덤화로 지름길 차단\nacc={a_dr:.2f} (회복)')
    axes[0].legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    fig.suptitle('그림 3. sim2real 갭 = 분포이동. 시뮬의 우연한 상관(색↔라벨)을 정책이 익스플로잇 →\n'
                 '실제에서 붕괴. 도메인 랜덤화가 그 상관을 깨 참 원인(마찰)만 남긴다 (WE-2)',
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_sim2real_dr.png', bbox_inches='tight')
    plt.close(fig)
    print('fig3_sim2real_dr.png 저장')
    return a_ss,a_sr,a_dr,frac_s,frac_d

# ============================================================
# fig4: 물리충실도-속도 트레이드오프 (52강 회수)
#   안정 최대 타임스텝 h* = 2/omega = 2 sqrt(m/k),
#   시뮬 비용 ∝ 1/h*  ->  강성↑ 이면 스텝↑ 이면 느려짐.
# ============================================================
def fig_fidelity_speed():
    k = np.logspace(2, 9, 400)        # 접촉 강성 [N/m], m=1
    m = 1.0
    hstar = 2*np.sqrt(m/k)            # 안정 최대 타임스텝 [s] (52강 E3)
    steps_per_sec = 1.0/hstar         # 물리 1초당 필요한 스텝 수

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    axA.loglog(k, hstar*1e3, 'tab:blue', lw=2.2)
    axA.set_xlabel('접촉 강성 k [N/m]  (충실도 →)', fontsize=10)
    axA.set_ylabel('안정 최대 타임스텝 h* [ms]', fontsize=10)
    axA.set_title('(a) h* = 2√(m/k) — 딱딱할수록 잘게', fontsize=11)
    axA.grid(True, which='both', alpha=0.3)
    # 대표점
    for kv, lab in [(1e4,'soft\n(MuJoCo류)'), (1e7,'금속 접촉'), (1e9,'강체 근사')]:
        hv = 2*np.sqrt(1/kv)*1e3
        axA.scatter([kv],[hv], s=70, c='tab:red', zorder=5)
        axA.annotate(lab, (kv,hv), (kv, hv*3), fontsize=8, ha='center')

    axB.loglog(k, steps_per_sec, 'tab:red', lw=2.2)
    axB.set_xlabel('접촉 강성 k [N/m]  (충실도 →)', fontsize=10)
    axB.set_ylabel('물리 1초당 스텝 수 (∝ 비용)', fontsize=10)
    axB.set_title('(b) 시뮬 비용 ∝ 1/h* = √(k/m)/2', fontsize=11)
    axB.grid(True, which='both', alpha=0.3)
    axB.annotate('강성 100배\n→ 비용 10배', (1e6, 1.0/(2*np.sqrt(1/1e6))),
                 (3e4, 3e3), fontsize=9, color='tab:red',
                 arrowprops=dict(arrowstyle='->', color='tab:red'))

    fig.suptitle('그림 4. 물리 충실도–속도 트레이드오프 (52강 E3 회수): '
                 '접촉을 딱딱하게 할수록 타임스텝이 줄고 비용이 는다.\n'
                 'soft 접촉은 k를 낮춰 h를 버는 거래 — 렌더 충실도(합성 이미지)는 이와 별개의 축이다.',
                 fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_fidelity_speed.png', bbox_inches='tight')
    plt.close(fig)
    print('fig4_fidelity_speed.png 저장')
    # 검증: 강성 100배 -> 비용 10배
    r = (1.0/(2*np.sqrt(1/1e6))) / (1.0/(2*np.sqrt(1/1e4)))
    print(f'  [fig4] k 100배(1e4->1e6) 시 비용비 = {r:.4f} (이론 10)')

if __name__ == '__main__':
    fig_landscape()
    hrs = fig_throughput()
    print(f'  [WE-1] 학습시간 h: 1env={hrs[0]:.1f} MJX={hrs[1]:.2f} Isaac={hrs[2]:.2f} MS3={hrs[3]:.3f}')
    print(f'  [WE-1] 1env {hrs[0]/24:.1f}일 -> ManiSkill3 {hrs[3]*60:.1f}분, 향상 {hrs[0]/hrs[3]:.0f}배')
    fig_sim2real()
    fig_fidelity_speed()
    print('완료.')
