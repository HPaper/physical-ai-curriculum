# Lec 66 그림 생성 스크립트 — World Model 기반 Physical AI
# 실행: python3 gen_figs.py  (이 디렉토리에서)
# 순수 numpy/matplotlib. 결정적(시드 고정). numpy 1.26 / scipy 1.15 / matplotlib 3.5.
#
# 이 토이는 "학습된 f_theta 위의 CEM 플래닝(=MPC)"과 "WM 안 평가의 함정"을
# CPU numpy로 재현한다. 실제 V-JEPA 2 / DreamZero / 1X WM 같은 대형 모델이 아니다.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 공통: 참 플랜트(관절 1축 이중적분기 + 토크 포화)와 학습된 WM (본문 WE-1과 동일)
# ============================================================
dt = 0.1
A = np.array([[1.0, dt], [0.0, 0.99]]); B = np.array([[0.0], [dt]])
a_max = 6.0

def f_true(s, a):
    return A @ s + B @ np.clip(np.atleast_1d(a), -a_max, a_max)

def learn(n, seed=1, noise=0.02):
    r = np.random.default_rng(seed)
    S = r.standard_normal((n, 2)); Ac = r.standard_normal((n, 1))
    Snx = np.stack([f_true(S[i], Ac[i]) for i in range(n)]) + noise*r.standard_normal((n, 2))
    th = np.linalg.lstsq(np.hstack([S, Ac]), Snx, rcond=None)[0].T
    return th[:, :2], th[:, 2:]

Ah, Bh = learn(200)
print("param err", round(float(np.linalg.norm(np.hstack([Ah, Bh]) - np.hstack([A, B]))), 5))

p_goal = 1.5

def energy(Am, Bm, s, acts):
    Ss = np.repeat(s[None, :], acts.shape[0], 0); J = np.zeros(acts.shape[0])
    for k in range(acts.shape[1]):
        Ss = Ss @ Am.T + acts[:, k:k+1] @ Bm.T
        J += (Ss[:, 0] - p_goal)**2
    return J + 0.001*(acts**2).sum(1)

def cem(Am, Bm, s, Hp, seed, mu0=None, bound=6.0):
    rng = np.random.default_rng(seed)
    mu = np.zeros(Hp) if mu0 is None else mu0.copy(); sd = (bound/3)*np.ones(Hp)
    for it in range(16):
        cand = np.clip(mu + sd*rng.standard_normal((256, Hp)), -bound, bound)
        cand[0] = np.clip(mu, -bound, bound)
        el = cand[np.argsort(energy(Am, Bm, s, cand))[:25]]
        mu, sd = el.mean(0), el.std(0) + (0.05*bound if it < 12 else 0.01)
    return mu

def mpc(Am, Bm, Hp=12, T=60, k_obs=1, seed0=100, bound=6.0):
    s_tr = np.array([0.0, 0.0]); s_bel = s_tr.copy(); plan = None; tr = [s_tr.copy()]; acts = []
    for t in range(T):
        if t % k_obs == 0: s_bel = s_tr.copy()
        mu0 = None if plan is None else np.r_[plan[1:], 0.0][:Hp]
        plan = cem(Am, Bm, s_bel, Hp, seed0 + t, mu0, bound)
        s_bel = Am @ s_bel + Bm @ np.atleast_1d(plan[0])
        s_tr = f_true(s_tr, plan[0])
        tr.append(s_tr.copy()); acts.append(plan[0])
    tr = np.array(tr)
    return float(np.mean(np.abs(tr[-15:, 0] - p_goal))), tr, np.array(acts)

INF = 10**9
print("== WE-1: 학습된 모델 위의 CEM 플래닝 ==")
st1, tr1, _ = mpc(A, B)
st2, tr2, _ = mpc(Ah, Bh)
st3, tr3, _ = mpc(Ah, Bh, k_obs=INF)
print(f"  참모델+폐루프 settle {st1:.4f} | 학습WM+폐루프 {st2:.4f} | 학습WM+개루프 {st3:.4f} "
      f"(폐 대비 {st3/st2:.1f}배)")
sweep = {}
for n in (20, 50, 200):
    An, Bn = learn(n)
    sweep[n] = (mpc(An, Bn)[0], mpc(An, Bn, k_obs=INF)[0])
    print(f"  n={n}: closed {sweep[n][0]:.4f} | open {sweep[n][1]:.4f}")

# ============================================================
# WE-2: WM 안 평가의 함정 (본문 WE-2와 동일)
# ============================================================
print("== WE-2: WM 안 평가 — 순위 역전 ==")
_, _, planA = mpc(Ah, Bh, T=40, k_obs=INF, seed0=200, bound=5.0)
_, _, planB = mpc(Ah, Bh, T=40, k_obs=INF, seed0=200, bound=25.0)

def J_traj(step, acts):
    s = np.array([0.0, 0.0]); J = 0.0; tr = [s.copy()]
    for a in acts:
        s = step(s, a); J += float((s[0] - p_goal)**2); tr.append(s.copy())
    return J, np.array(tr)

wm_step = lambda s, a: Ah @ s + Bh @ np.atleast_1d(a)
JwA, twA = J_traj(wm_step, planA); JwB, twB = J_traj(wm_step, planB)
JtA, ttA = J_traj(f_true, planA); JtB, ttB = J_traj(f_true, planB)
print(f"  WM 평가 J(A)={JwA:.2f} J(B)={JwB:.2f} -> {'B' if JwB < JwA else 'A'} 우수")
print(f"  참 평가 J(A)={JtA:.2f} J(B)={JtB:.2f} -> {'B' if JtB < JtA else 'A'} 우수")
print(f"  B 최종 위치: 상상 {twB[-1,0]:.2f} vs 실제 {ttB[-1,0]:.2f} | max|a| A {np.abs(planA).max():.1f} B {np.abs(planB).max():.1f}")

# ============================================================
# fig1 — 네 자리 지도: world model이 파이프라인에 꽂히는 곳
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2)

stages = [("실데이터 수집\n(teleop·영상, 55강)", 0.4), ("정책 학습\n(BC·RL, 37~41강)", 2.9),
          ("평가·선별\n(벤치마크, 57강)", 5.4), ("배포·실행 루프\n(관측→행동, 0강)", 7.9)]
for txt, x in stages:
    ax.add_patch(plt.Rectangle((x, 3.9), 1.9, 1.0, facecolor='#f0ede4', edgecolor='#8a7a4a', lw=1.5))
    ax.text(x + 0.95, 4.4, txt, ha='center', va='center', fontsize=9)
for x in (2.3, 4.8, 7.3):
    ax.annotate('', xy=(x + 0.6, 4.4), xytext=(x, 4.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

ax.add_patch(plt.Rectangle((3.4, 0.25), 3.1, 1.05, facecolor='#dff0e2', edgecolor='#3a9a5a', lw=2))
ax.text(4.95, 0.78, "학습된 world model $f_\\theta$\n$z_{t+1}=f_\\theta(z_t,a_t)$ (54강)",
        ha='center', va='center', fontsize=10, color='#14512c')

plugs = [
    ("① 데이터 엔진", "DreamGen: 상상 궤적 합성\nunseen 28.5% (베이스라인 0%)", 1.35, '#c0392b', (4.0, 1.3)),
    ("② 평가자", "1X WM·WorldEval\nr=0.94 (real-to-sim 0.41)", 6.35, '#2c6fb0', (4.6, 1.3)),
    ("③ 플래너의 내부 모델", "V-JEPA 2-AC: CEM 플래닝\n= 23강 MPC의 학습판", 8.85, '#8a5cb0', (5.6, 1.3)),
    ("④ 정책 그 자체", "DreamZero(WAM): 비디오+행동\n공동 디노이징, 7Hz", 8.85, '#b0741f', (6.2, 1.3)),
]
targets = [(1.35, 3.9), (6.35, 3.9), (8.5, 3.9), (9.2, 3.9)]
for (title, body, xl, col, src), (tx, ty) in zip(plugs, targets):
    ax.annotate('', xy=(tx, ty), xytext=src,
                arrowprops=dict(arrowstyle='->', color=col, lw=2,
                                connectionstyle="arc3,rad=-0.15"))
y_lab = {0: 2.65, 1: 2.65, 2: 3.05, 3: 1.75}
x_lab = {0: 1.3, 1: 5.1, 2: 7.05, 3: 8.15}
for i, (title, body, xl, col, src) in enumerate(plugs):
    ax.text(x_lab[i], y_lab[i], f"{title}\n{body}", ha='center', va='center', fontsize=8.3,
            color=col, bbox=dict(facecolor='white', edgecolor=col, alpha=0.9, lw=1.2))
ax.annotate("대체의 깊이: 얕음(오프라인 데이터) → 깊음(실시간 행동)", xy=(5.0, 0.05),
            ha='center', fontsize=9, color='#555')
ax.set_title("world model이 로봇 학습 파이프라인에 꽂히는 네 자리 (①~④: 본문 §1)")
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_four_sockets.png', dpi=140); plt.close(fig)

# ============================================================
# fig2 — 스펙트럼(위) + 타임라인(아래)
# ============================================================
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.5, 7.2),
                               gridspec_kw={'height_ratios': [1, 2.4]})
ax0.axis('off'); ax0.set_xlim(0, 10); ax0.set_ylim(0, 2)
spec = [("수동 영상 예측", "행동 입력 없음\nSora류 비디오 생성", '#999999'),
        ("행동조건부 예측", "a를 주면 결과 예측\nCosmos·V-JEPA 2 (54강)", '#3a9a5a'),
        ("상호작용형 환경", "실시간으로 스텝\nGenie 1/2/3", '#2c6fb0'),
        ("WAM: 행동도 출력", "예측이 곧 정책\nDreamZero·GR00T N2", '#b0741f')]
for i, (t, b, c) in enumerate(spec):
    x = 0.3 + i*2.5
    ax0.add_patch(plt.Rectangle((x, 0.45), 2.1, 1.25, facecolor=c, alpha=0.15, edgecolor=c, lw=2))
    ax0.text(x + 1.05, 1.33, t, ha='center', fontsize=9.5, fontweight='bold', color=c)
    ax0.text(x + 1.05, 0.78, b, ha='center', fontsize=8, color='#333')
    if i < 3:
        ax0.annotate('', xy=(x + 2.5, 1.07), xytext=(x + 2.1, 1.07),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=2))
ax0.set_title("(a) 스펙트럼 — 예측이 정책이 되기까지: 조건(행동 입력)과 출력(행동)의 확장")

LY = {3: 4.2, 2: 2.8, 1: 1.4, 0: 0.0}          # 레인 y 좌표
lanes = {3: ("상상 RL 계보", '#c0392b'), 2: ("영상 생성 계보", '#2c6fb0'),
         1: ("표현 예측 계보", '#3a9a5a'), 0: ("로봇 파이프라인 배치", '#b0741f')}
events = [  # (year_frac, lane, label, above?, label_dx)
    (2018 + 2/12, 3, "World Models\n2018.3", 1, 0.35), (2018 + 10/12, 3, "PlaNet\n2018.11", 0, 0),
    (2019 + 11/12, 3, "Dreamer v1\n2019.12", 1, 0), (2023 + 0/12, 3, "DreamerV3\n2023.1", 0, 0),
    (2025 + 8/12, 3, "Dreamer 4\n2025.9", 1, 0),
    (2023 + 1/12, 2, "UniPi\n2023.2", 1, 0), (2023 + 9/12, 2, "UniSim\n2023.10", 0, 0),
    (2024 + 1/12, 2, "Genie\n2024.2", 1, 0), (2024 + 11/12, 2, "Genie 2\n2024.12", 0, -0.2),
    (2025 + 0/12, 2, "Cosmos\n2025.1", 1, 0.15), (2025 + 7/12, 2, "Genie 3\n2025.8", 0, 0),
    (2026 + 5/12, 2, "Cosmos 3\n2026.6", 1, 0),
    (2022 + 5/12, 1, "JEPA 제안\n2022.6", 1, 0), (2025 + 5/12, 1, "V-JEPA 2(-AC)\n2025.6", 0, -0.15),
    (2026 + 2/12, 1, "AMI Labs\n2026.3", 1, 0),
    (2025 + 4/12, 0, "DreamGen\n2025.5", 1, -0.25), (2025 + 5/12, 0, "1X WM\n2025.6", 0, -0.1),
    (2025 + 9/12, 0, "Ctrl-World\n2025.10", 1, 0.3), (2026 + 1/12, 0, "DreamZero\n2026.2", 0, 0.15),
    (2026 + 2/12, 0, "GR00T N2\n프리뷰 2026.3", 1, 0.45),
]
for lane, (name, col) in lanes.items():
    ax1.axhline(LY[lane], color=col, lw=1, alpha=0.35)
    ax1.text(2017.72, LY[lane] + 0.1, name, fontsize=9, color=col, fontweight='bold')
for yr, lane, lab, above, dx in events:
    col = lanes[lane][1]
    ax1.plot(yr, LY[lane], 'o', ms=7, color=col)
    ax1.annotate(lab, xy=(yr, LY[lane]), xytext=(yr + dx, LY[lane] + (0.18 if above else -0.18)),
                 ha='center', va='bottom' if above else 'top', fontsize=7.6, color=col)
ax1.set_xlim(2017.6, 2027.1); ax1.set_ylim(-0.95, 5.05)
ax1.set_yticks([]); ax1.set_xticks(range(2018, 2027))
ax1.set_xlabel("연도")
ax1.set_title("(b) 타임라인 — 세 계보가 2025~26에 로봇 파이프라인의 네 자리로 흘러든다")
ax1.grid(alpha=0.2, axis='x')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_spectrum_timeline.png', dpi=140); plt.close(fig)

# ============================================================
# fig3 — WE-1: 학습된 모델 위의 CEM 플래닝 (폐루프 vs 상상 개루프)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
tt = np.arange(len(tr1))
ax[0].axhline(p_goal, color='k', ls=':', lw=1, label=f'목표 위치 {p_goal}')
ax[0].plot(tt, tr1[:, 0], color='tab:green', lw=2.2, label='참 모델 + 폐루프(재관측)')
ax[0].plot(tt, tr2[:, 0], color='tab:blue', lw=1.8, ls='--', label='학습 WM + 폐루프')
ax[0].plot(tt, tr3[:, 0], color='tab:red', lw=1.8, ls='-.', label='학습 WM + 개루프(순수 상상)')
ax[0].set_xlabel('스텝 t'); ax[0].set_ylabel('위치 p')
ax[0].set_title('(a) CEM-MPC 목표 도달 — 재관측이 모델 오차를 먹는다')
ax[0].legend(fontsize=8, loc='lower right'); ax[0].grid(alpha=0.3)

ns = list(sweep.keys()); x = np.arange(len(ns)); w = 0.36
ax[1].bar(x - w/2, [sweep[n][0] for n in ns], w, color='tab:blue', label='폐루프 (매 스텝 재관측)')
ax[1].bar(x + w/2, [sweep[n][1] for n in ns], w, color='tab:red', label='개루프 (상상만)')
ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels([f'n={n}' for n in ns])
for i, n in enumerate(ns):
    ax[1].text(i - w/2, sweep[n][0]*1.2, f'{sweep[n][0]:.3f}', ha='center', fontsize=8)
    ax[1].text(i + w/2, sweep[n][1]*1.2, f'{sweep[n][1]:.3f}', ha='center', fontsize=8)
ax[1].set_ylabel('settle 오차 |p−목표| (로그)')
ax[1].set_title('(b) 모델 품질(학습 데이터 n) vs 폐/개루프')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y', which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_wm_mpc.png', dpi=140); plt.close(fig)

# ============================================================
# fig4 — WE-2: WM 안 평가의 함정 (순위 역전)
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
tt2 = np.arange(len(twA))
ax[0].axhline(p_goal, color='k', ls=':', lw=1)
ax[0].plot(tt2, ttA[:, 0], color='tab:blue', lw=2, label='정책 A — 실제 실행')
ax[0].plot(tt2, twB[:, 0], color='tab:orange', lw=2, ls='--', label='정책 B — WM 상상 (환각)')
ax[0].plot(tt2, ttB[:, 0], color='tab:red', lw=2, label='정책 B — 실제 실행')
ax[0].set_xlabel('스텝 t'); ax[0].set_ylabel('위치 p')
ax[0].set_title('(a) 정책 B: 상상은 목표 도달, 실제는 반대로 폭주')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

xb = np.arange(2); w = 0.36
ax[1].bar(xb - w/2, [JwA, JwB], w, color='tab:green', label='WM 안 평가 (상상)')
ax[1].bar(xb + w/2, [JtA, JtB], w, color='tab:red', label='참 환경 평가')
for i, v in enumerate([JwA, JwB]): ax[1].text(i - w/2, v*1.25, f'{v:.1f}', ha='center', fontsize=9)
for i, v in enumerate([JtA, JtB]): ax[1].text(i + w/2, v*1.25, f'{v:.1f}', ha='center', fontsize=9)
ax[1].set_yscale('log'); ax[1].set_ylim(top=3000)
ax[1].set_xticks(xb); ax[1].set_xticklabels(['정책 A\n(분포 안)', '정책 B\n(WM 익스플로잇)'])
ax[1].set_ylabel('누적 비용 J (로그, 낮을수록 좋음)')
ax[1].set_title(f'(b) 순위 역전 — WM: B 우수({JwB:.1f}<{JwA:.1f}), 참: A 우수({JtA:.1f}<{JtB:.1f})')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y', which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_eval_trap.png', dpi=140); plt.close(fig)

print("figs saved:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
