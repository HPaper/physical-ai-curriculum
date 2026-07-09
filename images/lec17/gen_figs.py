"""Lec R17 그림 생성 스크립트.
실행: python3 gen_figs.py  (images/lec17/ 안에서)
본문 수치와 동일한 시뮬레이션 사용 — 수치 재현성 각주 참조.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy import signal

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = "."

C = {'p': '#1f77b4', 'pd': '#2ca02c', 'pid': '#d62728', 'k1': '#1f77b4',
     'k4': '#ff7f0e', 'k16': '#d62728', 'gray': '#888888'}


# ---------- 공통: 플랜트 q'' + q' = u + d, P 제어(지연 포함) 시뮬 ----------
def sim_delay_P(K, tau, T=40.0, dt=1e-3, d=0.0):
    n = int(T / dt)
    q = qd = 0.0
    ds = max(int(round(tau / dt)), 0)
    buf = np.zeros(ds + 1)
    qs = np.zeros(n)
    for i in range(n):
        e = 1.0 - q
        buf = np.roll(buf, 1); buf[0] = e
        u = K * buf[-1] if i * dt >= tau else 0.0
        qdd = u - qd + d
        qd += qdd * dt; q += qd * dt
        qs[i] = q
    return np.arange(n) * dt, qs


def sim_pid(Kp, Ki, Kd, T=20.0, dt=1e-4, d=0.0):
    n = int(T / dt)
    q = qd = ei = 0.0
    qs = np.zeros(n)
    for i in range(n):
        e = 1.0 - q
        ei += e * dt
        u = Kp * e + Ki * ei - Kd * qd     # r 상수 → ė = -q̇
        qd += (u - qd + d) * dt; q += qd * dt
        qs[i] = q
    return np.arange(n) * dt, qs


# ---------- fig1: 한 장 요약 — 같은 게인, 지연만 추가하면 ----------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for K, c in [(1, C['k1']), (4, C['k4']), (16, C['k16'])]:
        t, q = sim_delay_P(K, 0.0, T=15.0)
        axes[0].plot(t, q, color=c, lw=1.6, label=f"$K_p={K}$")
    axes[0].axhline(1, color=C['gray'], ls=':', lw=1)
    axes[0].set(title="(a) 지연 없음 — 게인을 올리면 빨라진다\n(진동은 늘지만 어떤 게인에도 안정)",
                xlabel="시간 [s]", ylabel="출력 $q$", ylim=(0, 1.9))
    axes[0].legend(loc='lower right')

    for K, c in [(1, C['k1']), (4, C['k4']), (16, C['k16'])]:
        t, q = sim_delay_P(K, 0.1, T=15.0)
        axes[1].plot(t, np.clip(q, -3, 5), color=c, lw=1.6, label=f"$K_p={K}$")
    axes[1].axhline(1, color=C['gray'], ls=':', lw=1)
    axes[1].set(title="(b) 지연 $\\tau=0.1$ s — 임계 게인 $K_u=10.2$ 초과 시 발산\n(같은 플랜트, 같은 게인)",
                xlabel="시간 [s]", ylim=(-3, 5))
    axes[1].annotate("$K_p=16 > K_u$ → 발산", xy=(0.4, 4.3), fontsize=10, color=C['k16'])
    axes[1].legend(loc='lower right')
    fig.suptitle("피드백의 근본 트레이드오프: 게인 = 속도, 지연 = 게인의 한계", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_gain_delay_tradeoff.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------- fig2: 극점 위치 ↔ 응답 모양 ----------
def fig2():
    cases = [  # (label, zeta, wn, color)
        ("A", 1.0, 1.0, '#1f77b4'),
        ("B", 0.5, 1.0, '#2ca02c'),
        ("C", 0.15, 1.0, '#ff7f0e'),
        ("D", 0.5, 3.0, '#9467bd'),
        ("E", -0.08, 1.0, '#d62728'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.axvline(0, color='k', lw=0.8); ax.axhline(0, color='k', lw=0.8)
    ax.axvspan(0, 1.2, color='#d62728', alpha=0.06)
    ax.text(0.45, 3.2, "불안정\n(RHP)", color='#d62728', fontsize=9, ha='center')
    for lbl, z, wn, c in cases:
        re, im = -z * wn, wn * np.sqrt(abs(1 - z**2))
        ax.plot([re, re], [im, -im], 'x', color=c, ms=10, mew=2.4)
        ax.annotate(lbl, (re, im), textcoords="offset points", xytext=(6, 5),
                    color=c, fontsize=12, fontweight='bold')
    ax.set(title="(a) $s$-평면의 극점 위치", xlabel="Re($s$)  ← 감쇠 빠름",
           ylabel="Im($s$) = 진동 주파수", xlim=(-3.6, 1.2), ylim=(-3.6, 3.6))
    ax.grid(alpha=0.3)

    ax = axes[1]
    t = np.linspace(0, 12, 800)
    for lbl, z, wn, c in cases:
        sys = signal.TransferFunction([wn**2], [1, 2 * z * wn, wn**2])
        _, y = signal.step(sys, T=t)
        ax.plot(t, np.clip(y, -0.5, 2.6), color=c, lw=1.6,
                label=f"{lbl}: $\\zeta$={z}, $\\omega_n$={wn}")
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.set(title="(b) 대응하는 스텝 응답", xlabel="시간 [s]", ylim=(-0.5, 2.6))
    ax.legend(fontsize=9, loc='upper right')
    fig.suptitle("극점을 보면 응답이 보인다: 반경 $\\omega_n$ = 빠르기, 각도 $\\cos^{-1}\\zeta$ = 출렁임", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_pole_response_map.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------- fig3: WE-1 — P → PD → PID ----------
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    t, q = sim_pid(1, 0, 0)
    ax.plot(t, q, color=C['p'], lw=1.6, label="P: $K_p{=}1$ (극점 $-0.5\\pm0.866j$)")
    t, q = sim_pid(4, 0, 3)
    ax.plot(t, q, color=C['pd'], lw=1.6, label="PD: $K_p{=}4, K_d{=}3$ (극점 $-2,-2$)")
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.annotate("오버슛 16.3%", xy=(3.63, 1.163), xytext=(6, 1.35),
                arrowprops=dict(arrowstyle='->', color=C['p']), color=C['p'], fontsize=9)
    ax.set(title="(a) 외란 없음: D가 감쇠를 산다", xlabel="시간 [s]", ylabel="$q$", ylim=(0, 1.5))
    ax.legend(fontsize=9, loc='lower right')

    ax = axes[1]
    t, q = sim_pid(4, 0, 3, d=-2.0)
    ax.plot(t, q, color=C['pd'], lw=1.6, label="PD: 정상오차 $-d/K_p=0.5$")
    t, q = sim_pid(4, 2, 3, d=-2.0)
    ax.plot(t, q, color=C['pid'], lw=1.6, label="PID($K_i{=}2$): 오차 0")
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.axhline(0.5, color=C['pd'], ls='--', lw=0.9, alpha=0.6)
    ax.set(title="(b) 정상 외란 $d=-2$ (중력 같은 것): I가 오프셋을 지운다",
           xlabel="시간 [s]", ylim=(0, 1.5))
    ax.legend(fontsize=9, loc='lower right')
    fig.suptitle("P → PD → PID: 각 항이 사는 것 (플랜트 $\\ddot q + \\dot q = u + d$)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_p_pd_pid.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------- fig4: 지연이 만드는 안정 게인 경계 ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    taus = np.logspace(np.log10(0.01), np.log10(2.0), 200)
    Kus = []
    for tau in taus:
        w = brentq(lambda w: np.arctan(w) + tau * w - np.pi / 2, 1e-3, 1e4)
        Kus.append(w * np.sqrt(w**2 + 1))
    Kus = np.array(Kus)
    ax = axes[0]
    ax.loglog(taus, Kus, color='k', lw=2)
    ax.fill_between(taus, 1e-2, Kus, color='#2ca02c', alpha=0.15)
    ax.text(0.02, 0.35, "안정", color='#2ca02c', fontsize=12)
    ax.text(0.35, 40, "불안정", color='#d62728', fontsize=12)
    ax.plot([0.1], [10.16], 'o', color='#d62728')
    ax.annotate("WE-2: $\\tau{=}0.1$,\n$K_u{=}10.16$", xy=(0.1, 10.16), xytext=(0.022, 3.2),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    ax.plot([1.151], [1.0], 's', color='#1f77b4')
    ax.annotate("E3: $K_p{=}1$,\n$\\tau_{max}{=}1.15$ s", xy=(1.151, 1.0), xytext=(0.35, 0.12),
                arrowprops=dict(arrowstyle='->'), fontsize=9)
    ax.set(title="(a) 임계 게인 $K_u(\\tau)$ — 지연이 클수록 쓸 수 있는 게인이 준다",
           xlabel="루프 지연 $\\tau$ [s]", ylabel="임계 게인 $K_u$", ylim=(5e-2, 3e2))
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    for K, c, lbl in [(8.0, '#2ca02c', "$K=8$: 감쇠"), (10.16, '#ff7f0e', "$K=K_u=10.16$: 지속 진동"),
                      (12.0, '#d62728', "$K=12$: 발산")]:
        t, q = sim_delay_P(K, 0.1, T=30.0)
        ax.plot(t, np.clip(q, -6, 8), color=c, lw=1.3, label=lbl)
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.annotate("측정 주기 $T_u = 2.02$ s\n(이론 $2\\pi/\\omega_{180}$과 일치)", xy=(0.8, 6.3), fontsize=9,
                color='#ff7f0e')
    ax.set(title="(b) $\\tau=0.1$ s에서 게인 3종의 운명", xlabel="시간 [s]", ylim=(-6, 8))
    ax.legend(fontsize=9, loc='lower left')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_delay_stability_boundary.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------- fig5: 이산 구현 — 100Hz vs 1kHz ----------
def sim_discrete_pd(Kp, Kd, Ts, extra=0, T=1.6, dt=1e-5):
    n = int(T / dt)
    q = qd = 0.0
    u, e_prev = 0.0, None
    uq = [0.0] * (extra + 1)
    sp = int(round(Ts / dt))
    qs = np.zeros(n)
    for i in range(n):
        if i % sp == 0:
            e = 1.0 - q
            de = 0.0 if e_prev is None else (e - e_prev) / Ts
            e_prev = e
            uq.append(Kp * e + Kd * de); u = uq.pop(0)
        qd += (u - qd) * dt; q += qd * dt
        qs[i] = q
    return np.arange(n) * dt, qs


def fig5():
    Kp, Kd = 900.0, 59.0
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    t, q = sim_discrete_pd(Kp, Kd, 1e-3)
    ax.plot(t, q, color='#2ca02c', lw=1.7,
            label="1 kHz: $\\tau_{eff}{=}1$ ms → $\\varphi_m$ 73° (연속 설계 그대로)")
    t, q = sim_discrete_pd(Kp, Kd, 1e-2)
    ax.plot(t, q, color='#ff7f0e', lw=1.7,
            label="100 Hz: $\\tau_{eff}{=}10$ ms → $\\varphi_m$ 42° (오버슛 10%, 링잉)")
    t, q = sim_discrete_pd(Kp, Kd, 1e-2, extra=1)
    ax.plot(t, np.clip(q, -1.5, 3.2), color='#d62728', lw=1.7,
            label="100 Hz + 계산지연 1주기: $\\tau_{eff}{=}20$ ms ≈ 예산 22 ms → 발산")
    ax.axhline(1, color=C['gray'], ls=':', lw=1)
    ax.set(title="같은 PD 게인($K_p{=}900, K_d{=}59$, 연속 설계 극점 $-30,-30$), 다른 제어 주기\n"
                 "이산화 = 지연: $\\tau_{eff} \\approx T/2$(ZOH) $+\\, T/2$(D 차분) $+$ 계산지연",
           xlabel="시간 [s]", ylabel="$q$", xlim=(0, 1.6), ylim=(-1.5, 3.2))
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_sampling_rate.png", dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figs done")
