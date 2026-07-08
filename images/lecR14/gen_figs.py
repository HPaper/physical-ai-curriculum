"""Lec R14 그림 생성 스크립트.
fig1: 토크-속도 곡선 — 전압 한계·전류 한계·열 한계가 만드는 3개 영역 + 출력/효율
fig2: 전류 스텝 응답 — 개루프 RL vs 폐루프 PI, 그리고 샘플링 주파수의 효과
fig3: 열 모델 시뮬 — 연속/피크 전류의 온도 궤적과 듀티 사이클
fig4: FOC dq 변환 — 3상 AC가 회전 좌표계에서 DC가 된다
모든 수치는 본문 Worked Example과 동일 파라미터:
Kt = Ke = 0.1, R = 0.5, L = 0.3e-3, V = 24, i_cont = 10, i_peak = 30,
R_th = 2.0, C_th = 60, T_amb = 25, T_max = 125.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lecR14"

# ---------------- 모터 파라미터 (가상 카탈로그) ----------------
Kt = 0.1        # Nm/A
Ke = 0.1        # V·s/rad (= Kt, SI)
R = 0.5         # Ohm
L = 0.3e-3      # H
V = 24.0        # V (버스 전압)
i_cont = 10.0   # A (연속, 열 한계에서 유도)
i_peak = 30.0   # A (피크, 컨트롤러 제한)
Rth = 2.0       # K/W
Cth = 60.0      # J/K
T_amb, T_max = 25.0, 125.0

w_nl = V / Ke                 # 무부하 속도
i_stall = V / R               # 스톨 전류
tau_stall = Kt * i_stall      # 스톨 토크
tau_cont = Kt * i_cont
tau_peak = Kt * i_peak

print(f"무부하 속도 w_nl = {w_nl:.1f} rad/s = {w_nl*60/2/np.pi:.0f} rpm")
print(f"스톨 전류 = {i_stall:.1f} A, 스톨 토크 = {tau_stall:.2f} Nm")
print(f"연속 토크 = {tau_cont:.2f} Nm, 피크 토크 = {tau_peak:.2f} Nm")

# ============================================================
# fig 1 — 토크-속도 곡선
# ============================================================
w = np.linspace(0, w_nl, 500)
tau_v = Kt * (V - Ke * w) / R          # 전압 한계선

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax1.plot(w, tau_v, 'k-', lw=2.2, label=r'전압 한계 $\tau=K_t(V-K_e\omega)/R$')
ax1.fill_between(w, 0, np.minimum(tau_cont, tau_v), color='mediumseagreen',
                 alpha=0.45, label='연속 작동 영역 (열 OK)')
ax1.fill_between(w, np.minimum(tau_cont, tau_v), np.minimum(tau_peak, tau_v),
                 where=tau_v > tau_cont, color='orange', alpha=0.40,
                 label='단속 영역 (열 시계 작동, ~수 초)')
ax1.fill_between(w, np.minimum(tau_peak, tau_v), tau_v,
                 where=tau_v > tau_peak, color='0.75', alpha=0.55,
                 hatch='//', label='전류 제한(30 A)으로 차단')
ax1.axhline(tau_cont, color='seagreen', ls='--', lw=1)
ax1.axhline(tau_peak, color='darkorange', ls='--', lw=1)
w_corner = (tau_stall - tau_cont) / (Kt * Ke / R)
print(f"연속 토크-전압 한계 교점(코너 속도) = {w_corner:.1f} rad/s")
ax1.annotate(f'무부하 속도\n{w_nl:.0f} rad/s', (w_nl, 0), xytext=(-58, 40),
             textcoords='offset points', fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.annotate(f'스톨 {tau_stall:.1f} Nm\n(전류 {i_stall:.0f} A — 열로는 순간도 위험)',
             (0, tau_stall), xytext=(8, -14), textcoords='offset points', fontsize=9)
ax1.annotate(f'코너 {w_corner:.0f} rad/s', (w_corner, tau_cont),
             xytext=(-75, 12), textcoords='offset points', fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.set_xlabel(r'속도 $\omega$ (rad/s)'); ax1.set_ylabel(r'토크 $\tau$ (Nm)')
ax1.set_title('(a) 토크-속도 평면: 세 가지 한계가 만드는 작동 영역')
ax1.legend(fontsize=8, loc='upper right'); ax1.set_xlim(0, 260); ax1.set_ylim(0, 5.2)

# (b) 전압 한계선 위에서의 기계 출력과 효율
P_mech = tau_v * w
eta = w / w_nl                          # 전압 한계선 위: eta = Ke*w/V (구리손만 고려)
ax2.plot(w, P_mech, 'b-', lw=2, label=r'기계 출력 $P=\tau\omega$ (W)')
i_of_w = (V - Ke * w) / R
ax2.plot(w, i_of_w**2 * R, 'r--', lw=1.6, label=r'구리손 $i^2R$ (W)')
w_pmax = w[np.argmax(P_mech)]
print(f"최대 기계 출력 = {P_mech.max():.1f} W at {w_pmax:.1f} rad/s")
ax2.annotate(f'최대 {P_mech.max():.0f} W\n@ {w_pmax:.0f} rad/s',
             (w_pmax, P_mech.max()), xytext=(10, -5),
             textcoords='offset points', fontsize=9)
ax2b = ax2.twinx()
ax2b.plot(w, eta, 'g-', lw=1.6)
ax2b.set_ylabel(r'효율 $\eta=\omega/\omega_{nl}$ (구리손만)', color='g')
ax2b.tick_params(axis='y', colors='g'); ax2b.set_ylim(0, 1.05)
ax2.set_xlabel(r'속도 $\omega$ (rad/s)'); ax2.set_ylabel('전력 (W)')
ax2.set_title('(b) 전압 한계선 위의 출력·손실·효율')
ax2.legend(fontsize=8, loc='center right'); ax2.set_xlim(0, 260)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_torque_speed.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 2 — 전류 스텝 응답
# ============================================================
tau_e = L / R
f_open = 1 / (2 * np.pi * tau_e)
print(f"\n전기 시정수 tau_e = {tau_e*1e3:.2f} ms, 개루프 대역폭 = {f_open:.1f} Hz")

w_bw = 2 * np.pi * 1000.0              # 목표 폐루프 대역폭 1 kHz
Kp, Ki = L * w_bw, R * w_bw            # 극영점 상쇄 튜닝
print(f"PI 게인: Kp = {Kp:.3f} V/A, Ki = {Ki:.1f} V/(A s)")

t = np.linspace(0, 3e-3, 3000)
i_ref = 10.0
i_open = i_ref * (1 - np.exp(-t / tau_e))            # V=5V 개루프 스텝
i_closed = i_ref * (1 - np.exp(-w_bw * t))           # 극영점 상쇄 → 1차

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax1.plot(t * 1e3, i_open, 'b-', lw=2,
         label=f'개루프 (V=5 V 스텝): $\\tau_e$={tau_e*1e3:.1f} ms')
ax1.plot(t * 1e3, i_closed, 'r-', lw=2,
         label='폐루프 PI (대역폭 1 kHz): $\\tau$=0.16 ms')
ax1.axhline(0.632 * i_ref, color='0.6', ls=':', lw=1)
ax1.annotate('63.2%', (2.45, 0.632 * i_ref), fontsize=8, va='bottom')
ax1.axvline(tau_e * 1e3, color='b', ls=':', lw=1)
ax1.axvline(1 / w_bw * 1e3, color='r', ls=':', lw=1)
ax1.set_xlabel('시간 (ms)'); ax1.set_ylabel('전류 (A)')
ax1.set_title('(a) 10 A 전류 스텝: 개루프 vs 폐루프')
ax1.legend(fontsize=9); ax1.set_xlim(0, 3); ax1.set_ylim(0, 12)

# (b) 같은 PI를 서로 다른 샘플링 주파수의 디지털 루프로
#     실제 컨트롤러처럼 "k에서 계산한 전압은 k+1에서 인가" (1샘플 연산·PWM 지연)
def digital_loop(fs, t_end=3e-3):
    Ts = 1 / fs
    n = int(t_end / Ts)
    a = np.exp(-Ts / tau_e)             # ZOH 정확 이산화
    i, acc, v_apply = 0.0, 0.0, 0.0
    ts, ys = [], []
    for k in range(n):
        e = i_ref - i
        v_cmd = Kp * e + Ki * acc
        if abs(v_cmd) <= V:              # 조건부 적분 (anti-windup)
            acc += e * Ts
        v_cmd = np.clip(v_cmd, -V, V)
        i = a * i + (1 - a) * (v_apply / R)
        v_apply = v_cmd                  # 1샘플 지연
        ts.append((k + 1) * Ts); ys.append(i)
    return np.array(ts), np.array(ys)

for fs, c in [(20000, 'tab:green'), (10000, 'tab:orange'), (5000, 'tab:red')]:
    ts, ys = digital_loop(fs)
    ax2.step(ts * 1e3, ys, where='post', color=c, lw=1.6,
             label=f'루프 주파수 {fs/1000:.1f} kHz')
    print(f"fs={fs} Hz: 최대 |i| = {np.abs(ys).max():.2f} A, "
          f"최종 i = {ys[-1]:.2f} A")
ax2.axhline(i_ref, color='0.5', ls=':', lw=1)
ax2.set_xlabel('시간 (ms)'); ax2.set_ylabel('전류 (A)')
ax2.set_title('(b) 같은 1 kHz 대역폭 PI를 디지털로: 샘플링이 느리면 폭발')
ax2.legend(fontsize=9); ax2.set_xlim(0, 3); ax2.set_ylim(-15, 30)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_current_step.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 3 — 열 모델
# ============================================================
tau_th = Rth * Cth
print(f"\n열 시정수 tau_th = {tau_th:.0f} s")
print(f"연속 허용 손실 P_cont = (T_max-T_amb)/Rth = {(T_max-T_amb)/Rth:.1f} W "
      f"→ i_cont = {np.sqrt((T_max-T_amb)/Rth/R):.1f} A")

def thermal_sim(i_profile, dt=0.05, t_end=600.0):
    n = int(t_end / dt)
    T = T_amb
    ts, Ts_ = [], []
    for k in range(n):
        i = i_profile(k * dt)
        P = i**2 * R
        T += dt / Cth * (P - (T - T_amb) / Rth)
        ts.append((k + 1) * dt); Ts_.append(T)
    return np.array(ts), np.array(Ts_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for iA, c in [(10, 'seagreen'), (12, 'darkorange'), (30, 'crimson')]:
    ts, Tt = thermal_sim(lambda _t, iA=iA: iA, t_end=600)
    ax1.plot(ts, Tt, color=c, lw=2, label=f'i = {iA} A ({Kt*iA:.1f} Nm)')
    over = ts[Tt > T_max]
    if len(over):
        print(f"i={iA} A: T_max({T_max}C) 도달 시각 = {over[0]:.1f} s")
        ax1.plot(over[0], T_max, 'o', color=c, ms=7)
    else:
        print(f"i={iA} A: {ts[-1]:.0f} s 시점 T = {Tt[-1]:.1f} C (한계 미도달 — 정상상태는 {T_amb + iA**2*R*Rth:.1f} C)")
ax1.axhline(T_max, color='k', ls='--', lw=1)
ax1.annotate('권선 한계 125°C', (410, T_max), xytext=(0, 5),
             textcoords='offset points', fontsize=9)
ax1.set_xscale('log'); ax1.set_xlim(1, 600); ax1.set_ylim(20, 260)
ax1.set_xlabel('시간 (s, 로그축)'); ax1.set_ylabel('권선 온도 (°C)')
ax1.set_title('(a) 일정 전류의 온도 궤적 — 피크는 시한부다')
ax1.legend(fontsize=9, loc='upper left')

# (b) 듀티 사이클: 30 A를 2 s, 쉬기 38 s 반복
duty = lambda t: 30.0 if (t % 40.0) < 2.0 else 0.0
ts, Tt = thermal_sim(duty, t_end=600)
ax2.plot(ts, Tt, 'b-', lw=1.5,
         label='30 A × 2 s / 휴지 38 s (평균 22.5 W)')
Pavg = 30**2 * R * 2 / 40
ax2.axhline(T_amb + Pavg * Rth, color='0.5', ls='--', lw=1,
            label=f'평균 전력 예측 정상상태 {T_amb + Pavg*Rth:.0f}°C')
ax2.axhline(T_max, color='k', ls='--', lw=1)
print(f"듀티 사이클: 평균 전력 = {Pavg:.1f} W, 최종 피크 온도 = {Tt.max():.1f} C")
ax2.set_xlabel('시간 (s)'); ax2.set_ylabel('권선 온도 (°C)')
ax2.set_title('(b) 듀티 사이클: 열 시정수(120 s)가 피크를 평균 내 준다')
ax2.legend(fontsize=9); ax2.set_ylim(20, 140)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_thermal.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 4 — FOC dq 변환
# ============================================================
f_e = 40.0                              # 전기 주파수 (Hz)
t = np.linspace(0, 0.1, 2000)
th = 2 * np.pi * f_e * t                # 전기각
I = 10.0
ia = -I * np.sin(th)                    # 순수 q축 전류가 되도록 구성
ib = -I * np.sin(th - 2 * np.pi / 3)
ic = -I * np.sin(th + 2 * np.pi / 3)

# Clarke (진폭 불변) + Park
ialpha = (2/3) * (ia - 0.5 * ib - 0.5 * ic)
ibeta = (ib - ic) / np.sqrt(3)
i_d = ialpha * np.cos(th) + ibeta * np.sin(th)
i_q = -ialpha * np.sin(th) + ibeta * np.cos(th)
print(f"\nFOC 검증: id = {i_d.mean():.3f} ± {i_d.std():.2e} A, "
      f"iq = {i_q.mean():.3f} ± {i_q.std():.2e} A")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
for y, c, lab in [(ia, 'tab:red', '$i_a$'), (ib, 'tab:green', '$i_b$'),
                  (ic, 'tab:blue', '$i_c$')]:
    ax1.plot(t * 1e3, y, color=c, lw=1.6, label=lab)
ax1.set_xlabel('시간 (ms)'); ax1.set_ylabel('상전류 (A)')
ax1.set_title(f'(a) 고정자 좌표계: 3상 AC (전기 주파수 {f_e:.0f} Hz)')
ax1.legend(fontsize=10, ncol=3); ax1.set_ylim(-14, 14)

ax2.plot(t * 1e3, i_d, 'tab:purple', lw=2, label='$i_d$ (자속 방향) = 0')
ax2.plot(t * 1e3, i_q, 'tab:orange', lw=2, label='$i_q$ (토크 방향) = 10 A')
ax2.set_xlabel('시간 (ms)'); ax2.set_ylabel('dq 전류 (A)')
ax2.set_title('(b) 로터와 함께 도는 좌표계: 같은 전류가 DC로 보인다')
ax2.legend(fontsize=10); ax2.set_ylim(-14, 14)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_foc_dq.png", dpi=140)
plt.close(fig)

print("\n그림 4장 생성 완료:", OUT)
