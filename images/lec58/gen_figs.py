# Lec R25 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
# fig2의 지터 측정은 실행 머신·순간 부하에 따라 달라진다(본문 참조).
# 나머지(fig3~fig5)는 시드 고정으로 결정적이다.
import time, subprocess, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
import mujoco

OUT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# WE-1: 1kHz 루프 지터 측정 (idle vs CPU 부하) → fig2
# ============================================================
def measure(T=0.001, n=5000, strategy="absolute"):
    periods = []
    t_prev = time.monotonic()
    t_next = t_prev + T
    for _ in range(n):
        if strategy == "naive":            # 상대 대기: "T초 자면 되겠지"
            time.sleep(T)
        else:                              # 절대 마감: 다음 마감 시각까지 잔여만
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            t_next += T
        now = time.monotonic()
        periods.append(now - t_prev)
        t_prev = now
    return np.array(periods) * 1e3   # ms

def jit_stats(p, name):
    line = (f"{name:8s} 평균 {p.mean():.4f} ms | σ {p.std()*1e3:6.1f} µs | "
            f"p99 {np.percentile(p,99):.3f} ms | 최악 {p.max():.3f} ms | "
            f"마감(1.5ms) 초과율 {(p > 1.5).mean()*100:.2f}%")
    print(line)

print("== WE-1: 1kHz 루프 지터 (time.monotonic) ==")
ncpu = os.cpu_count()
cache_i, cache_l = os.path.join(OUT, "p_idle.npy"), os.path.join(OUT, "p_load.npy")
if os.path.exists(cache_i) and os.path.exists(cache_l):
    p_idle, p_load = np.load(cache_i), np.load(cache_l)   # 재측정하려면 .npy 삭제
else:
    p_idle = measure()
    load = [subprocess.Popen([sys.executable, "-c", "while True: pass"])
            for _ in range(ncpu * 2)]
    time.sleep(1.0)
    try:
        p_load = measure()
    finally:
        for pr in load:
            pr.kill()
    np.save(cache_i, p_idle); np.save(cache_l, p_load)
cache_n = os.path.join(OUT, "p_naive.npy")
if os.path.exists(cache_n):
    p_naive = np.load(cache_n)
else:
    p_naive = measure(strategy="naive")
    np.save(cache_n, p_naive)
jit_stats(p_idle, "idle")
jit_stats(p_load, "loaded")
print(f"드리프트(5000주기 목표 5.000 s): 상대 sleep {p_naive.sum()/1e3:.3f} s "
      f"(+{p_naive.sum()-5000:.0f} ms) vs 절대 마감 {p_idle.sum()/1e3:.4f} s "
      f"(+{p_idle.sum()-5000:.1f} ms)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
bins = np.linspace(0.5, 4.0, 120)
for ax, p, name in ((axes[0], p_idle, "한가한 머신 (idle)"),
                    (axes[1], p_load, f"CPU 포화 부하 (busy-loop ×{ncpu*2})")):
    ax.hist(np.clip(p, bins[0], bins[-1]), bins=bins, color='#4878cf', alpha=0.85)
    ax.set_yscale('log')
    ax.axvline(1.0, color='k', lw=1, ls='--')
    ax.axvline(1.5, color='crimson', lw=1.2, ls=':')
    ax.set_title(f"{name}\n평균 {p.mean():.3f} ms · p99 {np.percentile(p,99):.2f} ms · 최악 {p.max():.2f} ms",
                 fontsize=10)
    ax.set_xlabel("실측 주기 [ms]")
axes[0].set_ylabel("빈도 (log)")
axes[0].annotate("목표 1 ms", xy=(1.02, 2e3), fontsize=9)
axes[1].annotate("마감 1.5 ms", xy=(1.55, 2e3), fontsize=9, color='crimson')
fig.suptitle("같은 코드, 같은 머신 — 평균은 같지만 꼬리(worst-case)가 다르다", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_jitter_histogram.png"), dpi=140)
plt.close(fig)

# ============================================================
# fig1: 한 장 요약 — 소프트웨어 스택과 주기 계층
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.2))
layers = [
    ("VLA 정책 (파이썬, GPU)", "액션 청크 생성", 0.1, "~1–10 Hz", "#c7d7f0", "비실시간"),
    ("ROS 2 노드·토픽 (DDS)", "지각·정책·보간기 연결", 0.02, "10–100 Hz", "#a9c4e8", "비실시간(soft RT)"),
    ("ros2_control 루프", "read→update→write", 0.001, "0.5–1 kHz", "#f2c894", "RT 스레드 (PREEMPT_RT)"),
    ("필드버스 (EtherCAT/CAN)", "관절 명령·상태 왕복", 0.0005, "1–4 kHz", "#f0b077", "버스 마스터 + DC 동기화"),
    ("드라이브 FOC 전류 루프", "전류≈토크 (R14)", 0.00005, "10–40 kHz", "#e89a5e", "펌웨어 (MCU/FPGA)"),
]
y = len(layers)
for i, (name, role, T, rate, color, rt) in enumerate(layers):
    yy = y - i
    ax.add_patch(plt.Rectangle((0.0, yy - 0.42), 4.6, 0.84, color=color, ec='k', lw=0.6))
    ax.text(0.15, yy + 0.16, name, fontsize=11, weight='bold', va='center')
    ax.text(0.15, yy - 0.20, f"{role} · {rt}", fontsize=9, va='center', color='#333')
    ax.text(4.45, yy, rate, fontsize=10, va='center', ha='right', weight='bold')
    # 오른쪽: 주기를 log 스케일 막대로
    barlen = np.interp(np.log10(T), [-4.5, -0.5], [0.3, 3.2])
    ax.barh(yy, barlen, left=5.1, height=0.5, color=color, ec='k', lw=0.6)
    ax.text(5.1 + barlen + 0.08, yy, f"T = {T*1000:g} ms", fontsize=9, va='center')
for i in range(len(layers) - 1):
    ax.annotate("", xy=(2.3, y - i - 0.45), xytext=(2.3, y - i - 0.55),
                arrowprops=dict(arrowstyle='-|>', color='k'))
ax.plot([5.0, 5.0], [0.4, y + 0.55], color='gray', lw=0.8)
ax.text(5.1, y + 0.62, "주기 T (log 눈금, 막대가 길수록 느린 루프)", fontsize=9, color='gray')
ax.text(0.0, y + 0.62, "아래로 갈수록: 주기 짧아지고, 지터 허용량 줄고, 보장 수단이 소프트웨어→하드웨어로", fontsize=10)
ax.axhline(y - 1.5, color='crimson', lw=1.0, ls='--', xmin=0.0, xmax=0.52)
ax.text(4.7, y - 1.5, "← 실시간 경계", fontsize=9, color='crimson', va='center')
ax.set_xlim(-0.1, 9.2); ax.set_ylim(0.3, y + 1.0)
ax.axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_stack_layers.png"), dpi=140)
plt.close(fig)

# ============================================================
# WE-2: R17 진자 PID에 지터 주입 → fig3
# ============================================================
XML = """
<mujoco model="pendulum1dof">
  <option timestep="0.0005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0  0 0 -0.5" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-20 20"/></actuator>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
DT = m.opt.timestep
GAINS = (6.0, 22.0, 0.41)          # R17 실습의 Z-N 게인 (100 Hz)

def run_pid(gains=GAINS, f_ctrl=100.0, jit_ms=0.0, p_stall=0.0, stall_ms=50.0,
            T_end=8.0, seed=0, target=np.pi/3):
    Kp, Ki, Kd = gains
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(m, d)
    T = 1.0/f_ctrl
    ei, e_prev, u = 0.0, None, 0.0
    t_upd, k = 0.0, 0
    qs = []
    for i in range(int(T_end/DT)):
        t = i*DT
        if t >= t_upd - 1e-12:                    # 갱신 시각 도달?
            e = target - d.qpos[0]
            ei += e*T                             # 순진한 구현: 고정 T 가정
            de = 0.0 if e_prev is None else (e - e_prev)/T
            e_prev = e
            u = Kp*e + Ki*ei + Kd*de
            k += 1
            late = rng.uniform(0.0, jit_ms*1e-3)  # 마감 지각(지터)
            if p_stall > 0 and rng.random() < p_stall:
                late += stall_ms*1e-3             # 드물게 긴 스톨(GC pause 유사)
            t_upd = k*T + late
        d.ctrl[0] = u
        mujoco.mj_step(m, d)
        qs.append(d.qpos[0])
    qs = np.array(qs)
    ts = np.arange(len(qs))*DT
    rms = np.sqrt(np.mean((qs[ts >= 3.0] - target)**2))
    return ts, qs, rms

print("\n== WE-2: 진자 PID(Z-N 게인, 100Hz)에 지터 주입 ==")
ts, q_clean, rms_clean = run_pid(jit_ms=0)
print(f"clean 100Hz: 정착 후 RMS = {rms_clean*1e3:.2f} mrad")
jits = [0, 2, 5, 10, 15, 20]
means, maxs = [], []
for jit in jits:
    rmss = [run_pid(jit_ms=jit, seed=s)[2] for s in range(5)]
    means.append(np.mean(rmss)); maxs.append(np.max(rmss))
    print(f"지터 U(0,{jit:2d}ms): RMS = {np.mean(rmss)*1e3:7.2f} mrad (5시드, 최악 {np.max(rmss)*1e3:.2f})")
for stall in (30, 50):
    rmss = [run_pid(p_stall=0.02, stall_ms=stall, seed=s)[2] for s in range(5)]
    print(f"2% 확률 {stall}ms 스톨: RMS = {np.mean(rmss)*1e3:.2f} mrad (최악 {np.max(rmss)*1e3:.2f})")
_, q_j20, _ = run_pid(jit_ms=20, seed=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.plot(ts, np.degrees(q_clean), lw=1.2, label="정시 100 Hz (지터 0)")
ax.plot(ts, np.degrees(q_j20), lw=1.0, alpha=0.9, label="지터 U(0, 20 ms)")
ax.axhline(60, color='k', ls='--', lw=0.8)
ax.set_xlabel("시간 [s]"); ax.set_ylabel("관절각 [°]")
ax.set_title("같은 게인, 같은 100 Hz — 갱신 시각만 흔들었다", fontsize=10)
ax.legend(fontsize=9); ax.set_xlim(0, 8)
ax = axes[1]
ax.errorbar(jits, np.array(means)*1e3, yerr=(np.array(maxs)-np.array(means))*1e3,
            marker='o', capsize=3, lw=1.4, label="지터 주입 (5시드 평균, 막대=최악)")
ax.axhline(rms_clean*1e3, color='gray', ls='--', lw=1, label="정시 100 Hz 기준")
ax.set_xlabel("지터 상한 j [ms]  (지각 ~ U(0, j))")
ax.set_ylabel("정착 후 RMS 오차 [mrad]")
ax.set_title("열화는 완만하다가 급해진다 — 평균 지연 j/2가\n위상 예산을 갉아먹는 만큼", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_jitter_degradation.png"), dpi=140)
plt.close(fig)

# ============================================================
# WE-3: QoS 시뮬 (best_effort vs reliable) → fig4
# ============================================================
def qos_sim(p=0.3, T=0.010, RTT=0.020, d=0.0002, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    t_pub = np.arange(n)*T                     # 표본 생성 시각(타임스탬프)
    ok = rng.random(n) >= p                    # 1차 전송 성공 여부
    arr_be = np.where(ok, t_pub + d, np.inf)   # BEST_EFFORT: 유실은 영영 안 옴
    n_retry = rng.geometric(1-p, size=n) - 1   # RELIABLE: 실패 횟수 ~ 기하분포
    raw = t_pub + d + RTT*n_retry
    arr_rel = np.maximum.accumulate(raw)       # 순서 보장 → head-of-line blocking
    polls = t_pub + T/2                        # 소비자는 주기 중간에 최신 표본을 사용
    def staleness(arrivals):
        ages = []
        for tp in polls:
            idx = np.where(arrivals <= tp)[0]  # 지금까지 도착한 표본들
            if len(idx):
                ages.append(tp - t_pub[idx.max()])
        return np.array(ages) * 1e3            # ms
    return staleness(arr_be), staleness(arr_rel)

print("\n== WE-3: QoS 시뮬 (손실 p=0.3, T=10ms) ==")
a_be, a_rel20 = qos_sim(RTT=0.020)
_,    a_rel4  = qos_sim(RTT=0.004)
for name, a in (("best_effort", a_be), ("reliable RTT=20ms", a_rel20),
                ("reliable RTT=4ms", a_rel4)):
    print(f"{name:18s} 평균 나이 {a.mean():6.2f} ms | p99 {np.percentile(a,99):6.1f} ms | 최악 {a.max():6.1f} ms")
p = 0.3
print(f"이론: E[재전송] = p/(1-p) = {p/(1-p):.3f} → BE 추가 나이 {10*p/(1-p):.2f} ms, "
      f"REL 추가 지연 RTT={20}ms 기준 {20*p/(1-p):.2f} ms")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
tt = np.arange(len(a_be))*0.01
ax.step(tt[:300], a_be[:300], lw=1.1, label="best_effort")
ax.step(tt[:300], a_rel20[:300], lw=1.1, alpha=0.85, label="reliable (RTT 20 ms)")
ax.set_xlabel("시간 [s]"); ax.set_ylabel("소비 시점의 표본 나이 [ms]")
ax.set_title("유실률 30%의 링크: 재전송은 신선도를 해친다", fontsize=10)
ax.legend(fontsize=9)
ax = axes[1]
for a, name in ((a_be, "best_effort"), (a_rel20, "reliable, RTT 20 ms > T"),
                (a_rel4, "reliable, RTT 4 ms < T")):
    xs = np.sort(a); ys = np.arange(1, len(xs)+1)/len(xs)
    ax.plot(xs, ys, lw=1.4, label=name)
ax.set_xscale('log')
ax.set_xlabel("표본 나이 [ms] (log)"); ax.set_ylabel("CDF")
ax.set_title("승부는 RTT vs 주기 T — 재전송이 다음 표본보다\n빨리 올 때만 reliable이 이긴다", fontsize=10)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_qos_staleness.png"), dpi=140)
plt.close(fig)

# ============================================================
# E2: 지연 예산 — R17의 PD 설계(Kp=900, Kd=59) 재계산 + fig5
# ============================================================
print("\n== E2: 지연 예산 (플랜트 1/(s(s+1)), PD 900/59) ==")
w = np.logspace(-1, 3, 200000)
L = (59*1j*w + 900) / (1j*w * (1j*w + 1))
i_c = np.argmin(np.abs(np.abs(L) - 1.0))
wc = w[i_c]
pm = 180 + np.degrees(np.angle(L[i_c]))
tau_max = np.radians(pm) / wc
print(f"ω_c = {wc:.1f} rad/s, φ_m = {pm:.1f}°, τ_max = φ_m/ω_c = {tau_max*1e3:.1f} ms")

scen = [
    ("A. 온보드 100 Hz",           [("ZOH T/2", 5.0), ("미분차분 T/2", 5.0), ("계산 1주기", 10.0)]),
    ("B. 온보드 200 Hz + DDS 로컬", [("ZOH", 2.5), ("미분차분", 2.5), ("계산", 5.0), ("DDS(가정 0.3)", 0.3)]),
    ("C. 오프보드 Wi-Fi (평균)",    [("ZOH", 5.0), ("미분차분", 5.0), ("계산", 10.0), ("Wi-Fi 왕복(가정)", 10.0)]),
    ("D. 오프보드 Wi-Fi (스파이크)", [("ZOH", 5.0), ("미분차분", 5.0), ("계산", 10.0), ("Wi-Fi 스파이크", 80.0)]),
]
fig, ax = plt.subplots(figsize=(10, 3.9))
colors = ['#4878cf', '#6fa8dc', '#f2c894', '#e06666']
for i, (name, parts) in enumerate(scen):
    left, total = 0.0, sum(v for _, v in parts)
    for j, (pname, v) in enumerate(parts):
        ax.barh(len(scen)-1-i, min(v, 40-left), left=left, height=0.55,
                color=colors[j % 4], ec='k', lw=0.5)
        if v >= 2.2 and left < 38:
            ax.text(left + min(v, 40-left)/2, len(scen)-1-i, pname, fontsize=8,
                    ha='center', va='center')
        left += min(v, 40-left)
    phi_rem = pm - np.degrees(wc * total * 1e-3)
    verdict = f"Στ = {total:.1f} ms → 남는 위상 {phi_rem:+.1f}°" + ("  (불안정!)" if phi_rem < 0 else "")
    ax.text(40.6, len(scen)-1-i, verdict, fontsize=9, va='center',
            color=('crimson' if phi_rem < 0 else 'k'))
    print(f"{name}: Στ = {total:.1f} ms, 남는 위상여유 = {phi_rem:+.1f}°")
ax.axvline(tau_max*1e3, color='crimson', lw=1.5, ls='--')
ax.text(tau_max*1e3 + 0.4, 1.5, f"τ_max = {tau_max*1e3:.1f} ms", color='crimson', fontsize=10, va='center')
ax.set_yticks(range(len(scen)))
ax.set_yticklabels([s[0] for s in scen][::-1], fontsize=10)
ax.set_xlim(0, 62); ax.set_xlabel("왕복 지연 예산 소비 [ms]")
ax.set_title("같은 제어기, 다른 배치 — 지연 예산 회계 (D의 스파이크 80 ms는 축 밖까지 이어진다)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_latency_budget.png"), dpi=140)
plt.close(fig)

print("\n그림 5장 생성 완료:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
