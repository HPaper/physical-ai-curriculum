"""Lec R16 그림 생성 스크립트.
fig1: 액추에이터 설계 공간 지도(토크 밀도 vs 역구동성) + 발끝 등가 질량 막대
fig2: 전류 기반 토크 추정 오차 — 부하 의존성과 감속비 의존성
fig3: 착지 충격 시뮬 — QDD(n=6) vs 하모닉(n=100) 등가 관성의 접촉력
fig4: 토크 명령→출력 대역폭 — 개루프 전류 vs 토크센서 피드백 vs SEA
fig5: 실습 기대 결과 — MuJoCo 1-DoF 다리 낙하 (armature × 제어 모드)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.rcParams.update({'font.family': 'Noto Sans CJK JP',
                     'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lec16"
g = 9.81

# ============================================================
# fig 1 — 설계 공간 지도 (개념도) + 발끝 등가 질량
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                               gridspec_kw={'width_ratios': [1.5, 1]})

# (a) 개념 지도: x = 역구동성/투명성(정성), y = 토크 밀도(정성)
acts = [  # (x, y, 이름, 색, 부가설명)
    (0.92, 0.22, "다이렉트 드라이브\n(n=1)", 'gray',
     "완전 투명하지만\n토크 밀도 부족"),
    (0.75, 0.62, "QDD\n(대구경 모터+n≈6)", 'crimson',
     "MIT Cheetah의 선택"),
    (0.18, 0.80, "하모닉\n(n≈100)", 'royalblue',
     "무백래시·고토크밀도\n비역구동"),
    (0.30, 0.72, "싸이클로이드", 'teal', "내충격·고강성"),
    (0.28, 0.88, "하모닉+토크센서\n(Franka류)", 'navy',
     "센서로 힘을 '측정'"),
    (0.55, 0.35, "SEA\n(고감속+스프링)", 'darkorange',
     "스프링으로 힘을 측정\n대역폭 낮음"),
]
for x, y, name, c, note in acts:
    ax1.scatter(x, y, s=160, color=c, zorder=5, edgecolor='k', linewidth=0.8)
    ax1.annotate(name, (x, y), textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=9, color=c, fontweight='bold')
    ax1.annotate(note, (x, y), textcoords='offset points', xytext=(0, -14),
                 ha='center', va='top', fontsize=7.5, color='0.35')
ax1.add_patch(plt.Rectangle((0.62, 0.45), 0.36, 0.42, fill=True,
                            facecolor='mistyrose', edgecolor='crimson',
                            linestyle='--', alpha=0.5, zorder=1))
ax1.text(0.80, 0.94, "보행(충격+힘 제어)이 요구하는 영역", fontsize=9,
         color='crimson', ha='center')
ax1.set_xlim(0, 1.05); ax1.set_ylim(0, 1.05)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_xlabel("역구동성 / 힘 투명성  →  (정성 축)")
ax1.set_ylabel("토크 밀도 (Nm/kg)  →  (정성 축)")
ax1.set_title("(a) 액추에이터 설계 공간 — 어디에도 공짜는 없다")

# (b) 발끝 등가 질량 (WE-2의 수치)
names = ["다이렉트\n드라이브", "QDD\n$n$=6", "하모닉\n$n$=100"]
m_leg, L = 1.0, 0.3
vals = [m_leg,
        m_leg + 6**2 * 1.5e-4 / L**2,
        m_leg + 100**2 * 5e-5 / L**2]
colors = ['gray', 'crimson', 'royalblue']
bars = ax2.bar(names, vals, color=colors, alpha=0.85)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.2f} kg",
             ha='center', fontsize=10)
ax2.axhline(m_leg, color='k', lw=0.8, ls=':')
ax2.text(2.4, m_leg + 0.06, "다리 자체 질량 1.0 kg", fontsize=8, ha='right')
ax2.set_ylabel("발끝 등가 질량 $m_{app}$ (kg)")
ax2.set_title("(b) 반사 관성이 만드는 '보이지 않는 질량'\n"
              r"$m_{app} = m_{leg} + n^2 J_m / L^2$")
ax2.set_ylim(0, 7.4)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_design_space.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 2 — 전류 기반 토크 추정 오차
# ============================================================
Kt = 0.1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (a) 상대 오차 vs 부하 토크: e = (1-eta) + tau_c / tau_load
tau_load = np.linspace(0.5, 40, 400)
for name, eta, tau_c, c in [("QDD ($n$=6): $\\eta$=0.96, $\\tau_c$=0.2 Nm",
                             0.96, 0.2, 'crimson'),
                            ("하모닉 ($n$=100): $\\eta$=0.75, $\\tau_c$=2.0 Nm",
                             0.75, 2.0, 'royalblue')]:
    e = (1 - eta) + tau_c / tau_load
    ax1.plot(tau_load, 100 * e, color=c, lw=2.2, label=name)
for tl, e_pt, c in [(5, 8.0, 'crimson'), (20, 5.0, 'crimson'),
                    (5, 65.0, 'royalblue'), (20, 35.0, 'royalblue')]:
    ax1.plot(tl, e_pt, 'o', color=c, ms=7, zorder=5)
    ax1.annotate(f"{e_pt:.0f}%", (tl, e_pt), textcoords='offset points',
                 xytext=(6, 4), fontsize=9, color=c)
ax1.set_xlabel(r"출력 부하 토크 $\tau_{load}$ (Nm)")
ax1.set_ylabel(r"추정 상대 오차 $|\tau - \hat\tau|/\tau$  (%)")
ax1.set_ylim(0, 100)
ax1.set_title("(a) 추정 오차는 부하 의존적 — 가벼운 접촉일수록 나쁘다\n"
              "(WE-1의 마찰 모델)")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# (b) 오차 vs 감속비 (유성 다단 모델: 단당 eta 0.97, 입력측 쿨롱 0.02 Nm)
n_arr = np.logspace(0, np.log10(200), 200)
eta_in, tauc_in = 0.97, 0.02
stages = np.maximum(1, np.ceil(np.log(n_arr) / np.log(8)))
eta_n = eta_in ** stages
tauc_n = tauc_in * n_arr
for tl, ls in [(20, '-'), (5, '--')]:
    e = (1 - eta_n) + tauc_n / tl
    ax2.plot(n_arr, 100 * e, color='seagreen', ls=ls, lw=2,
             label=f"유성 다단 모델, 부하 {tl} Nm")
ax2.plot(6, 100 * ((1 - 0.96) + 0.2 / 20), 'o', color='crimson', ms=9,
         zorder=5)
ax2.annotate("QDD ($n$=6)", (6, 5), textcoords='offset points',
             xytext=(8, 4), fontsize=9, color='crimson')
ax2.plot(100, 35, 's', color='royalblue', ms=9, zorder=5)
ax2.annotate("하모닉 ($n$=100)\n(별도 기술 — 곡선 위 점 아님)", (100, 35),
             textcoords='offset points', xytext=(-10, 10), fontsize=8,
             color='royalblue', ha='right')
ax2.set_xscale('log')
ax2.set_xlabel("감속비 $n$")
ax2.set_ylabel("추정 상대 오차 (%)")
ax2.set_title("(b) 감속비가 클수록 마찰이 신호를 삼킨다\n"
              "(모델 가정: 단당 $\\eta$=0.97, 입력측 쿨롱 0.02 Nm)")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_torque_estimation.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 3 — 착지 충격: 접촉력 시계열 (WE-2의 RK4)
# ============================================================
m_leg, L, kg_ = 1.0, 0.3, 5e4
v0 = np.sqrt(2 * g * 0.5)

def contact_sim(m_app, dt=1e-6, tmax=0.06):
    s = np.array([0.0, v0])
    ts, Fs = [], []
    t = 0.0
    while t < tmax:
        f = lambda s: np.array([s[1], g - kg_ * max(s[0], 0.0) / m_app])
        k1 = f(s); k2 = f(s + dt/2*k1); k3 = f(s + dt/2*k2); k4 = f(s + dt*k3)
        s = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        ts.append(t); Fs.append(kg_ * max(s[0], 0.0))
        if s[0] < 0 and s[1] < 0:
            break
    return np.array(ts), np.array(Fs)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
for name, Jm, n, c in [("QDD: $n$=6, $J_m$=1.5e-4", 1.5e-4, 6, 'crimson'),
                       ("하모닉: $n$=100, $J_m$=5e-5", 5e-5, 100, 'royalblue')]:
    m_app = m_leg + n**2 * Jm / L**2
    ts, Fs = contact_sim(m_app)
    pk = Fs.max()
    ax.plot(1e3 * ts, Fs, color=c, lw=2.2,
            label=f"{name} → $m_{{app}}$={m_app:.2f} kg, 피크 {pk:.0f} N")
    ax.annotate(f"{pk:.0f} N", (1e3 * ts[Fs.argmax()], pk),
                textcoords='offset points', xytext=(6, 4), fontsize=10,
                color=c, fontweight='bold')
ax.set_xlabel("접촉 후 시간 (ms)")
ax.set_ylabel("지면 접촉력 (N)")
ax.set_title("같은 0.5 m 낙하(3.13 m/s), 같은 지면($k$=5·10$^4$ N/m) — "
             "다른 것은 반사 관성뿐")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_impact.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 4 — 토크 대역폭 Bode
# ============================================================
w = np.logspace(-1, 4.2, 4000) * 2 * np.pi
Jr = 0.5

def closed_pi(ks, Kp, Ki, zeta=0.4):
    b = 2 * zeta * np.sqrt(ks * Jr)
    num_L = np.array([Kp * ks, Ki * ks])
    den_L = np.array([Jr, b, ks, 0.0])
    den_T = np.polyadd(den_L, np.pad(num_L, (len(den_L) - len(num_L), 0)))
    return signal.TransferFunction(num_L, den_T)

wi = 2 * np.pi * 1000
systems = [
    ("QDD 개루프 전류 (전류루프 1 kHz)", signal.TransferFunction([wi], [1, wi]),
     'crimson'),
    ("토크센서 관절: $k_s$=10$^4$, PI(0.5, 60)", closed_pi(1e4, 0.5, 60.0),
     'royalblue'),
    ("SEA: $k_s$=300, PI(0.5, 10)", closed_pi(300.0, 0.5, 10.0),
     'darkorange'),
]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for name, sys, c in systems:
    _, mag, _ = signal.bode(sys, w=w)
    f_hz = w / (2 * np.pi)
    ax.semilogx(f_hz, mag, color=c, lw=2.2, label=name)
    idx = np.where(mag < mag[0] - 3.0)[0]
    if len(idx):
        f3 = f_hz[idx[0]]
        ax.plot(f3, mag[idx[0]], 'o', color=c, ms=7, zorder=5)
        ax.annotate(f"−3 dB @ {f3:.3g} Hz", (f3, mag[idx[0]]),
                    textcoords='offset points', xytext=(6, -14),
                    fontsize=9, color=c)
ax.axhline(-3, color='k', lw=0.8, ls=':')
ax.set_xlabel("주파수 (Hz)")
ax.set_ylabel(r"$|\tau_{out}/\tau_{cmd}|$ (dB)")
ax.set_ylim(-30, 6)
ax.set_title("토크 명령 → 출력 토크 전달함수 (WE-3의 모델)")
ax.legend(fontsize=9, loc='lower left'); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_bandwidth.png", dpi=140)
plt.close(fig)

# ============================================================
# fig 5 — 실습 기대 결과: MuJoCo 1-DoF 다리 낙하
# ============================================================
import mujoco

XML = """
<mujoco model="leg1dof">
  <option timestep="1e-4" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" solref="0.002 1"/>
    <body name="torso" pos="0 0 0.93">
      <joint name="z" type="slide" axis="0 0 1"/>
      <geom name="body" type="sphere" size="0.08" mass="5" contype="0" conaffinity="0"/>
      <body name="leg" pos="0 0 -0.25">
        <joint name="ext" type="slide" axis="0 0 1" range="-0.15 0.15"
               armature="{ARM}"/>
        <geom name="foot" type="sphere" size="0.03" mass="0.5" pos="0 0 -0.15"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="leg_motor" joint="ext" ctrlrange="-400 400"/></actuator>
</mujoco>
"""

def drop(armature, mode, kv=2000.0, bv=100.0, kp_pos=4e4, kd_pos=200.0, T=0.6):
    m = mujoco.MjModel.from_xml_string(XML.format(ARM=armature))
    d = mujoco.MjData(m)
    foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    ts, Fs = [], []
    for k in range(int(T / m.opt.timestep)):
        q, qd = d.qpos[1], d.qvel[1]
        if mode == "impedance":
            d.ctrl[0] = kv * (0.0 - q) - bv * qd
        else:
            d.ctrl[0] = kp_pos * (0.0 - q) - kd_pos * qd
        mujoco.mj_step(m, d)
        F = 0.0
        for c in range(d.ncon):
            con = d.contact[c]
            if foot_id in (con.geom1, con.geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, f6)
                F += f6[0]
        ts.append(d.time); Fs.append(F)
    return np.array(ts), np.array(Fs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={'width_ratios': [1.4, 1]})
cases = [(0.06, "impedance", 'crimson', '-', "QDD급 armature=0.06, 임피던스"),
         (0.06, "stiff", 'crimson', ':', "QDD급, 강성 위치 유지"),
         (5.56, "impedance", 'royalblue', '-', "하모닉급 armature=5.56, 임피던스"),
         (5.56, "stiff", 'royalblue', ':', "하모닉급, 강성 위치 유지")]
for arm, mode, c, ls, lab in cases:
    ts, Fs = drop(arm, mode)
    pk = Fs.max()
    ax1.plot(1e3 * (ts - 0.3), Fs, color=c, ls=ls, lw=1.8,
             label=f"{lab} (피크 {pk:.0f} N)")
ax1.set_xlim(15, 120)
ax1.set_xlabel("시간 (ms, 접촉 부근)")
ax1.set_ylabel("발-지면 접촉력 (N)")
ax1.set_title("(a) 가상 스프링-댐퍼는 QDD에서만 충격을 줄인다")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

arms = np.array([0.0, 0.06, 0.25, 0.5, 1.0, 2.0, 3.5, 5.56])
pks = []
for a in arms:
    _, Fs = drop(a, "impedance")
    pks.append(Fs.max())
pks = np.array(pks)
ax2.plot(arms, pks, 'o-', color='seagreen', lw=2, label="측정 (임피던스 제어 켬)")
m_foot = 0.5
ref = pks[-1] * np.sqrt((m_foot + arms) / (m_foot + arms[-1]))
ax2.plot(arms, ref, 'k--', lw=1.2,
         label=r"강체 낙하 한계 $\propto\sqrt{m_{foot}+n^2 J_m/L^2}$"
               "\n(E2 스케일링, 최대점 기준)")
ax2.fill_between(arms, pks, ref, where=ref > pks, color='seagreen',
                 alpha=0.15)
ax2.annotate("임피던스 제어가\n벌어주는 마진", (0.5, 660), fontsize=9,
             color='seagreen')
ax2.annotate("반사 관성이 크면\n제어로 못 구한다", (3.3, 1250), fontsize=9,
             color='0.3')
ax2.set_xlabel("armature (= 관절 반사 관성, kg)")
ax2.set_ylabel("피크 접촉력 (N)")
ax2.set_title("(b) 피크 힘 vs 반사 관성 — 측정이 강체 한계로 수렴")
ax2.legend(fontsize=8, loc='upper left'); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_lab_drop.png", dpi=140)
plt.close(fig)

print("figs done")
