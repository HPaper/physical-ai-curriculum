# -*- coding: utf-8 -*-
# Lec R30 그림 생성 스크립트 — 미니 통합 데모 (가짜 VLA → 보간 → 제어기 → 2R 플랜트)
# 2R 파라미터는 R10/R19와 동일. 실행: python3 gen_figs.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

# ---------------------------------------------------------------
# 2R 동역학 (R10 WE-3 / R19와 동일 파라미터)
# ---------------------------------------------------------------
m1, m2, l1, l2, lc1, lc2 = 1.0, 1.0, 1.0, 1.0, 0.5, 0.5
I1 = I2 = 1.0 / 12
grav = 9.81

def M_mat(q):
    c2 = np.cos(q[1])
    return np.array([[I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2),
                      I2 + m2 * (lc2**2 + l1 * lc2 * c2)],
                     [I2 + m2 * (lc2**2 + l1 * lc2 * c2), I2 + m2 * lc2**2]])

def C_mat(q, qd):
    h = m2 * l1 * lc2 * np.sin(q[1])
    return np.array([[-h * qd[1], -h * (qd[0] + qd[1])], [h * qd[0], 0.0]])

def g_vec(q):
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return grav * np.array([(m1 * lc1 + m2 * l1) * c1 + m2 * lc2 * c12, m2 * lc2 * c12])

def fk(q):
    return np.array([l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
                     l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])])

def jac(q):
    s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                     [l1 * c1 + l2 * c12, l2 * c12]])

def ik2r(p):
    x, y = p
    c2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])

def rk4(q, qd, tau, fext_fn, dt):
    """토크 ZOH 유지, EEF 외력 fext_fn(q, qd) 포함 RK4 한 스텝."""
    def deriv(s):
        qq, vv = s[:2], s[2:]
        f = fext_fn(qq, vv)
        a = np.linalg.solve(M_mat(qq), tau + jac(qq).T @ f - C_mat(qq, vv) @ vv - g_vec(qq))
        return np.concatenate([vv, a])
    s = np.concatenate([q, qd])
    k1 = deriv(s); k2 = deriv(s + dt / 2 * k1)
    k3 = deriv(s + dt / 2 * k2); k4 = deriv(s + dt * k3)
    s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s[:2], s[2:]

no_force = lambda qq, vv: np.zeros(2)
dt = 1e-3                      # 제어·시뮬 공통 1 kHz
Kp, Kd = 400.0, 40.0           # CT 오차 동역학: 이중극 -20 (ζ=1, ωn=20)

# ---------------------------------------------------------------
# 공통 "가짜 VLA" 기준 궤적 (자유 공간): 5초, 시작 정지
# ---------------------------------------------------------------
def ref(t):
    t = np.asarray(t, dtype=float)
    x = 1.0 + 0.15 * (1 - np.cos(2 * np.pi * 0.4 * t))
    y = 0.45 * np.cos(2 * np.pi * 0.2 * t)
    return np.stack([x, y], axis=-1)

T_end = 5.0
ts = np.arange(0, T_end, dt)
ref_fine = ref(ts)

# ===============================================================
# 실험 (a): 보간 유무 — 10 Hz 청크를 1 kHz 셋포인트로
# ===============================================================
t_wp = np.arange(0, T_end + 1e-9, 0.1)          # 10 Hz 웨이포인트 51개
q_wp = np.array([ik2r(p) for p in ref(t_wp)])

def build_setpoints(mode):
    if mode == 'zoh':
        idx = np.minimum((ts / 0.1).astype(int), len(t_wp) - 1)
        qd_ = q_wp[idx]
        return qd_, np.zeros_like(qd_), np.zeros_like(qd_)
    if mode == 'lin':
        qd_ = np.stack([np.interp(ts, t_wp, q_wp[:, j]) for j in range(2)], axis=1)
        vd_ = np.gradient(qd_, dt, axis=0)  # 조각별 상수 기울기
        return qd_, vd_, np.zeros_like(qd_)
    cs = CubicSpline(t_wp, q_wp, axis=0)
    return cs(ts), cs(ts, 1), cs(ts, 2)

def run_tracking(mode):
    qd_, vd_, ad_ = build_setpoints(mode)
    q, qdv = q_wp[0].copy(), np.zeros(2)
    Q = np.zeros((len(ts), 2)); TAU = np.zeros((len(ts), 2))
    for i in range(len(ts)):
        e, ed = qd_[i] - q, vd_[i] - qdv
        tau = M_mat(q) @ (ad_[i] + Kd * ed + Kp * e) + C_mat(q, qdv) @ qdv + g_vec(q)
        Q[i] = q; TAU[i] = tau
        q, qdv = rk4(q, qdv, tau, no_force, dt)
    eef = np.array([fk(qq) for qq in Q])
    err_mm = np.linalg.norm(eef - ref_fine, axis=1) * 1000
    return dict(Q=Q, TAU=TAU, eef=eef, err=err_mm, qd=qd_)

print("=" * 60)
print("실험 (a): 보간 유무 (10Hz 청크 → 1kHz 셋포인트, CT 제어)")
res_a = {}
mask = ts >= 0.5
for mode, name in [('zoh', 'ZOH'), ('lin', '선형'), ('cub', '3차 스플라인')]:
    r = run_tracking(mode)
    res_a[mode] = r
    rms = np.sqrt(np.mean(r['err'][mask]**2))
    print(f"  {name:10s} RMS EEF 오차 {rms:8.2f} mm | 피크 오차 {r['err'][mask].max():7.2f} mm"
          f" | 피크 |tau| {np.abs(r['TAU'][mask]).max():7.1f} N·m")

# 그림 1
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
ax.plot(ref_fine[:, 0], ref_fine[:, 1], 'k--', lw=1.5, label='정책 의도 (연속)')
ax.plot(res_a['zoh']['eef'][:, 0], res_a['zoh']['eef'][:, 1], 'C3', lw=1, label='ZOH 추종')
ax.plot(res_a['cub']['eef'][:, 0], res_a['cub']['eef'][:, 1], 'C0', lw=1, label='3차 추종')
wp = ref(t_wp)
ax.plot(wp[:, 0], wp[:, 1], 'k.', ms=3, alpha=0.5)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('(a) EEF 궤적 (점 = 10 Hz 액션)')
ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax = axes[1]
for mode, name, c in [('zoh', 'ZOH', 'C3'), ('lin', '선형', 'C1'), ('cub', '3차', 'C0')]:
    ax.semilogy(ts, np.maximum(res_a[mode]['err'], 1e-3), c, lw=1, label=name)
ax.set_xlabel('t [s]'); ax.set_ylabel('|EEF 오차| [mm]'); ax.set_title('(b) 추종 오차 (로그)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[2]
ax.plot(ts, res_a['zoh']['TAU'][:, 0], 'C3', lw=0.8, label='ZOH')
ax.plot(ts, res_a['cub']['TAU'][:, 0], 'C0', lw=1.2, label='3차')
ax.set_xlabel('t [s]'); ax.set_ylabel(r'$\tau_1$ [N·m]'); ax.set_title('(c) 관절 1 토크')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('실험 (a) — 같은 10 Hz 액션 청크, 보간 계층만 다르다 (00강 ZOH 실험의 완결판)', y=1.02)
fig.tight_layout()
fig.savefig('fig1_interp_ablation.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ===============================================================
# 실험 (b): 추론 지연 주입 + 청크 이어붙이기 (RTC 미니어처)
# ===============================================================
# 가짜 VLA는 시계가 없다: 관측된 EEF 위치에서 경로 위상 s_obs 를 추정하고,
# 그 위상부터의 웨이포인트 H개를 내놓는다. 관측이 L 만큼 묵으면 청크는
# "이미 지나온 길"부터 다시 시작한다 — naive 교체는 그 낡은 앞부분을 그대로
# 실행해 로봇을 뒤로 홱 당기고(명령 점프), 위상 지연이 청크마다 누적된다.
# freeze 는 지연 구간에 해당하는 앞 n_L 개를 버리고 위상을 맞춰 이어붙인다
# (RTC 의 "추론 중 실행될 앞부분 동결"과 등가인 미니어처).
Delta, H, T_c = 0.1, 14, 1.0
s_grid_step = 1e-3

def estimate_phase(x_obs, s_center):
    """관측 EEF에서 경로 위상 추정 (예상 위상 ±0.4s 창에서 최근접점)."""
    s_c = np.arange(max(0.0, s_center - 0.4), s_center + 0.4, s_grid_step)
    d = np.linalg.norm(ref(s_c) - x_obs[None, :], axis=1)
    return s_c[np.argmin(d)]

def run_latency(L, strategy):
    n_L = int(np.ceil(L / Delta - 1e-9))
    q, qdv = q_wp[0].copy(), np.zeros(2)
    eef_hist = np.zeros((len(ts), 2))
    Q = np.zeros((len(ts), 2)); TAU = np.zeros((len(ts), 2))
    CMD = np.zeros((len(ts), 2))
    spline, cmd_jumps, s_prev = None, [], 0.0
    for i in range(len(ts)):
        t = ts[i]
        eef_hist[i] = fk(q)
        if abs(t / T_c - round(t / T_c)) < dt / 2 and round(t / T_c) < 5:  # 청크 경계
            k = int(round(t / T_c)); t_k = k * T_c
            if k == 0:
                s_obs = 0.0
            else:
                i_obs = max(0, i - int(round(L / dt)))          # L 만큼 묵은 관측
                s_exp = s_prev + (T_c - L)                       # 예상 위상
                s_obs = estimate_phase(eef_hist[i_obs], s_exp)
            shift = n_L if (strategy == 'freeze' and k > 0) else 0
            phases = s_obs + (shift + np.arange(H)) * Delta      # freeze: 낡은 앞부분 스킵
            wp_new = ref(phases)
            t_nodes = t_k + np.arange(H) * Delta
            new_spline = CubicSpline(t_nodes, np.array([ik2r(p) for p in wp_new]), axis=0)
            if spline is not None:                               # 명령 불연속 측정
                jump = np.linalg.norm(fk(new_spline(t_k)) - fk(spline(t_k))) * 1000
                cmd_jumps.append(jump)
            spline, s_prev = new_spline, s_obs + shift * Delta
        qd_c, vd_c, ad_c = spline(t), spline(t, 1), spline(t, 2)
        e, ed = qd_c - q, vd_c - qdv
        tau = M_mat(q) @ (ad_c + Kd * ed + Kp * e) + C_mat(q, qdv) @ qdv + g_vec(q)
        Q[i] = q; TAU[i] = tau; CMD[i] = qd_c
        q, qdv = rk4(q, qdv, tau, no_force, dt)
    eef = np.array([fk(qq) for qq in Q])
    err_mm = np.linalg.norm(eef - ref_fine, axis=1) * 1000
    w = ts >= 0.5
    return dict(TAU=TAU, eef=eef, err=err_mm, cmd=CMD,
                rms=np.sqrt(np.mean(err_mm[w]**2)), peak_tau=np.abs(TAU[w]).max(),
                max_jump=max(cmd_jumps) if cmd_jumps else 0.0)

print("=" * 60)
print("실험 (b): 추론 지연 L 주입 — naive 교체 vs 동결(RTC 미니어처)")
L_list = [0.0, 0.1, 0.2, 0.3]
res_b = {s: [] for s in ['naive', 'freeze']}
for L in L_list:
    for s in ['naive', 'freeze']:
        r = run_latency(L, s)
        res_b[s].append(r)
        print(f"  L={int(L * 1000):3d}ms {s:6s}: 최대 명령 점프 {r['max_jump']:6.1f} mm | "
              f"RMS 오차 {r['rms']:7.2f} mm | 피크 |tau| {r['peak_tau']:6.1f} N·m")

# 그림 2
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
iL = L_list.index(0.3)
zoom = (ts >= 0.6) & (ts <= 3.4)
cmd_n = np.array([fk(qq) for qq in res_b['naive'][iL]['cmd'][zoom]])
cmd_f = np.array([fk(qq) for qq in res_b['freeze'][iL]['cmd'][zoom]])
ax.plot(ts[zoom], ref_fine[zoom, 1], 'k--', lw=1.2, label='정책 의도 (정시 스케줄)')
ax.plot(ts[zoom], cmd_n[:, 1], 'C3', lw=1.2, label='명령 (naive 교체)')
ax.plot(ts[zoom], cmd_f[:, 1], 'C0', lw=1.2, label='명령 (동결)')
for tb in [1, 2, 3]:
    ax.axvline(tb, color='k', lw=0.6, ls=':')
ax.set_xlabel('t [s]'); ax.set_ylabel('EEF y 명령 [m]')
ax.set_title('(a) L=300ms: 경계마다 과거를 다시 산다')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
Lms = [int(L * 1000) for L in L_list]
ax.plot(Lms, [r['max_jump'] for r in res_b['naive']], 'C3o-', label='naive 교체')
ax.plot(Lms, [r['max_jump'] for r in res_b['freeze']], 'C0s-', label='동결')
ax.set_xlabel('추론 지연 L [ms]'); ax.set_ylabel('최대 명령 불연속 [mm]')
ax.set_title('(b) 청크 경계의 명령 점프'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[2]
ax.plot(Lms, [r['rms'] for r in res_b['naive']], 'C3o-', label='naive 교체')
ax.plot(Lms, [r['rms'] for r in res_b['freeze']], 'C0s-', label='동결')
ax.set_xlabel('추론 지연 L [ms]'); ax.set_ylabel('RMS EEF 오차 [mm]')
ax.set_title('(c) 성능 vs 지연 — 동결은 평평하다'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle('실험 (b) — 지연 주입과 청크 이어붙이기 (RTC의 미니어처: 지연 구간 동결)', y=1.02)
fig.tight_layout()
fig.savefig('fig2_latency_freeze.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ===============================================================
# 실험 (c): 벽 닦기 — 위치 직결 vs 임피던스 하위층
# ===============================================================
x_wall, k_e = 1.30, 1.0e4          # 벽 위치·강성 (R21과 동일 k_e)

def wall_force(qq, vv):
    p = fk(qq)
    pen = p[0] - x_wall
    if pen <= 0:
        return np.zeros(2)
    vx = (jac(qq) @ vv)[0]
    return np.array([-k_e * pen - 2.0 * vx, 0.0])   # 스프링 + 소량 접촉 감쇠

def minjerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5

def wipe_ref(t, x_cmd):
    """0~1s: 접근 (x: 1.20 → x_cmd), 1~4s: y 훑기 (+0.35 → -0.35)."""
    x = 1.20 + (x_cmd - 1.20) * minjerk(t / 1.0)
    y = 0.35 - 0.70 * minjerk((t - 1.0) / 3.0)
    return np.array([x, y])

def run_wipe(Kx, Bx, d_press, d_err, T=4.0):
    x_cmd = (x_wall + d_err) + d_press           # 정책이 믿는 벽 + 누름 명령
    n = int(T / dt)
    q, qdv = ik2r(np.array([1.20, 0.35])), np.zeros(2)
    F_hist = np.zeros(n)
    for i in range(n):
        t = i * dt
        xd = wipe_ref(t, x_cmd)
        xd_dot = (wipe_ref(t + dt, x_cmd) - wipe_ref(max(t - dt, 0.0), x_cmd)) / (2 * dt)
        p, v = fk(q), jac(q) @ qdv
        tau = jac(q).T @ (Kx * (xd - p) + Bx * (xd_dot - v)) + g_vec(q)
        F_hist[i] = k_e * max(p[0] - x_wall, 0.0)   # 벽 수직력
        q, qdv = rk4(q, qdv, tau, wall_force, dt)
    w = (np.arange(n) * dt >= 1.3)
    Fw = F_hist[w]
    return dict(F=F_hist, F_mean=Fw.mean(), F_peak=Fw.max(),
                contact=np.mean(Fw > 0.1))

CTRL = {'stiff': dict(Kx=2.0e4, Bx=400.0, d_press=0.005, label='위치 직결 (K=2×10⁴)'),
        'imp':   dict(Kx=200.0, Bx=40.0,  d_press=0.030, label='임피던스 (K_d=200)')}
F_MAX, C_MIN = 50.0, 0.9

print("=" * 60)
print("실험 (c): 벽 닦기 — 벽 위치 오차 스윕 (성공: F_peak ≤ 50 N & 접촉 유지 ≥ 90%)")
d_errs = np.arange(-0.020, 0.0201, 0.0025)
res_c = {}
for key, c in CTRL.items():
    rows = []
    for d in d_errs:
        r = run_wipe(c['Kx'], c['Bx'], c['d_press'], d)
        r['ok'] = (r['F_peak'] <= F_MAX) and (r['contact'] >= C_MIN)
        rows.append(r)
    res_c[key] = rows
    sub = [r for d, r in zip(d_errs, rows) if abs(d) <= 0.0151]
    rate = np.mean([r['ok'] for r in sub]) * 100
    i0 = np.argmin(np.abs(d_errs))
    print(f"  {c['label']:22s}: 성공률(±15mm) {rate:5.1f}% | δ_err=0: F_mean {rows[i0]['F_mean']:7.2f} N"
          f" (해석 {c['Kx'] * k_e / (c['Kx'] + k_e) * c['d_press']:7.2f} N)")
    for d, r in zip(d_errs, rows):
        if np.isclose(np.abs(d), [0.0, 0.010, 0.015], atol=1e-6).any():
            print(f"      δ_err={d * 1000:+5.1f}mm  F_mean {r['F_mean']:7.2f} N  F_peak {r['F_peak']:7.2f} N"
                  f"  접촉율 {r['contact'] * 100:5.1f}%  {'성공' if r['ok'] else '실패'}")

# 그림 3
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
tt = np.arange(len(res_c['stiff'][0]['F'])) * dt
i10 = int(np.argmin(np.abs(d_errs - 0.010)))
ax.plot(tt, res_c['stiff'][i10]['F'], 'C3', lw=1.2, label=CTRL['stiff']['label'])
ax.plot(tt, res_c['imp'][i10]['F'], 'C0', lw=1.2, label=CTRL['imp']['label'])
ax.axhline(F_MAX, color='k', ls='--', lw=1, label='허용 한계 50 N')
ax.set_xlabel('t [s]'); ax.set_ylabel('벽 수직력 [N]')
ax.set_title('(a) 접촉력 시계열 (δ_err = +10 mm)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
dmm = d_errs * 1000
for key, c, col in [('stiff', CTRL['stiff'], 'C3'), ('imp', CTRL['imp'], 'C0')]:
    Fm = [r['F_mean'] for r in res_c[key]]
    ok = np.array([r['ok'] for r in res_c[key]])
    ax.plot(dmm, Fm, col + 'o-', ms=4, lw=1.4, label=c['label'])
    ks = c['Kx'] * k_e / (c['Kx'] + k_e)
    ax.plot(dmm, np.maximum(ks * (c['d_press'] + d_errs), 0), col, ls=':', lw=1,
            label=f"해석 $k_s\\,\\delta$ ($k_s$={ks:.0f})")
    ax.plot(dmm[ok], np.array(Fm)[ok], col + 'o', ms=9, mfc='none')
ax.axhline(F_MAX, color='k', ls='--', lw=1)
ax.axhline(0.0, color='k', lw=0.5)
ax.set_yscale('symlog', linthresh=1.0)
ax.set_xlabel('벽 위치 오차 δ_err [mm] (정책의 착각)'); ax.set_ylabel('평균 접촉력 [N]')
ax.set_title('(b) 오차 스윕 — 큰 원 = 성공')
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.suptitle('실험 (c) — 같은 위치 오차가 강성 $k_s$에 비례한 힘 오차가 된다 (E2의 실증)', y=1.02)
fig.tight_layout()
fig.savefig('fig3_wall_wiping.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ===============================================================
# 그림 4: 지연 예산 파이 차트 (π0 배치 시나리오)
# ===============================================================
budget = [('관측 (카메라 30 fps 최악 신선도)', 33.3, 'C0'),
          ('추론 (π0 청크 생성, RTX 4090)', 73.0, 'C3'),
          ('네트워크 (유선 LAN 왕복, 가정)', 20.0, 'C1'),
          ('보간·큐 처리', 2.0, 'C2')]
total = sum(b[1] for b in budget)
freeze_ms = 73.0 + 20.0
print("=" * 60)
print(f"지연 예산 합계: {total:.1f} ms = 50Hz 액션 {total / 20:.1f}개 | "
      f"동결 필요(추론+네트워크) {freeze_ms:.0f} ms → ceil = {int(np.ceil(freeze_ms / 20))} 액션")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={'width_ratios': [1, 1.3]})
ax = axes[0]
ax.pie([b[1] for b in budget], labels=[f"{b[0]}\n{b[1]:.0f} ms" for b in budget],
       colors=[b[2] for b in budget], autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 8}, pctdistance=0.75)
ax.set_title(f'(a) 관측→실행 지연 예산: 합계 ≈ {total:.0f} ms')
ax = axes[1]
y0 = 0
ax.barh(1, 1000, color='lightgray', edgecolor='k', height=0.5)
left = 0
for name, v, c in budget:
    ax.barh(0, v, left=left, color=c, edgecolor='white', height=0.5)
    left += v
ax.text(1000 / 2, 1, 'π0 청크 1개 = H=50 @50Hz = 1000 ms', ha='center', va='center', fontsize=9)
ax.text(total + 20, 0, f'← 예산 {total:.0f} ms = 청크의 {total / 10:.0f}%', va='center', fontsize=9)
ax.axvline(freeze_ms, color='k', ls=':', lw=1)
ax.text(freeze_ms + 12, 0.5, f'동결 필요 구간(추론+네트워크) {freeze_ms:.0f} ms ≈ 액션 5개', fontsize=8, va='center')
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(['지연 예산', '청크 길이'])
ax.set_xlabel('시간 [ms]'); ax.set_xlim(0, 1050)
ax.set_title('(b) 예산 vs 청크 — 개루프 구간이 지연을 흡수한다')
fig.tight_layout()
fig.savefig('fig4_delay_budget.png', dpi=140, bbox_inches='tight')
plt.close(fig)

print("완료: fig1~fig4 저장")
