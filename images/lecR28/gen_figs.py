# Lec R28 그림 생성 + 본문 수치 검증 스크립트
# 실행: python3 gen_figs.py  (numpy / scipy 1.15 / matplotlib)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as Rot

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = os.path.dirname(os.path.abspath(__file__))
C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GRAY = '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#7f7f7f'

# ============================================================
# 0. 2R 동역학 (R10과 동일 파라미터) + 회귀자 Y
# ============================================================
m1, m2, l1, lc1, lc2 = 1.0, 1.0, 1.0, 0.5, 0.5
I1 = I2 = 1.0 / 12
G = 9.81
TH_TRUE = np.array([I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2),   # th1
                    m2 * l1 * lc2,                                   # th2
                    I2 + m2 * lc2**2,                                # th3
                    m1 * lc1 + m2 * l1,                              # th4
                    m2 * lc2])                                       # th5
print("[0] 참 base parameters θ* =", np.round(TH_TRUE, 5))


def M_of(th, q):
    c2 = np.cos(q[1])
    return np.array([[th[0] + 2 * th[1] * c2, th[2] + th[1] * c2],
                     [th[2] + th[1] * c2, th[2]]])


def C_of(th, q, qd):
    h = th[1] * np.sin(q[1])
    return np.array([[-h * qd[1], -h * (qd[0] + qd[1])], [h * qd[0], 0.0]])


def g_of(th, q):
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    return G * np.array([th[3] * c1 + th[4] * c12, th[4] * c12])


def tau_of(th, q, qd, qdd):
    return M_of(th, q) @ qdd + C_of(th, q, qd) @ qd + g_of(th, q)


def regressor(q, qd, qdd):
    c1, c2, s2, c12 = np.cos(q[0]), np.cos(q[1]), np.sin(q[1]), np.cos(q[0] + q[1])
    return np.array([
        [qdd[0], c2 * (2 * qdd[0] + qdd[1]) - s2 * (qd[1]**2 + 2 * qd[0] * qd[1]),
         qdd[1], G * c1, G * c12],
        [0.0, c2 * qdd[0] + s2 * qd[0]**2, qdd[0] + qdd[1], 0.0, G * c12]])


# 검증: 랜덤 상태에서 Yθ = Mq̈+Cq̇+g
rng = np.random.default_rng(0)
err = max(np.abs(regressor(q, qd, qdd) @ TH_TRUE - tau_of(TH_TRUE, q, qd, qdd)).max()
          for q, qd, qdd in [(rng.uniform(-np.pi, np.pi, 2), rng.uniform(-3, 3, 2),
                              rng.uniform(-5, 5, 2)) for _ in range(200)])
print(f"[0] 랜덤 200 상태에서 max|Yθ* - (Mq̈+Cq̇+g)| = {err:.2e}")

# ============================================================
# WE-1: 손계산 상태
# ============================================================
q, qd, qdd = np.array([0.0, np.pi / 2]), np.array([1.0, 2.0]), np.array([1.0, 0.0])
Y1 = regressor(q, qd, qdd)
print("\n[WE-1] q=(0, π/2), q̇=(1,2), q̈=(1,0)")
print("Y =\n", np.round(Y1, 4))
print("Yθ* =", np.round(Y1 @ TH_TRUE, 4), " / M q̈+C q̇+g =", np.round(tau_of(TH_TRUE, q, qd, qdd), 4))


# ============================================================
# WE-2: 여기 궤적, 최소자승, 조건수
# ============================================================
def multisine(T, N, f0, H, rng, amp=0.6, same_phase=False):
    """멀티사인 여기 궤적. 반환: t, q, qd, qdd (N×2). 해석적 미분."""
    t = np.linspace(0, T, N)
    q = np.zeros((N, 2))
    qd = np.zeros((N, 2))
    qdd = np.zeros((N, 2))
    for j in range(2):
        for k in range(1, H + 1):
            w = 2 * np.pi * k * f0
            ph = 0.2 * j if same_phase else rng.uniform(0, 2 * np.pi)
            a = amp / k
            q[:, j] += a * np.sin(w * t + ph)
            qd[:, j] += a * w * np.cos(w * t + ph)
            qdd[:, j] += -a * w**2 * np.sin(w * t + ph)
    return t, q, qd, qdd


def stack_Y_tau(q, qd, qdd, sigma, rng):
    N = len(q)
    Y = np.vstack([regressor(q[i], qd[i], qdd[i]) for i in range(N)])
    tau = np.hstack([tau_of(TH_TRUE, q[i], qd[i], qdd[i]) for i in range(N)])
    return Y, tau + rng.normal(0, sigma, 2 * N)


SIG = 0.1  # 토크 잡음 [N·m]
N_S = 1000

# 나쁜 여기: 0.05 Hz 단일 사인, 두 관절 거의 동상
rng = np.random.default_rng(1)
_, qb, qdb, qddb = multisine(20.0, N_S, 0.05, 1, rng, amp=0.7, same_phase=True)
# 좋은 여기: 0.5 Hz 기본파 + 5 고조파, 무작위 위상
rng = np.random.default_rng(2)
_, qg, qdg, qddg = multisine(10.0, N_S, 0.5, 5, rng, amp=0.6)

Yb, _ = stack_Y_tau(qb, qdb, qddb, 0, np.random.default_rng(0))
Yg, _ = stack_Y_tau(qg, qdg, qddg, 0, np.random.default_rng(0))
kb, kg = np.linalg.cond(Yb), np.linalg.cond(Yg)
print(f"\n[WE-2] κ(Y) 나쁜 여기 = {kb:.1f} / 좋은 여기 = {kg:.2f}")
print(f"[WE-2] 속도 범위: 나쁜 max|q̇|={np.abs(qdb).max():.3f}, 좋은 max|q̇|={np.abs(qdg).max():.2f} rad/s")

# 100회 잡음 실현에서 파라미터별 상대오차
REL_B, REL_G = [], []
for trial in range(100):
    r = np.random.default_rng(100 + trial)
    _, taub = stack_Y_tau(qb, qdb, qddb, SIG, r)
    _, taug = stack_Y_tau(qg, qdg, qddg, SIG, r)
    thb = np.linalg.lstsq(Yb, taub, rcond=None)[0]
    thg = np.linalg.lstsq(Yg, taug, rcond=None)[0]
    REL_B.append(np.abs(thb - TH_TRUE) / np.abs(TH_TRUE))
    REL_G.append(np.abs(thg - TH_TRUE) / np.abs(TH_TRUE))
REL_B, REL_G = np.array(REL_B), np.array(REL_G)
print("[WE-2] 파라미터별 평균 상대오차(%) 나쁜 여기:", np.round(REL_B.mean(0) * 100, 2))
print("[WE-2] 파라미터별 평균 상대오차(%) 좋은 여기:", np.round(REL_G.mean(0) * 100, 3))
tot_b = np.mean([np.linalg.norm(r * np.abs(TH_TRUE)) / np.linalg.norm(TH_TRUE) for r in REL_B])
tot_g = np.mean([np.linalg.norm(r * np.abs(TH_TRUE)) / np.linalg.norm(TH_TRUE) for r in REL_G])
print(f"[WE-2] 전체 상대오차 ‖θ̂-θ*‖/‖θ*‖: 나쁜 {tot_b*100:.2f}% / 좋은 {tot_g*100:.3f}%")

# 이론 예측: cov(θ̂) = σ²(YᵀY)⁻¹
pred_b = SIG * np.sqrt(np.trace(np.linalg.inv(Yb.T @ Yb))) / np.linalg.norm(TH_TRUE)
pred_g = SIG * np.sqrt(np.trace(np.linalg.inv(Yg.T @ Yg))) / np.linalg.norm(TH_TRUE)
print(f"[WE-2] 이론 예측 E‖θ̂-θ*‖/‖θ*‖ ≈ 나쁜 {pred_b*100:.2f}% / 좋은 {pred_g*100:.3f}%")

# 대표 1회(seed 100)로 훈련/검증 잔차
r = np.random.default_rng(100)
_, taub = stack_Y_tau(qb, qdb, qddb, SIG, r)
_, taug = stack_Y_tau(qg, qdg, qddg, SIG, r)
th_b = np.linalg.lstsq(Yb, taub, rcond=None)[0]
th_g = np.linalg.lstsq(Yg, taug, rcond=None)[0]
rms_train_b = np.sqrt(np.mean((Yb @ th_b - taub)**2))
rms_train_g = np.sqrt(np.mean((Yg @ th_g - taug)**2))

# 검증 궤적 (다른 주파수·위상)
rng = np.random.default_rng(7)
tv, qv, qdv, qddv = multisine(10.0, N_S, 0.4, 4, rng, amp=0.6)
Yv, tauv = stack_Y_tau(qv, qdv, qddv, SIG, np.random.default_rng(500))
rms_val_b = np.sqrt(np.mean((Yv @ th_b - tauv)**2))
rms_val_g = np.sqrt(np.mean((Yv @ th_g - tauv)**2))
print(f"[WE-2] 훈련 잔차 RMS: 나쁜 {rms_train_b:.4f} / 좋은 {rms_train_g:.4f} N·m (σ={SIG})")
print(f"[WE-2] 검증 잔차 RMS: 나쁜 여기 θ̂ → {rms_val_b:.3f} / 좋은 여기 θ̂ → {rms_val_g:.4f} N·m")
print(f"[WE-2] 나쁜 여기 θ̂ = {np.round(th_b, 3)}")
print(f"[WE-2] 좋은 여기 θ̂ = {np.round(th_g, 4)}")

# ---------- 주파수 스윕 (fig3) ----------
f0s = np.logspace(np.log10(0.02), np.log10(2.0), 13)
KAPPA, ERR, PRED = [], [], []
for f0 in f0s:
    rng = np.random.default_rng(2)  # 같은 형태의 궤적, 속도만 바꿈
    T = max(10.0, 2.0 / f0)
    _, qs, qds, qdds = multisine(T, N_S, f0, 5, rng, amp=0.6)
    Ys, _ = stack_Y_tau(qs, qds, qdds, 0, np.random.default_rng(0))
    KAPPA.append(np.linalg.cond(Ys))
    PRED.append(SIG * np.sqrt(np.trace(np.linalg.inv(Ys.T @ Ys))) / np.linalg.norm(TH_TRUE))
    errs = []
    for trial in range(30):
        r = np.random.default_rng(1000 + trial)
        _, taus = stack_Y_tau(qs, qds, qdds, SIG, r)
        ths = np.linalg.lstsq(Ys, taus, rcond=None)[0]
        errs.append(np.linalg.norm(ths - TH_TRUE) / np.linalg.norm(TH_TRUE))
    ERR.append(np.mean(errs))
KAPPA, ERR, PRED = np.array(KAPPA), np.array(ERR), np.array(PRED)
i_best = np.argmin(KAPPA)
print(f"\n[스윕] κ 최소 = {KAPPA.min():.2f} @ f0 = {f0s[i_best]:.2f} Hz (오차 {ERR[i_best]*100:.3f}%)")
print(f"[스윕] 최저속 f0={f0s[0]:.2f} Hz: κ = {KAPPA[0]:.0f}, 오차 {ERR[0]*100:.1f}%")
print(f"[스윕] 최고속 f0={f0s[-1]:.2f} Hz: κ = {KAPPA[-1]:.1f}, 오차 {ERR[-1]*100:.3f}%")

# ============================================================
# WE-3: 마찰 피팅
# ============================================================
Fc_t, Fv_t, Fs_t, vs_t = 0.8, 0.3, 1.2, 0.05


def fric_true(v):
    return (Fc_t + (Fs_t - Fc_t) * np.exp(-(v / vs_t)**2)) * np.sign(v) + Fv_t * v


rng = np.random.default_rng(3)
v_pos = np.logspace(np.log10(0.005), np.log10(3.0), 30)
v_dat = np.hstack([-v_pos[::-1], v_pos])
tau_f = fric_true(v_dat) + rng.normal(0, 0.03, len(v_dat))

# (a) 쿨롱+점성 선형 최소자승 (전 구간)
A_cv = np.column_stack([np.sign(v_dat), v_dat])
cv_all = np.linalg.lstsq(A_cv, tau_f, rcond=None)[0]
res_all = np.sqrt(np.mean((A_cv @ cv_all - tau_f)**2))
# (b) |v| ≥ 0.3만 사용
mask = np.abs(v_dat) >= 0.3
cv_hi = np.linalg.lstsq(A_cv[mask], tau_f[mask], rcond=None)[0]
res_hi_all = np.sqrt(np.mean((A_cv @ cv_hi - tau_f)**2))
res_hi_own = np.sqrt(np.mean((A_cv[mask] @ cv_hi - tau_f[mask])**2))


# (c) Stribeck 비선형 피팅
def stribeck(v, Fc, Fv, Fs, vs):
    return (Fc + (Fs - Fc) * np.exp(-(v / vs)**2)) * np.sign(v) + Fv * v


p_st, _ = curve_fit(stribeck, v_dat, tau_f, p0=[0.5, 0.5, 1.0, 0.1])
res_st = np.sqrt(np.mean((stribeck(v_dat, *p_st) - tau_f)**2))
print(f"\n[WE-3] 참값: Fc={Fc_t}, Fv={Fv_t}, Fs={Fs_t}, vs={vs_t}")
print(f"[WE-3] C+V 전구간:   F̂c={cv_all[0]:.3f}, F̂v={cv_all[1]:.3f}, RMS 잔차={res_all:.4f}")
print(f"[WE-3] C+V |v|≥0.3: F̂c={cv_hi[0]:.3f}, F̂v={cv_hi[1]:.3f}, 자기구간 RMS={res_hi_own:.4f}, 전구간 RMS={res_hi_all:.4f}")
print(f"[WE-3] Stribeck:    F̂c={p_st[0]:.3f}, F̂v={p_st[1]:.3f}, F̂s={p_st[2]:.3f}, v̂s={p_st[3]:.4f}, RMS={res_st:.4f}")
lowmask = np.abs(v_dat) < 0.15
print(f"[WE-3] 저속(|v|<0.15) 잔차 RMS: C+V 전구간 {np.sqrt(np.mean((A_cv[lowmask]@cv_all - tau_f[lowmask])**2)):.3f}"
      f" / Stribeck {np.sqrt(np.mean((stribeck(v_dat[lowmask],*p_st) - tau_f[lowmask])**2)):.4f}")

# ============================================================
# WE-4: hand-eye AX = XB
# ============================================================
R_X = Rot.from_euler('zx', [90, 12], degrees=True).as_matrix()
t_X = np.array([0.03, -0.06, 0.10])
T_X = np.eye(4); T_X[:3, :3] = R_X; T_X[:3, 3] = t_X
T_Xi = np.linalg.inv(T_X)


def rand_motion(rng, ax=None, angle_rng=(20, 60)):
    axis = ax if ax is not None else rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(*angle_rng))
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(ang * axis).as_matrix()
    T[:3, 3] = rng.uniform(-0.2, 0.2, 3)
    return T


def make_pairs(K, rng, s_rot=np.deg2rad(0.2), s_t=0.5e-3, parallel=False):
    As, Bs = [], []
    for _ in range(K):
        jitter = rng.normal(0, 0.01, 3)
        A = rand_motion(rng, ax=np.array([0, 0, 1.0]) + jitter if parallel else None)
        B = T_Xi @ A @ T_X
        # 카메라 관측 잡음
        B[:3, :3] = Rot.from_rotvec(rng.normal(0, s_rot, 3)).as_matrix() @ B[:3, :3]
        B[:3, 3] += rng.normal(0, s_t, 3)
        As.append(A); Bs.append(B)
    return As, Bs


def solve_handeye(As, Bs):
    # 회전: α_i = R_X β_i  →  Wahba/Procrustes (R02의 SVD 사영)
    al = np.array([Rot.from_matrix(A[:3, :3]).as_rotvec() for A in As])
    be = np.array([Rot.from_matrix(B[:3, :3]).as_rotvec() for B in Bs])
    H = be.T @ al                       # Σ β_i α_iᵀ  (3×3)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    # 병진: (R_A − I) t_X = R_X t_B − t_A
    L = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    rhs = np.hstack([R @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    t, *_ = np.linalg.lstsq(L, rhs, rcond=None)
    return R, t, np.linalg.svd(L, compute_uv=False), S


# 무잡음 검증
As0, Bs0 = make_pairs(10, np.random.default_rng(4), s_rot=0, s_t=0)
R0, t0, _, _ = solve_handeye(As0, Bs0)
print(f"\n[WE-4] 무잡음 K=10: 회전오차 {np.rad2deg(np.linalg.norm(Rot.from_matrix(R0 @ R_X.T).as_rotvec())):.2e}°, "
      f"병진오차 {np.linalg.norm(t0 - t_X)*1e3:.2e} mm")

# 잡음 있는 대표 케이스 K=10
As1, Bs1 = make_pairs(10, np.random.default_rng(5))
R1, t1, svL, svH = solve_handeye(As1, Bs1)
e_rot1 = np.rad2deg(np.linalg.norm(Rot.from_matrix(R1 @ R_X.T).as_rotvec()))
e_t1 = np.linalg.norm(t1 - t_X) * 1e3
print(f"[WE-4] 잡음(0.2°, 0.5mm) K=10: 회전오차 {e_rot1:.3f}°, 병진오차 {e_t1:.2f} mm, σ_min(L)={svL[-1]:.2f}")

# 퇴화: 회전축 전부 z 근방
As2, Bs2 = make_pairs(10, np.random.default_rng(6), parallel=True)
R2, t2, svL2, svH2 = solve_handeye(As2, Bs2)
e_rot2 = np.rad2deg(np.linalg.norm(Rot.from_matrix(R2 @ R_X.T).as_rotvec()))
e_t2 = np.linalg.norm(t2 - t_X) * 1e3
print(f"[WE-4] 퇴화(축 평행) K=10: 회전오차 {e_rot2:.1f}°, 병진오차 {e_t2:.1f} mm, "
      f"σ_min(L)={svL2[-1]:.4f}, σ(H)={np.round(svH2, 3)}")

# K 스윕
Ks = np.array([3, 5, 8, 12, 20, 30, 50])
EROT, ET = [], []
for K in Ks:
    er, et = [], []
    for trial in range(50):
        As_, Bs_ = make_pairs(K, np.random.default_rng(9000 + 37 * K + trial))
        R_, t_, _, _ = solve_handeye(As_, Bs_)
        er.append(np.rad2deg(np.linalg.norm(Rot.from_matrix(R_ @ R_X.T).as_rotvec())))
        et.append(np.linalg.norm(t_ - t_X) * 1e3)
    EROT.append(np.median(er)); ET.append(np.median(et))
EROT, ET = np.array(EROT), np.array(ET)
print(f"[WE-4] K 스윕 중앙값: K=3 → {EROT[0]:.3f}°/{ET[0]:.2f}mm, K=50 → {EROT[-1]:.3f}°/{ET[-1]:.2f}mm")

# ============================================================
# 그림 1 (한 장 요약): 마찰 곡선 + 여기별 파라미터 오차
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
vv = np.linspace(-3, 3, 2000); vv = vv[np.abs(vv) > 1e-4]
ax.plot(v_dat, tau_f, '.', color=C_GRAY, ms=5, label='측정 (정속 실험 + 잡음)')
ax.plot(vv, fric_true(vv), '-', color='k', lw=1.0, alpha=0.5, label='참 마찰 (Stribeck 포함)')
ax.plot(vv, cv_all[0] * np.sign(vv) + cv_all[1] * vv, '-', color=C_RED, lw=2,
        label=f'쿨롱+점성 적합 (Fc={cv_all[0]:.2f}, Fv={cv_all[1]:.2f})')
ax.set_xlabel('관절 속도 $\\dot q$ [rad/s]'); ax.set_ylabel('마찰 토크 $\\tau_f$ [N·m]')
ax.set_title('(a) 구조를 고정하고 파라미터를 회귀한다')
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
axins = ax.inset_axes([0.06, 0.55, 0.36, 0.4])
mask_in = np.abs(v_dat) < 0.25
axins.plot(v_dat[mask_in], tau_f[mask_in], '.', color=C_GRAY, ms=4)
vvi = np.linspace(-0.25, 0.25, 800); vvi = vvi[np.abs(vvi) > 1e-4]
axins.plot(vvi, fric_true(vvi), '-', color='k', lw=1.0, alpha=0.5)
axins.plot(vvi, cv_all[0] * np.sign(vvi) + cv_all[1] * vvi, '-', color=C_RED, lw=1.5)
axins.set_title('저속 확대: 구조가 틀리면 남는 오차', fontsize=8)
axins.tick_params(labelsize=7); axins.grid(alpha=0.3)

ax = axes[1]
x = np.arange(5); w = 0.38
labels = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$']
ax.bar(x - w / 2, REL_B.mean(0) * 100, w, color=C_RED, alpha=0.85,
       label=f'나쁜 여기 (κ={kb:.0f})')
ax.bar(x + w / 2, REL_G.mean(0) * 100, w, color=C_BLUE, alpha=0.85,
       label=f'좋은 여기 (κ={kg:.1f})')
ax.set_yscale('log'); ax.set_xticks(x, labels)
ax.set_ylabel('파라미터 상대오차 [%] (잡음 100회 평균)')
ax.set_title('(b) 데이터(여기)가 나쁘면 회귀가 무너진다')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
fig.suptitle('고전판 학습 = 물리가 정한 구조(모델) × 잘 설계한 데이터(여기) × 최소자승', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT}/fig1_classical_learning.png', dpi=140)
plt.close(fig)

# ============================================================
# 그림 2: 회귀 적합 — 훈련은 둘 다 좋고, 검증에서 갈린다
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
seg = slice(0, 300)
tau1_v = tauv[0::2]  # 관절 1 토크
ax.plot(tv[seg], tau1_v[seg], '.', color=C_GRAY, ms=3, label='측정 $\\tau_1$ (검증 궤적)')
ax.plot(tv[seg], (Yv @ th_g)[0::2][seg], '-', color=C_BLUE, lw=1.5,
        label=f'좋은 여기로 추정한 θ 예측 (RMS {rms_val_g:.2f})')
ax.plot(tv[seg], (Yv @ th_b)[0::2][seg], '-', color=C_RED, lw=1.2, alpha=0.9,
        label=f'나쁜 여기로 추정한 θ 예측 (RMS {rms_val_b:.2f})')
ax.set_xlabel('t [s]'); ax.set_ylabel('$\\tau_1$ [N·m]')
ax.set_title('(a) 검증 궤적에서의 토크 예측')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
bars = ['나쁜 여기\n훈련', '나쁜 여기\n검증', '좋은 여기\n훈련', '좋은 여기\n검증']
vals = [rms_train_b, rms_val_b, rms_train_g, rms_val_g]
cols = [C_RED, C_RED, C_BLUE, C_BLUE]
b = ax.bar(bars, vals, color=cols)
for bi, a in zip(b, [0.5, 0.95, 0.5, 0.95]):
    bi.set_alpha(a)
ax.axhline(SIG, color='k', ls='--', lw=1, label=f'잡음 바닥 σ = {SIG}')
for bi, v in zip(b, vals):
    ax.text(bi.get_x() + bi.get_width() / 2, v * 1.05, f'{v:.3f}', ha='center', fontsize=9)
ax.set_ylabel('토크 잔차 RMS [N·m]'); ax.set_yscale('log')
ax.set_title('(b) 훈련 잔차는 못 가른다 — 검증이 가른다')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT}/fig2_fit_validation.png', dpi=140)
plt.close(fig)

# ============================================================
# 그림 3: 여기 속도 스윕 — 조건수가 정확도를 결정
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.loglog(f0s, KAPPA, 'o-', color=C_GREEN)
ax.axvline(f0s[i_best], color=C_GRAY, ls=':')
ax.annotate('너무 느림:\n중력만 보인다\n(관성 열 ≈ 0)', xy=(f0s[1], KAPPA[1]), xytext=(0.05, 200),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=C_GRAY))
ax.annotate('너무 빠름:\n관성이 중력 열을\n압도', xy=(f0s[-1], KAPPA[-1]), xytext=(0.5, 300),
            fontsize=8, arrowprops=dict(arrowstyle='->', color=C_GRAY))
ax.set_xlabel('여기 기본 주파수 $f_0$ [Hz]'); ax.set_ylabel('κ(Y)')
ax.set_title('(a) 조건수: 빠르기의 U자 곡선'); ax.grid(alpha=0.3, which='both')
ax = axes[1]
ax.loglog(PRED * 100, ERR * 100, 'o', color=C_ORANGE, label='실측 (여기 속도별)')
pp = np.logspace(np.log10(PRED.min() * 100 * 0.7), np.log10(PRED.max() * 100 * 1.4), 50)
ax.loglog(pp, pp, '--', color=C_GRAY, label='이론 = 실측 (y = x)')
ax.set_xlabel(r'이론 예측  $\sigma\sqrt{\mathrm{tr}\,(Y^\top Y)^{-1}}\,/\,\|\theta^*\|$  [%]')
ax.set_ylabel('실측 평균 상대오차 [%] (30회)')
ax.set_title(r'(b) 오차는 $\mathrm{cov}(\hat\theta)=\sigma^2(Y^\top Y)^{-1}$이 예측한다')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_condition_sweep.png', dpi=140)
plt.close(fig)

# ============================================================
# 그림 4: 마찰 피팅 상세
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.plot(v_dat, tau_f, '.', color=C_GRAY, ms=5, label='측정')
ax.plot(vv, fric_true(vv), '-', color='k', lw=1, alpha=0.45, label='참')
ax.plot(vv, cv_all[0] * np.sign(vv) + cv_all[1] * vv, '-', color=C_RED, lw=1.8,
        label=f'쿨롱+점성 (RMS {res_all:.3f})')
ax.plot(vv, stribeck(vv, *p_st), '-', color=C_BLUE, lw=1.8,
        label=f'+Stribeck (RMS {res_st:.3f})')
ax.set_xlabel('$\\dot q$ [rad/s]'); ax.set_ylabel('$\\tau_f$ [N·m]')
ax.set_title('(a) 전 속도 구간: 둘 다 그럴듯해 보인다')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
m_ = np.abs(v_dat) < 0.4
ax.plot(v_dat[m_], tau_f[m_], '.', color=C_GRAY, ms=6)
vvz = np.linspace(-0.4, 0.4, 1200); vvz = vvz[np.abs(vvz) > 2e-3]
ax.plot(vvz, fric_true(vvz), '-', color='k', lw=1, alpha=0.45)
ax.plot(vvz, cv_all[0] * np.sign(vvz) + cv_all[1] * vvz, '-', color=C_RED, lw=1.8)
ax.plot(vvz, stribeck(vvz, *p_st), '-', color=C_BLUE, lw=1.8)
ax.annotate('Stribeck 언덕:\n정지 직전이 더 무겁다', xy=(0.05, fric_true(0.05)), xytext=(0.16, 0.45),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=C_GRAY))
ax.annotate('sgn 불연속:\n0 근처는 본질적으로 어렵다', xy=(0.0, 0.0), xytext=(-0.38, -0.6),
            fontsize=9, arrowprops=dict(arrowstyle='->', color=C_GRAY))
ax.set_xlabel('$\\dot q$ [rad/s]'); ax.set_ylabel('$\\tau_f$ [N·m]')
ax.set_title('(b) 저속 확대: 모델 구조의 차이가 드러나는 곳')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_friction_fit.png', dpi=140)
plt.close(fig)

# ============================================================
# 그림 5: hand-eye 기하 + 수렴
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
ax.set_aspect('equal'); ax.axis('off')
ax.set_xlim(-0.5, 5.2); ax.set_ylim(-0.4, 3.6)
# 베이스, 두 팔 자세, 카메라, 보드
ax.add_patch(plt.Rectangle((-0.3, -0.25), 1.0, 0.25, fc='#bbbbbb', ec='k'))
ax.text(0.2, -0.38, 'base {b}', ha='center', fontsize=9)
for (g, col, lab) in [((1.7, 1.9), C_BLUE, '자세 1'), ((2.9, 2.5), C_GREEN, '자세 2')]:
    ax.plot([0.2, 0.9 * g[0] / 2, g[0]], [0.0, 1.1 * g[1] / 2, g[1]], '-', color=col, lw=3,
            solid_capstyle='round', alpha=0.85)
    ax.plot(*g, 'o', color=col, ms=7)
    cam = (g[0] + 0.35, g[1] + 0.22)
    ax.add_patch(plt.Rectangle((cam[0] - 0.12, cam[1] - 0.09), 0.24, 0.18, fc=col, alpha=0.5, ec=col))
    ax.annotate('', xy=cam, xytext=g, arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
    ax.text(cam[0] + 0.13, cam[1] + 0.1, 'X', fontsize=11, fontweight='bold')
ax.text(1.55, 2.62, 'A: 그리퍼의 움직임\n(FK로 안다)', fontsize=9, color='k', ha='center')
ax.annotate('', xy=(2.9, 2.5), xytext=(1.7, 1.9),
            arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.6, ls='--',
                            connectionstyle='arc3,rad=-0.3'))
# 체커보드
bx, by = 4.3, 1.2
for i in range(4):
    for j in range(4):
        if (i + j) % 2 == 0:
            ax.add_patch(plt.Rectangle((bx + 0.16 * i, by + 0.16 * j), 0.16, 0.16, fc='k'))
ax.add_patch(plt.Rectangle((bx, by), 0.64, 0.64, fill=False, ec='k'))
ax.text(bx + 0.32, by - 0.18, '체커보드 {t} (고정)', ha='center', fontsize=9)
for (g, col) in [((1.7, 1.9), C_BLUE), ((2.9, 2.5), C_GREEN)]:
    cam = (g[0] + 0.35, g[1] + 0.22)
    ax.annotate('', xy=(bx + 0.32, by + 0.7), xytext=cam,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.0, alpha=0.6))
ax.text(4.45, 3.15, 'B: 카메라가 본\n보드의 움직임', fontsize=9, ha='center')
ax.text(2.2, 0.4, r'같은 강체 이동을 두 좌표계가 다르게 본다:  $A\,X = X\,B$', fontsize=11)
ax.set_title('(a) hand-eye 기하: X = 그리퍼→카메라 변환 (불변)')
ax = axes[1]
ax.loglog(Ks, EROT, 'o-', color=C_BLUE, label='회전 오차 [°]')
ax.loglog(Ks, ET, 's-', color=C_RED, label='병진 오차 [mm]')
ax.loglog(Ks, EROT[0] * np.sqrt(Ks[0] / Ks), '--', color=C_GRAY, lw=1, label=r'$1/\sqrt{K}$ 가이드')
ax.set_xlabel('자세 쌍 수 K'); ax.set_ylabel('오차 (50회 중앙값)')
ax.set_title(f'(b) 수렴: 잡음 0.2°/0.5mm, 비평행 회전축\n(퇴화: 축 평행이면 K=10에도 {e_t2:.0f} mm)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f'{OUT}/fig5_handeye.png', dpi=140)
plt.close(fig)

print("\n그림 5장 저장 완료:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
