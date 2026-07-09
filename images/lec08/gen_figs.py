# Lec R08 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = "/home/hjkim/frontier_ws/vla-study/images/lec08"

# ---------- 공용: 최소저크(5차 rest-to-rest) 형상 ----------
def mj_s(tau):            # smootherstep = 10t^3 - 15t^4 + 6t^5
    return 10*tau**3 - 15*tau**4 + 6*tau**5
def mj_sd(tau):
    return 30*tau**2 - 60*tau**3 + 30*tau**4
def mj_sdd(tau):
    return 60*tau - 180*tau**2 + 120*tau**3

# =====================================================================
# fig1 — 한 장 요약: 50Hz 액션 청크를 500Hz 셋포인트로 펴기 (ZOH/선형/3차)
# =====================================================================
T, dq = 1.0, 1.2
t50 = np.linspace(0, T, 51)             # 시작 상태 + 50개 액션 (50Hz)
q50 = dq*mj_s(t50/T)
t500 = np.arange(0, T, 0.002)           # 500Hz 셋포인트
zoh = q50[np.searchsorted(t50, t500, side='right') - 1]
lin = np.interp(t500, t50, q50)
cub = CubicSpline(t50, q50)(t500)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
# (a) 위치 확대창
m = (t500 >= 0.30) & (t500 <= 0.42)
ax[0].step(t500[m], zoh[m], where='post', color='tab:red', lw=1.8, label='ZOH (계단)')
ax[0].plot(t500[m], lin[m], color='tab:orange', lw=1.6, label='선형')
ax[0].plot(t500[m], cub[m], color='tab:blue', lw=1.6, label='3차 스플라인')
ax[0].plot(t50, q50, 'ko', ms=4.5, zorder=5, label='50Hz 액션 (청크)')
ax[0].set_xlim(0.30, 0.42)
ax[0].set_ylim(q50[15]-0.02, q50[21]+0.02)
ax[0].set_xlabel('시간 [s]'); ax[0].set_ylabel('관절각 [rad]')
ax[0].set_title('(a) 위치 — 확대. 선형·3차는 겹쳐 보인다')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].grid(alpha=0.3)

# (b) 함의 속도 (500Hz 유한차분)
for q, c, lb in [(zoh, 'tab:red', 'ZOH'), (lin, 'tab:orange', '선형'), (cub, 'tab:blue', '3차')]:
    v = np.diff(q)/0.002
    ax[1].plot(t500[1:], v, color=c, lw=1.2, label=lb)
ax[1].axhline(15*dq/(8*T), color='k', ls='--', lw=1, label='참값 피크 2.25')
ax[1].set_xlabel('시간 [s]'); ax[1].set_ylabel('함의 속도 [rad/s]')
ax[1].set_title('(b) 함의 속도 (Δq/2ms) — ZOH의 정체')
ax[1].legend(fontsize=8, loc='upper right'); ax[1].grid(alpha=0.3)

# (c) 함의 가속 — 선형 vs 3차 (ZOH는 축 밖)
for q, c, lb in [(lin, 'tab:orange', '선형'), (cub, 'tab:blue', '3차')]:
    acc = np.diff(q, 2)/0.002**2
    ax[2].plot(t500[2:], acc, color=c, lw=1.2, label=lb)
ax[2].set_xlabel('시간 [s]'); ax[2].set_ylabel('함의 가속 [rad/s²]')
ax[2].set_title('(c) 함의 가속 — 선형의 대가 (ZOH는 축 밖)')
ax[2].legend(fontsize=8, loc='upper right'); ax[2].grid(alpha=0.3)
fig.suptitle('같은 50Hz 청크, 세 가지 펴기 — 위치는 비슷해도 미분은 전혀 다르다', y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_chunk_to_setpoints.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# fig2 — 프로파일 3종: 사다리꼴 / S-커브 / 5차(최소저크), 위치·속도·가속·저크
# 공통 제약: L=0.8 rad, v_max=2 rad/s, a_max=8 rad/s², (S-커브) j_max=80 rad/s³
# =====================================================================
L, vm, am, jm = 0.8, 2.0, 8.0, 80.0
dt = 1e-4

# --- 사다리꼴 ---
ta, tc = vm/am, L/vm - vm/am
T_trap = 2*ta + tc
t1 = np.arange(0, T_trap, dt)
acc1 = np.where(t1 < ta, am, np.where(t1 < ta+tc, 0.0, -am))
v1 = np.cumsum(acc1)*dt
q1 = np.cumsum(v1)*dt

# --- S-커브 (7세그먼트) ---
tj = am/jm
Tacc = vm/am + tj
tc2 = (L - vm*Tacc)/vm
T_s = 2*Tacc + tc2
segs = [(jm, tj), (0, vm/am - tj), (-jm, tj), (0, tc2),
        (-jm, tj), (0, vm/am - tj), (jm, tj)]
t2, jerk2 = [], []
tcur = 0.0
for jv, dur in segs:
    n = int(round(dur/dt))
    t2.append(tcur + np.arange(n)*dt)
    jerk2.append(np.full(n, jv))
    tcur += dur
t2 = np.concatenate(t2); jerk2 = np.concatenate(jerk2)
acc2 = np.cumsum(jerk2)*dt
v2 = np.cumsum(acc2)*dt
q2 = np.cumsum(v2)*dt

# --- 5차 (같은 제약을 만족하는 최소 T) ---
T_q = max(15*L/(8*vm), np.sqrt(10*L/(np.sqrt(3)*am)))
t3 = np.arange(0, T_q, dt)
tau = t3/T_q
q3 = L*mj_s(tau)
v3 = L*mj_sd(tau)/T_q
a3 = L*mj_sdd(tau)/T_q**2
j3 = L*(60 - 360*tau + 360*tau**2)/T_q**3

fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
names = [f'사다리꼴 (T={T_trap:.2f}s)', f'S-커브 (T={T_s:.2f}s)', f'5차 최소저크 (T={T_q:.2f}s)']
colors = ['tab:red', 'tab:green', 'tab:blue']
for (t, q, v, a, j), c, nm in [((t1, q1, v1, acc1, None), colors[0], names[0]),
                               ((t2, q2, v2, acc2, jerk2), colors[1], names[1]),
                               ((t3, q3, v3, a3, j3), colors[2], names[2])]:
    ax[0].plot(t, q, color=c, lw=1.7, label=nm)
    ax[1].plot(t, v, color=c, lw=1.7)
    ax[2].plot(t, a, color=c, lw=1.7)
    if j is not None:
        ax[3].plot(t, j, color=c, lw=1.7)
# 사다리꼴 저크 = 임펄스 (스위칭 순간에 ±∞)
for ts in [0, ta, ta+tc, T_trap]:
    ax[3].annotate('', xy=(ts, 110), xytext=(ts, 0),
                   arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.6))
ax[3].text(0.02, 118, '사다리꼴: 저크 = 임펄스(±∞)', color='tab:red', fontsize=9)
ax[0].set_ylabel('위치 [rad]'); ax[1].set_ylabel('속도 [rad/s]')
ax[2].set_ylabel('가속 [rad/s²]'); ax[3].set_ylabel('저크 [rad/s³]')
ax[3].set_xlabel('시간 [s]')
ax[1].axhline(vm, color='gray', ls=':', lw=1); ax[1].text(0.7, vm+0.06, 'v_max', color='gray', fontsize=8)
ax[2].axhline(am, color='gray', ls=':', lw=1); ax[2].axhline(-am, color='gray', ls=':', lw=1)
ax[3].axhline(jm, color='gray', ls=':', lw=1); ax[3].axhline(-jm, color='gray', ls=':', lw=1)
ax[0].legend(fontsize=9, loc='lower right')
for a_ in ax: a_.grid(alpha=0.3)
ax[3].set_ylim(-140, 140)
fig.suptitle('같은 이동(0.8 rad), 같은 한계(v≤2, a≤8) — 매끄러움과 시간의 교환', y=0.995)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_profiles.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# fig3 — 청크 경계: 순진한 교체 vs 5차 브리지 스티칭 (RTC 미니어처)
# =====================================================================
# 이전 청크: 최소저크 0 → 1.2 rad, [0,1]s. t=0.5에서 새 청크 도착.
# 새 청크: 낡은 관측 기반이라 q_pred=0.58에서 정지 출발로 계산됨: 0.58 → 1.4, [0.5,1.5]s
q_old = lambda t: 1.2*mj_s(t/1.0)
v_old = lambda t: 1.2*mj_sd(t/1.0)/1.0
q_new = lambda t: 0.58 + 0.82*mj_s((t-0.5)/1.0)
v_new = lambda t: 0.82*mj_sd((t-0.5)/1.0)/1.0
a_new = lambda t: 0.82*mj_sdd((t-0.5)/1.0)/1.0**2

t_sw, t_br = 0.5, 0.15          # 교체 시각, 브리지 길이
# 브리지: (q_old(0.5), v_old(0.5), 0) → (q_new(0.65), v_new(0.65), a_new(0.65)) 5차
def quintic_coeffs(T, b):
    A = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
                  [1,T,T**2,T**3,T**4,T**5],
                  [0,1,2*T,3*T**2,4*T**3,5*T**4],
                  [0,0,2,6*T,12*T**2,20*T**3]])
    return np.linalg.solve(A, b)
b = np.array([q_old(t_sw), v_old(t_sw), 0.0,
              q_new(t_sw+t_br), v_new(t_sw+t_br), a_new(t_sw+t_br)])
a_br = quintic_coeffs(t_br, b)
q_br = lambda t: sum(a_br[k]*(t-t_sw)**k for k in range(6))

tt = np.arange(0, 1.5, 0.002)
def traj(mode):
    q = np.empty_like(tt)
    for i, t in enumerate(tt):
        if t < t_sw:
            q[i] = q_old(t)
        elif mode == 'naive':
            q[i] = q_new(t)
        else:
            q[i] = q_br(t) if t < t_sw + t_br else q_new(t)
    return q
qn, qs = traj('naive'), traj('stitch')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
ax[0].plot(tt, qn, color='tab:red', lw=1.6, label='순진한 교체 (점프)')
ax[0].plot(tt, qs, color='tab:blue', lw=1.6, label='5차 브리지 스티칭')
ax[0].axvline(t_sw, color='gray', ls=':', lw=1)
ax[0].text(t_sw+0.01, 0.15, '새 청크 도착', color='gray', fontsize=9, rotation=90)
ax[0].axvspan(t_sw, t_sw+t_br, color='tab:blue', alpha=0.08)
ax[0].set_xlabel('시간 [s]'); ax[0].set_ylabel('셋포인트 [rad]')
ax[0].set_title('(a) 위치 셋포인트'); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
for q, c, lb in [(qn, 'tab:red', '순진한 교체'), (qs, 'tab:blue', '스티칭')]:
    ax[1].plot(tt[1:], np.diff(q)/0.002, color=c, lw=1.3, label=lb)
ax[1].axvline(t_sw, color='gray', ls=':', lw=1)
ax[1].set_xlabel('시간 [s]'); ax[1].set_ylabel('함의 속도 [rad/s]')
ax[1].set_title('(b) 함의 속도 — 경계의 불연속'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.suptitle('청크 경계에서 무슨 일이 벌어지는가 — 실습에서 만들 스티처의 목표', y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_stitching.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("saved: fig1_chunk_to_setpoints.png, fig2_profiles.png, fig3_stitching.png")
print(f"[fig2] T_trap={T_trap:.4f}, T_s={T_s:.4f}, T_quintic={T_q:.4f}")
print(f"[fig2] 적분 검증: 사다리꼴 끝위치={q1[-1]:.4f}, S-커브 끝위치={q2[-1]:.4f} (목표 0.8)")
print(f"[fig3] 순진 교체: 위치 점프={qn[np.searchsorted(tt,t_sw)]-qn[np.searchsorted(tt,t_sw)-1]:+.4f} rad, "
      f"속도 {v_old(t_sw):.3f}→{0.0:.3f}")
