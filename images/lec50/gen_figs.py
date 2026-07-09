# Lec 50 그림 생성 스크립트 — Action의 여정 (청크 실행 계층)
# 실행: python3 gen_figs.py   (numpy / matplotlib 만 필요, CPU 전용)
# 개념 재현: temporal ensembling / Real-Time Chunking(동결+soft mask) / 다중율 보간(ZOH·선형·3차)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

rng = np.random.default_rng(0)

# =====================================================================
# 공통: "정책이 겨냥하는 참 궤적" 하나 (1D 스칼라 관절각으로 단순화)
#   실제 VLA는 다차원이지만, 청크 실행 전략의 수학은 스칼라에서 그대로 드러난다.
# =====================================================================
FPS = 50.0            # 실효 제어/기록 주기 (ACT·π0 급)
dt  = 1.0 / FPS
T_end = 2.0
t = np.arange(0, T_end, dt)             # 50 Hz 격자 (100 스텝)
N = len(t)

def true_traj(tt):
    # 부드러운 참 궤적 (rad)
    return 0.6*np.sin(2*np.pi*0.5*tt) + 0.15*np.sin(2*np.pi*1.3*tt + 0.7)

y_true = true_traj(t)

# =====================================================================
# FIG 1 — Temporal ensembling: 겹치는 청크의 지수가중 평균이 지터를 죽인다
#   ACT 방식: 매 스텝 추론 → 각 시각 t 를 겨냥한 예측이 여러 청크에서 나온다.
#   각 예측에 독립 노이즈(추론 다봉성·표본 분산)를 섞고, 그 시각을 예측한
#   "예측 나이 i"(0=방금 만든 청크, 클수록 오래 전 청크)에 대해 w_i=exp(-m i).
# =====================================================================
H = 20                     # 청크 길이 (스텝)
m_ens = 0.1                # 가중 감쇠율 (ACT 기본값 관습)
noise_std = 0.05           # 청크별 예측 노이즈 (rad)

# preds[k] : 시각 index k 를 겨냥한, 나이 i=0..K_k 인 예측들의 리스트
# 매 스텝 s(=0..N-1)에서 청크 하나 생성: 시각 s..s+H-1 을 예측. 그 청크가
# 시각 k 를 예측하면, k 에서 본 그 예측의 '나이'는 (k - s).
preds = [[] for _ in range(N)]
for s in range(N):
    chunk_bias = rng.normal(0, noise_std)          # 청크 전체에 걸리는 표본 바이어스
    for j in range(H):
        k = s + j
        if k >= N:
            break
        step_noise = rng.normal(0, noise_std*0.5)  # 스텝별 소음
        val = true_traj(t[k]) + chunk_bias + step_noise
        age = k - s                                 # 이 예측의 나이 (0=갓 생성)
        preds[k].append((age, val))

def ensemble_at(k):
    ages = np.array([a for a, _ in preds[k]])
    vals = np.array([v for _, v in preds[k]])
    w = np.exp(-m_ens * ages)
    w = w / w.sum()                                 # 정규화: sum w_i = 1
    return (w * vals).sum()

# naive: 앙상블 없이 '가장 최근 청크(나이 0)의 값'만 실행
naive = np.array([min(preds[k], key=lambda p: p[0])[1] for k in range(N)])
ens   = np.array([ensemble_at(k) for k in range(N)])

def jitter(sig):
    # 지터 = 스텝간 2차 차분(저크 대용)의 RMS. 부드러울수록 작다.
    return np.sqrt(np.mean(np.diff(sig, 2)**2))

j_naive, j_ens = jitter(naive), jitter(ens)
rms_naive = np.sqrt(np.mean((naive - y_true)**2))
rms_ens   = np.sqrt(np.mean((ens   - y_true)**2))

# 가중치 프로파일 (그림 삽입용): 나이 0..9, m=0.1 과 m=0.25 비교
ii = np.arange(10)
w01 = np.exp(-0.10*ii); w01 /= w01.sum()
w025 = np.exp(-0.25*ii); w025 /= w025.sum()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(t, y_true, 'k-', lw=2.2, label='참 궤적', zorder=1)
ax[0].plot(t, naive, color='#c0392b', lw=1.0, alpha=0.9,
           label=f'naive(최신 청크만) · 지터 {j_naive*1e3:.1f}e-3')
ax[0].plot(t, ens, color='#2c6fb0', lw=1.8,
           label=f'temporal ensembling · 지터 {j_ens*1e3:.1f}e-3')
ax[0].set_xlabel('시간 [s]'); ax[0].set_ylabel('관절각 [rad]')
ax[0].set_title('(a) 겹치는 청크의 지수가중 평균 → 지터 감소', fontsize=11)
ax[0].legend(fontsize=8, loc='upper right'); ax[0].grid(alpha=0.3)

ax[1].stem(ii, w01, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='m=0.10 (부드러움↑)')
ax[1].stem(ii+0.15, w025, linefmt='C3-', markerfmt='C3s', basefmt=' ', label='m=0.25 (반응성↑)')
ax[1].set_xlabel('예측 나이 i  (0 = 갓 생성한 청크)')
ax[1].set_ylabel('정규화 가중 w_i')
ax[1].set_title('(b) w_i = exp(-m·i) / Σ  — m이 저역필터 폭', fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_temporal_ensembling.png', dpi=130)
plt.close()

# =====================================================================
# FIG 2 — Real-Time Chunking: 추론 지연 동안 실행분을 '동결'하면 경계가 잇는다.
#   시나리오: 청크 A 실행 중 시각 t_req 에서 새 관측으로 청크 B 추론 시작,
#   B 는 d 스텝 뒤 t_req+d 에 도착(추론 지연). 도착 시점의 새 청크가
#   관측 변화로 A 와 다른 값을 겨냥한다(정책 다봉성).
#   naive: 도착 즉시 B 로 교체 → 경계 점프.
#   RTC:   B 의 앞 d 스텝은 A 값으로 '동결'(inpainting), 이어지는 구간은
#          soft mask 로 A→B 를 선형 블렌딩 → 연속.
# =====================================================================
d_delay = 6                 # 추론 지연 (스텝) ; 50Hz면 120 ms
t_req = 20                  # 재요청 시각 index
guard = d_delay             # 동결 구간 길이 = 추론 지연
blend = 20                  # soft mask 블렌딩 길이 (동결 직후, 청크 겹침 구간)

tc = np.arange(0, 70)       # 스텝 격자
base = 0.5*np.sin(2*np.pi*0.02*tc)          # 두 청크 공통 기저 모양 (완만)
OFFSET = 0.30               # B 가 A 대비 겨냥을 옮긴 양 (새 관측·정책 다봉성)

# 청크 A: 기저 그대로.  청크 B: 같은 모양을 OFFSET 만큼 옮겨 겨냥 (관측이 바뀌었으므로).
A = base.copy()
def chunkB(idx):
    return base[idx] + OFFSET

arrive = t_req + d_delay     # B 가 실제로 쓸 수 있게 되는 시각

# naive: arrive 시점에 즉시 B 로 갈아탐 → 한 스텝에 OFFSET 이 통째로 튄다.
naive_exec = A.copy()
for k in range(arrive, len(tc)):
    naive_exec[k] = chunkB(k)

# RTC: [arrive, arrive+guard) 는 A 로 '동결'(inpainting: 추론 지연 구간을 이전 청크로),
#      이어 blend 스텝 동안 soft mask(smoothstep)로 A→B 를 이어붙임, 이후 B.
rtc_exec = A.copy()
gstart = arrive
gend = arrive + guard
for k in range(gstart, gend):
    rtc_exec[k] = A[k]                      # 동결(freeze)
for k in range(gend, gend+blend):
    u = (k - gend + 1) / (blend + 1)        # 0→1
    a = u*u*(3 - 2*u)                       # smoothstep: 양 끝 도함수 0 (C¹ 연속 이음)
    rtc_exec[k] = (1 - a)*A[k] + a*chunkB(k)
for k in range(gend+blend, len(tc)):
    rtc_exec[k] = chunkB(k)

# 경계 점프 = "청크 전환이 만든 최악의 스텝 불연속" (전환 창 [arrive-1, gend+blend] 안).
# 참 궤적 자체의 완만한 기울기가 아니라, 전략이 통제하는 '이음매'만 재는 것이 핵심.
def switch_jump(sig):
    win = np.abs(np.diff(sig))[arrive-1:gend+blend]
    return win.max()

jump_naive = switch_jump(naive_exec)
jump_rtc   = switch_jump(rtc_exec)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for a_ in ax:
    a_.plot(tc, A, '--', color='#888', lw=1.3, label='청크 A (이전 계획)')
    a_.plot(tc[t_req:], [chunkB(k) for k in tc[t_req:]], ':', color='#27ae60', lw=1.3,
            label='청크 B (새 관측)')
    a_.axvspan(t_req, arrive, color='#f1c40f', alpha=0.18)
    a_.axvline(t_req, color='#f39c12', ls='-', lw=1, alpha=0.7)
    a_.axvline(arrive, color='#f39c12', ls='-', lw=1, alpha=0.7)
    a_.set_xlabel('스텝')

ax[0].plot(tc, naive_exec, color='#c0392b', lw=2.2,
           label=f'naive 교체 · 경계 점프 {jump_naive:.3f}')
ax[0].set_ylabel('관절각 [rad]')
ax[0].set_title('(a) naive: 도착 즉시 교체 → 불연속', fontsize=11)
ax[0].text(t_req+0.3, ax[0].get_ylim()[0]+0.05, '추론\n지연', fontsize=8, color='#b9770e')
ax[0].legend(fontsize=8, loc='lower left')

ax[1].plot(tc, rtc_exec, color='#2c6fb0', lw=2.2,
           label=f'RTC(동결+soft mask) · 점프 {jump_rtc:.3f}')
ax[1].axvspan(gstart, gend, color='#5dade2', alpha=0.18)
ax[1].set_title('(b) RTC: 지연구간 동결 → 연속 이음', fontsize=11)
ax[1].text(gstart+0.3, ax[1].get_ylim()[1]-0.14, '동결\n(guard)', fontsize=8, color='#1a5276')
ax[1].legend(fontsize=8, loc='lower left')
for a_ in ax:
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_rtc_boundary.png', dpi=130)
plt.close()

# =====================================================================
# FIG 3 — 다중율 계층: 50Hz 셋포인트를 1kHz 로 올릴 때 ZOH vs 선형 vs 3차
#   (0강 E3 의 확장) 같은 50Hz 표본을 세 방식으로 업샘플, 저크/오차 비교.
# =====================================================================
f_lo, f_hi = 50.0, 1000.0
up = int(f_hi / f_lo)              # 20
Tw = 0.4
t_lo = np.arange(0, Tw, 1/f_lo)    # 50 Hz 표본 시각
t_hi = np.arange(0, Tw, 1/f_hi)    # 1 kHz 격자

def ref(tt):
    return 0.4*np.sin(2*np.pi*3.0*tt)   # 3 Hz 목표

y_lo = ref(t_lo)
y_ref_hi = ref(t_hi)                     # '진짜' 연속 궤적 (기준)

# ZOH: 계단
zoh = np.array([y_lo[min(int(tt*f_lo), len(y_lo)-1)] for tt in t_hi])
# 선형 보간
lin = np.interp(t_hi, t_lo, y_lo)
# 3차 (natural cubic spline; scipy 없이 numpy 로 구현)
def cubic_spline(xk, yk, xq):
    n = len(xk)
    h = np.diff(xk)
    # natural spline: 2차 도함수 M
    A = np.zeros((n, n)); b = np.zeros(n)
    A[0,0] = 1; A[-1,-1] = 1
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2*(h[i-1]+h[i])
        A[i, i+1] = h[i]
        b[i] = 6*((yk[i+1]-yk[i])/h[i] - (yk[i]-yk[i-1])/h[i-1])
    M = np.linalg.solve(A, b)
    out = np.zeros_like(xq)
    idx = np.clip(np.searchsorted(xk, xq)-1, 0, n-2)
    for j, xx in enumerate(xq):
        i = idx[j]
        dx = xx - xk[i]
        hi = h[i]
        a0 = yk[i]
        a1 = (yk[i+1]-yk[i])/hi - hi*(2*M[i]+M[i+1])/6
        a2 = M[i]/2
        a3 = (M[i+1]-M[i])/(6*hi)
        out[j] = a0 + a1*dx + a2*dx**2 + a3*dx**3
    return out

cub = cubic_spline(t_lo, y_lo, t_hi)

def jerk_rms(sig, dt_):
    return np.sqrt(np.mean(np.diff(sig, 3)**2)) / dt_**3

def tracking_rms(sig):
    return np.sqrt(np.mean((sig - y_ref_hi)**2))

dt_hi = 1/f_hi
rows = [
    ('ZOH (마지막값 유지)', zoh),
    ('선형 보간',          lin),
    ('3차 스플라인',        cub),
]
metrics = {name: (tracking_rms(s), jerk_rms(s, dt_hi)) for name, s in rows}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
# 왼쪽: 파형 (확대 구간)
win = (t_hi >= 0.10) & (t_hi <= 0.24)
ax[0].plot(t_hi[win], y_ref_hi[win], 'k-', lw=2.4, label='참 궤적(연속)')
ax[0].plot(t_hi[win], zoh[win], color='#c0392b', lw=1.4, label='ZOH')
ax[0].plot(t_hi[win], lin[win], color='#f39c12', lw=1.4, label='선형')
ax[0].plot(t_hi[win], cub[win], color='#2c6fb0', lw=1.6, label='3차')
ax[0].plot(t_lo[(t_lo>=0.10)&(t_lo<=0.24)],
           y_lo[(t_lo>=0.10)&(t_lo<=0.24)], 'ko', ms=5, label='50Hz 표본')
ax[0].set_xlabel('시간 [s]'); ax[0].set_ylabel('셋포인트 [rad]')
ax[0].set_title('(a) 50Hz → 1kHz 업샘플 (확대)', fontsize=11)
ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)

# 오른쪽: 저크 RMS 막대 (log)
names = [r[0] for r in rows]
jerks = [metrics[n][1] for n in names]
bars = ax[1].bar(range(3), jerks, color=['#c0392b', '#f39c12', '#2c6fb0'])
ax[1].set_yscale('log')
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, fontsize=9)
ax[1].set_ylabel('저크 RMS [rad/s³] (log)')
ax[1].set_title('(b) 매끄러움: 계단이 저크를 폭발시킨다', fontsize=11)
for b, j in zip(bars, jerks):
    ax[1].text(b.get_x()+b.get_width()/2, j*1.15, f'{j:.2e}',
               ha='center', fontsize=8)
ax[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig3_multirate_interp.png', dpi=130)
plt.close()

# =====================================================================
# FIG 4 — 모델별 action space 지형도 (1절 표의 시각화)
#   x: action 차원, y: 실효 주기(Hz, log), 버블 크기: 청크 길이, 색: 표현(이산/연속)
#   수치는 본문 표(1차 자료 검증)에서 그대로. 스텝단위 모델은 청크=1로 표시.
#   name, dim, rate(Hz), chunk, repr('discrete'/'continuous')
# =====================================================================
models = [
    ('RT-2',       7,   2,   1,  'discrete'),   # 1~3Hz → 대표 2
    ('OpenVLA',    7,   6,   1,  'discrete'),
    ('ACT',        14,  50,  100,'continuous'),
    ('Diffusion\nPolicy', 7, 10, 8, 'continuous'),  # 실행 8
    ('π0/π0.5',    18,  50,  50, 'continuous'),
    ('SmolVLA',    6,   30,  50, 'continuous'),
    ('GR00T\nN1.7',132, 120, 40, 'continuous'),
    ('RDT-1B',     128, 10,  64, 'continuous'),   # 주기 미보고 → 대략 위치만(라벨서 표기)
]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
col = {'discrete': '#c0392b', 'continuous': '#2c6fb0'}
for name, dim, rate, chunk, rep in models:
    ax.scatter(dim, rate, s=40 + chunk*7, c=col[rep], alpha=0.55,
               edgecolors='k', linewidths=0.8, zorder=3)
    dy = 1.12 if name not in ('OpenVLA',) else 0.80
    ax.annotate(name, (dim, rate), (dim, rate*dy),
                fontsize=8.5, ha='center', va='bottom')
ax.set_xscale('symlog'); ax.set_yscale('log')
ax.set_xlim(4, 200); ax.set_ylim(1, 400)
ax.set_xlabel('action 차원  (제로패딩/통일공간 포함)')
ax.set_ylabel('실효 주기 [Hz] (log)')
ax.set_title('모델별 action space 지형도 — 버블=청크 길이, 색=표현', fontsize=12)
# 범례
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=col['discrete'],
              markeredgecolor='k', markersize=10, label='이산 토큰 (스텝단위)'),
       Line2D([0],[0], marker='o', color='w', markerfacecolor=col['continuous'],
              markeredgecolor='k', markersize=10, label='연속 (청크)')]
ax.legend(handles=leg, fontsize=9, loc='lower right')
ax.grid(alpha=0.3, which='both')
ax.text(4.4, 1.5, '이산·스텝단위(RT-2·OpenVLA)는\n청크가 없어 버블이 작다',
        fontsize=8, color='#7d2b1c')
plt.tight_layout()
plt.savefig('fig4_action_space_landscape.png', dpi=130)
plt.close()

# =====================================================================
# 콘솔 출력 — 본문이 인용하는 모든 수치
# =====================================================================
print("=== FIG1 temporal ensembling ===")
print(f"jitter naive = {j_naive*1e3:.2f}e-3, jitter ens = {j_ens*1e3:.2f}e-3, "
      f"ratio = {j_naive/j_ens:.2f}x")
print(f"tracking RMS naive = {rms_naive*1e3:.2f} mrad, ens = {rms_ens*1e3:.2f} mrad")
print(f"w(m=0.1): i=0 {w01[0]:.3f}, i=5 {w01[5]:.3f}, half-mass age ~ {np.log(2)/0.1:.1f}")
print(f"w(m=0.25): i=0 {w025[0]:.3f}, i=5 {w025[5]:.3f}")
print()
print("=== FIG2 RTC ===")
print(f"delay d = {d_delay} steps ({d_delay/FPS*1e3:.0f} ms @50Hz), guard = {guard}")
print(f"boundary jump naive = {jump_naive:.4f} rad, RTC = {jump_rtc:.4f} rad, "
      f"reduction = {jump_naive/jump_rtc:.1f}x")
print()
print("=== FIG3 multirate ===")
for n in names:
    tr, jk = metrics[n]
    print(f"{n:20s}: tracking RMS = {tr*1e6:8.2f} urad,  jerk RMS = {jk:.3e}")
print(f"ZOH/linear jerk ratio = {metrics[names[0]][1]/metrics[names[1]][1]:.1f}x")
print(f"linear/cubic jerk ratio = {metrics[names[1]][1]/metrics[names[2]][1]:.1f}x")
print(f"ZOH/cubic tracking ratio = {metrics[names[0]][0]/metrics[names[2]][0]:.1f}x")
