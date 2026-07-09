# Lec 46 그림 생성 스크립트 — GR00T 패밀리 (N1~N1.7)
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델/가중치/GPU 없음.
#   (1) embodiment 사영 MLP: 이종 DoF(7/14/29)를 공유 임베딩 차원 D로 사영 (shape 검증)
#   (2) dual-rate: System2 ~10Hz ZOH 명령을 System1 ~120Hz가 추종 (대역 분리)
#   (3) DreamGen 효과: novel task 0% → 43.2%(본 환경)/28.5%(새 환경) 막대
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

rng = np.random.default_rng(0)
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공통 파라미터
# ============================================================================
F2 = 10.0     # System 2 (이해) 주파수 [Hz] — 대표값(논문+2차 종합)
F1 = 120.0    # System 1 (행동) 주파수 [Hz] — 대표값
RATE_RATIO = F1 / F2                 # ~12배
D_EMB = 1536                         # 공유 임베딩 차원(예시값; 실제 N1 백본 hidden에 상당)

# ============================================================================
# (A) embodiment 사영 MLP — 이종 DoF를 공유 차원으로
#   로봇마다 상태·행동 차원이 다르다: 단일 팔(7), 양팔(14), 휴머노이드 전신(29).
#   각 embodiment는 자기만의 인코더 MLP W_e: R^{d_e} -> R^{D}로 공유 임베딩에 사영.
#   핵심: 하류(트랜스포머/DiT)는 항상 D차원만 본다 → 백본을 공유(cross-embodiment).
# ============================================================================
def make_encoder(d_in, D, seed):
    r = np.random.default_rng(seed)
    # 2층 MLP: d_in -> H -> D  (He 초기화 흉내)
    H = 256
    W1 = r.standard_normal((d_in, H)) * np.sqrt(2.0 / d_in)
    b1 = np.zeros(H)
    W2 = r.standard_normal((H, D)) * np.sqrt(2.0 / H)
    b2 = np.zeros(D)
    return (W1, b1, W2, b2)

def encode(x, params):
    # x: (B, d_in) -> (B, D). GELU 근사 대신 ReLU (개념 검증용)
    W1, b1, W2, b2 = params
    h = np.maximum(0.0, x @ W1 + b1)
    return h @ W2 + b2

# 세 embodiment: (이름, 상태 DoF)
EMBS = [("단일 팔\n(예: SO-101류)", 7),
        ("양팔\n(예: GR-1 상체)", 14),
        ("휴머노이드 전신\n(예: N1 29-DoF)", 29)]
B = 4  # 배치(관측 4개)
enc_params = {d: make_encoder(d, D_EMB, seed=100 + d) for _, d in EMBS}
shape_log = []
embs_emb = {}
for name, d in EMBS:
    x = rng.standard_normal((B, d))          # (B, d_e) 상태 벡터
    z = encode(x, enc_params[d])             # (B, D) 공유 임베딩
    embs_emb[d] = z
    shape_log.append((name.replace("\n", " "), d, x.shape, z.shape))

# 파라미터 수 계산 (embodiment별 인코더 vs 공유 백본)
def enc_param_count(d_in, D, H=256):
    return d_in * H + H + H * D + D
per_emb_params = {d: enc_param_count(d, D_EMB) for _, d in EMBS}
# "공유 백본"은 여기선 개념상 D->D 트랜스포머 블록 하나로 대리 (실제 N1은 2.2B)
shared_backbone_params = D_EMB * D_EMB * 4   # 대략 self-attn+MLP 한 블록 규모 대리

# 검증: 하류가 보는 차원은 embodiment 무관하게 항상 D
all_shared_dim_equal = all(embs_emb[d].shape[1] == D_EMB for _, d in EMBS)

# ============================================================================
# (B) dual-rate 시뮬 — System2(10Hz) ZOH 명령을 System1(120Hz)이 추종
#   0강 E3 재사용: 상위가 계단(ZOH) 잠재 명령을 내고, 하위가 대역 T1으로 추종.
#   상위 명령 변화가 하위 폐루프 대역보다 느리면 → 하위 출력이 부드러워진다.
# ============================================================================
T_END = 1.0
dt1 = 1.0 / F1                       # System1 스텝 (~8.33 ms)
t1 = np.arange(0, T_END, dt1)        # System1 타임그리드 (fine)
t2 = np.arange(0, T_END, 1.0 / F2)   # System2 타임그리드 (coarse, 10Hz)

# 상위(System2)가 '의도한' 연속 잠재 궤적 (부드러운 서브골 궤적 흉내)
def latent_intent(t):
    return 0.7 * np.sin(2 * np.pi * 1.3 * t) + 0.3 * np.sin(2 * np.pi * 0.5 * t + 0.6)

intent_cont = latent_intent(t1)               # 이상적 연속 의도(참조)
cmd2 = latent_intent(t2)                       # System2가 10Hz로 실제 내는 표본

# ZOH: 10Hz 명령을 120Hz 그리드로 계단 확장 (마지막 명령 유지)
idx = np.minimum((t1 * F2).astype(int), len(cmd2) - 1)
zoh = cmd2[idx]                                # System1이 매 스텝 보는 상위 명령(계단)

# System1: 1차 저역 추종기 (하위 폐루프의 최소 모형).  tau_follow가 대역을 정함.
#   x[k+1] = x[k] + dt1 * (zoh[k] - x[k]) / tau_follow
tau_follow = 0.03                              # 하위 추종 시상수 [s] (~5.3 Hz 대역)
x = 0.0
follow = np.zeros_like(t1)
for k in range(len(t1)):
    follow[k] = x
    x = x + dt1 * (zoh[k] - x) / tau_follow

# 정량 지표: (1) 계단 명령의 저크(불연속) vs 추종 출력의 부드러움
#   RMS 2차 차분(가속도 대리)으로 부드러움 측정
def rms_second_diff(y):
    d2 = np.diff(y, 2)
    return np.sqrt(np.mean(d2 ** 2))
rough_zoh = rms_second_diff(zoh)
rough_follow = rms_second_diff(follow)
smooth_ratio = rough_zoh / rough_follow        # 추종이 몇 배 부드러운가
# (2) 추종 오차 (연속 의도 대비)
track_rms = np.sqrt(np.mean((follow - intent_cont) ** 2))

# ============================================================================
# 그림 1: dual-system 타이밍 (10Hz 계단 + 120Hz 추종)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                               gridspec_kw={'width_ratios': [1.35, 1]})

# (a) 타이밍: 연속 의도 / 10Hz ZOH 계단 / 120Hz 추종
ax1.plot(t1, intent_cont, 'k:', lw=1.4, alpha=0.7, label='상위의 연속 의도(참조)')
ax1.step(t1, zoh, where='post', color='C1', lw=1.8,
         label=f'System 2 명령 ZOH ({F2:.0f}Hz 계단)')
ax1.plot(t2, cmd2, 'o', color='C1', ms=6, zorder=5)
ax1.plot(t1, follow, 'C0', lw=2.0,
         label=f'System 1 추종 ({F1:.0f}Hz, 부드러움)')
ax1.set_xlabel('시간 t [s]'); ax1.set_ylabel('잠재 명령 / 행동 값')
ax1.set_title(f'(a) 대역 분리: 상위 {F2:.0f}Hz 계단을 하위 {F1:.0f}Hz가 추종\n'
              f'주파수 비 {RATE_RATIO:.0f}배 — 하위가 계단 모서리를 부드럽게 채운다')
ax1.grid(alpha=0.3); ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_xlim(0, T_END)

# (b) 스텝 예산 막대: 상위 1스텝 동안 하위가 도는 횟수
steps = int(round(RATE_RATIO))
ax2.bar([0], [1], width=0.5, color='C1', label=f'System 2: 1 스텝 / {1000/F2:.0f} ms')
ax2.bar([1], [steps], width=0.5, color='C0',
        label=f'System 1: {steps} 스텝 / 같은 구간')
ax2.text(0, 1 + 0.5, '1', ha='center', fontsize=11, weight='bold')
ax2.text(1, steps + 0.5, f'{steps}', ha='center', fontsize=11, weight='bold')
ax2.set_xticks([0, 1]); ax2.set_xticklabels(['System 2\n(이해)', 'System 1\n(행동)'])
ax2.set_ylabel('상위 1주기 동안의 스텝 수')
ax2.set_title(f'(b) 하위는 상위 1스텝당 {steps}회 돈다\n(대역 예산 = 주파수 비)')
ax2.set_ylim(0, steps + 2.5); ax2.grid(alpha=0.3, axis='y')
ax2.legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(OUT + 'fig1_dual_rate_timing.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 2: embodiment 사영 shape — 이종 DoF → 공유 차원 D
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                               gridspec_kw={'width_ratios': [1.25, 1]})

# (a) 도식: 서로 다른 입력 차원이 각자의 MLP를 거쳐 같은 D로
ax1.axis('off')
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ys = [7.8, 5.0, 2.2]
cols = ['#2c6fb0', '#9a6a3a', '#5a8a3a']
for (name, d), y, c in zip(EMBS, ys, cols):
    # 입력 상태 벡터 박스(높이 = DoF에 비례)
    hgt = 0.6 + d * 0.09
    ax1.add_patch(plt.Rectangle((0.3, y - hgt/2), 1.5, hgt, fc=c, ec='k', alpha=0.75))
    ax1.text(1.05, y, f'{d}', ha='center', va='center', color='w',
             fontsize=11, weight='bold')
    ax1.text(1.05, y - hgt/2 - 0.45, name, ha='center', va='top', fontsize=8)
    # embodiment별 MLP 박스
    ax1.add_patch(plt.Rectangle((3.2, y - 0.55), 2.0, 1.1, fc='#e1eefb', ec=c, lw=2))
    ax1.text(4.2, y, f'$W_{{e}}$ MLP\n$R^{{{d}}}\\!\\to\\! R^{{D}}$',
             ha='center', va='center', fontsize=8.5)
    ax1.annotate('', xy=(3.2, y), xytext=(1.8, y),
                 arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
    ax1.annotate('', xy=(6.6, y), xytext=(5.2, y),
                 arrowprops=dict(arrowstyle='->', color=c, lw=1.8))
# 공유 임베딩/백본 박스
ax1.add_patch(plt.Rectangle((6.7, 1.4), 2.9, 7.0, fc='#efe4d8', ec='#9a6a3a', lw=2.5))
ax1.text(8.15, 6.5, f'공유 임베딩\n$R^{{D}}$\n(D={D_EMB})', ha='center', va='center',
         fontsize=10, weight='bold')
ax1.text(8.15, 3.6, '공유 트랜스포머\n+ DiT 액션 헤드\n(embodiment 무관)',
         ha='center', va='center', fontsize=8.5)
ax1.set_title('(a) embodiment별 인코더 MLP가 이종 DoF를 공유 차원 D로 사영\n'
              '하류(백본·헤드)는 항상 D만 본다 → 하나의 모델이 여러 몸을 쓴다', fontsize=10)

# (b) shape·파라미터 표를 막대로: 입력 DoF vs 인코더 파라미터 수
names = [n.replace("\n", " ") for n, _ in EMBS]
dofs = [d for _, d in EMBS]
pcounts = [per_emb_params[d] for _, d in EMBS]
xb = np.arange(len(EMBS))
ax2b = ax2.twinx()
bars = ax2.bar(xb - 0.2, dofs, width=0.4, color='C0', label='입력 DoF $d_e$')
bars2 = ax2b.bar(xb + 0.2, np.array(pcounts) / 1e3, width=0.4, color='C1',
                 label='인코더 파라미터 [K]')
ax2.set_xticks(xb)
ax2.set_xticklabels(['단일 팔\n7', '양팔\n14', '전신\n29'], fontsize=9)
ax2.set_ylabel('입력 DoF $d_e$', color='C0')
ax2.tick_params(axis='y', labelcolor='C0')
ax2b.set_ylabel('embodiment 인코더 파라미터 [K]', color='C1')
ax2b.tick_params(axis='y', labelcolor='C1')
for x, d in zip(xb, dofs):
    ax2.text(x - 0.2, d + 0.4, f'{d}', ha='center', fontsize=9, color='C0')
ax2.set_title(f'(b) 출력은 모두 $R^{{{D_EMB}}}$로 통일\n'
              f'입력 DoF는 달라도 하류가 보는 차원은 같다', fontsize=10)
ax2.set_ylim(0, 34)
fig.tight_layout()
fig.savefig(OUT + 'fig2_embodiment_projection.png', dpi=140)
plt.close(fig)

# ============================================================================
# 그림 3: 데이터 피라미드 + DreamGen 효과 (0% → 43.2%/28.5%)
# ============================================================================
DREAM_SEEN = 43.2    # DreamGen 데이터 추가 후 novel task 성공률(본 환경) [%]  (참고문헌[3])
DREAM_NEW = 28.5     # (새 환경) [%]
BASELINE = 0.0       # pick-and-place만 배운 N1의 novel task 성공률 [%]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                               gridspec_kw={'width_ratios': [1.05, 1]})

# (a) 데이터 피라미드 (3층) — 위로 갈수록 좁아지는 사다리꼴
ax1.axis('off'); ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
tri = [
    (0.3, 9.7, 0.4, 3.0, '#cfe3f7',
     '바닥 · 웹 + 인간 1인칭 영상\n거대·저비용 (행동 라벨 없음)\nlatent action / IDM 의사 라벨'),
    (2.0, 8.0, 3.2, 5.9, '#f2dfc4',
     '중간 · 합성 데이터\nGR00T-Mimic(시뮬 증식)\n+ DreamGen neural trajectories'),
    (3.7, 6.3, 6.1, 8.8, '#f7c9c9',
     '꼭대기 · 실기 teleop\n희소 · 고가'),
]
for x0, x1, y0, y1, c, txt in tri:
    ax1.add_patch(plt.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                              closed=True, fc=c, ec='k', lw=1.3))
    ax1.text((x0 + x1) / 2, (y0 + y1) / 2, txt, ha='center', va='center', fontsize=8)
ax1.annotate('PI: 꼭대기를\n돈으로 채운다', xy=(5.0, 7.5), xytext=(7.2, 9.1), fontsize=8.5,
             ha='center', color='C3', arrowprops=dict(arrowstyle='->', color='C3'))
ax1.annotate('NVIDIA: 바닥·중간을\n도구로 채운다', xy=(1.2, 2.0), xytext=(7.4, 3.4),
             fontsize=8.5, ha='center', color='C0',
             arrowprops=dict(arrowstyle='->', color='C0'))
ax1.set_title('(a) 데이터 피라미드 — 층마다 라벨링 방법이 다르다', fontsize=10)

# (b) DreamGen 효과 막대
cats = ['baseline\n(pick&place만)', 'DreamGen 추가\n(본 환경)', 'DreamGen 추가\n(새 환경)']
vals = [BASELINE, DREAM_SEEN, DREAM_NEW]
bcols = ['#b0b0b0', 'C0', 'C2']
bars = ax2.bar(cats, vals, color=bcols, width=0.6, edgecolor='k')
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.0, f'{v:.1f}%',
             ha='center', fontsize=10, weight='bold')
ax2.set_ylabel('novel task(새 동사) 성공률 [%]')
ax2.set_ylim(0, 52)
ax2.set_title('(b) DreamGen: 새 동사 0% → 43.2%(본)/28.5%(새)\n'
              '"생성 모델이 데이터 엔진이 된다"의 실증 1호 [3]', fontsize=10)
ax2.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig3_data_pyramid_dreamgen.png', dpi=140)
plt.close(fig)

# ============================================================================
# (D) FLARE 토이 — 픽셀 재구성 vs 잠재 정합 (E3)
#   미래 관측을 저차원 잠재로 인코딩한 뒤, 예측기가 그 잠재를 맞춘다(L2).
#   요점: 픽셀에는 태스크 무관 자유도(배경 잡음)가 섞여 있어 재구성이 비싸고 산만.
#         잠재는 태스크 유관 정보만 추려 정합 대상 차원이 작다.
#   토이: '미래 프레임' = 태스크 신호(저차원) + 배경 잡음(고차원).
#         인코더 g = 태스크 신호만 뽑는 사영. 예측기 h = z_t로 미래 잠재 예측.
# ============================================================================
P_PIX = 400          # '픽셀' 차원 (관측이 사는 고차원)
K_LAT = 8            # 잠재 차원 (태스크 유관 정보만)
n_frames = 300
rng_f = np.random.default_rng(3)

# 태스크 유관 잠재 (진짜 신호): 시간에 따라 부드럽게 변하는 K차원 궤적
tt_f = np.linspace(0, 1, n_frames)
true_lat = np.stack([np.sin(2*np.pi*(1+0.5*k)*tt_f + k) for k in range(K_LAT)], axis=1)  # (n,K)
# 픽셀 생성: 잠재를 무작위 사전 B로 픽셀에 올리고 + 큰 배경 잡음(태스크 무관)
Bdict = rng_f.standard_normal((K_LAT, P_PIX)) / np.sqrt(K_LAT)
bg_noise_std = 1.0
pixels = true_lat @ Bdict + bg_noise_std * rng_f.standard_normal((n_frames, P_PIX))  # (n,P)

# 인코더 g: 픽셀 -> 잠재 (최소자승 사영; 실제론 학습되지만 여기선 해석적으로)
#   g가 Bdict의 유사역행렬이면 태스크 잠재를 복원하고 배경 잡음은 평균적으로 걸러짐
g = np.linalg.pinv(Bdict)                        # (P, K)
enc_lat = pixels @ g                             # (n, K)  인코더가 뽑은 잠재

# 정합 대상 차원 비교: 픽셀 재구성 목표(P=400) vs 잠재 정합 목표(K=8)
recon_dim = P_PIX
align_dim = K_LAT
dim_ratio = recon_dim / align_dim               # 50배

# '예측기' h: z_t(현재 잠재)로 z_{t+1}(미래 잠재) 예측 (여기선 1스텝 선형 AR로 대리)
Z = enc_lat
Zt, Ztk = Z[:-1], Z[1:]
A = np.linalg.lstsq(Zt, Ztk, rcond=None)[0]     # h: z_t -> z_{t+1}
pred = Zt @ A
flare_loss = np.mean(np.sum((pred - Ztk)**2, axis=1))     # 잠재 공간 L2
# 픽셀 공간에서 같은 예측을 재구성했을 때의 손실(참고: 배경 잡음 때문에 훨씬 큼)
pred_pix = pred @ Bdict
pix_loss = np.mean(np.sum((pred_pix - (Ztk @ Bdict))**2, axis=1)) / P_PIX  # 픽셀당

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                               gridspec_kw={'width_ratios': [1.1, 1]})
# (a) 정합 대상 차원 막대: 픽셀 재구성 vs 잠재 정합
ax1.bar([0], [recon_dim], width=0.5, color='#b0b0b0', label='픽셀 재구성 목표')
ax1.bar([1], [align_dim], width=0.5, color='C0', label='FLARE 잠재 정합 목표')
ax1.text(0, recon_dim+8, f'{recon_dim}\n(픽셀 전부)', ha='center', fontsize=9)
ax1.text(1, align_dim+8, f'{align_dim}\n(태스크 latent)', ha='center', fontsize=9, color='C0')
ax1.set_xticks([0, 1]); ax1.set_xticklabels(['픽셀 생성\n(순진한 방법)', 'FLARE\n(잠재 정합)'])
ax1.set_ylabel('정합해야 하는 차원 수')
ax1.set_ylim(0, recon_dim*1.18)
ax1.set_title(f'(a) FLARE는 정합 대상 차원을 {dim_ratio:.0f}배 줄인다\n'
              f'픽셀 {recon_dim} → 잠재 {align_dim} (태스크 무관 배경 제외)', fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# (b) 미래 잠재 예측 정합: 예측 vs 실제 (2개 잠재 차원)
ax2.plot(tt_f[1:], Ztk[:, 0], 'C0', lw=1.8, label='실제 미래 잠재 (차원 1)')
ax2.plot(tt_f[1:], pred[:, 0], 'C1--', lw=1.8, label='예측 $h(z_t)$ (차원 1)')
ax2.plot(tt_f[1:], Ztk[:, 1], 'C2', lw=1.4, alpha=0.8, label='실제 미래 잠재 (차원 2)')
ax2.plot(tt_f[1:], pred[:, 1], 'C3--', lw=1.4, alpha=0.8, label='예측 (차원 2)')
ax2.set_xlabel('시간 (정규화)'); ax2.set_ylabel('잠재 값')
ax2.set_title(f'(b) 미래 잠재 정합 (L2={flare_loss:.2e})\n'
              f'행동 라벨 없이 "다음 표현"을 예측 → 인간 영상 학습 통로', fontsize=10)
ax2.grid(alpha=0.3); ax2.legend(fontsize=7.5, ncol=2, loc='upper right')
fig.tight_layout()
fig.savefig(OUT + 'fig4_flare_latent_alignment.png', dpi=140)
plt.close(fig)

# ============================================================================
# 본문 인용 수치 출력 — 본문과 일치해야 함
# ============================================================================
print("=== lec46 그림 4개 생성 완료 ===")
print(f"[dual-rate] F2={F2:.0f}Hz, F1={F1:.0f}Hz → 주파수 비 = {RATE_RATIO:.1f}배 "
      f"(상위 1스텝당 하위 {int(round(RATE_RATIO))}스텝)")
print(f"[dual-rate] ZOH 계단 거칠기(RMS 2차차분) {rough_zoh:.4f} vs 추종 {rough_follow:.6f} "
      f"→ 추종이 {smooth_ratio:.1f}배 부드러움; 추종 오차 RMS(연속 의도 대비)={track_rms:.4f}")
print("[embodiment 사영] 이종 DoF → 공유 차원 shape:")
for nm, d, xs, zs in shape_log:
    print(f"    {nm:24s}: 상태 {xs} --(W_e MLP)--> 임베딩 {zs}")
print(f"    하류가 보는 차원이 embodiment 무관하게 모두 D={D_EMB}? {all_shared_dim_equal}")
print(f"[파라미터] embodiment 인코더(D={D_EMB}, H=256): "
      + ", ".join(f"{d}DoF={per_emb_params[d]/1e3:.1f}K" for _, d in EMBS))
print(f"[DreamGen] baseline novel task {BASELINE:.0f}% "
      f"→ DreamGen 추가 {DREAM_SEEN:.1f}%(본 환경)/{DREAM_NEW:.1f}%(새 환경)")
print(f"[FLARE] 정합 대상 차원: 픽셀 재구성 {recon_dim} vs 잠재 정합 {align_dim} "
      f"→ {dim_ratio:.0f}배 축소; 미래 잠재 정합 L2 = {flare_loss:.3e}")

# ----- Worked Example 손계산 검증 (그림 없음) -----
# WE-A: 작은 수치로 사영 shape·행렬곱 손검증
d_e, D_small = 3, 4                    # 손계산용 축소: 상태 3차원 -> 임베딩 4차원
Wsmall = np.array([[1., 0., -1., 2.],
                   [0., 2.,  1., 0.],
                   [1., 1.,  0., -1.]])   # (3,4)
x_we = np.array([1.0, -2.0, 0.5])          # 상태 벡터 (3,)
z_we = x_we @ Wsmall                        # (4,)
print(f"[WE-A 손계산] x={x_we.tolist()} @ W(3x4) = {z_we.tolist()} "
      f"(상태 3 → 임베딩 4; 하류는 4차원만 본다)")

# WE-B: dual-rate ZOH 손검증 — 10Hz 명령 [0.0, 0.5, 1.0]을 120Hz(=12x)로 계단 확장
cmd_we = np.array([0.0, 0.5, 1.0])
n_sub = 12
zoh_we = np.repeat(cmd_we, n_sub)           # 각 명령이 12스텝 유지
print(f"[WE-B 손계산] 10Hz 명령 {cmd_we.tolist()} → 120Hz ZOH 길이 {len(zoh_we)} "
      f"(명령당 {n_sub}스텝 유지); 처음 15개 = {zoh_we[:15].tolist()}")
# 1차 추종기 3스텝 손계산 (tau=0.03, dt=1/120): 명령 0.5로 점프 시
dt_we = 1/120; tau_we = 0.03; xf = 0.0; traj_we = []
for c in [0.5]*3:
    xf = xf + dt_we*(c - xf)/tau_we
    traj_we.append(round(xf, 4))
print(f"[WE-B 손계산] 명령 0.5 점프 후 추종 3스텝: {traj_we} "
      f"(dt=1/120, tau=0.03 → 스텝당 {dt_we/tau_we:.4f}씩 접근)")
