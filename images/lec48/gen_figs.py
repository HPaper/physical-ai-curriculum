# Lec 48 그림 생성 스크립트 — 비공개 진영: 계층 주파수·대역 분리·온보드 지연
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만. 모델 다운로드/GPU 없음)
# 개념을 numpy 토이로 재현한다 — 공개된 주파수/파라미터 수치만으로 대역 분리와
# 온보드 추론 지연 예산을 계산한다. 본문이 인용하는 모든 수치는 이 스크립트 출력이다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 회사별 계층 스택 (공개 1차 자료 수치).  각 층 = (이름, 대표 주파수 Hz, System번호 or None)
#   System 번호가 None인 층 = "번호 밖"(전류/전류루프·엔지니어링 상위 로직)
#   Helix 02: S2 추론(7~9Hz, 대표 8) / S1 200Hz / S0 1kHz  [figure.ai/news/helix-02]
#   GR00T N1: S2 VLM(~10Hz) / S1 DiT(120Hz)  — 저수준 제어기는 번호 밖  [arXiv 2503.14734]
#   전류루프는 모두 번호 밖(~kHz~수십 kHz FOC)
# ============================================================================
COMPANIES = {
    'Figure Helix 02': [('S2 추론(VLM)', 8.0, 'S2'),
                        ('S1 트랜스포머', 200.0, 'S1'),
                        ('S0 전신제어기', 1000.0, 'S0'),
                        ('전류루프(FOC)', 20000.0, None)],
    'GR00T N1': [('S2 VLM', 10.0, 'S2'),
                 ('S1 DiT', 120.0, 'S1'),
                 ('저수준 제어기', 1000.0, None),
                 ('전류루프(FOC)', 20000.0, None)],
    'Figure Helix': [('S2 VLM', 8.0, 'S2'),
                     ('S1 트랜스포머', 200.0, 'S1'),
                     ('저수준 제어기', 1000.0, None),
                     ('전류루프(FOC)', 20000.0, None)],
    '1X Redwood': [('온보드 VLA', 5.0, None),
                   ('저수준 제어기', 1000.0, None),
                   ('전류루프(FOC)', 20000.0, None)],
    'Atlas LBM': [('DiT 정책', 30.0, None),
                  ('MPC 전신제어', 500.0, None),
                  ('전류루프(FOC)', 20000.0, None)],
}

print("=" * 68)
print("[E1] 계층 대역 분리 조건:  f_i  <<  f_{i+1}   (아래층이 위층보다 빨라야)")
print("=" * 68)
# 분리 배율 = 아래층 주파수 / 위층 주파수.  경험칙: >= 5(반)~10(충분)이면 분리 성립.
SEP_OK, SEP_MARGINAL = 10.0, 5.0

def separation_ratios(layers):
    freqs = [f for (_, f, _) in layers]
    return [freqs[i+1] / freqs[i] for i in range(len(freqs) - 1)]

for name, layers in COMPANIES.items():
    ratios = separation_ratios(layers)
    names = [n for (n, _, _) in layers]
    print(f"\n{name}:")
    for i, r in enumerate(ratios):
        verdict = '충분' if r >= SEP_OK else ('경계' if r >= SEP_MARGINAL else '위반')
        print(f"  {names[i]:>14s} → {names[i+1]:<14s}: 배율 {r:6.1f}x  [{verdict}]")

# Helix 02 세 학습층 사이 두 경계(핵심 인용 수치)
helix_ratios = separation_ratios(COMPANIES['Figure Helix 02'])
print(f"\n>>> Helix 02 대역 배율: S2→S1={helix_ratios[0]:.0f}x, "
      f"S1→S0={helix_ratios[1]:.0f}x, S0→전류루프={helix_ratios[2]:.0f}x")

# ---------------------------------------------------------------------------
# fig1: 회사별 계층 주파수 사다리 + System 번호 경계(GR00T 2층 vs Helix 3층)
# ---------------------------------------------------------------------------
order = ['Figure Helix 02', 'Figure Helix', 'GR00T N1', 'Atlas LBM', '1X Redwood']
sys_color = {'S2': '#2c6fb0', 'S1': '#e08a1e', 'S0': '#3a9a5a', None: '#9a9a9a'}
sys_label = {'S2': 'System 2 (느림·전역)', 'S1': 'System 1 (빠름·국소)',
             'S0': 'System 0 (전신 서보)', None: '번호 밖 (제어기·전류루프)'}

fig, ax = plt.subplots(figsize=(11.6, 5.4))
for xi, name in enumerate(order):
    layers = COMPANIES[name]
    for (lname, f, sysnum) in layers:
        y = np.log10(f)
        ax.scatter(xi, y, s=260, color=sys_color[sysnum], zorder=4,
                   edgecolor='k', linewidth=0.8)
        txt = f"{lname}\n{f:.0f} Hz" if f < 1000 else f"{lname}\n{f/1000:.0f} kHz"
        ax.annotate(txt, (xi, y), xytext=(xi + 0.12, y), fontsize=7.3,
                    va='center', ha='left')
    ys = [np.log10(f) for (_, f, _) in layers]
    ax.plot([xi] * len(ys), ys, color='#bbbbbb', lw=1.4, zorder=1)
    # System 번호가 "끝나는" 경계선 표시
    numbered = [np.log10(f) for (_, f, s) in layers if s is not None]
    if numbered:
        yb = (min(numbered) + max([np.log10(f) for (_, f, s) in layers
                                   if s is None] + [min(numbered) - 0.5])) / 2
        ax.axhspan(0, 0, xmin=0, xmax=0)  # no-op keep axis
# 경계 주석: GR00T=2층, Helix 02=3층
ax.annotate('GR00T: 번호는 S1에서 끝\n(제어기=번호 밖)',
            (2, np.log10(120)), xytext=(2.05, np.log10(120) - 0.95),
            fontsize=7.8, color='#7a4a1a',
            arrowprops=dict(arrowstyle='->', color='#7a4a1a', lw=1.2))
ax.annotate('Helix 02: S0까지 번호 안\n(학습형 1kHz 전신제어기)',
            (0, np.log10(1000)), xytext=(0.05, np.log10(1000) + 0.42),
            fontsize=7.8, color='#1a5a3a',
            arrowprops=dict(arrowstyle='->', color='#1a5a3a', lw=1.2))
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, fontsize=9)
ax.set_xlim(-0.5, len(order) - 0.05)   # 오른쪽 라벨 공간 확보
ax.set_ylim(0.4, 4.7)
yt = [0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.3]
ax.set_yticks(yt)
ax.set_yticklabels([f'{10**t:.0f}' if 10**t < 1000 else f'{10**t/1000:.0f}k' for t in yt])
ax.set_ylabel('제어 주파수 [Hz] (log)')
ax.set_title('회사별 계층 주파수 사다리 — 같은 "느린 위 / 빠른 아래"이나 System 번호의 경계는 회사마다 다르다',
             fontsize=11)
ax.grid(alpha=0.25, axis='y')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markeredgecolor='k', markersize=11, label=sys_label[k])
           for k, c in sys_color.items()]
ax.legend(handles=handles, fontsize=8.3, loc='center', bbox_to_anchor=(0.50, 0.62),
          framealpha=0.96)
fig.tight_layout()
fig.savefig(OUT + 'fig1_frequency_ladder.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# fig2: 대역 분리 검증 — (a) 회사별 인접층 배율 막대,  (b) 위상여유/지연 예산
#   τ_max = φm / ωc  (lec00 E3·lec17와 동일).  ωc ~= 2π·(폐루프 대역폭 ≈ 층 주파수/여유)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

# (a) 인접층 분리 배율
bar_names, bar_vals, bar_cols = [], [], []
for name in ['Figure Helix 02', 'GR00T N1', 'Atlas LBM', '1X Redwood']:
    layers = COMPANIES[name]
    ratios = separation_ratios(layers)
    lnames = [n for (n, _, _) in layers]
    for i, r in enumerate(ratios):
        bar_names.append(f"{name}\n{lnames[i].split('(')[0][:6]}→{lnames[i+1].split('(')[0][:6]}")
        bar_vals.append(r)
        bar_cols.append('#3a9a5a' if r >= SEP_OK else ('#e08a1e' if r >= SEP_MARGINAL else '#c0392b'))
ypos = np.arange(len(bar_vals))
ax1.barh(ypos, bar_vals, color=bar_cols, edgecolor='k', lw=0.5)
ax1.axvline(SEP_OK, color='#3a9a5a', ls='--', lw=1.5, label=f'충분 분리 (≥{SEP_OK:.0f}x)')
ax1.axvline(SEP_MARGINAL, color='#e08a1e', ls=':', lw=1.5, label=f'경계 (≥{SEP_MARGINAL:.0f}x)')
ax1.set_yticks(ypos)
ax1.set_yticklabels(bar_names, fontsize=6.8)
ax1.set_xscale('log')
ax1.set_xlabel('인접층 주파수 배율 $f_{i+1}/f_i$ (log)')
ax1.set_title('(a) 대역 분리 검증 — 모든 인접 경계가\n최소 5x, 대개 ≥10x 떨어져 있다')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(alpha=0.25, axis='x', which='both')

# (b) 아래층 정착시간 vs 위층 명령주기.  분리 성립 ⇔ 위층 주기 > 아래층 정착시간
#   아래층 폐루프 대역폭 bw ~= 층 주파수의 1/5(보수적).  정착시간 t_s ~= 5/(2π·bw)
#   (1차 근사: 시상수 τ=1/(2π·bw), 5τ에서 ~99% 정착)
def settling_ms(loop_hz, bw_frac=0.2):
    bw = loop_hz * bw_frac                    # 폐루프 대역폭 [Hz]
    return 5.0 / (2 * np.pi * bw) * 1000.0    # 5τ 정착시간 [ms]

# Helix 02 세 하위층: (층 주파수, 위층 명령 주기)
layer_hz = np.array([200., 1000., 20000.])            # S1, S0, 전류루프
layer_nm = ['S1 (200Hz)\n위=S2 8Hz', 'S0 (1kHz)\n위=S1 200Hz', '전류루프\n위=S0 1kHz']
settle = np.array([settling_ms(f) for f in layer_hz])
upper_period = np.array([1000. / 8, 1000. / 200, 1000. / 1000])  # 위층 주기 ms
sep_margin = upper_period / settle                    # >1 이면 분리 성립
xb = np.arange(len(layer_hz))
w = 0.36
ax2.bar(xb - w/2, settle, w, color='#2c6fb0', edgecolor='k', lw=0.5,
        label='아래층 정착시간 $t_s\\approx5/(2\\pi\\,bw)$')
ax2.bar(xb + w/2, upper_period, w, color='#e08a1e', edgecolor='k', lw=0.5,
        label='위층 명령 주기 $1/f_{upper}$')
ax2.set_yscale('log')
ax2.set_xticks(xb)
ax2.set_xticklabels(layer_nm, fontsize=8.0)
ax2.set_ylabel('시간 [ms] (log)')
ax2.set_title('(b) 아래층 정착시간 < 위층 명령 주기여야 분리 성립\n(주황 > 파랑: 아래층이 명령 사이에 정착)')
ax2.legend(fontsize=7.8, loc='upper right')
ax2.grid(alpha=0.25, axis='y', which='both')
for i, (s, u) in enumerate(zip(settle, upper_period)):
    ax2.text(i - w/2, s * 1.18, f'{s:.1f}', ha='center', fontsize=7.5)
    ax2.text(i + w/2, u * 1.18, f'{u:.1f}', ha='center', fontsize=7.5)
    ax2.text(i, max(s, u) * 2.4, f'여유 {u/s:.1f}x', ha='center', fontsize=7.8,
             color='#1a5a3a' if u/s >= 1 else '#c0392b')
ax2.set_ylim(top=upper_period.max() * 6)
print(f"\n>>> Helix 02 하위층 분리 여유(위층주기/정착시간): "
      f"S1={sep_margin[0]:.1f}x, S0={sep_margin[1]:.1f}x, 전류루프={sep_margin[2]:.1f}x")
fig.tight_layout()
fig.savefig(OUT + 'fig2_band_separation.png', dpi=140)
plt.close(fig)

print("\n" + "=" * 68)
print("[E2] 온보드 추론 지연:  latency = 2N·T_ctx / (peak·util)   (N=params, T=처리 토큰)")
print("=" * 68)
# 한 번의 추론이 처리하는 토큰 수 T_ctx만큼 FLOP가 붙는다: FLOP ≈ 2·N·T_ctx.
#   - S2(VLM): 이미지 패치+언어 프리필 ~수백 토큰 (T_ctx=256로 근사)
#   - S1/S0(제어 헤드): 짧은 상태·액션 청크 (T_ctx=32)
# 온보드 임베디드 가속기: 실효 처리량 ~10 TFLOP/s (배치1은 memory-bound라 peak의 일부만).
ONBOARD_TFLOPS = 10.0e12        # 실효 FLOP/s (임베디드 온보드, 배치1 실측 스케일)
def latency_ms(params, t_ctx, tput=ONBOARD_TFLOPS):
    flops = 2.0 * params * t_ctx             # forward 1회 근사
    return flops / tput * 1000.0

def max_rate_hz(params, t_ctx, **kw):
    return 1000.0 / latency_ms(params, t_ctx, **kw)

# (params, T_ctx, 층 성격)
MODELS = [
    ('Helix S2',    7e9,   256, 'S2'), ('π0(참고)',   3.3e9, 256, 'S2'),
    ('Helix S1',    80e6,  32,  'S1'), ('Atlas LBM',  450e6, 32,  'S1'),
    ('1X Redwood',  160e6, 32,  'S1'), ('SmolVLA(참고)', 450e6, 64, 'S2'),
    ('Helix S0',    10e6,  32,  'S0'),
]
for nm, p, tc, _ in MODELS:
    print(f"  {nm:>14s}: {p/1e6:8.0f}M · T={tc:3d} → 지연 {latency_ms(p, tc):8.2f} ms "
          f"→ 최대 {max_rate_hz(p, tc):9.1f} Hz")

# 핵심 대비: 7B S2가 200Hz(5ms 예산)로 온보드에서 돌 수 있는가?
lat_7b = latency_ms(7e9, 256)
lat_80m = latency_ms(80e6, 32)
print(f"\n>>> 7B S2를 200Hz(=5ms)로? 실제 지연 {lat_7b:.0f}ms vs 예산 5ms "
      f"→ {lat_7b/5:.0f}배 초과. 최대 {max_rate_hz(7e9, 256):.1f}Hz "
      f"→ S2는 7~9Hz 저주파로만 온보드 가능")
print(f">>> 80M S1은 {lat_80m:.2f}ms → 최대 {max_rate_hz(80e6, 32):.0f}Hz "
      f"→ 200Hz(5ms) 예산 안에 여유. '큰 위층은 느리게, 작은 아래층은 빠르게'가 강제된다")

# ---------------------------------------------------------------------------
# fig3: 모델 크기 → 온보드 지연 → 달성 가능 주파수
#   곡선은 두 컨텍스트 레짐(S2형 T=256, 제어헤드형 T=32)으로 그린다.
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
p_grid = np.logspace(np.log10(5e6), np.log10(1e10), 200)
lat_256 = np.array([latency_ms(p, 256) for p in p_grid])
lat_32 = np.array([latency_ms(p, 32) for p in p_grid])

# (a) 크기 vs 지연 (두 컨텍스트 레짐)
ax1.loglog(p_grid / 1e6, lat_256, 'C0', lw=2.2, label='S2형 프리필 (T=256)')
ax1.loglog(p_grid / 1e6, lat_32, 'C2', lw=2.2, ls='--', label='제어헤드 (T=32)')
for nm, p, tc, _ in MODELS:
    ax1.scatter(p / 1e6, latency_ms(p, tc), s=55, zorder=5, edgecolor='k', lw=0.6,
                color='#c0392b' if p >= 1e9 else '#2c6fb0')
    ax1.annotate(nm, (p / 1e6, latency_ms(p, tc)), fontsize=6.6,
                 xytext=(p / 1e6 * 1.12, latency_ms(p, tc) * 0.62))
ax1.axhline(5, color='#e08a1e', ls=':', lw=1.4)
ax1.text(6, 5.6, '200Hz 예산 (5ms)', fontsize=7.6, color='#e08a1e')
ax1.axhline(125, color='#2c6fb0', ls=':', lw=1.4)
ax1.text(6, 140, 'S2대 예산 (~8Hz, 125ms)', fontsize=7.6, color='#2c6fb0')
ax1.set_xlabel('파라미터 수 [M] (log)')
ax1.set_ylabel('온보드 추론 지연 [ms] (log)')
ax1.set_title(f'(a) 크기→지연  (온보드 실효 {ONBOARD_TFLOPS/1e12:.0f} TFLOP/s)')
ax1.grid(alpha=0.25, which='both')
ax1.legend(fontsize=8, loc='upper left')

# (b) 달성 가능 주파수 + 목표 주파수대 밴드
rate_256 = 1000.0 / lat_256
ax2.loglog(p_grid / 1e6, rate_256, 'C0', lw=2.2, label='달성 가능 최대 Hz (T=256)')
for hz, lab, c in [(9, 'S2대 (7~9Hz)', '#2c6fb0'),
                   (200, 'S1대 (200Hz)', '#e08a1e'),
                   (1000, 'S0대 (1kHz)', '#3a9a5a')]:
    ax2.axhline(hz, color=c, ls='--', lw=1.4)
    ax2.text(6, hz * 1.15, lab, fontsize=7.8, color=c)
for nm, p, tc, _ in MODELS:
    ax2.scatter(p / 1e6, max_rate_hz(p, tc), s=55, zorder=5, edgecolor='k', lw=0.6,
                color='#c0392b' if p >= 1e9 else '#2c6fb0')
ax2.set_xlabel('파라미터 수 [M] (log)')
ax2.set_ylabel('달성 가능 제어 주파수 [Hz] (log)')
ax2.set_title('(b) 큰 모델은 저주파에만 — 소형 온보드가 필요한 이유')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig(OUT + 'fig3_size_vs_latency.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# fig4: System 번호 경계 비교 — GR00T(2층) vs Helix 02(3층).  "어디서 번호가 끝나나"
#   블록 다이어그램. 색은 fig1과 동일(파랑 S2 / 주황 S1 / 초록 S0 / 회색 번호 밖).
# ---------------------------------------------------------------------------
fig, (axG, axH) = plt.subplots(1, 2, figsize=(12.2, 5.6))

def draw_stack(ax, title, blocks, boundary_after):
    # blocks: [(라벨, Hz텍스트, System표기 or None, 색)]  위→아래
    ax.set_xlim(0, 10); ax.set_ylim(0, len(blocks) * 2 + 1.5)
    ax.axis('off'); ax.set_title(title, fontsize=12, pad=12)
    n = len(blocks)
    for i, (lab, hz, sysnum, col) in enumerate(blocks):
        y = (n - 1 - i) * 2 + 0.8
        ax.add_patch(plt.Rectangle((1.2, y), 7.6, 1.5, facecolor=col,
                                   edgecolor='k', lw=1.3, alpha=0.9))
        tag = f"[{sysnum}]" if sysnum else "[번호 밖]"
        ax.text(5.0, y + 0.95, f"{lab}", ha='center', va='center',
                fontsize=10, color='white' if sysnum else '#333', weight='bold')
        ax.text(5.0, y + 0.40, f"{hz}   {tag}", ha='center', va='center',
                fontsize=8.5, color='white' if sysnum else '#333')
        if i < n - 1:
            ax.annotate('', xy=(5.0, y - 0.05), xytext=(5.0, y + 0.05 - 0.0),
                        arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.6))
    # System 번호 경계선(굵은 빨강 점선)
    yb = (n - 1 - boundary_after) * 2 + 0.55
    ax.plot([0.4, 9.6], [yb, yb], color='#c0392b', ls='--', lw=2.2)
    ax.text(9.55, yb + 0.18, 'System 번호 여기서 끝', ha='right', va='bottom',
            fontsize=9, color='#c0392b', weight='bold')

# GR00T N1: S2 → S1 → (번호 밖) 저수준 제어기 → (번호 밖) 전류루프
draw_stack(axG, 'GR00T N1 — 2층 (번호는 S1에서 끝)',
           [('System 2  VLM', '~10 Hz · 이해·계획', 'S2', '#2c6fb0'),
            ('System 1  DiT', '120 Hz · 행동 a 생성', 'S1', '#e08a1e'),
            ('저수준 제어기', '~1 kHz · 관절 서보', None, '#c9c9c9'),
            ('전류루프 (FOC)', '~20 kHz · 펌웨어·물리', None, '#c9c9c9')],
           boundary_after=1)   # S1(index1) 다음에 경계

# Helix 02: S2 → S1 → S0(학습형 1kHz) → (번호 밖) 전류루프
draw_stack(axH, 'Figure Helix 02 — 3층 (S0까지 번호 안)',
           [('System 2  추론', '7~9 Hz · 이해·추론', 'S2', '#2c6fb0'),
            ('System 1  트랜스포머', '200 Hz · 전신 관절목표', 'S1', '#e08a1e'),
            ('System 0  전신제어기', '1 kHz · 학습형(옛 C++ 대체)', 'S0', '#3a9a5a'),
            ('전류루프 (FOC)', '~20 kHz · 펌웨어·물리', None, '#c9c9c9')],
           boundary_after=2)   # S0(index2) 다음에 경계
fig.suptitle('System 번호가 "끝나는 자리"는 회사마다 다르다 — 같은 1kHz 전신 서보를 GR00T는 번호 밖, Helix는 System 0으로 부른다',
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT + 'fig4_system_boundary.png', dpi=140)
plt.close(fig)

print("\nfigures written: fig1_frequency_ladder, fig2_band_separation, "
      "fig3_size_vs_latency, fig4_system_boundary")
