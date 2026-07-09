# Lec 47 그림 생성 스크립트 — 작은 모델들·계보도
# 실행: python3 gen_figs.py  (numpy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 모델 다운로드/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')

# ============================================================================
# 공통 토이 1 — 시각 토큰 예산:  tokens = (res / patch)^2,  pixel-shuffle r배 → /r^2
#   pixel-shuffle: 공간 r×r 패치를 채널로 접어 하나의 토큰으로 → 토큰 수 1/r^2
# ============================================================================
def vis_tokens(res, patch, r=1):
    n_patch = (res // patch) ** 2          # ViT 패치 토큰 수
    return n_patch // (r * r)              # pixel-shuffle r배 압축

print("=" * 64)
print("[E1] 시각 토큰 예산  tokens = (res/patch)^2,  pixel-shuffle → /r^2")
print("=" * 64)
# SmolVLA: SmolVLM-2 백본(patch=16), 512px 입력, pixel-shuffle r=4
smolvla_tok = vis_tokens(512, 16, 4)
print(f"SmolVLA(512px, patch16, shuffle r=4): (512/16)^2={ (512//16)**2 } → /16 = {smolvla_tok} 토큰/카메라")
# 대조: 대형 VLM 관행(고해상 896px, patch14, shuffle 없음)
big_tok = vis_tokens(896, 14, 1)
print(f"고해상 baseline(896px, patch14, shuffle 없음): (896/14)^2 = {big_tok} 토큰/카메라")
print(f"토큰 절감 배율: {big_tok // smolvla_tok}배  (4096 → 64)")
# 3카메라 컨텍스트 길이 비교
print(f"3카메라 시각 컨텍스트:  SmolVLA {3*smolvla_tok} vs baseline {3*big_tok} 토큰")

# ---------- fig1: 해상도 vs 토큰수 (shuffle 유무) ----------
res_grid = np.array([224, 384, 512, 640, 768, 896])
patch = 16
tok_base = np.array([vis_tokens(r, patch, 1) for r in res_grid])
tok_sh2  = np.array([vis_tokens(r, patch, 2) for r in res_grid])
tok_sh4  = np.array([vis_tokens(r, patch, 4) for r in res_grid])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(res_grid, tok_base, 'o-', color='#c0392b', lw=2, label='shuffle 없음 (r=1)')
ax.plot(res_grid, tok_sh2, 's-', color='#e67e22', lw=2, label='pixel-shuffle r=2  (/4)')
ax.plot(res_grid, tok_sh4, '^-', color='#2c6fb0', lw=2, label='pixel-shuffle r=4  (/16)')
ax.axhline(64, color='#27ae60', ls='--', lw=1.4, alpha=0.8)
ax.annotate('SmolVLA 예산 ≈ 64토큰/카메라', xy=(512, 64), xytext=(560, 260),
            fontsize=10, color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60'))
ax.scatter([512], [smolvla_tok], s=140, facecolors='none', edgecolors='#2c6fb0', lw=2.2, zorder=5)
ax.set_yscale('log')
ax.set_xlabel('입력 해상도 (px, 정사각 · patch=16)')
ax.set_ylabel('카메라당 시각 토큰 수 (log)')
ax.set_title('그림 1 · 해상도 vs 시각 토큰 수 — pixel-shuffle가 예산을 접는다')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(OUT + 'fig1_tokens_vs_resolution.png', dpi=130)
plt.close(fig)

# ============================================================================
# 공통 토이 2 — 모델 크기 vs 추론 메모리:  가중치 바닥 ≈ (bytes/param) × params
#   fp16 = 2 bytes/param (원리적 바닥),  fp32 = 4 bytes/param.
#   실측 추론 풋프린트 = 가중치 + 활성/KV/런타임 오버헤드 (대략 fp32 바닥 + 여유).
# ============================================================================
print("=" * 64)
print("[E2] 모델 크기 → 추론 메모리 (가중치 바닥 ≈ bytes/param × params)")
print("=" * 64)
models = [
    ("SmolVLA",   0.45e9),
    ("SimVLA",    0.50e9),
    ("TinyVLA",   0.90e9),
    ("RDT-1B",    1.20e9),
    ("π0",        3.30e9),
    ("OpenVLA",   7.00e9),
    ("CogACT",    7.30e9),
]
for name, p in models:
    fp16 = p * 2 / 1e9
    fp32 = p * 4 / 1e9
    print(f"  {name:9s} {p/1e9:4.2f}B → fp16 가중치 {fp16:5.2f} GB · fp32 {fp32:5.2f} GB")
print(f"  → SmolVLA fp16 바닥 {0.45e9*2/1e9:.2f}GB(실측 추론 ~2GB) vs π0 fp32 {3.3e9*4/1e9:.1f}GB(실측 ~14GB): 약 {round((3.3e9*4)/(0.45e9*2))}배 차")

# ---------- fig2: 모델 크기 vs 메모리 (로그) ----------
names = [m[0] for m in models]
params = np.array([m[1] for m in models])
fp16 = params * 2 / 1e9
fp32 = params * 4 / 1e9

fig, ax = plt.subplots(figsize=(7.4, 4.8))
order = np.argsort(params)
xs = params[order] / 1e9
ax.plot(xs, fp16[order], 'o-', color='#2c6fb0', lw=2, label='fp16 가중치 바닥 (2 B/param)')
ax.plot(xs, fp32[order], 's--', color='#c0392b', lw=2, label='fp32 가중치 바닥 (4 B/param)')
ax.set_xscale('log'); ax.set_yscale('log')
for name, p in models:
    ax.annotate(name, xy=(p/1e9, p*2/1e9), xytext=(3, -11), textcoords='offset points',
                fontsize=8.5, color='#0c3a63')
# 소비자 GPU 경계선
for gb, lab in [(8, '8GB (노트북/온보드)'), (24, '24GB (RTX 4090)')]:
    ax.axhline(gb, color='#7f8c8d', ls=':', lw=1.1)
    ax.annotate(lab, xy=(0.45, gb), fontsize=8, color='#7f8c8d', va='bottom')
ax.set_xlabel('파라미터 수 (B, log)')
ax.set_ylabel('가중치 메모리 (GB, log)')
ax.set_title('그림 2 · 모델 크기 vs 추론 메모리 — 왜 450M이 노트북에 들어가는가')
ax.grid(True, which='both', alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(OUT + 'fig2_size_vs_memory.png', dpi=130)
plt.close(fig)

# ============================================================================
# 공통 토이 3 — latent action 의 원리적 한계 (역동역학의 null space)
#   무라벨 영상 = 운동학 x(t)만 관측.  힘/강성은 관측의 null space에 있다.
#   같은 x(t)를 서로 다른 (질량, 힘, 접촉)이 만든다 → many-to-one.
# ============================================================================
print("=" * 64)
print("[E3] latent action: 영상(운동학)만으로 힘·강성 복원 불가 (null space)")
print("=" * 64)
t = np.linspace(0, 1, 200)
x = 0.3 * (1 - np.cos(np.pi * t)) / 2        # 블록을 0→0.3m 부드럽게 민다
xd = np.gradient(x, t)
xdd = np.gradient(xd, t)
# 카메라가 보는 것은 x(t) 하나. 두 가설이 완전히 같은 x(t)를 만든다:
#   H1: 가벼운 블록(m=0.5), 자유공간 → F1 = m1*xdd
#   H2: 무거운 블록(m=2.0) + 눈에 안 보이는 내부 접촉/파지력 Fc → F2 = m2*xdd + Fc
m1, m2 = 0.5, 2.0
Fc = 5.0                                      # 카메라에 보이지 않는 내부 힘 (일정)
F1 = m1 * xdd
F2 = m2 * xdd + Fc
ratio = np.abs(F2).max() / np.abs(F1).max()
print(f"  동일한 x(t)를 만드는 두 가설의 최대 힘: H1 {np.abs(F1).max():.2f}N vs H2 {np.abs(F2).max():.2f}N")
print(f"  → 운동학은 같은데 힘은 {ratio:.1f}배 차이. 힘·강성은 vision→action 사상의 null space.")
print("  정지 프레스 구간(xd=xdd=0): 어떤 파지력이든 같은 영상 → 강성/힘 완전 미결정.")

# ---------- fig4: latent action null space — 같은 영상, 다른 힘 ----------
# 정지 프레스 구간을 붙여 힘 미결정을 시각화: t in [1,1.5]에서 x 고정(쥐고 힘만)
t2 = np.linspace(1.0, 1.5, 100)
x_hold = np.full_like(t2, x[-1])            # 위치 고정 = 영상 동일
t_all = np.concatenate([t, t2])
x_all = np.concatenate([x, x_hold])
# 두 가설의 파지력 (프레스 구간): H1 약함, H2 강함 — 영상은 둘 다 동일
Fgrip_all_1 = np.concatenate([F1, np.full_like(t2, 1.0)])
Fgrip_all_2 = np.concatenate([F2, np.full_like(t2, 12.0)])

fig, (axT, axF) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.25]})
axT.plot(t_all, x_all, color='#2c3e50', lw=2.4)
axT.axvspan(1.0, 1.5, color='#f2f4f4', zorder=0)
axT.text(1.25, x[-1] - 0.02, '정지 프레스\n(위치 고정)', fontsize=8.5, ha='center', va='top', color='#7f8c8d')
axT.set_ylabel('관측 x(t)  [m]')
axT.set_title('그림 4 · latent action의 null space — 카메라는 힘·강성을 못 본다')
axT.grid(True, alpha=0.3)
axT.annotate('카메라가 보는 유일한 신호', xy=(0.4, 0.11), fontsize=9, color='#2c3e50')

axF.plot(t_all, Fgrip_all_1, color='#2c6fb0', lw=2, label='가설 H1: 가벼운 블록 · 약한 힘 (peak %.2fN)' % np.abs(F1).max())
axF.plot(t_all, Fgrip_all_2, color='#c0392b', lw=2, label='가설 H2: 무거운 블록 · 숨은 접촉/파지력 (peak %.2fN)' % np.abs(F2).max())
axF.axvspan(1.0, 1.5, color='#fdecea', zorder=0)
axF.fill_between(t2, 1.0, 12.0, color='#e74c3c', alpha=0.12)
axF.text(1.25, 6.5, '같은 영상 →\n어떤 힘이든 가능\n(완전 미결정)', fontsize=8.5, ha='center', color='#c0392b')
axF.set_ylabel('실제 가한 힘  [N]')
axF.set_xlabel('시간  [s]')
axF.legend(loc='upper left', fontsize=8.2)
axF.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + 'fig4_latent_action_nullspace.png', dpi=130)
plt.close(fig)

# ---------- fig3: 계보 그래프 (세 논쟁 축 위의 작은 모델들) ----------
# 노드: (라벨, x=시간축, y=축 위치, 색)  y: 데이터/구조/표현 축을 은유적으로 배치
fig, ax = plt.subplots(figsize=(9.6, 5.8))
ax.axis('off')

# 배경 축 띠 (세 논쟁 축)
bands = [
    (2.55, 3.45, '#eaf2fb', '표현 축 · 연속 헤드 계보'),
    (1.55, 2.45, '#eef7ee', '구조 축 · 경량/분리 계보'),
    (0.55, 1.45, '#fdf0e6', '데이터 축 · 무라벨/커뮤니티 계보'),
]
for y0, y1, col, lab in bands:
    ax.axhspan(y0, y1, color=col, zorder=0)
    ax.text(0.05, (y0 + y1) / 2, lab, fontsize=9.5, va='center', ha='left',
            color='#555', style='italic')

# 노드 정의: name, x(시간), y(축), facecolor
nodes = {
    'RT-2':      (1.0, 3.05, '#d5d8dc'),
    'π0':        (3.5, 3.05, '#f5cba7'),
    'LAPA':      (3.5, 0.95, '#aed6f1'),
    'TinyVLA':   (2.4, 1.75, '#a9dfbf'),
    'CogACT':    (4.2, 1.75, '#a9dfbf'),
    'RDT-1B':    (2.4, 2.62, '#a9dfbf'),
    'SpatialVLA':(5.3, 2.62, '#a9dfbf'),
    'GO-1':      (5.5, 0.95, '#aed6f1'),
    'SmolVLA':   (6.5, 1.75, '#f9e79f'),
    'SimVLA':    (7.6, 2.35, '#d5d8dc'),
}
labels = {
    'RT-2': 'RT-2\n(웹지식·2023.7)',
    'π0': 'π0\n(flow expert·2024.10)',
    'LAPA': 'LAPA\n(latent action·2024.10)',
    'TinyVLA': 'TinyVLA\n(경량+연속 헤드·2024.9)',
    'CogACT': 'CogACT\n(인지/행동 분리·2024.11)',
    'RDT-1B': 'RDT-1B\n(순수 DiT·2024.10)',
    'SpatialVLA': 'SpatialVLA\n(3D prior·2025.1)',
    'GO-1': 'GO-1/ViLLA\n(latent planner·2025.3)',
    'SmolVLA': 'SmolVLA\n(450M·커뮤니티·2025.6)',
    'SimVLA': 'SimVLA\n(미니멀 0.5B·2026.2)',
}
# 엣지: (from, to, style, label)
edges = [
    ('RT-2', 'π0', '-', ''),
    ('π0', 'SmolVLA', '--', '템플릿 경량화'),
    ('TinyVLA', 'SmolVLA', '--', '경량+연속 헤드'),
    ('LAPA', 'GO-1', '-', 'latent action'),
    ('π0', 'RDT-1B', ':', '대조군(VLM 제거)'),
    ('π0', 'CogACT', ':', '분리 vs 공유'),
    ('SmolVLA', 'SimVLA', ':', '"정말 필요한가"'),
]
# 라벨 위치 미세 오프셋 (겹침 방지): edge -> (dx, dy)
lab_off = {
    ('π0', 'SmolVLA'): (-0.55, 0.24),
    ('TinyVLA', 'SmolVLA'): (-0.3, -0.16),
    ('LAPA', 'GO-1'): (0.0, 0.12),
    ('π0', 'RDT-1B'): (-0.15, 0.22),
    ('π0', 'CogACT'): (0.35, 0.12),
    ('SmolVLA', 'SimVLA'): (-0.35, 0.05),
}
for a, b, ls, lab in edges:
    xa, ya, _ = nodes[a]; xb, yb, _ = nodes[b]
    ax.annotate('', xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle='-|>', ls=ls, color='#7f8c8d', lw=1.5,
                                shrinkA=26, shrinkB=26))
    if lab:
        dx, dy = lab_off.get((a, b), (0.0, 0.08))
        ax.text((xa + xb) / 2 + dx, (ya + yb) / 2 + dy, lab, fontsize=7.3,
                color='#566573', ha='center', style='italic')

for name, (x, y, fc) in nodes.items():
    ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
        (x - 0.42, y - 0.20), 0.84, 0.40, boxstyle='round,pad=0.02',
        facecolor=fc, edgecolor='#34495e', lw=1.2, zorder=3))
    ax.text(x, y, labels[name], fontsize=7.3, ha='center', va='center', zorder=4)

ax.set_xlim(0, 8.5)
ax.set_ylim(0.5, 3.5)
ax.set_title('그림 3 · 작은 모델 계보 — 세 논쟁 축 위의 조합들 (SmolVLA로 수렴)',
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT + 'fig3_lineage_graph.png', dpi=130)
plt.close(fig)

# ============================================================================
# Worked Example 표 재현 — 해상도·패치·shuffle 조합별 카메라당 토큰수
# ============================================================================
print("=" * 64)
print("[WE] 해상도·패치·pixel-shuffle 조합별 카메라당 토큰 수")
print("=" * 64)
combos = [
    (896, 14, 1, "고해상 baseline"),
    (768, 16, 1, "중해상 shuffle 없음"),
    (512, 16, 1, "SmolVLA 해상도, shuffle 없음"),
    (512, 16, 2, "shuffle r=2"),
    (512, 16, 4, "SmolVLA 실제 (shuffle r=4)"),
    (384, 16, 3, "저해상 shuffle r=3"),
]
print(f"  {'res':>4} {'patch':>5} {'r':>2} {'토큰/카메라':>10}   설명")
for res, p, r, desc in combos:
    print(f"  {res:>4} {p:>5} {r:>2} {vis_tokens(res, p, r):>10}   {desc}")

print("\n생성 완료: fig1_tokens_vs_resolution.png, fig2_size_vs_memory.png, "
      "fig3_lineage_graph.png, fig4_latent_action_nullspace.png")
