# Lec 55 그림 생성 스크립트 — 데이터셋과 수집
# 실행: python3 gen_figs.py   (numpy / matplotlib 만 필요, CPU 전용)
# 개념 재현:
#   (1) 데이터셋 지형도: 규모(에피소드/궤적) vs 다양성(embodiment 축)
#   (2) LeRobotDataset v3 구조: 에피소드→프레임(parquet)+비디오(mp4)+meta(stats)
#   (3) q01/q99 정규화: 원신호→[-1,1], 잘못된 stats(퇴화 차원)로 폭발
#   (4) 수집도구→플랫폼 매핑: action space·좌표계 정합, embodiment gap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
rng = np.random.default_rng(0)

C_BG   = '#f7f7f9'
C_A    = '#2c7fb8'   # 파랑
C_B    = '#d95f0e'   # 주황
C_C    = '#31a354'   # 초록
C_D    = '#756bb1'   # 보라
C_BAD  = '#e34a33'   # 빨강(경고)
C_GREY = '#969696'

# =====================================================================
# FIG 1 — 데이터셋 지형도: 에피소드 규모(로그) vs embodiment 다양성
#   버블 크기 = 대략적 총 시간/규모, 색 = 수집 방식(단일로봇 vs cross-embodiment)
#   수치는 본문 참고문헌 [1]~[5]의 1차 자료 값(코드가 만드는 수치 아님, 시각화 좌표).
# =====================================================================
# (이름, 에피소드/궤적 수, embodiment 종류 수, 버블 규모 힌트, 색, 라벨오프셋)
datasets = [
    ("BridgeData V2", 60096,   1,  60,  C_A, (-40, 20)),   # WidowX 단일 [3]
    ("DROID",         76000,   1,  95,  C_A, (-8, -26)),   # Franka 단일, 350h [2]
    ("RT-1",          130000,  1,  55,  C_A, (14, 14)),    # EDR 단일 [4]
    ("OXE / RT-X",    1000000, 22, 160, C_C, (-40, 26)),   # cross-embodiment [1]
    ("AgiBot World",  1003672, 1,  180, C_B, (14, -4)),    # 단일 플랫폼 대량, 2976h [5]
]
fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.set_facecolor(C_BG)
for name, eps, emb, sz, col, off in datasets:
    ax.scatter(eps, emb, s=sz*22, c=col, alpha=0.55, edgecolors='k', linewidths=1.1, zorder=3)
    ax.annotate(name, (eps, emb), textcoords="offset points", xytext=off,
                fontsize=10.5, fontweight='bold', zorder=4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('에피소드/궤적 수 (로그)', fontsize=11)
ax.set_ylabel('embodiment 종류 수 (로그)', fontsize=11)
ax.set_title('데이터셋 지형도 — 규모 vs embodiment 다양성\n(버블 크기 ∝ 총 수집 시간·규모)',
             fontsize=12.5, fontweight='bold')
ax.set_ylim(0.6, 40)
ax.set_xlim(3e4, 3e6)
ax.grid(True, which='both', ls=':', alpha=0.4, zorder=0)
# 두 축을 나누는 화살표 주석
ax.annotate('cross-embodiment\n(embodiment 축으로 다양성↑)', xy=(1.0e6, 22), xytext=(1.3e5, 15),
            fontsize=9.5, color=C_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_C, lw=1.6))
ax.annotate('단일 플랫폼 대량\n(규모 축으로 성장↑)', xy=(1.0e6, 1.0), xytext=(2.0e5, 1.4),
            fontsize=9.5, color=C_B, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_B, lw=1.6))
fig.tight_layout()
fig.savefig('fig1_dataset_landscape.png', dpi=130)
plt.close(fig)
print("[fig1] dataset landscape saved")

# =====================================================================
# FIG 2 — LeRobotDataset v3 구조도 (numpy로 좌표만 계산, 나머지는 도형)
#   에피소드(논리) → 파일 기반 저장(parquet 샤드 + mp4 샤드) + meta(stats/tasks/episodes)
# =====================================================================
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
ax.set_facecolor('white')

def box(x, y, w, h, color, text, fs=9.5, tc='k', alpha=0.9, bold=True):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                       fc=color, ec='k', lw=1.3, alpha=alpha, zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal', zorder=3)

def arrow(x1, y1, x2, y2, color='k'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=16, lw=1.6, color=color, zorder=1))

ax.text(50, 96, 'LeRobotDataset v3 — 파일 기반 저장 + 관계형 메타데이터',
        ha='center', fontsize=13, fontweight='bold')

# 논리 뷰(위): 에피소드들
box(4, 78, 92, 10, '#dbe9f6',
    '논리 뷰:  episode 0  ·  episode 1  ·  episode 2  ·  …  ·  episode N-1\n(각 에피소드 = 프레임 시퀀스: observation.state / action / images / timestamp)',
    fs=9.5, bold=False)

# 화살표: 논리 → 물리
arrow(30, 78, 22, 64, C_GREY)
arrow(50, 78, 50, 64, C_GREY)
arrow(70, 78, 78, 64, C_GREY)
ax.text(50, 71.5, '여러 에피소드를 큰 파일로 concatenate (샤딩)', ha='center',
        fontsize=8.6, color=C_GREY, style='italic')

# 물리 저장 3기둥
box(4, 44, 28, 18, '#fdd0a2',
    'data/  (Parquet)\nfile-0000.parquet …\n프레임별 저수준 신호\n(state·action·timestamp)\n다수 에피소드/파일', fs=8.6, bold=False)
box(36, 44, 28, 18, '#c7e9c0',
    'videos/  (MP4)\nfile-0000.mp4 …\n카메라별 mp4 샤드\n프레임 concat·인코딩\n스트리밍 가능', fs=8.6, bold=False)
box(68, 44, 28, 18, '#dadaeb',
    'meta/\ninfo.json (스키마·fps)\nstats.json (mean/std/min/max)\ntasks.jsonl · episodes/\n(오프셋·길이)', fs=8.6, bold=False)
ax.text(4, 63.5, '물리 저장', fontsize=9.5, fontweight='bold')

# meta가 나머지를 인덱싱
arrow(68, 53, 64, 53, C_D)
arrow(68, 49, 32, 49, C_D)
ax.text(50, 40, 'meta/episodes 가 각 에피소드의 (파일, 시작·끝 오프셋)을 저장 → 파일 경계≠에피소드 경계',
        ha='center', fontsize=8.4, color=C_D, style='italic')

# 아래: 정책 학습이 소비
box(4, 20, 92, 12, '#f0f0f0',
    '정책 학습 소비:  LeRobotDataset[i] → dict of tensors    ·    StreamingLeRobotDataset (다운로드 없이 Hub 스트리밍)\n'
    'delta_timestamps 로 시간창(관측 히스토리·액션 청크) 슬라이스    ·    stats 로 정규화 (다음 그림)', fs=8.8, bold=False)
arrow(50, 44, 50, 32, 'k')

box(28, 4, 44, 10, '#cfe8cf',
    'meta/stats.json 은 mean/std/min/max 를 저장.\n정책 훈련(openpi 등)은 여기서 q01/q99 분위수를 계산해 정규화 계약을 만든다 (50강).',
    fs=8.3, tc='#204020', bold=False)
arrow(90, 20, 72, 12, C_C)

fig.tight_layout()
fig.savefig('fig2_lerobot_structure.png', dpi=130)
plt.close(fig)
print("[fig2] lerobot structure saved")

# =====================================================================
# FIG 3 — q01/q99 정규화: 정상 vs 잘못된 stats(퇴화 차원 폭발)
#   50강 회수: x_hat = 2(x - q01)/(q99 - q01) - 1
#   (a) 원신호 3개 차원(활발/보통/거의정지) 히스토그램 + q01/q99
#   (b) 올바른 stats로 정규화 → [-1,1] 안. 퇴화 차원에 잘못된(전역) stats 쓰면 폭발.
# =====================================================================
# 세 관절 차원: 큰 진폭, 중간, 거의 정지(퇴화)
N = 4000
d_active = rng.normal(0.0, 0.8, N)                    # 활발한 관절 (rad)
d_mid    = rng.normal(0.2, 0.25, N)                   # 보통
d_frozen = rng.normal(-0.5, 0.004, N)                 # 거의 안 움직임 (퇴화 차원)

def q0199(x):
    return np.quantile(x, 0.01), np.quantile(x, 0.99)

def normalize(x, q01, q99):
    return 2.0 * (x - q01) / (q99 - q01) - 1.0

qa = q0199(d_active); qm = q0199(d_mid); qf = q0199(d_frozen)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

# (a) 원신호 + q01/q99
ax0 = axes[0]
ax0.set_facecolor(C_BG)
for x, col, lab, q in [(d_active, C_A, '활발한 관절', qa),
                       (d_mid, C_C, '보통 관절', qm),
                       (d_frozen, C_D, '거의 정지(퇴화)', qf)]:
    ax0.hist(x, bins=60, color=col, alpha=0.5, density=True, label=f'{lab}')
    ax0.axvline(q[0], color=col, ls='--', lw=1.1, alpha=0.9)
    ax0.axvline(q[1], color=col, ls='--', lw=1.1, alpha=0.9)
ax0.set_title('(a) 원신호와 q01/q99 (점선)', fontsize=11, fontweight='bold')
ax0.set_xlabel('관절각 (rad)', fontsize=10)
ax0.set_ylabel('밀도', fontsize=10)
ax0.legend(fontsize=8.5, loc='upper left')
ax0.text(qf[0], ax0.get_ylim()[1]*0.55, f'퇴화 차원 폭 q99-q01\n= {qf[1]-qf[0]:.4f} rad\n(거의 0)',
         fontsize=8.2, color=C_D, ha='center')

# (b) 정규화 결과: 올바른 stats vs 퇴화 차원에 전역 stats 오용
ax1 = axes[1]
ax1.set_facecolor(C_BG)
# 올바른: 각 차원 자기 stats로
n_active = normalize(d_active, *qa)
n_frozen_ok = normalize(d_frozen, *qf)
# 잘못된: 퇴화 차원에 "활발한 차원의 넓은 stats"를 잘못 적용 (스케일 미스매치)
# — 반대로 흔한 실전 버그: 퇴화 차원의 q01==q99(0폭)로 나눠 폭발.
# 여기선 q99-q01 이 거의 0인 퇴화 차원 자기 stats로 나눌 때,
# 배포 로봇의 값이 훈련 분포에서 아주 살짝만 벗어나도 폭발함을 보인다.
d_frozen_deploy = d_frozen + 0.02                     # 배포 시 미세 오프셋(2 centirad)
n_frozen_bad = normalize(d_frozen_deploy, *qf)        # 퇴화 stats로 나눔 → 폭발

bins = np.linspace(-6, 6, 80)
ax1.hist(np.clip(n_active, -6, 6), bins=bins, color=C_A, alpha=0.6, density=True,
         label='활발 차원 (올바른 stats)')
ax1.hist(np.clip(n_frozen_ok, -6, 6), bins=bins, color=C_C, alpha=0.6, density=True,
         label='퇴화 차원 (훈련분포 그대로)')
ax1.hist(np.clip(n_frozen_bad, -6, 6), bins=bins, color=C_BAD, alpha=0.55, density=True,
         label='퇴화 차원 (배포 +0.02 오프셋)')
ax1.axvline(-1, color='k', ls=':', lw=1.2)
ax1.axvline(1, color='k', ls=':', lw=1.2)
ax1.axvspan(-1, 1, color='green', alpha=0.06)
ax1.set_title('(b) 정규화 후 [-1,1] — 퇴화 차원의 폭발', fontsize=11, fontweight='bold')
ax1.set_xlabel(r'정규화값  $\hat{x}=2(x-q_{01})/(q_{99}-q_{01})-1$', fontsize=10)
ax1.set_ylabel('밀도', fontsize=10)
ax1.legend(fontsize=8.2, loc='upper center')
ax1.set_xlim(-6, 6)
ax1.text(0, ax1.get_ylim()[1]*0.9, '계약 범위\n[-1,1]', ha='center', fontsize=8.5, color='green')

fig.tight_layout()
fig.savefig('fig3_normalization.png', dpi=130)
plt.close(fig)
print("[fig3] normalization saved")
# 수치 출력 (본문 캡션과 일치시킬 값)
print(f"  q01/q99 active = ({qa[0]:.3f},{qa[1]:.3f}) width={qa[1]-qa[0]:.3f}")
print(f"  q01/q99 frozen = ({qf[0]:.4f},{qf[1]:.4f}) width={qf[1]-qf[0]:.5f}")
print(f"  frozen 배포오프셋 0.02rad → 정규화값 범위 [{n_frozen_bad.min():.1f}, {n_frozen_bad.max():.1f}]")
print(f"  frozen 배포오프셋 중앙값 정규화 = {np.median(n_frozen_bad):.1f} (계약 [-1,1] 밖)")

# =====================================================================
# FIG 4 — 수집도구 → 플랫폼 매핑 (action space·좌표계 정합, embodiment gap)
# =====================================================================
fig, ax = plt.subplots(figsize=(9.4, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
ax.text(50, 96, '수집 도구 → 데이터 표현 → 배포 로봇 (embodiment gap)',
        ha='center', fontsize=13, fontweight='bold')

tools = [
    ("ALOHA\n(양팔 텔레옵)", "리더=팔로워\n동일 기구\n절대 관절각 14D", "gap ≈ 0\n(리더≅팔로워)", 80, C_A),
    ("GELLO\n(운동학 등가 리더)", "관절 매핑\n절대 관절각\n(IK 불필요)", "gap 작음\n(리더=타깃 복제)", 62, C_C),
    ("SO-101\n(리더-팔로워)", "절대 관절각 6D\nFeetech 레지스터", "gap 작음\n(동일 팔로워)", 44, C_C),
    ("UMI\n(핸드헬드 그리퍼)", "상대 EEF 궤적\n(로봇 무관)\nGoPro 관측", "gap 큼→상쇄\n지연매칭·상대표현\n하드웨어 무관", 26, C_B),
]
for name, repr_, gap, y, col in tools:
    box_w, box_h = 24, 14
    # 도구
    r = FancyBboxPatch((3, y-box_h/2), box_w, box_h, boxstyle="round,pad=0.3,rounding_size=1.2",
                       fc=col, ec='k', lw=1.2, alpha=0.35, zorder=2)
    ax.add_patch(r); ax.text(3+box_w/2, y, name, ha='center', va='center', fontsize=9.2, fontweight='bold')
    # 표현
    r2 = FancyBboxPatch((37, y-box_h/2), box_w, box_h, boxstyle="round,pad=0.3,rounding_size=1.2",
                        fc='#f0f0f0', ec='k', lw=1.0, zorder=2)
    ax.add_patch(r2); ax.text(37+box_w/2, y, repr_, ha='center', va='center', fontsize=8.3)
    # gap
    gap_fc = '#fee0d2' if 'gap 큼' in gap else '#e5f5e0'   # 큰 gap만 붉게
    r3 = FancyBboxPatch((71, y-box_h/2), box_w, box_h, boxstyle="round,pad=0.3,rounding_size=1.2",
                        fc=gap_fc, ec='k', lw=1.0, zorder=2)
    ax.add_patch(r3); ax.text(71+box_w/2, y, gap, ha='center', va='center', fontsize=8.0)
    ax.add_patch(FancyArrowPatch((27, y), (37, y), arrowstyle='-|>', mutation_scale=13, lw=1.4, color='k'))
    ax.add_patch(FancyArrowPatch((61, y), (71, y), arrowstyle='-|>', mutation_scale=13, lw=1.4, color='k'))

ax.text(15, 90, '수집 도구', ha='center', fontsize=10, fontweight='bold')
ax.text(49, 90, '데이터 action space', ha='center', fontsize=10, fontweight='bold')
ax.text(83, 90, 'embodiment gap', ha='center', fontsize=10, fontweight='bold')
ax.text(50, 10, '핵심: 절대 관절각(ALOHA·GELLO·SO-101)은 리더=팔로워라 gap이 작지만 embodiment에 묶인다.\n'
        'UMI의 상대 EEF는 로봇에 무관해 gap을 표현·지연매칭으로 상쇄한다 — 좌표계·action space 정합이 데이터 품질의 계약.',
        ha='center', fontsize=8.6, style='italic', color='#333')
fig.tight_layout()
fig.savefig('fig4_collection_mapping.png', dpi=130)
plt.close(fig)
print("[fig4] collection mapping saved")
print("ALL FIGS DONE")
