# Lec 33 그림 생성 스크립트 — 사후학습(SFT · RLHF/DPO · LoRA/PEFT)
# 실행: cd images/lec33 && python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
rng = np.random.default_rng(0)
OUT = __file__.replace('gen_figs.py', '')

# 공통 색
C_FROZEN = '#c9c4b8'   # 얼린 가중치(회색)
C_TRAIN  = '#2c6fb0'   # 학습 파라미터(파랑)
C_TRAIN2 = '#e08a1a'   # 학습 파라미터2(주황)
C_DATA   = '#8a7a4a'
C_GOOD   = '#1a8a4a'
C_BAD    = '#c0392b'

# ============================================================================
# 그림 1: 사후학습 파이프라인 (사전학습 → SFT → RLHF/DPO)  — 3단계 블록도
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.3); ax.axis('off')

stages = [
    (0.4, "사전학습\n(Pretraining)", "웹 텍스트 수조 토큰\nnext-token 예측\n→ 원시 지식·문법", '#e1eefb', '#2c6fb0'),
    (4.0, "SFT\n(지도 파인튜닝)", "(지시,응답) 큐레이션\n같은 next-token 손실\n→ 지시-따르기", '#e6f4ea', '#1a8a4a'),
    (7.6, "RLHF / DPO\n(선호 정렬)", "사람 선호쌍 (y+ > y-)\n보상모델+PPO 또는 DPO\n→ 유용·무해·스타일", '#faf0e1', '#e08a1a'),
]
w, h, y0 = 3.0, 2.7, 0.9
for i, (x, title, body, fc, ec) in enumerate(stages):
    box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                         fc=fc, ec=ec, lw=2.2)
    ax.add_patch(box)
    ax.text(x + w/2, y0 + h - 0.42, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color=ec)
    ax.text(x + w/2, y0 + h/2 - 0.35, body, ha='center', va='center', fontsize=10.3, color='#222')
    if i < 2:
        ar = FancyArrowPatch((x + w + 0.02, y0 + h/2), (stages[i+1][0] - 0.02, y0 + h/2),
                             arrowstyle='-|>', mutation_scale=22, lw=2.4, color='#555')
        ax.add_patch(ar)

# 아래: 데이터 규모/비용 화살표
ax.annotate('', xy=(10.6, 0.42), xytext=(0.4, 0.42),
            arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.6))
ax.text(2.0, 0.18, "대량·저품질 (조 단위 토큰)", ha='center', fontsize=9.2, color='#777')
ax.text(9.0, 0.18, "소량·고품질 (수만 예시)", ha='center', fontsize=9.2, color='#777')
ax.text(5.5, 4.05, "새 지식은 대부분 여기서 → 정렬·스타일은 여기서 (지식을 넣는 게 아니다)",
        ha='center', fontsize=10.5, color='#444', style='italic')
ax.text(0.4, 0.62, "지식 대부분 여기", fontsize=8.8, color='#2c6fb0')
fig.tight_layout()
fig.savefig(OUT + 'fig1_pipeline.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print("[fig1] 파이프라인 저장")

# ============================================================================
# 그림 2: LoRA 저랭크 도식 — W 얼림 + B,A 학습, ΔW = BA
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis('off')

# 입력 x
ax.text(0.35, 2.3, "x\n(d)", ha='center', va='center', fontsize=12, fontweight='bold')

# W (얼림) — 큰 정사각
Wx, Wy, Ws = 1.6, 1.0, 2.6
ax.add_patch(FancyBboxPatch((Wx, Wy), Ws, Ws, boxstyle="round,pad=0.02",
                            fc=C_FROZEN, ec='#7a756a', lw=2))
ax.text(Wx + Ws/2, Wy + Ws/2, "W\n(d×d)\n얼림(frozen)", ha='center', va='center',
        fontsize=12.5, fontweight='bold', color='#4a4a4a')

# 하단 경로: A (r×d, 좁음) → B (d×r, 좁음)
Ax, Ay, Aw, Ah = 5.0, 0.55, 0.55, 1.5   # A: d 입력, r 출력 (얇은 세로)
Bx, By, Bw, Bh = 6.4, 0.55, 1.5, 0.55    # B: r 입력, d 출력 (얇은 가로)
ax.add_patch(FancyBboxPatch((Ax, Ay), Aw, Ah, boxstyle="round,pad=0.02",
                            fc=C_TRAIN, ec='#1a4f80', lw=2))
ax.text(Ax + Aw/2, Ay + Ah/2, "A\n(r×d)", ha='center', va='center',
        fontsize=10.5, color='white', fontweight='bold')
ax.add_patch(FancyBboxPatch((Bx, By), Bw, Bh, boxstyle="round,pad=0.02",
                            fc=C_TRAIN2, ec='#a8630f', lw=2))
ax.text(Bx + Bw/2, By + Bh/2, "B (d×r)", ha='center', va='center',
        fontsize=10.5, color='white', fontweight='bold')

# 화살표들
def arrow(x0, y0, x1, y1, color='#555', lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=18, lw=lw, color=color))
arrow(0.7, 2.3, Wx - 0.05, Wy + Ws/2)                 # x → W
# x → A: W 박스 아래를 돌아가는 경로(겹침 방지)
ax.add_patch(FancyArrowPatch((0.35, 2.0), (Ax + Aw/2, Ay + Ah + 0.05),
             connectionstyle="arc3,rad=-0.25", arrowstyle='-|>',
             mutation_scale=18, lw=2.0, color=C_TRAIN))
arrow(Ax + Aw/2, Ay + Ah/2, Bx - 0.05, By + Bh/2, color='#888')  # A → B (내부 r차원)
ax.text((Ax+Aw+Bx)/2, Ay + Ah/2 + 0.28, "r 차원\n(병목)", ha='center', fontsize=8.6, color=C_BAD)

# 합류 ⊕
sumx, sumy = 8.6, 2.3
ax.add_patch(plt.Circle((sumx, sumy), 0.28, fc='white', ec='#333', lw=2))
ax.text(sumx, sumy, "+", ha='center', va='center', fontsize=20, fontweight='bold')
arrow(Wx + Ws + 0.05, Wy + Ws/2, sumx - 0.3, sumy + 0.05)          # W → +
arrow(Bx + Bw + 0.05, By + Bh/2, sumx - 0.2, sumy - 0.28, color=C_TRAIN2)  # BA → +
arrow(sumx + 0.3, sumy, 9.6, sumy)
ax.text(9.75, 2.3, "h", ha='center', va='center', fontsize=13, fontweight='bold')

ax.text(5.0, 4.15, r"$h = Wx + \Delta W\,x = Wx + BA\,x$   (rank $r \ll d$)",
        ha='center', fontsize=14)
ax.text(5.0, 3.7, "회색 W는 얼리고(gradient 0), 파랑 A·주황 B만 학습 → 파라미터 2dr개",
        ha='center', fontsize=10.5, color='#444')
ax.text(Wx + Ws/2, Wy - 0.35, "학습 파라미터 d² (전체)", ha='center', fontsize=9, color='#888')
ax.text(6.2, 0.2, "학습 파라미터 2dr (LoRA)", ha='center', fontsize=9, color=C_TRAIN)
fig.savefig(OUT + 'fig2_lora_schematic.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print("[fig2] LoRA 도식 저장")

# ============================================================================
# 그림 3: (a) 파라미터 절감 막대(로그)  (b) 랭크 r vs 담긴 에너지 비율
# ============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))

# (a) d별 전체 d^2 vs LoRA 2dr (r=8)  로그 막대
ds = [512, 1024, 2048, 4096]
r = 8
full = [d*d for d in ds]
lora = [2*d*r for d in ds]
xx = np.arange(len(ds)); bw = 0.36
axL.bar(xx - bw/2, full, bw, color=C_FROZEN, ec='#7a756a', label=f'전체 파인튜닝 d²')
axL.bar(xx + bw/2, lora, bw, color=C_TRAIN, ec='#1a4f80', label=f'LoRA 2dr (r={r})')
axL.set_yscale('log')
axL.set_xticks(xx); axL.set_xticklabels([f'd={d}' for d in ds])
axL.set_ylabel('가중치 하나당 학습 파라미터 수 (로그)')
axL.set_title('(a) 한 층 W(d×d) 파인튜닝 파라미터: 전체 vs LoRA')
for i, d in enumerate(ds):
    ratio = full[i] / lora[i]
    axL.text(i, full[i]*1.3, f'{ratio:.0f}배↓', ha='center', fontsize=9.5, color=C_BAD, fontweight='bold')
axL.legend(loc='upper left', fontsize=9.5)
axL.grid(axis='y', ls=':', alpha=0.5)

# (b) 실제 파인튜닝 업데이트 ΔW의 저랭크성: SVD 상위 r로 담기는 에너지
# 토이 ΔW: 소수의 지배적 방향 + 잡음 (파인튜닝 업데이트가 저랭크라는 관찰 재현)
d = 128
U, _ = np.linalg.qr(rng.standard_normal((d, d)))
V, _ = np.linalg.qr(rng.standard_normal((d, d)))
# 특이값: 급감(저랭크 구조) + 작은 꼬리 잡음
sv = np.array([10.0/(1+k)**1.1 for k in range(d)]) + 0.02
dW = U @ np.diag(sv) @ V.T
s = np.linalg.svd(dW, compute_uv=False)
energy = np.cumsum(s**2) / np.sum(s**2)
rs = np.arange(1, d+1)
axR.plot(rs, energy, color=C_TRAIN, lw=2.2)
for rr in [1, 2, 4, 8, 16]:
    axR.plot(rr, energy[rr-1], 'o', color=C_TRAIN2, ms=7)
    axR.annotate(f'r={rr}: {energy[rr-1]*100:.1f}%', (rr, energy[rr-1]),
                 textcoords='offset points', xytext=(8, -4), fontsize=8.8, color='#333')
axR.axhline(0.9, ls='--', color=C_GOOD, alpha=0.7)
axR.text(25, 0.855, '90% 에너지선', color=C_GOOD, fontsize=9)
axR.set_xlim(0, 40); axR.set_ylim(0, 1.02)
axR.set_xlabel('유지 랭크 r (SVD 상위 r개 특이값)')
axR.set_ylabel('담긴 에너지 비율 (상위 r개 σ² 합 / 전체)')
axR.set_title('(b) ΔW의 저랭크성: 소수 r로 대부분 에너지')
axR.grid(ls=':', alpha=0.5)
fig.tight_layout()
fig.savefig(OUT + 'fig3_param_savings.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print("[fig3] 파라미터 절감/에너지 저장")
# 출력용 수치
print("  r별 에너지:", {rr: round(float(energy[rr-1]), 4) for rr in [1,2,4,8,16]})

# ============================================================================
# 그림 4: RLHF 보상모델→정책 KL제약 루프  (좌: 개념도, 우: DPO 선호정확도 상승)
# ============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# (좌) 개념 루프 다이어그램
axL.set_xlim(0, 10); axL.set_ylim(0, 8); axL.axis('off')
def box(ax, x, y, w, h, txt, fc, ec, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                 fc=fc, ec=ec, lw=2))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs, color='#222')
def ar(ax, x0, y0, x1, y1, txt='', color='#555', off=0.25):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=17,lw=2,color=color))
    if txt:
        ax.text((x0+x1)/2, (y0+y1)/2 + off, txt, ha='center', fontsize=9, color=color)

box(axL, 0.4, 6.0, 3.0, 1.4, "정책 πθ\n(SFT에서 시작)", '#e1eefb', '#2c6fb0')
box(axL, 6.3, 6.0, 3.2, 1.4, "선호 데이터\ny+ > y- (사람)", '#faf0e1', '#e08a1a')
box(axL, 6.3, 3.2, 3.2, 1.4, "보상모델 rφ\n=학습된 비용함수", '#e6f4ea', '#1a8a4a')
box(axL, 0.4, 3.2, 3.0, 1.4, "정책 갱신 (PPO)\nmax r − β·KL", '#e1eefb', '#2c6fb0')
box(axL, 2.0, 0.5, 5.6, 1.2, "참조 πref (SFT 얼림) — KL 닻", '#f2efe6', '#8a7a4a', fs=10)

ar(axL, 8.0, 6.0, 8.0, 4.65, 'RM 학습', color='#e08a1a', off=0.0)
ar(axL, 6.3, 3.9, 3.4, 3.9, 'r(x,y) 점수', color='#1a8a4a')
ar(axL, 1.9, 4.6, 1.9, 5.95, '갱신 θ', color='#2c6fb0')
ar(axL, 3.4, 6.6, 6.25, 6.6, '응답 y 생성', color='#555')
ar(axL, 1.9, 3.15, 3.5, 1.75, 'β·KL 제약', color=C_BAD, off=-0.35)
ar(axL, 6.1, 1.4, 4.0, 3.15, '', color=C_BAD)
axL.set_title('(a) RLHF 루프: 보상모델 + KL제약 정책최적화', fontsize=11)

# (우) DPO/보상모델 토이: 선호쌍에서 로짓차를 학습 → 선호 정확도 상승
#   각 응답을 8차원 특징 phi로 두고, 숨은 선형 보상 r*=w*·phi 이 존재.
#   사람은 r*(y_a) vs r*(y_b) 로 승자를 라벨(Bradley-Terry, 잡음 포함).
#   우리가 학습하는 보상모델 r_hat = w·phi 를 DPO/BT 손실로 맞춘다.
#   선호 정확도 = 학습 보상이 사람 라벨의 승자를 맞히는 비율 → 50%→고점 상승.
def sigmoid(z): return 1/(1+np.exp(-z))
Dfeat = 8
Ntr = 400
w_star = rng.standard_normal(Dfeat)
phiA = rng.standard_normal((Ntr, Dfeat)); phiB = rng.standard_normal((Ntr, Dfeat))
gap_star = (phiA - phiB) @ w_star                 # 참 보상차
p_true = sigmoid(gap_star)                         # BT: y_a가 이길 확률
win_a = (rng.uniform(size=Ntr) < p_true)           # 잡음 섞인 사람 라벨
# 승자/패자 특징으로 정리 (선호 y+ = 승자)
phi_win = np.where(win_a[:,None], phiA, phiB)
phi_los = np.where(win_a[:,None], phiB, phiA)
dphi = phi_win - phi_los                            # 승자-패자 특징차 (항상 +가 이겨야)
w = rng.standard_normal(Dfeat) * 0.001             # 무작위 초기(정확도 ~50%)
lr = 0.15; beta = 1.0
acc_hist = [np.mean((dphi @ w) > 0)]               # step 0: 학습 전 정확도(~50%)
for step in range(60):
    z = beta * (dphi @ w)                          # 학습 보상차 (승자-패자)
    p = sigmoid(z)                                 # 승자가 이길 확률
    grad = beta * ((p - 1.0)[:,None] * dphi).mean(0)   # BT/DPO 손실 gradient
    w -= lr * grad
    acc_hist.append(np.mean((dphi @ w) > 0))       # 학습 보상이 사람 승자를 맞힌 비율
axR.plot(range(0, 61), np.array(acc_hist)*100, color=C_TRAIN, lw=2.3)
axR.axhline(50, ls='--', color=C_BAD, alpha=0.7); axR.text(2, 51.5, '무작위 50%', color=C_BAD, fontsize=9)
# 사람 라벨 자체의 상한(잡음 때문에 100% 불가): 참 w*로의 정확도
acc_star = np.mean((dphi @ w_star) > 0)
axR.axhline(acc_star*100, ls=':', color=C_GOOD, alpha=0.8)
axR.text(30, acc_star*100+0.6, f'참 보상 상한 {acc_star*100:.0f}%', color=C_GOOD, fontsize=8.8)
axR.set_xlim(0, 60); axR.set_ylim(45, 100)
axR.set_xlabel('보상모델/DPO 학습 스텝'); axR.set_ylabel('선호 정확도 (%)')
axR.set_title('(b) 선호학습 토이: 선호쌍에서 정확도 상승')
axR.grid(ls=':', alpha=0.5)
axR.text(24, 58, f'초기 {acc_hist[0]*100:.0f}% → 최종 {acc_hist[-1]*100:.0f}%',
         fontsize=10, color='#333',
         bbox=dict(boxstyle='round', fc='white', ec='#ccc'))
fig.tight_layout()
fig.savefig(OUT + 'fig4_rlhf_dpo.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print("[fig4] RLHF/DPO 저장")
print(f"  DPO 정확도: 초기 {acc_hist[0]:.3f} → 최종 {acc_hist[-1]:.3f}")
print("모든 그림 생성 완료.")
