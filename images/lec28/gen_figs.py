# Lec 28 그림 생성 스크립트 — CNN과 시각 표현
# 실행: python3 gen_figs.py  (numpy / scipy / matplotlib, CPU만, 결정론적)
# 개념을 numpy 토이로 재현한다 — 실제 대형 모델/GPU 없음.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.replace('gen_figs.py', '')
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# 공용: 유효(valid) 상관연산 (I*K)[i,j] = ΣΣ I[i+m,j+n]K[m,n]  — E1의 형식
# ----------------------------------------------------------------------------
def corr2d(I, K):
    kh, kw = K.shape
    H, W = I.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(I[i:i + kh, j:j + kw] * K)
    return out


# ============================================================================
# 그림 1: 필터가 만드는 특징맵 — 원본 → conv 출력 (Sobel/Laplacian)
#   WE-1과 계층적 특징(2층 conv)을 시각화
# ============================================================================
# 32x32 토이 장면: 밝은 사각형 물체 하나
img = np.zeros((32, 32))
img[8:24, 10:22] = 1.0                       # 물체
img += 0.04 * rng.standard_normal((32, 32))  # 소량 센서 잡음

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)   # 수직 엣지(가로 기울기)
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)   # 수평 엣지
Klap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)   # 라플라시안(코너/윤곽)

gx = corr2d(img, Kx)
gy = corr2d(img, Ky)
grad = np.sqrt(gx**2 + gy**2)               # 엣지 강도 = 얕은 층 특징
lap = corr2d(img, Klap)

# 2층 conv: |gx| 에 다시 필터 → 조합적 특징 (부품 수준)
K2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], float)
feat2 = corr2d(np.abs(gx), K2)

fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
for a, m, tt in zip(
        ax,
        [img, gx, gy, grad, feat2],
        ['원본 이미지 (32×32)', '층1: 수직엣지 Kx', '층1: 수평엣지 Ky',
         '엣지 강도 √(gx²+gy²)', '층2: 조합 특징']):
    im = a.imshow(m, cmap='gray')
    a.set_title(tt, fontsize=11)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle('하나의 작은 커널을 이미지 전역에 슬라이딩 = "어디서든 같은 특징 검출" (E1)  ·  '
             '층을 쌓으면 엣지→조합 특징 (E2)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + 'fig1_feature_maps.png', dpi=110)
plt.close(fig)


# ============================================================================
# 그림 2: 수용영역(RF) 성장 — 깊이별 RF 크기.  RF_l = RF_{l-1}+(k_l-1)∏s_i
# ============================================================================
def rf_growth(kernels, strides):
    rf, jump, rfs = 1, 1, []
    for k, s in zip(kernels, strides):
        rf = rf + (k - 1) * jump
        jump = jump * s
        rfs.append(rf)
    return rfs

layers = np.arange(1, 9)
rf_plain = rf_growth([3] * 8, [1] * 8)            # k=3, s=1: 선형 성장
rf_pool = rf_growth([3] * 8, [1, 2, 1, 2, 1, 2, 1, 2])  # 풀링(s=2) 삽입: 가속

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(layers, rf_plain, 'o-', label='k=3, stride=1 (선형)', color='#2c6fb0')
ax[0].plot(layers, rf_pool, 's-', label='k=3, 풀링 s=2 삽입 (가속)', color='#c0392b')
ax[0].set_xlabel('층 깊이 l'); ax[0].set_ylabel('수용영역 RF (픽셀)')
ax[0].set_title('수용영역 성장 (E2)')
ax[0].legend(); ax[0].grid(alpha=0.3)
for x, y in zip(layers, rf_pool):
    ax[0].annotate(str(y), (x, y), textcoords='offset points', xytext=(0, 6),
                   ha='center', fontsize=8, color='#c0392b')

# 오른쪽: RF를 이미지 위에 사각형으로 (얕은 층은 작은 창, 깊은 층은 큰 창)
ax[1].imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
cx = 32
for rf, c, lab in zip([3, 9, 21, 45], ['#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'],
                      ['층1 RF=3', '층3 RF=9', '층5 RF=21', '층7 RF=45']):
    r = rf / 2
    ax[1].add_patch(plt.Rectangle((cx - r, cx - r), rf, rf, fill=False,
                                  edgecolor=c, lw=2, label=lab))
ax[1].set_title('깊어질수록 한 뉴런이 보는 창이 커진다\n(얕은 층=국소 엣지, 깊은 층=물체 전체)',
                fontsize=10)
ax[1].legend(loc='upper right', fontsize=8); ax[1].set_xticks([]); ax[1].set_yticks([])
fig.tight_layout()
fig.savefig(OUT + 'fig2_receptive_field.png', dpi=110)
plt.close(fig)


# ============================================================================
# 그림 3: FC vs conv 파라미터 수 (로그 막대) — E1
#   입력 32x32x3 → 같은 출력 텐서(32x32xCout)를 내는 FC vs conv
# ============================================================================
Cin, k, Hh, Ww = 3, 3, 32, 32
couts = [8, 16, 32, 64]
conv_p = [co * (Cin * k * k) + co for co in couts]
in_dim = Hh * Ww * Cin
fc_p = [in_dim * (Hh * Ww * co) + Hh * Ww * co for co in couts]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(couts)); w = 0.38
ax.bar(x - w / 2, fc_p, w, label='완전연결(FC)', color='#c0392b')
ax.bar(x + w / 2, conv_p, w, label='합성곱(conv)', color='#2c6fb0')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels([f'Cout={c}' for c in couts])
ax.set_ylabel('파라미터 수 (로그)')
ax.set_title('같은 출력 크기, FC vs conv 파라미터 수 (E1)\n'
             '32×32×3 입력 → 32×32×Cout 출력', fontsize=11)
for xi, f, c in zip(x, fc_p, conv_p):
    ax.annotate(f'{f/c:,.0f}배', (xi, max(f, c) * 1.3), ha='center',
                fontsize=9, color='#8e44ad')
ax.legend(); ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUT + 'fig3_fc_vs_conv.png', dpi=110)
plt.close(fig)


# ============================================================================
# 그림 4: 잔차 유무 역전파 기울기 노름 vs 깊이 (E3 · WE-2의 심장)
#   plain: h=φ(Wh),  residual: h=h+φ(Wh).  게인<1 → plain은 소멸.
# ============================================================================
def grad_profile(L, d=16, gain=0.7, seed=0, residual=False):
    r = np.random.default_rng(seed)
    Ws = [r.standard_normal((d, d)) * (gain / np.sqrt(d)) for _ in range(L)]
    h = r.standard_normal(d); pre = []
    for l in range(L):
        z = Ws[l] @ h; pre.append(z); a = np.tanh(z)
        h = h + a if residual else a
    g = np.ones(d); gn = [np.linalg.norm(g)]
    for l in reversed(range(L)):
        D = 1 - np.tanh(pre[l])**2
        J = (np.eye(d) + D[:, None] * Ws[l]) if residual else (D[:, None] * Ws[l])
        g = J.T @ g; gn.append(np.linalg.norm(g))
    return np.array(gn[::-1])

L = 50
gp = grad_profile(L, residual=False)
gr = grad_profile(L, residual=True)
gpN = gp / gp[-1]                     # 출력 기울기=1 로 정규화
grN = gr / gr[-1]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].semilogy(np.arange(L + 1), np.maximum(gpN, 1e-20), 'o-', ms=3,
               label='잔차 없음 (plain)', color='#c0392b')
ax[0].semilogy(np.arange(L + 1), grN, 's-', ms=3,
               label='잔차 있음 (residual)', color='#2c6fb0')
ax[0].set_xlabel('층 인덱스 l (0=입력 쪽)')
ax[0].set_ylabel('역전파 기울기 노름 (출력=1 정규화)')
ax[0].set_title('깊이 50에서 층별 기울기 노름 (E3)')
ax[0].legend(); ax[0].grid(alpha=0.3, which='both')

# ratio g0/gL vs depth
depths = [5, 10, 20, 30, 50, 80, 100]
rp = [grad_profile(L, residual=False)[0] / grad_profile(L, residual=False)[-1] for L in depths]
rr = [grad_profile(L, residual=True)[0] / grad_profile(L, residual=True)[-1] for L in depths]
ax[1].semilogy(depths, rp, 'o-', label='잔차 없음', color='#c0392b')
ax[1].semilogy(depths, rr, 's-', label='잔차 있음', color='#2c6fb0')
ax[1].axhline(1.0, color='gray', ls='--', lw=1)
ax[1].set_xlabel('망 깊이 L'); ax[1].set_ylabel('입력층/출력층 기울기 비 (g0/gL)')
ax[1].set_title('깊어질수록 plain은 지수적으로 소멸\n잔차는 O(1)~O(10) 유지')
ax[1].legend(); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(OUT + 'fig4_residual_gradient.png', dpi=110)
plt.close(fig)

# ---- 콘솔 출력: 본문/캡션 수치 확정 ----
print("[fig1] Sobel 물체 이미지 gx max =", round(gx.max(), 3),
      " grad max =", round(grad.max(), 3), " feat2 shape =", feat2.shape)
print("[fig2] RF plain(k=3,s=1):", rf_plain)
print("[fig2] RF pool (s=2):", rf_pool)
print("[fig3] conv params:", conv_p)
print("[fig3] FC params:", fc_p)
print("[fig3] ratios FC/conv:", [round(f / c) for f, c in zip(fc_p, conv_p)])
print("[fig4] plain g0/gL @L=50 :", f"{gp[0]/gp[-1]:.3e}")
print("[fig4] resid g0/gL @L=50 :", f"{gr[0]/gr[-1]:.3e}")
print("[fig4] depth sweep plain :", [f"{v:.2e}" for v in rp])
print("[fig4] depth sweep resid :", [f"{v:.2e}" for v in rr])
print("saved 4 PNGs to", OUT)
