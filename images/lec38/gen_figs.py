"""
Lec 38 (ACT · action chunking) 그림 생성.
순수 numpy/scipy/matplotlib. 결정론적 시드. 개념 재현용 CPU 토이 — 실제 ACT/ALOHA 모델 아님.
cd images/lec38 && python3 gen_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                     'figure.dpi': 120, 'savefig.dpi': 120})

C_TRUE = '#333333'; C_CHUNK = '#2c6fb0'; C_NAIVE = '#c0392b'; C_ENS = '#1a7a4a'
C_BOX = '#e1eefb'; C_BOX2 = '#efe4d8'; C_Z = '#f3eede'

# =====================================================================
# 공통 토이 (WE와 동일 수식)
# =====================================================================
def rollout_drift(H, eps, dist, T=120, k=0.30, x_goal=1.0, n_trials=600, seed=0):
    rng = np.random.default_rng(seed)
    drifts = []
    for _ in range(n_trials):
        x, traj, t = 0.0, [0.0], 0
        while t < T:
            dev  = abs(x - x_goal)
            kick = eps * (1.0 + 2.0*dev) * rng.standard_normal()
            x_int = x
            for h in range(H):
                if t >= T: break
                a = k*(x_goal - x_int) + (kick if h == 0 else 0.0)
                x_int += a
                x     += a + dist*rng.standard_normal()
                traj.append(x); t += 1
        traj = np.array(traj)
        drifts.append(np.sqrt(np.mean((traj[T//2:] - x_goal)**2)))
    return float(np.mean(drifts))

def make_chunks(T, L, noise, seed):
    rng = np.random.default_rng(seed)
    tt = np.arange(T + L)
    truth = 0.5*np.sin(2*np.pi*0.015*tt)
    chunks = {}
    for s in range(T):
        off = noise*rng.standard_normal()
        chunks[s] = truth[s:s+L] + off
    return truth[:T], chunks

def ensemble(T, L, chunks, m):
    # ACT 논문 규약: w_i = e^{-mi}, i=0 이 '가장 오래된' 예측(가중 1). 최신일수록 i↑·가중↓.
    out = np.zeros(T)
    for t in range(T):
        srange = [s for s in range(max(0, t-L+1), t+1) if s in chunks and t < s + L]
        preds = np.array([chunks[s][t-s] for s in srange])   # s 오름차순 = 오래된 것 먼저
        i = np.arange(len(srange))                            # i=0: 가장 오래된 예측
        w = np.exp(-m*i); w /= w.sum()
        out[t] = (w*preds).sum()
    return out

def naive_latest(T, L, chunks):
    return np.array([chunks[t][0] for t in range(T)])

def jerk(x):
    return np.sqrt(np.mean(np.diff(x, n=3)**2))


# =====================================================================
# FIG 1: H스텝 청크 예측 도식 — 관측 1회로 H스텝을 한꺼번에 낸다
# =====================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    T = 12
    tt = np.arange(T+1)
    truth = 0.5*(1 - np.cos(np.pi*tt/T))            # S자 셋포인트 이동
    ax.plot(tt, truth, '-', color=C_TRUE, lw=2.2, alpha=0.35, label='참(전문가) 궤적', zorder=1)

    # per-step (H=1): 매 스텝 관측→행동
    for t in range(0, T, 1):
        ax.scatter([t],[truth[t]], s=26, color=C_NAIVE, zorder=4)
    ax.scatter([],[], s=26, color=C_NAIVE, label='per-step 예측 (H=1): 매 스텝 관측·재조준')

    # chunk (H=4): 관측 1회로 4스텝 청크
    H = 4
    for s in range(0, T, H):
        seg = np.arange(s, min(s+H, T)+1)
        ax.plot(seg, truth[seg] + 0.055, 'o-', color=C_CHUNK, lw=2.3, ms=6, zorder=5)
        # 관측 시점 표시
        ax.annotate('관측', (s, truth[s]+0.055), textcoords='offset points',
                    xytext=(0, 16), ha='center', fontsize=8.5, color=C_CHUNK,
                    arrowprops=dict(arrowstyle='->', color=C_CHUNK, lw=1.2))
        # 개루프 구간 음영
        ax.axvspan(s, min(s+H, T), color=C_CHUNK, alpha=0.05, zorder=0)
    ax.plot([],[], 'o-', color=C_CHUNK, label='청크 예측 (H=4): 관측 1회 → 4스텝 한 번에')

    ax.set_xlabel('시간 스텝 t'); ax.set_ylabel('관절각 (정규화)')
    ax.set_title('action chunking: 의사결정 빈도 T → T/H, 관측 사이는 개루프 실행',
                 fontsize=11.5)
    ax.legend(loc='lower right', fontsize=8.6, framealpha=0.95)
    ax.set_ylim(-0.15, 1.28); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig('fig1_chunking_schematic.png'); plt.close(fig)
    print("fig1 저장")


# =====================================================================
# FIG 2: 드리프트 vs H — chunking 이득과 개루프 트레이드오프
# =====================================================================
def fig2():
    Hs = [1, 2, 4, 8, 16, 32]
    d_free = [rollout_drift(H, 0.06, 0.0)  for H in Hs]
    d_dist = [rollout_drift(H, 0.06, 0.03) for H in Hs]
    print("  H:", Hs)
    print("  외란0   :", [round(v,4) for v in d_free])
    print("  외란0.03:", [round(v,4) for v in d_dist])

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(Hs, d_free, 'o-', color=C_ENS, lw=2.2, ms=7,
            label='외란 없음: chunking 순이득 (드리프트 ~ 1/√H)')
    ax.plot(Hs, d_dist, 's-', color=C_NAIVE, lw=2.2, ms=7,
            label='외란 있음: U자 — 큰 H는 개루프로 손해')
    imin = int(np.argmin(d_dist))
    ax.scatter([Hs[imin]],[d_dist[imin]], s=160, facecolors='none',
               edgecolors=C_NAIVE, lw=2.2, zorder=5)
    ax.annotate(f'최적 H≈{Hs[imin]}\n(드리프트 최소)', (Hs[imin], d_dist[imin]),
                textcoords='offset points', xytext=(24, 18), fontsize=9, color=C_NAIVE)
    ax.set_xscale('log', base=2); ax.set_xticks(Hs); ax.set_xticklabels(Hs)
    ax.set_xlabel('청크 길이 H (로그축)'); ax.set_ylabel('정상상태 드리프트 (RMS)')
    ax.set_title('청크가 유효 지평선을 T→T/H로 줄이지만, 개루프 길이는 트레이드오프',
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig('fig2_error_vs_H.png'); plt.close(fig)
    print("fig2 저장")


# =====================================================================
# FIG 3: CVAE 구조 — 인코더 q(z|a,o) / 잠재 z / 디코더 p(a|z,o)
# =====================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec='#555', lw=1.3))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2, text='', off=0.18):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                     mutation_scale=15, color='#444', lw=1.5))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2+off, text, ha='center', fontsize=8.6, color='#333')

    # 훈련(윗줄): a*, o -> 인코더 -> z ~ N(mu,sigma) -> 디코더 (+o) -> a_hat
    ax.text(5, 5.72, '훈련: q(z | a*, o) 로 스타일 z를 흡수 → p(a | z, o) 로 청크 재구성',
            ha='center', fontsize=10.5, weight='bold')
    box(0.2, 3.6, 1.7, 1.0, '전문가 청크 a*\n+ 관측 o', C_BOX2, 9)
    box(2.5, 3.6, 1.8, 1.0, '인코더\nq(z | a*, o)', C_BOX)
    box(5.0, 3.7, 1.5, 0.85, '잠재 z\n~ N(μ, σ)', C_Z)
    box(7.1, 3.6, 1.8, 1.0, '디코더\np(a | z, o)', C_BOX)
    box(9.0, 3.7, 0.85, 0.85, 'â\n청크', C_BOX2, 8.5)
    arrow(1.9, 4.1, 2.5, 4.1)
    arrow(4.3, 4.1, 5.0, 4.12, 'μ, σ')
    arrow(6.5, 4.12, 7.1, 4.1, '재파라미터화')
    arrow(8.9, 4.1, 9.0, 4.12)
    # o 를 디코더에도
    ax.add_patch(FancyArrowPatch((1.05, 3.6), (8.0, 3.6), connectionstyle="arc3,rad=-0.28",
                 arrowstyle='-|>', mutation_scale=13, color='#9a6a3a', lw=1.3, ls='--'))
    ax.text(4.6, 2.55, '관측 o 는 디코더에도 조건으로 (그래서 "조건부" VAE)',
            ha='center', fontsize=8.8, color='#9a6a3a')

    # 손실
    box(0.6, 0.5, 4.0, 1.0, 'ELBO 손실:\n재구성 ‖a* − â‖²  +  β·KL(q ‖ N(0,I))', '#f7f7f7', 9)
    box(5.2, 0.5, 4.2, 1.0, '추론(배포): z = 0 (사전평균) 고정\n→ 디코더가 결정적 청크 생성', '#eef7ee', 9)
    ax.text(2.6, 1.72, 'z=스타일 파라미터: 다봉·변동을 흡수 → mode averaging 회피',
            ha='center', fontsize=8.6, color='#2c6fb0')

    fig.tight_layout(); fig.savefig('fig3_cvae_structure.png'); plt.close(fig)
    print("fig3 저장")


# =====================================================================
# FIG 4: temporal ensembling 블렌딩 — 겹치는 청크 → 매끈
# =====================================================================
def fig4():
    T, L, noise = 80, 20, 0.04
    truth, chunks = make_chunks(T, L, noise, seed=0)
    nv = naive_latest(T, L, chunks)
    en = ensemble(T, L, chunks, m=0.01)
    j_nv, j_en = jerk(nv), jerk(en)
    print(f"  jerk naive={j_nv*1e3:.2f}e-3  ens(m=0.01)={j_en*1e3:.2f}e-3  ratio={j_nv/j_en:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3),
                                   gridspec_kw={'width_ratios':[1.7, 1]})
    tt = np.arange(T)
    # 겹치는 몇몇 청크 예측을 옅게
    for s in range(0, T, 6):
        seg = np.arange(s, min(s+L, T))
        ax1.plot(seg, chunks[s][:len(seg)], '-', color=C_CHUNK, alpha=0.18, lw=1.0)
    ax1.plot([],[], '-', color=C_CHUNK, alpha=0.4, lw=1.2, label='겹치는 청크 예측들')
    ax1.plot(tt, truth, '-', color=C_TRUE, lw=1.6, alpha=0.5, label='참 궤적')
    ax1.plot(tt, nv, '-', color=C_NAIVE, lw=1.6,
             label=f'naive(최신 청크만): 저크 {j_nv*1e3:.0f}e-3')
    ax1.plot(tt, en, '-', color=C_ENS, lw=2.2,
             label=f'지수가중 앙상블(m=0.01): 저크 {j_en*1e3:.1f}e-3')
    ax1.set_xlabel('시간 스텝 t'); ax1.set_ylabel('관절각 (rad)')
    ax1.set_title(f'temporal ensembling: 이음매 저크 {j_nv/j_en:.0f}배 감소', fontsize=10.5)
    ax1.legend(fontsize=8.4, loc='upper right'); ax1.grid(alpha=0.25)

    # 가중 프로파일
    ages = np.arange(0, L)
    for m, col in [(0.01, C_ENS), (0.1, '#2c6fb0'), (0.5, '#c0392b')]:
        w = np.exp(-m*ages); w /= w.sum()
        ax2.plot(ages, w, 'o-', color=col, ms=4, lw=1.8, label=f'm={m}')
    ax2.set_xlabel('예측 인덱스 i (0=가장 오래된 예측, ACT 규약)'); ax2.set_ylabel('정규 가중 w_i (합=1)')
    ax2.set_title('w_i = e^{−m·i} (저역필터 폭 = m)', fontsize=10)
    ax2.legend(fontsize=8.6); ax2.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig('fig4_temporal_ensembling.png'); plt.close(fig)
    print("fig4 저장")


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
    print("== 모든 그림 저장 완료 ==")
