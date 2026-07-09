#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lec 56. LeRobot 딥다이브 — 그림 생성 및 Worked Example 수치 검증.

순수 numpy/scipy/matplotlib (CPU, 결정론적 시드). torch/LeRobot/HF 미사용.
그림 4장 + WE-1/WE-2 수치를 한 스크립트에서 계산해 본문 수치와 정확히 일치시킨다.

실행:  python3 gen_figs.py
출력:  fig1_architecture.png, fig2_async_timing.png,
       fig3_policy_zoo.png, fig4_pipeline.png
       + 표준출력에 WE-1/WE-2 검증 수치.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---- 폰트/렌더 결정론 ----
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["savefig.dpi"] = 120

C_DATA = "#2563eb"   # dataset - blue
C_POL  = "#dc2626"   # policy  - red
C_ENV  = "#16a34a"   # env     - green
C_LOOP = "#7c3aed"   # loop    - purple
C_GRAY = "#6b7280"
C_BG   = "#f3f4f6"

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))


# =====================================================================
# WE-1: async 청크 큐 시뮬 (0강 WE-2 확장, RTC 유무 비교)
# =====================================================================
def we1_async_queue():
    """
    정책 주기(느림) vs 제어 주기(빠름). 제어 루프는 매 tick 큐에서 액션 하나를
    소비한다. 큐 잔량이 threshold*H 아래로 내려가면 새 관측을 정책 서버로 보내고,
    정책은 latency tick 뒤 새 청크 H개를 돌려준다. 큐 언더런(=로봇 idle) 을 센다.
    """
    H = 50               # actions_per_chunk (LeRobot SmolVLA 기본)
    n_ticks = 600        # 제어 tick 수 (예: 30Hz면 20초)

    def run(threshold, latency):
        queue = list(range(H))       # 초기 청크 (부팅 시 이미 하나 있음)
        pending = None               # 도착 예정 tick (None=요청 없음)
        n_req = starved = 0
        qlog = []
        for tick in range(n_ticks):
            # 1) 도착: 잔량 위에 새 청크의 '겹치지 않는 꼬리'를 이어붙인다
            if pending is not None and tick >= pending:
                need = H - len(queue)
                if need > 0:
                    queue += list(range(need))     # weighted_average 겹침 병합 자리
                pending = None
            # 2) 소비: 제어 루프가 매 tick 하나 실행
            if queue:
                queue.pop(0)
            else:
                starved += 1                       # 큐 고갈 = ZOH 홀드 = 멈칫
            qlog.append(len(queue))
            # 3) 재요청: 임계 미만이고 진행 중 요청 없으면
            if pending is None and len(queue) < threshold * H:
                pending = tick + latency
                n_req += 1
        return n_req, starved, float(np.mean(qlog)), qlog

    # (a) threshold 스윕 (정상 지연 lat=6 tick)
    rows = []
    for th in (0.3, 0.5, 0.7, 0.9):
        r = run(th, 6)
        rows.append((th, r[0], r[1], r[2]))

    # (b) 지연 악화 (threshold 0.7 고정): RTC/큐가 지연을 못 흡수하면 언더런
    lat_rows = []
    for lat in (6, 20, 40, 55):
        r = run(0.7, lat)
        lat_rows.append((lat, r[0], r[1], r[2]))

    # 타이밍 그림용 큐 궤적 두 개
    _, _, _, q_good = run(0.7, 6)     # 지연 흡수 성공
    _, _, _, q_bad = run(0.7, 55)     # 지연이 청크 길이에 육박 → 언더런

    print("=" * 62)
    print("WE-1: async 청크 큐 (H=%d, n_ticks=%d)" % (H, n_ticks))
    print("-" * 62)
    print(" (a) threshold 스윕 (lat=6 tick):")
    for th, nq, st, avg in rows:
        print("     th=%.1f | 재요청 %2d회 | idle %3d tick | 평균잔량 %4.1f/%d"
              % (th, nq, st, avg, H))
    print(" (b) 지연 악화 (th=0.7 고정):")
    for lat, nq, st, avg in lat_rows:
        print("     lat=%2d tick | 재요청 %2d회 | idle %3d tick | 평균잔량 %4.1f/%d"
              % (lat, nq, st, avg, H))
    return dict(H=H, n_ticks=n_ticks, rows=rows, lat_rows=lat_rows,
               q_good=q_good, q_bad=q_bad)


# =====================================================================
# WE-2: 정규화 → 더미 정책 → 역정규화 → temporal ensembling 파이프라인 토이
#        + stats 불일치 시 실패 재현
# =====================================================================
def we2_pipeline():
    rng = np.random.default_rng(0)
    D = 6                      # action 차원 (SO-101 6관절)
    H = 16                     # 청크 길이 (토이)

    # --- 데이터셋 통계: 차원별 q01/q99 (dataset.meta.stats 자리) ---
    # 각 차원 스케일이 크게 다르다 (관절각 rad vs 그리퍼 정규화값)
    lo = np.array([-2.6, -1.8, -2.5, -1.6, -3.0, -0.1])   # q01
    hi = np.array([ 2.6,  1.8,  2.5,  1.6,  3.0,  1.1])    # q99

    def normalize(a, lo, hi):
        # 50강/55강 규약: [lo,hi] → [-1,1]
        return 2.0 * (a - lo) / (hi - lo) - 1.0

    def denormalize(a_n, lo, hi):
        return (a_n + 1.0) / 2.0 * (hi - lo) + lo

    # --- 물리 단위 원시 액션 청크 (정책이 목표로 하는 참값) ---
    tt = np.linspace(0, 1, H)
    raw = np.stack([
        1.8 * np.sin(2 * np.pi * 0.8 * tt),
        1.2 * np.sin(2 * np.pi * 0.5 * tt + 0.5),
        2.0 * np.sin(2 * np.pi * 0.6 * tt + 1.0),
        1.0 * np.cos(2 * np.pi * 0.7 * tt),
        2.4 * np.sin(2 * np.pi * 0.4 * tt),
        0.5 + 0.4 * np.sin(2 * np.pi * 0.9 * tt),
    ], axis=1)  # (H, D)

    # === 정상 경로 ===
    # 1) 정규화 (참값을 정규화하면 정책이 배운 [-1,1] 공간의 타깃)
    tgt_n = normalize(raw, lo, hi)
    # 2) 더미 정책: 정규화 공간에서 타깃 + 소량 잡음 (실제 정책 출력 흉내)
    pred_n = tgt_n + rng.normal(0, 0.03, size=tgt_n.shape)
    pred_n = np.clip(pred_n, -1.0, 1.0)     # [-1,1] 클리핑 (정규화 이점)
    # 3) 역정규화 (같은 stats!)
    pred_raw = denormalize(pred_n, lo, hi)

    err_norm = np.abs(pred_n - tgt_n).max()
    err_raw = np.abs(pred_raw - raw).max()

    # === stats 불일치 실패 재현 ===
    # 배포 시 잘못된(다른 로봇의) stats로 역정규화 → 스케일/오프셋 붕괴
    lo_wrong = lo * 0.5          # 관절 범위를 절반으로 오인
    hi_wrong = hi * 0.5
    pred_raw_bad = denormalize(pred_n, lo_wrong, hi_wrong)
    err_raw_bad = np.abs(pred_raw_bad - raw).max()

    # === temporal ensembling (50강 E1 회수) ===
    # 매 스텝 추론했다고 가정: 시각 k를 겨냥한 예측 여러 개 (나이 i)
    # 여기선 한 차원(관절 0)에서 지터 감소를 수치화
    FPS = 30.0
    Ttot = 3.0
    ts = np.arange(0, Ttot, 1 / FPS)
    Nt = len(ts)
    true = 0.9 * np.sin(2 * np.pi * 0.5 * ts) + 0.2 * np.sin(2 * np.pi * 1.4 * ts + 0.6)
    Hc, m, sig = 16, 0.1, 0.04
    preds = [[] for _ in range(Nt)]
    rng2 = np.random.default_rng(1)
    for s in range(Nt):
        bias = rng2.normal(0, sig)              # 청크별 표본 바이어스
        for j in range(Hc):
            k = s + j
            if k >= Nt:
                break
            preds[k].append((k - s, true[k] + bias + rng2.normal(0, sig * 0.5)))

    def ens(k):
        ages = np.array([a for a, _ in preds[k]])
        vals = np.array([v for _, v in preds[k]])
        w = np.exp(-m * ages)
        w /= w.sum()
        return (w * vals).sum()

    naive = np.array([min(preds[k])[1] for k in range(Nt)])   # 최신 청크(나이 0)만
    ens_v = np.array([ens(k) for k in range(Nt)])
    jit = lambda x: np.sqrt(np.mean(np.diff(x, 2) ** 2))
    j_naive, j_ens = jit(naive), jit(ens_v)

    print("=" * 62)
    print("WE-2: 정규화→정책→역정규화→앙상블 파이프라인 (D=%d, H=%d)" % (D, H))
    print("-" * 62)
    print(" 정상 경로 (같은 stats):")
    print("   정규화 공간 max|err| = %.4f  (정책 잡음+클리핑)" % err_norm)
    print("   역정규화 후 max|err| = %.4f rad  (참값 복원 성공)" % err_raw)
    print(" stats 불일치 (역정규화 stats를 0.5배로 오인):")
    print("   역정규화 후 max|err| = %.4f rad  (붕괴 — %.1f배 악화)"
          % (err_raw_bad, err_raw_bad / err_raw))
    print(" temporal ensembling (관절0, m=%.1f):" % m)
    print("   지터 naive=%.2fe-3, ens=%.2fe-3, 비율=%.1fx"
          % (j_naive * 1e3, j_ens * 1e3, j_naive / j_ens))
    return dict(D=D, H=H, lo=lo, hi=hi, raw=raw, tgt_n=tgt_n, pred_n=pred_n,
                pred_raw=pred_raw, pred_raw_bad=pred_raw_bad,
                err_norm=err_norm, err_raw=err_raw, err_raw_bad=err_raw_bad,
                ts=ts, true=true, naive=naive, ens_v=ens_v,
                j_naive=j_naive, j_ens=j_ens, m=m)


# =====================================================================
# 그림 1: LeRobot 아키텍처 (dataset / policy / env / loop)
# =====================================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color, fc=None, fs=10, bold=True):
        fc = fc if fc else "white"
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                           ec=color, fc=fc, lw=2.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#111827",
                fontweight="bold" if bold else "normal")

    ax.text(50, 96, "LeRobot 아키텍처 = 표준 인터페이스로 붙인 4개 축",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(50, 91.5, "(0강 인터페이스 계약의 구현체 — 정책은 교체 가능)",
            ha="center", fontsize=9, color=C_GRAY)

    # 4 축
    box(4, 68, 26, 15,
        "LeRobotDataset (v3)\n멀티에피소드 parquet+mp4\nmeta/stats.json (q01/q99)\ndelta_timestamps",
        C_DATA, fc="#eff6ff", fs=8.5)
    box(37, 68, 26, 15,
        "PreTrainedPolicy\nACT · Diffusion · VQ-BeT\nπ0 · SmolVLA\nselect_action / forward",
        C_POL, fc="#fef2f2", fs=8.5)
    box(70, 68, 26, 15,
        "Env (gym)\nLIBERO · Meta-World\nMuJoCo / 실물 Robot\nstep / reset",
        C_ENV, fc="#f0fdf4", fs=8.5)

    # factory 층
    box(20, 47, 60, 10,
        "config + factory  —  make_policy() / make_dataset() / make_env()\n"
        "PreTrainedConfig 로 정책 종류 선택 (--policy.type / --policy.path)",
        C_GRAY, fc=C_BG, fs=9)

    # 학습 루프 / 추론 루프
    box(6, 24, 40, 13,
        "학습 루프 (lerobot-train)\nbatch = dataset.sample()\n"
        "loss = policy.forward(batch)\nloss.backward(); opt.step()",
        C_LOOP, fc="#faf5ff", fs=8.5)
    box(54, 24, 40, 13,
        "추론 루프 (record / eval / async)\nobs = env/robot.get_observation()\n"
        "act = policy.select_action(obs)\nenv/robot.send_action(act)",
        C_LOOP, fc="#faf5ff", fs=8.5)

    # 하단 메시지
    box(20, 6, 60, 10,
        "같은 API → ACT/DP/π0/SmolVLA/GR00T 를 코드 거의 그대로 교체\n"
        "학습 <100줄 · 추론 ~40줄 (LeRobot 논문)",
        "#111827", fc="#fffbeb", fs=9)

    def arrow(x1, y1, x2, y2, color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14, lw=1.6, color=color))

    # dataset/policy/env → factory
    arrow(17, 68, 30, 57)
    arrow(50, 68, 50, 57)
    arrow(83, 68, 70, 57)
    # factory → loops
    arrow(35, 47, 26, 37)
    arrow(65, 47, 74, 37)
    # loops → bottom
    arrow(26, 24, 40, 16)
    arrow(74, 24, 60, 16)

    fig.tight_layout()
    out = HERE + "/fig1_architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# 그림 2: async 타이밍 (정책 주기 vs 제어 주기 · 큐 잔량 · 언더런)
# =====================================================================
def fig2_async_timing(we1):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2),
                             gridspec_kw=dict(height_ratios=[1.0, 1.15]))

    # --- (a) 타이밍 다이어그램: 제어 tick(빠름) vs 정책 추론(느림) ---
    ax = axes[0]
    ax.set_title("(a) 다중주기: 제어 루프(30Hz)는 매 tick 큐를 소비, 정책은 청크를 비동기로 채운다",
                 fontsize=10.5, fontweight="bold")
    # 제어 tick
    for k in range(30):
        ax.plot([k, k], [2.4, 3.0], color=C_LOOP, lw=1.0)
    ax.text(-1.2, 2.7, "제어\ntick", ha="right", va="center", fontsize=9, color=C_LOOP)
    # 정책 추론 블록 (느림, ~6 tick)
    infer_blocks = [(2, 6), (14, 6), (26, 6)]
    for (s, w) in infer_blocks:
        ax.add_patch(Rectangle((s, 1.1), w, 0.6, fc="#fecaca", ec=C_POL, lw=1.4))
        ax.text(s + w / 2, 1.4, "정책 추론\n(~100ms)", ha="center", va="center",
                fontsize=7.5, color="#7f1d1d")
    ax.text(-1.2, 1.4, "정책\n서버", ha="right", va="center", fontsize=9, color=C_POL)
    # 청크 배달 화살표
    for (s, w) in infer_blocks:
        ax.add_patch(FancyArrowPatch((s + w, 1.4), (s + w + 0.3, 2.4),
                     arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C_ENV))
    ax.text(9, 0.4, "추론이 도는 동안에도 제어 tick은 멈추지 않는다 — 큐가 그 간극을 흡수",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")
    ax.set_xlim(-4, 33)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    # --- (b) 큐 잔량 궤적: 지연 흡수 성공 vs 언더런 ---
    ax = axes[1]
    H = we1["H"]
    q_good = np.array(we1["q_good"])
    q_bad = np.array(we1["q_bad"])
    x = np.arange(len(q_good))
    ax.plot(x, q_good, color=C_ENV, lw=1.8,
            label="지연 흡수 성공 (th=0.7, lat=6 tick): idle 0")
    ax.plot(x, q_bad, color=C_POL, lw=1.8,
            label="지연 과다 (th=0.7, lat=55 tick): 언더런")
    ax.axhline(0.7 * H, color=C_GRAY, ls="--", lw=1.1)
    ax.text(len(x) * 0.55, 0.7 * H + 1.2, "재요청 임계 = 0.7·H = %.0f" % (0.7 * H),
            fontsize=8.5, color=C_GRAY)
    # 언더런 구간(잔량 0) 강조
    under = q_bad == 0
    ax.fill_between(x, 0, H, where=under, color="#fee2e2", alpha=0.6, step="mid")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.text(len(x) * 0.5, -6, "잔량 0 = 로봇 idle (ZOH 홀드 · 멈칫)",
            fontsize=8.5, color=C_POL, ha="center")
    ax.set_xlim(0, len(x))
    ax.set_ylim(-9, H + 4)
    ax.set_xlabel("제어 tick", fontsize=10)
    ax.set_ylabel("큐 잔량 (액션 수)", fontsize=10)
    ax.set_title("(b) 큐 잔량 궤적 — 서버 지연이 청크 길이에 육박하면 큐가 고갈된다 (WE-1)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = HERE + "/fig2_async_timing.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# 그림 3: 탑재 정책 비교표 (ACT/DP/π0/SmolVLA/GR00T)
# =====================================================================
def fig3_policy_zoo():
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.axis("off")
    ax.set_title("LeRobot에 탑재된 정책들 — 같은 PreTrainedPolicy API, 다른 액션 헤드",
                 fontsize=12.5, fontweight="bold", pad=14)

    cols = ["정책", "액션 헤드", "표현", "액션 차원", "청크 H", "실효 주기", "LeRobot 상태"]
    rows = [
        ["ACT",      "CVAE 디코더",       "연속(회귀)",   "14 (양팔)", "100", "~50Hz",  "코어 IL"],
        ["Diffusion","DDPM/DDIM U-Net",  "연속(디퓨전)", "태스크별",  "16→8", "~10Hz",  "코어 IL"],
        ["VQ-BeT",   "VQ + 트랜스포머",   "이산 코드+오프셋","태스크별","~5",  "~10Hz",  "코어 IL"],
        ["π0",       "flow expert(~300M)","연속(flow)",  "≤18(패딩)", "50",  "≤50Hz",  "파운데이션"],
        ["SmolVLA",  "flow expert(~100M)","연속(flow)",  "6~7",       "50",  "~30Hz",  "파운데이션 ★"],
        ["GR00T*",   "DiT flow 헤드",     "연속(flow)",  "~132",      "16→40","~120Hz","외부 연계"],
    ]
    colcolors = [C_POL, "#111827", "#111827", "#111827", "#111827", "#111827", C_GRAY]
    ncol = len(cols)
    x0, x1 = 0.5, 10.5
    xs = np.linspace(x0, x1, ncol + 1)
    widths = np.diff(xs)
    ytop = 4.4
    rowh = 0.62

    # 헤더
    for j, c in enumerate(cols):
        ax.add_patch(Rectangle((xs[j], ytop), widths[j], rowh,
                     fc="#1f2937", ec="white"))
        ax.text(xs[j] + widths[j] / 2, ytop + rowh / 2, c, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    # 행
    for i, row in enumerate(rows):
        y = ytop - (i + 1) * rowh
        band = "#f9fafb" if i % 2 == 0 else "white"
        if row[0].startswith("SmolVLA"):
            band = "#fffbeb"
        for j, cell in enumerate(row):
            ax.add_patch(Rectangle((xs[j], y), widths[j], rowh, fc=band,
                         ec="#e5e7eb", lw=0.6))
            col = colcolors[j]
            fw = "bold" if j == 0 else "normal"
            ax.text(xs[j] + widths[j] / 2, y + rowh / 2, cell, ha="center",
                    va="center", fontsize=8.3, color=col, fontweight=fw)
    ax.set_xlim(0, 11)
    ax.set_ylim(y - 0.9, ytop + rowh + 0.3)
    ax.text(0.5, y - 0.55,
            "* GR00T는 NVIDIA Isaac-GR00T 스택이 정본 — LeRobot 데이터/포맷과 연계. "
            "청크 H는 예측 길이, 실효 주기는 논문/문서 보고값 (50강 표와 정합).",
            fontsize=7.8, color=C_GRAY, ha="left")
    fig.tight_layout()
    out = HERE + "/fig3_policy_zoo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# =====================================================================
# 그림 4: 정규화·청크·앙상블 파이프라인 (WE-2 시각화)
# =====================================================================
def fig4_pipeline(we2):
    fig = plt.figure(figsize=(11.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.28)

    lo, hi = we2["lo"], we2["hi"]
    raw, tgt_n = we2["raw"], we2["tgt_n"]
    pred_raw, pred_raw_bad = we2["pred_raw"], we2["pred_raw_bad"]
    D, H = we2["D"], we2["H"]

    # --- (a) 정규화: 물리단위 → [-1,1] ---
    ax = fig.add_subplot(gs[0, 0])
    xk = np.arange(H)
    for d in range(D):
        ax.plot(xk, raw[:, d], lw=1.3, alpha=0.85)
    ax.set_title("(a) 원시 액션 (물리 단위 rad)\n차원마다 스케일이 다르다", fontsize=9.5)
    ax.set_xlabel("청크 스텝", fontsize=8.5)
    ax.set_ylabel("액션 값", fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    for d in range(D):
        ax.plot(xk, tgt_n[:, d], lw=1.3, alpha=0.85)
    ax.axhline(1, color=C_GRAY, ls="--", lw=0.8)
    ax.axhline(-1, color=C_GRAY, ls="--", lw=0.8)
    ax.set_title("(b) 정규화 후 [-1,1] (q01/q99, meta/stats.json)\n"
                 "정책은 이 공간에서 학습·추론", fontsize=9.5)
    ax.set_xlabel("청크 스텝", fontsize=8.5)
    ax.set_ylim(-1.4, 1.4)
    ax.grid(alpha=0.25)

    # --- (c) 역정규화: 같은 stats vs 잘못된 stats ---
    ax = fig.add_subplot(gs[1, 0])
    d0 = 4     # 스케일 큰 차원
    ax.plot(xk, raw[:, d0], color="#111827", lw=2.4, label="참값 (원시)")
    ax.plot(xk, pred_raw[:, d0], color=C_ENV, lw=1.6, ls="--",
            label="같은 stats 역정규화 (max|err|=%.3f)" % we2["err_raw"])
    ax.plot(xk, pred_raw_bad[:, d0], color=C_POL, lw=1.6, ls=":",
            label="틀린 stats(0.5×) (max|err|=%.2f)" % we2["err_raw_bad"])
    ax.set_title("(c) 역정규화 — stats 정합이 인터페이스 계약\n"
                 "틀린 stats → %.0f배 붕괴" % (we2["err_raw_bad"] / we2["err_raw"]),
                 fontsize=9.5)
    ax.set_xlabel("청크 스텝 (차원 %d)" % d0, fontsize=8.5)
    ax.set_ylabel("액션 값 (rad)", fontsize=8.5)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.grid(alpha=0.25)

    # --- (d) temporal ensembling: 지터 감소 ---
    ax = fig.add_subplot(gs[1, 1])
    ts = we2["ts"]
    ax.plot(ts, we2["true"], color=C_GRAY, lw=2.2, alpha=0.7, label="참 궤적")
    ax.plot(ts, we2["naive"], color=C_POL, lw=1.1,
            label="naive 최신청크 (지터 %.0fe-3)" % (we2["j_naive"] * 1e3))
    ax.plot(ts, we2["ens_v"], color=C_DATA, lw=1.6,
            label="앙상블 (지터 %.0fe-3, %.1f배↓)"
            % (we2["j_ens"] * 1e3, we2["j_naive"] / we2["j_ens"]))
    ax.set_title("(d) temporal ensembling (50강 E1 회수)\n"
                 "겹치는 청크의 지수가중 평균 → 지터 감소", fontsize=9.5)
    ax.set_xlabel("시간 (s)", fontsize=8.5)
    ax.set_ylabel("관절0 (rad)", fontsize=8.5)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("정규화 → 정책 → 역정규화 → 청크 → temporal ensembling (WE-2)",
                 fontsize=12.5, fontweight="bold", y=0.98)
    out = HERE + "/fig4_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    we1 = we1_async_queue()
    we2 = we2_pipeline()
    print("=" * 62)
    fig1_architecture()
    fig2_async_timing(we1)
    fig3_policy_zoo()
    fig4_pipeline(we2)
    print("=" * 62)
    print("done.")
