# Lec R26 그림 생성 스크립트
# 실행: python3 gen_figs.py  (이 디렉토리에서)
# 모든 실험은 결정적이다(난수 없음). numpy 1.26 / mujoco 3.2.5 기준.
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
import mujoco

OUT = os.path.dirname(os.path.abspath(__file__))
C = dict(ee='tab:red', ie='tab:purple', se='tab:blue', rk='tab:green', mj='tab:orange')

# ============================================================
# WE-1: 스프링-질량, 세 적분기 (fig1, fig2a)
# ============================================================
m_, k_ = 1.0, 100.0
w = np.sqrt(k_/m_)                # 10 rad/s
h = 0.01                          # h·w = 0.1
n = int(10.0/h)                   # 10초 = 1000스텝

def energy(x, v): return 0.5*(v**2 + w**2*x**2)

def run_explicit():
    x, v = 1.0, 0.0; tr = [(x, v)]
    for _ in range(n):
        x, v = x + h*v, v - h*w**2*x
        tr.append((x, v))
    return np.array(tr)

def run_implicit():
    x, v = 1.0, 0.0; tr = [(x, v)]
    for _ in range(n):
        v = (v - h*w**2*x)/(1 + (h*w)**2); x = x + h*v
        tr.append((x, v))
    return np.array(tr)

def run_symplectic():
    x, v = 1.0, 0.0; tr = [(x, v)]
    for _ in range(n):
        v = v - h*w**2*x; x = x + h*v
        tr.append((x, v))
    return np.array(tr)

def run_rk4():
    f = lambda s: np.array([s[1], -w**2*s[0]])
    s = np.array([1.0, 0.0]); tr = [s]
    for _ in range(n):
        k1 = f(s); k2 = f(s+h/2*k1); k3 = f(s+h/2*k2); k4 = f(s+h*k3)
        s = s + h/6*(k1+2*k2+2*k3+k4)
        tr.append(s)
    return np.array(tr)

tre, tri, trs, trr = run_explicit(), run_implicit(), run_symplectic(), run_rk4()
Ee, Ei, Es, Er = (energy(t[:, 0], t[:, 1]) for t in (tre, tri, trs, trr))
t_ax = np.arange(n+1)*h
print("== WE-1 ==")
print(f"explicit  E(10s)/E0 = {Ee[-1]/Ee[0]:.6g}  (이론 (1+h²ω²)^1000 = {(1+(h*w)**2)**n:.6g})")
print(f"implicit  E(10s)/E0 = {Ei[-1]/Ei[0]:.6g}  (이론 (1+h²ω²)^-1000 = {(1+(h*w)**2)**-n:.6g})")
print(f"symplectic max|E/E0-1| = {np.abs(Es/Es[0]-1).max():.4g}")
print(f"rk4       1-E(10s)/E0 = {1-Er[-1]/Er[0]:.4g}")

# 증폭 인자 (fig2a)
def amp(hw, which):
    if which == 'ee':
        A = np.array([[1, hw], [-hw, 1]])
    elif which == 'ie':
        A = np.array([[1, hw], [-hw, 1]])/(1 + hw**2)
    elif which == 'se':
        A = np.array([[1-hw**2, hw], [-hw, 1]])
    else:
        z = 1j*hw
        return abs(1 + z + z**2/2 + z**3/6 + z**4/24)
    return np.abs(np.linalg.eigvals(A)).max()

hws = np.linspace(1e-3, 3.4, 2000)
amps = {k: np.array([amp(x, k) for x in hws]) for k in ('ee', 'ie', 'se', 'rk')}
for name, key in [('explicit', 'ee'), ('implicit', 'ie'), ('symplectic', 'se'), ('rk4', 'rk')]:
    over = hws[amps[key] > 1 + 1e-9]
    print(f"{name:10s} 첫 불안정 hω ≈ {over[0]:.4f}" if len(over) else f"{name:10s} 범위 내 안정")

# ============================================================
# WE-3: MuJoCo timestep 발산 경계 (fig2b, fig3a)
# ============================================================
XML_SPRING = """
<mujoco>
  <option gravity="0 0 0" integrator="Euler"/>
  <worldbody><body>
    <joint name="z" type="slide" axis="0 0 1" stiffness="{k}" damping="0"/>
    <geom type="sphere" size="0.05" mass="1"/>
  </body></worldbody>
</mujoco>
"""

def run_spring(k, h, nstep=3000, q0=0.01, record=False):
    m = mujoco.MjModel.from_xml_string(XML_SPRING.format(k=k))
    m.opt.timestep = h
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    qs = []
    for _ in range(nstep):
        mujoco.mj_step(m, d)
        q = d.qpos[0]
        if record: qs.append(q)
        if not np.isfinite(q) or abs(q) > 1e3*q0:
            return (False, np.array(qs)) if record else False
    return (True, np.array(qs)) if record else True

def boundary(k, iters=40):
    ht = 2/np.sqrt(k)
    lo, hi = 0.5*ht, 1.5*ht
    assert run_spring(k, lo) and not run_spring(k, hi)
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if run_spring(k, mid) else (lo, mid)
    return 0.5*(lo+hi)

print("== WE-3: 발산 경계 ==")
ks = np.array([1e2, 1e4, 1e6])
hstars = np.array([boundary(k) for k in ks])
for k, hs in zip(ks, hstars):
    print(f"k={k:8.0f}  h*(측정)={hs:.6g}  h*(이론 2/ω)={2/np.sqrt(k):.6g}  비율={hs*np.sqrt(k)/2:.4f}")

# 경계 바로 안/밖 시계열 (k=1e4)
_, q_in  = run_spring(1e4, 0.99*0.02, nstep=150, record=True)
ok_out, q_out = run_spring(1e4, 1.01*0.02, nstep=150, record=True)
print(f"h=0.99·h*: max|q|/q0 = {np.abs(q_in).max()/0.01:.3g} (유계) | "
      f"h=1.01·h*: {'발산' if not ok_out else '?'} ({len(q_out)}스텝 만에 |q|>10³·q0 돌파)")

# ============================================================
# WE-2: solref/solimp — 침투·반발 (fig3b, fig4)
# ============================================================
XML_BALL = """
<mujoco>
  <option gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" solref="{tc} {dr}" solimp="{d0} {d1} {w}"/>
    <body><joint type="free"/>
      <geom name="ball" type="sphere" size="0.05" mass="0.1" solref="{tc} {dr}" solimp="{d0} {d1} {w}"/>
    </body>
  </worldbody>
</mujoco>
"""
R = 0.05

def make_ball(tc, dr, d1=0.95, wd=1e-6, z0=None, h=1e-4):
    m = mujoco.MjModel.from_xml_string(
        XML_BALL.format(tc=tc, dr=dr, d0=min(0.9, d1), d1=d1, w=wd))
    m.opt.timestep = h
    d = mujoco.MjData(m)
    d.qpos[2] = R if z0 is None else z0
    return m, d

def pen_rest(tc, dr=1.0, d1=0.95, h=1e-4, T=1.0):
    m, d = make_ball(tc, dr, d1=d1, h=h)
    for _ in range(int(T/h)):
        mujoco.mj_step(m, d)
    return R - d.qpos[2]

print("== WE-2: 침투 vs timeconst (ζ=1, dmax=0.95) ==")
tcs = np.logspace(np.log10(0.005), np.log10(0.08), 7)
pens95 = np.array([pen_rest(tc) for tc in tcs])
slope = np.polyfit(np.log(tcs), np.log(pens95), 1)[0]
print(f"  log-log 기울기 = {slope:.4f} (이론 2)")
print(f"  tc=0.02 침투 = {pen_rest(0.02)*1000:.4f} mm (폐형식 (1-0.95)·9.81·0.02² = {0.05*9.81*4e-4*1000:.4f} mm)")
pens50 = np.array([pen_rest(tc, d1=0.5) for tc in tcs])
pens99 = np.array([pen_rest(tc, d1=0.99) for tc in tcs])
for d1 in [0.5, 0.9, 0.95, 0.99, 0.999]:
    print(f"  dmax={d1:5.3f}: 침투(tc=0.02) = {pen_rest(0.02, d1=d1)*1000:.4f} mm")

def restitution(dr, tc=0.02, h0=0.5, h=1e-4):
    m, d = make_ball(tc, dr, z0=R+h0, h=h)
    apex, hit, up = 0.0, False, False
    for _ in range(int(3.0/h)):
        mujoco.mj_step(m, d)
        z = d.qpos[2]
        if z < R: hit = True
        if hit and z > R:
            up = True; apex = max(apex, z)
        if up and d.qvel[2] < 0 and z < apex - 1e-4:
            break
    h1 = max(apex - R, 0.0)
    return np.sqrt(h1/h0)

print("== WE-2: 반발 vs dampratio (tc=0.02) ==")
drs = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5])
es = np.array([restitution(dr) for dr in drs])
for dr, e in zip(drs, es):
    print(f"  ζ={dr:4.2f}  e={e:.3f}")

print("== WE-3b: tc=0.01 고정, h 스윕 — 조용한 클램프 ==")
hs_sweep = np.array([1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7.5e-3, 9e-3, 1e-2])
pen_h = np.array([pen_rest(0.01, h=hh, T=2.0) for hh in hs_sweep])
clamp_model = 0.05*9.81*np.maximum(0.01, 2*hs_sweep)**2
for hh, p, c in zip(hs_sweep, pen_h, clamp_model):
    print(f"  h={hh:7.4f} (tc/h={0.01/hh:5.1f}): 침투={p*1000:.4f} mm | 클램프 모델={c*1000:.4f} mm")

# ============================================================
# fig1 — 한 장 요약: 세 적분기의 위상 궤적과 에너지 드리프트
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(tre[:, 0], tre[:, 1]/w, C['ee'], lw=0.7, label='explicit Euler')
ax[0].plot(tri[:, 0], tri[:, 1]/w, C['ie'], lw=0.9, label='implicit Euler')
ax[0].plot(trs[:, 0], trs[:, 1]/w, C['se'], lw=0.9, label='semi-implicit(symplectic) Euler')
ax[0].plot(trr[:, 0], trr[:, 1]/w, C['rk'], lw=0.9, ls='--', label='RK4')
ax[0].set_xlim(-14, 14); ax[0].set_ylim(-14, 14)
ax[0].set_xlabel('x'); ax[0].set_ylabel('v/ω'); ax[0].set_aspect('equal')
ax[0].set_title('(a) 위상 궤적 — 같은 물리, 같은 h, 다른 미래\n(스프링-질량, hω=0.1, 10초)')
ax[0].legend(fontsize=8, loc='upper right'); ax[0].grid(alpha=0.3)
ax[1].semilogy(t_ax, Ee/Ee[0], C['ee'], label='explicit: ×(1+h²ω²)/스텝')
ax[1].semilogy(t_ax, Ei/Ei[0], C['ie'], label='implicit: ÷(1+h²ω²)/스텝 (인공 감쇠)')
ax[1].semilogy(t_ax, Es/Es[0], C['se'], label='symplectic: 유계 진동(±5.3%)')
ax[1].semilogy(t_ax, Er/Er[0], C['rk'], ls='--', label='RK4: −1.4e-5/10초 (서서히 감쇠)')
ax[1].annotate('10초 만에 ×20,959', xy=(10, 2.1e4), xytext=(4.3, 2e3),
               color=C['ee'], fontsize=10, arrowprops=dict(arrowstyle='->', color=C['ee']))
ax[1].annotate('정확히 그 역수: ÷20,959', xy=(9.4, 1.1e-4), xytext=(2.6, 3e-3),
               color=C['ie'], fontsize=10, arrowprops=dict(arrowstyle='->', color=C['ie']))
ax[1].set_xlabel('시간 [s]'); ax[1].set_ylabel('E(t)/E(0)')
ax[1].set_title('(b) 에너지 드리프트 — 발산은 접촉이 아니라 적분기에서 시작된다')
ax[1].legend(fontsize=8, loc='upper left'); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_integrator_energy.png', dpi=140)
plt.close(fig)

# ============================================================
# fig2 — 안정성 지도: 증폭 인자와 h*–강성 지도
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(hws, amps['ee'], C['ee'], label='explicit Euler')
ax[0].plot(hws, amps['ie'], C['ie'], label='implicit Euler (항상 <1)')
ax[0].plot(hws, amps['se'], C['se'], label='symplectic Euler')
ax[0].plot(hws, amps['rk'], C['rk'], ls='--', label='RK4')
ax[0].axhline(1, color='k', lw=0.8)
ax[0].axvline(2, color=C['se'], ls=':', lw=1)
ax[0].axvline(2*np.sqrt(2), color=C['rk'], ls=':', lw=1)
ax[0].text(2.02, 1.25, 'hω=2', color=C['se'], fontsize=9)
ax[0].text(2.86, 1.25, 'hω=2√2', color=C['rk'], fontsize=9)
ax[0].text(0.25, 1.06, '항상 >1\n(진동계에서 무조건 발산)', color=C['ee'], fontsize=8)
ax[0].set_xlabel('hω (타임스텝 × 고유진동수)'); ax[0].set_ylabel('스텝당 증폭 인자')
ax[0].set_ylim(0.55, 1.6); ax[0].set_title('(a) 증폭 인자 — 1을 넘으면 지수 발산')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
kk = np.logspace(1.5, 6.5, 100)
ax[1].loglog(kk, 2/np.sqrt(kk), C['se'], label='이론 h*=2/ω=2√(m/k) (symplectic)')
ax[1].loglog(kk, 2*np.sqrt(2)/np.sqrt(kk), C['rk'], ls='--', label='이론 h*=2√2/ω (RK4)')
ax[1].loglog(ks, hstars, 'o', color=C['mj'], ms=9, mec='k',
             label='MuJoCo Euler 측정 (이분법)')
ax[1].fill_between(kk, 2/np.sqrt(kk), 1, alpha=0.12, color=C['ee'])
ax[1].text(3e3, 0.09, '발산', color=C['ee'], fontsize=11)
ax[1].text(70, 2.2e-3, '안정', color=C['se'], fontsize=11)
ax[1].text(1.1e4, 6e-4, '강성 ×100 → h ×1/10\n→ 시뮬 10배 느려짐', fontsize=8, rotation=-24)
ax[1].set_xlabel('강성 k [N/m] (m=1 kg)'); ax[1].set_ylabel('안정 최대 타임스텝 h* [s]')
ax[1].set_ylim(1e-4, 1); ax[1].set_title('(b) 타임스텝–강성 지도: 시뮬 속도의 근본 한계')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_stability_map.png', dpi=140)
plt.close(fig)

# ============================================================
# fig3 — MuJoCo: 경계 시계열 + 조용한 클램프
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
t_in = np.arange(len(q_in))*0.99*0.02
t_out = np.arange(len(q_out))*1.01*0.02
ax[0].semilogy(t_in, np.abs(q_in)/0.01, C['se'], lw=0.9, marker='.', ms=3,
               label='h=0.99·h* : 유계 (max 7.1배)')
ax[0].semilogy(t_out, np.abs(q_out)/0.01, C['ee'], lw=1.4, marker='.', ms=4,
               label='h=1.01·h* : 20스텝 만에 발산')
ax[0].axhline(1e3, color='k', ls=':', lw=0.8)
ax[0].text(1.0, 1.5e3, '발산 판정선 (10³·q0)', fontsize=8)
ax[0].set_xlim(0, 3.05)
ax[0].set_xlabel('시간 [s]'); ax[0].set_ylabel('|q|/q0')
ax[0].set_title('(a) 경계의 날카로움 — h* 양쪽 1%의 운명 (MuJoCo, k=10⁴)')
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which='both')
ax[1].plot(hs_sweep*1e3, pen_h*1e3, 'o-', color=C['mj'], label='측정 침투 (정지 공, tc=0.01)')
ax[1].plot(hs_sweep*1e3, clamp_model*1e3, 'k--', lw=1,
           label='모델: (1−d_max)·g·max(tc, 2h)²')
ax[1].axvline(5, color='gray', ls=':', lw=1)
ax[1].text(5.1, 0.06, 'tc = 2h\n(문서 권고 경계)', fontsize=8, color='gray')
ax[1].set_xlabel('타임스텝 h [ms]'); ax[1].set_ylabel('정지 침투 깊이 [mm]')
ax[1].set_title('(b) 발산 대신 조용한 변질 — h가 크면 접촉이 몰래 물렁해진다')
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_mujoco_timestep.png', dpi=140)
plt.close(fig)

# ============================================================
# fig4 — solref/solimp: 침투·반발 손잡이
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for pens, d1, c in [(pens50, 0.5, 'tab:gray'), (pens95, 0.95, C['mj']), (pens99, 0.99, 'tab:purple')]:
    ax[0].loglog(tcs*1e3, pens*1e3, 'o-', color=c, label=f'd_max={d1} 측정')
    ax[0].loglog(tcs*1e3, (1-d1)*9.81*tcs**2*1e3, 'k--', lw=0.7)
ax[0].text(0.6, 0.12, '점선: 폐형식\n(1−d_max)·g·tc²·ζ²', fontsize=8,
           transform=ax[0].transAxes)
ax[0].set_xlabel('solref timeconst [ms]'); ax[0].set_ylabel('정지 침투 깊이 [mm]')
ax[0].set_title(f'(a) 침투 = solref·solimp의 함수 (log-log 기울기 {slope:.4f})')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')
ax[1].plot(drs, es, 'o-', color=C['se'])
ax[1].axvline(1.0, color='gray', ls=':', lw=1)
ax[1].text(1.02, 0.5, 'ζ=1 임계감쇠\n(반발 없음)', fontsize=8, color='gray')
ax[1].annotate('ζ=0.3 → e=0.44\n"통통 튀는 세계"', xy=(0.3, 0.44), xytext=(0.5, 0.6),
               fontsize=9, arrowprops=dict(arrowstyle='->'))
ax[1].set_xlabel('solref dampratio ζ'); ax[1].set_ylabel('등가 반발계수 e = √(h1/h0)')
ax[1].set_title('(b) 반발 = dampratio의 함수 (0.5 m 낙하)')
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_solref_solimp.png', dpi=140)
plt.close(fig)

print("figs saved:", sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
