# Lec R12 그림 생성 스크립트
# 실행: python3 gen_figs.py  (numpy, matplotlib, mujoco 필요)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mujoco

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})

OUT = __file__.rsplit('/', 1)[0] if '/' in __file__ else '.'

# ---------------------------------------------------------------- fig 1
# (a) 마찰 원뿔의 기하  (b) 상보성 조건의 L자 집합과 soft contact 완화
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
mu = 0.5
alpha = np.arctan(mu)
# 접촉면
ax.fill_between([-1.4, 1.4], -0.5, 0, color='0.85', zorder=0)
ax.axhline(0, color='0.4', lw=1)
# 원뿔 (법선 z축 기준 ±alpha)
h = 1.25
ax.fill_betweenx([0, h], [0, -h*np.tan(alpha)], [0, h*np.tan(alpha)],
                 color='tab:blue', alpha=0.22, zorder=1)
for s in (+1, -1):
    ax.plot([0, s*h*np.tan(alpha)], [0, h], color='tab:blue', lw=2)
ax.annotate('', xy=(0, 1.15), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color='k', lw=1.6))
ax.text(0.03, 1.16, r'$f_n$ (법선)', fontsize=11)
# 원뿔 안 힘 (stick)과 밖 힘 (slip)
ax.annotate('', xy=(0.30, 0.85), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2.2))
ax.text(0.24, 0.90, '원뿔 안\n→ 정지(stick)', color='tab:green', fontsize=10)
ax.annotate('', xy=(0.85, 0.62), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=2.2))
ax.text(0.72, 0.42, '원뿔 밖 → 불가능\n(미끄러지며 경계로 포화)', color='tab:red', fontsize=10)
# 반각 표시
th = np.linspace(np.pi/2, np.pi/2 - alpha, 30)
ax.plot(0.55*np.cos(th), 0.55*np.sin(th), color='k', lw=1)
ax.text(0.13, 0.60, r'$\alpha=\tan^{-1}\mu$', fontsize=12)
ax.plot([0], [0], 'ko', ms=5)
ax.text(-1.3, -0.35, r'접촉력의 허용 집합: $\Vert f_t\Vert \leq \mu f_n$', fontsize=11)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.5, 1.45)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title(f'(a) 쿨롱 마찰 원뿔 (μ={mu})', fontsize=12)

ax = axes[1]
# 상보성: 허용 집합 = 두 반직선 (L자)
ax.plot([0, 2.2], [0, 0], color='tab:blue', lw=4,
        solid_capstyle='round', label='허용 집합 (강체 접촉)')
ax.plot([0, 0], [0, 2.2], color='tab:blue', lw=4, solid_capstyle='round')
# soft contact 완화 곡선
dd = np.linspace(-0.0, 2.2, 200)
soft = 1.2*np.exp(-dd/0.18)
ax.plot(dd, soft, '--', color='tab:orange', lw=2,
        label='soft contact 완화 (MuJoCo류)')
ax.plot([0], [0], 'o', color='tab:red', ms=9, zorder=5)
ax.annotate('미분 불가능한 모서리\n(불연속성의 근원)', xy=(0, 0), xytext=(0.55, 0.9),
            fontsize=11, color='tab:red',
            arrowprops=dict(arrowstyle='->', color='tab:red'))
ax.text(1.05, 0.10, r'$f_n=0$, 떨어져 있음 ($d>0$)', fontsize=10, va='bottom')
ax.text(0.06, 1.55, '$d=0$, 접촉 유지\n($f_n>0$)', fontsize=10)
ax.set_xlabel('간격(gap) $d$'); ax.set_ylabel('법선력 $f_n$')
ax.set_xlim(-0.15, 2.3); ax.set_ylim(-0.15, 2.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_title(r'(b) 상보성 조건  $0 \leq f_n \perp d \geq 0$', fontsize=12)

fig.tight_layout()
fig.savefig(f'{OUT}/fig1_cone_complementarity.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# force closure 성립/실패 파지 배치 (WE-2와 동일 기하)
def draw_grasp(ax, contacts, mu, title, box_hw=(1.0, 0.8)):
    w, hgt = box_hw
    ax.add_patch(plt.Rectangle((-w, -hgt), 2*w, 2*hgt, facecolor='0.9',
                               edgecolor='0.4'))
    alpha = np.arctan(mu)
    pts = [np.asarray(p, float) for p, n in contacts]
    for p, n in contacts:
        p = np.asarray(p, float); n = np.asarray(n, float)
        base = np.arctan2(n[1], n[0])
        L = 1.1
        e1 = p + L*np.array([np.cos(base+alpha), np.sin(base+alpha)])
        e2 = p + L*np.array([np.cos(base-alpha), np.sin(base-alpha)])
        ax.fill([p[0], e1[0], e2[0]], [p[1], e1[1], e2[1]],
                color='tab:blue', alpha=0.25)
        ax.plot(*zip(p, e1), color='tab:blue', lw=1.5)
        ax.plot(*zip(p, e2), color='tab:blue', lw=1.5)
        ax.annotate('', xy=p + 0.55*n, xytext=p,
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
        ax.plot(*p, 'ko', ms=6)
    ax.plot(*zip(*pts), '--', color='tab:red', lw=2, label='접촉점 연결선')
    ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.45, 1.45)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower center', fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
antipodal = [((-1, 0.0), (1, 0)), ((1, 0.0), (-1, 0))]
offset = [((-1, 0.4), (1, 0)), ((1, -0.4), (-1, 0))]
draw_grasp(axes[0], antipodal, 0.5,
           '(a) 마주보는 파지, μ=0.5 → force closure 성립\n(연결선이 두 원뿔 안)')
draw_grasp(axes[1], offset, 0.3,
           '(b) 어긋난 파지, μ=0.3 → 실패\n(연결선 기울기 0.4 > μ: 원뿔 밖)')
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(f'{OUT}/fig2_force_closure.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
# (a) 경사면 미끄럼: MuJoCo vs 이론   (b) 파지 μ 스윕: 성공/실패 경계
INCLINE_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" friction="0.5 0.005 0.0001"/>
    <body pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

def incline_disp(th_deg, T=1.0):
    m = mujoco.MjModel.from_xml_string(INCLINE_XML)
    th = np.radians(th_deg)
    m.opt.gravity[:] = 9.81*np.array([np.sin(th), 0, -np.cos(th)])
    d = mujoco.MjData(m)
    while d.time < T:
        mujoco.mj_step(m, d)
    return d.qpos[0]

GRASP_XML = """
<mujoco model="two_finger_grasp">
  <option timestep="0.001" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <worldbody>
    <body name="box" pos="0 0 0.2">
      <freejoint/>
      <geom name="box" type="box" size="0.02 0.02 0.02" mass="0.1"
            friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="finger_l" pos="-0.03 0 0.2">
      <joint name="slide_l" type="slide" axis="1 0 0"/>
      <geom name="pad_l" type="box" size="0.005 0.03 0.03" mass="0.05"
            friction="{mu} 0.005 0.0001"/>
    </body>
    <body name="finger_r" pos="0.03 0 0.2">
      <joint name="slide_r" type="slide" axis="-1 0 0"/>
      <geom name="pad_r" type="box" size="0.005 0.03 0.03" mass="0.05"
            friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide_l" gear="1"/>
    <motor joint="slide_r" gear="1"/>
  </actuator>
</mujoco>
"""

def grasp_drop(mu, squeeze=2.0, T=1.0):
    m = mujoco.MjModel.from_xml_string(GRASP_XML.format(mu=mu))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    z0 = d.qpos[2]
    d.ctrl[:] = squeeze
    while d.time < T:
        mujoco.mj_step(m, d)
    return z0 - d.qpos[2]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
mu = 0.5
thetas = np.arange(15, 41, 1)
sim = [incline_disp(t) for t in thetas]
th_f = np.linspace(15, 40, 300)
a = 9.81*(np.sin(np.radians(th_f)) - mu*np.cos(np.radians(th_f)))
ax.plot(th_f, np.maximum(a, 0)/2, '-', color='0.4',
        label=r'이론: $\frac{1}{2}g(\sin\theta-\mu\cos\theta)t^2$')
ax.plot(thetas, sim, 'o', ms=5, color='tab:blue', label='MuJoCo (1초 후)')
ax.axvline(np.degrees(np.arctan(mu)), ls='--', color='tab:red')
ax.text(np.degrees(np.arctan(mu))+0.4, 0.55,
        r'$\theta^*=\tan^{-1}0.5=26.6°$', color='tab:red', fontsize=10)
ax.set_xlabel(r'경사각 $\theta$ [deg]'); ax.set_ylabel('1초간 미끄러진 거리 [m]')
ax.set_title('(a) 경사면 블록: 미끄럼의 문턱 (μ=0.5)', fontsize=12)
ax.legend(fontsize=10)

ax = axes[1]
mus = np.array([0.10, 0.15, 0.20, 0.22, 0.23, 0.24, 0.245, 0.25,
                0.26, 0.28, 0.32, 0.40, 0.50])
drops = np.array([grasp_drop(m_) for m_ in mus])*1000
ax.semilogy(mus, np.maximum(drops, 0.3), 'o-', color='tab:blue')
ax.axvline(0.981/4, ls='--', color='tab:red')
ax.text(0.981/4 + 0.008, 300, r'$\mu^*=\frac{mg}{2F}=0.245$',
        color='tab:red', fontsize=11)
ax.axhspan(0.3, 5, color='tab:green', alpha=0.12)
ax.text(0.40, 1.1, '유지(성공)', color='tab:green', fontsize=11)
ax.text(0.12, 300, '낙하(실패)', color='tab:blue', fontsize=11)
ax.set_xlabel('마찰계수 μ'); ax.set_ylabel('1초 후 상자 낙하량 [mm] (log)')
ax.set_title('(b) 2지 파지 μ 스윕: 성공/실패 경계 (F=2N/손가락)', fontsize=12)

fig.tight_layout()
fig.savefig(f'{OUT}/fig3_mu_sweep.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# soft contact의 creep: 같은 파지(μ=0.5)가 솔버 설정에 따라 흘러내리는 시간 궤적
GRASP_TPL = GRASP_XML.replace('cone="elliptic" impratio="10"',
                              'cone="{cone}" impratio="{ir}"')

def creep_trace(cone, ir, T=1.0):
    m = mujoco.MjModel.from_xml_string(GRASP_TPL.format(cone=cone, ir=ir, mu=0.5))
    d = mujoco.MjData(m)
    z0 = d.qpos[2]
    d.ctrl[:] = 2.0
    ts, drops = [], []
    while d.time < T:
        mujoco.mj_step(m, d)
        ts.append(d.time)
        drops.append((z0 - d.qpos[2]) * 1000)
    return np.array(ts), np.array(drops), d.qvel[2] * 1000

fig, ax = plt.subplots(figsize=(7.5, 4.4))
settings = [('pyramidal', '1'), ('elliptic', '1'),
            ('elliptic', '10'), ('elliptic', '100')]
colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
for (cone, ir), c in zip(settings, colors):
    ts, drops, vz = creep_trace(cone, ir)
    ax.plot(ts, drops, color=c, lw=2,
            label=f'cone={cone}, impratio={ir}  ({-vz:.2f} mm/s)')
ax.set_xlabel('시간 [s]'); ax.set_ylabel('상자 낙하량 [mm]')
ax.set_title('같은 파지(μ=0.5, F=2N)의 creep — 물리는 그대로, 솔버 설정만 변경', fontsize=12)
ax.legend(fontsize=10, title='설정 (정상상태 creep 속도)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/fig4_creep.png', dpi=140)
plt.close(fig)

print('figures written to', OUT)
