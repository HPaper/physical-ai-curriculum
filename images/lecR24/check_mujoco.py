# Lec R24 실습 검증 스크립트 — MuJoCo humanoid 간이 WBC (선 자세 유지 + 토크 분배)
# 준비: 이 디렉토리에 humanoid.xml 다운로드
#   curl -sLO https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml
# 실행: python3 check_mujoco.py  (이 디렉토리에서, ~1분)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
C1, C2, C3, C4 = '#0072B2', '#D55E00', '#009E73', '#CC79A7'

m = mujoco.MjModel.from_xml_path('humanoid.xml')
d = mujoco.MjData(m)
dt = m.opt.timestep

# ------------------------------------------------------------
# Part 0. 부족구동 확인
# ------------------------------------------------------------
print('=== Part 0: 모델 ===')
print(f'nq={m.nq} nv={m.nv} nu={m.nu}  → 구동 안 되는 DoF = nv-nu = {m.nv - m.nu} (floating base)')
mg = sum(m.body_mass) * 9.81
print(f'총질량 {sum(m.body_mass):.2f} kg, mg = {mg:.1f} N')

act_dofadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
gear = m.actuator_gear[:, 0]
taulim = np.zeros(m.nv)
taulim[act_dofadr] = gear          # 관절별 토크 한계 = 모터 사양(gear)
jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
dof = lambda nm: m.jnt_dofadr[jid[nm]]
ankle_y = [dof('ankle_y_right'), dof('ankle_y_left')]   # pitch 균형용
ankle_x = [dof('ankle_x_right'), dof('ankle_x_left')]   # roll 균형용
torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')

# ------------------------------------------------------------
# Part 1. 접촉이 내야 하는 베이스 렌치 (mj_inverse)
# ------------------------------------------------------------
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
d.qvel[:] = 0
d.qacc[:] = 0
mujoco.mj_inverse(m, d)
print('=== Part 1: 정지에 필요한 베이스 6 성분 (qfrc_inverse[:6]) ===')
print(np.round(d.qfrc_inverse[:6], 2), ' ← 모터가 못 내는 힘. 접촉(지면반력)만이 낼 수 있다')

# ------------------------------------------------------------
# 접촉 정합 토크 분배 (본문 E3의 QP를 등식 KKT로 축소한 것)
#   min f^T W f  s.t. (Jc^T f)[:6] = qfrc_bias[:6]
# ------------------------------------------------------------
def grf_dist(d, w_t=10.0):
    nc = d.ncon
    if nc == 0:
        return np.zeros(m.nv), np.zeros((0, 3))
    Jc = np.zeros((3 * nc, m.nv))
    for i in range(nc):
        c = d.contact[i]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, jacr, c.pos, m.geom_bodyid[max(c.geom1, c.geom2)])
        Jc[3 * i:3 * i + 3] = jacp
    A = Jc[:, :6].T                       # (6, 3nc): 베이스 행
    b = d.qfrc_bias[:6]
    n = 3 * nc
    W = np.diag(np.tile([w_t, w_t, 1.0], nc))   # 접선력에 벌점 → 원뿔 위반 억제
    KKT = np.block([[W, A.T], [A, np.zeros((6, 6))]])
    sol = np.linalg.lstsq(KKT, np.concatenate([np.zeros(n), b]), rcond=None)[0]
    f = sol[:n]
    return Jc.T @ f, f.reshape(-1, 3)

def foot_fz(d):
    fz = {'l': 0.0, 'r': 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        fn = c.frame[:3] * f6[0] + c.frame[3:6] * f6[1] + c.frame[6:9] * f6[2]
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[max(c.geom1, c.geom2)]))
        fz['l' if 'left' in nm else 'r'] += fn[2]
    return fz['l'], fz['r']

# ------------------------------------------------------------
# Part 2. 정착(settle): WBC 피드포워드 + 관절 감쇠만으로 2초
# ------------------------------------------------------------
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
for k in range(int(2.0 / dt)):
    tau = np.zeros(m.nv)
    tau[6:] = d.qfrc_bias[6:] - grf_dist(d)[0][6:] - 5.0 * d.qvel[6:]
    tau[6:] = np.clip(tau[6:], -taulim[6:], taulim[6:])
    d.qfrc_applied[:] = 0
    d.qfrc_applied[6:] = tau[6:]
    mujoco.mj_step(m, d)
qs, vs = d.qpos.copy(), d.qvel.copy()
com0 = d.subtree_com[0].copy()
q_ref = qs[7:].copy()
_, f_settle = grf_dist(d)
print('=== Part 2: 정착 상태 ===')
print(f'com = {np.round(com0, 4)}, ncon = {d.ncon}, |qvel| = {np.linalg.norm(vs):.4f}')
print(f'분배된 수직력 합 = {f_settle[:, 2].sum():.1f} N (mg = {mg:.1f} N)')

# ------------------------------------------------------------
# Part 3. 세 제어기 비교 + 밀기
# ------------------------------------------------------------
def run(mode, push=0.0, pdir=0, t0=2.0, dur=0.3, T=8.0, kc=400.0, dc=300.0, log=False):
    d.qpos[:] = qs; d.qvel[:] = vs; d.time = 0
    mujoco.mj_forward(m, d)
    maxdev, rows = 0.0, []
    for k in range(int(T / dt)):
        t = k * dt
        d.xfrc_applied[:] = 0
        if t0 <= t < t0 + dur:
            d.xfrc_applied[torso, pdir] = push
        mujoco.mj_subtreeVel(m, d)
        tau = np.zeros(m.nv)
        if mode == 'naive':      # 접촉을 모르는 중력보상
            tau[6:] = d.qfrc_bias[6:] - 5.0 * d.qvel[6:]
        elif mode == 'wbc':      # 접촉 정합 분배 + 자세 PD + CoM 발목 태스크
            tau[6:] = d.qfrc_bias[6:] - grf_dist(d)[0][6:] - 5.0 * d.qvel[6:]
            tau[6:] += 20.0 * (q_ref - d.qpos[7:])
            com, comv = d.subtree_com[0], d.subtree_linvel[0]
            u = kc * (com[0] - com0[0]) + dc * comv[0]      # pitch: +발목토크 → CoM 뒤로
            for a in ankle_y:
                tau[a] += 0.5 * u
            v = kc * (com[1] - com0[1]) + dc * comv[1]      # roll (부호는 스텝응답으로 결정)
            for a in ankle_x:
                tau[a] += +0.5 * v
        tau[6:] = np.clip(tau[6:], -taulim[6:], taulim[6:])
        d.qfrc_applied[:] = 0
        d.qfrc_applied[6:] = tau[6:]
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return 'NAN', t, maxdev, np.array(rows)
        maxdev = max(maxdev, abs(d.subtree_com[0][0] - com0[0]))
        if log and k % 4 == 0:
            rows.append((t, d.qpos[2], d.subtree_com[0][0], *foot_fz(d)))
        if d.qpos[2] < 0.9:
            return 'falls', t, maxdev, np.array(rows)
    return 'STANDS', T, maxdev, np.array(rows)

print('=== Part 3: 세 제어기 (외란 없음, 8 s) ===')
logs = {}
for mode in ['zero', 'naive', 'wbc']:
    st, t, md, rows = run(mode, log=True)
    logs[mode] = rows
    print(f'{mode:6s}: {st} t={t:.2f} s  max|Δcom_x|={md * 1000:.1f} mm')

print('=== Part 4a: 전방 밀기(+x) 스윕, 0.3 s ===')
for push in [10, 20, 30, 40]:
    st, t, md, _ = run('wbc', push=push, pdir=0)
    print(f'F=+{push:2d} N x: {st:6s} (t={t:.2f})  max|Δcom_x|={md * 1000:5.1f} mm')

print('=== Part 4a-보충: 밀기 한계의 기하 (capture point 환산) ===')
d.qpos[:] = qs; d.qvel[:] = vs        # 정착 자세로 복원 후 기하 계산
mujoco.mj_forward(m, d)
xmax = -1e9        # 발끝 최대 전방 x (capsule 끝점 + 반지름)
for gi in range(m.ngeom):
    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi])
    if bn and 'foot' in bn:
        pos = d.geom_xpos[gi]; R = d.geom_xmat[gi].reshape(3, 3)
        r, hl = m.geom_size[gi][0], m.geom_size[gi][1]
        for s in (-1, 1):
            xmax = max(xmax, pos[0] + s * hl * R[0, 2] + r)
h_com = com0[2]
omega = np.sqrt(9.81 / h_com)
print(f'발끝 전방 여유 = {(xmax - com0[0]) * 100:.1f} cm (기하 한계)')
for F in (30, 40):
    dv = F * 0.3 / sum(m.body_mass)
    print(f'F={F} N, 0.3 s 임펄스 → capture point 이동 {dv / omega * 100:.1f} cm')

print('=== Part 4b: 측방 밀기(+y) — 발 사이 하중 재분배 ===')
st, t, md, rows_lat = run('wbc', push=20, pdir=1, log=True)
i_pre = int(1.9 / dt / 4); i_push = int(2.25 / dt / 4)
print(f'push 전  fz L/R = {rows_lat[i_pre, 3]:.0f}/{rows_lat[i_pre, 4]:.0f} N')
print(f'push 끝  fz L/R = {rows_lat[i_push, 3]:.0f}/{rows_lat[i_push, 4]:.0f} N  → {st}')
for push in [40, 60]:
    st, t, md, _ = run('wbc', push=push, pdir=1)
    print(f'F=+{push:2d} N y: {st}')

# ------------------------------------------------------------
# Part 5. 발목 피치 스텝 응답 — 균형 피드백 부호 결정용 (본문 실습 6)
#   기본 제어(분배+감쇠+자세 PD, CoM 피드백 없음)에 두 발목 피치 +4 N·m을 0.5 s
# ------------------------------------------------------------
print('=== Part 5: 발목 피치 스텝 응답 (+4 N·m each, 0.5 s) ===')
d.qpos[:] = qs; d.qvel[:] = vs; d.time = 0
mujoco.mj_forward(m, d)
com_x0 = d.subtree_com[0][0]
dev = 0.0
for k in range(int(2.0 / dt)):
    t = k * dt
    tau = np.zeros(m.nv)
    tau[6:] = d.qfrc_bias[6:] - grf_dist(d)[0][6:] - 5.0 * d.qvel[6:]
    tau[6:] += 20.0 * (q_ref - d.qpos[7:])
    if 1.0 <= t < 1.5:
        for a in ankle_y:
            tau[a] += 4.0
    tau[6:] = np.clip(tau[6:], -taulim[6:], taulim[6:])
    d.qfrc_applied[:] = 0
    d.qfrc_applied[6:] = tau[6:]
    mujoco.mj_step(m, d)
    if 1.0 <= t < 1.5:
        dx = d.subtree_com[0][0] - com_x0
        if abs(dx) > abs(dev):
            dev = dx
print(f'스텝 구간 내 CoM x 최대 이동: {dev * 1000:.1f} mm  ← 부호가 피드백 부호를 결정')

# ------------------------------------------------------------
# 그림 5: 실습 요약
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for mode, col, lab in [('zero', '0.6', '토크 0'), ('naive', C2, '순진한 중력보상 (접촉 무시)'),
                       ('wbc', C1, '간이 WBC (접촉 정합 분배)')]:
    r = logs[mode]
    ax.plot(r[:, 0], r[:, 1], color=col, lw=2.2, label=lab)
ax.axhline(0.9, color='k', lw=0.8, ls=':')
ax.text(6.5, 0.93, '낙상 판정선', fontsize=8.5)
ax.set_xlabel('t [s]'); ax.set_ylabel('골반 높이 [m]')
ax.set_ylim(0, 1.45); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='center right')
ax.set_title('(a) 같은 중력보상, 접촉을 아느냐가 서고 넘어짐을 가른다')
ax = axes[1]
ax.plot(rows_lat[:, 0], rows_lat[:, 3], color=C1, lw=2, label='왼발 $f_z$')
ax.plot(rows_lat[:, 0], rows_lat[:, 4], color=C2, lw=2, label='오른발 $f_z$')
ax.axvspan(2.0, 2.3, color=C4, alpha=0.2)
ax.text(2.32, 360, '측방 밀기\n20 N', color=C4, fontsize=9)
ax.axhline(mg / 2, color='0.5', lw=1, ls='--')
ax.text(6.0, mg / 2 + 8, 'mg/2', fontsize=8.5, color='0.4')
ax.set_xlabel('t [s]'); ax.set_ylabel('수직 지면반력 [N]')
ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax.set_title('(b) 밀리는 동안 하중이 반대발로 이동한다 (WE-3의 실물판)')
fig.tight_layout()
fig.savefig('fig5_mujoco_stand.png', dpi=140, bbox_inches='tight')
print('fig5 저장')
