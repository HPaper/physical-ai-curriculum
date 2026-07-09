# Lec R20 실습 수치 검증: MuJoCo op-space 제어 (mj_jacSite + qfrc_bias)
# 실행: python3 check_mujoco.py  (mujoco 3.2.5 기준)
import mujoco
import numpy as np

# ---------- Part A: 2R op-space (R10 WE-3 모델 + site) ----------
XML2R = """
<mujoco><option gravity="0 -9.81 0" timestep="0.001"/><worldbody>
  <body><joint type="hinge" axis="0 0 1"/>
    <inertial pos="0.5 0 0" mass="1" diaginertia="0.001 0.0833333333 0.0833333333"/>
    <body pos="1 0 0"><joint type="hinge" axis="0 0 1"/>
      <inertial pos="0.5 0 0" mass="1" diaginertia="0.001 0.0833333333 0.0833333333"/>
      <site name="ee" pos="1 0 0"/>
    </body></body></worldbody></mujoco>"""

Kp_t, Kd_t = 100.0, 20.0

def make_circle_refs(cx, cy, Rc, T_lap):
    w = 2*np.pi/T_lap
    def refs(t):
        x_d  = np.array([cx + Rc*np.cos(w*t), cy + Rc*np.sin(w*t)])
        xd_d = np.array([-Rc*w*np.sin(w*t),  Rc*w*np.cos(w*t)])
        a_d  = np.array([-Rc*w*w*np.cos(w*t), -Rc*w*w*np.sin(w*t)])
        return x_d, xd_d, a_d
    return refs

def ik2r(x, y, l1=1.0, l2=1.0):
    c2 = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
    q2 = np.arccos(np.clip(c2, -1, 1))
    q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
    return np.array([q1, q2])

def run_2r(refs, T_end=4.0, damp=0.0, blow=1e6):
    """2R op-space 제어 실행 → (times, errs[mm], |τ|max 시계열, 폭발 시각).
    본문 실습 1(Part A)·2(특이점 체감)의 공용 러너."""
    m = mujoco.MjModel.from_xml_string(XML2R)
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
    nv = m.nv; dt = m.opt.timestep
    # 초기 상태: 기준 궤적 위에서 기준 속도로 시작 (해석 IK)
    x0, v0, _ = refs(0.0)
    d.qpos[:] = ik2r(*x0)
    mujoco.mj_forward(m, d)
    jacp = np.zeros((3, nv)); mujoco.mj_jacSite(m, d, jacp, None, sid)
    d.qvel[:] = np.linalg.solve(jacp[:2], v0)
    n = int(T_end/dt)
    times, errs, taus = np.zeros(n), np.zeros(n), np.zeros(n)
    t_blow = None
    for i in range(n):
        t = i*dt
        mujoco.mj_forward(m, d)
        x = d.site(sid).xpos[:2].copy()
        jacp = np.zeros((3, nv)); mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:2].copy()
        # J̇q̇: qpos를 q̇ 방향으로 미소 전진시켜 유한차분
        qpos0 = d.qpos.copy(); eps = 1e-6
        mujoco.mj_integratePos(m, d.qpos, d.qvel, eps)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        jacp2 = np.zeros((3, nv)); mujoco.mj_jacSite(m, d, jacp2, None, sid)
        d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
        Jd_qd = ((jacp2[:2] - J)/eps) @ d.qvel
        Mfull = np.zeros((nv, nv)); mujoco.mj_fullM(m, Mfull, d.qM)
        Minv = np.linalg.inv(Mfull)
        Lam = np.linalg.inv(J @ Minv @ J.T + damp*np.eye(2))   # damp>0: Λ_δ (WE-3)
        x_d, xd_d, a_d = refs(t)
        a_x = a_d + Kd_t*(xd_d - J @ d.qvel) + Kp_t*(x_d - x)
        F = Lam @ (a_x - Jd_qd) + Lam @ J @ Minv @ d.qfrc_bias
        tau = J.T @ F
        times[i], errs[i] = t, np.linalg.norm(x - x_d)*1000
        taus[i] = np.abs(tau).max()
        if not np.all(np.isfinite(tau)) or np.abs(tau).max() > blow:   # 수치 폭발
            t_blow = t
            times, errs, taus = times[:i+1], errs[:i+1], taus[:i+1]
            break
        d.qfrc_applied[:] = tau
        mujoco.mj_step(m, d)
    return times, errs, taus, t_blow

# 실습 1 (Part A): 기준 원 (R19·WE-2와 동일: 중심 (1.1,0.3), 반경 0.3, 주기 2 s)
times, errs, taus_base, _ = run_2r(make_circle_refs(1.1, 0.3, 0.3, 2.0))
lap2 = times >= 2.0
print(f"Part A (MuJoCo 2R op-space): 2바퀴째 EEF RMS = {np.sqrt(np.mean(errs[lap2]**2)):.4f} mm "
      f"(NumPy판 기대값 0.0755 mm), max|τ| = {taus_base.max():.1f} N·m")

# 실습 2: 특이점 체감 — 경계 스침(원 중심 1.65) vs 경계 넘기기(WE-3의 직선)
_, _, taus_graze, _ = run_2r(make_circle_refs(1.65, 0.3, 0.3, 2.0))
print(f"실습 2 (원 중심 (1.65,0.3), 최원점 r≈1.98 — 경계 스침): max|τ| = {taus_graze.max():.1f} N·m")

def line_refs(t):   # WE-3의 반경 방향 직선 — 목표가 t=2s에 경계 r=2를 넘는다
    return np.array([1.5 + 0.25*t, 0.0]), np.array([0.25, 0.0]), np.zeros(2)

_, _, taus_line, t_blow = run_2r(line_refs, T_end=3.0)
print(f"실습 2 (직선 목표, 정확한 Λ): 토크 폭발 t = {t_blow:.2f} s, max|τ| = {taus_line.max():.3g} N·m")
_, _, taus_dls, _ = run_2r(line_refs, T_end=3.0, damp=0.01)
print(f"실습 2 (직선 목표, damp=0.01): max|τ| = {taus_dls.max():.1f} N·m (폭발 없음)")

# ---------- Part B: 3R 여유자유도 + null-space 자세 태스크 ----------
XML3R = """
<mujoco><option gravity="0 -9.81 0" timestep="0.001"/><worldbody>
  <body><joint type="hinge" axis="0 0 1" name="q1"/>
    <geom type="capsule" fromto="0 0 0  0.6 0 0" size="0.04" density="1000"/>
    <body pos="0.6 0 0"><joint type="hinge" axis="0 0 1" name="q2"/>
      <geom type="capsule" fromto="0 0 0  0.5 0 0" size="0.04" density="1000"/>
      <body pos="0.5 0 0"><joint type="hinge" axis="0 0 1" name="q3"/>
        <geom type="capsule" fromto="0 0 0  0.4 0 0" size="0.04" density="1000"/>
        <site name="ee" pos="0.4 0 0"/>
      </body></body></body></worldbody></mujoco>"""

def run_3r(use_posture, T_end=5.0, full_bias_comp=False):
    m = mujoco.MjModel.from_xml_string(XML3R)
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
    nv = m.nv
    q_rest = np.array([0.6, 0.8, -0.5])
    d.qpos[:] = q_rest
    mujoco.mj_forward(m, d)
    x_goal = d.site(sid).xpos[:2].copy()          # 초기 EEF를 그대로 유지하는 태스크
    Kp_t, Kd_t = 100.0, 20.0
    Kp_n, Kd_n = 20.0, 5.0
    dt = m.opt.timestep; n = int(T_end/dt)
    e_hist, qdev_hist = np.zeros(n), np.zeros(n)
    for i in range(n):
        mujoco.mj_forward(m, d)
        x = d.site(sid).xpos[:2].copy()
        jacp = np.zeros((3, nv)); mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:2].copy()
        qpos0 = d.qpos.copy(); eps = 1e-6
        mujoco.mj_integratePos(m, d.qpos, d.qvel, eps)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        jacp2 = np.zeros((3, nv)); mujoco.mj_jacSite(m, d, jacp2, None, sid)
        d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
        Jd_qd = ((jacp2[:2] - J)/eps) @ d.qvel
        Mfull = np.zeros((nv, nv)); mujoco.mj_fullM(m, Mfull, d.qM)
        Minv = np.linalg.inv(Mfull)
        Lam = np.linalg.inv(J @ Minv @ J.T)
        xdot = J @ d.qvel
        a_x = Kd_t*(-xdot) + Kp_t*(x_goal - x)
        if full_bias_comp:      # 변형: bias를 관절공간에서 전부 보상 (실습 5)
            tau = d.qfrc_bias.copy() + J.T @ (Lam @ (a_x - Jd_qd))
        else:                   # Khatib 원형: 태스크 공간 사영분(p, μ)만 보상 (E1)
            F = Lam @ (a_x - Jd_qd) + Lam @ J @ Minv @ d.qfrc_bias
            tau = J.T @ F
        if use_posture:
            Jbar = Minv @ J.T @ Lam                    # 동역학적으로 일관된 일반화 역행렬
            N = np.eye(nv) - J.T @ Jbar.T              # 토크 수준 null-space 사영자
            tau0 = Kp_n*(q_rest - d.qpos) - Kd_n*d.qvel
            tau = tau + N @ tau0
        d.qfrc_applied[:] = tau
        mujoco.mj_step(m, d)
        e_hist[i] = np.linalg.norm(x - x_goal)*1000
        qdev_hist[i] = np.degrees(np.abs(d.qpos - q_rest).max())
    return e_hist, qdev_hist

for flag, fb, name in [(False, False, "자세 태스크 없음"),
                       (True, False, "null-space 자세 태스크"),
                       (True, True, "자세 태스크 + 관절공간 전체 bias 보상")]:
    e_h, qd_h = run_3r(flag, full_bias_comp=fb)
    print(f"Part B (3R, {name}): EEF 최대 이탈 = {e_h.max():.3f} mm, "
          f"관절 최대 이탈 = {qd_h.max():.1f}°, 최종 관절 이탈 = {qd_h[-1]:.1f}°")
