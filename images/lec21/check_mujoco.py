# Lec R21 실습 검증 스크립트 — MuJoCo 1-DoF 벽 접촉
# 실행: python3 check_mujoco.py  (mujoco 3.x 필요)
# gen_figs.py 의 페널티 시뮬(그림 1·2)과 같은 실험을 MuJoCo 접촉 솔버로 재현한다.
#
# 벽 강성 지정 노트 (R26 예고):
#  - solref="-2500 -1" 의 음수는 (강성, 감쇠)를 직접 지정하는 문법.
#  - 단 유효 강성은 solimp 의 d 로 스케일된다: k_eff = 2500/d_max² (d=0.5 → 4배 = 10^4 N/m).
#  - 구 반경(0.05)보다 침투가 깊어지면 접촉 모델이 왜곡되므로 반경을 넉넉히 잡았다.
import mujoco
import numpy as np

XML = """
<mujoco model="wall1d">
  <option timestep="1e-4" gravity="0 0 0"/>
  <worldbody>
    <body name="mass">
      <joint name="x" type="slide" axis="1 0 0"/>
      <geom name="tip" type="sphere" size="0.05" mass="2"
            solref="-2500 -1" solimp="0.5 0.5 0.001" condim="1"/>
    </body>
    <!-- 벽 왼면이 x=0.15 → 반경 0.05 구의 중심이 x_w=0.10 에서 접촉. k_eff = 10^4 N/m -->
    <geom name="wall" type="box" pos="0.25 0 0" size="0.1 0.2 0.2"
          solref="-2500 -1" solimp="0.5 0.5 0.001" condim="1"/>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(XML)
x_w, x_goal, v_cmd, ke = 0.10, 0.15, 0.10, 1.0e4   # gen_figs.py 와 동일한 명령

def run(K, B, T=6.0):
    d = mujoco.MjData(m)
    f6 = np.zeros(6)
    F_hist, in_prev, bounce = [], False, 0
    for i in range(int(T / m.opt.timestep)):
        t = i * m.opt.timestep
        xd = min(x_goal, v_cmd * t)
        vd = v_cmd if v_cmd * t < x_goal else 0.0
        d.qfrc_applied[0] = K * (xd - d.qpos[0]) + B * (vd - d.qvel[0])  # 임피던스 법칙 (E1)
        mujoco.mj_step(m, d)
        F = 0.0
        for c in range(d.ncon):
            mujoco.mj_contactForce(m, d, c, f6)
            F += f6[0]                                # 법선 성분
        in_c = F > 0
        bounce += int(in_prev and not in_c)
        in_prev = in_c
        F_hist.append(F)
    F_hist = np.array(F_hist)
    F_ss = F_hist[-5000:].mean()
    pen = d.qpos[0] - x_w
    return F_ss, F_hist.max(), bounce, (F_ss / pen if pen > 0 else float('nan'))

print("MuJoCo 1-DoF 벽 접촉 (k_eff = 10^4 N/m, 명령: 벽 뒤 5 cm) — 그림 1·2의 MuJoCo 판")
for name, K, B in [("WE-1 임피던스 K=200,  B=40  ", 200.0, 40.0),
                   ("B  저감쇠     K=200,  B=8   ", 200.0, 8.0),
                   ("A  고강성     K=1200, B=60  ", 1200.0, 60.0),
                   ("   위치 제어  K=2e4,  임계감쇠", 2.0e4, 2 * np.sqrt(2 * 2e4))]:
    F_ss, F_pk, bc, ke_eff = run(K, B)
    ks = K * ke / (K + ke)
    print(f"{name}: F_ss = {F_ss:7.2f} N (직렬 스프링 예측 {ks*0.05:7.2f}), "
          f"F_peak = {F_pk:7.2f} N, 바운스 {bc}회, 유효 벽 강성 {ke_eff:,.0f} N/m")
