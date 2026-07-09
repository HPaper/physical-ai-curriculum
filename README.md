# Physical AI 마스터 커리큘럼 — 로봇의 몸에서 지능까지, 하나의 흐름

> 로봇 시스템을 **감지 → 판단 → 제어 → 구동 → 로봇 → 환경**의 닫힌 루프로 보고, 그 각 부분을 아래에서 위로 쌓아 올라가는 단일 학습 흐름(66강).
> 최종 목표: 새로 나오는 VLA 논문/시스템을 봤을 때 **새로운 점만 파악하면 나머지는 이미 아는 상태**.
>
> **배경별 진입점** — 파트는 하나의 흐름이지만, 각 파트에 추천 배경 태그가 있다. 로봇 배경자는 Part 6(딥러닝)부터, 딥러닝 배경자는 Part 2(기구학)부터 훑어도 되고, 모두 **0강(전체 지도)**을 먼저 본다.
>
> v4 (2026-07-09): 상/하위 트랙 병합 → 단일 흐름 15파트·전역 재번호(00–65), 환경=시뮬레이션 파트 신설(Part 12), 0강 전체 지도 재설계(닫힌 루프 · 연산/인터페이스/물리).
> 이력: v3 2트랙(상위 AI·VLA / 하위 고전 로봇), 강의 깊이 기준(수식·worked example·코드·이미지), 강의별 참고문헌 규약.

---

## 🚧 진행 상황 (Work in Progress)

이 저장소는 **현재 집필 중**입니다. AI(Claude)와 함께 강의를 한 강씩 만들며, 아래가 파트별 현황입니다. (v3 = 수식 3단·실행 검증 코드·그림 ≥5 / v2 = 이론 골격, v3 증보 예정)

| Part | 강의 | 상태 |
|---|---|---|
| 1 · 전체 지도 | 0 | ✅ v3 |
| 2 · 로봇의 몸(기구학) | 1–8 | ✅ v3 |
| 3 · 로봇의 물리(동역학) | 9–13 | ✅ v3 |
| 4 · 구동(액추에이터) | 14–16 | ✅ v3 |
| 5 · 제어 | 17–24 | ✅ v3 |
| 6 · 딥러닝 기초 | 25–28 | ⬜ 미작성 |
| 7 · Transformer·LLM | 29–33 | ⬜ 미작성 |
| 8 · VLM | 34–36 | ⬜ 미작성 |
| 9 · 행동을 배우다 | 37–41 | ⬜ 미작성 |
| 10 · VLA 계보 | 42–48 | 🟡 v2 (증보 예정) |
| 11 · 실물 통합 | 49–50 | 🟡 v2 (증보 예정) |
| 12 · 환경·시뮬레이션 | 51–54 | 부분(52 ✅ v3 · 51·53·54 ⬜) |
| 13 · 데이터·평가 | 55–57 | ⬜ 미작성 |
| 14 · 시스템 통합 | 58–62 | ✅ v3 |
| 15 · 프론티어·자립 | 63–65 | ⬜ 미작성 |

- **지금 바로 학습 가능**: 0강 + Part 2–5(1–24강, 로봇의 몸·물리·구동·제어) + Part 14(58–62 시스템 통합) + 52강(시뮬 내부) + Part 10·11(42–50, VLA 계보·실물 통합, v2). 모두 코드·수치 검증 완료.
- **작성 방식**: 각 강의는 수식(직관→의미→형식 3단), 손계산 + 실행 가능한 검증 코드, 그림(≥5장)을 포함한다. 본문의 모든 인용 수치는 `images/lecNN/gen_figs.py` 실행으로 재현된다.

### 저장소 구조

```
part01-orientation/            0강 전체 지도
part02-kinematics/             1–8   로봇의 몸: DoF·SO(3)·SE(3)·FK·자코비안·IK·보간
part03-dynamics/               9–13  동역학·접촉·보행
part04-actuators/              14–16 모터·감속기·QDD
part05-control/                17–24 PID→LQR→computed torque→임피던스→MPC→WBC
part06-deep-learning/          25–28 딥러닝 기초           (예정)
part07-transformers-llm/       29–33 Transformer·LLM       (예정)
part08-vlm/                    34–36 VLM                   (예정)
part09-robot-learning/         37–41 모방·생성·RL          (예정)
part10-vla-lineage/            42–48 VLA 계보
part11-real-robot-integration/ 49–50 하드웨어·Action 파이프라인
part12-environment-simulation/ 51–54 시뮬레이터·물리엔진·합성데이터·world model
part13-data-evaluation/        55–57 데이터·벤치마크        (예정)
part14-system-integration/     58–62 ROS2·추정·식별·안전·종합
part15-frontier/               63–65 프론티어·논문읽기·캡스톤 (예정)
images/lecNN/                  강의별 그림 + 재현 스크립트(gen_figs.py)
```

---

## 이 자료의 사용법 (Claude와 함께 학습하기)

이 커리큘럼은 혼자 읽는 교과서가 아니라 **AI와 함께 진행하는 수업의 강의계획서**다.
각 강의는 md 파일 하나이며(`partNN-topic/lecNN-슬러그.md`), 학습 세션은 이렇게 진행한다:

1. Claude Code 세션을 열고 해당 강의 파일을 읽게 한다 → Claude가 강사 역할.
2. 본문을 함께 읽으며 질문 리스트의 질문을 던지고, 이해가 안 되는 부분은 그림/수식 전개를 요청한다.
3. worked example을 손으로 따라 풀고 검증 코드를 함께 실행한다.
4. 실습 섹션을 수행한다 (Colab 또는 로컬 GPU / 로봇·제어 파트는 NumPy·MuJoCo).
5. 마지막에 Claude에게 퀴즈를 받고, **내가 설명하면 Claude가 채점**한다 (Feynman 기법).

**강의 MD가 단일 원본(SSOT)이다.** 슬라이드(Marp)·조판 PDF(pandoc)는 정규 교육 운영 시점에 파생 생성한다.

### 강의 파일 공통 템플릿 (v3)

```markdown
# Lec NN. 제목
## 한 장 요약            ← 이 강의의 핵심을 그림 1개로
## 학습 목표              ← 3~5개, "~를 설명/유도/구현할 수 있다"
## 왜 이 강의가 필요한가    ← 문제의식: 이걸 모르면 무엇이 안 되는가
## 본문                   ← 개념 서술 + 그림. 아래 세 요소를 반드시 포함:
### 핵심 수식             ← 강의당 3개± . 각 수식을 3단으로: ① 직관 ② 물리·기하적 의미 ③ 형식(유도 요점)
### Worked Example        ← 손으로 푸는 예제 1~3개 + 검증 코드(실행 가능해야 함)
### {상대 배경}을 위한 번역 ← 로봇↔딥러닝 개념 대응 비유
## 흔한 오해              ← 반복되는 오개념 2~4개와 교정
## 실습                   ← 1.5~2시간, Colab/단일 GPU/시뮬레이터에서 실행 가능한 것만
## Claude와 토론할 질문    ← 5~7개
## 읽을거리               ← 논문/블로그 1~3개 + "어디까지만 읽으면 되는지"
## 자가 점검              ← 5개
## 참고문헌               ← 인용 규약(아래) 준수
```

### 깊이·분량 기준 (v3, 정규 교육 전제)

- **한 강의 = 하루 6시간 모듈**: 이론 3~4h(수식 3단 + worked example + 코드) / 실습 1.5~2h / 토론·자가 점검 ~1h. 전체 66강 ≈ 하루 6시간·주 5일 기준 약 3개월.
- **수식**: 핵심 수식을 피하지 않되 유도 전체가 아니라 "직관 → 의미 → 형식" 3단 제시. 쓰이지 않는 수식은 넣지 않는다.
- **코드**: 개념당 실행 가능한 최소 코드(NumPy/PyTorch/MuJoCo) ≥2개. "코드가 곧 정의"가 되도록.
- **이미지**: 강의당 ≥5개 — ① mermaid/SVG, ② 논문·웹 원본 그림(`images/lecNN/`, 캡션에 출처+참고문헌 번호), ③ 필요 시 생성형 AI 이미지(생성 표기).

### 인용 규약 (모든 강의 공통)

핵심 수치(파라미터 수, 성공률, 제어 주파수, 데이터 규모, 날짜 등)와 조사 기반 주장은 강의 말미 `## 참고문헌`에 출처를 남긴다:

1. **형식**: `[n] 저자/기관, "제목," arXiv:ID 또는 매체, 연도.월. 링크` — 논문은 arXiv 초록 페이지, 웹 문서는 원본 URL + 접속일.
2. **뒷받침 주석**: 각 항목에 `— **뒷받침**: ...`로 근거가 되는 본문의 구체적 수치·주장을 명시.
3. **출처 등급**: 언론·블로그 등 2차 출처는 `(2차)`, 검증 불가 항목은 "미검증/재확인 필요" 명시.
4. **선순위**: 논문 > 공식 기술 블로그·문서·저장소 > 언론. 회사 발표 수치는 "회사 발표"임을 밝힌다.
5. **미검증 내부 자료·비공개 문서는 인용 출처로 쓰지 않는다** — 구조·아이디어 참고는 가능하되, 내용·수치는 반드시 1차 자료로 크로스체크한 뒤 그 1차 자료를 인용 (설계 근거: 부록 E).

---

## 전체 흐름 (0강의 시스템 지도를 따라)

```
                        0강 · 전체 지도 (감지→판단→제어→구동→로봇→환경 닫힌 루프)
   ── 로봇의 몸에서 시작해 지능으로, 다시 통합·프론티어로 ──
   Part 2 기구학(1–8) → Part 3 동역학(9–13) → Part 4 구동(14–16) → Part 5 제어(17–24)
        → Part 6 딥러닝 기초(25–28) → Part 7 Transformer·LLM(29–33) → Part 8 VLM(34–36)
        → Part 9 행동을 배우다(37–41) → Part 10 VLA 계보(42–48)
        → Part 11 실물 통합(49–50) → Part 12 환경·시뮬(51–54)
        → Part 13 데이터·평가(55–57) → Part 14 시스템 통합(58–62) → Part 15 프론티어(63–65)
```

**왜 이 순서인가**: 큰 그림의 신호는 위(지능)→아래(물리)로 흐르지만, 학습은 만질 수 있는 아래(로봇·제어)부터 지능으로 쌓는 것이 forward-reference 없이 깔끔하다. 11강(실물 통합)이 제어(Part 5)와 VLA(Part 10) *뒤에* 오는 이유 — "정책이 낸 행동 a가 제어 스택으로 흘러 τ가 되어 로봇으로" 라는 다리가 앞의 두 흐름을 잇는다. 배경별로 진입점만 바꾸면 된다(로봇 배경자 → Part 6부터 / 딥러닝 배경자 → Part 2부터).

---

# 커리큘럼

## Part 1 — 전체 지도 [공통]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [0](part01-orientation/lec00-system-anatomy.md) | Physical AI 로봇 시스템 해부 | 감지→판단→제어→구동→로봇→환경의 **닫힌 루프**. 요소를 **연산(코드)·인터페이스(계약)·물리(로봇&환경)**로 구분. 판단(고/저)·제어(상/하)의 상·하 갈림, **행동 a→제어기→τ→로봇** 사슬, 정책 설계 3축(아키텍처·학습목적·표현), 환경 3형태(실제/시뮬/world model). 각 요소를 어느 강의가 파는지의 지도 — 이후 모든 강의가 참조 | 아는 시스템 하나를 이 지도로 분해 |

## Part 2 — 로봇의 몸: 기구학 [로봇·제어]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [1](part02-kinematics/lec01-robot-anatomy.md) | 로봇 해부학: 링크, 관절, 자유도 | 직렬/병렬 구조, DoF(Grübler), 여유자유도, 작업공간 vs 관절공간, 그리퍼·핸드 DoF | MuJoCo에서 URDF/MJCF 관절 트리 분석 |
| [2](part02-kinematics/lec02-rotation-so3.md) | 회전의 수학: SO(3) | 회전행렬, 오일러각·짐벌락, 축-각(Rodrigues), 쿼터니언. **비유: 임베딩 다양체 — 회전은 벡터공간이 아니다** | 표현 4종 상호 변환 + scipy 검증 |
| [3](part02-kinematics/lec03-se3-transforms.md) | 강체 변환: SE(3)와 좌표계 | 동차변환, 좌표계 합성, twist 맛보기, DH vs URDF | 카메라-로봇-물체 좌표계 체인 (hand-eye 맛보기) |
| [4](part02-kinematics/lec04-forward-kinematics.md) | 정기구학 (FK) | 링크 변환의 곱, **PoE(지수곱) 본류**, DH 대조, UR5e 예제 | UR5e FK NumPy 구현 + Pinocchio 대조 |
| [5](part02-kinematics/lec05-jacobian.md) | 자코비안 | 관절속도→EEF 속도, 기하/해석 자코비안, **정역학 쌍대 τ=JᵀF**. **비유: backprop chain rule** | 해석식+유한차분 자코비안 대조 |
| [6](part02-kinematics/lec06-singularity-manipulability.md) | 특이점과 조작성 | 특이점(SVD), 조작성 타원체, null-space | 2R/3R 조작성 히트맵 |
| [7](part02-kinematics/lec07-inverse-kinematics.md) | 역기구학 (IK) | 해석해·수치해(DLS), 다해성, null-space | DLS IK 구현·특이점 통과 |
| [8](part02-kinematics/lec08-interpolation-timing.md) | 보간과 시간 파라미터화 | **학습 스택 아래 최소 궤적론**: 다항식/사다리꼴, 저크 제한 — 액션 청크를 셋포인트로 펴기(50강 보간 계층). 고전 모션 플래닝은 의도적 제외 | 최소 저크 보간기 |

## Part 3 — 로봇의 물리: 동역학·접촉·보행 [로봇·제어]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [9](part03-dynamics/lec09-dynamics-ingredients.md) | 동역학의 재료 | 질량·관성텐서·운동량, 평행축 정리, Dzhanibekov 효과 | 링크 관성 파라미터 검증 |
| [10](part03-dynamics/lec10-lagrangian-dynamics.md) | 라그랑주 동역학 | **매니퓰레이터 방정식 M(q)q̈+C(q,q̇)q̇+g(q)=τ**, 수동성, 2링크 완전 유도 | 2링크 M,C,g sympy 유도 + 시뮬 |
| [11](part03-dynamics/lec11-newton-euler.md) | 뉴턴-오일러·계산 동역학 | RNEA/ABA/역동역학. **비유: forward/backward pass의 쌍** | RNEA 구현 + MuJoCo 대조 |
| [12](part03-dynamics/lec12-contact-friction-grasping.md) | 접촉·마찰·파지 | 쿨롱 마찰 원뿔, 접촉 불연속, form/force closure. **접촉 태스크가 어려운 물리적 근거** | MuJoCo 접촉 파라미터 스윕 |
| [13](part03-dynamics/lec13-underactuation-locomotion.md) | 부족구동·보행 역학 | 부족구동, ZMP, LIP, capture point, 수동보행 극한 사이클. **비유: 안정 어트랙터 학습** | LIP 보행 패턴 시뮬 |

## Part 4 — 구동: 액추에이터 [로봇·제어]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [14](part04-actuators/lec14-motors-current-loop.md) | 전기 모터와 전류 루프 | BLDC/PMSM, 토크 상수 Kt, 전류≈토크, FOC(kHz), 열 한계 — **50강 최하층의 정체** | 토크-속도 곡선·전류 스텝 응답 |
| [15](part04-actuators/lec15-gears-transmissions.md) | 감속기와 전동 | 하모닉/유성/**싸이클로이드(DYD)**/텐던 — 백래시·강성·역구동성·충격내성, 반사 관성 n²Jm | 반사 관성·가속 성능 계산 |
| [16](part04-actuators/lec16-qdd-integrated-actuators.md) | QDD와 통합 액추에이터 | MIT Cheetah proprioceptive actuation, 전류 기반 토크 추정, 관절 토크센서(Franka)와 비교 | QDD vs 하모닉 충격 응답 시뮬 |

## Part 5 — 제어: 명령을 운동으로 [로봇·제어]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [17](part05-control/lec17-feedback-control-basics.md) | 피드백 제어 최소 코스 | PID, 극점·안정성, 대역폭·위상여유·지연. **비유: 학습률·안정성, 지연=stale gradient** | 2차 플랜트 PID 튜닝 |
| [18](part05-control/lec18-state-space-lqr-kalman.md) | 상태공간·LQR·칼만 필터 | 가제어/가관측, LQR, 칼만 필터. **비유: LQR ↔ 가치함수가 2차식인 RL 해석해** | 도립진자 LQR + 칼만 |
| [19](part05-control/lec19-joint-control-computed-torque.md) | 관절 제어·computed torque | 독립 관절 PID 한계 → 역동역학 제어, 선형 오차 동역학, 중력 보상 | PID vs computed torque 추종 |
| [20](part05-control/lec20-operational-space-control.md) | 작업공간 제어 | operational space, 태스크 공간 관성, Jᵀ vs 역행렬 제어 | 관절 PID vs op-space 원 추종 |
| [21](part05-control/lec21-impedance-compliance.md) | 임피던스·컴플라이언스 제어 | **접촉 다루기**: 임피던스(M-B-K) 성형, 임피던스 vs 어드미턴스, Franka 내장 임피던스. **비유: 접촉 응답의 손실함수 성형** | 1-DoF 접촉 파라미터 스윕 |
| [22](part05-control/lec22-force-hybrid-control.md) | 힘 제어·하이브리드 제어 | 직접 힘 제어, 힘/위치 하이브리드, F/T vs 전류 추정 | peg-in-hole 위치 vs 하이브리드 |
| [23](part05-control/lec23-mpc.md) | MPC | 유한 구간 최적화 + receding horizon, QP 제약, 볼록 MPC 보행. **비유: planning 있는 model-based RL — 39강 Diffusion Policy와 같은 구도** | 카트폴 MPC(scipy) 제약 유무 비교 |
| [24](part05-control/lec24-whole-body-control.md) | 전신 제어 (WBC) | null-space 위계, QP 기반 WBC, 접촉 제약 토크 분배. **48강 Helix 02 S0·Atlas LBM 아래층** | 2태스크 우선순위 제어 시뮬 |

### Modern Robotics 장 매핑 (Part 2–5 기초 참고서)

기초 참고서 [Lynch & Park, *Modern Robotics*](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf)의 활용 범위. **Ch.9(궤적)·Ch.10(모션 플래닝)은 제외** — 그 층은 학습 정책(VLA)이 대체(8강은 최소 보간만 별도 자료).

| MR 장 | 대응 강의 | MR 장 | 대응 강의 |
|---|---|---|---|
| Ch.2 Configuration Space | 1 | Ch.6 Inverse Kinematics | 7 |
| Ch.3 Rigid-Body Motions | 2·3 | Ch.8 Dynamics of Open Chains | 9~11 |
| Ch.4 Forward Kinematics | 4 | Ch.11 Robot Control | 17(부분)·19~22 |
| Ch.5 Velocity Kinematics & Statics | 5·6 | Ch.12 Grasping & Manipulation | 12 |
| MR 범위 밖 | 13(보행)·14~16(액추에이터)·18(LQR/칼만)·23(MPC)·24(WBC)·Part 14 | | |

## Part 6 — 딥러닝 기초 [AI 기초]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 25 | 딥러닝, 왜 로봇에 필요한가 | 왜 역기구학+PID로는 빨래를 못 개는가. 룰베이스 → 학습 기반 패러다임 전환(0강 지도 위에서 AI 파트 로드맵) | 없음 (오리엔테이션) |
| 26 | 신경망 = 함수 근사기 | MLP, 활성함수, 경사하강법, 역전파. **비유: 최적화 기반 제어기 튜닝의 일반화** | Karpathy micrograd; 2링크 IK를 MLP로 근사 |
| 27 | 학습 파이프라인 해부 | 데이터셋/손실함수/미니배치/과적합/일반화. GPU가 하는 일 | PyTorch 학습 루프 바닥부터 |
| 28 ★ | CNN과 시각 표현 | 합성곱, 계층적 특징, 사전학습(ImageNet), ResNet + detection/SAM·affordance 조감 | 사전학습 ResNet 전이학습 |

핵심 자료: 3Blue1Brown, Karpathy "Zero to Hero" 1-4, CS231n.

## Part 7 — Transformer와 LLM [AI 기초]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 29 | 토큰과 임베딩 | BPE 토크나이저, 임베딩 공간. **복선: 로봇 행동도 토큰이 된다(42강 RT-2, 44강 FAST)** | HF tokenizers; 임베딩 시각화 |
| 30 | Attention 해부 | Q/K/V, self-attention, multi-head. **비유: 상태 의존적 gain scheduling** | attention 가중치 시각화 |
| 31 | Transformer 완성 | residual, LayerNorm, positional encoding, causal mask, KV 캐시 | nanoGPT 훈련 (수 분) |
| 32 ★ | LLM의 탄생 | 사전학습, 스케일링 법칙, 창발, in-context learning | nanoGPT 스케일 실험 또는 생략 |
| 33 | 사후학습: 모델 길들이기 | SFT, RLHF, **LoRA/PEFT(→ VLA 파인튜닝 핵심)**. **복선: 45강 RECAP의 RL post-training** | SmolLM2 LoRA 파인튜닝 |

핵심 자료: Karpathy "Let's build GPT", HF LLM Course.

## Part 8 — VLM: 로봇의 눈과 언어 [AI 기초]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 34 | ViT: 이미지를 패치 토큰으로 | 16×16 패치=단어, 해상도↔토큰 수, DINOv2(기하) vs CLIP류(의미) | ViT 추론 + 패치 임베딩 |
| 35 | CLIP → SigLIP | 대조학습, zero-shot, SigLIP이 VLA 백본 표준이 된 이유. SSL 계보(→ 63강 복선) | CLIP zero-shot 물체 분류 |
| 36 | VLM 조립: LLaVA 템플릿 | **encoder + projector + LLM** 도식 = 모든 VLA 백본의 뼈대. PaliGemma·Eagle·SmolVLM 미리보기 | SmolVLM LoRA VQA 파인튜닝 |

핵심 자료: HF CV Course, Umar Jamil "PaliGemma from scratch".

## Part 9 — 행동을 배우다: 모방·생성·RL [공통]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 37 | 모방학습이 무너지는 방식 | BC=지도학습, compounding error O(εT²), DAgger, causal confusion. **비유: 개루프 드리프트** | CS285 Lec 2 + BC+DAgger 축소판 |
| 38 | ACT와 action chunking | ALOHA, CVAE, temporal ensembling | LeRobot ACT를 PushT에서 훈련 |
| 39 | Diffusion Policy | 행동 다봉성, DDPM/DDIM 최소 수학, receding horizon. **비유: MPC의 확률적 사촌(23강)** | Diffusion Policy PushT |
| 40 | Flow matching | 직선 보간 + ODE(4~10스텝). **π0/GR00T/SmolVLA 액션 헤드 공통 선택** | MIT 6.S184 flow matching from scratch |
| 41 ★ | 강화학습 압축 코스 | MDP·정책경사·**advantage**·오프라인 RL·POMDP — 45강 RECAP을 읽기 위한 최소. **비유: 최적제어/HJB** (심화: 부록 D) | HF Deep RL PPO 또는 CS285 HW2 |

핵심 자료: **"Robot Learning: A Tutorial"(HF/LeRobot, arXiv 2510.12403)** — 이 파트의 교과서. MIT 6.S184.

## Part 10 — VLA 계보 [공통]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [42](part10-vla-lineage/lec42-birth-of-vla.md) | VLA의 탄생 (2022-23) | RT-1 → **RT-2("VLA" 명명·웹 지식 전이)** → OXE/RT-X(cross-embodiment) | RT-2 논문 그림 분석 |
| [43](part10-vla-lineage/lec43-open-generation.md) | 오픈 세대 (2024) | Octo, **OpenVLA**(7B>55B), OpenVLA-OFT(디코딩만 바꿔 26배 — 액션 헤드 중요성) | OpenVLA 코드 구조 읽기 |
| [44](part10-vla-lineage/lec44-pi-family-1.md) | π 패밀리 I: π0와 action expert | PaliGemma + flow 전문가(50Hz), **π0-FAST**(DCT 토크나이저) — 이산 vs 연속 논쟁 | openpi로 π0 추론 |
| [45](part10-vla-lineage/lec45-pi-family-2.md) | π 패밀리 II: 일반화와 RL | π0.5(이종 co-training·계층 추론), KI, **π*0.6/RECAP(RL post-training)**, π0.7 | π0.5 논문 Fig 정독 |
| [46](part10-vla-lineage/lec46-groot-family.md) | GR00T 패밀리 | dual-system, 데이터 피라미드, DreamGen, N1→N1.7, **N2=world action model 예고** | GR00T N1.7 LeRobot 로드 |
| [47](part10-vla-lineage/lec47-small-models-lineage.md) | 작은 모델들·계보도 총정리 | **SmolVLA(450M)**, TinyVLA, SimVLA, SpatialVLA, GO-1, RDT·CogACT. 계보도 완성 | 계보도 직접 그리기 |
| [48](part10-vla-lineage/lec48-proprietary-vla.md) | 비공개 진영의 지향점 | 회사별 베팅: Figure Helix(3계층), Skild, Tesla, 1X Redwood, BD+TRI LBM, Gemini Robotics, Agility. **여기서 System 2/1/0 벤더 라벨 첫 등장(경계 차이 단서 포함)** | 각 사 1차 자료 읽기 |

⚠️ **SmolVLA**(HF, 2025.6)와 **SimVLA**(2026.2)는 다른 모델. 실습은 SmolVLA.

## Part 11 — 실물 통합: 행동에서 로봇으로 [공통·교차]

> Part 5(제어)와 Part 10(VLA)이 만나는 지점.

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [49](part11-real-robot-integration/lec49-robot-hardware.md) | VLA 로봇 하드웨어 지형도 | 연구용 매니퓰레이터(Franka·UR·SO-101·OMY)·휴머노이드(G1·GR-1·Figure·NEO). 액추에이터 비교(하모닉·싸이클로이드·QDD·서보). 왜 RGB-only인가. 플랫폼↔데이터셋 지도 | 자기 로봇을 같은 틀로 스펙 분석 |
| [50](part11-real-robot-integration/lec50-action-pipeline.md) | Action의 여정: VLA에서 액추에이터까지 | **모델별 action space**(RT-2/OpenVLA/ACT/DP/π0/SmolVLA/GR00T/RDT). 청크 실행(temporal ensembling·RTC·async). **제어 계층**: VLA 1~50Hz → 보간/IK 100~1000Hz → 전류루프. 배포 토폴로지 | LeRobot async 코드 추적 |

## Part 12 — 환경: 시뮬레이션과 합성 데이터 [공통]

> 0강의 "환경 = 실제 / 물리 시뮬 / 학습된 world model"에서 뒤 두 형태를 다룬다.

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 51 ★ | 시뮬레이터 지형도 | MuJoCo/robosuite, ManiSkill3, Isaac Lab, Genesis, RoboCasa+MimicGen. sim2real 갭 | ManiSkill3/robosuite 구동 |
| [52](part12-environment-simulation/lec52-simulation-internals.md) | 시뮬레이션의 내부 | 물리엔진: 적분기(explicit/implicit), 접촉 솔버(LCP/soft), 타임스텝·안정성. **sim2real 물리 원인 목록** | 타임스텝·솔버 파라미터 실험 |
| 53 | 합성 데이터와 도메인 랜덤화 | MimicGen(데이터 증식), DreamGen/Cosmos(neural trajectory), 도메인 랜덤화 — "데이터를 시뮬로 만든다" | MimicGen류 데이터 증식 실습 |
| 54 | 학습된 world model (맛보기) | 행동 조건부 영상 생성이 환경을 대체(V-JEPA 2·Cosmos·1X WM·DreamZero). 63강 심화의 다리 | world model 추론 데모 |

## Part 13 — 데이터와 평가 [공통]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 55 | 데이터셋과 수집 | OXE·DROID·BridgeData·AgiBot World, LeRobotDataset v3. 수집 도구(ALOHA·GELLO·UMI·SO-101) | HF Hub 데이터셋 로드·시각화 |
| 56 | LeRobot 딥다이브 | 라이브러리 구조, 탑재 정책(ACT/DP/π0/SmolVLA/GR00T), async. **메인 실습 도구** | **SmolVLA를 LIBERO에 파인튜닝** — 최대 실습 |
| 57 | 벤치마크와 평가의 함정 | LIBERO(포화), **LIBERO-Plus(교란 시 붕괴)**, SimplerEnv, RoboArena/RoboChallenge. 평가 통계 현실(N=10~20, CI 없음) | LIBERO에서 파인튜닝 모델 평가 |

## Part 14 — 시스템 통합 [로봇·통합]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [58](part14-system-integration/lec58-robot-software-stack.md) | 로봇 소프트웨어 스택 | ROS 2(노드/토픽), 실시간성, 필드버스(EtherCAT/CAN), ros2_control | ROS 2 관절 명령 퍼블리시 |
| [59](part14-system-integration/lec59-state-estimation.md) | 상태 추정·센서 융합 | 엔코더/IMU/F-T, 상보 필터→EKF, 모멘텀 옵저버(외력 추정) | IMU+엔코더 융합 자세 추정 |
| [60](part14-system-integration/lec60-system-identification-calibration.md) | 시스템 식별·캘리브레이션 | 관성 파라미터 동정, 마찰 피팅, hand-eye. **비유: 고전판 "학습"** | 2링크 관성 파라미터 회귀 |
| [61](part14-system-integration/lec61-safety-layers.md) | 안전 계층 | 속도/힘/전력 제한, ISO 10218/TS 15066, e-stop, 학습 정책 아래 안전 필터(CBF) | 안전 필터(속도 제한 투영) |
| [62](part14-system-integration/lec62-two-stacks-meet.md) | 종합: 두 스택의 접점 | VLA 액션 청크가 보간→제어기→전류루프로 흐르는 전체 경로 재구성. 임피던스 하위층의 가치, 지연·주기 예산 합동 설계 | 통합 데모 + 다이어그램 완성 |

## Part 15 — 프론티어와 자립 [공통]

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 63 | 프론티어 지도 (2025-26) | ① RL post-training 물결 ② latent action(LAPA·GO-1) ③ **world model 수렴(V-JEPA 2·Cosmos·DreamZero: "VLA 다음은 WAM?")** ④ 효율화. **System 2/1/0 경계 엄밀 논의(GR00T vs Helix)** | 서베이 "An Anatomy of VLA Models" |
| 64 | 논문 읽기 프레임워크 | **6축 지도**(액션 디코딩/아키텍처/학습 레시피/데이터/평가/효율) + 비판적 읽기 체크리스트 10항목 + **층위 진단**(0강 지도로 "아키텍처⊥학습목적" 판별) | 체크리스트로 π0.7 논문 분석 |
| 65 ★ | 캡스톤 | 최근 논문 2편을 프레임워크로 완전 분석 → "새로운 점"만 요약. 정보 파이프라인 구축 | 논문 분석 리포트 2편 |

---

## 부록 A — 모델 계보 치트시트 (2026-07 기준)

```
2022.12  RT-1        스케일된 모방학습 실증 (35M, 행동=이산 토큰)
2023.07  RT-2        "VLA" 명명. 웹 사전학습 VLM에 행동 토큰 출력 (12B/55B, 비공개)
2023.10  OXE/RT-X    데이터 전환점: 1M 궤적, 22 로봇, cross-embodiment
2024.05  Octo        첫 완전 오픈 generalist (디퓨전 헤드, VLM 아님)
2024.06  OpenVLA     RT-2 레시피의 오픈 재현 (7B > 55B RT-2-X)
2024.10  π0          VLM + flow matching action expert 템플릿 확립 (50Hz)
2024.10  RDT-1B      순수 디퓨전 트랜스포머 / LAPA: latent action
2024.11  CogACT      인지(VLM)와 행동(DiT) 분리
2025.01  π0-FAST     DCT 기반 행동 토크나이저 (이산 진영 반격)
2025.02  Helix       dual-system 상용 실증 (7-9Hz VLM + 200Hz 제어, 비공개)
2025.02  OpenVLA-OFT 디코딩만 바꿔 26배 가속 — 액션 헤드의 중요성
2025.03  GR00T N1    휴머노이드 dual-system + 데이터 피라미드 / GO-1: ViLLA
2025.04  π0.5        open-world 일반화 (이종 co-training + 계층 추론)
2025.06  SmolVLA     450M 민주화 (커뮤니티 데이터, 소비자 GPU) / 1X Redwood (160M 온보드)
2025.07  TRI LBM     "신중한 검증" 논문: 디퓨전 LBM + 엄밀한 평가 방법론
2025.09  Gemini Robotics 1.5  ER 플래너 ↔ VLA 실행기, thinking-before-acting
2025.10  Atlas LBM   450M DiT 30Hz 전신 loco-manipulation (BD+TRI)
2025.11  π*0.6/RECAP RL post-training으로 실전 신뢰성 (18h 연속 작업)
2025.12  GR00T N1.6  Cosmos Reason 백본, 전신 제어
2026.01  Helix 02    3계층: S2 / S1 200Hz / S0 1kHz sim-trained 전신제어기
2026.02  SimVLA      0.5B 미니멀 베이스라인 (SmolVLA와 혼동 주의)
2026.03  GR00T N2 예고  world action model (DreamZero) — VLA→WAM 패러다임 이동?
2026.04  π0.7        스티어러블 generalist, 창발적 조합 일반화 (~5B)
```

오픈 가중치 + 단일 GPU 파인튜닝 난이도: SmolVLA(가장 쉬움) > Octo > TinyVLA > OpenVLA LoRA(24GB+) > π0/GR00T(40-80GB) > RDT-1B.

## 부록 B — 검증된 핵심 링크

**교과서 격 (AI·VLA)**
- Robot Learning: A Tutorial (HF, 2025.10): https://huggingface.co/spaces/lerobot/robot-learning-tutorial
- HF Robotics Course: https://huggingface.co/learn/robotics-course
- An Anatomy of VLA Models (살아있는 서베이): https://suyuz1.github.io/VLA-Survey-Anatomy/
- MIT 6.S184 Flow Matching & Diffusion: https://diffusion.csail.mit.edu/

**교과서 격 (로봇·제어)**
- Lynch & Park, Modern Robotics — **Part 2–5 기초 참고서**: PDF https://hades.mech.northwestern.edu/images/7/7f/MR.pdf | http://modernrobotics.org — **Ch.9-10(궤적·플래닝) 제외**(본문 매핑표)
- Russ Tedrake, Robotic Manipulation (무료, Drake): https://manipulation.csail.mit.edu/
- MuJoCo: https://mujoco.readthedocs.io | Pinocchio: https://github.com/stack-of-tasks/pinocchio

**기초 다지기**
- Karpathy Zero to Hero: https://karpathy.ai/zero-to-hero.html | 3Blue1Brown: https://www.3blue1brown.com/topics/neural-networks
- HF LLM Course: https://huggingface.co/learn/llm-course | HF CV Course: https://huggingface.co/learn/computer-vision-course

**모델 공식 소스**
- openpi (π0/π0.5): https://github.com/Physical-Intelligence/openpi | PI 블로그: https://www.pi.website/blog
- Isaac GR00T: https://github.com/NVIDIA/Isaac-GR00T | GEAR: https://research.nvidia.com/labs/gear/
- LeRobot: https://github.com/huggingface/lerobot | SmolVLA: https://huggingface.co/blog/smolvla

**비공개 진영 1차 자료 (48강)**
- Figure: Helix https://www.figure.ai/news/helix | Helix 02 https://www.figure.ai/news/helix-02 | Go-Big https://www.figure.ai/news/project-go-big
- Skild: https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain | LocoFormer: arXiv 2509.23745
- 1X Redwood: https://www.1x.tech/discover/redwood-ai | TRI LBM: https://toyotaresearchinstitute.github.io/lbm1/ | Atlas LBM: https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- Gemini Robotics 1.5: arXiv 2510.03342 | Agility WBC: https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model

**하드웨어·제어 스택 (49–50강, Part 2–5)**
- Franka FCI/libfranka: https://frankarobotics.github.io/docs/ (1kHz, 토크지령 중력·마찰 자동보상)
- UR RTDE: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html (e-Series 500Hz)
- SO-101: https://huggingface.co/docs/lerobot/so101 | LeRobot async: https://huggingface.co/docs/lerobot/en/async
- Real-Time Chunking: https://www.pi.website/research/real_time_chunking | DROID(Polymetis): arXiv 2403.12945
- Unitree G1: https://www.unitree.com/g1 | Fourier GR-1: http://support.fftai.cn/main/en/concepts/about_gr1/
- ROBOTIS DYD/Dynamixel-Y/OMY: https://emanual.robotis.com/docs/en/all-dyd/ | https://ai.robotis.com/omy/hardware_omy

**정보 파이프라인 (64–65강에서 구축)**
- HF Daily Papers: https://huggingface.co/papers | alphaXiv: https://www.alphaxiv.org/ | arXiv cs.RO RSS: https://rss.arxiv.org/rss/cs.RO
- Humanoids Daily: https://www.humanoidsdaily.com/ | Import AI: https://importai.substack.com/
- 학회: CoRL(11월) > RSS > ICRA/IROS. Papers with Code는 2025.7 폐쇄 — HF Papers로 대체.

## 부록 C — 실습 환경 요구사항

- **CPU만으로 충분**: Part 2–5(NumPy/sympy/MuJoCo-CPU), Part 6–8 개념, cvxpy 없이 scipy로 MPC
- **Colab 무료/T4**: Part 6–9 딥러닝 실습, ACT/Diffusion Policy 훈련, SmolVLA 추론, LIBERO, SimplerEnv
- **Colab A100 또는 로컬 24GB**: SmolVLA 파인튜닝(~4h/20k steps), OpenVLA LoRA
- **로컬 8GB+**: π0 추론 (openpi)
- **선택**: ROS 2(도커 권장, Part 14), Isaac Lab(RTX, Part 12)
- **선택 하드웨어**: SO-101(~$220) 또는 Robotis OMY — 실물 teleop→수집→훈련 루프

## 부록 D — CS285 → VLA 심화 트랙 (선택)

CS285(버클리, Sergey Levine)는 2026 봄학기에 "CS 185/285"로 개편(BC 2강 분리, "LLM RL" 신설, Inverse RL·Meta-RL 삭제). **공개 영상은 Fall 2023 녹화본**이므로 아래 번호는 Fall 2023 기준.

| CS285 강의 | 내용 | 풀리는 VLA 개념 | 우선순위 |
|---|---|---|---|
| L2 (+HW1) | 모방학습, DAgger | 본편 37강 | 필수 (본편) |
| L4 | MDP, RL 표기법 | 모든 RL-VLA 논문의 언어 | 높음 |
| L5 (+HW2) | Policy gradient, baseline | VLA RL 파인튜닝(PPO/GRPO) | 높음 |
| L6 | Actor-critic, **advantage**, GAE | **45강 RECAP의 advantage conditioning** | 높음 |
| L15-16 | **오프라인 RL** (CQL, IQL, AWR) | RECAP = 오프라인 RL 정책 추출의 VLA 적용. **VLA 시대 최고 가치의 2강** | 높음 |
| L9 | Importance sampling, TRPO→PPO | RL 파인튜닝이 base 정책 근처에 머무는 이유 | 중간 |
| L7(-8) | 가치함수, Q러닝 | RECAP critic 절반, HIL-SERL | 중간 |
| L11(-12) | 모델 기반 RL | **world model**(Dreamer, WAM) 기반 | 중간 |
| L21 | 시퀀스 모델 RL | VLA = VLM 정책이므로 직결 | 높음 (짧음) |
| L20 | Inverse RL | 보상 모델, VLM-as-reward | 낮음 |
| L18-19 | VI/생성모델, control as inference | 디퓨전/FM 헤드 유도까지 원할 때만 | 선택 |

**건너뛰어도 되는 것**: L3(PyTorch — 본편 27강 대체), **L10(LQR/MPC — 18·23강 대체)**, L13-14(exploration), L17(이론), L22(meta-RL), L23.
**과제**: HW1(필수급) → HW2 → (여유 시) HW5 오프라인 RL. HW3(Atari DQN) 스킵.
**Levine의 VLA 시대 자료**: Dwarkesh 팟캐스트 2025.9, Substack "Sporks of AGI"(2025.7).

## 부록 E — 설계 근거: 전체 지도와 커리큘럼 원칙

이 커리큘럼이 지금의 형태(0강의 닫힌-루프 지도, 단일 흐름 15파트, v3 강의 템플릿)를 갖게 된 설계 원칙.

**핵심 설계 판단 — 왜 "계층 스택"이 아니라 "닫힌 루프 + 종류 구분"인가.** Physical AI를 흔히 "지각→…→플랜트"의 단일 계층 스택으로 그린다. 그러나 그 요소들은 종류가 다르다: 어떤 것은 **연산(코드)**, 어떤 것은 **인터페이스(계약)**, 어떤 것은 **물리(로봇 & 환경, 코드 아님)**이고, 태스크·목표는 **입력(명세)**이다. 이들을 한 줄에 번호로 세우면 "VLA vs RL"(아키텍처 vs 학습 목적), "controller를 없앴다"(물리·계약은 제거 불가), "행동 a가 로봇으로 직행"(실제로는 제어기를 거쳐 τ로) 같은 **범주 오류**를 부른다. 그래서 0강은 단일 스택 대신 **감지→판단→제어→구동→로봇→환경의 닫힌 루프**를 세우고, 각 요소를 연산/인터페이스/물리로 구분하며, 어느 강의가 그것을 파는지까지 매핑한다.

| 설계 원칙 | 커리큘럼에서의 구현 |
|---|---|
| 종류(코드/계약/물리)를 구분하는 닫힌-루프 지도로 개념 혼동 제거 | **0강** — 전체 지도, 64강 "층위 진단"으로 연결 |
| 두 배경(로봇 ↔ AI)을 하나의 흐름에 통합, 진입점은 자율 선택 | **단일 15파트**(트랙 폐지) + 파트별 배경 태그 |
| 판단·제어의 **상/하 대칭**(느린 전역 상위 / 빠른 국소 하위) | 0강 §2 확대도, Part 5(제어)·Part 9-10(판단) |
| 환경은 세 형태(실제/시뮬/world model) — 시뮬도 환경 | **Part 12 신설**(51–54) |
| 벤더 "System 2/1/0"은 경계가 제각각이라 전체 지도엔 미사용 | 모델별(48강)·프론티어(63)·논문읽기(64)에서만, 엄밀한 단서와 함께 |
| 개념마다 문제의식 → 직관 → 수식 3단 → worked example(손계산+실행 코드) → 흔한 오해 | **강의 템플릿 v3**, 인용 수치는 코드 실행으로 재현 |
| "아키텍처 ⊥ 학습 목적 ⊥ 행동 표현" 직교 분해를 논문 읽기 도구로 | 0강 §3, 64강 프레임워크 |

이 커리큘럼만의 차별점: 모델 계보의 역사(Part 10), 생태계·벤치마크(Part 13), 하드웨어 스펙(49강), 비공개 기업 분석(48강), 실배포 수치(50강), 전 강의의 실행 코드·재현 그림.
