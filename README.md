# Physical AI 마스터 커리큘럼 — 상위제어(AI·VLA) × 하위제어(고전 로봇)

> 두 방향의 학습자를 위한 팀 교육용 커리큘럼:
> **① 로봇은 알지만 딥러닝은 모르는 사람** → 상위제어 트랙 (딥러닝 → Transformer → VLM → VLA)
> **② 딥러닝은 알지만 로봇은 모르는 사람** → 하위제어 트랙 (기구학 → 동역학 → 액추에이터 → 제어)
> 최종 목표: 새로 나오는 VLA 논문/시스템을 봤을 때 **새로운 점만 파악하면 나머지는 이미 아는 상태**.
>
> v3 (2026-07-08): 하위제어 트랙 신설(30강), 강의 깊이 기준 도입(수식·worked example·코드·이미지), 공용 0강 신설,
> 분량 재설계(트랙당 하루 6시간 × 주 5일 × 약 6주 ≈ 180시간). 팀 정규 교육과정 승격을 전제로 설계.
> v2 이하 이력: 33강 단일 트랙(상위제어), 강의별 참고문헌 규약, 리서치 기반 팩트체크.

---

## 🚧 진행 상황 (Work in Progress)

이 저장소는 **현재 집필 중**입니다. AI(Claude)와 함께 강의를 한 강씩 만들어 나가고 있으며, 아래 표가 실시간 현황입니다.

| 구획 | 강의 | 상태 |
|---|---|---|
| 공용 오리엔테이션 | `00` | ✅ 완료 (v3: 수식·실행 코드·그림) |
| **하위제어 트랙** (고전 로봇) | `R01`–`R30` (전 30강) | ✅ **완료** (v3, 검증 완료 — 그림 118장·코드 전부 실행 검증) |
| 상위제어 트랙 (AI·VLA) | `18`–`26` (9강) | 🟡 초안(v2 골격) — v3 깊이로 증보 예정 |
| 상위제어 트랙 (AI·VLA) | `01`–`17`, `27`–`33` | ⬜ 미작성 |

- **지금 바로 학습 가능**: 공용 00강 + 하위제어 트랙 R01~R30 (딥러닝 배경자가 고전 로봇 제어를 처음부터 배우는 6주 과정, 전부 검증 완료).
- **작성 방식**: 각 강의는 수식(직관→의미→형식 3단), 손계산 + 실행 가능한 검증 코드, 논문/생성 그림(≥5장)을 포함한다. 본문의 모든 인용 수치는 `images/<강의>/gen_figs.py` 실행으로 재현된다.
- 진행 방식·깊이 기준·인용 규약은 아래 "이 자료의 사용법"과 "깊이·분량 기준" 절 참고.


### 저장소 구조

```
00-common/                              공용 오리엔테이션 (Lec 00)
upper-part5-vla-lineage/                상위 트랙 Part 5 (Lec 18–24: VLA 계보)
upper-part6-real-robot-integration/     상위 트랙 Part 6 (Lec 25–26: 하드웨어·Action 파이프라인)
lower-part1-robot-body/                 하위 트랙 Part R1 (R01–R04: DoF·SO(3)·SE(3)·FK)
lower-part2-differential-kinematics/    하위 트랙 Part R2 (R05–R08: 자코비안·IK·보간)
lower-part3-dynamics/                   하위 트랙 Part R3 (R09–R13: 동역학·접촉·보행)
lower-part4-actuators/                  하위 트랙 Part R4 (R14–R16: 모터·감속기·QDD)
lower-part5-control/                    하위 트랙 Part R5 (R17–R24: PID→임피던스→MPC→WBC)
lower-part6-system-integration/         하위 트랙 Part R6 (R25–R30: ROS2·심·추정·안전·종합)
images/<강의>/                          강의별 그림 + 재현 스크립트(gen_figs.py)
```

---

## 이 자료의 사용법 (Claude와 함께 학습하기)

이 커리큘럼은 혼자 읽는 교과서가 아니라 **AI와 함께 진행하는 수업의 강의계획서**다.
각 강의는 md 파일 하나이며(상위 트랙 `part{N}/lec{NN}-제목.md`, 하위 트랙 `partR{N}/lecR{NN}-제목.md`), 학습 세션은 이렇게 진행한다:

1. Claude Code 세션을 열고 해당 강의 파일을 읽게 한다 → Claude가 강사 역할.
2. 본문을 함께 읽으며 질문 리스트의 질문을 던지고, 이해가 안 되는 부분은 그림/수식 전개를 요청한다.
3. worked example을 손으로 따라 풀고 검증 코드를 함께 실행한다.
4. 실습 섹션을 수행한다 (Colab 또는 로컬 GPU / 하위 트랙은 NumPy·MuJoCo).
5. 마지막에 Claude에게 퀴즈를 받고, **내가 설명하면 Claude가 채점**한다 (Feynman 기법).

**강의 MD가 단일 원본(SSOT)이다.** 슬라이드(Marp)·조판 PDF(pandoc)는 정규 교육 운영 시점에 파생 생성한다.

### 강의 파일 공통 템플릿 (v3)

```markdown
# Lec NN. 제목

## 한 장 요약            ← 이 강의의 핵심을 그림 1개로
## 학습 목표              ← 3~5개, "~를 설명/유도/구현할 수 있다"
## 왜 이 강의가 필요한가    ← 문제의식: 이걸 모르면 무엇이 안 되는가
## 본문                   ← 개념 서술 + 그림. 아래 세 요소를 반드시 포함:
### 핵심 수식             ← 강의당 3개± . 각 수식을 3단으로: ① 직관(말로) ② 물리·기하적 의미 ③ 형식(유도 요점)
### Worked Example        ← 손으로 푸는 예제 1~3개 + 검증 코드(NumPy/PyTorch, 실행 가능해야 함)
### {상대 배경}을 위한 번역 ← 상위 트랙: 제어 개념으로 비유 / 하위 트랙: 딥러닝 개념으로 비유
## 흔한 오해              ← 이 주제에서 반복되는 오개념 2~4개와 교정
## 실습                   ← 1.5~2시간, Colab/단일 GPU/시뮬레이터에서 실행 가능한 것만
## Claude와 토론할 질문    ← 5~7개
## 읽을거리               ← 논문/블로그 1~3개 + "어디까지만 읽으면 되는지"
## 자가 점검              ← 5개
## 참고문헌               ← 인용 규약(아래) 준수
```

### 깊이·분량 기준 (v3, 정규 교육 전제)

- **한 강의 = 하루 6시간 모듈**: 이론 본문 3~4h (수식 3단 제시 + worked example + 코드) / 실습 1.5~2h / 토론·자가 점검 ~1h.
- **수식**: 핵심 수식을 피하지 않는다. 단 유도 전체가 아니라 "직관 → 의미 → 형식"의 3단 제시. 쓰이지 않는 수식은 넣지 않는다.
- **코드**: 개념당 실행 가능한 최소 코드(NumPy/PyTorch/MuJoCo) ≥2개. "코드가 곧 정의"가 되도록 (예: 자코비안을 유한차분으로 검증, attention을 20줄로 구현).
- **이미지**: 강의당 ≥5개 — ① mermaid/SVG 다이어그램, ② **논문·웹 원본 그림** (`images/lecNN/`에 저장, 캡션에 출처 + 참고문헌 번호), ③ 필요 시 **생성형 AI 이미지** (생성 표기). 플로우차트+텍스트만으로 구성된 강의는 v3 기준 미달.
- 기존 작성분(18~26강)은 v2 깊이(이론 골격+실습) 기준 — **v3 기준으로 증보 대상**이다. 증보 요청 시 강의 단위로 진행.

### 인용 규약 (모든 강의 공통)

본문에 등장하는 핵심 수치(파라미터 수, 성공률, 제어 주파수, 데이터 규모, 날짜 등)와 조사 기반 주장은 강의 말미 `## 참고문헌`에 출처를 남긴다:

1. **형식**: `[n] 저자/기관, "제목," arXiv:ID 또는 매체, 연도.월. 링크` — 논문은 arXiv 초록 페이지, 웹 문서는 원본 URL + 접속일.
2. **뒷받침 주석**: 각 항목에 `— **뒷받침**: ...`로 이 문헌이 근거가 되는 본문의 구체적 수치·주장을 명시한다.
3. **출처 등급 표시**: 언론·블로그 등 2차 출처는 `(2차)`, 검증 불가 항목은 "미검증/재확인 필요"를 명시한다.
4. **선순위**: 논문 > 공식 기술 블로그·문서·저장소 > 언론 보도. 회사 발표 수치는 "회사 발표"임을 밝힌다.
5. **미검증 내부 자료·비공개 문서는 인용 출처로 쓰지 않는다** — 구조·아이디어 참고는 가능하되, 내용·수치는 반드시 1차 자료로 크로스체크한 뒤 그 1차 자료를 인용한다 (설계 근거: 부록 E).

---

## 트랙 구조와 분량

| | 상위제어 트랙 (AI·VLA) | 하위제어 트랙 (고전 로봇) |
|---|---|---|
| 대상 | 로봇 전문, 딥러닝 초보 | 딥러닝 전문, 로봇 초보 |
| 내용 | 딥러닝 → Transformer/LLM → VLM → 로봇 학습 → VLA 계보 → 실물 통합 → 생태계 → 프론티어 | 기구학 → 동역학 → 액추에이터 → 제어(임피던스·MPC·WBC) → 시스템 통합 |
| 분량 | 공용 0강 + 33강 ≈ **34일 (약 7주)** | 30강 = **30일 (6주)** |
| 리듬 | 하루 6시간(이론 4h + 실습 2h), 주 5일 | 동일 |
| 실습 도구 | Colab/GPU, LeRobot, LIBERO | NumPy, MuJoCo, Pinocchio, (가능 시) 실기 |
| 번역 박스 | 제어 개념으로 비유 | 딥러닝 개념으로 비유 |

- **공용 0강**은 두 트랙 모두의 1일차다.
- **교차 강의**: 상위 25·26강 ↔ 하위 R15·R30은 두 트랙이 만나는 지점 — 팀 합동 세션으로 운영하기 좋다.
- 개인 학습(비정규) 시 빠른 트랙: 상위 트랙에서 ★ 표시(04·08·17·29·33) 생략 + Part 1은 아는 만큼 건너뛰기 (~26강).

## 전체 지도

```
                        ┌── 공용 Lec 00: Physical AI 시스템 해부 (계층 좌표계) ──┐
                        │                                                      │
   상위제어 트랙 (34일)   ▼                                          하위제어 트랙 (30일)
   Part 1 딥러닝 기초 ──▶ Part 2 Transformer·LLM ──▶ Part 3 VLM      Part R1 로봇이라는 기계 (DoF·SO(3)·SE(3)·FK)
        └──▶ Part 4 로봇 학습 (IL·디퓨전·flow·RL)                     Part R2 자코비안·IK·궤적
   Part 5 VLA 계보 (RT-x → π → GR00T → 오픈/비공개)                   Part R3 동역학 (라그랑주·접촉·보행 역학)
   Part 6 실물 로봇 통합 (하드웨어·Action 파이프라인) ◀━━ 교차 ━━▶     Part R4 액추에이터 (모터·감속기·QDD)
   Part 7 생태계 (LeRobot·데이터·벤치마크)                            Part R5 제어 (PID→토크→임피던스→MPC→WBC)
   Part 8 프론티어 (world model·RL 물결·논문 읽기)                    Part R6 시스템 통합 (ROS2·심·추정·안전)
```

VLA = **VLM 백본**(상위 Part 2-3) + **액션 헤드**(상위 Part 4)의 결합이고, 그 출력은 **하위제어 스택**(하위 Part R4-R5)으로 흘러 들어간다. 두 트랙을 모두 마친 사람이 그 경계면(상위 26강 = 하위 R30)을 완전히 이해한다.

---

# 공용 오리엔테이션

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [00](00-common/lec00-system-anatomy.md) | Physical AI 시스템 해부: 개념 좌표계 | 로봇 시스템을 **연산 블록(코드) · 인터페이스(계약) · 물리 기질(코드 아님)** 세 종류 + 입력(명세)으로 구분하는 공용 좌표계 도입 — 무엇이 코드이고 무엇이 물리이고 무엇이 계약인지. "VLA는 정책, 컨트롤러는 정책 아님", "RL은 아키텍처가 아니라 학습 목적" 같은 범주 오류를 걷어낸다. **각 구성요소를 어느 강의가 파고드는지의 지도** 포함 — 이후 모든 강의가 이 좌표를 참조 | 아는 시스템 하나(자율주행, 로봇셀, 드론)를 세 종류로 분해하고 각 인터페이스의 신호를 명시해 보기 |

*이 강의의 세 종류 좌표계는 본 커리큘럼의 자체 설계다. 설계 근거·판단 기준은 부록 E 참조.*

---

# 상위제어 트랙 — AI·VLA (Part 1~8, 33강)

## Part 1 — 딥러닝 기초: 로봇공학자의 언어로 (4강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 01 | Physical AI 전체 지도 | 왜 역기구학+PID로는 빨래를 못 개는가. 룰베이스 → 학습 기반 제어 패러다임 전환. 상위 트랙 전체의 로드맵과 용어 지도 (00강의 좌표계 위에서) | 없음 (오리엔테이션) |
| 02 | 신경망 = 함수 근사기 | MLP, 활성함수, 경사하강법, 역전파. **비유: 뉴턴법/최적화 기반 제어기 튜닝의 일반화** | Karpathy micrograd 따라 만들기; 2링크 암 역기구학을 MLP로 근사해보기 |
| 03 | 학습 파이프라인 해부 | 데이터셋/손실함수/미니배치/과적합/일반화/하이퍼파라미터. GPU가 하는 일 | PyTorch 학습 루프를 바닥부터 작성 |
| 04 ★ | CNN과 시각 표현 | 합성곱, 계층적 특징, 사전학습(ImageNet) 개념. ResNet까지 + **detection/segmentation/SAM·affordance 조감** (VLA 논문에 등장하는 만큼만) | 사전학습 ResNet으로 전이학습 |

핵심 자료: 3Blue1Brown Neural Networks 시리즈(시각적 직관), Karpathy "Zero to Hero" 1-4편, CS231n 노트.

## Part 2 — Transformer와 LLM (5강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 05 | 토큰과 임베딩 | 세상을 시퀀스로: BPE 토크나이저, 임베딩 공간. **복선: 나중에 로봇 행동도 토큰이 된다(RT-2, FAST)** | HF tokenizers 실험; 임베딩 시각화 |
| 06 | Attention 해부 | Q/K/V, self-attention, multi-head. **비유: 상태 의존적 gain scheduling / 연관 메모리** | attention 가중치 시각화 (3B1B 영상과 병행) |
| 07 | Transformer 완성 | residual, LayerNorm, positional encoding, causal mask, KV 캐시. autoregressive 생성 | nanoGPT를 셰익스피어 데이터로 훈련 (GPU 수 분) |
| 08 ★ | LLM의 탄생 | 사전학습, 스케일링 법칙, 창발 능력, in-context learning. GPT 계열사 | nanoGPT 스케일 실험 또는 생략 |
| 09 | 사후학습: 모델 길들이기 | SFT, RLHF 개념, **LoRA/PEFT (→ VLA 파인튜닝의 핵심 기술)**. **복선: RECAP 등 VLA의 RL post-training** | SmolLM2를 LoRA로 파인튜닝 (HF LLM course 노트북) |

핵심 자료: Karpathy "Let's build GPT" + "Deep Dive into LLMs", HF LLM Course, 3B1B GPT/attention 챕터.

## Part 3 — VLM: 로봇의 눈과 언어를 연결 (3강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 10 | ViT: 이미지를 패치 토큰으로 | 16×16 패치 = 단어. 해상도↔토큰 수 계산 (896px = 4096토큰 — 카메라 3대면 지연 폭발). DINOv2(기하) vs CLIP류(의미) 특징의 차이 | ViT 추론 + 패치 임베딩 뜯어보기 |
| 11 | CLIP → SigLIP: 시각-언어 정렬 | 대조학습, zero-shot 분류. SigLIP이 VLA 백본 표준이 된 이유. **SSL 계보 조감**(contrastive → non-contrastive → JEPA — 31강의 복선) | CLIP zero-shot으로 로봇 작업장 물체 분류 |
| 12 | VLM 조립: LLaVA 템플릿 | **encoder + projector + LLM** 이 하나의 도식이 모든 VLA 백본의 뼈대. PaliGemma(→π0), Eagle/Cosmos-Reason(→GR00T), SmolVLM(→SmolVLA) 미리보기 | SmolVLM을 LoRA로 소형 VQA 파인튜닝; 로봇 장면 이해 테스트 |

핵심 자료: HF CV Course Unit 3-4, HF 블로그 "Vision Language Models Explained"(2024)/"VLMs 2025", Umar Jamil "PaliGemma from scratch" 영상.

## Part 4 — 로봇 학습: VLA의 나머지 반쪽 (5강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 13 | 모방학습이 무너지는 방식 | BC = 지도학습. compounding error O(εT²), distribution shift, DAgger, causal confusion. **비유: 개루프 제어의 드리프트** | CS285 Lec 2 시청 + **CS285 HW1(BC+DAgger) 축소판** |
| 14 | ACT와 action chunking | ALOHA, CVAE, temporal ensembling. 청크 예측이 compounding error를 나누는 원리 | **LeRobot ACT를 PushT에서 훈련** (공식 Colab) |
| 15 | Diffusion Policy | 행동의 다봉성(pass-left vs pass-right): MSE 회귀는 평균을 내서 장애물로 돌진한다. DDPM/DDIM 최소 수학, receding horizon. **비유: MPC의 확률적 사촌** | Diffusion Policy PushT Colab |
| 16 | Flow matching | 직선 보간 + ODE 적분 = 디퓨전보다 적은 스텝(4~10). **π0/GR00T/SmolVLA 액션 헤드의 공통 선택**인 이유 | MIT 6.S184 랩 1 (flow matching from scratch) |
| 17 ★ | 강화학습 압축 코스 | MDP·정책경사·**advantage**(CS285 L4-L6 요약) + **오프라인 RL 개념(L15-16)** + POMDP 한 절 — π*0.6/RECAP의 advantage conditioning을 읽기 위한 최소 세트. **비유: 최적제어/HJB와의 관계** (심화: 부록 D CS285 트랙) | HF Deep RL course PPO 노트북 또는 CS285 HW2 |

핵심 자료: **"Robot Learning: A Tutorial"(HF/LeRobot, 2025.10, arXiv 2510.12403)** — 이 파트 전체의 교과서. MIT 6.S184(2026) 강의노트. CS285는 Lec 2만 본편에서 다루고, 심화는 부록 D.
수학 범위 가이드: ε-예측 MSE, DDIM, FM 선형보간까지만. score-SDE/ELBO 유도는 생략해도 논문 읽기에 지장 없음.

## Part 5 — VLA 계보: 모델들의 진화사 (7강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [18](upper-part5-vla-lineage/lec18-birth-of-vla.md) | VLA의 탄생 (2022-23) | RT-1(멀티태스크 transformer 정책 실증) → **RT-2("VLA" 명명: 행동=텍스트 토큰, 웹 지식 전이)** → OXE/RT-X(cross-embodiment 데이터 전환, ~50% 향상) | RT-2 논문 그림 분석 |
| [19](upper-part5-vla-lineage/lec19-open-generation.md) | 오픈 세대 (2024) | Octo(디퓨전 헤드), **OpenVLA**(7B가 55B RT-2-X를 이김, 오픈 표준), OpenVLA-OFT(디코딩 방식만 바꿔 26배 빠름 — 액션 헤드가 백본만큼 중요하다는 교훈) | OpenVLA 코드 구조 읽기 (Claude와 함께) |
| [20](upper-part5-vla-lineage/lec20-pi-family-1.md) | π 패밀리 I: π0와 action expert | PaliGemma + 300M flow 전문가(50Hz 청크). **π0-FAST**: DCT+BPE 행동 토크나이저 — 이산 AR vs 연속 flow 논쟁의 양쪽을 한 회사가 다 만듦 | **openpi로 π0 추론 실행** (8GB+ VRAM) |
| [21](upper-part5-vla-lineage/lec21-pi-family-2.md) | π 패밀리 II: 일반화와 RL | π0.5(이종 데이터 co-training, 계층적 추론 → 처음 보는 집), Knowledge Insulation, **π*0.6/RECAP(RL post-training: 보정 teleop + advantage conditioning, 18시간 에스프레소)**, π0.7(steerable generalist, 2026.4) | π0.5 논문 Fig 정독 |
| [22](upper-part5-vla-lineage/lec22-groot-family.md) | GR00T 패밀리 | dual-system(System2 VLM ~10Hz + System1 DiT ~120Hz), 데이터 피라미드(웹/합성/실기), DreamGen 합성 데이터, N1→N1.5(백본 동결)→N1.6(Cosmos Reason)→N1.7. **N2 = world action model 예고** | GR00T N1.7을 LeRobot에서 로드 |
| [23](upper-part5-vla-lineage/lec23-small-models-lineage.md) | 작은 모델들과 계보도 총정리 | **SmolVLA(450M, 구조 상세 분석 — 커뮤니티 데이터만으로 훈련)**, TinyVLA, SimVLA(2026.2, 동명이인 주의), SpatialVLA(3D prior), GO-1(latent action), RDT-1B·CogACT. 오픈 진영 계보도 1장 완성 | 계보도를 직접 그려보고 Claude와 검증 |
| [24](upper-part5-vla-lineage/lec24-proprietary-vla.md) | 비공개 진영의 지향점 | 회사별 "베팅" 분류로 배우는 설계 철학: **Figure Helix→Helix 02**(7B S2 7-9Hz / 80M S1 200Hz / 10M S0 1kHz 3계층), **Skild**(omni-bodied, RL+심 우선, "VLM+어댑터" 비판 — LocoFormer로 방법 유추), **Tesla**(단일 end-to-end + neural world simulator, FSD 인프라 공유), **1X Redwood**(160M 온보드 VLA + world model 평가 + Expert Mode teleop 플라이휠), **BD+TRI LBM**(450M DiT 30Hz를 MPC급 전신 제어 위에), **Gemini Robotics 1.5**(ER 플래너 ↔ VLA 실행기, Motion Transfer), **Agility**(<1M LSTM 제어기 + 엔지니어링 스택 — 고전 로봇공학과 가장 가까움) | 각 사 기술 블로그 1차 자료 읽기 (부록 B) + "누가 옳은가" 토론 |

⚠️ 명칭 주의: **SmolVLA**(HF, 2025.6, 유명한 것)와 **SimVLA**(2026.2, 0.5B 미니멀 베이스라인 논문)는 다른 모델. 둘 다 다루되 실습은 SmolVLA.

## Part 6 — 실물 로봇 통합: 하드웨어와 제어 스택 (2강)

> 두 트랙의 교차 지점. 하위제어 트랙의 R15·R30과 합동 세션 권장.

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [25](upper-part6-real-robot-integration/lec25-robot-hardware.md) | VLA 로봇 하드웨어 지형도 | **연구용 매니퓰레이터**: Franka Panda/FR3(7DoF, 하모닉+전관절 토크센서, FCI 1kHz), UR5e(6DoF, RTDE 500Hz), WidowX-250(→Bridge), ALOHA 2(Dynamixel 리더-팔로워), SO-101(Feetech STS3215 버스서보, ~$130), Robotis OMY(Dynamixel Y). **휴머노이드**: Unitree G1/H1(QDD형 저감속 유성기어), Fourier GR-1(FSA, GR00T 실기), AgiBot G1(8캠+손목 F/T), Figure 02/03(촉각 ~3g), Optimus·1X NEO(텐던). **액추에이터 비교**: 하모닉(무백래시·비역구동) vs 싸이클로이드(DYD/다이나믹셀 Y: 하모닉급 정밀+내충격, OMY 리그) vs QDD(역구동·충격내성·전류∝토크 고유수용) vs SEA vs 취미서보(온보드 위치PID만). **왜 RGB-only가 지배적인가**. 플랫폼↔데이터셋 지도 | 자기 로봇(또는 SO-101)을 같은 틀로 스펙 분석해서 표로 정리 |
| [26](upper-part6-real-robot-integration/lec26-action-pipeline.md) | Action의 여정: VLA에서 액추에이터까지 | **모델별 action space** (전부 1차 자료 검증됨): RT-2/OpenVLA=ΔEEF 7차원×256빈(q01-q99 분위수 정규화, ~6Hz), ACT=절대 관절각 50Hz 청크100, Diffusion Policy=EEF 위치(16예측/8실행), π0=최대 18차원 패딩 관절공간 H=50@50Hz(청크 생성 73ms), SmolVLA=관절각 청크50, GR00T N1=H16→**N1.7=상대 EEF(인간·로봇 공유) H40**, RDT-1B=128차원 통일 공간. **청크 실행**: temporal ensembling, receding horizon, **Real-Time Chunking**, LeRobot 비동기 추론. **제어 계층**: VLA 1~50Hz → 보간/IK/필터 100~1000Hz(Franka 토크지령+중력·마찰 자동보상, UR servoj 500Hz, DROID: 15Hz→Polymetis→1kHz) → 모터 전류루프. **배포 토폴로지**: 오프보드 GPU vs 온로봇 Jetson(TensorRT ~3.6배) | LeRobot async inference 코드 추적: 관측이 들어가 서보 명령이 나올 때까지 경로를 Claude와 함께 따라가기 |

## Part 7 — 생태계: 데이터, 도구, 시뮬레이터 (4강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 27 | 데이터셋과 수집 | OXE(1M 궤적/22 로봇), DROID, BridgeData V2, AgiBot World(공장식 수집), **LeRobotDataset v3 포맷**. 수집 도구: ALOHA, GELLO, UMI, **SO-101(~30만원, 취미가 표준)** | HF Hub에서 데이터셋 로드 + 시각화 |
| 28 | LeRobot 딥다이브 | 라이브러리 구조, 탑재 정책(ACT/DP/π0/π0.5/SmolVLA/GR00T N1.7...), async inference. **이 과정의 메인 실습 도구** | **SmolVLA를 LIBERO 태스크에 파인튜닝** (Colab 가능) — 이 과정의 최대 실습 |
| 29 ★ | 시뮬레이터 지형도 | MuJoCo/robosuite(입문용), ManiSkill3(GPU 병렬), Isaac Lab(RTX 필요, sim2real), Genesis, RoboCasa+MimicGen(데이터 증식). sim2real 갭 | ManiSkill3 또는 robosuite 환경 구동 |
| 30 | 벤치마크와 평가의 함정 | LIBERO(포화됨: 상위권 95%+), **LIBERO-Plus(교란 넣으면 95%→30% 붕괴)**, SimplerEnv(real-to-sim), CALVIN, RoboArena/RoboChallenge(분산 실기 평가). 평가 통계의 현실(N=10~20, CI 없음) | LIBERO에서 파인튜닝 모델 평가 |

## Part 8 — 프론티어와 자립 (3강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| 31 | 프론티어 지도 (2025-26) | ① RL post-training 물결(RECAP, SimpleVLA-RL, π_RL) ② latent action(LAPA, GO-1, UniVLA — 행동 라벨 없는 비디오 활용) ③ **world model 수렴(V-JEPA 2, Cosmos, DreamZero/GR00T N2, Tesla·1X world simulator: "VLA 다음은 WAM?")** ④ 효율화(양자화, RTC, on-device) ⑤ 개념 배경: SSL 계보와 JEPA·latent world model (원논문 기준) | 최신 서베이 "An Anatomy of VLA Models" 훑기 |
| 32 | 논문 읽기 프레임워크 | **6축 설계공간 지도**: 액션 디코딩 / 아키텍처 / 학습 레시피 / 데이터 / 평가 / 효율·배포. + 비판적 읽기 체크리스트 10항목(시행 횟수·CI, sim vs real, 베이스라인 공정성, 데모 영상 vs 수치...) + **층위 진단**(00강의 좌표계로 "아키텍처⊥학습목적" 직교 분해 — "VLA vs RL" 류 가짜 논쟁 판별) | 체크리스트로 π0.7 논문 분석 |
| 33 ★ | 캡스톤 | 최근 1달 내 VLA 논문 2편을 골라 프레임워크로 완전 분석 → "새로운 점"만 요약. 정보 파이프라인 구축(HF Daily Papers, alphaXiv, LeRobot Discord, 학회 워크숍) | 논문 분석 리포트 2편 작성 |

---

# 하위제어 트랙 — 고전 로봇공학 (Part R1~R6, 30강)

> 대상: 딥러닝은 알지만 로봇은 모르는 팀원. 번역 박스가 반대 방향이다 — 로봇 개념을 **딥러닝 개념으로 비유**한다
> (예: 자코비안 ↔ 신경망의 야코비안/backprop, MPC ↔ planning이 있는 model-based RL, 임피던스 성형 ↔ 손실함수 성형).
> 실습 기본 도구: NumPy(손계산 검증), MuJoCo(물리), Pinocchio(기구학·동역학 라이브러리). 실기가 있으면 SO-101/OMY로 확장.
> **기초 참고서: Lynch & Park, Modern Robotics** (무료 PDF: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf) — 역학적 내용(기구학·동역학·제어·파지)의 표준 참고서로 사용. 단 **Ch.9(궤적 생성)·Ch.10(모션 플래닝)은 참고 범위에서 제외** — 그 층은 이 커리큘럼에서 학습 정책(VLA)이 맡는 것으로 보고 상위 트랙에서 다룬다. 강의별 장 매핑은 파트 표 아래 참조.

## Part R1 — 로봇이라는 기계 (4강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R01](lower-part1-robot-body/lecR01-robot-anatomy.md) | 로봇 해부학: 링크, 관절, 자유도 | 직렬/병렬 구조, DoF 세는 법(Grübler), 여유자유도, 작업공간 vs 관절공간, 그리퍼·핸드의 DoF 경제학. 상위 25강의 로봇들을 해부학으로 재방문 | MuJoCo에서 로봇 모델(URDF/MJCF) 열어 관절 트리 분석 |
| [R02](lower-part1-robot-body/lecR02-rotation-so3.md) | 회전의 수학: SO(3) | 회전행렬의 성질, 오일러각과 짐벌락, 축-각(Rodrigues), 쿼터니언 — 각 표현의 장단과 상호 변환. **비유: 임베딩 공간의 다양체 — 회전은 벡터공간이 아니다** | 표현 4종 상호 변환 라이브러리를 NumPy로 직접 구현 + scipy 검증 |
| [R03](lower-part1-robot-body/lecR03-se3-transforms.md) | 강체 변환: SE(3)와 좌표계 | 동차변환, 좌표계 합성 규칙, twist 맛보기, DH 규약 vs 최신 규약(URDF) | 카메라-로봇-물체 좌표계 체인 계산 (hand-eye 문제 맛보기) |
| [R04](lower-part1-robot-body/lecR04-forward-kinematics.md) | 정기구학 (FK) | 링크 변환의 곱으로서의 FK. **PoE(지수곱)를 본류로**(MR 방식), DH 규약은 산업 문서 해독용으로 대조. 실로봇(UR5e) 예제 | UR5e FK를 NumPy로 구현하고 Pinocchio와 대조 |

## Part R2 — 미분 기구학과 궤적 (4강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R05](lower-part2-differential-kinematics/lecR05-jacobian.md) | 자코비안 | 관절속도 → EEF 속도의 선형 사상, 기하/해석 자코비안, **정역학 쌍대 τ=JᵀF**. **비유: 신경망 야코비안, backprop의 chain rule과 동형** | 자코비안을 해석식+유한차분 양쪽으로 구해 대조 |
| [R06](lower-part2-differential-kinematics/lecR06-singularity-manipulability.md) | 특이점과 조작성 | 특이점에서 무슨 일이 벌어지나(SVD로 보기), 조작성 타원체, 여유자유도와 null-space | 2R/3R 암의 조작성 지도 그리기 (히트맵) |
| [R07](lower-part2-differential-kinematics/lecR07-inverse-kinematics.md) | 역기구학 (IK) | 해석해(2R 예제), 수치해(뉴턴법, damped least squares), 다해성·특이점 근처의 거동, null-space 과제 | DLS IK를 직접 구현해 특이점 통과 실험 |
| [R08](lower-part2-differential-kinematics/lecR08-interpolation-timing.md) | 보간과 시간 파라미터화 | **학습 스택 아래에서 살아남는 최소 궤적론만**: 다항식/사다리꼴 프로파일, 관절/작업공간 보간, 저크 제한 — VLA 액션 청크를 셋포인트로 펴는 데 필요한 만큼 (상위 26강의 "보간 계층"). 고전 모션 플래닝(RRT 등)과 본격 궤적 계획은 의도적으로 다루지 않음 — 그 층은 학습 정책이 대체 | 최소 저크 보간기 구현 + 속도/가속 프로파일 플롯 |

## Part R3 — 동역학 (5강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R09](lower-part3-dynamics/lecR09-dynamics-ingredients.md) | 동역학의 재료 | 질량·관성텐서·운동량, 무게중심, 평행축 정리. 직관 예제(Dzhanibekov 효과) | 링크 관성 파라미터를 CAD/URDF에서 읽고 검증 |
| [R10](lower-part3-dynamics/lecR10-lagrangian-dynamics.md) | 라그랑주 동역학 | 에너지 → 운동방정식, **매니퓰레이터 방정식 M(q)q̈+C(q,q̇)q̇+g(q)=τ**, M·C·g 각 항의 물리, 수동성(passivity). 2링크 암 완전 유도 | 2링크 M, C, g를 sympy로 유도 + 수치 시뮬레이션 |
| [R11](lower-part3-dynamics/lecR11-newton-euler.md) | 뉴턴-오일러와 계산 동역학 | 재귀 뉴턴-오일러(RNEA), 순동역학(ABA), 역동역학 — 라이브러리가 실제로 쓰는 알고리즘. **비유: forward/backward pass의 쌍** | Pinocchio로 RNEA/ABA 호출, 계산 시간 측정, MuJoCo와 대조 |
| [R12](lower-part3-dynamics/lecR12-contact-friction-grasping.md) | 접촉, 마찰, 파지 | 쿨롱 마찰 원뿔, 접촉의 불연속성(왜 시뮬이 어려운가), form/force closure, 파지 행렬. **상위 트랙의 "접촉 태스크가 어려운 이유"의 물리적 근거** | MuJoCo 접촉 파라미터 스윕 — 같은 파지가 성공/실패하는 경계 찾기 |
| [R13](lower-part3-dynamics/lecR13-underactuation-locomotion.md) | 부족구동과 보행의 역학 | 부족구동 정의, ZMP, 선형 도립진자(LIP), capture point, 수동보행의 극한 사이클. **비유: 안정 어트랙터 학습** | LIP 모델로 보행 패턴 생성 시뮬레이션 |

## Part R4 — 액추에이터와 전동 (3강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R14](lower-part4-actuators/lecR14-motors-current-loop.md) | 전기 모터와 전류 루프 | BLDC/PMSM 동작 원리, 토크 상수 Kt, 전류≈토크, FOC 전류 제어(수 kHz~수십 kHz), 열 한계. **상위 26강 최하층의 정체** | 모터 파라미터로 토크-속도 곡선 그리기; 전류 스텝 응답 시뮬 |
| [R15](lower-part4-actuators/lecR15-gears-transmissions.md) | 감속기와 전동 (상위 25강과 합동) | 하모닉/유성/**싸이클로이드(DYD)**/벨트/텐던 — 감속비·백래시·강성·역구동성·충격내성 트레이드오프, 반사 관성 (n²Jm) | 감속비에 따른 반사 관성·가속 성능 계산 실습 |
| [R16](lower-part4-actuators/lecR16-qdd-integrated-actuators.md) | 통합 액추에이터와 QDD 철학 | MIT Cheetah 계보의 proprioceptive actuation: 저감속+대구경 모터 → 전류로 토크 추정 → 센서리스 힘 제어. 관절 토크센서(Franka) 방식과의 비교. **왜 이 선택이 보행 RL을 가능하게 했나** | QDD vs 하모닉 관절의 충격 응답 시뮬 비교 |

## Part R5 — 제어 (8강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R17](lower-part5-control/lecR17-feedback-control-basics.md) | 피드백 제어 최소 코스 | 피드백의 원리, PID, 극점과 안정성, 대역폭·위상여유·지연의 대가 — **딥러닝 배경자를 위한 고전 제어 1일 압축**. **비유: 학습률과 안정성, 지연 = stale gradient** | 2차 플랜트에 PID 튜닝 실험 (시뮬) |
| [R18](lower-part5-control/lecR18-state-space-lqr-kalman.md) | 상태공간, LQR, 칼만 필터 | 상태공간 모델, 가제어/가관측, LQR(2차 비용 최적 피드백), 칼만 필터(최적 추정), LQG. **비유: LQR ↔ 가치함수가 2차식인 RL의 해석해** | 도립진자 LQR + 칼만 필터 구현 |
| [R19](lower-part5-control/lecR19-joint-control-computed-torque.md) | 관절 제어와 computed torque | 독립 관절 PID의 한계 → 역동역학(computed torque) 제어: 비선형을 지우고 선형 오차 동역학 만들기, 중력 보상 | 2링크에 PID vs computed torque 추종 오차 비교 |
| [R20](lower-part5-control/lecR20-operational-space-control.md) | 작업공간 제어 | operational space 정식화, 태스크 공간 관성, 자코비안 전치 vs 역행렬 제어 | EEF 원 그리기: 관절 PID vs 작업공간 제어 비교 |
| [R21](lower-part5-control/lecR21-impedance-compliance.md) | 임피던스·컴플라이언스 제어 | **접촉을 다루는 법**: 기계 임피던스(M-B-K) 성형, 임피던스 vs 어드미턴스(어느 쪽이 무엇을 측정하고 무엇을 내나), 수동 컴플라이언스(RCC)와 능동 컴플라이언스, Franka 내장 임피던스의 실체. **비유: 접촉 응답의 "손실함수 성형"** | 1-DoF 접촉 시뮬로 임피던스 파라미터 스윕; 벽에 부딪히는 실험 |
| [R22](lower-part5-control/lecR22-force-hybrid-control.md) | 힘 제어와 하이브리드 제어 | 직접 힘 제어, 힘/위치 하이브리드(방향 분해), F/T 센서 vs 전류 기반 추정. 삽입·연마·조립 태스크의 제어 설계 | peg-in-hole 시뮬: 위치 제어 vs 하이브리드 비교 |
| [R23](lower-part5-control/lecR23-mpc.md) | MPC | 유한 구간 최적화 + receding horizon, 제약 처리(QP), 볼록 MPC로 보행(질량중심 동역학), 계산 예산과 주기. **비유: planning 있는 model-based RL — 그리고 Diffusion Policy의 receding horizon과 같은 구도(상위 15강)** | cvxpy로 카트폴 MPC 구현; 제약 유무 비교 |
| [R24](lower-part5-control/lecR24-whole-body-control.md) | 전신 제어 (WBC) | 태스크 우선순위(null-space 위계), QP 기반 WBC, 접촉 제약 하의 토크 분배 — 휴머노이드 하위 스택의 표준. **상위 24강 Helix 02의 S0·Atlas LBM 아래층이 바로 이것** | 간이 WBC: 2태스크 우선순위 제어 시뮬 |

## Part R6 — 시스템 통합 (6강)

| # | 제목 | 핵심 내용 | 실습 |
|---|---|---|---|
| [R25](lower-part6-system-integration/lecR25-robot-software-stack.md) | 로봇 소프트웨어 스택 | ROS 2 개념(노드/토픽/서비스), 실시간성(RT 커널, 주기 보장), 필드버스(EtherCAT/CAN), ros2_control 구조 | ROS 2로 시뮬 로봇에 관절 명령 퍼블리시 |
| [R26](lower-part6-system-integration/lecR26-simulation-internals.md) | 시뮬레이션의 내부 | 물리엔진이 하는 일: 적분기(explicit/implicit), 접촉 솔버(LCP/soft), 타임스텝과 안정성, MuJoCo vs Isaac의 설계 차이. **sim2real 갭의 물리적 원인 목록** | 타임스텝·솔버 파라미터로 같은 장면이 달라지는 실험 |
| [R27](lower-part6-system-integration/lecR27-state-estimation.md) | 상태 추정과 센서 융합 | 엔코더/IMU/F-T의 특성, 상보 필터→EKF, 관절 토크로 외력 추정(모멘텀 옵저버), 오도메트리 드리프트 | IMU+엔코더 융합으로 자세 추정 구현 |
| [R28](lower-part6-system-integration/lecR28-system-identification-calibration.md) | 시스템 식별과 캘리브레이션 | 관성 파라미터 동정, 마찰 모델 피팅, 기구학 캘리브레이션, 카메라-로봇(hand-eye) 캘리브레이션. **비유: 이것이 고전판 "학습"이다 — 무엇이 다른가** | 시뮬 데이터로 2링크 관성 파라미터 회귀 |
| [R29](lower-part6-system-integration/lecR29-safety-layers.md) | 안전 계층 | 속도/힘/전력 제한, 협동로봇 규격(ISO 10218/TS 15066 개념), e-stop과 안전 등급, 학습 정책 아래의 안전 필터(CBF 맛보기) | 시뮬에서 안전 필터(속도 제한 투영) 끼워 넣기 |
| [R30](lower-part6-system-integration/lecR30-two-stacks-meet.md) | 종합: 두 스택의 접점 (상위 26강과 합동) | VLA 액션 청크가 이 트랙에서 배운 스택(보간→제어기→전류루프)으로 흘러 들어오는 전체 경로 재구성. 임피던스 아래층이 학습 정책에 주는 가치, 안전·주기·지연 예산의 합동 설계 | 상위 26강 실습(LeRobot async 추적)을 하위제어 관점에서 재수행 + 통합 다이어그램 완성 |

### 하위 트랙 ↔ Modern Robotics 장 매핑

기초 참고서 MR(위 PDF)의 활용 범위. 강의 생성 시 해당 장의 정의·표기·연습문제를 기준으로 삼고, 인용은 참고문헌 규약대로.

| MR 장 | 대응 강의 | 비고 |
|---|---|---|
| Ch.2 Configuration Space | R01 | DoF·Grübler·C-space |
| Ch.3 Rigid-Body Motions | R02·R03 | SO(3)/SE(3), 지수좌표, twist |
| Ch.4 Forward Kinematics | R04 | PoE 본류 (DH는 부록 C) |
| Ch.5 Velocity Kinematics & Statics | R05·R06 | 자코비안, τ=JᵀF, 특이점·조작성 |
| Ch.6 Inverse Kinematics | R07 | 뉴턴법·수치 IK |
| Ch.8 Dynamics of Open Chains | R09~R11 | 라그랑주·뉴턴-오일러·M/C/g |
| Ch.11 Robot Control | R17(부분)·R19~R22 | 모션/힘/하이브리드/임피던스 |
| Ch.12 Grasping & Manipulation | R12 | 마찰 원뿔, form/force closure |
| **Ch.9 궤적 생성 · Ch.10 모션 플래닝** | **참고 범위 제외** | 학습 정책(VLA)이 대체하는 층 — R08은 최소 보간만 별도 자료로 |
| Ch.7 폐체인 · Ch.13 모바일 로봇 | 선택 | 필요할 때만 |
| MR 범위 밖 | R13(보행 — Tedrake·LIP 문헌), R14~R16(액추에이터), R18(LQR/칼만), R23(MPC), R24(WBC), Part R6 | 강의별 참고문헌에서 확정 |

보조 자료: Russ Tedrake "Robotic Manipulation"(manipulation.csail.mit.edu — 무료, Drake 실습), Craig "Introduction to Robotics"(DH 관점 대조), MuJoCo/Pinocchio 공식 문서.

---

## 부록 A — 모델 계보 치트시트 (2026-07 기준)

```
2022.12  RT-1        스케일된 모방학습 실증 (35M, 행동=이산 토큰)
2023.07  RT-2        "VLA" 명명. 웹 사전학습 VLM에 행동 토큰 출력 (12B/55B, 비공개)
2023.10  OXE/RT-X    데이터 전환점: 1M 궤적, 22 로봇, cross-embodiment
2024.05  Octo        첫 완전 오픈 generalist (디퓨전 헤드, VLM 아님)
2024.06  OpenVLA     RT-2 레시피의 오픈 재현 (7B > 55B RT-2-X)
2024.10  π0          VLM + flow matching action expert 템플릿 확립 (50Hz)
2024.10  RDT-1B      순수 디퓨전 트랜스포머 경로 / LAPA: latent action
2024.11  CogACT      인지(VLM)와 행동(DiT) 분리
2025.01  π0-FAST     DCT 기반 행동 토크나이저 (이산 진영의 반격)
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

**교과서 격 (상위 트랙)**
- Robot Learning: A Tutorial (HF, 2025.10): https://huggingface.co/spaces/lerobot/robot-learning-tutorial
- HF Robotics Course: https://huggingface.co/learn/robotics-course
- An Anatomy of VLA Models (살아있는 서베이): https://suyuz1.github.io/VLA-Survey-Anatomy/
- MIT 6.S184 Flow Matching & Diffusion: https://diffusion.csail.mit.edu/

**교과서 격 (하위 트랙)**
- Lynch & Park, Modern Robotics — **하위 트랙 기초 참고서**: PDF https://hades.mech.northwestern.edu/images/7/7f/MR.pdf | 사이트(영상·연습문제) http://modernrobotics.org — 활용 범위는 역학·제어·파지 장. **Ch.9-10(궤적·플래닝)은 제외** (본문 매핑표 참조)
- Russ Tedrake, Robotic Manipulation (무료, Drake 실습): https://manipulation.csail.mit.edu/
- MuJoCo 문서: https://mujoco.readthedocs.io | Pinocchio: https://github.com/stack-of-tasks/pinocchio

**기초 다지기**
- Karpathy Zero to Hero: https://karpathy.ai/zero-to-hero.html
- 3Blue1Brown Neural Networks: https://www.3blue1brown.com/topics/neural-networks
- HF LLM Course: https://huggingface.co/learn/llm-course
- HF CV Course: https://huggingface.co/learn/computer-vision-course

**모델 공식 소스**
- openpi (π0/π0.5): https://github.com/Physical-Intelligence/openpi | PI 블로그: https://www.pi.website/blog
- Isaac GR00T: https://github.com/NVIDIA/Isaac-GR00T | GEAR lab: https://research.nvidia.com/labs/gear/
- LeRobot: https://github.com/huggingface/lerobot | SmolVLA: https://huggingface.co/blog/smolvla

**비공개 진영 1차 자료 (24강)**
- Figure: Helix https://www.figure.ai/news/helix | Helix 02 https://www.figure.ai/news/helix-02 | Go-Big https://www.figure.ai/news/project-go-big
- Skild: https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain | LocoFormer(방법 유추용): arXiv 2509.23745
- 1X: Redwood https://www.1x.tech/discover/redwood-ai (world model 포함)
- TRI LBM: https://toyotaresearchinstitute.github.io/lbm1/ | Atlas LBM: https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
- Gemini Robotics 1.5: arXiv 2510.03342 | Agility WBC 모델: https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model
- Tesla는 공식 기술 문서 없음 — Elluswamy ICCV 2025 / CVPR 2026 강연 녹화가 최선

**하드웨어·제어 스택 (25-26강, R트랙)**
- Franka FCI/libfranka 문서: https://frankarobotics.github.io/docs/ (1kHz, 토크지령 중력·마찰 자동보상)
- UR RTDE 가이드: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html (e-Series 500Hz)
- SO-101 문서: https://huggingface.co/docs/lerobot/so101 | LeRobot async inference: https://huggingface.co/docs/lerobot/en/async
- Real-Time Chunking: https://www.pi.website/research/real_time_chunking | DROID 스택(Polymetis): arXiv 2403.12945
- Unitree G1: https://www.unitree.com/g1 | Fourier GR-1: http://support.fftai.cn/main/en/concepts/about_gr1/
- ROBOTIS DYD/Dynamixel-Y/OMY: https://emanual.robotis.com/docs/en/all-dyd/ | https://ai.robotis.com/omy/hardware_omy

**정보 파이프라인 (32-33강에서 구축)**
- HF Daily Papers: https://huggingface.co/papers | alphaXiv: https://www.alphaxiv.org/
- arXiv cs.RO RSS: https://rss.arxiv.org/rss/cs.RO
- Humanoids Daily: https://www.humanoidsdaily.com/ | Import AI: https://importai.substack.com/
- 학회: CoRL(11월) > RSS > ICRA/IROS. Papers with Code는 2025.7 폐쇄됨 — HF Papers로 대체.

## 부록 C — 실습 환경 요구사항

**상위 트랙**
- **Colab 무료/T4로 충분**: Part 1-3 전부, ACT/Diffusion Policy 훈련, SmolVLA 추론, LIBERO, SimplerEnv
- **Colab A100 또는 로컬 24GB**: SmolVLA 파인튜닝(~4h/20k steps), OpenVLA LoRA
- **로컬 8GB+**: π0 추론 (openpi)

**하위 트랙**
- **CPU만으로 충분**: NumPy/sympy 실습 전부, Pinocchio, MuJoCo(CPU), cvxpy MPC
- **선택**: ROS 2 환경(도커 권장), Isaac Lab(RTX)

**선택 하드웨어(공통)**: SO-101 로봇팔(~$220 키트) 또는 Robotis OMY — 실물 teleop→수집→훈련 루프와 하위 트랙 실기를 원할 때

## 부록 D — CS285 → VLA 심화 트랙 (선택)

CS285(버클리, Sergey Levine)는 2026 봄학기에 "CS 185/285"로 개편됨(BC 2개 강의로 분리, **"LLM RL" 강의 신설**, Inverse RL·Meta-RL 삭제). **공개 영상은 여전히 Fall 2023 녹화본**이므로 아래 번호는 Fall 2023 기준.

| CS285 강의 | 내용 | 풀리는 VLA 개념 | 우선순위 |
|---|---|---|---|
| L2 (+HW1) | 모방학습, DAgger | 본편 13강 | 필수 (본편) |
| L4 | MDP, RL 표기법 | 모든 RL-VLA 논문의 언어 | 높음 |
| L5 (+HW2) | Policy gradient, baseline | VLA RL 파인튜닝(PPO/GRPO), BC=log-likelihood의 관계 | 높음 |
| L6 | Actor-critic, **advantage**, GAE | **RECAP의 advantage conditioning** | 높음 |
| L15-16 | **오프라인 RL** (CQL, IQL, AWR) | RECAP = 오프라인 RL 정책 추출을 VLA에 적용한 것. **VLA 시대 최고 가치의 2강** | 높음 |
| L9 | Importance sampling, TRPO→PPO, KL 제약 | RL 파인튜닝이 base 정책 근처에 머무는 이유 | 중간 |
| L7(-8) | 가치함수, Q러닝 | RECAP의 critic 절반, HIL-SERL | 중간 (L8은 HW3 할 때만) |
| L11(-12) | 모델 기반 RL | **world model**(Dreamer, WAM) 이해의 기반 | 중간 |
| L21 | 시퀀스 모델 RL | VLA = VLM 정책이므로 직결; 2026 "LLM RL" 강의의 조상 | 높음 (짧음) |
| L20 | Inverse RL | 보상 모델, VLM-as-reward | 낮음 |
| L18-19 | VI/생성모델, control as inference | 디퓨전/FM 헤드를 유도까지 하고 싶을 때만; AWR의 exp(A/β) 출처 | 선택 |

**건너뛰어도 되는 것**: L3(PyTorch — 본편 3강이 대체), **L10(LQR/MPC — 하위 트랙 R18·R23이 대체)**, L13-14(exploration), L17(이론), L22(meta-RL — 필드에서도 퇴조), L23.
**과제**: HW1(필수급) → HW2 → (여유 시) 2026 HW5 오프라인 RL, HW4 LLM RL. HW3(Atari DQN)은 스킵.
**Levine의 VLA 시대 자료**: Dwarkesh 팟캐스트 2025.9 (~88분, 최고의 세계관 개관), Substack "Sporks of AGI"(2025.7 — 왜 실기 데이터인가), "The Promise of Generalist Robotic Policies"(2024.10).

## 부록 E — 설계 근거: 세 종류 좌표계와 커리큘럼 원칙

이 커리큘럼이 왜 지금의 형태(공용 00강의 세 종류 좌표계, 두 트랙, v3 강의 템플릿)를 갖게 됐는지의 설계 원칙을 정리한다.

**핵심 설계 판단 — "계층"을 종류로 나눈 이유.** Physical AI를 흔히 "지각→…→플랜트"의 단일 계층 스택으로 그린다. 그러나 그 요소들은 레벨(종류)이 다르다: 어떤 것은 **연산 블록(코드)**, 어떤 것은 **인터페이스(계약)**, 어떤 것은 **물리 기질(코드 아님)**이고, 태스크·목표는 아예 **입력(명세)**이다. 이들을 한 줄에 번호로 세우면 "VLA vs RL"(아키텍처 vs 학습 목적), "controller를 없앴다"(물리·계약은 제거 불가) 같은 **범주 오류**를 부른다. 그래서 00강은 단일 스택 대신 **세 종류 + 입력**으로 좌표계를 세우고, 각 구성요소를 어느 강의가 파고드는지까지 매핑한다(00강 §1 표).

| 설계 원칙 | 커리큘럼에서의 구현 |
|---|---|
| 종류(코드/계약/물리)를 구분하는 좌표계로 개념 혼동을 제거 | **공용 00강** — 세 종류 좌표계 도입, 상위 32강의 "층위 진단"으로 연결 |
| "로봇 배경 ↔ AI 배경" 양방향 독자 경로 | **두 트랙**: 상위(AI·VLA) + 하위제어(R01~R30, 딥러닝 배경자용 고전 로봇 30강) |
| 개념마다 문제의식 → 직관 → 수식 3단(직관/의미/형식) → worked example(손계산+실행 코드) → 흔한 오해 | **강의 템플릿 v3** — 위 섹션들을 필수화, 인용 수치는 코드 실행으로 재현 |
| CV 지형(detection/SAM/affordance)을 VLA 논문에 필요한 만큼 조감 | 상위 4강 범위 확장 |
| SSL 계보(contrastive→JEPA→world model)의 체계적 취급 | 상위 11강 SSL 조감 + 31강 원논문 심화 |
| "아키텍처 ⊥ 학습 목적 ⊥ 행동 표현"의 직교 분해를 논문 읽기 도구로 | 상위 32강 프레임워크의 "층위 진단" |

이 커리큘럼만의 차별점(다른 교과서·서베이와 구분되는 축): 모델 계보의 역사(상위 Part 5), 생태계·벤치마크(Part 7), 하드웨어 스펙(25강), 비공개 기업 분석(24강), 실배포 수치(26강), 그리고 전 강의의 실행 코드·재현 그림.
