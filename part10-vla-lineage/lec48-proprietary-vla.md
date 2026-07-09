# Lec 48. 비공개 진영의 지향점 — 회사별 베팅으로 읽는 VLA 설계 철학

> Part 5 마지막 강의. 선수 지식: 42~47강 (특히 46강 G0강T의 dual-system, 45강 RECAP).
> 정보 기준일: 2026-07-08. 비공개 기업 정보는 유통기한이 짧다 — 학습 시점에 Claude에게 최신 소식 확인을 요청할 것.

## 한 장 요약

상용 로봇 AI 기업들을 두 축 위에 놓으면 각자의 "베팅"이 보인다.
가로축은 웹 지식(VLM 백본) 의존도, 세로축은 계층 분리 정도다. 참고용으로 오픈 진영의 π0.5와 G0강T도 함께 찍었다.

```mermaid
quadrantChart
    x-axis VLM 의존 낮음 --> VLM 의존 높음
    y-axis 단일 네트워크 --> 계층 분리
    Agility: [0.10, 0.95]
    Figure Helix 02: [0.75, 0.90]
    Gemini Robotics 1.5: [0.90, 0.85]
    BD+TRI Atlas LBM: [0.35, 0.75]
    G0강T N1 참고: [0.70, 0.70]
    Skild Brain: [0.15, 0.60]
    pi0.5 참고: [0.80, 0.45]
    1X Redwood: [0.45, 0.25]
    Tesla Optimus: [0.30, 0.15]
```

## 학습 목표

1. 주요 상용 기업 7곳(Figure, Skild, Tesla, 1X, BD+TRI, DeepMind, Agility)이 각각 어떤 모델 형태에 베팅하는지 한 문장씩으로 설명할 수 있다.
2. 상용 배포 제약(신뢰성, 온보드 추론, 지연)이 연구용 VLA와 다른 설계를 낳는 이유를 설명할 수 있다.
3. 회사 공개 자료에서 공학적 신호(파라미터 수, 제어 주파수, 데이터 시간)와 마케팅 신호를 구분할 수 있다.
4. "계층화", "데이터 플라이휠", "world model", "sim-trained 저수준 제어기"라는 4개 공통 패턴을 각 회사 사례로 뒷받침할 수 있다.

## 본문

### 0. 왜 비공개 진영을 공부하는가

오픈 논문은 벤치마크(LIBERO 성공률)에 최적화되고, 회사는 **배포**(하루 8시간 무중단, 온보드 컴퓨트, 고객 앞 신뢰성)에 최적화된다. 같은 문제를 다른 제약으로 풀 때 설계가 어떻게 갈라지는지 보여주는 자연 실험이다. 또한 이들은 논문 대신 기술 블로그로 공개하므로, "부분 공개 자료에서 아키텍처를 재구성하는 훈련"이 논문 읽기와는 다른 근육을 만들어 준다.

각 회사에 자료 신뢰도 등급을 붙였다:
**[A] 상세 공학 공개** (수치·구조 명시) / **[B] 블로그+논문 유추** / **[C] 강연·마케팅 수준**.

### 1. Figure AI — 계층 VLA의 끝까지 가보기 **[A]**

비공개 진영에서 가장 상세한 축에 드는 공개. 진화 순서가 그대로 교재다.

- **Helix** (2025.2): S2 = 7B VLM(7~9Hz, 장면·언어 이해) → 잠재 벡터 → S1 = 80M 트랜스포머(200Hz, 상반신 35-DoF 연속 제어). ~500시간 teleop + 자동 언어 라벨링. 온보드 임베디드 GPU에서 구동. G0강T·π0와 같은 시기에 같은 결론(주파수 분리)에 도달했다는 점이 중요.
- **Helix 물류 업데이트** (2025.2): 암묵적 스테레오 비전(토큰 수 불변), 학습 기반 시각 고유수용감각(EEF 6D 포즈 자가 캘리브레이션), "스포츠 모드"(테스트타임 청크 보간으로 시연자보다 빠르게 실행). 데이터 큐레이션만으로 1/3 데이터로 40% 향상 — 데이터 품질 신호.
- **Helix 02** (2026.1): 3계층으로 확장. S2(추론) → S1(200Hz, 머리+손바닥 카메라·지문 촉각 ~3g·전신 고유수용감각 입력, 전신 관절 목표 출력) → **S0 = 10M 전신 제어기(1kHz), 순수 시뮬레이션 훈련**(20만 병렬 환경 + 1,000시간+ 리타게팅 인간 모션). 공식 표현: "손으로 짠 C++ 109,504줄을 대체". 61개 loco-manipulation 동작의 4분 자율 시연.
- **Project Go-Big** (2025.9): 1인칭 인간 비디오로 인터넷 규모 사전학습, 인간→로봇 zero-shot 내비게이션 전이. Brookfield(주거 10만+ 유닛) 데이터 파트너십.
- 하드웨어·양산: Figure 03(2025.10, 촉각 지문·손바닥 카메라·양산 설계), BotQ 자체 공장(연 12,000대 캐파 — 회사 발표 수치).

**베팅 요약**: 웹 지식은 크게(7B), 제어는 빠르게(1kHz), 그 사이를 잠재 벡터로 연결. 전 스택 수직 통합.

### 2. Skild AI — "VLM 이식"에 반대하는 omni-bodied RL **[B]**

- **Skild Brain** 블로그(2025.7)의 주장: 로봇공학에는 "수조 개" 예시가 필요한데 실기 데이터로는 불가능 → **대규모 시뮬레이션 + 인터넷 인간 비디오**(인간 신체 = 또 하나의 형태일 뿐)로 사전학습, 실기 데이터는 post-training에만. VLA 주류를 정면 비판: "VLM에 로봇 데이터 1% 미만을 접붙이는 방식은 물리적 접지가 부족하다."
- 구조는 2계층: 저주파 고수준 조작/내비게이션 정책 → 고주파 저수준 정책(관절각/토크 출력). 파라미터 수 비공개.
- "omni-bodied": 하나의 모델이 사족·휴머노이드·팔·모바일 매니퓰레이터를 제어하고, 다리 하나가 고장 나도 재학습 없이 적응한다고 주장(데모 기반, 일부는 마케팅으로 걸러 볼 것).
- 방법을 유추할 창은 창업자(Pathak/Gupta, CMU)의 논문: **LocoFormer**(2025.9, arXiv 2509.23745) — 절차 생성된 로봇 수만 종에 대규모 RL + 강한 도메인 랜덤화, 에피소드 경계를 넘는 긴 컨텍스트로 in-context 적응(자기 넘어짐에서 배움). RMA(rapid motor adaptation) 계보의 연장.
- 규모: 시리즈 C $1.4B, 기업가치 $14B+(2026.1). ABB·UR·Foxconn 파트너십.

**베팅 요약**: 45강 RECAP과 대비하라 — PI는 "실기 데이터 + RL로 신뢰성", Skild는 "심 스케일 + 적응력". 데이터 철학의 양극단.

### 3. Tesla Optimus — 단일 end-to-end + 신경 시뮬레이터 **[C]**

- 공식 기술 문서가 없다. 확인 가능한 것: FSD의 end-to-end 아키텍처·인프라를 공유("FSD의 발전이 Optimus로 그대로 이전"), 카메라 8대 비전+고유수용감각 → 모터 명령의 단일 네트워크, 인간 비디오 모방 + 심 RL.
- 가장 실질적인 공개는 **신경 world simulator**(Elluswamy, ICCV 2025): 행동 조건부 비디오 생성 모델을 FSD와 Optimus 양쪽의 훈련·평가에 사용. 63강(world model)의 예고편.
- 파라미터·제어 주파수 비공개. 양산 일정은 반복 연기 이력이 있어 회의적으로 볼 것.

**베팅 요약**: "차와 로봇은 같은 문제"라는 인프라 재활용 베팅. 검증 불가능한 부분이 가장 많은 회사.

### 4. 1X Technologies — 작은 온보드 VLA + world model 평가 **[A-]**

- **Redwood**(2025.6): **~160M 파라미터, ~5Hz**, NEO의 임베디드 GPU에서 완전 온보드 구동. 비전·언어 임베딩·고유수용감각 입력, **디퓨전 정책 디코딩**, 보행+조작 통합(기대기, 버티기 같은 전신 동작). 실패 에피소드에서도 학습. 음성 대화는 오프보드 LLM.
- **1X World Model**: 행동 조건부 비디오 생성 모델을 학습된 시뮬레이터/정책 평가기로 사용 — 실행 전에 "이 행동을 하면 무슨 일이 벌어지는가"를 예측.
- **NEO 제품 전략**(2025.10 출시, $20,000 또는 월 $499): 모르는 작업은 **Expert Mode** — 예약된 인간 VR teleop이 대신 수행하고 그 데이터가 훈련으로 들어간다. teleop 폴백을 숨기지 않고 데이터 플라이휠로 제품화한 사례. 자율성 수준은 마케팅-현실 논쟁의 최전선.

**베팅 요약**: π0(3.3B)의 1/20 크기로 온보드 우선. "모델은 작게, 데이터 루프는 크게".

### 5. Boston Dynamics + TRI — LBM: 고전 전신 제어 위의 디퓨전 정책 **[A]**

- **TRI LBM 논문**(2025.7, arXiv 2507.05331): 언어+비전+고유수용감각 조건부 디퓨전 트랜스포머, flow matching 목적함수, 16-스텝 액션 청크, ~1,700시간 로봇 데이터. 핵심 발견: **멀티태스크 사전학습이 새 태스크에서 3~5배 데이터 효율** + 통계적으로 엄밀한 평가 방법론(57강과 연결).
- **Atlas LBM**(2025.10): **450M DiT, 30Hz**, 스테레오 이미지+고유수용감각+언어 → 전신 ~50 DoF loco-manipulation(스텝, 웅크리기, 무게중심 이동 포함)을 하나의 정책으로. BD의 기존 MPC급 전신 제어 역량 위에 얹힌다는 점이 G0강T System1(관절 직접 출력)과의 차이.
- 2026: 양산형 전기 Atlas(CES 2026), 2026년 물량 전부 현대 RMAC + **Google DeepMind**에 배정. BD-DeepMind 파트너십으로 Atlas에 Gemini Robotics 탑재 발표 — TRI LBM과 DeepMind 스택이 한 하드웨어에서 공존하게 됨.

**베팅 요약**: "제어는 우리가 세계 최고니까, 학습은 그 위에". 고전 로봇공학 자산을 버리지 않는 노선.

### 6. Google DeepMind (+Apptronik) — 플래너-실행기 2모델 **[A]**

- **Gemini Robotics 1.5 + ER 1.5**(2025.9, arXiv 2510.03342): ER(embodied reasoning VLM — 계획, 공간 추론, 웹/도구 호출)이 오케스트레이터로서 VLA(GR 1.5)를 부린다. GR 1.5는 **행동 전에 자연어로 사고**(thinking-before-acting)하고, **Motion Transfer** — ALOHA·양팔 Franka·Apptronik Apollo를 아우르는 통일 모션 표현 — 로 embodiment 간 zero-shot 스킬 전이.
- Apptronik: 하드웨어 파트너. Apollo 2 + "Robot Park"(2026.6, 90,000 sq ft 데이터 수집 공장). 시리즈 A 누적 $935M.

**베팅 요약**: 계층화를 한 모델 안(π0.5)이 아니라 **두 모델 사이**에 둔 형태. 48강 지도에서 Figure와 같은 사분면, 다른 구현.

### 7. Agility Robotics — 가장 보수적인, 그래서 흥미로운 **[A-]**

- "전신 제어 파운데이션 모델" = **1M 파라미터 미만 LSTM**("motor cortex"). Isaac Sim에서 수십 년치 심 시간을 3~4일에 학습, zero-shot sim-to-real, 자유공간 포즈 목표로 프롬프트. 상위 계층(태스크 로직, WorkOS)은 여전히 엔지니어링.
- Digit은 GXO 물류센터에 RaaS로 실전 배치 중 — **수익을 내는 로봇 중 가장 "덜 학습된" 스택**이라는 역설.

**베팅 요약**: 룰베이스 스택에서 WBC 한 층만 RL로 교체. 고전 로봇공학자가 가장 공감할 노선이자, "전부 end-to-end"가 정말 필요한가라는 질문 그 자체.

### 8. 공통 패턴 4가지 (시험에 나올 부분)

1. **주파수 계층화가 상용의 대세**: Helix(7-9/200/1000Hz), G0강T(10/120Hz), Gemini(ER↔VLA), Skild(고/저주파 2계층). 단일 모델·단일 주파수는 연구 진영(OpenVLA)과 소형 온보드(Redwood)에서만.
2. **teleop 폴백의 제품화 = 데이터 플라이휠**: 1X Expert Mode, Figure Go-Big/Brookfield, Apptronik Robot Park. "자율성 실패"를 "데이터 수집"으로 재정의.
3. **world model의 침투**: Tesla 신경 시뮬레이터(훈련·평가), 1X WM(실행 전 평가), NVIDIA Cosmos/DreamZero(합성 데이터·zero-shot 정책). 63강의 주제가 이미 상용에 들어와 있다.
4. **저수준 제어기의 sim-trained 대체**: Helix 02의 S0, Agility의 LSTM — 고전 WBC가 담당하던 층이 RL로 교체되는 중. 단, 그 위 계층 설계는 회사마다 정반대.

### 로봇공학자를 위한 번역

계층형 VLA는 **cascade 제어의 학습판**이다. Helix 02의 S2/S1/S0는 태스크 플래너 / 궤적 생성기 / 전신 서보 제어기의 3-루프 구조와 1:1 대응하고, 루프 간 인터페이스가 "정의된 신호(궤적, 셋포인트)"에서 "학습된 잠재 벡터"로 바뀌었을 뿐이다. 바깥 루프가 느리고 안쪽 루프가 빠른 이유(대역폭 분리)도 동일하다. Agility는 아예 기존 cascade에서 최내곽 루프 하나만 신경망으로 바꾼 경우다. 반대로 Tesla/1X의 단일 네트워크는 "루프 분리 없이 하나의 고차 제어기"에 해당하며, 그래서 제어 주파수가 전체 시스템에서 가장 느린 요소(추론 지연)에 묶인다.

## 실습 (45분, GPU 불필요)

**7사 비교표 완성하기.** Claude와 함께 아래 표의 빈칸을 1차 자료(부록 B 링크)에서 찾아 채운다. 못 찾는 칸은 "비공개"로 표시 — 어느 칸이 비어 있는지 자체가 각 사의 공개 전략을 보여준다.

| 회사 | 백본/파라미터 | 계층 수 | 각 층 주파수 | 액션 디코딩 | 데이터 전략 | 신뢰도 |
|---|---|---|---|---|---|---|
| Figure | | | | | | |
| Skild | | | | | | |
| Tesla | | | | | | |
| 1X | | | | | | |
| BD+TRI | | | | | | |
| DeepMind | | | | | | |
| Agility | | | | | | |

완성 후: 표를 보고 "5년 뒤 어느 베팅이 이길 것 같은가"를 논거와 함께 Claude에게 설명하고 반박을 받아본다.

## Claude와 토론할 질문

1. Skild의 "VLM에 로봇 데이터 1% 접붙이기" 비판은 타당한가? π0.5의 co-training 결과(45강)는 이 비판에 대한 반박 근거가 되는가?
2. Helix 02의 S0(1kHz, sim-trained)가 대체한 것은 정확히 무엇인가? 고전 WBC(QP 기반 전신 제어)와 입출력·보장성 측면에서 뭐가 다른가?
3. 왜 상용 진영은 거의 전부 계층형인가? 연구 논문(OpenVLA 등)이 단일 모델을 선호하는 이유와 어떻게 갈리나?
4. 1X의 Expert Mode는 데이터 전략인가, 자율성 미달의 은폐인가? 두 해석 각각의 근거는?
5. BD+TRI처럼 "MPC 위에 정책"을 얹는 방식과 G0강T처럼 정책이 관절을 직접 내는 방식 — 접촉이 많은 작업에서 각각 어떻게 실패할까?
6. Agility의 <1M LSTM이 "파운데이션 모델"이라 불릴 자격이 있는가? 파운데이션 모델의 정의는 크기인가 역할인가?
7. 7개 회사 중 공개 자료만으로 재현 시도가 가능한 곳은 어디까지인가?

## 읽을거리

1. **Figure Helix 블로그 3부작** (helix → helix-logistics → helix-02, 각 10~15분): 비공개 진영에서 가장 논문에 가까운 공개. 전문을 읽을 것.
2. **Skild "Building the general-purpose robotic brain"** 블로그 (15분): 주장-근거 구조를 비판적으로 읽을 것. LocoFormer(arXiv 2509.23745)는 초록과 Fig 1만.
3. (선택) TRI LBM 프로젝트 페이지: 초록 + 평가 방법론 섹션만 — 57강에서 다시 만난다.

## 자가 점검

1. 7개 회사 각각의 베팅을 한 문장씩, 자료를 안 보고 말할 수 있는가?
2. Helix 02의 세 층(S2/S1/S0)의 파라미터 규모·주파수·훈련 방식을 말할 수 있는가?
3. "데이터 플라이휠" 패턴을 세 회사 사례로 설명할 수 있는가?
4. 계층형이 상용에서 지배적인 이유를 대역폭/지연 관점에서 설명할 수 있는가?
5. 회사 발표 자료에서 신뢰할 수치와 걸러야 할 주장을 구분하는 기준 세 가지를 말할 수 있는가?

## 참고문헌

> 본문 수치·주장의 출처. 웹 문서는 2026-07-08 접속 기준. (2차) = 언론 보도. 비공개 기업 특성상 회사 발표 수치가 다수 — 본문의 신뢰도 등급([A]/[B]/[C])과 함께 볼 것.

[1] Figure AI, "Helix: A Vision-Language-Action Model for Generalist Humanoid Control," 기술 블로그, 2025.2. https://www.figure.ai/news/helix
— **뒷받침**: S2 7B VLM 7~9Hz / S1 80M 200Hz, 상반신 35-DoF, ~500시간 teleop+자동 언어 라벨, 온보드 임베디드 GPU.

[2] Figure AI, "Helix in Logistics," 2025.2. https://www.figure.ai/news/helix-logistics
— **뒷받침**: 암묵적 스테레오(토큰 수 불변), 학습 기반 시각 고유수용감각, 스포츠 모드, 큐레이션 데이터 1/3로 40% 향상, 스테레오로 처리량 60% 증가.

[3] Figure AI, "Helix 02," 2026.1. https://www.figure.ai/news/helix-02
— **뒷받침**: 3계층(S2/S1 200Hz/S0 10M@1kHz), S0 순수 시뮬 훈련(20만 병렬 환경, 1,000시간+ 리타게팅 인간 모션), "C++ 109,504줄 대체", 61개 loco-manipulation 동작 4분 자율 시연, 지문 촉각 ~3g.

[4] Figure AI, "Project Go-Big," 2025.9. https://www.figure.ai/news/project-go-big — **뒷받침**: 인간 1인칭 영상 사전학습, 인간→로봇 zero-shot 내비게이션, Brookfield 파트너십.

[5] Figure AI, "Introducing Figure 03" (2025.10) · "BotQ". https://www.figure.ai/news/introducing-figure-03 · https://www.figure.ai/news/botq — **뒷받침**: 양산 설계 전환, 연 12,000대 캐파(회사 발표 수치).

[6] Skild AI, "Building the general-purpose robotic brain," 블로그, 2025.7. https://www.skild.ai/blogs/building-the-general-purpose-robotic-brain
— **뒷받침**: omni-bodied 논지, 2계층(저주파 고수준→고주파 관절/토크), 심+인간 영상 사전학습, "로봇 데이터 1% 미만 접붙이기" 비판, 파라미터 비공개.

[7] LocoFormer, arXiv:2509.23745, 2025.9. https://arxiv.org/abs/2509.23745 — **뒷받침**: 절차 생성 로봇 대규모 RL, 에피소드 경계 넘는 컨텍스트의 in-context 적응 (Skild 방법 유추의 창).

[8] (2차) The Robot Report, "Skild AI raises $1.4B...," 2026.1. https://www.therobotreport.com/skild-ai-raises-1-4b-building-omni-bodied-robot-skild-brain/ — **뒷받침**: 시리즈 C $1.4B, 기업가치 $14B+, 파트너십.

[9] (2차) Teslarati, Elluswamy 강연 보도. https://www.teslarati.com/tesla-vp-explains-why-end-to-end-ai-future-self-driving/ — **뒷받침**: FSD-Optimus 인프라 공유, 신경 world simulator(ICCV 2025). Tesla는 공식 기술 문서 부재 — 본문 [C] 등급의 근거.

[10] 1X Technologies, "Redwood AI," 2025.6. https://www.1x.tech/discover/redwood-ai — **뒷받침**: ~160M@~5Hz 완전 온보드, 디퓨전 정책 디코딩, 보행+조작 통합, 1X World Model(행동 조건부 영상 생성 평가기).

[11] 1X, NEO 제품 페이지 + (2차) Humanoids Daily 출시 보도, 2025.10. https://www.1x.tech/neo · https://www.humanoidsdaily.com/news/1x-details-neo-human-in-the-loop-strategy-and-hardware-as-pre-orders-go-live — **뒷받침**: $20,000/월 $499, Expert Mode(예약 VR teleop) 데이터 전략.

[12] Toyota Research Institute, "A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation," arXiv:2507.05331, 2025.7. https://arxiv.org/abs/2507.05331 · 프로젝트: https://toyotaresearchinstitute.github.io/lbm1/
— **뒷받침**: 언어 조건부 DiT+flow matching, 16스텝 청크, ~1,700시간, 멀티태스크 사전학습 3~5배 데이터 효율, 엄밀한 통계 평가.

[13] Boston Dynamics, "Large behavior models help Atlas find new footing," 블로그, 2025.10. https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/ — **뒷받침**: Atlas LBM 450M DiT@30Hz, ~50 DoF 전신 loco-manipulation, 스테레오+고유수용감각+언어 입력.

[14] Boston Dynamics, "Boston Dynamics & Google DeepMind form new AI partnership," 2026.1. https://bostondynamics.com/blog/boston-dynamics-google-deepmind-form-new-ai-partnership/ — **뒷받침**: Atlas에 Gemini Robotics 탑재, 2026년 물량 현대 RMAC+DeepMind 배정.

[15] Google DeepMind, "Gemini Robotics 1.5" 기술 보고서, arXiv:2510.03342, 2025.9. https://arxiv.org/abs/2510.03342 — **뒷받침**: ER 1.5 오케스트레이터 + GR 1.5 실행기, thinking-before-acting, Motion Transfer(ALOHA/양팔 Franka/Apollo).

[16] Apptronik 보도자료. https://apptronik.com/news-collection/apptronik-partners-with-google-deepmind-robotics · https://apptronik.com/news-collection/welcome-to-robot-park-where-apptroniks-apollo-goes-to-work — **뒷받침**: DeepMind 하드웨어 파트너, Robot Park(90,000 sq ft, 2026.6), 시리즈 A 누적 $935M.

[17] Agility Robotics, "Training a whole-body control foundation model," 기술 블로그. https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model — **뒷받침**: <1M LSTM "motor cortex", Isaac Sim 3~4일(수십 년치 심 시간), zero-shot sim2real, 상위 계층은 엔지니어링(WorkOS).
