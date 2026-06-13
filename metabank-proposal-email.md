# MetaBank 제안 메일 초안 (jihoonpa 님 사전 공유용)

> 발송 전 jihoonpa 님께 먼저 공유 → 검토 후 고객 발송.
> AI 티 제거, 비즈니스 톤, 분기로직 없음.

---

**제목:** MetaBank Qwen-Image-Edit / 이미지 편집 모델의 AWS Trainium·Inferentia 활용 방안 제안

지훈 님,

안녕하세요. MetaBank의 이미지 편집 모델을 AWS Trainium/Inferentia에서 활용하는 건에 대해, 사전 기술 검증을 마치고 제안 방향을 정리했습니다. 고객 발송 전에 먼저 공유드리니 검토 부탁드립니다.

## 배경

MetaBank가 관심 있는 **Qwen-Image-Edit**(Apache-2.0, 상업 활용 가능)을 Trainium/Inferentia에서 훈련·서빙할 수 있는지 실기(`trn1.32xlarge`) 검증했습니다. 결론부터 말씀드리면 Qwen-Image-Edit은 현재 Neuron에서 훈련·추론 모두 구조적 블로커가 있어, 이를 명확히 짚고 현실적인 대안 경로를 함께 제안드립니다.

## 1. Qwen-Image-Edit은 Trainium·Inferentia 모두 현재 큰 블로커가 있습니다 (훈련·추론 공통)

Qwen-Image-Edit 내부의 **Qwen2.5-VL 비전-언어 인코더가 동적 텐서 형상(dynamic tensor shape)** 을 사용합니다. Neuron 컴파일러(`neuronx-cc`)는 정적 형상을 전제로 하므로 이 연산을 지원하지 않아 컴파일이 차단됩니다.

중요한 점은, **이 블로커가 훈련뿐 아니라 추론에도 동일하게 적용된다**는 것입니다. 컴파일을 막는 것은 가중치 값이 아니라 모델의 연산 구조(인코더의 동적 형상)이므로, **GPU에서 훈련을 마친 모델이라도 그 가중치를 Neuron으로 컴파일하려 하면 동일하게 막힙니다.** 즉 아래 2번의 "GPU 훈련 → Neuron 서빙" 경로도 Qwen-Image-Edit에는 적용되지 않습니다. 이는 모델 아키텍처와 정적 컴파일러 간의 구조적 비호환이며(AWS 공개 이슈 aws-neuron-sdk #1144), 단순 설정 변경이나 학습 위치 변경으로는 해소되지 않습니다.

따라서 Qwen-Image-Edit을 Neuron(Trainium/Inferentia)에 올리려면, 인코더를 정적 형상으로 재구성하거나 Neuron SDK 차원의 지원이 선행되어야 합니다. 현 시점에서는 권장하지 않습니다.

## 2. 권장 경로 — GPU 훈련 자산을 Inferentia 서빙으로 연결 (정적 형상 모델 대상)

GPU(CUDA)로 훈련을 마친 모델을 **Neuron으로 컴파일하여 SageMaker Endpoint(Inferentia)로 서빙**하는 방안입니다. 추론은 상시·대량으로 비용이 집중되는 구간이라, 추론을 Inferentia로 옮기는 것만으로 비용 효율의 핵심을 확보할 수 있습니다. 훈련은 익숙한 GPU 생태계를 그대로 쓰고, 추론만 Neuron으로 전환하는 구성입니다.

**단, 이 경로는 정적 형상 구조의 모델에 한해 적용됩니다.** 위 1번에서 설명드린 대로 Qwen-Image-Edit은 동적 형상 인코더 때문에 GPU 훈련을 마쳤더라도 추론 컴파일 단계에서 막히므로, 이 경로의 대상이 아닙니다. 정적 형상 모델인 **FLUX.1-Kontext가 이 경로의 적합한 대상**이며(3번 참조), 실기에서 컴파일 성공을 확인했습니다.

## 3. 대안 경로 — 동급 디퓨전 모델 FLUX.1-Kontext 활용

Black Forest Labs의 **FLUX.1-Kontext**(Qwen-Image-Edit과 유사한 이미지 편집 디퓨전 모델)는 정적 형상 구조라 Neuron 친화적이며, 실기 검증에서 **컴파일에 성공**했습니다(`.neff` 산출 확인). 두 가지로 활용 가능합니다.

- **3-a. Trainium 훈련 + Inferentia 서빙** — 훈련 경로의 통합 블로커를 해소하면(현재 작업 진행 중) Trainium에서 훈련한 모델을 그대로 Inferentia로 서빙.
- **3-b. GPU 훈련 + Inferentia 서빙** — 2번 경로를 FLUX.1-Kontext에 적용. FLUX은 정적 형상이라 GPU 훈련 모델의 Neuron 추론 컴파일이 가능합니다(Qwen과 달리 막히지 않음).

즉 FLUX.1-Kontext는 Qwen-Image-Edit과 달리 2번·3번 경로가 모두 열려 있습니다. 다만 FLUX.1-Kontext [dev]는 비상업 라이선스이므로, 상업 배포 시 라이선스 검토가 선행되어야 합니다(연구·평가 목적은 무방).

## 4. 결론

- 고객이 `Qwen-Image-Edit` 으로 CUDA 훈련 PoC 를 마친바, 해당 모델에 lock-in 된 상황일 가능성이 높으므로, 추론 서빙 인프라 (Amazone SageMaker)를 NDS 가이드 받아 매끄럽게 구축하시면 좋겠고
- 여전히 Trainium 에 대한 수용성을 보이시는 경우 (비용 절감 요건 등), FLUX.1-Kontext 모델과 참고 애셋을 활용하여 NDS 인력이 Neuron 컴파일 및 Inferentia 서빙 서포트해 주시기 바랍니다.

## 참고 자산

- **`neuron-agentic-development`** — Neuron 포팅을 지원하는 에이전트·스킬 모음. 모델별 Neuron 호환성 사전 판별 및 포팅 자동화에 활용.
- **Neuron 포팅 간 주의사항 (PPTX)** — 고객 엔지니어가 Trainium 작업 중 자주 겪는 문제와, 본인이 그 상황에 처했는지 식별하는 신호, 권장 조치를 한 장표당 한 문제로 정리한 의사결정 가이드.

세부 검증 내용(블로커 분석, 컴파일 로그, 아키텍처 다이어그램)은 필요하시면 함께 전달드리겠습니다. 검토 후 코멘트 주시면 고객 발송본으로 다듬겠습니다.

감사합니다.
재빈 드림
