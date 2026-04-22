# 🧠 ResNet Transfer Learning & Fine-Tuning on CIFAR-10
### **ResNet50 vs ResNet101** — ImageNet 사전학습 모델을 CIFAR-10에 전이학습/파인튜닝하여 Skip Connection의 원리, Stage A/B 학습 전략, 모델 효율성을 비교 분석한 프로젝트



---

## 📌 프로젝트 요약 (Project Overview)

딥러닝에서 "깊으면 깊을수록 좋다"는 직관은 오랫동안 사실이 아니었습니다. 20층만 넘어가도 gradient vanishing으로 인해 얕은 네트워크보다 성능이 떨어지는 **Degradation Problem**이 발생했기 때문입니다. ResNet 은 이 문제를 **Skip Connection(잔차 연결)** 이라는 단순하지만 강력한 아이디어로 해결하여, 152층짜리 네트워크도 안정적으로 학습할 수 있게 만들었습니다.

---

## 🎯 핵심 목표 (Motivation)

1. **왜 ResNet인가?** — Skip Connection이 실제로 학습에 어떤 영향을 주는지
2. **어떻게 파인튜닝 하는가?** — Stage A/B 전략의 설계 근거와 하이퍼파라미터 선택 이유
3. **더 크면 더 좋은가?** — ResNet50 vs ResNet101, 파라미터 73% 증가 대비 실제 성능 이득

---

## 📂 프로젝트 구조 (Project Structure)
```
resnet-transfer-learning-cifar10/
│
├── resnet_transfer_learning_cifar10.py   # 메인 학습 스크립트 (Kaggle 실행용)
├── README.md
│
└── outputs/                              # 학습 후 자동 생성
    ├── fig_01_cifar10_samples.png        # 클래스별 샘플 시각화
    ├── fig_02_curves_ResNet50.png        # ResNet50 학습 곡선
    ├── fig_02_curves_ResNet101.png       # ResNet101 학습 곡선
    ├── fig_03_accuracy_comparison.png    # 4모델 정확도 비교 막대 그래프
    └── fig_04_efficiency_comparison.png  # 파라미터/정확도/속도 효율 비교
```

---

## 🏗️ Architecture

### ResNet 핵심 원리: Skip Connection

```
[ 기존 방식 ]           [ ResNet 방식 ]

  Input (x)               Input (x)
     ↓                      ↓    ↘ (shortcut)
  Conv layers            Conv layers
     ↓                      ↓      ↓
  H(x) = F(x)            F(x)  +  x
                             ↓
                          H(x) = F(x) + x
                             ↓
                           ReLU
```

Skip Connection의 핵심은 네트워크가 **전체 출력 H(x)를 직접 학습하는 대신, 입력 대비 잔차 F(x) = H(x) − x 만을 학습**하게 만드는 구조. 학습이 실패하더라도 F(x) → 0 이 되어 H(x) → x (identity mapping)가 보장되므로, 깊이가 늘어도 성능 저하가 발생하지 않음. 이것이 Gradient Vanishing 없이 152층까지 학습 가능한 핵심 이유...

### Bottleneck Block (ResNet50 / ResNet101 적용 구조)

| 단계 | 연산 | 채널 변화 | 목적 |
|:----:|------|:---------:|------|
| 1 | 1×1 Conv + BN + ReLU | 256 → 64 | 차원 압축 (계산량 감소) |
| 2 | 3×3 Conv + BN + ReLU | 64 → 64 | 핵심 특징 추출 |
| 3 | 1×1 Conv + BN | 64 → 256 | 차원 복원 |
| + | Skip Connection | 256 → 256 | 잔차 덧셈 |
| → | ReLU | — | 비선형 활성화 |

> Basic Block(3×3 → 3×3) 대비 계산량 약 50% 절감, 동일 출력 채널 유지.  
> 이 구조로 ResNet50 / 101 / 152처럼 50층 이상의 깊은 네트워크 실용적 구현 가능.

### 모델 비교: VGG16 vs ResNet50 vs ResNet101

| 항목 | VGG16 (2014) | ResNet50 (2015) | ResNet101 (2015) |
|------|:------------:|:---------------:|:----------------:|
| **레이어 수** | 16 | 50 | 101 |
| **파라미터 수** | 138M | 25.6M | 44.5M |
| **ImageNet Top-5 Error** | 7.3% | 5.25% | 4.60% |
| **계산량 (GFLOPs)** | 15.5 | 3.8 | 7.6 |
| **깊이 확장 가능 여부** | ✗ (16층 한계) | ✔ (Skip Connection) | ✔ (Skip Connection) |
| **블록 구조** | 3×3 Conv 반복 | Bottleneck + Skip | Bottleneck + Skip |

---

## 🎯 파인튜닝 전략 (Fine-Tuning Strategy: Stage A → Stage B)

### 전체 학습 흐름

```
[Stage A] Backbone 완전 동결 (trainable = False)
  → Head (GAP → Dense(256) → Dropout → Dense(10)) 단독 학습
  → LR = 1e-3  |  Epochs = 5 (EarlyStopping 적용)
  → 목표: ImageNet 특징 기반 헤드 초기 적응

          ↓  Stage A 완료

[Stage B] Backbone 상위 30개 레이어 해동 (Unfreeze)
  → 해동된 백본 상위 레이어 + 헤드 동시 학습
  → LR = 5e-6  |  clipnorm = 1.0  |  Epochs = 5 (EarlyStopping 적용)
  → 목표: CIFAR-10 특화 고수준 특징 미세 조정
```

### 하이퍼파라미터 설계 근거

| 파라미터 | Stage A 값 | Stage B 값 | 선택 이유 |
|----------|:----------:|:----------:|-----------|
| **Learning Rate** | 1e-3 | 5e-6 | 사전학습 가중치 파괴 방지 — 두 단계 간 200배 차이 |
| **Gradient Clipping** | 미적용 | clipnorm=1.0 | 깊은 경로 gradient explosion 억제 |
| **Unfreeze 레이어 수** | 0개 | 상위 30개 | 범용 특징(하위) 보존, 고수준 특징(상위) 재조정 |
| **EarlyStopping patience** | 3 | 3 | val_loss 기준 조기 종료, 과적합 방지 |
| **ReduceLROnPlateau factor** | 0.5 | 0.5 | 정체 감지 시 LR 자동 절감 |
| **Epochs** | 5 | 5 | EarlyStopping으로 실제 종료 시점 자동 결정 |

### 레이어별 Freeze 전략

| 레이어 위치 | 학습되는 특징 수준 | 전략 | 근거 |
|:-----------:|-------------------|:----:|------|
| 하위 레이어 | edges, corners, textures (저수준·범용) | **Freeze** | ImageNet 범용 특징 그대로 재활용 |
| 중간 레이어 | shapes, patterns (중간 수준) | **Freeze** | CIFAR-10으로도 충분히 전이 가능 |
| 상위 레이어 | 객체 부분, 고수준 의미 정보 | **Unfreeze** | 도메인 특화 재조정 필요 구간 |

> ResNet50  기준: 약 175개 레이어 중 30개 해동 → 상위 **17%** 조정  
> ResNet101 기준: 약 345개 레이어 중 30개 해동 → 상위 **9%** 조정

---

## 📈 학습 성과 및 지표 (Results)

### 성능 요약

| Model | Val Acc (best) | Test Acc | Test Loss | 비고 |
|-------|:--------------:|:--------:|:---------:|------|
| ResNet50  Stage A | ~0.8500 | ~0.8490 | ~0.4600 | 헤드 단독 학습 |
| ResNet50  Stage B | ~0.8700 | ~0.8680 | ~0.3900 | Stage A 대비 +1~2%p |
| ResNet101 Stage A | ~0.8490 | ~0.8480 | ~0.4650 | 헤드 단독 학습 |
| ResNet101 Stage B | ~0.8750 | ~0.8720 | ~0.3850 | Stage A 대비 +1~3%p |

> 실제 수치는 GPU 환경 및 랜덤 시드에 따라 소폭 차이 발생 가능.  
> 위 수치는 원본 노트북 기준 예상 범위.

### 시각화 결과물

#### 📷 fig_01 — CIFAR-10 클래스별 샘플

| 항목 | 내용 |
|------|------|
| **파일명** | `fig_01_cifar10_samples.png` |
| **구성** | 10개 클래스 × 5장 = 총 50장 그리드 |
| **확인 포인트** | 클래스 간 시각적 다양성, 32×32 저해상도 특성 |

#### 📉 fig_02 — 학습 곡선 (Stage A → B 통합)

| 항목 | 내용 |
|------|------|
| **파일명** | `fig_02_curves_ResNet50.png` / `fig_02_curves_ResNet101.png` |
| **구성** | Loss + Accuracy 동시 표시, Stage 경계 수직 점선 구분 |
| **확인 포인트** | 과적합 여부, Stage 전환 시 loss 정체 구간, train/val 곡선 간격 |

**구간별 관찰 포인트:**

| 구간 | 예상 패턴 | 해석 |
|------|-----------|------|
| Stage A 초반 | val_loss 빠른 감소 | 헤드의 급속 적응 |
| Stage A → B 전환 | loss 일시 정체 | LR 1e-3 → 5e-6 급감 영향 |
| Stage B 전체 | train/val 간격 좁음 | clipnorm + 낮은 LR의 안정화 효과 |

#### 📊 fig_03 — 모델별 Test Accuracy 비교

| 항목 | 내용 |
|------|------|
| **파일명** | `fig_03_accuracy_comparison.png` |
| **구성** | ResNet50/101 × Stage A/B = 4개 막대, 색상 구분 |
| **확인 포인트** | Stage B의 일관된 성능 향상, 두 모델 간 격차 |

#### 🔬 fig_04 — 파라미터 효율성 비교

| 항목 | 내용 |
|------|------|
| **파일명** | `fig_04_efficiency_comparison.png` |
| **구성** | 파라미터 수(M) / Test Accuracy(%) / 추론 속도(ms/img) 3축 가로 막대 |
| **확인 포인트** | ResNet50의 단위 파라미터 효율 우위 |

### 추론 속도 & 파라미터 효율성

| Model | Parameters | Test Acc (Stage B) | Acc / M params | Inference (ms/img) |
|-------|:----------:|:-----------------:|:--------------:|:-----------------:|
| ResNet50  | 25.6 M | ~86.8% | ~3.39 | 기준 (1×) |
| ResNet101 | 44.5 M | ~87.2% | ~1.96 | ~1.5~2× 증가 |

> 파라미터 **73% 증가** 대비 정확도 향상 **약 0.4~1%p** 수준.  
> 단위 파라미터당 정확도(Acc/M) — ResNet50이 약 **1.7배 더 효율적**.

---

## ✨ 주요 결과 및 분석 (Key Findings & Analysis)

### 1. Stage A → Stage B 전이 효과

Stage A만으로도 약 85%의 높은 정확도가 확보됩니다. 이는 ImageNet 사전학습 가중치의 범용성이 CIFAR-10처럼 이질적인 저해상도 데이터셋에서도 강력하게 작동한다는 것을 의미합니다. Stage B(파인튜닝)는 여기서 추가로 1~3%p를 끌어올리는데, 이 향상은 상위 레이어의 고수준 특징이 CIFAR-10의 이미지 패턴에 맞게 재조정된 결과입니다.

### 2. 왜 ResNet101이 항상 더 좋지 않은가?

CIFAR-10은 32×32 저해상도, 10개 클래스라는 **상대적으로 단순한 분류 과제**입니다. ResNet101의 추가 깊이는 ImageNet처럼 1,000개 클래스를 세밀하게 구분해야 하는 복잡한 문제에서 진가를 발휘합니다. 단순한 과제에 과도하게 깊은 모델을 적용하면 Over-parameterization 위험이 커지고, 파라미터 대비 성능 이득이 체감적으로 감소합니다.

### 3. 실무 적용 시 모델 선택 기준

| 시나리오 | 추천 모델 | 이유 |
|----------|:---------:|------|
| 모바일 / 엣지 디바이스 | ResNet50 | 속도 우선, 충분한 정확도 확보 |
| 실시간 추론 서비스 | ResNet50 | 낮은 지연시간 필수 환경 |
| 소규모 데이터셋 (수만 장 이하) | ResNet50 | 과적합 위험 최소화 |
| 의료 영상 / 정밀 분류 | ResNet101+ | 정확도 최우선 환경 |
| 대규모 데이터셋 (100만 장+) | ResNet101+ | 충분한 데이터로 깊이의 이점 극대화 |

---

## 💡 회고록 (Retrospective)
이 프로젝트는 단순히 프레임워크의 내장 모델을 불러와 fit()을 실행하는 수준을 넘어, 전이학습의 본질을 뜯어보고 통제하는 계기가 되었습니다. 미세 조정(Fine-tuning)을 두 단계(Stage A, Stage B)로 명확히 분리하여 설계해 보면서, 각 학습 단계의 목적에 따라 하이퍼파라미터와 네트워크 동결(Freezing) 전략이 완전히 달라져야 함을 실무적으로 체감했습니다.

가장 인상 깊었던 엔지니어링 포인트는 Learning Rate(LR)의 극단적인 단차였습니다. Classifier 헤드만 학습하는 Stage A에서는 1e-3의 다소 높은 LR을 적용했지만, Backbone의 일부를 개방한 Stage B에서는 5e-6으로 200배나 낮춰야 했습니다. 이는 "미세 조정을 할 때는 LR을 단순히 작게 설정해야 한다"는 피상적인 암기를 넘어, ImageNet 대규모 데이터로 이미 훌륭한 최적점(Global Minimum 근처)에 도달해 있는 가중치 공간을 무너뜨리지 않기 위한 필수적인 수학적 조치임을 이해하게 되었습니다.

또한, 데이터 분할 시 Test Set을 Early Stopping이나 LR Scheduler의 모니터링 지표로 사용할 경우 발생하는 'Data Leakage(데이터 누수)' 문제의 치명성을 깊이 고찰했습니다. 모델이 간접적으로 Test Set에 과적합되어 실제 환경(In-the-wild)에서의 일반화(Generalization) 성능을 심각하게 과대평가하게 되는 메커니즘을 방지하고자, Train Set 내부에서 10%의 Validation Set을 층화 추출(Stratified split)하여 엄격하게 분리했습니다. 이는 단순히 코드를 에러 없이 작동시키는 것을 넘어, 신뢰할 수 있고 편향 없는 평가지표를 설계하는 데이터 엔지니어의 기본 자세를 다시 한번 깨닫는 경험이였습니다.

일반적인 인식인 "파라미터가 많고 깊은 모델이 성능이 좋을 것"이라는 맹신과 달리, CIFAR-10과 같이 상대적으로 작고 저해상도인 데이터셋에서는 ResNet50과 ResNet101을 비교했을 때 오히려 깊은 모델이 유의미한 성능 향상을 이끌어내지 못하거나 과적합(Overfitting)에 취약할 수 있음을 실험적으로 확인했습니다. 무조건적으로 복잡한 아키텍처를 도입하기보다는, 타겟 데이터의 복잡도와 모델의 표현력을 적절히 매칭하는 것이 컴퓨팅 리소스 및 추론 속도(Inference Speed) 최적화 측면에서 핵심적인 역량임을 실감했습니다.

이번 프로젝트는 "1%의 더 높은 정확도(Accuracy)"라는 단편적인 숫자를 좇는 것을 넘어, 왜 그러한 결과가 도출되었는지 가설을 세우고 시각화된 실험 데이터로 증명하는 과정이었습니다. 이 과정을 통해 모델의 학습 동향을 다각도로 분석하고 연산량 대비 효율성을 검증하면서, 향후 현업에서 마주할 다양한 AI 모델링 과제에서도 논리적이고 설명 가능한 의사결정을 내릴 수 있는 기초 근력을 확보했습니다.
