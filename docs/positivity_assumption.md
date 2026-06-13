# Causal Inference에서의 Positivity Assumption

> 본 문서는 Dunnhumby 인과추론 프로젝트(Track 2)의 핵심 식별(identification) 가정인 **Positivity (Overlap) Assumption**을 다루는 standalone 튜토리얼이다.
> 맥락(데이터·모델·정책)은 [Track 2 리포트](track2_report.md)를, 프로젝트 전체 개요는 [README](../README.md)를 참고하라.

---

## 한눈에 보기 (30초)

**TL;DR (3줄)**
1. 본 분석의 Treatment/Control은 **거의 완전히 분리**된다 — **PS AUC = 0.989**, Overlap 영역 `[0.1, 0.9]`에 전체의 **~17%**만 존재.
2. 그 결과 ATE 추정치는 추정기에 따라 **+$471 ~ -$65**로 요동치고(20배 이상 범위), CATE는 모델 간 **부호조차** 갈린다 — 모델 품질이 아니라 **식별(identification)의 문제**다.
3. 대응은 한 가지 추정치가 아니라 **PS trimming + ATO overlap weighting + Manski partial-identification bounds + sensitivity analysis**의 묶음이며, 모든 결과는 **가설 생성용(hypothesis-generating)** — 최종 확증은 **A/B 실험**으로 한다.

**핵심 수치 (verified)**

| 지표 | 값 | 진단 |
|------|-----|------|
| PS AUC | **0.989** | 심각 (> 0.9): 거의 완벽한 분리 |
| Overlap `[0.1, 0.9]` | **~17%** | 심각 (< 50%) |
| 균형 잡힌 공변량 | **9 / 21** | 절반 미만이 균형 |
| ATE 범위 (Naive→DML) | **+$471 → -$65** | 식별 불안정 (20배+) |
| CATE 부호 (모델 간) | +$15 ~ -$200 | 방향조차 불일치 |

**내비게이션 (30초 → 5분 → 30분)**
- [1. 개요 — 정의와 중요성](#1-개요)
- [2. Positivity Violation 진단](#2-positivity-violation-진단)
- [3. Violation의 결과](#3-violation의-결과)
- [4. 해결책 (PS trimming / ATO / Manski / sensitivity)](#4-해결책)
- [5. Partial-ID Bounds vs Conformal CATE](#5-cate-bounds-vs-conformal-cate)
- [6. 실무 권고 + 보고 템플릿](#6-실무-권고)
- [부록 A. PS 영역 분해 / B. 추정기 전체 표 / C. 참고문헌](#부록-a-ps-region-breakdown)

---

## 1. 개요

### 정의
Positivity (Overlap) Assumption은 다음과 같이 정의된다:
```
0 < P(T=1|X=x) < 1  for all x with positive density
```

모든 Covariate 조합은 Treatment와 Control 모두를 받을 양의 확률을 가져야 한다.

### 중요성
Positivity는 Causal Identification에 필수적이다:
- CATE 추정: τ(x) = E[Y|T=1,X=x] - E[Y|T=0,X=x]
- 두 Conditional Expectation 모두 양쪽 그룹의 관측치를 필요로 함
- Positivity 없이는 추정치가 관측이 아닌 외삽(Extrapolation)이 됨

> **본 프로젝트에서의 함의:** Dunnhumby 캠페인 노출(Treatment)은 무작위 배정이 아니라 기업의 타겟팅 규칙을 따른다. 따라서 노출 고객과 비노출 고객은 사전(pre-period, 주 1–31) 행동이 체계적으로 다르고, 이것이 아래에서 보이는 심각한 positivity 위반의 근원이다.

## 2. Positivity Violation 진단

### Propensity Score 메트릭

| 메트릭 | 양호 | 경고 | 심각 |
|--------|------|------|------|
| PS AUC | 0.5-0.7 | 0.7-0.9 | > 0.9 |
| Overlap [0.1, 0.9] | > 80% | 50-80% | < 50% |
| SMD (평균) | < 0.1 | 0.1-0.25 | > 0.25 |

### 현재 분석 결과
- **PS AUC: 0.989** (거의 완벽한 분리 — "심각" 구간)
- **Overlap [0.1, 0.9]: ~17%** (전체의 1/6만 비교 가능 영역에 존재)
- **균형 잡힌 공변량: 21개 중 9개** (절반 미만 — 나머지는 가중치/trimming 후에도 잔차 불균형 가능)
- 의미: Treatment와 Control은 근본적으로 다른 모집단임

> 일부 구버전 문서가 PS AUC를 0.990 또는 overlap 31%로 기재했으나, committed 결과물 기준 표준값은 **AUC 0.989 / overlap ~17%**이다. 본 문서는 이 값으로 통일한다.

### 높은 PS AUC의 의미
```
PS AUC ≈ 0.989는 다음을 의미:
- Covariate가 주어지면 Treatment 상태를 거의 완벽하게 예측 가능
- Treatment 관측치가 많은 곳에 Control 관측치가 거의 없음 (역도 마찬가지)
- 높은 PS 영역의 CATE는 순전히 모델 외삽
```

## 3. Violation의 결과

### ATE 불안정성
서로 다른 추정기가 매우 다른 결과를 제공 (n=2,430, `ate_results_purchase_amount.csv`):

| 방법 | 추정치 | 95% CI | 해석 |
|------|--------|--------|------|
| Naive | +$471 | [442, 501] | Selection Bias 포함 |
| IPW | +$151 | [-10, 313] | 극단적 가중치, CI가 0을 포함 |
| AIPW | +$24 | [-56, 104] | 모델 의존적, 0 포함 |
| OLS | +$65 | [29, 102] | 선형 외삽 |
| DML | -$65 | [-220, 90] | Cross-fitting이 도움이 되나 면역은 아님 |
| ATO (overlap) | +$60 | [-14, 134] | Overlap 모집단에 한정한 효과 |

Naive(+$471)부터 DML(-$65)까지 추정치가 **부호를 바꾸며 20배 이상** 벌어지는 것은 근본적인 Identification 문제를 나타낸다. Naive를 제외한 모든 추정기의 95% CI가 0을 포함한다.

### CATE 모델 불일치
모델들이 Treatment Effect **방향**에서도 불일치한다 (test set mean CATE, `cate_summary_purchase_amount.csv`):
- CausalForestDML: **+$15** (양수, std $52 — 저분산)
- LinearDML: **-$139** (음수, std $452 — 고분산/불안정)
- T-Learner: **-$200** (음수, std $397)

이것은 모델 품질의 문제가 아니라 Identification의 문제이다. (전체 코호트 486명에 대한 CausalForestDML 평균 CATE는 +$10으로 일관되며, headline mean으로 인용 가능하다.)

### Anti-Predictive / 불안정한 AUUC
높은 AUUC가 곧 좋은 모델을 뜻하지 않는다. 본 분석에서 **LinearDML의 AUUC(357.0)가 가장 높지만** 평균 -$139 / std $452로 사실상 사용 불가하고, NonParamDML(AUUC 304.4)은 평균 추정이 발산($1M+)한다. CausalForestDML은 AUUC 4위(271.6)지만 저분산·양(+)의 그럴듯한 분포를 가져 선택되었다 — 자세한 모델 선택 논리는 [Track 2 리포트](track2_report.md) 참고. positivity 위반 하에서는 raw AUUC보다 **안정성·타당성**이 우선한다.

## 4. 해결책

위반을 "고치는" 단일 추정치는 없다. 아래 네 가지를 **함께** 사용해 식별 불확실성을 정직하게 노출한다.

### 4.1 PS Trimming
```python
# 극단적 PS 값 제외
mask = (ps >= 0.1) & (ps <= 0.9)
data_trimmed = data[mask]
```

**장점:** 외삽 제거 (overlap 영역에서만 추정)
**단점:** 샘플 손실 (본 데이터에서는 overlap이 ~17%이므로 **80% 이상 손실** 가능), Local Effect만 추정

### 4.2 Overlap Weighting (ATO)
```python
# Propensity Overlap으로 가중치 부여
h = ps * (1 - ps)  # 0.5에서 최대
weights = h / ps        # for treated
weights = h / (1 - ps)  # for control
```

**장점:** 모든 데이터 사용, 극단값 Down-weight, trimming의 임계값 임의성 회피
**단점:** ATE가 아닌 Average Treatment effect on the Overlap (ATO) 추정 (본 분석 ATO = +$60 [-14, 134])

### 4.3 Partial Identification Bounds

#### ATE에 대한 Manski Bounds
```
ATE ∈ [E[Y|T=1] - Y_max, E[Y|T=1] - Y_min]
```

추가 가정 없이 이것이 Identified Set이다. 본 데이터에서는 결과 변수 범위가 넓어 **Manski bounds도 매우 넓게(wide)** 나오며, 이는 데이터가 점추정을 지지하지 못한다는 사실을 정직하게 드러낸다.

#### CATE Bounds
각 x에 대해:
```
τ(x) ∈ [μ₁(x) - Y_max, μ₁(x) - Y_min]
```

Bounds는 Overlap 영역에서 좁고, 외부에서 넓다.

### 4.4 Sensitivity Analysis
다음에 대해 안정성 테스트 (본 분석은 모두 수행):
- 다른 Trimming Threshold
- 다른 Covariate Set
- 다른 모델 (S/T/X-Learner, Linear/NonParam/CausalForest DML)

> **검증 상태 (정직한 보고):** 본 분석의 refutation test는 일부 **실패**한다 — Placebo Treatment(Amount) = 0.747(임계 <0.5, **FAIL**), Subset Stability = 0.561(임계 >0.7, **FAIL**). Placebo(Visits) = 0.052는 통과. 이 실패는 심각한 positivity 위반 하에서 **예상되는** 결과이며, 따라서 모든 CATE/정책 결과는 **가설 생성용**이고 A/B 실험(설계: n=5,748, 2,874/arm, power 0.80, detectable effect ~$34)으로 확증되어야 한다.

## 5. CATE Bounds vs Conformal CATE

| 측면 | Partial ID Bounds | Conformal CATE |
|------|-------------------|----------------|
| **측정 대상** | Identification Uncertainty | Prediction Uncertainty |
| **원인** | Missing Counterfactual | 모델/샘플링 변동성 |
| **Overlap 필요?** | 아니오 (Bounds가 넓어짐) | 예 |
| **Positivity Violation** | Bounds가 정직 (넓음) | 추정치가 편향 |
| **권고** | Primary | Supplementary |

**본 분석에서**: Partial ID Bounds를 Primary로 사용, Conformal은 Overlap 영역에서만 사용.

## 6. 실무 권고

### 분석을 위해
1. 항상 PS AUC와 Overlap Ratio 보고
2. Trimming과 Bounds를 함께 사용
3. Sensitivity Analysis는 필수
4. 결론에 명시적으로 주의사항 기재 (refutation 실패 포함 — 숨기지 않음)

### 비즈니스 의사결정을 위해
1. Overlap 영역에 권고사항 집중
2. Overlap 외부의 High-CATE 고객: 불확실
3. ROI 예측에 불확실성 범위 포함
4. CATE 예측 검증을 위한 A/B Testing 고려 (확증의 유일한 신뢰 경로)

### 보고 템플릿
```
ATE Estimate: $X (trimmed [0.1, 0.9])
95% CI: [$Y, $Z]
Manski Bounds: [$A, $B]

주의사항:
- PS AUC = 0.989는 Positivity Violation을 나타냄
- 샘플의 ~17%만 Overlap 영역에 존재
- Overlap 외부의 추정치는 외삽
- Refutation test 일부 실패 → 결과는 가설 생성용
- 결과는 무작위 실험(A/B test)으로 검증되어야 함
```

---

## 부록 A. PS Region Breakdown

진단 시 Propensity Score를 구간별로 분해해 어디에서 비교가 가능한지 명시한다. 본 분석은 overlap 영역 `[0.1, 0.9]`에 전체의 **~17%**만 존재한다(나머지 ~83%는 PS<0.1 또는 PS>0.9의 외삽 영역).

| PS 영역 | 라벨 | 비교 가능성 | 처리 |
|---------|------|-------------|------|
| PS < 0.1 | No-overlap (Control 지배) | 거의 불가 | Trim / down-weight, bounds로만 보고 |
| 0.1 ≤ PS ≤ 0.9 | **Overlap (~17%)** | 가능 | Primary 추정 영역 (ATO·trimmed) |
| PS > 0.9 | No-overlap (Treatment 지배) | 거의 불가 | Trim / down-weight, bounds로만 보고 |

> 임계 0.1/0.9는 관례적(Crump et al. 2009 계열)이며 sensitivity analysis에서 다른 임계값으로도 재확인한다.

## 부록 B. 추정기 전체 표 (참조용)

ATE 추정기 비교 (n=2,430, `ate_results_purchase_amount.csv`):

| 방법 | 추정치 | 95% CI | 식별 대상 |
|------|--------|--------|-----------|
| Naive | +$471 | [442, 501] | (편향) 단순 차이 |
| IPW | +$151 | [-10, 313] | ATE (역확률 가중) |
| AIPW | +$24 | [-56, 104] | ATE (이중 강건) |
| OLS | +$65 | [29, 102] | (선형 외삽) |
| DML | -$65 | [-220, 90] | ATE (cross-fitting) |
| ATO | +$60 | [-14, 134] | ATO (overlap 모집단) |

CATE 모델 비교 (test set; `cate_summary_purchase_amount.csv` / `auuc_comparison_purchase_amount.csv`):

| 모델 | mean CATE | std | AUUC | % positive |
|------|-----------|-----|------|------------|
| CausalForestDML **(선택)** | +$15 | $52 | 271.6 | 78% |
| LinearDML | -$139 | $452 | 357.0 | 42% |
| NonParamDML | +$1.1M (발산) | huge | 304.4 | 64% |
| S-Learner | -$21 | $46 | 289.5 | 21% |
| X-Learner | -$96 | $208 | 218.5 | 38% |
| T-Learner | -$200 | $397 | 212.0 | 43% |

선택 논리(안정성+타당성 우선)와 BLP 유의성(BLP p=0.094 borderline, X-Learner p=0.005)은 [Track 2 리포트](track2_report.md) 참조.

## 부록 C. 참고문헌

- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects.
- Li, F., Morgan, K. L., & Zaslavsky, A. M. (2018). Balancing covariates via propensity score weighting.
- Manski, C. F. (1990). Nonparametric bounds on treatment effects.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research.

---

[← README로 돌아가기](../README.md) · [Track 2 리포트 보기 →](track2_report.md)
