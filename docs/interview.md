# Critical Self-Assessment & 예상 면접 Q&A

> [← README로 돌아가기](../README.md)

본 문서는 **저자 본인의 비판적 자기평가(Critical Self-Assessment)**와 **예상 면접 Q&A**다. 외부 채점표가 아니라, 이 분석을 만든 사람이 스스로 강점과 한계를 정직하게 짚고, 면접에서 나올 법한 날카로운 질문에 미리 답을 준비한 기록이다. 평가 렌즈는 세 가지 — **Data Scientist / Marketer / Recruiter** — 로, 같은 결과물을 서로 다른 이해관계자의 눈으로 다시 본다.

이 프로젝트의 핵심 입장은 한 줄로 요약된다: **"심각한 positivity violation(PS AUC 0.989) 아래서 얻은 CATE는 confirmatory가 아니라 hypothesis-generating이다. 그래서 결론은 '쓸 수 없다'가 아니라 'A/B Test로 검증한 뒤 쓴다'이다."**

---

## ⏱️ 30초 요약 (TL;DR)

1. **정직성이 이 분석의 정체성이다.** PS AUC 0.989 / overlap 17% / refutation FAIL을 숨기지 않고 1급 결과로 보고했다. 한계를 드러내는 것 자체가 신뢰의 근거다.
2. **모델 선택은 "최고 AUUC"가 아니라 "안정성 + 타당성"으로 했다.** LinearDML이 AUUC는 더 높지만(357) 평균 −\$139·표준편차 \$452로 불안정하다. CausalForestDML(std \$52, 평균 +\$10~15, 78% 양수)을 선택한 근거를 정직하게 설명할 수 있다.
3. **결론은 검증 가능한 다음 단계로 연결된다.** 최적 타겟팅 31.3%(+\$2,426, ROI 125%) → A/B Test(n=5,748, MDE ~\$34)로 확정.

### 핵심 숫자 한눈에

| 항목 | 값 | 상태/해석 |
|------|-----|-----------|
| PS AUC | **0.989** | 심각한 positivity violation (정직 보고) |
| Overlap 구간 [0.1, 0.9] | **17%** | 83%가 외삽 영역 |
| 균형 잡힌 공변량 | **9 / 21** | 절반 미만 |
| 선택 모델 | **CausalForestDML** | 최고 AUUC 아님 — 안정성·타당성 기준 |
| CausalForestDML CATE | 평균 **+\$10~15**, std **\$52**, 78% 양수 | 캠페인 목적과 정합 |
| Breakeven CATE | **\$42.43** | = 비용 \$12.73 / 마진 0.30 |
| 최적 타겟팅 | **31.3%** (152/486) | profit **+\$2,426**, ROI **125%** |
| 전체 타겟팅(100%) | profit **−\$4,659**, ROI **−75%** | 최적 대비 −\$7,085 |
| Refutation (Placebo Amount) | **0.747** (임계 <0.5) | **FAIL** — 정직 보고 |
| A/B Test 설계 | n=**5,748** (2,874×2), power 0.80 | MDE ~**\$34** |

### 문서 내비게이션

- [1. Data Scientist 렌즈 — 방법론·인과추론](#1-data-scientist-렌즈)
- [2. Marketer 렌즈 — 실행 가능성·ROI](#2-marketer-렌즈)
- [3. Recruiter 렌즈 — 역량 증명](#3-recruiter-렌즈)
- [4. 예상 면접 Q&A](#4-예상-면접-qa-준비된-답변)
- [5. 종합 자기평가](#5-종합-자기평가)
- [부록 A. 자기평가의 근거가 된 정본 수치](#부록-a-자기평가의-근거가-된-정본-수치)

---

## 1. Data Scientist 렌즈

> *"관리자급 DS가 내 코드와 결과를 본다면 어디를 칭찬하고 어디를 찌를까?"*

### 1.1 내가 잘했다고 보는 부분

| 영역 | 자기평가 | 근거 |
|------|----------|------|
| **방법론 체계성** | 우수 | 2-Track Framework가 명확하고 Descriptive(Track 1) → Causal(Track 2) 순서가 논리적. 시간 분할(pre wk 1–31 / outcome wk 32–102)로 시계열 누수 차단 |
| **Positivity 진단** | 탁월 | PS AUC 0.989를 숨기지 않고 1급 결과로 보고. overlap 17%, 균형 공변량 9/21을 명시하고 완화책(PS trimming, ATO overlap weighting, Manski partial-identification bounds, sensitivity analysis) 적용 |
| **모델 선택의 정직성** | 양호 | "최고 AUUC를 골랐다"는 흔한 함정에 빠지지 않고, 안정성·타당성 기준으로 선택한 뒤 그 trade-off를 문서에 그대로 노출 |
| **한계점 인식** | 탁월 | Refutation Test FAIL(placebo 0.747, subset 0.561), 외삽 83%를 솔직하게 보고하고 결과를 hypothesis-generating으로 규정 |

### 1.2 내가 스스로 보완하고 싶은 부분

1. **Refutation Test 실패의 원인 분석을 더 깊게.**
   - Placebo Treatment(Amount) = 0.747 (임계 <0.5) → **FAIL**. Subset Stability = 0.561 (임계 >0.7) → **FAIL**.
   - 이는 positivity violation 하에서 **예상된** 결과이지만, spurious correlation의 출처(어떤 공변량이 placebo와 결합하는지)를 더 분해하지 못했다. *(참고: Placebo(Visits) = 0.052는 통과 — outcome에 따라 결과가 갈린다는 점 자체가 추가 분석 단서다.)*

2. **모델 간 부호 불일치 해석.**
   - CausalForestDML 평균 **+\$10~15** vs LinearDML 평균 **−\$139** vs T-Learner **−\$200**. 부호가 모델마다 갈린다.
   - 이는 곧 추정치의 식별 불안정성을 보여주는 증거인데, 나는 이를 "CausalForestDML이 안정·타당하니 선택"으로 정리했지 *왜* 다른 모델이 음수로 발산하는지(positivity 영역 밖 외삽 때문)를 정량적으로 더 파고들 여지가 있다.

3. **Unobserved confounder 민감도(E-value) 미포함.**
   - Manski bounds로 partial identification은 했지만, E-value 류의 unmeasured confounding 민감도 지표를 추가하면 "얼마나 강한 교란이 있어야 결론이 뒤집히나"를 정량화할 수 있다.

### 1.3 모델 선택에 대한 나의 입장 (가장 자주 받는 지적)

> **흔한 오해:** "CausalForestDML이 AUUC가 제일 높아서 골랐다."
> **사실(committed CSV 기준):** 그렇지 않다. main run에서 **AUUC가 가장 높은 건 LinearDML(357.0)**이고 CausalForestDML(271.6)은 4위다.

내가 CausalForestDML을 선택한 진짜 이유는 **안정성(stability) + 타당성(plausibility)**이다.

| Model | 평균 CATE | std | AUUC | 양수 비율 | 판정 |
|-------|----------|-----|------|----------|------|
| **CausalForestDML** | **+\$15** | **\$52** | 271.6 | 78% | **선택** — 저분산 + 타당한 양수 분포 |
| LinearDML | −\$139 | \$452 | **357.0** | 42% | 최고 AUUC지만 불안정·음수 평균 |
| NonParamDML | +\$1.1M (발산) | 거대 | 304.4 | 64% | 발산 → 사용 불가 |
| S-Learner | −\$21 | \$46 | 289.5 | 21% | 분산은 낮지만 79%가 음의 효과 → 비현실적 |
| X-Learner | −\$96 | \$208 | 218.5 | 38% | 음수 평균 |
| T-Learner | −\$200 | \$397 | 212.0 | 43% | 음수 평균·고분산 |

논리는 이렇다: **positivity가 심각하게 깨진 상황에서는 raw AUUC보다 추정의 안정성과 분포의 타당성을 우선해야 한다.** AUUC 상위 모델들은 쓸 수 없다 — LinearDML은 평균 −\$139·std \$452로 불안정하고, NonParamDML은 +\$1.1M로 발산한다. S-Learner는 분산이 비슷하게 낮지만 고객의 79%가 음의 효과를 가진다고 말하는데, 이는 "캠페인이 대부분 고객에게 해롭다"는 뜻이라 캠페인 목적과 모순된다. **CausalForestDML만이 (a) 낮은 분산(std \$52)과 (b) 캠페인 의도에 부합하는 대체로 양(+)인 분포(평균 +\$10~15, 78% 양수)를 동시에 만족하는 유일한 모델이다.**

이 선택을 뒷받침하는 nuance도 정직하게 남겨둔다: **BLP(Best Linear Predictor) test p = 0.094로 경계선**이고, **X-Learner는 p = 0.005**다. 즉 heterogeneity의 통계적 증거가 강하지 않다 — 이 또한 결과를 hypothesis-generating으로 봐야 하는 이유를 강화한다.

> **기술 코멘트(자기 검열):** "PS AUC 0.989는 사실상 treatment를 거의 완벽히 예측한다는 뜻이라, 이 데이터는 observational study로 causal effect를 confirmatory하게 추정하기 어렵다. 나는 이를 인지하고 모든 CATE를 hypothesis-generating으로 규정했으며, 다음 단계를 A/B Test로 명시했다. 다만 본문 곳곳에서 이 단서(caveat)를 더 반복적으로 강조했어야 한다."

---

## 2. Marketer 렌즈

> *"마케팅 실무자가 이걸 받으면 내일 무엇을 할 수 있나?"*

### 2.1 내가 잘했다고 보는 부분

| 영역 | 자기평가 | 근거 |
|------|----------|------|
| **실행 가능성** | 탁월 | 세그먼트별 구체 액션(Test&Learn / Reduce)으로 번역 |
| **ROI 정량화** | 우수 | 전체 타겟팅(**−\$4,659**, ROI −75%) vs 최적 31.3%(**+\$2,426**, ROI 125%) — 개선폭 **+\$7,085** 명확 비교 |
| **페르소나 해석** | 우수 | "VIP Heavy / Bulk Shoppers가 음의 CATE" → 고가치 세그먼트 과다 타겟팅 인사이트 |
| **예산 배분 가이드** | 양호 | 정책 포트폴리오(보수적 → 최적 → 공격적)로 위험성향별 선택지 제시 |

### 2.2 즉시 활용 가능한 인사이트 (방향성으로 해석)

> ⚠️ 아래의 "현재/권장 타겟팅 %"는 **source CSV에 없는 illustrative 수치**다. 검증된 사실은 (a) **세그먼트별 평균 CATE**, (b) **Track-2 cohort N**, (c) **권장 액션(Reduce vs Test&Learn)** 세 가지뿐이며, 비율은 방향성(확대/유지/축소)으로만 읽어야 한다.

**세그먼트별 권장 액션 (segment_recommendations.csv · 486 cohort 기준):**

| Seg | 이름 | N (486 cohort) | 평균 CATE | 권장 액션 |
|-----|------|----------------|-----------|-----------|
| 0 | Active Loyalists | 97 | **+\$33** | Test & Learn (소폭 확대) |
| 6 | Regular + H&B | 62 | **+\$34** | Test & Learn (소폭 확대) |
| 4 | Light Grocery | 91 | **+\$30** | Test & Learn (확대) |
| 3 | Fresh Lovers | 73 | **+\$27** | Test & Learn (확대) |
| 2 | Lapsed H&B | 27 | **+\$19** | Test & Learn |
| 1 | VIP Heavy | 59 | **−\$38** | **REDUCE / TypeA 제외** |
| 5 | Bulk Shoppers | 77 | **−\$40** | **REDUCE / TypeA 제외** |

> 📌 **N 혼동 주의:** 위 N(97, 62, 91, …)은 **Track 2 인과분석 cohort 인원**이며, Track 1 전체 세그먼트 규모(509, 299, 193, …)와 **다른 수치**다. 같은 세그먼트라도 분석 모집단이 다르므로 직접 비교하지 말 것.

**비즈니스 액션 우선순위:**

1. **[축소] VIP Heavy(−\$38), Bulk Shoppers(−\$40) TypeA 타겟팅 축소.**
   - 두 세그먼트는 음의 CATE → TypeA 캠페인이 오히려 손실. TypeA 대상에서 제외/축소 권장.
   - 다만 "중단" 전에 2.3의 우려를 반드시 검토.

2. **[확대] Light Grocery(+\$30), Fresh Lovers(+\$27) 타겟팅 확대.**
   - 양의 CATE인데 상대적으로 과소 타겟팅으로 추정 → Test & Learn으로 확대.

3. **[검증] A/B Test 설계 (n=5,748).**
   - arm당 2,874명, power 0.80, alpha 0.05, **detectable effect ~\$34**.
   - 2–4주 캠페인 사이클로 hypothesis를 confirmatory하게 전환.

### 2.3 마케팅 관점에서의 내 우려

> "VIP Heavy의 음의 CATE가 **'캠페인 자체가 해롭다'**인지 **'TypeA가 이 세그먼트에 안 맞는다'**인지 구분이 필요하다. TypeB/C 캠페인 테스트 없이 타겟팅을 전면 중단하면 고가치 고객 이탈 리스크가 있다. 따라서 'REDUCE'는 'TypeA에서 제외'이지 '모든 접촉 중단'이 아니다 — 이 구분을 액션 문서에 못박아야 한다."

---

## 3. Recruiter 렌즈

> *"이 포트폴리오가 증명하는 역량은 무엇이고, 무엇이 더 있으면 완벽한가?"*

### 3.1 이 프로젝트가 증명한다고 보는 역량

| 역량 | 수준 | 증거 |
|------|------|------|
| **Causal Inference** | Senior | HTE 추정, DML 계열 다중 추정량, policy learning 전 파이프라인 구현. positivity 진단·partial identification까지 |
| **ML Engineering** | Mid-Senior | Optuna 튜닝, 6종 CATE 모델 비교, Ray 병렬화 |
| **비즈니스 커뮤니케이션** | Senior | 기술 결과를 ROI/세그먼트 액션으로 번역 |
| **연구 윤리·정직성** | Senior | refutation FAIL·외삽 83%를 숨기지 않고 보고, AUUC 과장 없이 모델 선택 근거를 투명하게 노출 |

### 3.2 포트폴리오 강점

- **End-to-End:** EDA → Segmentation(Track 1) → Causal(Track 2) → Policy → A/B 설계까지 완결된 흐름.
- **실제 데이터 복잡성 대응:** positivity violation, 불균형 그룹, 모델 부호 불일치를 회피하지 않고 정면 처리.
- **재현성:** Config → Function → Result NamedTuple 패턴으로 코드 구조화, 파라미터 문서화, committed CSV로 수치 추적 가능.
- **학문적 엄밀성 + 비즈니스 감각 균형:** "정직해서 못 쓴다"가 아니라 "정직하게 한계를 규정하고 검증 경로를 제시"하는 실무적 마무리.

### 3.3 더 있으면 좋을 부분 (스스로 인정하는 갭)

1. **A/B Test 실제 수행 결과** → hypothesis-generating에서 confirmatory까지 완결.
2. **MLOps 관점** → 모델 재훈련 파이프라인, CATE 모니터링 대시보드, drift 감지.
3. **TypeB/C 캠페인 비교** → "TypeA가 안 맞는다 vs 캠페인이 해롭다"를 분리해 일반화.

---

## 4. 예상 면접 Q&A (준비된 답변)

면접에서 나올 법한 날카로운 질문과, 내가 준비한 자신 있는 답변이다. 핵심은 **숨기지 않고, 그러나 다음 단계로 연결하는 것**이다.

**Q1. "PS AUC 0.989면 사실상 treatment가 완전히 예측되는 상황인데, 왜 분석을 진행했나?"**
> 맞다. AUC 0.989는 propensity로 treatment를 거의 완벽히 분리한다는 뜻이고, overlap 구간 [0.1, 0.9]에 든 고객은 17%뿐이다(83%가 외삽). 그래서 나는 처음부터 이 분석을 **confirmatory가 아니라 hypothesis-generating**으로 규정했다. observational data로는 여기서 멈추는 게 정직하다 — 대신 partial identification(Manski bounds), ATO overlap weighting으로 결론이 가장 신뢰될 영역만 강조하고, 최종 의사결정은 **A/B Test(n=5,748, MDE ~\$34)**로 넘긴다. 즉 "쓸 수 없다"가 아니라 "검증 후 쓴다"이다.

**Q2. "Refutation Test가 실패했는데 결과를 신뢰할 수 있나?"**
> 실패를 인정한다 — Placebo(Amount) = 0.747(임계 <0.5), Subset Stability = 0.561(임계 >0.7)로 둘 다 **FAIL**이다. 하지만 이건 positivity가 심각하게 깨진 상황에서 **예상된** 결과다. 오히려 통과했다면 그게 더 의심스러웠을 것이다. 흥미롭게도 Placebo(Visits) = 0.052는 통과했는데, outcome에 따라 결과가 갈린다는 점 자체가 추가 분석 단서다. 결론적으로 이 실패는 "결과를 프로덕션에 그대로 배포하면 안 되고 A/B로 confirmatory 검증이 필요하다"는 신호로 해석하며, 나는 이를 처음부터 결론에 반영했다.

**Q3. "왜 CausalForestDML인가? AUUC가 제일 높은 모델이 아니지 않나?"**
> 정확하다. main run에서 AUUC가 가장 높은 건 LinearDML(357.0)이고 CausalForestDML(271.6)은 4위다. 내가 raw AUUC로 골랐다면 LinearDML을 선택했어야 한다. 하지만 LinearDML은 평균 −\$139·std \$452로 불안정하고 음의 평균을 내며, NonParamDML은 +\$1.1M로 발산한다. S-Learner는 분산은 낮지만 고객의 79%가 음의 효과라 캠페인 목적과 모순된다. **CausalForestDML만이 저분산(std \$52)과 타당한 양수 분포(평균 +\$10~15, 78% 양수)를 동시에 만족하는 유일한 모델**이다. positivity violation 아래서는 raw AUUC보다 안정성·타당성을 우선하는 게 옳다고 본다. 덧붙여 BLP test p=0.094(경계선), X-Learner p=0.005 — heterogeneity의 통계 증거가 강하지 않다는 점도 정직하게 함께 보고한다.

**Q4. "모델마다 CATE 부호가 다른데(+\$15 vs −\$200) 어떻게 신뢰하나?"**
> 부호 불일치 자체가 positivity violation의 증상이다. 외삽 영역(83%)에서는 모델의 함수 형태 가정에 따라 추정이 크게 흔들린다. 그래서 나는 단일 추정치를 confirmatory하게 쓰지 않고, (a) 가장 안정적인 모델의 분포, (b) ATE 추정량 6종의 범위(naive +\$471 → dml −\$65, 대부분 CI가 0을 포함), (c) partial identification bounds를 함께 제시했다. 결론은 "효과의 부호조차 데이터만으로는 확정 못 한다 → 실험으로 확정한다"이다.

**Q5. "VIP Heavy가 음의 CATE인 이유는?"**
> 두 가설이 있다. (1) **Ceiling effect** — 이미 \$9,716을 쓰는 헤비 고객은 추가 캠페인으로 더 올릴 여지가 적다. (2) **Cannibalization** — 어차피 살 물건에 쿠폰을 줘 마진만 깎는다. 다만 이 음수가 "캠페인이 해롭다"인지 "TypeA가 이 세그먼트에 안 맞는다"인지는 데이터만으로 구분 못 한다. 그래서 권장은 "모든 접촉 중단"이 아니라 "**TypeA에서 제외**"이고, TypeB/C 테스트로 원인을 분리하자고 제안한다.

**Q6. "이 결과로 실제 마케팅 의사결정을 내릴 수 있나?"**
> 직접 내리진 않는다 — 한 단계가 더 있다. 분석은 **우선순위와 가설**을 준다: 최적 타겟팅 31.3%(+\$2,426, ROI 125%)가 전체 타겟팅(−\$4,659, ROI −75%) 대비 +\$7,085 개선이라는 *방향*과, VIP/Bulk를 줄이고 Grocery/Fresh를 늘리라는 *우선순위*다. 이걸 **A/B Test로 confirmatory 검증한 뒤** 롤아웃하는 게 내 권고다. breakeven sensitivity(마진 25~35%, 비용 \$10~15)에서도 최적 정책은 항상 양의 profit(+\$1,461 ~ +\$3,702)이라 방향성은 견고하다.

---

## 5. 종합 자기평가

| 렌즈 | 자기평가 | 한 줄 요약 |
|------|----------|-----------|
| **Data Scientist** | 4.0 / 5 | 방법론 탄탄, positivity 대응 우수. 모델 부호 불일치 원인·E-value 보완 여지 |
| **Marketer** | 4.5 / 5 | 즉시 실행 가능한 액션. TypeB/C 대안 분석 추가 권장 |
| **Recruiter** | 4.2 / 5 | Senior DS 역량 증명. A/B 실측 결과·MLOps가 더해지면 완성 |

### 최종 입장

> 이 프로젝트는 **"정직한 Causal Analysis"**를 지향한다. PS AUC 0.989라는 심각한 positivity violation을 숨기지 않았고, refutation FAIL과 외삽 83%를 1급 결과로 보고했으며, 모델 선택조차 "최고 AUUC"라는 매력적 서사 대신 "안정성·타당성"이라는 덜 화려하지만 정직한 근거로 했다. 핵심은 이 정직함이 **"결과를 못 쓴다"로 끝나지 않고 "A/B Test로 검증 후 활용"이라는 명확한 다음 단계로 연결**된다는 점이다. 한계를 드러내는 것과 실무 가치를 만드는 것이 충돌하지 않음을 보이는 것 — 그게 이 포트폴리오로 증명하려는 바다.

---

## 부록 A. 자기평가의 근거가 된 정본 수치

> 위 본문의 모든 수치는 committed result CSV에서 검증된 값이다. 빠른 교차확인용으로 핵심 표를 모아둔다. (deep-dive는 [README](../README.md) 및 `docs/positivity_assumption.md` 참조.)

### A.1 ATE 추정량 비교 (ate_results_purchase_amount.csv, n=2,430)

| 추정량 | ATE | 95% CI |
|--------|-----|--------|
| Naive | +\$471 | [442, 501] |
| IPW | +\$151 | [−10, 313] |
| AIPW | +\$24 | [−56, 104] |
| OLS | +\$65 | [29, 102] |
| DML | −\$65 | [−220, 90] |
| ATO (overlap) | +\$60 | [−14, 134] |

대부분의 doubly-robust 추정량 CI가 0을 포함 → ATE조차 데이터만으로 확정하기 어렵다.

### A.2 정책 포트폴리오 비교 (policy_comparison.csv)

| 정책 | 대상 수 | 비율 | Profit | ROI | 비고 |
|------|---------|------|--------|-----|------|
| **CATE > Breakeven** | 152 | **31.3%** | **+\$2,426** | **125%** | **최적** |
| Top 20% CATE | 97 | 20.0% | +\$2,259 | 183% | 예산제약, 최고 ROI |
| Conservative (Lower CI > BE) | 3 | 0.6% | +\$188 | 493% | A/B 전 초안전 |
| Risk-Adjusted (30%) | 54 | 11.1% | +\$1,603 | 233% | |
| PolicyTree (Tuned) | 130 | 26.7% | +\$1,684 | 102% | 해석 가능 규칙 |
| CATE > 0 | 314 | 64.6% | +\$1,447 | 36% | |
| Current Practice | 302 | 62.1% | −\$3,402 | −88% | |
| Full Targeting | 486 | 100% | −\$4,659 | −75% | |

### A.3 Breakeven 민감도 (breakeven_scenarios.csv)

| 시나리오 | Breakeven CATE | 최적 타겟 % | Profit |
|----------|----------------|-------------|--------|
| 비용 \$12.73 / 마진 30% (base) | \$42.43 | 31.3% | +\$2,426 |
| 마진 25% | \$50.92 | 26.1% | +\$1,726 |
| 마진 35% | \$36.37 | 36.0% | +\$3,173 |
| 비용 \$15 | \$50.00 | 26.5% | +\$2,107 |
| 비용 \$10 | \$33.33 | 39.7% | +\$2,888 |
| worst (비용 \$15 / 마진 25%) | — | — | +\$1,461 |
| best (비용 \$10 / 마진 35%) | — | — | +\$3,702 |

최악 시나리오에서도 최적 정책 profit은 양수(+\$1,461) → 방향성은 견고하다.

### A.4 Refutation Test 상세

| 테스트 | 값 | 임계 | 판정 |
|--------|-----|------|------|
| Placebo Treatment (Amount) | 0.747 | < 0.5 | **FAIL** |
| Placebo Treatment (Visits) | 0.052 | < 0.5 | pass |
| Subset Stability | 0.561 | > 0.7 | **FAIL** |

positivity violation 하에서 예상된 결과 — hypothesis-generating으로 해석, A/B Test로 확정.

### A.5 Positivity 진단 요약

- **PS AUC = 0.989** (treatment를 거의 완벽히 예측 → 심각한 violation)
- **Overlap [0.1, 0.9] ≈ 17%** (83%가 외삽 영역)
- **균형 잡힌 공변량 = 9 / 21**
- **Manski bounds:** wide (부호조차 확정 못 하는 경우 존재)
- **완화책:** PS trimming, ATO overlap weighting, partial-identification bounds, sensitivity analysis

---

> [← README로 돌아가기](../README.md)
