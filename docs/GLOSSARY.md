# 용어집 · Glossary (한 ↔ English)

본 프로젝트 문서에서 사용하는 인과추론·머신러닝·마케팅 용어의 한국어/영어 대조표이다.
영어 약어가 사실상 표준인 용어(CATE, ATE, AUUC, DML 등)는 **영어를 그대로 유지**하고, 한국어
서술과 병기한다. 이 표는 한/영 문서 간 용어 일관성을 보장하기 위한 단일 기준(single source)이다.

> A bilingual reference for the causal-inference, machine-learning, and marketing terminology used
> across this project. Established English acronyms are kept in English; this table is the single
> source of truth for term consistency between the Korean and English documents.

---

## 인과추론 · Causal Inference

| 한국어 | English | 비고 / Note |
|--------|---------|-------------|
| 인과추론 | causal inference | |
| 처치 / 처치효과 | treatment / treatment effect | "treatment"는 캠페인 노출을 의미 |
| 평균 처치효과 | Average Treatment Effect (**ATE**) | |
| 조건부 평균 처치효과 | Conditional Average Treatment Effect (**CATE**) | τ(x) = E[Y(1)−Y(0) \| X=x] |
| 이질적 처치효과 | Heterogeneous Treatment Effect (**HTE**) | 고객별로 다른 처치효과 |
| 잠재적 결과 | potential outcome | Y(1), Y(0) |
| 성향점수 | propensity score (**PS**) | e(x) = P(T=1 \| X=x) |
| 양정성 (가정) | positivity (assumption) | 0 < e(x) < 1 for all x |
| 겹침 (영역) | overlap (region) | 흔히 PS ∈ [0.1, 0.9] |
| 교란변수 | confounder | |
| 비교란성 / 무시가능성 | unconfoundedness / ignorability | |
| 외삽 | extrapolation | overlap 밖 추정 |
| 반증 검정 | refutation test | |
| 위약(플라시보) 검정 | placebo test | |
| 부분 식별 / 식별 구간 | partial identification / identified set | |
| Manski 경계 | Manski bounds | |
| 이중 강건 추정 | doubly robust (**DR**) estimation | |
| 역확률 가중 | inverse probability weighting (**IPW**) | |
| 겹침 가중 | overlap weighting (**ATO**) | |
| 이중 기계학습 | Double Machine Learning (**DML**) | |
| 교차적합 | cross-fitting | |
| SUTVA | stable unit treatment value assumption | 간섭/파급효과 없음 |
| 가설 생성적 | hypothesis-generating | 확증적(confirmatory)의 반대 |

## 세그멘테이션 · 머신러닝 / Segmentation · ML

| 한국어 | English | 비고 / Note |
|--------|---------|-------------|
| (고객) 세그먼트 | (customer) segment | |
| 잠재요인 | latent factor | |
| 비음수 행렬분해 | Non-negative Matrix Factorization (**NMF**) | V ≈ W·H |
| 군집화 | clustering (K-Means) | |
| 실루엣 점수 | silhouette score | |
| Calinski–Harabasz 지수 | Calinski–Harabasz index | |
| Davies–Bouldin 지수 | Davies–Bouldin index (**DBI**) | 낮을수록 좋음 |
| 부트스트랩 안정성 | bootstrap stability | |
| 조정 랜드 지수 | Adjusted Rand Index (**ARI**) | |
| 설명 분산 | explained variance | |
| 메타러너 | meta-learner (S / T / X-Learner) | |
| 업리프트 | uplift | |
| Qini 계수 | Qini coefficient | |
| 업리프트 곡선 하면적 | Area Under the Uplift Curve (**AUUC**) | |
| 최우수 선형 예측 검정 | Best Linear Predictor (**BLP**) test | 이질성 유의성 |

## 마케팅 · 비즈니스 / Marketing · Business

| 한국어 | English | 비고 / Note |
|--------|---------|-------------|
| 타겟팅 | targeting | |
| 정책 학습 | policy learning | |
| 손익분기 (CATE) | breakeven (CATE) | = 비용 / 마진 |
| (매출총)마진 | (gross) margin | |
| 투자수익률 | Return on Investment (**ROI**) | |
| 코호트 | cohort | treatment / control |
| 천장 효과 | ceiling effect | 이미 높은 고객 → 추가 효과 제한 |
| (자기)잠식 | cannibalization | 쿠폰이 기존 구매를 대체 |
| RFM (최근성·빈도·금액) | Recency, Frequency, Monetary | |
| 자체 브랜드 (PB) | private label | |
| 재방문 유도 / 윈백 | win-back | |
| 사전 지식(사전분포) | prior (knowledge) | |
