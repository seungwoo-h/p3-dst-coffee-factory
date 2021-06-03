# [Stage 3] Dialogue State Tracking

[개인 wrap-up report](https://github.com/seungwoo-h/p3-dst-coffee-factory/blob/seungwoohong/README.md)

**☕️  (1조) Coffee Factory - 민지원, 이애나, 정지훈, 진소정, 최병민, 홍승우**

**Private 0.6926 - 최종 3위**

*Boostcamp AI Tech* P-Stage 3 의 대화 상태 추적 (Dialogue State Tracking) 대회는 목적 지향형 대화에서 대화 턴 마다 미리 정의된 slot 들에 대한 value 값들을 예측하여 Joint Goal Accuracy → Slot F1 Score → Slot Accuracy 순으로 순위를 측정합니다.

---

## 사용한 모델 아키텍쳐

- [SUMBT](https://github.com/SKTBrain/SUMBT)
- [TRADE](https://github.com/jasonwu0731/trade-dst)
- [DST-STAR](https://arxiv.org/abs/2101.09374)

Encoder 모델은 한국어로 학습된 BERT 를 fine-tuning 하였습니다.

---

## Training Strategies

- Ensemble - 6개 모델 앙상블
- Post-processing - 코사인 유사도를 이용한 ontology 와 비교 후 대치
- Pseudo-labeling

---
