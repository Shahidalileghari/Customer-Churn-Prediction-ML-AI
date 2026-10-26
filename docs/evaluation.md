# Evaluation protocol

The training script uses a stratified 80/20 train/test split with random seed 42. A `DummyClassifier` that predicts the most frequent training class provides a baseline. Logistic regression is evaluated on the same held-out examples.

Metrics reported:
- **Accuracy:** overall fraction classified correctly; may mislead on imbalanced targets.
- **Precision:** fraction of positive predictions that are true positives.
- **Recall:** fraction of actual positive cases identified.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking quality across classification thresholds.
- **Confusion matrix:** counts of true/false positives and negatives.

The 0.5 decision threshold is a simple demo default, not a business-optimized threshold. In a real churn use case, tune the threshold using validation data and business costs. Do not report a single random split as proof of generalization; add cross-validation and external validation for serious use.
