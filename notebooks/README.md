# Notebook guidance

Keep notebooks focused on exploratory analysis and visual communication. Move reusable transformations, training, and inference logic into `src/applied_ml/` so it can be tested and run from the command line.

Suggested notebook:
1. Inspect data shape, data types, missingness, and target balance.
2. Plot target distribution and numeric feature distributions.
3. Compare features across target classes.
4. Record hypotheses and findings.
5. Confirm any transformation intended for production is implemented in the tested pipeline, not only in the notebook.
