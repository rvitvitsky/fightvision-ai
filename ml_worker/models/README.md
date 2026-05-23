# ML weights

Сюда положи обученные веса из ноутбука RASCAR. Файлы:

| Файл                       | Что внутри                                                       |
|----------------------------|-------------------------------------------------------------------|
| `full_pipeline.joblib`     | `{event_model, attr_models, feature_cols, cfg}` (из ячейки 19)    |
| `visual_reranker.joblib`   | `{pca, clf, frame_offsets}` (из ячейки 24)                        |
| `mlp_reranker.pt`          | `{state_dict, feature_cols, scaler, input_dim, hidden_dim}` (cell 26) |

Если файлов нет — pipeline переключится на эвристический fallback:
вероятность удара = нормированная скорость wrist'а.
Это **сильно** менее точно, но удобно для smoke-теста всего стека.

После того как положишь веса — рестарт worker'а не нужен, проверка идёт на каждый job.
