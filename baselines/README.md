# Baseline models

The paper compares FARM against:

| Baseline | Type | File |
|---|---|---|
| MLP | Deep learning — dense | `mlp.py` |
| CNN | Deep learning — 1-D convolution | `cnn.py` |
| Transformer | Deep learning — sequence transformer | `transformer.py` |
| Random Forest | Classical ML — sklearn `RandomForestClassifier` | `classical.py` |
| Elastic Net | Classical ML — sklearn `LogisticRegression(penalty='elasticnet')` | `classical.py` |
| VAMPr | Specialized AMR — per-drug logistic regression | Kim et al. 2020, code at https://github.com/jaehyunkim-rutgers/VAMPr |
| ResFinder | Specialized AMR — rule/database | https://bitbucket.org/genomicepidemiology/resfinder/ |
| RGI (CARD) | Specialized AMR — rule/database | https://github.com/arpcard/rgi |

Architectures and training settings for every baseline are summarised in
**Supplementary Table 8** of the paper.

The MLP / CNN / Transformer / RF / Elastic Net baselines share the FARM
training entry-point and are toggled with `--model {mlp,cnn,transformer,rf,enet}`.
VAMPr, ResFinder, and RGI are run from their upstream repositories — see
`scripts/run_specialized_baselines.sh` for the exact commands we used.
