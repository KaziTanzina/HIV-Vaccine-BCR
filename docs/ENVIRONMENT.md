# Computational environment

## Scope

This document describes the observed software environment
and the separately recorded historical AbLang-2 environment.

It is not a tested environment lockfile and does not
establish end-to-end reproducibility.

## Observed pfizer-vaccine environment

Python: 3.11.16

| Package | Observed version |
|---|---|
| ablang2 | 0.1.1 |
| matplotlib | 3.11.2 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| pyarrow | 25.0.0 |
| sadie-antibody | Not found |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| statsmodels | 0.15.0 |
| torch | 2.14.1 |
| transformers | Not found |

## Import resolution

| Module | Observed status |
|---|---|
| ablang2 | Found |
| sadie | Not found |
| sklearn | Found |
| statsmodels | Found |
| torch | Found |
| transformers | Not found |

Missing modules should not be interpreted as proof that
the historical computations were never executed.

## Historical AbLang-2 environment

The saved historical execution log reports:

- Environment: vaccine-ablang2
- Python: 3.11
- PyTorch: 2.5.1
- AbLang-2: 0.1.1
- Inference device: CPU
- Frozen representations: 2,739 x 480

The original input and saved embedding archive were
subsequently verified against the recorded SHA256 hashes.

The standalone embedding-generation source has not
been identified.

## Reproducibility limitations

SADIE and Transformers were not available in the
observed pfizer-vaccine environment.

Historical package versions, external pretrained model
resources, remaining workflow dependencies, and
execution instructions must be resolved or documented
before claiming full computational reproduction.

Do not install arbitrary latest package versions and
describe that environment as reproducing the published
historical workflow.
