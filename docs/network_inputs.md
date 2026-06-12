# Network Inputs Overview

This document summarizes the network inputs implemented in the current codebase. It describes the tensors passed to the neural-network filter ansatz, not the constant-filter ansatz.

## Where Inputs Are Built

- 1D network inputs are assembled in `1D/src/funcs_common.py::preprocess_features` and consumed by `SimpleNN`.
- 2D network inputs are assembled in `2D/src/funcs_common.py::preprocess_features` and consumed by `SimpleNN`.
- The `main.py` training path computes `params["num_features"]` with `nn_feature_count(N, params)`, so feature width follows the configured feature variant.
- Some legacy scripts still hard-code the baseline feature count (`2 * N + 4` in 1D or `2 * (N + 1) + 2` in 2D). Prefer `main.py -c ...` for variant-aware runs.

## Common Feature Variant Controls

Both 1D and 2D support the same `feature_variant` values:

| Variant | Input groups used |
| --- | --- |
| `baseline_norm` | normalized baseline groups |
| `log_norm` | log-magnitude versions of the baseline groups, then normalized |
| `baseline_plus_log` | normalized baseline groups concatenated with log-normalized groups |
| `log_material_only` | currently keeps normalized baseline groups and appends material-scale features |
| `no_norm_log` | log-magnitude baseline groups without normalization |

Aliases include `baseline`, `current`, and `paper` for `baseline_norm`; `log` for `log_norm`; `baseline+log` and `plus_log` for `baseline_plus_log`; and `material_log` or `material_only` for `log_material_only`.

Log features use:

```text
log1p(abs(value) / feature_log_scale)
```

then clamp with `feature_log_clip` unless clipping is disabled.

Normalization modes:

| Setting | Meaning |
| --- | --- |
| `sample`, `per_sample`, `batch` | subtract mean and divide by std over spatial cells, per sample and per channel |
| `none`, `identity`, `raw` | leave the feature tensor unchanged |
| `global`, `training`, `train` | use `feature_global_mean` and `feature_global_std` |

Material features are enabled when any of these are true:

- `feature_variant: log_material_only`
- `include_material_scale_features: true`
- `include_material_features: true`
- `include_material_ratios: true`

When enabled, material channels are:

```text
log(sigma_s), log(sigma_t), log(sigma_a)
```

where `sigma_a = max(sigma_t - sigma_s, 0)`. If `include_material_ratios: true`, two more channels are appended:

```text
sigma_s / sigma_t, sigma_a / sigma_t
```

Material features use `material_feature_normalization`, which defaults to `none`.

## 1D Inputs

The 1D NN filter is used for `filter_type in (1, 2)`. At each batch item and spatial cell, the network receives a vector with shape:

```text
[batch, num_x, num_features]
```

The baseline groups, in concatenation order, are:

| Group | Channels | Code quantity | Shape before concat |
| --- | ---: | --- | --- |
| streaming derivative | `N + 1` | `A_Dy` | `[batch, num_x, N + 1]` |
| collision term | `N + 1` | `sigma_t * y_prev` | `[batch, num_x, N + 1]` |
| scattering source | `1` | `sigma_s * y_prev[..., 0]` | `[batch, num_x, 1]` |
| physical source | `1` | `source` | `[batch, num_x, 1]` |

Baseline width:

```text
2 * (N + 1) + 2 = 2N + 4
```

For `filter_type == 1`, all four baseline groups are converted to absolute value before per-sample spatial normalization.

For `filter_type == 2`, the implementation differs only for `A_Dy` and `sigma_t * y_prev`: odd moment channels `1::2` are converted to absolute value, while the other channels keep their sign. Scattering and source are still absolute-valued before normalization.

### 1D Ablations

`ablation_idx` zeroes whole baseline groups before concatenation. The group indices are:

```text
0: A_Dy
1: sigma_t * y_prev
2: sigma_s * y_prev[..., 0]
3: source
```

The current mapping is:

| `ablation_idx` | Groups kept |
| ---: | --- |
| `0` | all groups |
| `1` | `A_Dy` only |
| `2` | `sigma_t * y_prev` only |
| `3` | scattering only |
| `4` | source only |
| `5` | none |
| `6` | all except `A_Dy` |
| `7` | all except `sigma_t * y_prev` |
| `8` | all except scattering |
| `9` | all except source |

The same ablation mask is applied to the log groups when a log variant is selected.

### 1D Feature Counts

Let `B = 2N + 4` and `M = 0`, `3`, or `5` material channels.

| Variant | Feature count |
| --- | ---: |
| `baseline_norm` | `B + M` |
| `log_norm` | `B + M` |
| `no_norm_log` | `B + M` |
| `baseline_plus_log` | `2B + M` |
| `log_material_only` | `B + M` |

Examples:

| N | baseline | baseline + 3 material | baseline + 5 material | baseline_plus_log, no material |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 13 | 15 | 20 |
| 7 | 18 | 21 | 23 | 36 |
| 9 | 22 | 25 | 27 | 44 |

Current `configs/reproduce_all.yaml` uses the paper-style baseline feature settings for 1D. Current `configs/reproduce_1d.yaml` uses `feature_variant: log_material_only` with material scale features and ratios enabled, so it uses `2N + 9` input features on the `main.py` path.

## 2D Inputs

The 2D NN filter is used for `filter_type == 0`. At each batch item and spatial cell, the network receives a vector with shape:

```text
[batch, num_y, num_x, num_features]
```

The 2D implementation first converts moment tensors into rotationally invariant per-degree norms. For each angular degree `ell = 0, ..., N`, the implementation groups the `ell + 1` retained 2D moment channels and takes an L2 norm.

The baseline groups, in concatenation order, are:

| Group | Channels | Code quantity | Shape before concat |
| --- | ---: | --- | --- |
| weighted state norms | `N + 1` | degree-wise norms of `sigma_t * psi_prev` | `[batch, num_y, num_x, N + 1]` |
| derivative norms | `N + 1` | degree-wise norms of `A_dxpsi` and `A_dypsi` | `[batch, num_y, num_x, N + 1]` |
| scattering source | `1` | `sigma_s * psi_prev[..., 0]` | `[batch, num_y, num_x, 1]` |
| physical source | `1` | `source` | `[batch, num_y, num_x, 1]` |

Baseline width:

```text
2 * (N + 1) + 2 = 2N + 4
```

Unlike 1D, the 2D baseline groups are norms or scalar non-moment fields, then per-sample spatially normalized over `[num_y, num_x]`.

### 2D Feature Counts

The feature-count formulas are the same as 1D, with `B = 2N + 4` and optional `M = 0`, `3`, or `5` material channels.

Examples:

| N | baseline | baseline + 3 material | baseline + 5 material | baseline_plus_log, no material |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 13 | 15 | 20 |
| 5 | 14 | 17 | 19 | 28 |
| 7 | 18 | 21 | 23 | 36 |
| 9 | 22 | 25 | 27 | 44 |

Current 2D reproduction configs use `feature_variant: baseline_norm` and do not enable material scale or ratio features.

## Constant Filter Inputs

Constant-filter modes do not consume feature vectors:

- 1D `filter_type == 3` uses `SimpleNN_const`, a single trainable scalar parameter.
- 2D `filter_type == 1` uses `SimpleNN_const`, a single trainable scalar parameter.
- 2D `filter_type == 2` uses a supplied constant value rather than a network module.

## Output

Both neural-network filters output one nonnegative scalar filter strength per spatial cell:

- 1D: `[batch, num_x]`
- 2D: `[batch, num_y, num_x]`

The scalar is applied as `sigma_f * state * filter_coeffs` inside the FPN update.
