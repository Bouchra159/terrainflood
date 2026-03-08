# TerrainFlood-UQ Results Summary

## Bolivia OOD Ablation

| Variant | IoU | F1 | Precision | Recall | ECE | Mean Variance |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.4082 | 0.5797 | 0.4129 | 0.9728 | 0.4023 | 0.0000 |
| B | 0.4411 | 0.6122 | 0.4529 | 0.9444 | 0.2404 | 0.0000 |
| C | 0.6623 | 0.7968 | 0.7892 | 0.8046 | 0.3699 | 0.0000 |
| D | 0.6902 | 0.8167 | 0.7880 | 0.8475 | 0.3643 | 0.0004 |
| D_plus | 0.6652 | 0.7990 | 0.7044 | 0.9229 | 0.3200 | 0.0045 |
| E | 0.1668 | 0.2859 | 0.1669 | 0.9953 | 0.4931 | 0.0010 |

## Bootstrap 95% CI

| Variant | IoU | CI Lower | CI Upper |
|---|---:|---:|---:|
| A | 0.4082 | 0.2114 | 0.5782 |
| B | 0.4411 | 0.1994 | 0.6580 |
| C | 0.6623 | 0.4497 | 0.7785 |
| D | 0.6819 | 0.4779 | 0.7850 |
| E | 0.1686 | 0.0687 | 0.3018 |

## Threshold Optimization

| Variant | tau* | IoU@tau* | IoU@0.5 |
|---|---:|---:|---:|
| A | 0.80 | 0.5695 | 0.4082 |
| B | 0.77 | 0.5235 | 0.4411 |
| C | 0.51 | 0.6668 | 0.6623 |
| D | 0.50 | 0.6860 | 0.6860 |
| D_plus | 0.52 | 0.6823 | 0.6481 |
| E | 0.64 | 0.1988 | 0.1675 |

## Alpha vs HAND

| Variant | Pearson r vs theory |
|---|---:|
| C | +0.5425 |
| D | -0.8887 |
| D_plus | +0.8121 |
| E | -0.8627 |

## Short interpretation

- Variant D is the best overall Bolivia OOD model.
- Variant C is the strongest non-dropout gated model.
- D_plus increases uncertainty magnitude but trails D on IoU.
- Variant E underperforms strongly in the current temporal-differencing setup.
- Threshold tuning does not change the top-ranked model.
