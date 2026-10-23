# Full-pipeline check on D1 (EXPLORATORY, post hoc)

Clip joins are **artificial** boundaries: two separately recorded segments are concatenated, so a join also carries a recording discontinuity. A D1 channel stands for one speaker, which is unverified (issue 4). Change window ±0.5 s; DER with a 0.5 s collar (±0.25 s), overlap scored.

## P1 (complete)

Device cuda (torch 2.8.0+cu128); real-time factor 0.008.

| Boundary | n | Change rate, pooled | Change rate, channel mean | Miss rate (cross-speaker) | Split / merge |
|---|---|---|---|---|---|
| switch | 300 | 1.3% | 1.8% | - | 0.0% |
| control | 300 | 0.3% | 0.6% | - | 0.0% |
| join_ss_same | 300 | 1.7% | 4.1% | - | 1.7% |
| join_xs_same | 300 | 60.3% | 59.6% | 39.7% | 37.7% |
| join_xs_cross | 300 | 65.3% | 68.6% | 34.7% | 29.3% |

| Contrast (A - B, per channel) | Channels | Mean | 95% BCa | Sign-flip p (one-sided, two-sided) |
|---|---|---|---|---|
| false_change: switch - control | 17 | 1.2% | -0.8% to 3.6% | 0.1562, 0.3125 |
| false_change: switch - join_ss_same | 17 | -2.3% | -9.3% to 0.6% | 0.7969, 0.4219 |
| false_change: join_ss_same - control | 17 | 3.5% | -0.2% to 10.5% | 0.1562, 0.3125 |
| miss: join_xs_cross - join_xs_same | 17 | -9.0% | -16.9% to -1.6% | 0.9800, 0.0410 |
| split: switch - control | 17 | 0.0% | 0.0% to 0.0% | 1.0000, 1.0000 |

| Files | n | Mean DER | Median DER | Confusion share | Files with >1 speaker found |
|---|---|---|---|---|---|
| join_ss_same | 300 | 10.5% | 3.5% | 0.4% | 10.7% |
| join_xs_cross | 300 | 19.7% | 13.2% | 9.2% | 76.3% |
| join_xs_same | 300 | 23.9% | 15.8% | 12.2% | 69.7% |
| natural | 579 | 7.1% | 1.9% | 0.1% | 5.5% |

## P2 (complete)

Device cuda (torch 2.8.0+cu128); real-time factor 0.015.

| Boundary | n | Change rate, pooled | Change rate, channel mean | Miss rate (cross-speaker) | Split / merge |
|---|---|---|---|---|---|
| switch | 300 | 0.3% | 0.4% | - | 0.0% |
| control | 300 | 0.3% | 0.8% | - | 0.0% |
| join_ss_same | 300 | 3.3% | 6.8% | - | 1.7% |
| join_xs_same | 300 | 94.0% | 95.0% | 6.0% | 3.7% |
| join_xs_cross | 300 | 96.0% | 96.6% | 4.0% | 3.0% |

| Contrast (A - B, per channel) | Channels | Mean | 95% BCa | Sign-flip p (one-sided, two-sided) |
|---|---|---|---|---|
| false_change: switch - control | 17 | -0.4% | -3.4% to 0.8% | 0.7500, 1.0000 |
| false_change: switch - join_ss_same | 17 | -6.4% | -14.7% to -2.0% | 1.0000, 0.0312 |
| false_change: join_ss_same - control | 17 | 6.0% | 2.3% to 12.5% | 0.0156, 0.0312 |
| miss: join_xs_cross - join_xs_same | 17 | -1.5% | -4.2% to 1.1% | 0.8584, 0.2852 |
| split: switch - control | 17 | 0.0% | 0.0% to 0.0% | 1.0000, 1.0000 |

| Files | n | Mean DER | Median DER | Confusion share | Files with >1 speaker found |
|---|---|---|---|---|---|
| join_ss_same | 300 | 10.8% | 3.5% | 0.9% | 10.0% |
| join_xs_cross | 300 | 11.4% | 5.6% | 1.3% | 98.7% |
| join_xs_same | 300 | 13.8% | 6.5% | 1.9% | 99.3% |
| natural | 579 | 6.2% | 1.6% | 0.1% | 2.2% |

