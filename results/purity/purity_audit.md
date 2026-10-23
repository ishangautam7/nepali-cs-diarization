# D1 speaker-purity audit (PREREGISTRATION §5 item 4)

Clips with both embeddings: 20802; assessable under both: 20444; flagged under both: 688 (single-model flags: {'titanet': 983, 'campplus': 1021}).

| language mix | clips | assessable | flagged | rate |
|---|---|---|---|---|
| cs|t0 | 5873 | 5814 | 199 | 0.0342 |
| cs|t1 | 6894 | 6847 | 226 | 0.033 |
| cs|t2 | 6819 | 6756 | 228 | 0.0337 |
| en|t0 | 651 | 584 | 21 | 0.036 |
| ne|t0 | 565 | 443 | 14 | 0.0316 |

Split-half null (1st percentile): titanet 0.92298 (n=3220), campplus 0.94431 (n=3220)
Split-off videos (both embeddings): 34 (single-model: {'titanet': 36, 'campplus': 43}).

| split | clips | flagged | kept | speakers after |
|---|---|---|---|---|
| validation | 2338 | 45 | 2293 | 6 |
| train | 16702 | 606 | 16096 | 39 |
| test | 1762 | 37 | 1725 | 7 |

Closest cross-channel video pairs (descriptive; percentile within the split-half null):
- 4qYik5d_rLw (UCOC6C8_U9PsAs3tQodDDmGw) vs zJYS-XNDP2Q (UCeq9rTo9W93NfJNP9n1eRGw): cos {'titanet': 0.84002, 'campplus': 0.8741}, percentile {'titanet': 0.16, 'campplus': 0.19}
- CCSTJTzx1YA (UCOC6C8_U9PsAs3tQodDDmGw) vs zJYS-XNDP2Q (UCeq9rTo9W93NfJNP9n1eRGw): cos {'titanet': 0.83827, 'campplus': 0.84594}, percentile {'titanet': 0.12, 'campplus': 0.03}
- 8ywUuBnUg0g (UCOC6C8_U9PsAs3tQodDDmGw) vs zJYS-XNDP2Q (UCeq9rTo9W93NfJNP9n1eRGw): cos {'titanet': 0.83144, 'campplus': 0.8782}, percentile {'titanet': 0.06, 'campplus': 0.22}
- DxI6929EuWg (UCOC6C8_U9PsAs3tQodDDmGw) vs zJYS-XNDP2Q (UCeq9rTo9W93NfJNP9n1eRGw): cos {'titanet': 0.815, 'campplus': 0.8354}, percentile {'titanet': 0.06, 'campplus': 0.03}
- 2B_-IR6oQsk (UCAh6gaPtNIBBW_qNss4flcA) vs MfuHAWNvE5w (UCdsTvXwaUcvV88UhqTHynTg): cos {'titanet': 0.64738, 'campplus': 0.65818}, percentile {'titanet': 0.0, 'campplus': 0.0}
