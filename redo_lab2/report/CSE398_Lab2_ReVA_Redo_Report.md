# CSE398 Lab 2 Redo — full ReVA test

- Test set: all 4,000 ReVA test questions; identical IDs for Qwen base, Qwen LoRA, and VILA
- Qwen3-VL base: **2702/4000 (67.55%)**, Wilson 95% CI 66.08–68.98%
- Qwen3-VL LoRA: **2701/4000 (67.53%)**, Wilson 95% CI 66.06–68.96%
- VILA1.5-3B: **1942/4000 (48.55%)**, Wilson 95% CI 47.00–50.10%
- Base vs LoRA: 99 vs 98 discordant; McNemar p = 1.0 (no detectable difference)
- Base vs VILA: 1132 vs 372 discordant; McNemar p ≈ 10⁻⁸⁵
- Weakest type: Temporal Grounding (base 302/640, LoRA 288/640, VILA 196/640)
