# Decision log

Dated decisions taken before the pre-registration freeze. Changes after the freeze go to `DEVIATIONS.md`.

## 2026-10-18 (Ishan)

1. **D2 size.** Record the full set: about 10–20 h of unscripted 2–3-person Nepali–English conversations with wired lavalier mics, enough for the confirmatory non-inferiority test H5. The exact size is fixed by simulation on the pilot hour. Superseded the same day by decision 6.
2. **Annotation.** Hired bilingual annotators do all annotation, including the D1 listening checks (§4.5–4.6), with a second annotator redoing 20% for agreement.
3. **Ethics.** Consent and the ethics statement go through Ishan's university or institute. Superseded the same day by decision 5.
4. **D1-LF.** The long-form audio of the 161 source videos was to be obtained for private, non-commercial analysis only: audio only, stored privately, never redistributed, deleted at the end of the project. The download failed (see the end of this log), so D1-LF does not exist.
5. **Permissions.** Ishan states that he holds the necessary permissions, so work starts now without waiting for a committee review. Every participant still gives informed written consent before recording, using the forms in `planning/ethics/`. The approving body and approval reference go into the paper's ethics statement.

   Note added 2026-10-23: no recordings were made and no participants were recruited, so no approval was sought; see `planning/ethics/README.md`.
6. **Conversation data (replaces decision 1).** Ishan asked to use public data instead of recording our own conversations. He can annotate a small amount himself.
   - DISPLACE 2024 becomes the primary DS-DL test for H5. It has human speaker and language references, but its conversations are in other Indian languages with English, not Nepali.
   - Public Nepali podcasts and talk shows (D2-P) are annotated as far as capacity allows and support H5 on Nepali. Their audio comes from public RSS feeds (decision 8) and is kept under the same conditions as decision 4.
   - The Nepali conversational audio in IndicVoices is audited as a possible extra source.
   - Recording our own conversations (D2-R) is deferred, not deleted; its protocol and forms stay in `planning/`.

   Consequence: the "no extra missed turns" claim is confirmed on Indian-language–English conversation, and only supported on Nepali unless D2-P reaches the simulated size.
7. **Non-commercial use.** Ishan confirmed on 2026-10-18 that the project, including its funding and intended outputs, is entirely non-commercial. Use of D1, the DiariZen weights and the alignment model remains subject to their licence terms. (First recorded in the box copy of this file; merged here.)
8. **Podcasts.**
   - Ishan approved the RSS route for D2-P: episodes come from podcasts' own public RSS feeds, found through the Apple Podcasts directory.
9. **DISPLACE-M.** Ishan reports that the DISPLACE 2026 (DISPLACE-M) organisers agreed to its use for this research. DISPLACE-M is Hindi–English clinical conversation (40 h dev + 15 h eval), and it becomes the primary H5 dataset unless DISPLACE 2024 access is also granted. Downloading needs Hugging Face access approval for Ishan's account and the archive password, which Ishan places on the box himself.
10. **Operational rules for switch points and controls** (2026-10-18, before the pilot; no outcome statistic had been computed). These fill gaps in PLAN.md §2, implemented in `csdiar/data/boundaries.py` and `csdiar/data/vad.py`.
    - **VAD:** Silero VAD 6.2.3 (MIT), 32 ms frames, speech threshold 0.5. It is chosen because it belongs to no system under test; pyannote's segmentation would make pause matching depend on P1.
    - **Pause** at a word boundary: the longest run of non-speech frames whose centres lie between the middles of the two words (ties go to the run nearest the aligned boundary). With no such frame, the point is the middle of the aligned gap and the pause bin is "none". At L = 1.0 s, 80% of train switch points have no detectable pause.
    - **Switch-point exclusion:** switch points with a neutral word between the two runs are excluded, because their timing is ambiguous (2,366 train, 828 validation, 407 test). They are counted in the summary.
    - **Window contents:** no analysis window, at a switch point or a control, may contain a neutral or loan word. This is stricter than v0.5 for switch points, so that both sides of the comparison are treated the same. An unaligned word next to a window counts as inside it.
    - **Window placement:**
      - A window's outer edge may lie at most 0.1 s outside its run's aligned span (alignment tolerance).
      - A switch window's inner edge sits at the pause.
      - Every window must lie inside the clip.
    - **Low-alignment clips:** clips whose mean word probability is below the 5th percentile of train clips (0.369) are flagged and excluded from every family: 838 train, 17 validation and 192 test clips. They stay in the files for a sensitivity analysis.
    - **Controls:** boundaries between consecutive language words of one run whose region [p0 − W, p1 + W] obeys the window rules.
      - Embedding families: W = L.
      - Detection family `det`: W = c = 0.5 s, the primary collar.
      - Sensitivity family `det_w1`: W = 1.0 s.
      A control region wider than the collar would leave almost no controls in this heavily switched corpus. Any switch just outside a control's region biases results towards the null.
    - **Matching, embedding families:** up to 3 controls per point, without replacement and in seeded random order. Matching is exact on:
      - video (same clip preferred, never across videos);
      - pause bin;
      - position bin (2 s from the clip start, last bin ≥ 10 s);
      - context length (shorter same-language side < 2L or ≥ 2L).
    - **Matching, detection families:** the primary comparison is cell standardisation. Each switch point is compared with every eligible control in its video × pause bin × position cell, and controls may be reused. 1:3 matching left 78% of train points without a control, while cell coverage is 99.6%. The 1:3 matches are kept as a sensitivity analysis. Run-length strata (< 0.5, 0.5–1.0, ≥ 1.0 s) are reported for switch points but cannot be matched, because a clean control cannot have a short side.
    - **Counts before the loan lexicon and name list exist** (counts only, no outcomes):

      | Family | Train | Validation | Test |
      |---|---|---|---|
      | Embedding, L = 1.0 s | 4,130 | 492 | 130 |
      | Embedding, L = 0.5 s | 26,565 | 3,899 | 824 |
      | Embedding, L = 1.5 s | 1,000 | 89 | 35 |
      | Detection | 148,944 | 21,975 | 6,846 |

      The final counts follow the lexicon freeze.
11. **E3 source, licence and loading** (2026-10-18).
    - **Model:** UniSpeech's speaker-verification model "WavLM large, Fix pre-train: No" (VoxCeleb1-O EER 0.431%).
    - **Checkpoint:** taken from the authors' Google Drive link and pinned by SHA-256 (51f07e3b…f94b).
    - **Code:** the UniSpeech `ECAPA_TDNN_SMALL` code is vendored in `third_party/unispeech`. It is CC BY-SA 3.0, and that folder keeps that licence.
    - **WavLM-Large upstream:** built from the pinned s3prl 0.4.18 package, not from torch.hub's moving GitHub head.
    - **Strict loading:** the upstream script loads with `strict=False`. Our loader requires every model weight to come from the checkpoint, and only the training-time AAM head is left unused. The fine-tuned WavLM is therefore really in use.
    - **Environment:** E3 runs in its own environment (`.venv-e3`, `ops/env_e3.sh`).
    - **Speed and memory:** about 15 GB of VRAM at 512 one-second windows per batch.
    - **Other GPU users:** the embedding launcher sets a 24 GB reserve in `~/run/gpu_reserve_gb` while it runs.
12. **Loan lexicon, name list and letter rule, v1, frozen 2026-10-18** (before the pilot; built from text only). Files are in `planning/lexicon/`.
    - **`loans.txt`:** 52 entries, sha256 38be59bb…7c26.
      - Contents: English words written in Devanagari, plus their inflected forms found in D1.
      - How they were found: candidates came from consonant-skeleton matching of uroman romanisations against English words (`scripts/lexicon_candidates.py`) and from Wiktionary's "Nepali terms borrowed from English". Every entry was checked by hand.
      - Excluded homographs: जुन ("which"), बस ("sit"/"enough") and चिज ("thing").
      - Why the list is short: the corpus writes English in Latin, so loans in Devanagari are rare (126 tokens).
    - **`names.txt`:** 144 entries, sha256 eb474023…efc3.
      - Contents: proper names whose script would mis-assign their language. These are Latin-script names of people, places, organisations, products, programming languages, days and months, plus Devanagari spellings of foreign names.
      - Excluded: Nepali names in Devanagari stay `ne`, and acronyms stay `en`.
      - How they were chosen: every capitalised Latin type with ≥ 2 occurrences was checked by hand.
    - **Letter rule** (`letters_neutral` in `csdiar.data.tags`): a single Latin letter other than the article "a" is `neutral`. In these lectures such letters are almost always variables.
    - **Effect on D1:** 2.02% of the 541,899 words are re-tagged. That is 10,820 en→neutral (mostly single letters), 126 ne→loan and 4 ne→neutral.
    - **Counts with v1** (counts only, no outcomes):

      | Family | Train | Validation | Test |
      |---|---|---|---|
      | Embedding, L = 1.0 s | 3,842 | 443 | 114 |
      | Detection | 140,641 | 20,388 | 5,585 |

    - **Residual tag noise:** low-frequency names and loans are not curated. The noise this leaves is measured by the listening check and on the 777 human-verified clips. Any v2 would have to be logged here before the M2a freeze.
13. **Pilot v1** (2026-10-18; 3 validation channels only; baselines only; `csdiar/analysis/pilot.py`; results in `results/pilot/pilot_v1.{json,md}`). No mitigation was run and no other split's outcomes were computed.
    - **Definitions fixed for the freeze:**
      - H1 dz is point-level: the grand mean of channel means divided by the pooled within-channel SD of the per-set Δcos.
      - The target-score SD used for standardising is the SD of control cosines (same speaker, same lag).
      - P0 uses the dense L = 1.0 s grid: d(k) = 1 − cos(e[k−10], e[k]) at t = 0.1k.
      - P0 peaks are local maxima within ±0.5 s.
      - A hypothesized change is a peak above τ within ±c of the point.
      - P0's AUC is cell-standardised (video × pause × position), weighted by the switch points in each cell.
    - **P0 τ (frozen):** E1 0.903, E2 0.851, E3 0.916, which gives FAR_CTRL = 10% on the pilot controls.
    - **H1 pilot (L = 1.0 s, 397 matched sets):** Δcos is positive in every channel for every model.

      | Model | Δcos | dz | Δcos in target-score SD |
      |---|---|---|---|
      | E1 | +0.042 | 0.32 | 0.36 |
      | E2 | +0.033 | 0.25 | 0.28 |
      | E3 | +0.025 | 0.19 | 0.21 |

      - The effect grows with window length. At L = 0.5 s dz is 0.06–0.12; at L = 1.5 s it is 0.44–0.67, but with only 68 sets.
      - Between-channel ICC is 0.003–0.014.
    - **P0 pilot:** the pooled cell-standardised AUC is 0.49 for every model. FAR at switch points varies by stratum:
      - < 0.5 s: 0.07;
      - 0.5–1.0 s: 0.07–0.08;
      - ≥ 1.0 s: 0.15–0.18;
      - controls: 0.10.

      Short insertions, which make up 78% of detection-family points, therefore show no excess, and long runs show one. Primary cells stay as pre-registered (strata pooled; strata reported). The stratum pattern is exploratory, because changing the primary cell after seeing pilot effects would bias the test.
    - **Power** for 17 confirmatory channels, Bonferroni α = 0.05/3, between-channel SD at its upper 80% bound, real per-channel counts:

      | Test | Effect | E1 | E2 | E3 |
      |---|---|---|---|---|
      | H1, P(p < α) | dz = 0.2 | 0.97 | 0.99 | 0.76 |
      | H1, P(p < α) | dz = 0.3 | 1.00 | 1.00 | 0.98 |
      | H2 (P0), P(p < α) | AUC = 0.55 | 1.00 | 0.91 | 0.98 |

      The probability that the estimate also clears the smallest effect is about 0.5 when the true effect equals it, as expected. The design stays as planned.
    - **Not in this pilot:** the plan's pilot also names "the first recorded D2 hour", but D2-R is deferred and no D2-P audio has been annotated yet. That part is pending.
14. **H3 definitions and pilot** (2026-10-18; validation only; `csdiar/data/windows.py`, `csdiar/analysis/h3.py`; results in `results/pilot/h3_v1.{json,md}`).
    - **Window units:** clean monolingual 1 s grid windows (`csdiar/data/windows.py`).
      - Every touched word is of one language; no neutral or loan word; at least 2 words.
      - At least 60% of VAD frames are speech.
      - Low-alignment clips are skipped.
      - Windows are thinned so their starts are at least 1 s apart.
      - Counts: train 66,463 (en 17,703), validation 9,156 (en 1,685), test 6,920 (en 2,152).
    - **Primary H3 trials:** same session (same video), with starts ≥ 2 s apart so the windows are non-adjacent.
      - Lag bins: [2,4), [4,8), [8,16), [16,64) and ≥ 64 s.
      - Per channel and bin, shift = mean(cross) − mean(mono-ne mean, mono-en mean). The two monolingual conditions get equal weight, so language-specific score levels cannot pass for a cross-language effect.
      - A bin counts only if it has ≥ 20 cross trials and ≥ 20 trials in at least one monolingual condition.
      - The channel shift is the cross-weighted mean over bins.
      - Standardised by the target-score SD (SD of same-session monolingual scores).
      - Up to 40,000 random pairs are drawn per video (seeded).
    - **Secondary:** the cross-session shift (same channel, different video), and EER and minDCF (P_target = 0.05, cross vs mono). For these, nontargets are different-channel pairs with the same language pattern.
    - **Pilot result:** every channel and every model shows a downward shift.

      | Model | Shift | Shift in target SD | EER cross / mono | minDCF cross / mono |
      |---|---|---|---|---|
      | E1 | −0.026 | −0.24 | 18.9% / 16.6% | 0.86 / 0.79 |
      | E2 | −0.023 | −0.21 | 12.4% / 9.9% | 0.63 / 0.54 |
      | E3 | −0.023 | −0.20 | 17.7% / 14.4% | 0.78 / 0.68 |

      - The cross-session shift is also negative everywhere (−0.007 to −0.024).
      - The between-channel SD of the shift is 0.002–0.006.
      - The pilot estimate sits at the smallest effect of interest (0.2 target SD), so H3's power at the SESOI is about 0.5 by construction. Its precision is high given the small between-channel SD.
15. **M2 λ(t) check before any tuning** (2026-10-18; validation only; `csdiar/analysis/m2_lambda.py`; results in `results/pilot/m2_lambda_v1.{json,md}`).
    - **Definition:** λ(t) = Jensen–Shannon divergence (base 2) between the renormalised (p_ne, p_en) of the 1 s LID windows [t−1, t] and [t, t+1].
    - **AUC against cell-matched controls:**
      - Detection family: 0.58 overall; 0.57 for runs under 0.5 s; 0.61 for 0.5–1.0 s; 0.71 for runs of at least 1.0 s.
      - Embedding family (L = 1.0 s): 0.72.
    - **λ also tracks pauses somewhat:** phrase-boundary controls score above other controls (AUC 0.57–0.59).
    - **Reading:** λ is a usable but weak language-change signal. It is weakest for single-word insertions, because 1 s LID windows cannot resolve one word. M2's oracle-λ variant (built from reference tags) bounds what a better LID could give.
    - **Not yet checked:** the DS-SL comparison needs conversation data.
    - **No change** to M2's design follows from this check.
16. **Revisions after the independent pilot review** (2026-10-18; 4 major and 8 minor findings). Made before the freeze; no confirmatory outcome had been computed. Boundaries were regenerated as `lex_v1b`.
    - **Symmetric detection rule (M1):** a detection-family switch point now needs the same cleanliness as its controls. Its region t ± W may contain no neutral or loan word, with W = 0.5 s for `det` and 1.0 s for `det_w1`. Both languages are allowed in the region.
    - **Primary P0 cell (M2; revises decision 13):** the primary cell is now `p0L1`.
      - **Switch points:** both runs ≥ 1.0 s, and P0's own windows [t − 1, t] and [t, t + 1] each lie inside one clean run.
      - **Controls:** the `det_w1` controls, which are clean over ±1.0 s inside one run.
      - **Statistic:** the cell-standardised AUC of d(t).
      - **τ:** now calibrated on the `det_w1` controls.
      - **Secondary and sensitivity:** the pooled detection family (`det`) is secondary, and `det_w1` is a sensitivity analysis.
      - **Rationale (construct):** a two-window detector with 1 s windows cannot register a sub-second insertion as a two-sided language change. Its ±1 s field must also be clean at controls, or switches inside it contaminate them.
      - **Disclosure:** the pilot showed that the pooled cell sits near 0.5. Choosing the estimand on pilot data does not bias the confirmatory test, because the 17 confirmatory channels are disjoint from the 3 pilot channels.
    - **Stratum statistics (M3):** every stratum statistic is cell-standardised. AUC uses the cell's controls. FAR is the switch point's hit minus the hit rate of its cell's controls.
    - **Outcome rules (M4):**
      - The primary test is the one-sided sign-flip test in the predicted direction; the two-sided p is also reported.
      - Equivalence (TOST) bounds are ± the smallest effect of interest: H1 dz ± 0.2; H2-P0 AUC 0.5 ± 0.05; P1/P2 RR within [0.8, 1.25]; H3 ± 0.2 target SD.
      - Categories: supported; nonzero but trivial; equivalent; inconclusive; reversed. "Reversed" means a two-sided p < .05 in the opposite direction, and it is reported as such.
      - For H1, dz is the only smallest-effect metric; Δcos in target-SD units is descriptive.
    - **Pause search (m1):** limited to the aligned gap ± 0.1 s, and never past the words' middles.
    - **Dropped points (m2, m6):** every dropped point is counted, whether near a clip edge (no P0 window), unmatched (H1) or without a cell control (P0). Match levels are reported per family.
    - **Power (m3):** α = 0.05/5 for H2. H1 counts are matched sets; H2 counts are `p0L1` points that have cell controls. The AUC standard error ignores dependence within a clip, so it is optimistic.
    - **Sensitivity analyses (m4, m5):** these are pre-registered:
      - "I" and capital "A" re-tagged as English;
      - points excluded when a boundary word's alignment score is below the 5th percentile of train word scores;
      - the purity-verified subset;
      - low-alignment clips included.

      A descriptive check compares the CMI of low-alignment clips with the CMI of the other clips. Result:
      - Low-alignment clips have slightly lower CMI (train mean 25.8 vs 28.9).
      - They over-represent monolingual clips: en 7.5% vs 1.8%, ne 6.9% vs 1.4%.
      - So the exclusion does not preferentially remove heavily code-switched clips.
    - **Hygiene (m7, m8):**
      - At the freeze, checksums of all confirmatory feature files are logged.
      - No confirmatory cosine is computed before the freeze.
      - Unit tests now cover `stratified_auc`, `peak_strength` and `signflip_p`.
    - **Deferred to the M2b freeze:** the D2 pilot hour (no annotated conversation yet) and H5's sample-size simulation (it needs DISPLACE dev).
    - **Counts with `lex_v1b`** (switch points / controls):

      | Family | Train | Validation | Test |
      |---|---|---|---|
      | `p0L1` | 3,954 / 33,562 | 446 / 3,240 | 115 / 6,735 |
      | `det` | 129,792 / 93,946 | 18,091 / 11,386 | 4,722 / 12,095 |
      | `emb1` | 4,000 / 33,562 | 451 / 3,240 | 121 / 6,735 |
17. **Pilot v2, after the review fixes** (2026-10-18; validation only; `lex_v1b`; results in `results/pilot/pilot_v2.{json,md}` and `m2_lambda_v2.*`). It replaces pilot v1's τ and numbers.
    - **P0 τ (frozen; τ is calibrated on the `det_w1` controls):** E1 0.917, E2 0.863, E3 0.929.
    - **H1 (L = 1.0 s, 404 matched sets, 47 unmatched):**

      | Model | Δcos | dz | Between-channel SD | Within-channel SD |
      |---|---|---|---|---|
      | E1 | +0.044 | 0.35 | 0.007 | 0.13 |
      | E2 | +0.040 | 0.30 | 0.017 | 0.13 |
      | E3 | +0.030 | 0.24 | 0.008 | 0.13 |

    - **Primary P0 cell `p0L1`:** 446 switch points scored, 8 without cell controls. The AUC is above 0.5 in every channel.

      | Model | AUC (channel range) | Cell-standardised ΔFAR |
      |---|---|---|
      | E1 | 0.615 (0.603–0.626) | +2.9 pp |
      | E2 | 0.586 (0.573–0.601) | +2.5 pp |
      | E3 | 0.592 (0.572–0.607) | +3.0 pp |

    - **Secondary pooled `det`:** AUC 0.49 for every model. Cell-standardised strata:
      - < 0.5 s: AUC 0.48, ΔFAR about −1 pp;
      - 0.5–1.0 s: AUC 0.51–0.52;
      - ≥ 1.0 s: AUC 0.59–0.61, ΔFAR +3.5 to +4.9 pp.
    - **Sensitivity `det_w1`:** AUC 0.49–0.50.
    - **Dropped points:** 1,868 switch points and 913 controls fall within 1 s of a clip edge, so P0 has no window there.
    - **Power** for the 17 confirmatory channels, with their counts (3,695 H1 matched sets; 3,962 `p0L1` points with cell controls):

      | Test | Effect | α | E1 | E2 | E3 |
      |---|---|---|---|---|---|
      | H1 | dz = 0.2 | 0.05/3 | 1.00 | 0.68 | 0.99 |
      | H1 | dz = 0.3 | 0.05/3 | 1.00 | 0.96 | 1.00 |
      | P0 | AUC = 0.55 | 0.05/5 | 0.76 | 0.76 | 0.70 |
      | P0 | AUC = 0.60 | 0.05/5 | 1.00 | 1.00 | 0.99 |

      E2's lower H1 power comes from its larger pilot between-channel SD, which is used at its upper 80% bound.
    - **M2 λ:** AUC 0.58 for detection-family switch points (0.72 for ≥ 1.0 s runs), and 0.72 at L = 1.0 s. Phrase-boundary controls against other controls give 0.56–0.58.

18. **Acoustic covariates and the attribution procedure** (2026-10-18; before the freeze; features only, computed on all channels, no outcome used). Code: `csdiar/data/acoustic.py`, `csdiar/analysis/attribution.py`, `csdiar/analysis/glmm.R`. Covariates are in `~/data/d1/acoustic/lex_v1b/`.
    - **Covariates**, per window, for the H1 windows and the P0 windows: F0 (semitones, per-channel Praat range), energy (dB over speech samples), articulation rate (syllable nuclei per second of speech), spectral tilt (dB per octave, 125 Hz–5 kHz) and an SNR proxy (window energy minus the side's 10th-percentile 10 ms level). They enter the models as |right − left|, z-scored. Definitions are in PREREGISTRATION.md §6.
    - **Deviations from PLAN.md §2:**
      - Speaking rate comes from the acoustics, not from the alignment. Syllable or letter counts from the transcript differ systematically between romanised Devanagari and English spelling, so they would confound the covariate with the language change itself.
      - SNR is a local proxy, and C50 is not estimated. The plan's source for both, Brouhaha (`pyannote/brouhaha`), is gated, and the box's Hugging Face login was refused (403) on 2026-10-18. If Ishan accepts its terms, a repeat with Brouhaha SNR and C50 is a pre-registered sensitivity analysis.
    - **Mixed models:** R 4.5.3 with lme4 2.0-6 from conda-forge (`ops/env_r.sh`; exact builds in `ops/locks/r-glmm.explicit.txt`). H1 uses a linear mixed model of the window cosine; H2-P0 uses a logistic GLMM of the false alarm at the frozen τ. The random effects and the singular-fit fallback order follow PLAN.md §7. The cell (video × pause bin × position bin) takes the place of the matched set for P0, because the detection family is compared by cell, not by 1:3 matching (decision 10).
    - **Grading:** "language-attributable" is an intersection–union rule (each condition at α = .05). It needs the primary test, the covariate-adjusted model, the pause > 150 ms phrase-boundary contrast and H3 for the same embedding all significant in the predicted direction.
    - **Dry run on the 3 pilot channels** (validation only):
      - **Random effects:** the random slope was singular for H1, and every structure was singular for P0. This is expected with 3 speakers.
      - **H1:** the covariates changed the switch coefficient from −0.041 to −0.039 (E1).
      - **P0:** they changed the switch coefficient from +0.93 to +0.83 log-odds (E1) and from +0.62 to +0.50 (E3).
      - **Strongest covariate:** tilt change (+0.24 to +0.56 log-odds per SD).
19. **Freeze scope and the confirmatory channel set** (2026-10-18).
    - **Staged freezes.** PLAN.md §9 puts H7 and the D1 part of H4 into M2a. They are instead frozen in their own supplements, each before any of its outcomes is computed:
      - **H7:** the D4 tagging rules are not yet fixed. MUCS transcripts write many English words in Devanagari or Bengali script, for example "ऑफिस" (office), so a transliteration lexicon is needed first.
      - **H4:** M1–M3 are not yet implemented.
      - **Why:** freezing either one now would mean a later deviation.
      - **D4 features** that do not depend on the tags (ingest, alignment, VAD, dense embeddings) may be computed before the supplement. No D4 outcome statistic is computed before it.
    - **Confirmatory channels for E-A:** the 17 non-pilot channels, as PLAN.md §6 and §7 say (2^17 sign patterns). PLAN.md §7's sentence that the 2-channel test split is descriptive only applies to analyses of that split on its own (E-B iii). The primary analysis repeated without the 2 test channels is a pre-registered sensitivity analysis.

20. **Freeze review, round 1** (2026-10-18; before the freeze; no confirmatory outcome computed). A single broad review ran out of its 280 s budget without a report. It was split into two focused reviews:
    - code: 0 major, 12 minor, 3 of them flagged as possibly major;
    - design: 3 major, 8 minor.

    All were addressed.
    - **Outcome categories (design M1):** PREREGISTRATION.md §4 now states the precedence. It says the smallest effect is judged on the point estimate, and it names the estimand of each interval.
    - **Which channels enter a test (design M2):**
      - **Primary tests:** a channel needs at least 10 units (H1 matched sets; switch points with cell controls). H3 keeps its lag-bin trial rule.
      - **Phrase-boundary contrasts:** a channel needs at least 5 units.
      - **Reporting:** excluded channels are listed, and the tests run over the k included channels.
      - **How the minimums were set:** from counts only. All 17 channels have at least 13 H1 sets and 13 eligible `p0L1` points; about 14 channels qualify for the contrasts.
      - **Mixed models:** they use the primary test's channels.
    - **Denominators (design M3):**
      - **dz:** the cluster-level form from the pilot: the unweighted mean of channel means over the pooled within-channel SD, with weights n_c − 1. It is recomputed in each bootstrap resample. PLAN.md §2's plain dz is reported as secondary.
      - **H3's target-score SD:** a single SD of all same-session monolingual target scores, computed once and held fixed in the bootstrap.
    - **Code:**
      - **Freeze guard:** it also checks that every input path is the frozen one. τ must round to the pre-registered values.
      - **attribution.py:** it refuses a confirm JSON from another mode or freeze, and it uses NA-safe lookups. It checks the pause-bin labels, and a contrast needs at least 2 channels to be assessable.
      - **confirm.py:** it writes the primary results before the sensitivity blocks run. It checks the needed columns up front, and asserts that no low-alignment point is eligible (verified: none is, in any split).
      - **BCa:** it returns NaN when there are fewer than 2 clusters or no finite replicates.
      - **glmm.R:** it guards single-level factors, catches fit errors, and refits with Nelder–Mead after a convergence warning. It counts a structure only when the fit converges and is not singular, and it reports counts by condition.
      - **New tests:** BCa against `scipy.stats.bootstrap`, the mirrored categories for H3, the precedence when a result is both trivial and equivalent, and sign-flip ties.
    - **Text:**
      - **Sources and roles:** the header now cites decisions 10–20, and `det_w1`'s role is reconciled.
      - **Holm across stages:** the M2a P0 verdict is final, because P1 and P2 at p = 1 can only make it conservative.
      - **Power and disclosure:** the power table and the post-pilot choice of the primary P0 cell, with the pilot numbers, are now in the pre-registration.
      - **Thresholds and missing prerequisites:** the alignment-sensitivity threshold is stated as 0.093. If the purity audit or the listening check never happens, it is reported as not done.
    - **Purity audit:** its subset rule (two non-tested embeddings, robust outlier flags, split-off videos) is now fixed in PREREGISTRATION.md §5. That way the subset cannot be tuned after the confirmatory results are seen.

21. **Freeze review, round 2** (2026-10-18; before the freeze; no confirmatory outcome computed). Code review: 0 major, 6 minor. Design review: 3 major, 6 minor.
    - **Criterion (b) is now a channel-level test (design M1).**
      - **The problem:** a Wald z from a mixed model without the random slope ignores between-speaker variation in the effect. That makes it anti-conservative, and so it would break the intersection–union rule. The pilot suggests the slope will be dropped.
      - **The fix:** within each channel, the condition coefficient is now estimated by within-set least squares with the five covariates. Those per-channel coefficients get an exact one-sided sign-flip over channels.
      - **What happens to lme4:** the models stay as a descriptive summary (PLAN.md §7's sensitivity model), with cumulative fallbacks and robust coefficient extraction. They no longer enter the grade, and a model failure is recorded rather than aborting the run.
      - **This deviates from PLAN.md §7,** which named the GLMM as the sensitivity model. The GLMM is still reported.
    - **Grading is fully determined (design M2):**
      - **Assessability:** (b) and (c) need at least 5 channels, since with k ≤ 4 an exact one-sided sign-flip cannot reach .05. With fewer, the criterion is "not assessable" and the grade cannot be language-attributable.
      - **Scope:** no claim is made across embeddings.
    - **H3 is fully specified (design M3):** the bin rule (≥ 20 cross trials and at least one monolingual language with ≥ 20), the reference (the mean of the qualifying monolingual means), the target-score SD (all same-session monolingual trials with lag ≥ 2 s), the lag from source-video timestamps, and the seeded pair sampling (up to 40,000 per video).
    - **Minor fixes:**
      - **Pre-registration text:**
        - the equivalence bounds of each hypothesis;
        - the Holm family for the two-sided p;
        - the power caveat for H1 with E2;
        - what counts as an H1 unit (a set with at least one control);
        - the F0 first-pass sampling seed and the SNR edge rule;
        - the H1 secondary is Δcos in control-cosine SD units, as the code computes it.
      - **attribution.py:** it refuses low-alignment eligible points, empty video IDs, empty rows and empty contrasts.
      - **confirm.py:** H3 lists its excluded channels, and the alignment threshold must equal the pre-registered 0.093.
      - **Tests:** new tests cover pooled SD, dz, plain dz, the inclusion split, the dry-run guard and the channel-level adjusted effect.

22. **Freeze review, round 3** (2026-10-18). Code: 0 major, 6 minor. Design: 0 major, 5 minor. Both say nothing blocks the freeze. Fixed before freezing:
    - **Empty contrasts:** an empty phrase-boundary contrast is flagged "not assessable" instead of stopping the run. The code uses MIN_CHANNELS = 5 for both (b) and (c), as decision 21 states.
    - **Excluded channels:** each test now lists every analysed channel that lacks enough units, including channels with none.
    - **BCa:** it reports how many bootstrap replicates were dropped.
    - **Tests:** a test confirms that 17 channels take the exact sign-flip branch, and another runs BCa with the dz ratio statistic.
    - **The P0 score and cell weighting:** PREREGISTRATION.md now states that the AUC uses d(t) at the point (peak strength only for the false alarm at τ), and that each cell is weighted by its number of switch points.
    - **The H1 secondary unit:** it is named as PLAN.md §2 does, the target-trial score SD. It is estimated from the control-window cosines, which are same-speaker pairs, so this is not a deviation.
    - **New sensitivity item 8:** PLAN.md §2's loan re-inclusion analysis was missing from §5 and is now there.
    - **The alignment sensitivity:** §5 states that a point whose boundary word has no score is removed.
    - **Code written after the freeze:** §5 and §7 say what the freeze record covers. Sensitivity code written after the freeze is committed, and its hash logged in DEVIATIONS.md, before it runs on confirmatory channels.

23. **Confirmatory D1 results, M2a** (2026-10-18; one run with the frozen code; `results/confirm/`; summary tables in `results/confirm/summary_m2a.md`).
    - **The run:** `confirm.py`, then `attribution.py`, with the commands in the freeze record. The attribution step ran after deviation 1.
    - **Channels and signs:** all 17 confirmatory channels entered every primary test.
      - Every channel's estimate had the predicted sign, with one exception: H3 with E2, where one channel (`…Mhog`) shifted by +0.016 SD.
      - The exact one-sided sign-flip p is therefore 2^−17 = 7.6 × 10^−6 everywhere except H3 E2 (2^−16). Every Holm p is ≤ 4 × 10^−5.

    | H | Model | Estimate (BCa CI) | Outcome |
    |---|---|---|---|
    | H1 | E1 | dz 0.271 (0.203–0.348) | supported |
    | H1 | E2 | dz 0.194 (0.135–0.266) | nonzero but trivial |
    | H1 | E3 | dz 0.254 (0.198–0.305) | supported |
    | H2-P0 | E1 | AUC 0.584 (0.559–0.618) | supported |
    | H2-P0 | E2 | AUC 0.573 (0.547–0.604) | supported |
    | H2-P0 | E3 | AUC 0.582 (0.564–0.601) | supported |
    | H3 | E1 | shift −0.215 SD (−0.240 to −0.175) | supported |
    | H3 | E2 | shift −0.173 SD (−0.204 to −0.124) | nonzero but trivial |
    | H3 | E3 | shift −0.203 SD (−0.239 to −0.152) | supported |

    - **Next to H2-P0, as PREREGISTRATION.md §7 requires:** the detection cells PLAN.md §7 first named go the other way.

      | Model | Pooled `det` AUC (99% CI) | `det_w1` AUC (99% CI) |
      |---|---|---|
      | E1 | 0.488 (0.475–0.504) | 0.480 (0.455–0.502) |
      | E2 | 0.485 (0.472–0.4996) | 0.477 (0.455–0.499) |
      | E3 | 0.493 (0.478–0.505) | 0.486 (0.462–0.505) |

      - For E2 both intervals exclude 0.5 from below.
      - **By stratum** (the shorter run's length): 0.48 below 0.5 s, 0.51–0.52 for 0.5–1.0 s, and 0.59–0.60 at 1 s or more.
      - **What this means:** under the pooled cell, switch points score slightly lower than their controls, and the excess appears only between runs of at least 1 s.
      - **One explanation, not tested:** a detector with 1 s windows registers a switch only when each run fills its window.
      - **Disclosure:** the primary cell `p0L1` was chosen after the pilot (§7). These numbers must be reported next to it.
    - **Operating point at τ:**
      - **Cell-standardised ΔFAR:** +2.46, +2.95 and +2.73 pp, with unadjusted one-sided sign-flip p of .046, .023 and .003.
      - **FAR at τ by group:** phrase-boundary controls (pause > 150 ms) 0.22–0.25, switch points 0.13–0.17, other controls 0.08–0.12.
      - **Reading:** pauses trigger the detector more often than language switches. Criterion (c) compares AUC, not FAR at τ, so it does not show that switches exceed pause controls at the operating point.
    - **Attribution:** each grade is given with its primary category. Every criterion was assessable.

      | Model | H1 | H2-P0 |
      |---|---|---|
      | E1 | language-attributable (supported) | switch-associated (supported) |
      | E2 | language-attributable (nonzero but trivial) | language-attributable (supported) |
      | E3 | language-attributable (supported) | language-attributable (supported) |

      - **Criterion (b)** over 17 channels:
        - **H1:** the adjusted coefficient is in the predicted direction in 16, 16 and 17 channels, with p ≤ 2 × 10^−4.
        - **P0:** only 9, 11 and 10 channels go the predicted way. The p-values are .079 (E1, not met), .031 and .0085.
      - **Criterion (c):** 14 channels for H1, p ≤ 6 × 10^−4; 13 channels for P0 (AUC 0.56–0.57), p ≈ .01.
      - **The P0 grades for E2 and E3 meet the rule but are fragile.**
        - The sign-flip on means gives weight to large coefficients from small channels. For E3 the largest coefficient, +0.137, comes from `UChx9…`, which has 10 switch points.
        - The descriptive lme4 condition z is only 1.15–1.95. Its full random-effects structure was accepted except for E3's P0 model (no_video).
      - **What "language-attributable" means here, operationally:** the effect survives five acoustic-change proxies and pause-matched controls, and H3 points the same way. It is not causal evidence for language identity.
        - Phonetic content and recording confounds remain.
        - H3 uses the same embeddings, so it is not independent evidence.
      - **Scope:** no comparison across embeddings is made. E2 being "nonzero but trivial" does not mean it is less affected.
    - **Descriptive:** H1 grows with window length: dz 0.10–0.13 at L = 0.5 s (17 channels, 21,721 sets), 0.19–0.27 at 1.0 s, and 0.24–0.36 at 1.5 s (16 channels, 862 sets).
      - Every L = 0.5 s value is below the smallest effect.
      - Longer windows also select longer runs, so this trend mixes window length with which switches qualify.
    - **Sensitivity analyses run so far:**
      - item 1, alignment score ≥ 0.093 (240,915 of 255,695 points kept);
      - item 6, without the 2 test channels.

      Both leave every outcome category unchanged. Items 2–5, 7 and 8 (variant tag runs, the purity audit, the listening check, Brouhaha, loan re-inclusion) are pending.
      - **Update 2026-10-20:** item 4 ran (decision 27). Items 2, 3 and 8 ran as variant boundary runs (DEVIATIONS.md entry 9; `results/sensitivity/`), each read against a reference run on the rebuilt, unvaried boundaries (`base`). No H2-P0 or H3 category changes in any of the four runs. E2's H1 sits at the smallest effect throughout: dz 0.205 (base), 0.211 (item 2), 0.209 (item 8), all supported, and 0.183 (item 3), nonzero but trivial like M2a's frozen 0.194. E1 and E3 stay supported (dz 0.24–0.28). M2a's verdicts stand as frozen. Item 7 is not run: Brouhaha access was not granted. Item 5 and the item 4 listening wait on the author's listening answers (deviation 8).
    - **Overall size:** all effects are modest: dz about 0.2–0.27, AUC about 0.57–0.58, H3 about −0.2 SD.
    - **Deviation 1** (DEVIATIONS.md) affected no number; its 16 points are listed there. The paper reports it.

24. **H7 design: D4 tagging and the supplement** (2026-10-18; text and features only; no D4 outcome computed). Code: `csdiar/data/tags_d4.py`, `csdiar/data/boundaries_d4.py`, `scripts/lexicon_candidates_d4.py`, `scripts/build_lexicon_d4.py`, `csdiar/analysis/confirm_h7.py`. The supplement is `planning/PREREGISTRATION_H7.md`.
    - **The D4 tagger:** a new module, so that D1's frozen tagger and boundary code stay unchanged. The D4 boundaries reuse D1's frozen boundary code with only the tagger swapped.
    - **The lexicon:**
      - **Why not spelling matching:** consonant-skeleton matching alone was too imprecise for Hindi and Bengali. Frequent native words such as करें and করে match English words like "cream" and "car".
      - **What was done instead:** the most frequent types, plus all skeleton candidates seen at least 5 times, were labelled in context by LLM-based labellers.
      - **Checks:** row alignment, then a gloss-similarity test, which caught one shifted file; it was rewritten. A 60-item spot-check gave 59/60 agreement.
    - **Clusters are MUCS speaker IDs,** which are in effect recordings. The minimum is 5 units, not D1's 10, because recordings are short: about 10 switch points per speaker. A voice-cluster sensitivity analysis guards against one narrator appearing under several IDs.
      - **Threshold calibration:** a within-recording split-half threshold (θ ≈ 0.99) merged nothing, even across one D1 channel's videos. The threshold was therefore set from D1's different-channel video pairs.
    - **Tests:** a Monte Carlo sign-flip (2^17 patterns) because k is large, and Holm over 12 tests.

25. **H7 confirmatory result on D4** (2026-10-18; frozen `h7-freeze` = `a68e5b1`, record `h7-freeze-record`; one-shot run after deviation 2 fixed a pre-run crash). Results in `results/h7/confirm_h7.{json,md}`.
    - **Verdict: the SS-DL effect replicates.** All 12 tests (2 hypotheses × 3 embeddings × 2 pairs) are significant in the predicted direction (Holm one-sided p ≤ 2.6e-3), so every pair/hypothesis is "replicated (3/3)".
      - **Hindi-English:** effects close to D1 — H1 dz 0.236 / 0.184 / 0.260 (E1/E2/E3), H2-P0 AUC 0.564 / 0.557 / 0.580. H1 meets D1's smallest effect for 2/3 embeddings, H2-P0 for 3/3.
      - **Bengali-English:** same direction but all "nonzero but trivial" — H1 dz 0.158 / 0.119 / 0.149, H2-P0 AUC 0.535 / 0.524 / 0.542; 0/3 meet D1's smallest effect. So the effect is real in both pairs but weaker in Bengali-English.
    - **Voice-cluster sensitivity is degenerate on D4 and is reported as such, not relied on.** The D1-calibrated threshold θ = 0.5137 over-merges MUCS: 551 hi-en recordings collapse to 6 clusters, 307 bn-en to 3. With so few clusters the sign-flip has almost no power, so the sensitivity pass cannot meaningfully test the speaker-ID-as-recording concern on D4. This is a disclosed limitation of the pre-registered (outcome-blind) threshold transfer, not a change to it: θ was fixed in advance precisely to avoid tuning on outcomes, and the primary per-speaker-ID result stands on its own.
    - **Runtime note:** the run took ~35 min CPU because `bca_ci` bootstraps (10,000 resamples) run in pure Python over 550+ clusters. Correct, just slow; a future speed-up (vectorising the bootstrap) would be a separate reviewed change, not applied to the frozen run.

26. **Full mitigation study approved; authorship; DISPLACE access route** (2026-10-19).
    - Ishan approved the full build: the P1-AHC back end, M2, M3 with its training runs, H4 on D1 and H5/H6 on DISPLACE, accepting the GPU time in PLAN §8 (M3 alone ≈ 100–200 GPU-h).
    - Author block: "Ishan Gautam" as Ishan gave it. Affiliation and email are taken from his DISPLACE applicant record until he confirms them. The venue, and with it the venue ethics line, is still open.
    - DISPLACE access, checked on the public pages on 2026-10-19. This corrects decision 9, which said Hugging Face access approval was needed:
      - DISPLACE-M: the Hugging Face repo `Displace-M/Displace2026` is public and not gated (API: `gated: false`), so no access request is needed. Its zip archives are password-encrypted. The password ("credentials") comes from the organisers after the registration form and the signed Terms & Conditions are emailed to displace2026@gmail.com. The card limits use to challenge participants unless the organisers consent; Ishan reports they did (decision 9), and the signed terms plus their written reply are the record.
      - DISPLACE 2024 (optional): Zenodo records 10669296 and 12166687, password-protected; the signed terms, saved as `Terms_and_Conditions_DISPLACE_2024_<team_name>.pdf`, go to displace2024@gmail.com.
      - Nothing is downloaded until the password is in `~/.config/csdiar/keys.env` on the box, placed there by Ishan.
    - Build order, dependencies first, keeping the GPU busy:
      1. Speaker-purity audit exactly as frozen (PREREGISTRATION.md §5 item 4: TitaNet-L and WeSpeaker CAM++ clip embeddings), because M3's AAM-softmax trains on purity-verified speakers.
      2. P1-AHC (Community-1 segmentation, E1–E3 or mitigated embeddings, AHC-cosine) with B0/B1/B2 and the M1/M2 hooks.
      3. M2 LanguageGatedSCD from the existing VoxLingua107 posteriors (`~/data/d1/lid/*.npz`), with the oracle and time-shuffled λ variants.
      4. M3 training code, smoke test, and the choice between OpenSLR 54 and IndicVoices Nepali as the extra speaker pool.
      5. The H4 D1-CV pre-registration supplement, reviewed until a round has no major issues, then frozen before any M3 or confirmatory mitigation outcome is computed.
      6. M3 runs. 7. H4 confirmatory. 8. DISPLACE once the password arrives: ingest, dev tuning, H5 power simulation, M2b freeze, eval. 9. Paper, re-reviewed to two clean rounds.
    - Two design issues the H4 supplement must settle, and the review must check:
      - The H4 pilot (commit `2a71d20`) computed M1/B1 outcomes on the 3 pilot channels. PLAN §6 says the pilot looks at no mitigation results, and PLAN's 18-channel H4 cross-validation includes those channels. Proposal: score H4 only on the 15 non-pilot, non-test channels (5 folds of 3), keep the pilot channels on the fitting/tuning side of every fold without ever scoring them, and log this in DEVIATIONS.md.
      - D1 clips are 3–14 s, the pipelines almost never split on clips (PLAN §6 E-A), and D1-LF does not exist. H4 on D1 therefore needs a back end that registers the SS-DL excess on clips. Proposal: P0, which PLAN §5 already runs "with every mitigation", is primary for H4 on D1; P1-AHC is descriptive on D1 clips and primary on DISPLACE.

27. **Speaker-purity audit result** (2026-10-19; code and operational readings in DEVIATIONS.md 3; results in `results/purity/`).
    - **Embeddings:** all 20,805 clips with TitaNet-L and CAM++; 15 clips had no speech run of at least 0.5 s and were embedded whole. The 3 clips without provenance are left out, so the audit covers 20,802.
    - **Flagged clips (both embeddings):** 688 of the 20,444 assessable clips (3.4%); single-model flags 983 (TitaNet-L) and 1,021 (CAM++). The rate is flat across language mix (cs tertiles 3.3–3.4%, en 3.6%, ne 3.2%), so the flags do not track language. A 3.5 robust-SD rule would flag far fewer clips from a normal distribution, so most flags are probably acoustically odd clips rather than other speakers. The 200-clip listening sample (`listening_sample.csv`) decides that, and it needs a listener.
    - **Split-off videos (both embeddings):** 34 of 161 videos (4,319 clips), giving 39 speaker clusters in train (from 15 channels), 6 in the pilot split (from 3) and 7 in test (from 2). The frozen threshold, the 1st percentile of within-video split-half cosines, is 0.923 (TitaNet-L) and 0.944 (CAM++). Different videos of one channel typically sit at 0.77–0.95 (within-channel cross-video quartile to median), so the rule also splits recording sessions of the same teacher.
    - **A speaker-level reference:** video pairs from different channels have a 99.9th percentile of 0.616 (TitaNet-L) and 0.636 (CAM++). Six split-off videos are below it under both embeddings and look like other speakers: `v7JNIi2Daw0`; `x-LA7qvv5ac` and `57CuMjls6OM` (channel `UCZfM_5l…`, the two are also different from each other); `s_qrQBTBEzc`; `zJYS-XNDP2Q` and `aybOawW9VLM` (pilot channel `UCeq9rTo…`).
    - **Same teacher on two channels:** `zJYS-XNDP2Q` (pilot channel `UCeq9rTo…`) matches all four videos of train channel `UCOC6C8_…` at 0.82–0.88 under both embeddings; the next cross-channel pair is at 0.65. This is very likely one teacher on a pilot and a confirmatory channel. It changes no M2a statistic (the pilot channels set only τ and the power numbers) and is disclosed.
    - **Use:**
      - The frozen subset (flagged clips dropped, all 34 split-off videos as their own clusters) is used as is for the pre-registered H1–H3 sensitivity run, with the over-splitting caveat.
      - M3's speaker labels use the speaker-level reference instead (H4 supplement §7): flagged clips dropped, and a video is a separate speaker only below the cross-channel 99.9th percentile under both embeddings. Training on the frozen clusters would teach M3 to separate sessions of one teacher.
      - `zJYS-XNDP2Q` is left out of the H4 calibration channels and M3's dev set, so no scored teacher shapes τ or early stopping.

28. **M3 learning rate chosen on dev; purity-subset sensitivity result** (2026-10-19, before the H4 freeze; no H4 statistic computed).
    - **Why a sweep:** the first draft of the M3 recipe (body lr 2e-4, no warm-up) made the dev EER worse than E3's (0.166 at step 0; 0.25 at step 500, 0.244 at 750), with the embedding drifting far from E3. Dev here is the cross-language EER on the calibration channels (`m3_data dev`), never a P0 statistic.
    - **Sweep:** fold 0, seed 0, primary configuration, the 6,000-step schedule, dev EER every 250 steps, stopped at 2,000 steps.

      | body lr | head warm-up | dev EER at 500 / 1000 / 1500 | best (step) |
      |---|---|---|---|
      | 2e-5 | 0 | 0.151 / 0.146 / stopped | 0.136 (250) |
      | 2e-5 | 500 | 0.144 / 0.135 / 0.147 | 0.135 (1000) |
      | 5e-5 | 500 | 0.144 / 0.141 / 0.159 | 0.120 (1250) |

    - **Choice: lr 2e-5 with a 500-step head warm-up.** The rule written down before the sweep was the lowest best dev EER. Applied at the recipe's own evaluation cadence (every 500 steps), it picks this configuration (0.135, against 0.141 for lr 5e-5 and 0.146 without warm-up). The 5e-5 run's 0.120 is a single evaluation at a 250-step point the recipe never evaluates; its neighbouring values (0.141, 0.159, 0.170) show it is unstable. All three configurations improve on E3's 0.166.
    - **Warm-up semantics are the evaluated ones:** train mode throughout, body weights fixed for 500 steps while batch-norm statistics adapt. Adapting the statistics alone already moved the dev EER from 0.166 to 0.142 by step 250.
    - **Purity-subset sensitivity (PREREGISTRATION.md §5 item 4; DEVIATIONS 4; `results/purity/sensitivity_m2a_purity.json`):** with the 688 flagged clips dropped and the 34 split-off videos as their own clusters (46 clusters; 27–29 enter H1/H2-P0 and 45 enter H3), every primary result keeps its direction and passes the smallest effect:
      - H1: dz 0.253 / 0.206 / 0.255 (E1/E2/E3), all supported. E2 was "nonzero but trivial" at 0.194 on all clips.
      - H2-P0: AUC 0.579 / 0.564 / 0.587, all supported.
      - H3: shift −0.242 / −0.203 / −0.246 target SD, all supported.
      - The sign-flip is Monte Carlo here (2^17 patterns), so the printed p = 0 means p < 1/131,072. Sessions of one teacher count as separate clusters (decision 27), so these p-values are optimistic, which is why the result is descriptive.

29. **DISPLACE-M probably cannot supply H5's cross-language turns; DISPLACE 2024 can** (2026-10-19; from the published papers, before any download).
    - H5 and M2's α need DS-DL turns: a real speaker change where the language also changes, located from reference annotations.
    - **DISPLACE-M** (challenge paper, arXiv 2603.02813 v3, §3.2–3.3 and appendix A.2): 35 h of Hindi health conversations, with Haryanvi and Bhojpuri dialects, in 25 h dev and 10 h eval. It has speaker segments, and its transcripts are verbatim but "rendered in the Devanagari script", with no language annotation. English words in it are therefore indistinguishable by script, and the paper describes the multilingualism as dialects of Hindi, not Hindi–English turns. DS-DL turns could only come from automatic LID, which would make H5's reference partly automatic.
    - **DISPLACE 2024** (arXiv 2406.09494) has a language-diarization track with segment-level language references on 158 h of code-mixed conversations in Indian languages and English. That is exactly the reference H5 needs.
    - **Consequence:** DISPLACE 2024 becomes the preferred H5 and M2-tuning dataset again (as in decision 6). DISPLACE-M stays useful for P1-AHC tuning and as a secondary set. Ishan is asked to pursue the 2024 access (Zenodo, password from displace2024@gmail.com after the signed terms) first. The M2b supplement will fix the choice once the data and its annotations are on the box.
    - **Confirmed from the release's own file lists** (2026-10-19, after Ishan asked whether DISPLACE-M would do): the zip directories of the 10 Track 1 (speaker diarization) and Track 2 (ASR) archives in the Hugging Face repo `Displace-M/Displace2026` were read with HTTP range requests; nothing was downloaded in full or extracted. Every file in them is encrypted, so the release needs its own password too. Track 1 dev holds only `<ID>_SPEAKER.rttm` and audio: 125 Hindi recordings in two archives (about 25 h) and 26 Kannada recordings (about 4.4 h), the Kannada set not described in the paper. Neither track has language references. The eval archives (71 Hindi recordings; and 130 Hindi plus 25 Kannada) hold audio only, with no speaker references or transcripts, and Track 2 dev has one transcript per recording. Hours assume 16 kHz 16-bit mono WAV, which matches the paper's 25 h dev and 10 h eval. On DISPLACE-M, H5 would therefore need automatic language labels and a test split carved out of dev, so it stays a secondary set and the M2b supplement's fallback (its §10).

30. **D1 listening check result** (2026-10-20; one rater, the author, deviation 8; first 352 of 450 items, deviation 11; scorer `scripts/score_listening.py` as hashed in entry 11; result `results/listening/listening_d1.json`). Descriptive (PREREGISTRATION.md §5 item 5).
    - **Primary cell, confirmatory splits, design-weighted:** π (switch points without an audible language change) 6.3% (6 of 96 judged; 5 unsure), 95% bootstrap interval 2.0–11.6%. δ (different speaker, overlap or noise on either side) 0 of 101. κ (det_w1 controls with a language change) 0 of 52 (2 unsure). Marker within about 0.1 s of a word boundary: 81.9% of primary-cell switch points (74.0–89.4%), 76.2% of all switch points (67.7–84.2%).
    - **Corrected excess:** ΔFAR / (1 − π − κ) gives +2.63 / +3.15 / +2.91 pp for E1 / E2 / E3 (observed +2.46 / +2.95 / +2.73), intervals 2.51–2.78, 3.02–3.34 and 2.79–3.09 pp from the contamination rates alone (ΔFAR held fixed, as entry 10 specifies).
    - **Reading:** label contamination is small and one-sided (switch points that are not switches, never controls that are), so it biases the excess slightly towards zero, as expected; no switch point has a second speaker. Item counts are small and there is one rater who knew the hypotheses, so this is a check, not a correction the paper relies on.

31. **Purity listening result** (2026-10-20; one rater, the author, deviation 8; first 117 of 230 pairs, deviation 12, count confirmed by the author before the run; scorer as hashed in entry 12; result `results/listening/listening_purity.json`). Descriptive (PREREGISTRATION.md §5 item 4, listening).
    - **Flagged pairs** (104 scored): 15 unusable (14.4%, Wilson 8.9–22.4%). Among the 84 usable pairs judged same or different (5 unsure), 43 judged a different speaker: 51.2% (40.7–61.6%); over all judged pairs 49 of 92, 53.3%.
    - **Control pairs** (13 scored, two unflagged clips of one video): 1 unusable; 1 of 12 judged different, 8.3% (1.5–35.4%), which bounds the rater's false "different" rate.
    - **Difference** flagged minus control: +42.9 points on usable pairs (+44.9 on all judged pairs).
    - **Reading:** about half of the audit's flagged clips are other voices by ear, far above the control rate, so the flags mark real impurities and the purity subset (decision 28) drops them. With 688 flagged clips that is roughly 1.7% of D1's clips not spoken by the channel's main speaker.

## Facts established the same day


- Dataset revision pinned: `dd00ca41dd38774cd6097c372c16939d7f3746df` (20,805 clips, 58.425 h, 20 channels, 161 videos: 125 train / 14 validation / 22 test). Metadata in `data/d1_meta/` (not committed).
- The clips tile their source videos almost completely. Spans from each video's first to last clip add up to 61.7 h, and the clips cover 95% of that (median 99% per video). Median video span is 11 min, maximum 4.0 h. The long-form audio therefore mainly adds continuity, not new speech. Its value is that pipelines can run on the real continuous recording, without the artificial joins that concatenating clips would create.
- The 3 rows without provenance are all in the test split, channel `UChx9-bynjfm6Js3k54Sf1Mg`. No other provenance key is duplicated.
- Gated access accepted by Ishan; Hugging Face login on the box is his read-only token (account `ishangautam7`), never printed. Pinned downloads on the box: pyannote Community-1 at revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` (`~/data/models/pyannote-community-1`), and the Nepali part of IndicVoices at revision `c96f9088f138cf89d419da7e8e643e1f05c00a87` (`~/data/indicvoices_nepali`, 74 parquet files, 28.2 GB).
- IndicVoices Nepali audit (metadata only): 249,422 utterances, 467.4 h, 1,076 speakers. Scenarios: extempore 143k utterances, conversation 81k (87.3 h), read 25k. Speakers are mostly from West Bengal (93%) and Assam, 87% rural. Only 0.8% of utterances contain any Latin-script word, so its transcripts cannot carry the script-based language tags, and it is not a code-switching source. It is a large real-speech Nepali speaker pool, better than OpenSLR 54's read speech for M3's speaker loss and for nontarget trials.
- DISPLACE: Ishan emailed the 2024 and 2026 organisers on 2026-10-18 to ask about post-challenge research access. The agreements are saved unsigned on the box in `~/data/displace/access`. Nothing is signed or downloaded until they reply and Ishan approves. The 2024 archive is listed at 411.9 GB; the box has room.
- Long-form audio (decision 4): the download attempt failed, so no long-form audio exists.
