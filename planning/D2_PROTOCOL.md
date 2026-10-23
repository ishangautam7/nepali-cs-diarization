# D2 recording protocol

v0.2 · companion to `PLAN.md` v0.4 §§2–3 and `DECISIONS.md`.

**Purpose.** Record real, unscripted 2–3-person bilingual Nepali–English conversations so diarization systems see only a room channel while per-speaker close-talk channels give the speaker reference. The study tests whether Nepali↔English code-switching makes systems hypothesize speaker changes that are not there (SS-DL false splits) without missing real cross-lingual turns (DS-DL misses).

**Ethics.** No approval-committee gate. Ishan holds the necessary permissions, so recording starts without waiting for a committee review (`DECISIONS.md` decision 5). Every participant still gives informed, signed consent *before* recording, using the forms in `planning/ethics/`. Record the approving body and the approval reference for the paper's ethics statement: log [APPROVING BODY] and [APPROVAL REFERENCE] once, in the project record. Only real recordings: no scripted reading, no simulated or mixed audio.

**Blinding.** Participants are **not** told the code-switching hypothesis. They are told the same purpose as the information sheet: we test how well speech technology can tell who is speaking ("who spoke when") in natural conversation. Eligibility is screened with a neutral language-background questionnaire (§2); there is no cover story about mixing and no screening item about mixing. Debrief after each participant's last session (§6).

---

## 1. Design and sampling

| Item | Target |
|---|---|
| Sessions | ~28, each ~45 min (≈ 21 h total) |
| Split counts | 20 test, 8 dev |
| Participants per session | 2–3 |
| Distinct participants | ~35–50 |
| Sessions per participant | at most 2, both in the **same split** |
| Test conversations | ≥ 20 |
| Same-gender sessions | ≥ half of all sessions |

The final number of sessions and participants is **set by the pilot power simulation** (`PLAN.md` §2b, M2b), not fixed here; the row above is the working target.

**Balance targets (soft, logged per session):** gender mix across dev and test; age spread [e.g. 18–25 / 26–40 / 41+]; home region / dialect [Kathmandu valley vs other]; everyday English use inferred from the neutral questionnaire. Aim so neither split is dominated by one cell.

**Eligibility.** Screen with a **neutral language-background questionnaire** containing filler items (schooling, media habits, languages used at home/work). Nothing singles out language mixing, and no item asks about switching. Pairs/triads should already be acquainted enough to talk freely.

**Split assignment at recruitment.** Each participant is assigned to **one split (dev or test) at recruitment, per recruitment group**. A session combines only participants of the same split, so dev and test stay participant-disjoint (`PLAN.md` §3). Once assigned, a participant never moves splits. **Pilot participants are all assigned to dev.**

---

## 2. Elicitation

Goal: natural Nepali–English mixing **without ever asking for it**. Give a topic, let them talk. Never mention language, mixing, or switching.

**Topic prompts (pick 4–6 per session, open-ended, rotate):**

1. Your typical day from morning to night.
2. Favourite food and where to get the best [momo/sel roti/local dish].
3. A recent trip or trek — what happened, what went wrong.
4. Phones and apps you use most and why.
5. A film or series you both watched recently.
6. Studies or work — what you actually do day to day.
7. Festivals — how your family celebrates [Dashain/Tihar].
8. Load-shedding, traffic, or city life frustrations.
9. Sports — a match or team you follow.
10. Cooking something: walk each other through a recipe.
11. Money and prices — how things have changed.
12. A gadget or purchase you regret or love.
13. Social media — what you post, what annoys you.
14. Future plans — study abroad, job, business idea.
15. A childhood memory or hometown story.
16. Weddings you attended recently.
17. Health, fitness, or a doctor visit.
18. Comparing Kathmandu with another city you know.
19. A skill you are learning (driving, a language, coding).
20. A news story you have been following.
21. The best and worst teacher you ever had.
22. Technology at work/school — tools, software, meetings.

**Facilitator rules.** The facilitator gives all instructions **before recording starts**, then either stays silent or leaves the room. The facilitator is **never a participant** and is never scored: any facilitator speech that lands on tape is excluded from scoring through the UEM (`PLAN.md` §2). There is no `FAC` channel; the facilitator is not wired.

- Hand over the topic before pressing record, then step back.
- If talk stalls (> ~20 s silence) and the facilitator stayed in the room, hand over the next topic card without joining the discussion. Prefer printed topic cards so no prompting is needed mid-session.
- Do not react to switches, do not correct language, and do not discuss language at any point.

**What not to say to participants.** Do not say: "mix languages", "use English", "switch", "speak Nepali", "code-switch", "say it in [language]". If asked what the study is about: "we're testing how well speech technology can tell who is speaking in a natural conversation" — nothing more until debrief.

---

## 3. Equipment

Give example models per item, **without prices**. Every item must be checked for local availability in Nepal (Kathmandu electronics/pro-audio dealers or import lead time) before purchase. 32-bit float recording is preferred because it makes clipping effectively a non-issue.

| Item | Requirement | Example models (verify availability + exact spec locally) |
|---|---|---|
| Multitrack recorder | **≥ 6 XLR inputs** (3 lavaliers + room + spare), phantom power (48 V), per-input tracks; 32-bit float preferred | Zoom F6 (6-in, 32-bit float); Zoom F8n Pro (8-in, 32-bit float); Tascam Portacapture X8 |
| Room mic (always required) | **One true omnidirectional** mic for the shared far-field channel. The F6/F8n Pro have **no onboard mic**, so a dedicated room mic is always needed | Rode NT55 with omni capsule; Audio-Technica AT4022 (omni). **Not the Rode NT5 — it is cardioid** |
| Wired lavaliers (one per speaker) | Omnidirectional, on an XLR track. Either a **phantom-powered XLR** lavalier, **or** a plug-in-power model through a **phantom-to-plug-in-power adapter** | Sennheiser ME 2-II (phantom XLR); Audio-Technica AT831b (phantom XLR); Rode Lavalier II (plug-in-power) via a Rode VXLR Pro adapter |
| Headphones | Closed-back for monitoring levels | Sony MDR-7506; Audio-Technica ATH-M40x |
| Cables | XLR cables, one per mic + spares; lavalier phantom-to-plug-in-power adapters as needed | — |
| Power | Recorder batteries + spares, mains adapter, power strip | — |
| Storage | 2× SD cards (recorder spec), card reader; ≥ [128] GB each | SanDisk Extreme Pro SDXC (recorder-approved speed class) |

**Minimum workable kit:** 1 recorder (≥ 6 XLR in), 3 lavaliers (one per speaker), 1 dedicated omni room mic, any phantom-to-plug-in-power adapters, headphones, cables, power, two cards.

---

## 4. Room and placement

**Room choice.** Quiet, low-reverb, indoor room with the door closed [ROOM: e.g. a small meeting room]. Avoid: open halls, glass/tile echo, fans/AC noise, fridge hum, street-facing windows, fluorescent buzz. Turn off or unplug noisy appliances. One consistent room across sessions if possible; log any change.

**Seating diagram (text).**

```
          [ Room mic, centre, ~0.8–1.0 m above table ]
                            |
         S1 ---------------+--------------- S2
                       S3 (if 3)
   speakers ~0.8–1.2 m apart, facing in, around a small
   table; recorder on table; facilitator outside the circle or absent
```

- **Room mic position/height.** Centred over the speakers, roughly equidistant from each, **~0.8–1.0 m above the table surface**, capsule aimed at the group. This is the only channel the systems ever see, so place it as a neutral far-field listener, not close to any one speaker.
- **Lavalier placement.** Clip ~15–20 cm below the chin, centred on the chest, capsule facing up. Keep each lavalier well away from the others so one person's mic mostly captures that person (crosstalk is handled in post, but less bleed is better).
- **Clothing-noise tips.** Clip to a firm edge (collar, placket, lapel), not loose fabric. Run the cable under clothing and strain-relief it with a loop. Avoid scarves, zippers, necklaces and hair over the capsule. Ask participants not to touch or tap the mic. Re-check after they settle.

---

## 5. Recorder settings and file naming

**Settings.**
- Sample rate: **48 kHz**.
- Bit depth: **32-bit float** if the recorder supports it (preferred); otherwise **24-bit**.
- One **WAV per channel** (per-track files — never a single downmix). Each lavalier is its own track; the room mic is its own track. **All tracks share the recorder's single clock.**
- Phantom power on for condenser lavaliers and the room mic (plug-in-power lavaliers via their adapter).
- Set input gains at soundcheck (see §6). With 32-bit float, keep gain conservative; with 24-bit, target peaks around −12 dBFS.
- No onboard limiting/EQ/noise-gate that alters the recording (monitoring-only processing is fine).

**Naming scheme.**
- Session folder: `D2_[SITE]_S[NN]_YYYYMMDD` (e.g. `D2_KTM_S03_20261101`).
- Channel files: `D2_S[NN]_[CHAN].wav`, where `CHAN` ∈ `ROOM`, `LAV1`, `LAV2`, `LAV3`.
  - e.g. `D2_S03_ROOM.wav`, `D2_S03_LAV1.wav`.
- Keep a `session.csv` and `participants.csv` (schema in §7) in the session folder.
- `LAVn` maps to a participant pseudonymous ID in `participants.csv`; real names never appear in filenames.

---

## 6. Session runbook (checklists)

### Before
- [ ] Consent form **signed** by every participant (the required boxes, plus their own choice on the four optional boxes) before anything is recorded.
- [ ] Confirm all participants in this session share the **same split**; assign/confirm **pseudonymous IDs** (e.g. `P031`); record split (dev/test). Pilot → dev.
- [ ] Brief participants with the neutral framing only (§2); facilitator gives **all instructions now, before recording**.
- [ ] Fit lavaliers; check clothing noise; confirm channel-to-person mapping.
- [ ] Set input **levels** on each channel while they chat casually (watch peaks; redo if clipping or too low).
- [ ] Confirm per-channel WAV, 48 kHz, 24-bit/32-bit float, cards have space, batteries/power OK.
- [ ] Remind everyone: **avoid saying real names on tape** where possible. Phones silent.

### Start
- [ ] Facilitator falls silent or leaves the room.
- [ ] Press record; confirm all channels armed and levels moving.
- [ ] **3 claps** in front of the room mic, in clear view (sync-verification marker).
- [ ] **30 s of room tone** (everyone silent) for noise profiling.
- [ ] Give the first topic (already handed over), then step back.

### During
- [ ] Monitor levels on headphones; watch for clipping, rustle, dropout.
- [ ] Keep participants talking; re-prompt only on long stalls by handing over the next topic card; never join or discuss language.
- [ ] Note timestamp of any disruption (phone, knock, cough fit) for the log.

### End
- [ ] Signal wind-down; **3 claps** again before stopping (end sync-verification marker).
- [ ] Stop recording.
- [ ] **Debrief** only after a participant's **last** session: explain the code-switching focus; answer questions; confirm consent still stands; note any withdrawal. A participant who may return for a second session is not yet debriefed.
- [ ] **Payment** per agreed amount [AMOUNT]; obtain receipt/acknowledgement.

### After
- [ ] **Copy** all channel WAVs off the card to the project box.
- [ ] Compute **checksums** (e.g. `sha256sum *.wav > manifest.sha256`); verify copy integrity.
- [ ] **Encrypted upload** to team-only storage; fill `session.csv` / `participants.csv`.
- [ ] **Wipe card only after** copy + checksum verification succeed.
- [ ] Log issues and timestamps for post-processing.

---

## 7. Metadata schema

### `session.csv` (one row per session)

| Column | Type | Notes |
|---|---|---|
| `session_id` | str | `D2_[SITE]_S[NN]_YYYYMMDD` |
| `date` | date | YYYY-MM-DD |
| `site` | str | `[ROOM/location code]` |
| `split` | enum | `dev` / `test` (pilot → `dev`) |
| `n_speakers` | int | 2 or 3 participants |
| `participant_ids` | str | comma-separated pseudonymous IDs |
| `same_gender` | bool | all participants same gender |
| `duration_min` | float | recorded minutes |
| `recorder_model` | str | — |
| `bit_depth` | enum | `24` / `32f` |
| `sample_rate_hz` | int | 48000 |
| `room_mic` | str | model |
| `start_claps_s` | float | approx time of start claps |
| `end_claps_s` | float | approx time of end claps |
| `room_tone_s` | float | room-tone duration |
| `topics` | str | prompt IDs used (from §2) |
| `issues` | str | free text (noise, dominance, interruptions) |
| `checksum_verified` | bool | after-copy verification |

### `participants.csv` (one row per participant per session)

| Column | Type | Notes |
|---|---|---|
| `participant_id` | str | pseudonymous (`P###`) |
| `session_id` | str | FK |
| `channel` | enum | `LAV1` / `LAV2` / `LAV3` |
| `split` | enum | `dev` / `test` (must match session; fixed at recruitment) |
| `session_index` | int | 1 or 2 (≤ 2 per participant) |
| `debriefed` | bool | true only after the participant's last session |
| `gender` | str | self-reported |
| `age_band` | str | `[18-25/26-40/41+]` |
| `region` | str | home region/dialect |
| `english_use` | enum | inferred from the neutral questionnaire |
| `l1` | str | first language |
| `consent_dua` | bool | optional box: share with other researchers under a data-use agreement (includes audio) |
| `consent_public_audio` | bool | optional box: public audio with names bleeped |
| `consent_public_text` | bool | optional box: public transcripts and annotations only |
| `consent_recontact` | bool | optional box: contact about future studies |
| `reconsent` | enum | `still_agree` / `withdrawn` / `changed` (debrief form, after the last session) |
| `withdrawn` | bool | set true if participant withdraws; the whole session's audio then leaves the study |

A session's audio is released at a tier only if every participant in it ticked that box (`consent_*` true for all, after re-consent).

No real names in either file; the name↔ID key is stored separately, encrypted, team-only.

---

## 8. Post-session processing

1. **Sync verification from claps.** Detect the 3 start claps (and end claps) on each channel and confirm the channels agree. A single recorder gives one shared clock, so **claps only verify sync — tracks are never shifted or re-aligned using them.** Log any unexpected residual offset as a fault.
2. **Close-talk speaker reference.** Run VAD on each lavalier with **per-channel thresholds**. Crosstalk is removed **only where another channel's louder speech explains it by a margin**, so genuine simultaneous speech (real overlap) is kept, not gated away. This yields the reference RTTM; it is corrected by hand (Pass 1 below).
3. **ASR draft transcript** per speaker from the close-talk channels (draft only). The draft VAD (e.g. Silero VAD) and the draft ASR (e.g. Whisper) are **tools, not components of any system under test** (P0–P3, P1-AHC, E1–E3). In particular, no pyannote segmentation is used to build the drafts.
4. **ELAN export.** The `.eaf` links **every lavalier WAV and the room WAV**, so annotators can hear each close-talk channel and the shared channel. There is **no editable language tier**: language tags come from the script rule after transcript correction, not from annotator judgement.
5. **Annotation order.**
   - **Pass 1 — transcript + speaker correction, full listen-through.** Annotators correct the full transcript and the speaker segments across the whole session, not only near points. This fixes any switch the ASR wrote in the wrong script before any tags exist.
   - **Pipeline regeneration.** From the corrected transcript the pipeline regenerates the script-based language tags (`PLAN.md` §2), switch points, controls and random filler points.
   - **Pass 2 — boundary checks at untyped markers.** Annotators check speaker and word boundaries at each marker. Markers are **untyped and show no label** of whether they are a transition, switch point, control or filler.
   - **20% second annotation.** A second annotator independently redoes 20% of sessions **starting from the close-talk VAD segments only, without the ASR draft** (`PLAN.md` §4.8). Inter-annotator DER and language-boundary F1 are reported.
   - **Backchannels** (short acknowledgements such as "हो" or "yeah") are tagged. Backchannel-only segments create **no speaker transitions in the primary analysis** (`PLAN.md` §2); a sensitivity analysis includes them.

Systems under test only ever receive the **ROOM** channel; the lavalier channels exist solely to build the reference.

---

## 9. Pilot session goals

Record one pilot session (assigned to dev) and measure, per audio hour:

- **DS-DL turns / hour** — real cross-lingual speaker turns (powers the H5 test size).
- **SS-DL switch points / hour** — within-speaker language switches.
- **Crosstalk level** — bleed from other speakers into each lavalier (dB), to set the per-channel VAD thresholds and margin.
- **Annotation hours / audio hour** — to budget annotator time and cost.

Feed these into the **power simulation** (`PLAN.md` §§2b, 6–7, M2b) to fix the final D2 size (expected ~600–1,000 DS-DL test turns). No mitigation results are examined at pilot.

---

## 10. Common failure modes and fixes

| Failure | Fix |
|---|---|
| **Crosstalk** (other speakers bleed into a lavalier) | Increase mic separation; conservative gain; remove crosstalk only where a louder channel explains it by a margin, keeping real overlap; log per-session crosstalk from the pilot |
| **Rustle / clothing noise** | Reclip to a firm edge; strain-relief loop; remove scarves/necklaces; re-check after settling; ask participants not to touch the mic |
| **Clipping** | Prefer 32-bit float; otherwise set peaks ~−12 dBFS and re-level at soundcheck; monitor on headphones |
| **Dominance by one speaker** | Choose engaging topics and rotate them; log imbalance. The facilitator does not intervene mid-session beyond handing over a topic card |
| **Participants avoiding English** | Choose topics tied to tech/study/media/urban life that naturally invite English terms; never ask them to use English; drop/replace topics that go fully monolingual |
| **Overlapping speech** | Expected and **kept** (scored with overlap); per-channel-threshold close-talk VAD disambiguates who spoke; flag heavy-overlap regions for annotators |
| **Interruptions / disruptions** (phone, knock, outside noise) | Note timestamp; pause only if severe; exclude the affected span from scoring if unusable; never fake a repair |
| **Lost sync / missing claps** | Always do start **and** end claps; if missed, add claps immediately and log. Single-recorder capture means per-channel clocks already match — claps only verify this, never re-align |
| **Level drift mid-session** | Set gain once and do not touch during recording; with 32-bit float, correct in post instead of re-gaining live |

---

*Freeze the D2-dependent parts of this protocol and the pre-registration at milestone M2b, after the pilot, before any D2-test scoring (`PLAN.md` §9).*
