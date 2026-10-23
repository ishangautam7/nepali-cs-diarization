# Annotation guidelines — Nepali–English conversations (D2) and listening checks (D1)

v0.2. No conversation (Job A) was annotated under these guidelines, and the pilot that was to fix the provisional values below was not run.

## 1. Purpose

You are annotating bilingual Nepali–English speech for a speech-research study on
who is speaking and in which language over time. Your labels are the ground truth
the study measures automatic systems against, so accuracy and consistency matter
more than speed. You do **not** need to know the research question; label what you
hear, exactly as described below.

There are two jobs. Most hires do **Job A** (conversations); some also do **Job B**
(short listening checks on an existing corpus). Each job is explained in full below.

## 2. Tools

**Job A — ELAN.** Install the latest ELAN release from the Max Planck Institute
(https://archive.mpi.nl/tla/elan) — currently the ELAN 6.x line; use whatever is
newest. We provide one `.eaf` file per session.

The `.eaf` links **every audio channel of the session**: one close-talk (lavalier)
WAV per speaker, plus the room WAV. In ELAN you can **solo any single channel** —
mute the others and listen to one speaker's lavalier alone — which is how you tell
apart who is talking when people overlap. Use this constantly.

Most sessions arrive in a **two-pass** form (Pass 1 then Pass 2 below). The file is
pre-filled with:

- an automatic **draft of speech segments per speaker**, derived from each speaker's
  close-talk channel, and
- an automatic **ASR draft transcript** inside those segments.

Some sessions (the double-annotated 20%) arrive with **no ASR draft and no prior
annotator's work** — just the speech segments from the close-talk channels. If you
are assigned one of those, you type the transcript from scratch; §6 says so
explicitly.

**Tier structure (do not rename or restructure):**

- One tier per participant, named `SPK_<participant id>` (e.g. `SPK_P01`,
  `SPK_P02`). Each holds that speaker's **transcript text**, segment by segment.
- A `FAC` tier for any **facilitator** speech (the project member running the
  session), if the file has one. Transcribe it so it is visible, but it is **not
  scored** and is not a participant — never move facilitator speech onto a `SPK_*`
  tier, and never move participant speech onto `FAC`.
- A `NOTES` tier for anything that is not transcript (tags, flags; see below).
- **Overlapping speech stays on the separate speaker tiers** — if two people talk
  at once, each one's words go on their own `SPK_*` tier for the time they are
  speaking. Solo each lavalier to attribute the words. Never merge overlapping
  speech onto one tier.

There is **no language tier and nothing to tag as a language.** The study reads each
word's language from its script automatically (see §4), so you never label a word
"Nepali" or "English" — you just write it in the right script.

**Job B — web review page.** A browser page we host. Each item plays a short audio
snippet and shows a spectrogram with a vertical line marking one point in time. You
answer three fixed questions per item (§7). No install needed.

## 3. Segmentation rules (Job A)

A **segment** is one continuous stretch of one speaker's speech.

- **Start and end at audible speech.** The segment boundary sits where the sound of
  the word actually begins or ends, not on the silence around it. Zoom in and use
  waveform + playback to place it; aim within ~50 ms of the true onset/offset.
- **Split at pauses of at least 0.5 s (provisional).** If one speaker pauses for 0.5 s or more,
  end the segment and start a new one after the pause. Shorter pauses stay inside
  one segment.
- **Backchannels** — short listener responses such as `हो`, `हजुर`, `yeah`, `ok`,
  `right` said while someone else holds the floor — get their **own short segment**
  on that listener's tier, transcribed normally, **and** a `[backchannel]` tag on
  the `NOTES` tier time-aligned to that segment. Do not fold them into the main
  speaker's segment. Tagging matters: a backchannel-only segment (one where the
  listener says nothing but a backchannel) is deliberately not counted as a speaker
  turn in the main analysis, and the tag is how the pipeline knows to set it aside.
- **Laughter, noise, cross-talk.** Use these tags on the `NOTES` tier, time-aligned
  to where they occur:
  - `[laughter]` — laughing, by itself or over speech.
  - `[noise]` — non-speech noise (door, mic bump, cough) loud enough to matter.
  - `[crosstalk]` — a stretch where speakers overlap so much that individual words
    cannot be reliably separated even after soloing each lavalier; still transcribe
    what you *can* attribute on the `SPK_*` tiers.
  - `[unintelligible]` — speech you cannot make out; put it on the speaker's tier in
    place of the words.

## 4. Transcription rules (script carries the language)

Transcripts drive the whole study, because it derives each word's language **purely
from its script**: **Devanagari = Nepali (ne), Latin = English (en)**. There is no
separate language label to set and no way to override this — if you put a word in the
wrong script, its language is wrong. So the script you choose *is* the language
annotation. Get it right at the word level.

- **English words in Latin script.** Fillers and discourse words too: write `so`,
  `okay`, `right`, `means`, `suppose`, `example`, `like` — never `सो`, `ओके`.
- **Nepali words in Devanagari.** Including Nepali discourse words: `हो`, `हजुर`,
  `अनि`, `त`, `नि`, `क्या`.
- **Numbers are written as words, in the language actually spoken.** A number spoken
  in Nepali is written as a **Devanagari word** (`तीन`, `सात`, `बाह्र`); a number
  spoken in English is written as a **Latin word** (`three`, `seven`, `twelve`).
  **Do not write digits.** This is on purpose and it **differs from the existing D1
  corpus**, which transcribes numbers as Latin digits (`3`, `7`, `12`); we do not,
  because a digit would otherwise be dropped as "neutral" and the switch it carries
  would be lost. Transcribe the number the way the speaker said it, in that
  language's script. Example: `आज चाहिँ chapter three को बारेमा discuss गर्दैछौँ`
  if "three" was said in English; `आज चाहिँ तेस्रो chapter को बारेमा ...` if it was
  said in Nepali.
- **Proper names get a name tag.** Write the name in the script that fits how it is
  spoken (`Jack`, `Madam`; `राम`, `सीता`), **and** add a `[name]` tag on the `NOTES`
  tier time-aligned to it. A name counts as neutral in the analysis — it is neither a
  Nepali nor an English content word — and the tag is what tells the pipeline to
  treat it that way, so a name written in Latin is not mistaken for an English switch.
- **Technical terms and symbols** stay in their spoken/written form: `L2 cache`,
  `system.out.println`, `A + B = C`, `0°`, `√`, `%`, `₹`. Math/operator symbols are
  written as symbols, not spelled out.
- **Punctuation is light.** `,` and `.` where a clear clause/sentence break is heard,
  `?` for questions, `।` (danda) for Nepali sentence ends. Do **not** add heavy
  punctuation, quotation marks around fillers, or capitalisation beyond what the word
  naturally takes (proper nouns, acronyms).
- Transcribe **what is said**, including false starts and repetitions; do not clean
  up grammar.

Short reference examples (match the pattern):

```
यसरी चाहिँ हामी Jack Dairy implement गर्न सक्छौँ hope you understood this.
most urgent error लाई पहिले मैले सल्ट्याएँ हो and ask a teammate to review
equation of axis so axis भनेको figure Madam
```

### Flagging loanwords and ambiguous words (you flag; the team decides)

Some English-origin words are spoken as everyday Nepali, and some Nepali words turn
up in Latin — the script is genuinely ambiguous (e.g. an English loan said with
Nepali inflection). You do **not** decide what counts as a loanword, and you do not
maintain any shared list. The team settles the loan lexicon centrally; it is frozen
at milestone M2b and applied the same way to every split.

Your only job with an ambiguous word is to **flag it**:

1. Write it in the script that best matches **how the speaker pronounces and uses
   it** in that moment.
2. Add a `[loan: <word>]` tag on the `NOTES` tier at that time if you think it may be
   a loanword, or `[ambiguous: <word>]` if you are simply unsure which script it
   belongs in.

Then move on. Do not agonise, do not keep a personal list, and do not change your
spelling of a word to be "consistent" with a decision you think the team will make —
flag it and transcribe what you hear. The team resolves the list in the weekly
meeting.

## 5. Two-pass workflow (Job A)

Work each session in two passes, in this order.

**Pass 1 — full transcript + speaker segments, whole session.**

1. **Open the `.eaf`** in ELAN; confirm every channel plays and the draft tiers are
   present. Solo-check each lavalier.
2. **Listen through the entire session, start to end**, and correct:
   - the **full transcript** on every `SPK_*` tier — right words, right script per §4
     — for the *whole* session, not just near any particular point. Getting the
     script right everywhere is what makes the language tags right everywhere.
   - the **speaker segments** — who is talking and when — across the whole session,
     catching turns the draft merged, dropped, or put on the wrong tier. Solo
     lavaliers to resolve overlap.
   - **boundaries** to ~50 ms as you go.
3. Add `NOTES`-tier tags as you work: `[backchannel]`, `[name]`, `[loan:/ambiguous:]`,
   `[laughter]`, `[noise]`, `[crosstalk]`, `[unintelligible]`.
4. Mark Pass 1 done. **The pipeline then runs**, off your corrected transcript: it
   recomputes the language tags from the script and places the boundary-check markers
   you will see in Pass 2. You do not place those markers and you never see what type
   each one is.

**Pass 2 — boundary checks at the markers.**

5. Re-open the session. It now carries a set of **marker points** on a dedicated
   points tier. Within **±2 s of every marker**, check and tighten:
   - the **segment boundaries** (onsets/offsets to ~50 ms),
   - **who** is speaking on each side (right `SPK_*` tier),
   - the **transcript** right around the marker.
   The markers are **untyped and all look identical.** Some sit at language switches,
   some at speaker transitions, some at backchannels, and some are random filler
   points — but nothing tells you which, and you must treat every marker exactly the
   same way. Do **not** try to guess a marker's type, and do not work harder at one
   marker than another.
6. **Self-check** against §9 and mark the session done.

> For a **double-annotation session** (no ASR draft, no prior work — see §6), Pass 1
> is the same but you type the transcript from scratch on top of the close-talk
> segments. Everything else is identical.

## 6. Second (independent) annotation

**20% of Job-A sessions are annotated a second time, independently.** If you are the
second annotator on a session:

- You start from **only the close-talk (lavalier) VAD speech segments.** You get
  **no ASR draft transcript** and **no first annotator's work** — you must not see
  either.
- Type the full transcript yourself (Pass 1), then do the Pass-2 boundary checks, by
  the same rules as everyone else.
- **Do not look at anyone else's file**, and do not discuss a session you share with
  another annotator. Independent means independent; that is the whole point of the
  measurement.

## 7. Job B — D1 listening checks

Each item on the web page plays a short snippet and marks one point on a
spectrogram. For every item, answer three yes/no (or "unsure") questions about that
marked point:

1. **Is there a real language switch here?** — i.e. does the speaker change from
   Nepali to English or English to Nepali at that point?
2. **Is it the same single speaker on both sides?** — one person speaking before and
   after, not two different people.
3. **Is the marked boundary within about 100 ms of the actual switch?** — is the line
   placed on the switch, or clearly early/late?

Rules:

- **Marked points are mixed, and the condition is hidden.** Some are genuine switch
  points and some are controls, and **you are not told which is which.** Judge each
  item on its own; do not try to guess the pattern or keep a running balance of
  answers.
- Use the "unsure" option sparingly, only when the audio is genuinely unclear; add a
  one-line note when you do.
- Use closed headphones; replay as many times as needed.

## 8. Quality assurance

- **Qualification test.** Before any paid work, pass a short test on a **gold
  sample** (a session/items we already annotated). You must reach the agreed
  accuracy bar to start; this is part of paid training.
- **Double annotation.** **20% of Job-A sessions** are annotated independently by a
  second annotator who works from the close-talk segments only (§6). Do not look at
  anyone else's file.
- **Agreement measures** we compute and track:
  - inter-annotator **DER** with a **0.25 s collar** (speaker labelling/segmentation),
  - **language-boundary F1 at ±0.2 s** (do switches land in the same place),
  - **word error rate near the markers** (transcript agreement where it matters most).
  - Target thresholds were to be set after the pilot, which was not run.
- **Weekly calibration meeting.** The whole team reviews disagreements and clarifies
  edge cases. The team — not individual annotators — decides the loan lexicon here,
  and once it is frozen at M2b it does not change. Attendance is paid.
- **Resolving disagreements.** Where two annotators differ on a double-annotated
  session, a lead adjudicates; the agreed version becomes the reference, and any rule
  clarified is written back into these guidelines so the fix is permanent.

## 9. Self-check before marking a session done (Job A)

- Every speaker's speech is on the correct `SPK_*` tier; overlaps are on separate
  tiers; facilitator speech (if any) is on `FAC`, never on a `SPK_*` tier.
- No empty/zero-length segments; boundaries sit on audible speech.
- Script per word is correct (English Latin, Nepali Devanagari); **numbers written as
  words in the spoken language, never as digits.**
- Pass 1: the **whole** transcript is corrected, not just near the markers.
- Pass 2: every marker (±2 s) has been checked, and you treated all markers alike.
- `NOTES` tags added: `[backchannel]`, `[name]`, `[loan:/ambiguous:]`, and the
  laughter/noise/crosstalk/unintelligible tags.

## 10. Time estimate (provisional)

Plan for roughly **4–6 hours of annotator time per audio hour** of conversation for
Job A — that covers Pass 1 (full transcript + segments + one full listen-through) and
Pass 2 (the marker boundary checks) together. This is **provisional**: the pilot that was to
measure it was not run.

## 11. Confidentiality and devices

- You sign a **confidentiality agreement** before starting.
- You work **only on project-provided equipment** — project laptops or a managed
  remote desktop. You do **not
  use a personal device** for any project data, and you do not need your own laptop to
  do this work; the project provides what you annotate on.
- Work **only on the provided storage/accounts.** Do not download, copy, screenshot,
  or move audio, transcripts, or annotations anywhere else.
- **No copies** of any material, ever; delete nothing from the provided storage
  either.
- **Do not discuss the content** of recordings — what was said, who spoke, any names
  or personal details — with anyone outside the project.

## 12. Reporting problems

If anything is wrong — audio won't play, a channel/tier/draft is missing, a session
seems corrupted, a recording raises a privacy concern, or a rule here does not cover
what you hear — **do not guess.** Flag it in the project's annotation channel with the session id and timestamp, and continue with
the next session. Ask in the weekly meeting if it is not urgent.
