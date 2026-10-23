# Vendored: UniSpeech speaker-verification ECAPA-TDNN

- **Source:** https://github.com/microsoft/UniSpeech, commit `6112826ac13a4327f4c9a7afa2a505e35b763514`, file `downstreams/speaker_verification/models/ecapa_tdnn.py`.
- **Licence:** Creative Commons Attribution-ShareAlike 3.0 Unported (`LICENSE` in this folder). This folder and any modified version of it stay under CC BY-SA 3.0. The rest of the repository is licensed separately.
- **Changes:**
  1. The SSL upstream is built with the pinned `s3prl` package (`s3prl.hub.<feat_type>()`), not with `torch.hub.load('s3prl/s3prl', ...)`, which fetches whatever GitHub head is current.
  2. The `config_path` branch (UniSpeech-SAT's own upstream code) is not vendored and raises `NotImplementedError`.
  3. A provenance header was added.
- **Used as:** E3 in PLAN.md §5. WavLM-Large plus ECAPA-TDNN, loaded with the authors' fine-tuned checkpoint "WavLM large, Fix pre-train: No" (VoxCeleb1-O EER 0.431%). The checkpoint's SHA-256 is pinned in `ops/env_e3.sh`. The same model is the starting point for M3.
