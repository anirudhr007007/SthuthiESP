# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
semantic versioning.

## [Unreleased]

### Added

- **NVIDIA NIM** as a supported answer provider (`AI_PROVIDER=nvidia`). It uses
  the OpenAI-compatible hosted endpoint
  `https://integrate.api.nvidia.com/v1/chat/completions` with `AI_API_KEY`
  (`nvapi-...`) and `AI_MODEL` (for example `z-ai/glm-5.3-flash`);
  `AI_BASE_URL` may point at a self-hosted NIM deployment instead. NVIDIA is
  rate-limited like the other cloud providers and is probed by `--doctor`.
  Hosted NVIDIA speech is a separate streaming/gRPC service, so voice input
  remains local Whisper or Google STT.
- `WEB_SEARCH` flag (default `true`) to opt out of the DuckDuckGo context
  added for unmatched general questions, and `VOICE_DEBUG` (default `false`)
  to gate the terminal printing of heard questions and answers.
- `make_archive.ps1` to build a shareable source archive that never contains
  `.env`, `.venv`, `.runtime`, `.cache`, or biometric training images.
- `setup_windows.ps1` and `cleanup_runtime.ps1` for a project-local runtime:
  dependencies go into `.venv`; pip, temporary, DeepFace, Whisper, Keras, and
  tool caches go into `.runtime`. Cleanup is a dry run unless `-Force` is used.
- `runtime_env.py` to apply the local cache layout before heavy imports.
- `--chat` for a text-only session and `--warmup-models` to pre-fetch/load the
  configured face and local Whisper models without opening the camera.
- `AnswerResult` with privacy-safe routing metadata, while `answer_question()`
  remains the string-only compatibility API.
- `--doctor` checks for project-local DeepFace/Whisper caches and embedding
  cache coverage.
- A FAR/FRR threshold sweep and resubstitution caveat in
  `tools/calibrate_threshold.py`, plus `docs/model-card.md`.
- Tests for runtime isolation, cache coverage, headless chat, model warm-up,
  dominant-face selection, calibration metrics, the NVIDIA adapter, and the
  voice debug gate.
- **Groq** as a supported answer provider (`AI_PROVIDER=groq`), using the
  OpenAI-compatible endpoint at `https://api.groq.com/openai/v1`. Configure
  with `GROQ_API_KEY` and `GROQ_MODEL` (default `llama-3.3-70b-versatile`).
  Groq is subject to cloud rate limits.

### Changed

- Enrollment now accepts one dominant face and ignores sub-1% background
  detections. Two plausibly sized faces are still rejected, so group photos do
  not enroll as one person. No biometric images were modified.
- `--offline` documentation is explicit that it disables cloud answers and
  Google STT but does not block first-run model downloads.
- Startup summaries include camera and recognition tuning without printing
  visitor data.
- `--chat` and `--ask` answer short greetings locally instead of returning the
  generic no-answer fallback.
- Voice listener status prints stay content-free; heard questions and answers
  appear in the terminal only with `VOICE_DEBUG=true`.

### Fixed

- The voice listener stops after five consecutive unexpected errors instead of
  logging forever; the camera keeps running meanwhile.
- `--doctor` no longer presents microphone index 0 as the "default" device when
  SpeechRecognition actually uses the system default input.
- CI `compileall` covers `runtime_env.py` too.
- `.gitignore` now also excludes `*.bak` files such as knowledge backups.
- Voice operation no longer prints raw questions, answers, or rejected
  wake-word transcripts to the terminal.
- The DeepFace weight path is reported correctly as
  `.runtime/deepface/.deepface/weights`, matching DeepFace's nested layout.
- Skipped enrollment images are surfaced by the embedding-cache check instead
  of being hidden behind a raw folder count.
- `run_windows.ps1` no longer treats TensorFlow's stderr warnings as fatal; the
  app starts and its real exit code is preserved.
- `setup_windows.ps1` now restores and verifies the GUI-enabled `opencv-python`
  build after `retina-face` installs the headless OpenCV package. `--doctor`
  reports a `FAIL` for a headless build instead of failing later at `cv2.imshow`.

### Security

- Web search (DuckDuckGo) receives unmatched question text whenever it is
  enabled; `WEB_SEARCH=false` restores a strictly local/on-provider flow.
- `make_archive.ps1` never packages `.env`, `.cache/`, biometric images, or
  virtual-environment/model caches, so shared archives no longer carry secrets
  or personal data.
- Runtime and model caches are now project-local and removable.
- Voice status messages are content-free.
- `.runtime/` is git-ignored and excluded from Ruff/compile checks.

## [2.0.0] - 2026-10-25

This release hardens the reviewed prototype. The local face-recognition,
greeting, verified-FAQ, and optional voice features are preserved; unsafe
defaults and fragile state handling were replaced.

### Added

- Explicit `.env` loading with real environment variables taking precedence.
- Source-checked knowledge schema (`schema_version: 2`) with per-fact
  `source_type`, `verified_on`, `reviewed_by`, aliases, keywords, and
  top-level provenance.
- `tools/audit_knowledge.py` for structure, provenance, uniqueness, and
  paraphrase-recall checks.
- `tools/calibrate_threshold.py` for genuine/impostor score analysis and a
  suggested `SIMILARITY_THRESHOLD` range.
- Duplicate FAQ detection in `python app.py --doctor`.
- Bounded embedding cache with atomic writes and model/detector invalidation.
- Duplicate-image rejection during enrollment so one image cannot enroll two
  people.
- Bounded speech queue, echo guard, and wake-word support.
- Stable identity tracker requiring consecutive confirmations.
- Cloud-provider rate limiting and secret-free provider probes.
- Dependency-free unit suite (139 tests), knowledge-recall regression tests,
  pinned requirements, Ruff configuration, and cross-platform CI.
- Privacy, architecture, configuration, operations, testing, and
  knowledge-curation documentation.

### Changed

- Default `STT_PROVIDER` is `none`; microphone input is opt-in. Google STT is
  explicitly opt-in and is disabled by `--offline`.
- `--offline` disables cloud answer providers and Google STT but keeps local
  Whisper available.
- Recognition defaults are `SIMILARITY_THRESHOLD=0.68` and
  `MATCH_MARGIN=0.08`; invalid or out-of-range values are clamped.
- Verified answers take priority over every provider, so a known FAQ never
  causes network traffic.
- Unknown college-specific questions stay offline instead of being answered
  from model memory.
- General-question prompt context uses a bounded, relevant subset of verified
  facts rather than truncating the whole knowledge base.
- Provider requests bound output tokens, response size, timeout, redirects,
  and endpoint policy (remote endpoints require HTTPS).
- Knowledge questions, aliases, and HOD answers were normalised; duplicate
  entries were merged (131 facts to 121, with 507 unique aliases).
- Canonical names are `Dr. Ananda Babu J` and `Dr. Naveen Kumar C M`.

### Fixed

- DeepFace full-frame "no face" placeholders are rejected instead of becoming
  accepted identities.
- Speech queue overflow no longer blocks the camera loop or loses the newest
  response.
- TTS and microphone activity are coordinated so the assistant does not
  transcribe its own answer.
- Enrollment cache corruption fails closed and rebuilds.
- Provider failures no longer leak questions or API keys into logs.
- Greeting-prefixed questions are no longer mistaken for bare greetings.
- Visitor welcome cooldown now applies across visits.
- Missing or malformed knowledge files fail closed with a safe message.

### Security

- Removed the legacy `app.txt` credential from the working copy. The original
  archive may still contain it; revoke that key separately.
- Added `.gitignore` rules for `.env`, biometric images, caches, and runtime
  artefacts.
- Added a training-data README covering consent, retention, and deletion.
- Documented the privacy boundary: face embeddings stay local; only opted-in
  STT and answer providers can send data off the machine.

### Known limitations

- Recognition is name-level and has no liveness/anti-spoofing protection.
- The office-fallback router is heuristic and can misclassify unusual
  paraphrases.
- Hardware, camera, DeepFace, and audio paths are not exercised by CI; run
  `python app.py --doctor` and the manual checklist in `docs/testing.md`.
- `verified` means source-checked against the supplied attachment, not
  independently confirmed by the college. Time-sensitive details must be
  rechecked with the college office.

## [1.0.0] - 2026-10-11

- Initial prototype: folder-based face enrollment, DeepFace recognition,
  spoken greetings, optional speech input, verified college FAQ answers, and
  Ollama/Gemini/OpenAI-compatible providers.
