# AI system card - espsthuthi2

This document describes the intended use, data, models, evaluation status,
failure modes, and limits of the `espsthuthi2` college-assistant prototype. It
is written to be included as a report appendix. It is not a claim of
production readiness.

## 1. System summary

| Item | Description |
| --- | --- |
| System | Local-first college face-recognition and question-answering assistant |
| Version | 2.0.0 |
| Purpose | Demonstration of face recognition, greeting, verified FAQ retrieval, optional speech, and optional local/cloud language models |
| Primary users | College project team, demonstrators, and evaluators |
| Deployment | Single-user desktop/laptop with a camera and graphical desktop |
| Operating systems | Windows 10/11; Ubuntu/Linux desktop |
| Runtime isolation | Python packages and model caches are stored in the project's `.venv` and `.runtime` folders |

## 2. Intended and non-intended use

### Intended use

- Demonstrating a local face-recognition pipeline with consented enrollment
  images.
- Answering a curated set of verified college FAQ questions from a local
  knowledge file.
- Greeting recognised people or first-time visitors in a controlled demo.
- Demonstrating optional local Whisper speech recognition, speech output, and
  a local or opt-in cloud language model.
- Teaching responsible-AI practices: consent, data minimisation, honest
  limitations, local processing, and reproducible evaluation.

### Non-intended use

Do **not** use this system for:

- attendance, access control, identity verification, payments, or security
  decisions;
- surveillance, continuous tracking, emotion detection, or profiling;
- decisions about admission, employment, discipline, or eligibility;
- identifying people who have not consented;
- unattended kiosk or multi-tenant operation;
- any use where a false accept or false reject could cause harm.

The recognition component has no liveness or anti-spoofing protection. A photo
or replay may be accepted. The question router is heuristic and can
misclassify an unusual paraphrase.

## 3. Components

| Component | Implementation | Notes |
| --- | --- | --- |
| Face detector | DeepFace with the configured OpenCV detector | Frames are processed locally |
| Face embedding | VGG-Face by default (configurable) | 2,622-dimensional embedding; model weights cached under `.runtime/deepface` |
| Identity decision | Cosine similarity + threshold + runner-up margin + consecutive-batch confirmation | Name-level only; no per-face track association |
| Enrollment | One folder per person; one dominant detectable face per image; duplicate-image rejection | Tiny background false positives may be ignored; two plausibly sized faces are rejected. 5-10 varied, consented images per person recommended |
| FAQ retrieval | Conservative lexical retrieval with exact/alias precedence | Not semantic RAG or an LLM knowledge graph |
| Answer provider | Local Ollama by default; offline, Gemini, OpenAI, or compatible endpoint are configurable | Verified FAQ answers never call a provider |
| Speech input | Disabled by default; optional local faster-whisper or Google STT | Google STT sends audio off the machine |
| Speech output | Windows SAPI or pyttsx3 | Local |
| Data store | JSON enrollment images and local embedding cache | No server database |

## 4. Data inventory

| Data | Location | Retention | Leaves the machine? |
| --- | --- | --- | --- |
| Enrollment photographs | `data/training_images/<person>/` | Until consent expires or cleanup | No |
| Face embeddings | `.cache/embeddings.json` | Rebuilt from images; delete with them | No |
| Camera frames | RAM only | Not persisted | No |
| Face names and similarity scores | RAM and OpenCV overlay | Not persisted | No |
| Microphone audio | RAM | Not recorded | Only if Google STT is explicitly enabled |
| Local Whisper model | `.runtime/huggingface` | Until cleanup | Download only; audio remains local |
| DeepFace weights | `.runtime/deepface` | Until cleanup | Download only |
| Questions and answers | Terminal only in text mode; camera overlay | Not persisted | General questions may go to an opted-in provider |
| `.env` | Project root | Until operator deletes it | No |
| Provider credentials | `.env` or process environment | Operator-controlled | Sent only to the configured provider |

Enrollment images and embeddings are biometric data. Consent, retention, and
deletion must be agreed before collection.

## 5. Decision logic and failure behaviour

```text
question
  -> exact/alias verified FAQ?       -> local verified answer
  -> unknown college-specific?       -> local office fallback
  -> provider = offline?             -> local offline fallback
  -> configured provider             -> bounded answer
  -> provider failure                -> local safe fallback
```

Recognition fails closed:

- invalid or zero embeddings are rejected;
- a score below `SIMILARITY_THRESHOLD` is `Unknown`;
- a runner-up gap below `MATCH_MARGIN` is `Unknown`;
- tiny boxes and full-frame no-face placeholders are rejected;
- enrollment rejects images with two plausibly sized faces and accepts only
  one dominant face;
- a name must repeat for `IDENTITY_CONFIRMATIONS` batches;
- stale recognition results expire.

The assistant does not claim to know a person merely because a detector
returned a box.

## 6. Evaluation status

### Automated evaluation

The dependency-free unit suite (171 tests) covers:

- configuration, `.env` parsing, clamping, and secret-free descriptions;
- verified-knowledge loading, provenance, uniqueness, and paraphrase recall;
- FAQ matching, routing, provider safety, transport policy, and rate limits;
- recognition scoring, margin/ties, face-box plausibility, and stale results;
- enrollment discovery, duplicate rejection, dominant-face selection, and cache;
- greeting, speech queue, echo guard, wake word, and voice helpers;
- doctor checks and project-local cache discovery;
- headless chat, cleanup metadata, and runtime environment isolation.

Run:

```text
python -B -m unittest discover -s tests -v
python -B -m tools.audit_knowledge
python -m ruff check .
```

CI runs the dependency-free checks on Ubuntu and Windows with Python 3.11 and
3.12. CI does not exercise a camera, microphone, speakers, TensorFlow, or
DeepFace.

### Recognition evaluation

The current checkout contains **4 people with 1 image each**. All four images
now produce embeddings after tiny OpenCV false positives were filtered, but the
set is still **not** enough for a meaningful threshold calibration:

- no genuine pairs exist within a person;
- impostor pair counts are too small;
- there are no held-out images for an unbiased estimate.

`tools/calibrate_threshold.py` therefore refuses to produce a stable estimate
for this dataset. After collecting at least 2 images for at least 2 people,
run:

```text
python -B -m tools.calibrate_threshold
python -B -m tools.calibrate_threshold --json
```

The tool reports genuine/impostor distributions, equal-error threshold,
FAR/FRR at the equal-error point, a threshold sweep, and a suggested range. It
uses enrollment images (resubstitution), so the numbers are optimistic. For a
report-quality result:

1. Split each person's images into enrollment and held-out sets.
2. Rebuild embeddings from the enrollment set only.
3. Evaluate held-out genuine and impostor pairs.
4. Record people, images, threshold, margin, FAR/FRR, date, and reviewer.
5. Repeat under the lighting, distance, glasses, and background conditions
   expected at the demo.

Do not present the equal-error threshold as an accuracy figure.

### Evaluation results to record

| Metric | Required record |
| --- | --- |
| Dataset | People, images per person, capture conditions, consent status |
| Threshold | `SIMILARITY_THRESHOLD`, `MATCH_MARGIN`, confirmations, interval |
| Face recognition | FAR, FRR, EER, genuine/impostor ranges, confusion examples |
| Robustness | Glasses/no glasses, lighting, distance, rotation, background, similar people, visitor, covered camera, low light |
| Latency | Recognition interval and observed lag on the demo machine |
| FAQ | Recall on curated questions and paraphrases; precision spot-check |
| Routing | Unknown college-question fallback rate; provider failure fallback |
| Privacy | Terminal output contains no transcripts, questions, or keys |

## 7. Known limitations and risks

- **No liveness or anti-spoofing.** Photos and replays may pass.
- **Resubstitution calibration.** Current calibration uses enrollment images;
  it is not a held-out benchmark.
- **Small, shallow dataset.** Four people with one image each cannot represent
  demographic, lighting, pose, or age variation.
- **Name-level identity confirmation.** The tracker counts names across
  batches, not individual faces. Multiple faces can complicate association.
- **No embedding encryption.** Local files depend on OS permissions and
  full-disk encryption.
- **Heuristic college-question routing.** Unusual paraphrases can be
  misclassified and sent to a cloud provider if one is enabled.
- **FAQ retrieval is lexical, not semantic.** A paraphrase with little token
  overlap can miss.
- **No fairness evaluation.** The dataset is too small to assess performance
  across skin tone, gender, age, glasses, disability, or camera quality.
- **Optional cloud exposure.** Cloud answer providers receive a general
  question and a bounded subset of verified FAQ context. Google STT receives
  audio. Both are opt-in.
- **Model weights are third-party artefacts.** Their licences and behaviour
  are outside this project.
- **No absolute offline guarantee on a cold start.** `--offline` disables
  cloud answers and Google STT but does not block first-run model downloads.
  Pre-fetch with `--warmup-models` and verify `--doctor` before the demo.

## 8. Privacy and security controls

- Microphone input is disabled by default (`STT_PROVIDER=none`).
- Cloud answer providers and Google STT are opt-in.
- Verified FAQs are answered locally.
- Unknown college-specific questions never call a provider.
- Provider errors and normal voice operation do not print question text,
  answers, or API keys to the terminal.
- Provider endpoints require HTTPS unless they are loopback.
- Cloud calls are rate-limited and responses are bounded.
- The embedding cache is atomic and invalidated when the model changes.
- All Python packages, pip downloads, temporary files, and model caches can be
  kept inside the checkout via `.venv` and `.runtime`.
- `.runtime/`, `.cache/`, `.env`, and enrollment images are git-ignored.

## 9. Deletion and consent

Before enrollment, record informed consent and a retention date. To remove the
project-local runtime and generated caches:

```powershell
.\cleanup_runtime.ps1 -Force
```

To also remove enrollment images:

```powershell
.\cleanup_runtime.ps1 -Force -DeleteBiometrics
```

To also remove local secrets/configuration:

```powershell
.\cleanup_runtime.ps1 -Force -DeleteEnv
```

Run without `-Force` to preview. Review `docs/privacy.md` and record the
deletion date, operator, and scope. Removing files cannot revoke a credential
that was already exposed; rotate it separately.

## 10. Accountability

- A human operator must remain present during demos.
- The operator decides whether cloud or microphone features are enabled.
- The operator must be able to explain what data is stored, where, why, and for
  how long.
- Reported metrics must state the dataset and evaluation method.
- Claims must not exceed the evidence: this is a name-level demo, not a
  security or attendance system.
