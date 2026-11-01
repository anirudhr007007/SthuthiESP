# Architecture

The project is deliberately small and flat so a college team can read it
without a framework. Each module owns one concern and imports the heavy
camera/ML libraries lazily so the unit suite stays dependency-free.
`runtime_env.py` redirects pip, temporary, and model caches into `.runtime`
before those libraries are imported.

## Runtime flow

```text
                         +----------------------+
                         |      settings.py     |
                         |  env/.env + clamping |
                         +----------+-----------+
                                    |
                                    v
+-------------+     +---------------------------+     +-------------------+
|  app.py     |---->|      enrollment.py        |---->| DeepFace + OpenCV |
| CLI / loop  |     | images, cache, embeddings |     | (lazy imports)    |
+------+------+     +---------------------------+     +-------------------+
       |
       |  camera frames
       v
+----------------------+       +----------------------+
| recognition_worker   |------>|     face_utils.py    |
| latest-frame thread  |       | cosine, margin, box  |
+----------+-----------+       | StableIdentityTracker|
           |                   +----------------------+
           v
+----------------------+       +----------------------+
|     greeting.py      |       |    speech_service    |
| visit/cooldown state |       | bounded TTS queue    |
+----------------------+       +----------+-----------+
                                         ^
                                         |
+----------------------+       +---------+------------+
|    voice_service     |------>|       app.py         |
| STT, wake, echo      |       | state + display      |
+----------+-----------+       +---------+------------+
           |                             |
           v                             v
+----------------------+       +----------------------+
|    ai_provider.py    |<------| verified FAQ / model |
| matcher + providers  |       | answer routing       |
+----------------------+       +----------------------+
```

## Modules

| Module | Responsibility |
| --- | --- |
| `app.py` | CLI parsing, overrides, startup summary, headless chat, model warm-up, camera loop, drawing, orchestration |
| `settings.py` | `.env` parser, environment configuration, bounds/clamping, safe description |
| `runtime_env.py` | Redirects DeepFace, Hugging Face, Keras, pip, Torch, and temp caches into `.runtime` |
| `enrollment.py` | Image discovery, single-dominant-face validation, embedding extraction, cache |
| `face_utils.py` | Normalisation, cosine similarity, margin decision, face-box validation, dominant-face selection, name tracker |
| `recognition_worker.py` | One background inference thread; keeps only the newest pending frame |
| `greeting.py` | Presence timeout, per-visit greeting state, visitor cooldown, minimum gap |
| `speech_service.py` | TTS backends, bounded queue, activity token, recent-answer history |
| `voice_service.py` | Microphone listener, Whisper/Google STT, wake word, echo guard |
| `ai_provider.py` | FAQ matching/retrieval, offline routing, `AnswerResult`, provider adapters, transport policy, rate limiting |
| `doctor.py` | Preflight checks for Python, files, enrollment, modules, model/embedding caches, camera, mic, provider, knowledge |
| `ui_state.py` | Thread-safe activity/answer snapshot for the OpenCV overlay |
| `tools/audit_knowledge.py` | Knowledge structure, provenance, uniqueness, paraphrase probes |
| `tools/calibrate_threshold.py` | Genuine/impostor score analysis, threshold sweep, recommendation |

## Startup sequence

1. `main()` parses arguments and handles `--version` immediately.
2. `configure_runtime_environment()` redirects caches into `.runtime` before
   DeepFace, Whisper, or any other heavy library is imported.
3. `Settings.from_env(PROJECT_PATH)` loads `.env` (without overriding real
   environment variables), applies defaults, clamps ranges, and prints only
   invalid/clamping warnings.
4. CLI overrides are applied immutably with `dataclasses.replace`.
5. `load_knowledge()` keeps only explicitly verified, complete, non-placeholder
   facts. Invalid files fail closed with no facts.
6. `--doctor` runs diagnostics and exits.
7. `--warmup-models` loads the configured models and exits.
8. `--ask` answers one question and exits; `--chat` runs a text-only loop.
   Both run **before** camera imports.
9. Otherwise enrollment embeddings load (or rebuild from cache), the speech
   worker starts, an optional voice listener starts, and the camera loop runs.

## Recognition pipeline

Enrollment uses a stricter rule than the camera path: exactly one detection
must cover at least 2% of the image, and every other detection must be below
1%. This accepts a clear face with tiny OpenCV false positives while rejecting
group photos and two plausibly sized faces.

1. The camera loop reads a frame and downscales it to `MAX_FRAME_WIDTH`.
2. Frames are submitted at most every `RECOGNITION_INTERVAL` seconds.
3. `LatestFrameWorker` runs inference on a background thread. If inference is
   slow, only the newest pending frame is kept; stale frames are dropped.
4. DeepFace returns one embedding per detected face. `is_plausible_face_area`
   rejects tiny boxes and DeepFace's full-frame no-face placeholder.
5. `identify()` compares the embedding with each person's enrolled embeddings
   using cosine similarity:
   - best score must be `>= SIMILARITY_THRESHOLD`
   - best score minus runner-up must be `>= MATCH_MARGIN`
   - invalid vectors, ties, and empty enrollment fail closed
6. `StableIdentityTracker` requires the same accepted name in
   `IDENTITY_CONFIRMATIONS` consecutive batches.
7. Results older than `RECOGNITION_RESULT_MAX_AGE` are discarded so a stopped
   camera cannot keep an old identity alive.
8. `GreetingController` greets a known person once per visit, or welcomes an
   unknown visitor once per cooldown. Known greetings win when both are
   present; the visitor welcome follows on a later update.

## Answer routing

```text
question
  |
  +-- empty / no meaningful tokens -----------> safe offline answer
  |
  +-- verified question or alias match -------> verified local answer (no network)
  |
  +-- unknown college-specific question ------> college-office fallback (no network)
  |
  +-- AI_PROVIDER=offline --------------------> general offline answer
  |
  +-- selected provider ----------------------> bounded prompt -> answer
                                               (failure -> safe offline answer)
```

### FAQ matching

Normalisation is NFKC + case-fold + punctuation removal. Exact question and
alias matches win. A greeting prefix is removed only when it is a true prefix;
a bare greeting fact cannot swallow a longer question. Fuzzy scoring:

- requires at least two meaningful overlapping terms, excluding generic terms
  such as `college`, `official`, `department`, and `hod`
- requires at least 50% query coverage unless a multi-word keyword phrase
  matches
- applies a small canonical-token map for plural/synonym differences
- rejects equal top scores rather than guessing

This is intentionally conservative. False negatives fall back to "check with
the college office" or the model; false positives would answer the wrong
question, which is worse.

### Prompt retrieval

`select_relevant_facts()` ranks verified facts by exact phrase match, then
token overlap and keyword phrases. It sends at most `MAX_CONTEXT_FACTS` (8)
facts and `MAX_CONTEXT_CHARS` (6,000) characters.

### Optional web context

When no verified fact matches and a cloud/local provider is selected,
`_search_web_context()` may fetch at most three DuckDuckGo snippets with the
`ddgs` package and label them unverified in the prompt. This only runs for
unmatched questions; `WEB_SEARCH=false`, `AI_PROVIDER=offline`, or `--offline`
disables it entirely. This replaced sending a
truncated dump of the whole knowledge base, which hid relevant facts.

### Provider adapters

| Provider | Transport | Notes |
| --- | --- | --- |
| `ollama` | Loopback HTTP `/api/chat` | `num_predict` bounded; local, no cloud rate limit |
| `gemini` | HTTPS `generateContent` | API key in `x-goog-api-key`, never URL |
| `openai` | OpenAI SDK Responses API, or HTTPS `/v1/responses` fallback | `max_retries=0`, bounded output |
| `groq` | HTTPS `/v1/chat/completions` | Bearer header; cloud rate limits apply |
| `nvidia` | HTTPS `/v1/chat/completions` | NVIDIA NIM; `AI_BASE_URL` defaults to the hosted API; cloud rate limits |
| `compatible` | HTTPS or loopback `/chat/completions` | Bearer header; remote HTTPS only |

All adapters bound timeout, response bytes, answer length, and redirects.
Errors are categorised (`timeout`, `network`, `http`, `response`, `rate_limit`,
`configuration`, `redirect`, `provider`) and logged without the question or
credentials. There is no automatic provider fallback: a failure returns the
safe offline answer. `answer_question_result()` also returns an `AnswerResult`
with a privacy-safe source category (`verified_faq`, `college_fallback`,
`offline_fallback`, `provider`, or `provider_fallback`) for chat/reporting;
`answer_question()` remains the string-only compatibility wrapper.

## Threading and shutdown

| Thread | Purpose | Shutdown |
| --- | --- | --- |
| Main | Camera loop, OpenCV UI, greeting/state orchestration | `q`/Esc sets the stop event |
| `recognition_worker` | DeepFace inference, newest-frame semantics | `stop(timeout=2)` |
| Speech worker | Serialises TTS through one backend | queue sentinel + `stop()` |
| Voice listener (optional) | Microphone capture, STT, wake/echo, answer call | daemon; observes the stop event |

Shared state is limited to `AssistantState` (lock-protected snapshot),
`SpeechService` (condition-protected queue/history), and
`LatestFrameWorker` (lock/event-protected latest frame and result). The TTS
queue is bounded; overflow drops the oldest queued message rather than growing
without limit. `wait_until_idle()` treats a failed backend as idle so the
microphone listener cannot hang.

## Failure philosophy

- **Fail closed on identity:** invalid vectors, ties, missing faces, and
  implausible boxes are `Unknown`, never a guess.
- **Fail closed on facts:** unverified or incomplete knowledge is ignored.
- **Fail closed on college questions:** unknown college facts go to the office
  fallback, not model memory.
- **Fail safe on providers:** errors return a bounded offline answer with no
  secret/question leakage and no provider switch.
- **Fail visibly in diagnostics:** `--doctor` reports `FAIL`/`WARN`/`SKIP`
  instead of silently running with a missing camera, module, or data file.

## Deliberate limits

- The tracker confirms a **name**, not a specific face track. It is not a
  multi-face identity association system.
- There is no liveness detection, anti-spoofing, or template encryption.
- The college-question router is heuristic; unusual paraphrases can be
  misclassified.
- The app is a single-process desktop tool, not a multi-tenant service.
- Hardware paths are validated manually, not in CI.
