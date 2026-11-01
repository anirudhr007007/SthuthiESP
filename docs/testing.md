# Testing and acceptance

The automated suite is intentionally dependency-free: it does **not** import
OpenCV, DeepFace, TensorFlow, SpeechRecognition, PyAudio, or a real audio
device. This keeps CI fast and makes logic regressions visible before hardware
is available. Hardware paths are covered by the manual checklist below.

## Automated checks

Run from the project root:

```text
python -B -m unittest discover -s tests -v
python -B -m tools.audit_knowledge
python -B -m compileall -q app.py ai_provider.py doctor.py enrollment.py face_utils.py greeting.py recognition_worker.py runtime_env.py settings.py speech_service.py ui_state.py version.py voice_service.py tests tools
python -m ruff check .
```

Expected at release: 171 tests pass, audit reports 121 facts / 507 aliases /
15-of-15 probes, compilation succeeds, and Ruff reports no errors.

### What the unit suite covers

| Area | Representative checks |
| --- | --- |
| Settings | `.env` parsing, environment precedence, aliases, clamping, secret-free descriptions |
| Knowledge loading | explicit `verified: true`, complete fields, placeholders rejected, templates excluded |
| Knowledge recall | every question/alias resolves to its own fact, 18 common paraphrases, names, provenance |
| Matching/routing | exact/alias precedence, greeting prefixes, ambiguity rejected, college questions stay offline |
| Retrieval | exact-first, limit, character budget, stable ordering |
| Providers | Ollama bounds, Gemini header key, OpenAI SDK/HTTP fallback, compatible endpoint, probes |
| Transport | HTTPS policy, URL credentials/query rejection, response size, redirects, secret-free errors |
| Rate limiting | minimum interval, hourly cap/window, reset, local Ollama exemption, safe fallback |
| Recognition | cosine validation, margin, ties, invalid vectors, plausible face area, dominant-face selection, tracker |
| Worker | latest-frame semantics, serial callback, errors once, stop/discard |
| Enrollment | discovery, ignored folders, cache round-trip/invalid entries, atomic save, duplicates, single-dominant-face rules |
| Greeting | per-visit once, presence timeout, minimum gap, visitor cooldown, reset |
| Speech | queue order/overflow, activity token, failure handling, timeout, echo similarity, SAPI mapping |
| Voice | greetings, wake-word boundaries, language code, local greeting answer, provider delegation |
| Doctor | knowledge ambiguity/provenance, missing files, training depth, consent note, model/embedding cache coverage, OpenCV GUI backend, no secret output |
| App helpers | greeting detection, offline overrides, no mutation, OpenCV text wrapping/ASCII replacement, headless chat, model warm-up |

### Knowledge audit details

`tools/audit_knowledge.py` is stricter than the loader: it checks schema,
provenance, source types, duplicate phrases, placeholder text, alias length,
review dates, and paraphrase probes. The probe list is the regression net for
common natural-language phrasing. Add a probe whenever a real visitor phrase
fails to match.

### CI

`.github/workflows/ci.yml` runs on Ubuntu and Windows with Python 3.11 and
3.12:

1. Ruff
2. `compileall`
3. full unit suite
4. knowledge audit

CI does not install TensorFlow/OpenCV/audio packages. A green CI run is
necessary but **not sufficient** for a demo.

## Manual acceptance checklist

Complete this on the machine and with the camera/speakers that will be used.

### Environment

- [ ] Python is 3.11 or 3.12 and `.venv` was created by `setup_windows.ps1`
      (Windows) or the documented manual steps (Linux).
- [ ] `pip check` reports no broken requirements.
- [ ] `python app.py --doctor` has no `FAIL`.
- [ ] `python app.py --version` prints `2.0.0`.
- [ ] `python app.py --warmup-models` reports the face model ready; with
      `STT_PROVIDER=whisper`, it also reports the Whisper model ready.
- [ ] `--doctor` confirms the DeepFace/Whisper caches and embedding cache are
      under `.runtime` / `.cache`.
- [ ] Ollama (if used) is running and the configured model is pulled.

### Camera and recognition

- [ ] The camera window opens at the expected index.
- [ ] Each enrolled person is recognised in normal lighting.
- [ ] Each enrolled person is recognised with glasses/no glasses as applicable.
- [ ] An unenrolled person is shown as `Unknown`.
- [ ] Two similar-looking enrolled people are not confused.
- [ ] Removing a person's images and rebuilding removes recognition.
- [ ] A covered camera shows `No face detected` without crashing.
- [ ] Unplugging/blocking the camera exits cleanly or reports a retry/error.

### Greeting and speech

- [ ] A known person is greeted once per visit, with the correct name.
- [ ] Leaving for longer than `PRESENCE_TIMEOUT` and returning greets again.
- [ ] An unknown visitor is welcomed once per cooldown.
- [ ] No greeting repeats every frame.
- [ ] Speech is audible, intelligible, and not clipped.
- [ ] Queue overflow during rapid events does not hang the UI.
- [ ] `q` and Esc stop the app and release the camera.

### Voice input

For each enabled STT provider:

- [ ] The microphone is detected (or `STT_DEVICE_INDEX` is set).
- [ ] A short greeting gets the local greeting response.
- [ ] A verified FAQ question is answered locally.
- [ ] A general question reaches the intended provider.
- [ ] The assistant does not transcribe its own speech (`ECHO_GUARD=true`).
- [ ] A wake word is required when configured; similar words do not match.
- [ ] `--offline` disables Google STT and keeps Whisper.

### Text and providers

- [ ] `python app.py --ask "Who is the principal of MCE?"` answers locally.
- [ ] `python app.py --chat` starts a text-only session and `/quit` exits.
- [ ] `python app.py --ask "What are the admission fees?"` gives the office
      fallback and makes **no** provider call.
- [ ] `python app.py --ask "What is photosynthesis?"` uses the selected
      provider or the general offline fallback.
- [ ] `--offline` makes no cloud answer or Google STT request. It does **not**
      block first-run DeepFace/Whisper downloads; pre-fetch models with
      `--warmup-models` before claiming a network-free demo.
- [ ] Stopping Ollama produces a safe fallback, not a crash.
- [ ] An invalid cloud key produces a safe fallback and never prints the key.
- [ ] The cloud rate limit produces a safe fallback after the configured cap.

### Privacy

- [ ] `.env` is not tracked by git.
- [ ] Training images are not tracked by git.
- [ ] `.cache/`, `.runtime/`, and logs are not tracked by git.
- [ ] Terminal output from startup/doctor/provider failure/voice operation
      contains no keys, visitor questions, or transcripts.
- [ ] Google STT is off unless the operator has announced it.
- [ ] The retention/deletion plan is agreed and recorded.
- [ ] `.\\cleanup_runtime.ps1` previews the generated runtime before deletion.
- [ ] `--offline` is used when the demo must keep all question/audio data local;
      model caches were pre-fetched.

## Negative and failure tests

Run these at least once before a public demo:

| Scenario | Expected |
| --- | --- |
| Delete `data/college_knowledge.json` | App warns and continues with no verified facts |
| Corrupt the knowledge JSON | Loader fails closed; no invented answers |
| Corrupt `.cache/embeddings.json` | Cache is ignored and rebuilt |
| Move/rename the training folder | Startup error with a clear message |
| Set `SIMILARITY_THRESHOLD=9` | Warning and clamped to `1.0` |
| Set `CAMERA_INDEX=99` | Clear camera error, clean exit |
| Set `STT_PROVIDER=google` + `--offline` | Google disabled, note printed |
| Stop Ollama mid-run | Safe offline answer, no auto-switch |
| Use a remote `http://` endpoint | Configuration error, no request |
| Put a key in `AI_BASE_URL` | URL rejected |
| Trigger queue overflow | Oldest queued answer dropped, UI responsive |
| Cover the camera | No stale identity/greeting persists |

## Threshold acceptance

After running `tools.calibrate_threshold.py`, record:

- number of people and images
- genuine/impostor ranges and the FAR/FRR threshold sweep
- chosen threshold and margin
- false accept/reject observations from the manual set
- date and reviewer

The tool uses the enrollment images themselves (resubstitution), so treat its
numbers as optimistic. For a report-quality result, enroll from one set and
evaluate on held-out images. The current dataset has four people with one image
each, so it cannot produce genuine pairs or a meaningful calibration.

Do not accept a threshold solely because the tool suggests it. The manual
similar-person test is mandatory.

## Adding tests

- Put logic tests in `tests/test_*.py` and keep them dependency-free.
- Use fakes for providers, DeepFace, and audio backends.
- Never require a camera, microphone, network, or API key.
- Add a knowledge probe when fixing a real paraphrase miss.
- Add a regression test for every bug fix.
- Prefer asserting user-visible behaviour and fail-closed outcomes over
  implementation details.

## CI limitations

CI cannot validate:

- camera drivers and permissions
- DeepFace/TensorFlow model loading, weights, and project-local caches
- SAPI/pyttsx3 voices and audio devices
- microphone/STT accuracy and echo behaviour in a real room
- GPU acceleration or memory pressure
- real provider credentials or quotas

Those require the manual checklist and should be repeated on the demo machine
after any dependency or OS change.
