# espsthuthi2 — local-first college face assistant

A production-hardened college-project assistant that combines:

- **Local face recognition** from folders of consented enrollment images
- **Spoken greetings** for recognised people and first-time visitors
- **Verified college FAQ answers** that never call a provider
- **Optional voice questions** through a microphone
- **Optional local or cloud language models** for general questions
- **Privacy-safe defaults**: microphone off, cloud answers opt-in, offline mode

It runs on a normal Windows laptop/desktop or Ubuntu/Linux mini-PC with a
webcam, speakers, and a graphical desktop. A Raspberry Pi is not required.

> **Status:** `2.1.0`. The hardware, camera, DeepFace, and audio paths require
> a real machine; CI covers the dependency-free logic, knowledge recall, and
> provider safety. See [docs/testing.md](docs/testing.md) for the manual
> acceptance checklist.

---

## What it does

1. Opens the camera and recognises faces against local enrollment images.
2. Confirms an identity across several inference batches, then greets the
   person once per visit.
3. Welcomes an unknown visitor once per cooldown period.
4. Answers typed (`--ask`) or spoken questions:
   - verified FAQ → answered locally from `data/college_knowledge.json`
   - unknown college-specific question → safe "check the college office"
     fallback, no provider call
   - general question → selected provider (local Ollama by default)
5. Speaks answers through Windows SAPI or `pyttsx3`, with a bounded queue and
   an echo guard so the microphone does not transcribe the assistant.

### Privacy at a glance

| Data | Default | Leaves the machine? |
| --- | --- | --- |
| Camera frames and face embeddings | Local | No |
| Enrollment images and embedding cache | Local | No |
| Verified FAQ answers | Local | No |
| Microphone audio | Off (`STT_PROVIDER=none`) | Only if Google STT is enabled |
| Local Whisper audio | Opt-in | No |
| General questions to Ollama | Local | No |
| General questions to cloud providers | Opt-in | Yes, including retrieved FAQ context |
| Web search context for unmatched questions | On (`WEB_SEARCH=false`, `--offline`, or `AI_PROVIDER=offline` disables it) | Yes — the question text goes to DuckDuckGo |

Read [docs/privacy.md](docs/privacy.md) before deploying with real visitors.

---

## Requirements

- **Python 3.11 or 3.12** (TensorFlow/DeepFace compatibility; 3.13 is not
  recommended)
- A webcam and a graphical desktop (OpenCV opens a window)
- Speakers for speech output
- A microphone only if voice input is enabled
- About 2.5-3 GB of disk for the Python runtime; model weights add roughly
  600 MB for VGG-Face and 500 MB for Whisper `small`
- 8 GB RAM recommended; the first DeepFace or Whisper run downloads weights
  unless they were pre-fetched with `--warmup-models`
- Optional: [Ollama](https://ollama.com/) for free local general answers

Windows and Linux are supported. macOS is not part of the tested matrix.

---

## Quick start

### Windows

```powershell
.\setup_windows.ps1
Copy-Item .env.example .env
.\run_windows.ps1 --doctor
.\run_windows.ps1 --warmup-models
.\run_windows.ps1
```

`setup_windows.ps1` installs every optional dependency group (core, local
Whisper, cloud SDK, and Ruff) into `.venv`. Pip downloads, temporary files, and
all model caches are redirected to `.runtime` in the project folder. It does
not install or upgrade global Python packages.

If PowerShell blocks scripts, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

To create the venv manually, set `PIP_CACHE_DIR` to
`.runtime\pip-cache` first and install the same four requirement files shown
in [docs/operations.md](docs/operations.md).

### Linux

```bash
sudo apt install portaudio19-dev espeak-ng   # Debian/Ubuntu example
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-linux.txt

cp .env.example .env
chmod +x run_linux.sh
python app.py --doctor
python app.py
```

After the first setup you can use the wrappers:

```powershell
.\run_windows.ps1 --doctor
.\run_windows.ps1
```

```bash
./run_linux.sh --doctor
./run_linux.sh
```

The wrappers expect `.venv` in the project root, point model caches at
`.runtime`, and forward all arguments.

### Local-only runtime

Everything installed for this checkout is removable in one step:

```powershell
.\cleanup_runtime.ps1            # show what would be deleted
.\cleanup_runtime.ps1 -Force     # delete .venv, .runtime, and caches
```

Enrollment images are retained by default. Add `-DeleteBiometrics` only after
checking consent and retention, and `-DeleteEnv` to remove local secrets.
Source code and the verified knowledge file are never removed. See
[docs/privacy.md](docs/privacy.md) for the full deletion procedure.

---

## First run checklist

1. **Add enrollment images** to `data/training_images/Person Name/`.
   Use 5-10 varied images with one dominant face per person. Read
   [data/training_images/README.md](data/training_images/README.md) first;
   these are biometric images and require consent.
2. **Copy `.env.example` to `.env`** and adjust the camera index, paths, and
   provider. `.env` is git-ignored; real environment variables override it.
3. **Run diagnostics:**
   ```text
   python app.py --doctor
   ```
   Fix every `FAIL` before a demo. `WARN` items are usually quality advice.
4. **Try a question without the camera:**
   ```text
   python app.py --ask "Who is the principal of MCE?"
   python app.py --ask "What is photosynthesis?"
   ```
5. **Pre-download model weights before an offline demo:**
   ```text
   python app.py --warmup-models
   ```
   This loads the configured DeepFace model and, when
   `STT_PROVIDER=whisper`, the local Whisper model. It needs no camera.
6. **Run the camera app** and press `q` in the OpenCV window to quit.

The first recognition run may take a while while DeepFace downloads weights.
Weights are cached under `.runtime/deepface`; faster-whisper models are cached
under `.runtime/huggingface`. Embeddings are cached in
`.cache/embeddings.json`; use `--rebuild-embeddings` after changing the face
model or detector.

---

## Command-line options

| Option | Meaning |
| --- | --- |
| `--doctor` | Check Python, files, enrollment images, modules, project-local model caches, camera, microphone, and provider configuration |
| `--ask "QUESTION"` | Answer one question, print it, and exit. No camera needed |
| `--chat` | Start a text-only headless chat. No camera, microphone, or speech |
| `--warmup-models` | Download/load the configured face and local Whisper models, then exit |
| `--offline` | Force local verified answers; disable cloud providers and Google STT, keep local Whisper. Does not block first-run model downloads |
| `--no-voice` | Disable both microphone and speech output for this run |
| `--camera-index N` | Override `CAMERA_INDEX` |
| `--stt-provider NAME` | Override `STT_PROVIDER` (`none`, `whisper`, `google`) |
| `--wake-word "PHRASE"` | Require a leading wake phrase; an empty value disables it |
| `--ai-provider NAME` | Override `AI_PROVIDER` (`ollama`, `offline`, `gemini`, `openai`, `groq`, `nvidia`, `compatible`) |
| `--rebuild-embeddings` | Ignore and rebuild the local embedding cache |
| `--version` | Print the project version |
| `-h`, `--help` | Show the built-in help |

Examples:

```text
python app.py --offline
python app.py --chat
python app.py --ask "Who is the HOD of CSE?"
python app.py --no-voice --camera-index 1
python app.py --stt-provider whisper --wake-word "hey assistant"
python app.py --warmup-models
python app.py --rebuild-embeddings
```

---

## Answer providers

### Local Ollama (default, free)

Install Ollama, then:

```text
ollama pull llama3.2:3b
ollama serve
python app.py --ask "Explain photosynthesis in simple terms."
```

Keep `AI_PROVIDER=ollama` and `OLLAMA_URL=http://127.0.0.1:11434`. The adapter
bounds generation length and timeout, and local Ollama is not subject to cloud
rate limits.

### Offline only

```text
AI_PROVIDER=offline
```

or use `--offline`. Verified FAQs still work; unknown general questions get a
safe offline message. This mode stops cloud answers and Google STT, but it is
not a network sandbox: DeepFace or Whisper can still download a missing model
on first use. Run `python app.py --warmup-models` before an offline demo.

### Gemini / OpenAI / compatible endpoints (opt-in)

```text
pip install -r requirements-cloud.txt
```

Then set, for example:

```text
AI_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
```

or:

```text
AI_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

### Groq (fast OpenAI-compatible API)

```text
AI_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

Groq provides an OpenAI-compatible endpoint with fast inference. The adapter
uses `https://api.groq.com/openai/v1/chat/completions`.

### NVIDIA NIM (hosted models)

```text
AI_PROVIDER=nvidia
AI_API_KEY=nvapi-...
AI_MODEL=z-ai/glm-5.3-flash
```

NVIDIA NIM exposes OpenAI-compatible chat completion models at
`https://integrate.api.nvidia.com/v1` (the default; override `AI_BASE_URL` for
a self-hosted NIM deployment). `AI_MODEL` is a build.nvidia.com model id such
as `z-ai/glm-5.3-flash`, `meta/llama-3.3-70b-instruct`, or
`nvidia/llama-3.3-nemotron-super-49b-v1`. NVIDIA is treated as a cloud
provider and is rate-limited like Gemini/OpenAI/Groq. NVIDIA's hosted speech
service is a separate streaming/gRPC API, so voice input stays on local
Whisper or Google STT.

or an OpenAI-compatible endpoint:

```text
AI_PROVIDER=compatible
AI_BASE_URL=https://your-endpoint.example/v1
AI_API_KEY=...
AI_MODEL=...
```

Remote endpoints must use HTTPS. Plain HTTP is accepted only for loopback
(LM Studio, local servers). Cloud providers (Gemini, OpenAI, Groq, NVIDIA)
are rate-limited by
`CLOUD_MIN_INTERVAL` and `CLOUD_MAX_CALLS_PER_HOUR`. Provider errors never
auto-switch to a different provider and never print the question or key.

---

## Voice input

Voice input is **off by default**. Speech output remains on unless
`--no-voice` is used.

### Local Whisper (recommended for privacy)

```text
pip install -r requirements-offline-voice.txt
```

```text
STT_PROVIDER=whisper
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

The model downloads on first use into `.runtime/huggingface`. Pre-fetch it
with `python app.py --warmup-models` (with `STT_PROVIDER=whisper`). Microphone
audio stays on the machine, and `--offline` keeps Whisper available.

### Google speech recognition (opt-in, sends audio to Google)

```text
STT_PROVIDER=google
STT_LANGUAGE=en-IN
```

The README and `--doctor` warn about this. `--offline` disables it
automatically and suggests Whisper.

### Wake word and echo guard

```text
WAKE_WORD=hey assistant
ECHO_GUARD=true
```

The wake word must be a leading phrase. `ECHO_GUARD` suppresses transcription
while speech is playing and ignores transcripts that closely match recent
answers. It is a usability safeguard, not a security boundary.

---

## Verified college knowledge

`data/college_knowledge.json` is curated, source-checked FAQ data. Every
loadable fact must be explicitly `"verified": true` and contain a question,
answer, source, source type, review date, reviewer, aliases, and keywords.
Incomplete facts and unverified templates are ignored at load time.

```text
python -B -m tools.audit_knowledge
python app.py --doctor
```

To add or change a fact, follow [docs/knowledge-base.md](docs/knowledge-base.md).
Do not mark a fact verified without checking a source.

The 121 facts and 507 aliases are regression-tested: every curated question
and alias must resolve to its own fact, and common paraphrases must hit the
intended answer. Current provenance is a source cross-check against the
supplied MCE FAQ attachment and public MCE pages checked on 2026-10-25 — not
independent confirmation by the college. Recheck time-sensitive details.

---

## Face recognition tuning

Defaults are `SIMILARITY_THRESHOLD=0.68` and `MATCH_MARGIN=0.08`. Values are
clamped to `0.0-1.0`. The app confirms an identity across
`IDENTITY_CONFIRMATIONS` batches before greeting.

Calibrate on your own enrollment set:

```text
python -B -m tools.calibrate_threshold
python -B -m tools.calibrate_threshold --json
```

The tool reports genuine/impostor score distributions, an equal-error
threshold, FAR/FRR across a threshold sweep, and a suggested range. It uses
the enrollment images themselves (resubstitution), so the numbers are
optimistic. Use separate held-out images for report-quality results. Treat the
output as a starting point, not a guarantee. Use a higher threshold and larger
margin if different people are confused.

The current checkout has 4 people with 1 image each, which is enough for a
pipeline smoke test but not a meaningful calibration. Add images before
quoting FAR/FRR. See [docs/model-card.md](docs/model-card.md) for the evaluation
protocol and limitations.

Recognition is **name-level**, not face tracking, and has **no liveness or
anti-spoofing protection**. Do not use it for access control, attendance, or
any security decision.

---

## Tests and diagnostics

The unit suite uses only the standard library, so it runs without TensorFlow,
OpenCV, or audio devices:

```text
python -B -m unittest discover -s tests -v
python -B -m tools.audit_knowledge
python -B -m compileall -q app.py ai_provider.py doctor.py enrollment.py face_utils.py greeting.py recognition_worker.py runtime_env.py settings.py speech_service.py ui_state.py version.py voice_service.py tests tools
```

Lint:

```text
pip install -r requirements-dev.txt
python -m ruff check .
```

CI runs lint, compilation, the full unit suite, and the knowledge audit on
Ubuntu and Windows with Python 3.11 and 3.12. Hardware paths still need the
manual checklist in [docs/testing.md](docs/testing.md).

---

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named cv2` / `deepface` / `tensorflow` | Activate the venv and install `requirements-windows.txt` or `requirements-linux.txt` |
| `Could not open camera` | Correct `CAMERA_INDEX`; close other camera apps; Windows camera privacy settings; Linux `/dev/video*` permissions |
| DeepFace Keras/TensorFlow error | Use Python 3.11/3.12 and keep matching `tensorflow==2.17.0` + `tf-keras==2.17.0` |
| No microphone input | Voice defaults to off. Set `STT_PROVIDER=whisper` or `google`; install PyAudio/PortAudio; check `STT_DEVICE_INDEX` |
| No speech output | `ENABLE_VOICE=true`, system volume, Windows SAPI voices, Linux espeak-ng, `SPEECH_OUTPUT`/`SPEECH_VOICE` |
| Ollama connection refused | Run `ollama serve`; confirm `OLLAMA_URL`; `ollama pull <model>` |
| Ollama model not pulled | `ollama list`, then pull the exact `OLLAMA_MODEL` |
| Face never recognised | More/varied images, better lighting, lower threshold slightly, run calibration |
| Different people confused | Raise `SIMILARITY_THRESHOLD` and `MATCH_MARGIN`; add distinct images |
| Wrong/garbled TTS voice | List installed voices; Windows SAPI and Linux `pyttsx3` discover different names |
| `--ask` works but camera fails | `--ask` does not need camera modules; install the full requirements for the GUI |
| Slow first startup | DeepFace weights and embedding extraction are one-time; later runs use the cache |

More detail: [docs/operations.md](docs/operations.md).

---

## Project layout

```text
espsthuthi2/
  app.py                  camera loop, CLI, greeting/voice orchestration
  ai_provider.py          FAQ matcher, retrieval, providers, transport safety
  enrollment.py           image discovery, embeddings, cache
  face_utils.py           similarity, match margin, face-box checks, tracker
  greeting.py             visit state and greeting cooldowns
  speech_service.py       bounded TTS queue, echo metadata
  voice_service.py        STT, wake word, echo guard
  recognition_worker.py   latest-frame background inference worker
  runtime_env.py          redirects pip/model/temp caches into .runtime
  settings.py             env/.env configuration and clamping
  doctor.py               preflight diagnostics and model-cache checks
  data/
    college_knowledge.json
    training_images/      local biometric images (git-ignored)
  tools/
    audit_knowledge.py
    calibrate_threshold.py
  tests/                  dependency-free unit and data regression tests
  docs/                   architecture, privacy, model card, operations, testing
  setup_windows.ps1       project-local dependency installer
  cleanup_runtime.ps1     dry-run/forced runtime deletion
  .github/workflows/ci.yml
  requirements*.txt
  .venv/                  local packages (git-ignored, removable)
  .runtime/               local models, pip, and temp caches (git-ignored)
```

Module details: [docs/architecture.md](docs/architecture.md).

---

## Security and privacy actions

- The original archive contained a credential in `app.txt`. It has been
  removed from this working copy. **Revoke that key separately** if the
  original archive was shared or run; deleting the file does not revoke it.
- Never put secrets in `app.txt`, `api.txt`, source code, screenshots, or a
  public repository. Use `.env` or environment variables.
- `.gitignore` excludes `.env`, training images, caches, recordings, and model
  downloads. Verify with `git status` before publishing.
- Biometric images require informed consent and a retention/deletion plan.
- Use `--offline` for demos where no visitor data should leave the machine,
  and pre-fetch models with `--warmup-models` first.
- Use `.\cleanup_runtime.ps1` to remove dependencies and caches after testing;
  add `-DeleteBiometrics` only when consent/retention permits.

See [docs/privacy.md](docs/privacy.md) for the full data-flow and threat model.

---

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration reference](docs/configuration.md)
- [Knowledge base curation](docs/knowledge-base.md)
- [Operations and deployment](docs/operations.md)
- [Privacy and security](docs/privacy.md)
- [AI system/model card](docs/model-card.md)
- [Testing and acceptance](docs/testing.md)
- [Changelog](CHANGELOG.md)

---

## Licence

MIT. See [LICENSE](LICENSE). The licence does not cover personal data such as
enrollment photographs, embeddings, or recordings.
