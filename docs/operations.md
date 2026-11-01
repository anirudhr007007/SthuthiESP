# Operations guide

This guide covers installation, first run, demo preparation, calibration,
updates, and failure recovery. For configuration values see
[configuration.md](configuration.md); for the data boundary see
[privacy.md](privacy.md).

## 1. Supported environment

| Component | Supported |
| --- | --- |
| Python | 3.11 or 3.12 |
| OS | Windows 10/11, Ubuntu/Linux desktop |
| Camera | Built-in or USB webcam visible to OpenCV |
| Audio | Speakers; microphone only for voice input |
| Desktop | Required (OpenCV window) |
| Optional | Ollama for local general answers, faster-whisper for local STT |

The application is a single-user desktop tool. It is not designed for a
multi-tenant server, remote camera streams, or unattended kiosk mode.

## 2. Install

### Windows

```powershell
cd D:\path\to\espsthuthi2
.\setup_windows.ps1
Copy-Item .env.example .env
```

`setup_windows.ps1` creates `.venv`, installs the core, local-Whisper, cloud,
and development requirement groups, runs `pip check`, and redirects pip,
temporary, and model caches into `.runtime`. It does not modify global Python
packages. For a minimal install, install only the requirement files you need
with the same `PIP_CACHE_DIR` set to `.runtime\pip-cache`.

### Linux

```bash
cd /path/to/espsthuthi2
sudo apt install portaudio19-dev espeak-ng
mkdir -p .runtime/pip-cache .runtime/tmp
export PIP_CACHE_DIR="$PWD/.runtime/pip-cache"
export TMPDIR="$PWD/.runtime/tmp"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-linux.txt -r requirements-offline-voice.txt \
  -r requirements-cloud.txt -r requirements-dev.txt
cp .env.example .env
```

The application redirects DeepFace, Whisper, Keras, Matplotlib, Torch, and
temporary caches into `.runtime` at startup. On Linux, set `PIP_CACHE_DIR` as
shown above if you also want pip downloads kept inside the checkout.

If `python3.12` is not available, install it from your distribution or
`pyenv`. Do not use Python 3.13 for TensorFlow 2.17/DeepFace 0.0.93.

## 3. Configure

Edit `.env` (never commit it):

```text
CAMERA_INDEX=0
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
STT_PROVIDER=none
```

Relative paths are resolved from the project root. Use absolute paths for a
shared data directory if needed.

### Ollama

Ollama is optional. Its server, not this project, decides where model files are
stored; a normal `ollama pull` uses Ollama's user-wide store (for example
`%USERPROFILE%\.ollama\models`), which is **outside** `.runtime` and is not
removed by `cleanup_runtime.ps1`.

For a project-local store, start a dedicated server on another loopback port:

```powershell
$env:OLLAMA_MODELS = "$PWD\.runtime\ollama\models"
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama serve
```

In a second terminal, pull the model from that server:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama pull llama3.2:3b
```

Then set `OLLAMA_URL=http://127.0.0.1:11435` in `.env`. If you use an existing
user-wide Ollama server instead, that model store remains outside this project
and is outside this project's cleanup scope.

Check:

```text
python app.py --doctor
```

The doctor probes loopback Ollama by default but does not send cloud requests.

## 4. Enroll people

Create one folder per person:

```text
data/training_images/
  Person Name/
    front.jpg
    left.jpg
    glasses.jpg
```

Guidance:

- 5-10 varied images per person
- one clearly dominant face per image; two plausibly sized faces are rejected
- tiny background detections may be ignored, but do not rely on that
- different lighting, angle, expression, and glasses/no-glasses
- no group photos, screenshots, or heavy filters
- consent recorded before capture

Read [../data/training_images/README.md](../data/training_images/README.md).
After adding or removing images:

```text
python app.py --rebuild-embeddings
```

The command rebuilds the cache and then continues to the camera app; quit with
`q` after the load messages. The cache is keyed by image content hash plus face
model/detector. Identical images used for two people are enrolled only for the
first person, and a warning is printed. `python app.py --doctor` warns when the
cache covers fewer images than the enrollment folders contain.

## 5. First run

```text
.\run_windows.ps1 --doctor
.\run_windows.ps1 --warmup-models
.\run_windows.ps1 --ask "Who is the principal of MCE?"
.\run_windows.ps1 --chat
.\run_windows.ps1
```

On Linux use `.venv/bin/python app.py` with the same arguments.

Expected:

- `--doctor` reports `FAIL` for missing modules/data and `WARN` for shallow
  enrollment, a missing consent note, an uncached model, incomplete embedding
  coverage, or a missing Ollama model.
- `--warmup-models` loads the configured DeepFace model and, when
  `STT_PROVIDER=whisper`, the faster-whisper model. It needs no camera.
- The `--ask` FAQ answer is local and immediate.
- `--chat` starts a text-only session; `/quit`, `quit`, or `exit` stops it.
- The general question uses Ollama if it is running; otherwise it prints a
  safe offline answer.
- The camera window shows status and detected names; `q` or Esc exits.

The first DeepFace call downloads weights into `.runtime/deepface`; the first
Whisper call downloads into `.runtime/huggingface`. Do this before a demo.

## 6. Demo runbook

### Recommended privacy-first demo

```text
python app.py --warmup-models
python app.py --offline
```

or use a wrapper:

```powershell
.\run_windows.ps1 --warmup-models
.\run_windows.ps1 --offline
```

```bash
./run_linux.sh --offline
```

Verified FAQs work. General questions get the offline fallback. Enable Whisper
if you need voice input without Google. `--offline` stops cloud answers and
Google STT but is **not** a network sandbox; the warm-up run ensures missing
models are already cached.

### Checklist before visitors arrive

- [ ] `python app.py --doctor` has no `FAIL`.
- [ ] Camera opens and the window is visible.
- [ ] Enrollment images match the people expected.
- [ ] Greeting volume is audible and not distorted.
- [ ] A known FAQ and a general question both answer correctly.
- [ ] Provider/network choice is intentional; Google STT is off unless
      announced.
- [ ] Consent and deletion process are understood.
- [ ] A backup plan exists if the camera, network, or audio fails.

### During the demo

- Keep the OpenCV window focused so `q` works.
- Avoid moving the camera rapidly; recognition is sampled, not continuous.
- If a false recognition occurs, raise `SIMILARITY_THRESHOLD`/`MATCH_MARGIN`
  and restart, or use `--no-voice` and answer via `--ask` in a second terminal.
- If audio feedback loops, confirm `ECHO_GUARD=true` and lower speaker volume.
- Do not claim liveness or security properties.

### After the demo

If consent covered only the event, preview and remove generated data:

```powershell
.\cleanup_runtime.ps1
.\cleanup_runtime.ps1 -Force
.\cleanup_runtime.ps1 -Force -DeleteBiometrics
```

Enrollment images are retained unless `-DeleteBiometrics` is explicitly used.
Record the deletion scope, date, and operator.

## 7. Calibrate recognition

```text
python -B -m tools.calibrate_threshold
python -B -m tools.calibrate_threshold --json
python -B -m tools.calibrate_threshold --rebuild
```

The tool:

- extracts/caches embeddings for the current enrollment set
- computes genuine pairs (same person) and impostor pairs (different people)
- reports distributions, an equal-error threshold, FAR/FRR at that point, a
  threshold sweep, and a suggested `SIMILARITY_THRESHOLD` range
- refuses to give a stable estimate when you have fewer than two people, fewer
  than two images for at least one person, or too few pairs

Interpretation:

- `genuine_min > impostor_max` → separable on this set; midpoint is a useful
  starting point, but validate on held-out images.
- overlapping distributions → add varied images and re-check; no threshold
  removes the risk.
- the tool evaluates the enrollment images themselves (resubstitution), so its
  numbers are optimistic. For a report-quality result, enroll from one set and
  evaluate on a separate held-out set.
- small datasets produce unreliable estimates. The current checkout has four
  people with one image each, which is enough for a smoke test but not a
  meaningful FAR/FRR calibration.

After changing the threshold, test:

1. each enrolled person in normal lighting
2. each enrolled person with glasses/no glasses
3. at least one unenrolled person (must be `Unknown`)
4. two similar-looking people (must not be confused)

## 8. Performance tuning

| Symptom | Options |
| --- | --- |
| Low FPS / high CPU | Lower `CAMERA_WIDTH`/`CAMERA_HEIGHT`, keep `MAX_FRAME_WIDTH=640`, increase `RECOGNITION_INTERVAL` |
| Slow first run | Pre-download models with `--warmup-models`; keep `USE_EMBEDDING_CACHE=true` |
| Camera window error `The function is not implemented` | A headless OpenCV build is active. Re-run `setup_windows.ps1`, or run `.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84` and restart |
| High memory | Use a smaller DeepFace model, close other apps, keep frame width low |
| Disk use | Inspect `.runtime` and `.venv`, then remove them with `cleanup_runtime.ps1` when finished |
| Delayed speech | Lower `SPEECH_QUEUE_SIZE`, shorten answers, check audio device |
| Stale identity | Lower `RECOGNITION_RESULT_MAX_AGE`, confirm camera is delivering frames |
| Camera drops frames | Increase `CAMERA_BUFFER_SIZE` slightly or lower resolution |

DeepFace model/detector changes invalidate the cache. Rebuild after changes.

## 9. Update dependencies

The pins are intentional. Do not blindly upgrade TensorFlow, Keras, NumPy, or
DeepFace; they interact.

1. Create a branch.
2. Update one group at a time:
   - TensorFlow + tf-keras together
   - NumPy + OpenCV together
   - DeepFace separately
3. Recreate the venv with `setup_windows.ps1` (or the documented Linux steps).
4. Run lint, unit tests, audit, and the manual hardware checklist in
   [testing.md](testing.md).
5. Update the changelog and pins only after all checks pass.
6. Keep the previous known-good pins available for rollback.

Security updates to `requests`, `Pillow`, or other transitive dependencies
should be reviewed promptly even if the top-level pins do not change.

## 10. Backup and restore

Back up:

- `data/college_knowledge.json`
- `.env.example` (never a real `.env` with secrets in an unsecured backup)
- source code and docs
- consent/retention records

Do **not** casually back up biometric images. If backups are required, encrypt
them and include them in the retention/deletion plan.

`data/training_images/`, `.cache/`, and `.runtime/` are intentionally
git-ignored. A fresh checkout requires re-adding consented images and
rebuilding embeddings. Do not back up `.runtime` (models and caches) unless
offline installation is required; it can be rebuilt.

## 11. Troubleshooting

### Camera

- `CAMERA_INDEX=1` or `2` for external cameras.
- Close Teams/Zoom/browser tabs using the camera.
- Windows: Settings → Privacy → Camera.
- Linux: check `ls /dev/video*` and group permissions.

### DeepFace / TensorFlow

- Confirm the venv is active: `python -c "import tensorflow, tf_keras, deepface; print('ok')"`.
- Python must be 3.11/3.12.
- Keep `tensorflow==2.17.0` and `tf-keras==2.17.0` together.
- Delete `.cache/embeddings.json` and rebuild after model/detector changes.
- Weights live under `.runtime/deepface/.deepface/weights`; the nested
  `.deepface` directory is DeepFace's own layout.
- `--doctor` reports the configured model as `WARN` until its weights are
  cached. Run `--warmup-models` before an offline demo.

### Audio input

- Voice input is off by default; set `STT_PROVIDER`.
- Install `PyAudio` and PortAudio headers.
- List devices and set `STT_DEVICE_INDEX`.
- Whisper: first run downloads the model into `.runtime/huggingface`; use
  `--warmup-models` and check disk/network.
- Google: requires network and sends audio off-machine.

### Audio output

- Windows uses SAPI via `pywin32`; Linux uses `pyttsx3`/espeak-ng.
- List available voices and set `SPEECH_VOICE` exactly.
- `SPEECH_OUTPUT` is Windows-only; other platforms warn and use the default.
- Check `SPEECH_VOLUME`, system mixer, and `STARTUP_SPEECH`.

### Provider

- `ollama serve` and `ollama list`.
- The model name in `OLLAMA_MODEL` must exist; `:latest` is accepted.
- `GROQ_API_KEY` and `GROQ_MODEL` required for `AI_PROVIDER=groq`.
- Remote endpoints need HTTPS and valid keys.
- A rate-limit message means `CLOUD_MIN_INTERVAL` or
  `CLOUD_MAX_CALLS_PER_HOUR` was reached; wait or adjust deliberately.
- Provider errors return a safe offline answer; check the terminal for the
  category, not the full question.

### Knowledge

- `python -B -m tools.audit_knowledge` reports schema/uniqueness/probe errors.
- `python app.py --doctor` reports duplicate phrases and missing provenance.
- A fact without `"verified": true` is ignored by design.
- Check aliases if a natural paraphrase falls through to the model.

## 12. Incident response

If a key is exposed:

1. Revoke/rotate it at the provider immediately.
2. Remove it from the working tree and shell history.
3. Check logs/screenshots/repositories for copies.
4. Record the incident and notify affected parties if personal data was
   involved.

If biometric data is exposed:

1. Stop the demo and isolate the machine.
2. Preserve non-personal logs for investigation.
3. Notify the data owner and follow the consent/retention agreement.
4. Delete copies and rotate any credentials on the host.
5. Do not post images or transcripts in a public issue.

## 13. Release checklist

- [ ] Version bumped in `version.py`, `pyproject.toml`, changelog.
- [ ] `python -m ruff check .` passes.
- [ ] `python -B -m unittest discover -s tests -v` passes.
- [ ] `python -B -m tools.audit_knowledge` passes.
- [ ] `python app.py --doctor` has no unexpected `FAIL`.
- [ ] `python app.py --warmup-models` succeeds for the configured models.
- [ ] `.\cleanup_runtime.ps1` (dry run) lists only generated files.
- [ ] Manual hardware checklist completed.
- [ ] No `.env`, images, `.cache/`, `.runtime/`, logs, or keys in `git status`.
- [ ] README/docs updated.
- [ ] Knowledge provenance dates current.
