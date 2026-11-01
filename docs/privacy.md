# Privacy and security

This project processes biometric data and can optionally send audio or text
off the machine. Treat privacy as an operational requirement, not a feature
flag.

## Data inventory

| Data | Where it lives | Sensitivity | Retention |
| --- | --- | --- | --- |
| Enrollment images | `data/training_images/<person>/` | High (biometric) | Delete after the demo unless consent covers longer |
| Face embeddings | `.cache/embeddings.json` | High (derived biometric) | Delete with the images; rebuild to regenerate |
| Camera frames | RAM only | High | Never written to disk by this project |
| Recognised names and scores | RAM / OpenCV window | Medium | Not persisted |
| Spoken answers | RAM and speakers | Low/medium | Not recorded by this project |
| Microphone audio | RAM | High | Sent to Google only if `STT_PROVIDER=google` |
| General questions and retrieved FAQ context | Sent to the selected provider | Medium | Provider policy applies |
| DeepFace and Whisper model files | `.runtime/deepface`, `.runtime/huggingface` | Low/medium (third-party model data) | Delete with `cleanup_runtime.ps1` |
| Ollama model files (optional) | Ollama's store (project-local store only if `OLLAMA_MODELS` points into `.runtime`) | Low | Managed by Ollama; `cleanup_runtime.ps1` removes only a project-local store |
| Pip downloads, temporary files, tool caches | `.runtime/pip-cache`, `.runtime/tmp`, `.runtime/*` | Low | Delete with `cleanup_runtime.ps1` |
| `.env` | Project root | Secret | Keep local; git-ignored |
| Logs | Terminal only; optional `.runtime/logs` | Low | Avoid running with verbose capture in public |

The project does not create a face database beyond the local image folders and
the embedding cache. It does not upload camera frames.

## Data flows

### Default configuration

```text
camera -> local DeepFace -> local embedding comparison -> greeting
question -> local verified FAQ -> local answer
general question -> local Ollama (loopback) -> answer
speech -> local SAPI / pyttsx3
microphone -> off
```

Nothing leaves the machine except optional local-network traffic to Ollama on
loopback.

### Google STT (opt-in)

```text
microphone -> Google Speech Recognition -> transcript -> local matcher/provider
```

Captured microphone audio leaves the machine. Google's terms and retention
policy apply. `--offline` disables this and recommends Whisper.

### Local Whisper (opt-in)

```text
microphone -> faster-whisper on this machine -> transcript
```

Audio stays local. The model files are downloaded on first use into
`.runtime/huggingface` (or after `--warmup-models`).

### Local runtime

```text
pip downloads -> .runtime/pip-cache
temporary files -> .runtime/tmp
DeepFace weights -> .runtime/deepface
Whisper/Keras/tool caches -> .runtime/*
```

The venv and every cache can be removed with `cleanup_runtime.ps1`. No global
Python package is installed or modified by `setup_windows.ps1`.

### Cloud answer providers (opt-in)

```text
general question + bounded verified FAQ context -> provider -> answer
```

Only general questions that are not classified as college-specific are sent.
Verified FAQ answers and unknown college-specific questions never trigger a
provider call. The prompt includes a small, relevant subset of verified facts
so the model does not invent college details. Cloud calls are rate-limited and
errors are logged without the question or key.

## Safeguards in the code

- Default `STT_PROVIDER=none`; Google STT requires explicit configuration.
- `--offline` forces `AI_PROVIDER=offline`, disables Google STT, and keeps
  local Whisper available.
- Verified facts are matched before provider selection.
- Unknown college-specific questions stay offline.
- Provider output is bounded to 1,600 characters; requests are bounded to 180
  output tokens, 64 KiB responses, a timeout, and no redirects.
- Remote endpoints must use HTTPS; plain HTTP is allowed only for loopback.
- URLs containing credentials, query strings, or fragments are rejected.
- Cloud credentials are sent in headers, never in URLs.
- Provider probe output and startup summaries omit secret values.
- Voice operation prints status only; it does not print visitor questions,
  answers, or rejected wake-word transcripts.
- `.gitignore` excludes `.env`, training images, `.cache/`, `.runtime/`,
  recordings, and models.
- `setup_windows.ps1` installs into `.venv` and redirects pip, temporary, and
  model caches to `.runtime`; `cleanup_runtime.ps1` previews or removes them.
- The embedding cache is written atomically, invalidated when the model or
  detector changes, and checked for image coverage by `--doctor`.
- The speech queue is bounded so a stuck audio device cannot grow memory
  without limit.

## What the code does **not** protect against

- **No liveness or anti-spoofing.** A printed photo or replay may be accepted.
  Do not use this for access control, attendance, payments, or identity proof.
- **No encryption at rest.** Enrollment images and the embedding cache rely on
  operating-system permissions. Use full-disk encryption for a shared laptop.
- **No per-visitor consent prompt.** Consent must be collected by the operator
  before images are placed in the training folder.
- **No central audit log.** Deletions and approvals must be recorded by the
  operator.
- **No guarantee against a compromised host.** Malware, screen capture, or
  another local user with file access can read the images and cache.
- **No absolute network guarantee.** Endpoint validation is URL policy; DNS,
  proxies, and a compromised local resolver are outside its control.
  `--offline` disables cloud answers and Google STT, but a missing DeepFace or
  Whisper model can still be downloaded on first use. Pre-fetch models with
  `--warmup-models` and verify `--doctor` before an offline demo.
- **Heuristic college-question router.** An unusual paraphrase could be
  classified as general and sent to a cloud provider. Use `--offline` or a
  local provider when that risk is unacceptable.

## Consent checklist

Before collecting enrollment images:

- [ ] Explain what is stored, where, why, and for how long.
- [ ] Explain that face embeddings are derived and stored locally.
- [ ] Explain which optional providers can receive audio/text.
- [ ] Obtain explicit consent from each person (or their guardian).
- [ ] Record the consent and the agreed retention date.
- [ ] Provide a deletion contact and process.
- [ ] Do not collect images of bystanders; use photos with one dominant face.
- [ ] Reconfirm consent before reusing images for a new purpose.

## Deletion procedure

For the project-local runtime and generated caches:

```powershell
.\\cleanup_runtime.ps1            # preview
.\\cleanup_runtime.ps1 -Force     # remove .venv, .runtime, .cache, and tool caches
```

Enrollment images are retained by default. Use `-DeleteBiometrics` only after
confirming consent and retention, and `-DeleteEnv` to remove local secrets:

```powershell
.\\cleanup_runtime.ps1 -Force -DeleteBiometrics -DeleteEnv
```

For one person, or when cleanup is not available:

1. Delete the person's folder under `data/training_images/`.
2. Delete `.cache/embeddings.json` (or rebuild it) so the removed embeddings
   are not reused.
3. Delete backups, exported reports, and any copied images.
4. Record the deletion date and who performed it.
5. Verify the person no longer matches in a test run.

## Secret handling

- Copy `.env.example` to `.env`; never commit `.env`.
- Use environment variables in shared or CI environments.
- Never put keys in `app.txt`, `api.txt`, source files, screenshots, shell
  history, or the knowledge JSON.
- If a key was ever exposed, **revoke it**; removing the file is not enough.
- The original archive contained a credential in `app.txt`. Revoke that key if
  the archive was shared, even though the file is gone from this checkout.

## Reporting a privacy issue

Stop using the affected configuration, preserve relevant logs without personal
data, and contact the project maintainer privately. Do not open a public issue
containing images, transcripts, keys, or visitor names.
