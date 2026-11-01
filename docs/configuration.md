# Configuration reference

All configuration is environment based. `.env` in the project root is loaded
automatically when present; **real environment variables always win**. Use
`.env.example` as the starting point and never commit the resulting `.env`.

Relative paths are resolved from the project root. Values marked *clamped* are
forced into a safe range; invalid numbers fall back to the default and print a
warning.

## College and data

| Variable | Default | Notes |
| --- | --- | --- |
| `COLLEGE_NAME` | `Malnad College of Engineering, Hassan` | Used in greetings, prompts, and college-name detection |
| `TRAINING_PATH` | `data/training_images` | One subfolder per person |
| `KNOWLEDGE_PATH` | `data/college_knowledge.json` | Verified FAQ schema 2 |
| `EMBEDDING_CACHE_PATH` | `.cache/embeddings.json` | Local cache; contains derived biometric data |
| `USE_EMBEDDING_CACHE` | `true` | Set `false` to re-extract every run |
| `ESSPTHUTHI_RUNTIME_DIR` | `.runtime` | Root for pip/temp/model caches; relative paths resolve from the project. Must resolve inside the checkout, or `.runtime` is used. `setup_windows.ps1` always installs under `.runtime` |

## Face recognition

| Variable | Default | Notes |
| --- | --- | --- |
| `FACE_MODEL` | `VGG-Face` | Any DeepFace-compatible model; changing it invalidates the cache |
| `FACE_DETECTOR` | `opencv` | DeepFace detector backend; changing it invalidates the cache |
| `SIMILARITY_THRESHOLD` | `0.68` | Clamped `0.0-1.0`; raise to reduce false accepts |
| `MATCH_MARGIN` | `0.08` | Clamped `0.0-1.0`; required gap over the runner-up |
| `RECOGNITION_INTERVAL` | `0.35` | Minimum seconds between submitted frames; minimum `0.1` |
| `RECOGNITION_RESULT_MAX_AGE` | `1.5` | Results older than this are discarded; minimum `0.2` |
| `MAX_FRAME_WIDTH` | `640` | Downscale for inference/display; `0` disables |
| `IDENTITY_CONFIRMATIONS` | `3` | Consecutive batches before a name is confirmed; minimum `1` |

## Camera

| Variable | Default | Notes |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | Try `1`, `2`, ... for external cameras |
| `CAMERA_WIDTH` | `1280` | Requested capture width; `0` leaves it to the driver |
| `CAMERA_HEIGHT` | `720` | Requested capture height |
| `CAMERA_FPS` | `30` | Requested frame rate |
| `CAMERA_BUFFER_SIZE` | `1` | Small buffers reduce latency |
| `CAMERA_READ_RETRIES` | `5` | Retries while opening the camera; minimum `1` |
| `CAMERA_MAX_READ_FAILURES` | `10` | Consecutive failed reads before stopping; minimum `1` |

## Voice input

| Variable | Default | Notes |
| --- | --- | --- |
| `ENABLE_VOICE` | `true` | Master switch for microphone **and** speech output |
| `STT_PROVIDER` | `none` | `none`, `whisper`, or `google`; aliases such as `faster-whisper` map to `whisper` |
| `STT_LANGUAGE` | `en-IN` | BCP-47 style language hint |
| `STT_DEVICE_INDEX` | unset | Optional PyAudio input device index |
| `STT_ENERGY_THRESHOLD` | unset | Optional recognizer energy threshold |
| `STT_PAUSE_THRESHOLD` | `0.8` | Silence before a phrase ends; minimum `0.2` |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, ... |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or another faster-whisper device |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `int8_float16`, `float16`, `float32` |
| `LISTEN_TIMEOUT` | `1.0` | Seconds to wait for speech; minimum `0.5` |
| `PHRASE_TIME_LIMIT` | `8` | Maximum seconds per phrase; minimum `1` |
| `WAKE_WORD` | empty | Leading phrase required before answering; empty disables |
| `ECHO_GUARD` | `true` | Skip transcription while the assistant speaks and for matching recent answers |
| `ECHO_SIMILARITY_THRESHOLD` | `0.72` | Clamped `0.3-0.99` |
| `ECHO_IDLE_GRACE` | `0.6` | Extra seconds after speech before listening; minimum `0` |
| `ECHO_RECENT_WINDOW` | `20.0` | Seconds of answers retained for echo checks; minimum `2` |
| `VOICE_DEBUG` | `false` | Print heard questions/answers in the terminal; keep off around visitors |

`STT_PROVIDER=google` sends microphone audio to Google. `--offline` forces it
back to `none`; use `whisper` for local voice input. Whisper models are cached
under `.runtime/huggingface` (pre-fetch with `--warmup-models`).

## Speech output

| Variable | Default | Notes |
| --- | --- | --- |
| `SPEECH_OUTPUT` | empty | Windows SAPI audio output name; other platforms use the default |
| `SPEECH_VOICE` | empty | Voice name; availability depends on the OS |
| `SPEECH_RATE` | `165` | Words per minute, clamped `80-320`; mapped to SAPI `-10..10` |
| `SPEECH_VOLUME` | `1.0` | Clamped `0.0-1.0` |
| `SPEECH_QUEUE_SIZE` | `25` | Maximum queued messages; minimum `1`. Overflow drops the oldest |
| `STARTUP_SPEECH` | `true` | Speak "The college assistant is ready." |

## Greeting and presence

| Variable | Default | Notes |
| --- | --- | --- |
| `GREETING_MIN_GAP` | `2.0` | Minimum seconds between greetings; minimum `0` |
| `PRESENCE_TIMEOUT` | `8.0` | Seconds without a face before a visit ends; minimum `1` |
| `VISITOR_WELCOME_COOLDOWN` | `20.0` | Seconds before another unknown-visitor welcome; minimum `0` |

## Answer providers

| Variable | Default | Notes |
| --- | --- | --- |
| `AI_PROVIDER` | `ollama` | `ollama`, `offline`, `gemini`, `openai`, `groq`, `nvidia`, or `compatible` |
| `AI_TIMEOUT` | `15.0` | Provider timeout in seconds; minimum `2` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Loopback HTTP is allowed |
| `OLLAMA_MODEL` | `llama3.2:3b` | Must be pulled locally |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Used only when `AI_PROVIDER=gemini` |
| `GEMINI_API_KEY` | empty | Never print or commit |
| `OPENAI_MODEL` | `gpt-4o-mini` | Used only when `AI_PROVIDER=openai` |
| `OPENAI_API_KEY` | empty | Never print or commit |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Used only when `AI_PROVIDER=groq` |
| `GROQ_API_KEY` | empty | Never print or commit |
| `AI_BASE_URL` | empty | Compatible/NVIDIA endpoint; empty means the hosted NVIDIA API for `AI_PROVIDER=nvidia`; remote URLs must be HTTPS |
| `AI_API_KEY` | empty | Bearer token for a compatible/NVIDIA endpoint (`nvapi-...` for NVIDIA) |
| `AI_MODEL` | empty | Required for compatible and NVIDIA endpoints (e.g. `nvidia/nemotron-3.5-lightning-30b-a3b`); stale NVIDIA ids return HTTP 410 |
| `WEB_SEARCH` | `true` | Send unmatched general questions to DuckDuckGo for context; `false` disables all web access for answers |
| `CLOUD_MIN_INTERVAL` | `3.0` | Minimum seconds between remote calls; minimum `0` |
| `CLOUD_MAX_CALLS_PER_HOUR` | `60` | Remote calls per rolling hour; `0` disables the cap |
| `PROVIDER_PROBE_TIMEOUT` | `3.0` | Ollama probe timeout; minimum `0.5` |

Cloud limits apply only to Gemini, OpenAI, Groq, NVIDIA, and non-loopback
compatible endpoints. Local Ollama and local LM Studio are not rate-limited.

## CLI overrides

The CLI applies overrides after `.env` and environment loading:

```text
--offline              AI_PROVIDER=offline; Google STT -> none; keeps Whisper
--no-voice             ENABLE_VOICE=false
--camera-index N       CAMERA_INDEX=N
--stt-provider NAME    normalised STT_PROVIDER
--wake-word PHRASE     WAKE_WORD=PHRASE (empty disables)
--ai-provider NAME     AI_PROVIDER=NAME
--chat                 text-only session; no camera, microphone, or speech
--warmup-models        load/cache the configured face and Whisper models, then exit
--rebuild-embeddings   ignore and rebuild the local embedding cache, then run
```

Overrides last for that process only; they never edit `.env`. `--offline` is
not a network sandbox: it stops cloud answers and Google STT, but a missing
DeepFace or Whisper model can still download on first use. Run
`--warmup-models` before an offline demo.

## Validation and fail-closed behaviour

- Invalid numbers print a warning and use the default.
- Out-of-range thresholds and margins print a warning and are clamped.
- Missing or malformed knowledge files produce no verified facts; the app
  warns instead of inventing answers.
- Unknown college-specific questions never reach a model.
- Provider failures return a bounded offline answer and do not auto-switch.
