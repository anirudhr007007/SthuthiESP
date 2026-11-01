# Enrollment images and consent

Each subfolder represents one person:

```text
data/training_images/
  Person Name/
    front.jpg
    left.jpg
    glasses.jpg
```

## Capture guidance

- Use 5-10 clear images per person.
- Vary lighting, angle, expression, distance, and glasses/no-glasses.
- Each image must contain one clearly dominant face. Tiny background
  detections may be ignored, but images with two plausibly sized faces are
  rejected as possible group photos.
- Avoid group photos, screenshots, heavy filters, and images where the face is
  smaller than roughly 10% of the frame.
- Rebuild embeddings after changing the face model or detector:

  ```text
  python app.py --rebuild-embeddings
  ```

## Consent and privacy

These are biometric images. Collect them only from people who have given
informed consent for this demonstration, and tell them what is stored, why,
where it is processed, and how to request deletion.

- Keep images local. Do not commit them to a public repository, paste them in
  chat, or upload them to a cloud provider.
- The repository `.gitignore` excludes files under this folder; only this
  README is tracked.
- Face embeddings are stored in `.cache/embeddings.json` on the local machine.
  They are derived from the images but are still sensitive.
- Delete a person's folder and rebuild the cache to remove them. Also delete
  old backups and any copies made outside this folder.
- Model files and temporary caches live under `.runtime`; remove them with
  `.\cleanup_runtime.ps1 -Force` when the demo runtime is no longer needed.

## Retention

Agree a retention period before collecting images. For a short college demo,
delete the images and the cache after the event unless participants have
consented to longer storage. Record who approved the collection and when
deletion happened; do not rely on memory.
