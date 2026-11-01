# Knowledge base curation

`data/college_knowledge.json` contains the only facts the assistant may answer
as verified college knowledge. The file is data, not a prompt: malformed or
unverified entries are ignored at load time.

## Schema (version 2)

```json
{
  "schema_version": 2,
  "college": {
    "name": "Malnad College of Engineering",
    "short_name": "MCE",
    "location": "Hassan, Karnataka, India",
    "website": "https://www.mcehassan.ac.in/"
  },
  "provenance": {
    "reviewed_on": "2026-10-25",
    "reviewed_by": "SthuthiBeam maintainers",
    "method": "How the facts were checked",
    "limitations": "What the review does not prove",
    "official_sources": ["https://www.mcehassan.ac.in/"]
  },
  "facts": [
    {
      "verified": true,
      "question": "Who is the current principal of MCE?",
      "answer": "Dr. ... is the principal of Malnad College of Engineering.",
      "source": "User-provided MCE FAQ attachment (cross-checked 2026-10-25)",
      "source_type": "user_attachment",
      "verified_on": "2026-10-25",
      "reviewed_by": "SthuthiBeam maintainers",
      "aliases": ["Who is the principal?", "Who is MCE's principal?"],
      "keywords": ["principal", "MCE"]
    }
  ],
  "college_fact_templates": [
    {
      "verified": false,
      "question": "What are the current admission fees and deadlines?",
      "answer": "",
      "source": "",
      "note": "Add a current official source before setting verified to true."
    }
  ]
}
```

### Required per-fact fields

| Field | Rule |
| --- | --- |
| `verified` | Must be the JSON boolean `true` to load |
| `question` | Non-empty, unique after normalisation |
| `answer` | Non-empty, non-placeholder, preferably 1-3 spoken sentences |
| `source` | Non-empty and specific enough to re-check |
| `source_type` | `user_attachment`, `assistant_help`, `answer_policy`, `official_website`, or `project_document` |
| `verified_on` | ISO date, not in the future |
| `reviewed_by` | Person or team that checked it |
| `aliases` | Natural paraphrases; at least two recommended |
| `keywords` | Short phrases useful for retrieval |

Questions and aliases must be unique across the whole file. Duplicate phrases
make routing ambiguous and are reported by both the audit and `--doctor`.

## What "verified" means here

`verified` means **source-checked against the supplied MCE FAQ attachment and,
where noted, public MCE pages on the review date**. It does **not** mean the
college independently confirmed the fact, and it does not make time-sensitive
data permanent.

Re-check officials, phone numbers, fees, intake, deadlines, accreditation, and
facility details with the college office before a public demo. Update
`verified_on` and `source` when you do.

## Adding a fact

1. Find the smallest question a visitor would actually ask.
2. Write a direct answer suitable for speech (avoid tables and markdown).
3. Record the exact source and source type.
4. Add two or more natural aliases, including common abbreviations.
5. Add a few retrieval keywords.
6. Set `verified_on` to today and `reviewed_by` to the reviewer.
7. Run the audit and tests:

   ```text
   python -B -m tools.audit_knowledge
   python -B -m unittest discover -s tests -v
   ```

8. Run `python app.py --ask "<new question>"` and at least three paraphrases.

### Alias guidance

Good aliases are complete questions or short noun phrases a visitor would say:

- `Who is the HOD of CSE?`
- `Does the college have a hostel?`
- `What is MCE's website?`
- `How many students study at MCE?`

Avoid:

- single generic words (`principal`, `library`) that could collide
- aliases that contain another fact's exact question
- time-sensitive claims that belong in the answer, not the alias
- duplicates that differ only by punctuation or case

The matcher normalises NFKC, case, and punctuation, so `MCE's website?` and
`mce website` are the same key.

### HOD and name conventions

Use the full canonical name and department:

```text
Dr. Ananda Babu J
Dr. Naveen Kumar C M
```

Do not reintroduce legacy variants such as `Dr. Anand Babu J` or
`Dr. C. M. Naveen Kumar`. The regression suite checks these names.

## Adding a prompt template (not a fact)

If you know a question but not a reliable answer, add it to
`college_fact_templates` with `"verified": false`, an empty answer/source, and
a note explaining what must be checked. Templates are never loaded as facts.

## Auditing

```text
python -B -m tools.audit_knowledge
python -B -m tools.audit_knowledge --knowledge path/to/file.json
```

The audit checks:

- top-level schema version and required provenance
- ISO review dates, not in the future
- explicit `verified: true` and non-placeholder fields
- allowed source types
- duplicate questions and aliases after normalisation
- alias length, empty strings, and self-referential aliases
- loader acceptance (every fact must survive `filter_verified_knowledge`)
- 15 paraphrase probes, including HOD acronyms, library/labs, PhD, NBA, and
  campus-location phrasing

`python app.py --doctor` adds a runtime duplicate check. The unit suite
`tests/test_knowledge_recall.py` additionally verifies that **every** curated
question and alias resolves to its own fact.

## Editing safely

- Keep valid UTF-8 and two-space JSON indentation.
- Preserve `schema_version: 2`.
- Prefer a scripted edit for bulk changes and review the diff.
- Do not hand-merge duplicate facts without preserving the best answer,
  aliases, source, and review metadata.
- Never remove provenance to make an audit pass.
- If a fact is wrong, fix the answer **and** the source; do not just change the
  wording.
- If a fact is uncertain, move it to `college_fact_templates` instead of
  guessing.

## Consolidation history

The 2.0.0 curation merged ten duplicate entries (CSE/ISE HOD, CSBS HOD,
principal phone, student-affairs contact, student counts, AI&ML research areas,
and campus location), normalized names, and added aliases. The current file
has 121 facts and 507 unique aliases. The knowledge regression test protects
that recall while allowing future additions.

## Privacy

The knowledge file contains public college information, not personal data.
Do not add visitor names, phone numbers of private individuals, enrollment
images, or credentials. Keep official sources as URLs and dates, not copied
private correspondence.
