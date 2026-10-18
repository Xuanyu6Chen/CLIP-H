# Prompts

One file per step. `prompts.load_prompt("<file>", "<section>")` reads a prompt.

A file that holds several prompts marks each with a line `=== name ===`. The text of a section
runs to the next marker, without the blank line that separates the two. Text in `{braces}` is
filled in by the code; a literal brace is written doubled (`{{`, `}}`). The text is sent exactly
as written.

| File | Step | Sections | Read by | Filled in |
|---|---|---|---|---|
| `generation.txt` | G1 generation | `system`: the role | `direct_generation.generate_pool` | |
| | | `user`: labeled profiles of one batch, propose `k` hypotheses | `direct_generation.generation_prompt` | `n`, `outcome`, `k`, `extra_rules`, `profiles_block` |
| `annotation_<task>.txt` | 3A annotation | the sections of CLIP-H's `annotation.txt` | `annotation.py`, after `use_prompt_file('annotation_<task>')` in Setup | see `clip_h/readmission/prompts/README.md` |

Each `annotation_<task>.txt` holds the same text as the `annotation.txt` of that task's CLIP-H
folder (`readmission`, `hosp_los3` = `clip_h/hosp_los`, `icu_los4` = `clip_h/icu_los`), so both
methods label patients with the same prompt. Only the `worked_example` and `pair` examples differ
between tasks; `mortality` and `discharge` share theirs.

The task wording of the generation prompt (`outcome`, `extra_rules`) is set per task in the
notebooks' Settings cell (`_TASKS`).
