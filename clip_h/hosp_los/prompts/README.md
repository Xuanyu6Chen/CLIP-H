# Prompts

One file per step. `prompts.load_prompt("<file>", "<section>")` reads a prompt.

A file that holds several prompts marks each with a line `=== name ===`. The text of a section
runs to the next marker, without the blank line that separates the two. Text in `{braces}` is
filled in by the code; a literal brace is written doubled (`{{`, `}}`). The text is sent exactly
as written.

| File | Step | Sections | Read by | Filled in |
|---|---|---|---|---|
| `interpret-neuron-binary.txt` | 2A interpretation | none (one prompt) | `interpret_neurons.interpret_neuron` | `task_specific_instructions`, `positive_texts`, `negative_texts` |
| `weak_groups.txt` | 2A+ weak-neuron groups | `system`: the role for the two calls below | | |
| | | `name_user`: a short name for one weak neuron | `weak_groups.name_weak_neuron` | `positive_texts`, `negative_texts` |
| | | `group_user`: sort the names into related groups, each stated as one hypothesis | `weak_groups.group_weak_neurons` | `conditions` |
| `annotation.txt` | Annotation, used by 2A, 2B, 2C, 2D, 2E, 3A | `pair_system`, `pair`, `pair_suffix`: one profile, one hypothesis | `annotation.score_pair` | `hypothesis`, `text` |
| | | `rubric_header`, `checklist_rules`, `checklist`: one profile, all hypotheses, reply style `checklist` | `annotation._checklist_prompt` | `properties`, `profile`, `last_index` |
| | | `rubric_header`, `worked_example`, `labels`: the same, reply style `labels` | `annotation._batch_prompt` | `properties`, `patient`, `n_properties`, `label_template` |
| `self_refine.txt` | 2B self-refine | `refine_system`, `refine`, `critic_system`, `critic` | `self_refine.make_refine_fn_agentic`, `make_critic_fn` | `hypotheses`, `feedback`; `task`, `pos_label`, `candidates` |
| `review.txt` | 2C review | none (one prompt) | the runner notebook, which passes it to `conversational_review.build_review_prompt` | `{n_reviewers}` and, where the file has it, `{hospital_url}`, by text replacement, so braces in this file are not doubled |
| `dedup.txt` | 2D de-duplication | `system`, `user` | `dedup_lib.make_dedup_adjudicator` | `task`, `pos_label`, `phi`, `concept_a`, `n_a`, `concept_b`, `n_b`, `n_both` |

The name `interpret-neuron-binary` is part of the key of the interpretation cache; renaming the
file makes saved interpretations miss.

The prompts that carry the task are `interpret-neuron-binary.txt`, `review.txt`, `weak_groups.txt`
and, in `annotation.txt`, the examples of `pair` and the `worked_example`. `self_refine.txt`,
`dedup.txt` and the other sections of `annotation.txt` are the same for every task; the task is
told to them through `{task}` and `{pos_label}`.
