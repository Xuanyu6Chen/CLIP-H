# Turns each row of a features CSV into a profile text using the PROFILE template of the task's run script, and writes the profile files (profiles alone, and profiles with their labels).
"""Template profiles. No LLM is involved.

A template is an ordered set of blocks. A block is a few sentences with
{column} slots. A sentence is written only when every slot has a value, so a
missing lab simply disappears instead of leaving empty text.

Block keys of PROFILE (set at the top of each run script):
    section     heading the block sits under; printed once, when the first
                block of that section produces text
    lead, join  text before the block, and between its sentences
    parts       the sentences
    show_if     column that must be non-zero for `parts` to show; when it is 0
                the block shows `empty_text` instead (e.g. "No ICU admission.")
    empty_text  shown when no sentence could be written
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

SLOT = re.compile(r"\{(\w+)\}")
NO_VALUE = {"unknown", "nan", "none", "?", ""}


def _value(row: dict[str, str], column: str) -> str | None:
    """The cell as text, or None when the column is absent or holds no real value."""
    value = row.get(column)
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in NO_VALUE else text


def _format(text: str) -> str:
    """Whole numbers lose ".0" ("3.0" -> "3"), other numbers keep 2 decimals,
    text is kept. A trailing period is dropped because the sentences add their
    own ("EW EMER." would otherwise print as "EW EMER..")."""
    text = text.strip().rstrip(".")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(number)) if number == int(number) else f"{round(number, 2):g}"


def fill_sentence(sentence: str, row: dict[str, str]) -> str | None:
    """Fill the sentence's {column} slots, or return None if any value is missing."""
    if any(_value(row, column) is None for column in SLOT.findall(sentence)):
        return None
    return SLOT.sub(lambda match: _format(_value(row, match.group(1))), sentence)


def render_block(row: dict[str, str], block: dict[str, Any]) -> str | None:
    """Write one block for one patient, or None when it has nothing to say."""
    show_if = block.get("show_if")
    if show_if is not None:
        value = _value(row, show_if)
        if value is None:                 # column absent or empty: skip the block
            return None
        try:
            shown = float(value) != 0
        except ValueError:
            shown = True
        if not shown:                     # present but 0: the "none" line
            return block.get("empty_text")
    parts = [filled for filled in (fill_sentence(s, row) for s in block.get("parts", [])) if filled]
    if not parts:
        return block.get("empty_text")
    return block.get("lead", "") + block.get("join", " ").join(parts)


def render_profile(row: dict[str, str], template: dict[str, dict[str, Any]]) -> str:
    """Write the whole profile for one patient: every block that has text,
    under its [SECTION] heading. Blocks sharing a section must be adjacent in
    the template, since the heading is printed when the section changes."""
    lines: list[str] = []
    current_section = None
    for block in template.values():
        text = render_block(row, block)
        if not text:
            continue
        section = block.get("section")
        if section and section != current_section:
            lines.append(f"[{section}]")
            current_section = section
        lines.append(text)
    profile = "\n".join(lines) if lines else "Patient record."
    tidy = [re.sub(r"[ \t]+", " ", line).strip() for line in profile.split("\n")]
    return "\n".join(line for line in tidy if line).strip()


def write_profiles(features_csv: Path, template: dict[str, dict[str, Any]],
                   out_jsonl: Path, overwrite: bool) -> list[str]:
    """Write one JSON line per row of the features CSV: {"idx", "profile"}.

    Profiles are built from the saved features file on purpose: whatever was
    kept out of the features cannot reach a profile. Returns the profile texts.
    Stops if the profile file is already there and `overwrite` is false.
    """
    if Path(out_jsonl).exists() and not overwrite:
        raise FileExistsError(f"{out_jsonl} already exists. Set OVERWRITE = True in the run script to replace it.")
    with Path(features_csv).open("r", encoding="utf-8") as file:
        profiles = [render_profile(row, template) for row in csv.DictReader(file)]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with Path(out_jsonl).open("w", encoding="utf-8") as file:
        for idx, profile in enumerate(profiles):
            file.write(json.dumps({"idx": idx, "profile": profile}) + "\n")
    return profiles


def write_profiles_with_labels(profiles: list[str], labels: Iterable[int],
                               out_jsonl: Path, overwrite: bool) -> None:
    """Write one JSON line per patient: {"id", "profile", "label"}.

    The profile and the label of each patient together, in the row order of
    the features and labels files (`id` = row number). This is the file the
    method folders read. Stops if it is already there and `overwrite` is false.
    """
    if Path(out_jsonl).exists() and not overwrite:
        raise FileExistsError(f"{out_jsonl} already exists. Set OVERWRITE = True in the run script to replace it.")
    labels = [int(label) for label in labels]
    if len(profiles) != len(labels):
        raise ValueError(f"{len(profiles):,} profiles but {len(labels):,} labels")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with Path(out_jsonl).open("w", encoding="utf-8") as file:
        for idx, (profile, label) in enumerate(zip(profiles, labels)):
            file.write(json.dumps({"id": idx, "profile": profile, "label": label}) + "\n")
