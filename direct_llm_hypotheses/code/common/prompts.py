"""Read a prompt from the ``prompts`` folder (one file per step)."""

import re
from pathlib import Path
from typing import Optional

_SECTION_MARKER = re.compile(r"^=== (\w+) ===\n", re.MULTILINE)


def load_prompt(prompt_name: str, section: Optional[str] = None) -> str:
    """Read a prompt from ``prompts/<prompt_name>.txt``.

    A file that holds several prompts marks each with a line ``=== name ===``; pass that name
    as ``section``. The text of a section runs to the next marker, without the blank line
    that separates the two (and without the final line break of the file).
    """
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / f"{prompt_name}.txt"
    try:
        with open(prompt_path, encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {prompt_path}; please ensure it's in the prompts/ folder")
    parts = _SECTION_MARKER.split(text)
    if len(parts) == 1:
        if section is not None:
            raise KeyError(f"{prompt_path.name} has no sections; asked for '{section}'")
        return text
    names, bodies = parts[1::2], parts[2::2]
    if section not in names:
        raise KeyError(f"{prompt_path.name} has sections {names}; asked for '{section}'")
    k = names.index(section)
    end = "\n" if k == len(names) - 1 else "\n\n"
    if not bodies[k].endswith(end):
        raise ValueError(f"{prompt_path.name}: section '{section}' must be followed by "
                         + ("one line break" if k == len(names) - 1 else "one blank line"))
    return bodies[k][:-len(end)]
