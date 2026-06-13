"""On-disk project: everything needed to resume any stage lives in files.

Layout (under <root>/<name>/):

    project.json            # config: language, length, instructions, providers
    memory/
      concept.json          # canonical structured artifacts (the story bible)
      characters.json
      world.json
      outline.json
      continuity.json       # {chapter_number: summary} — resumable continuity
    chapters/
      ch01.md, ch02.md, ...
    novel.md                # assembled novel
    critique.md             # critic review

Nothing the next command needs is held only in memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .models import Characters, Concept, Outline, World

DEFAULT_ROOT = "projects"


@dataclass
class ProjectConfig:
    name: str
    language: str
    length: int
    instructions: str | None = None
    planner: str | None = None  # provider name in secret.yaml (None -> first)
    drafter: str | None = None
    created: str = ""


class Project:
    def __init__(self, name: str, root: str | Path = DEFAULT_ROOT):
        self.name = name
        self.dir = Path(root) / name
        self.memory = self.dir / "memory"
        self.chapters = self.dir / "chapters"

    # ----- lifecycle -------------------------------------------------------------

    def exists(self) -> bool:
        return self.config_path.exists()

    def create(self, config: ProjectConfig) -> None:
        self.memory.mkdir(parents=True, exist_ok=True)
        self.chapters.mkdir(parents=True, exist_ok=True)
        config.created = config.created or datetime.now().isoformat(timespec="seconds")
        self.save_config(config)

    # ----- config ----------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.dir / "project.json"

    def save_config(self, config: ProjectConfig) -> None:
        self.config_path.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_config(self) -> ProjectConfig:
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        return ProjectConfig(**data)

    # ----- structured artifacts (json canonical + md mirror) ---------------------

    def _save_json(self, name: str, model) -> None:
        (self.memory / name).write_text(
            model.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load_json(self, name: str, schema):
        path = self.memory / name
        if not path.exists():
            return None
        return schema.model_validate_json(path.read_text(encoding="utf-8"))

    def save_concept(self, c: Concept) -> None:
        self._save_json("concept.json", c)

    def load_concept(self) -> Concept | None:
        return self._load_json("concept.json", Concept)

    def save_characters(self, c: Characters) -> None:
        self._save_json("characters.json", c)

    def load_characters(self) -> Characters | None:
        return self._load_json("characters.json", Characters)

    def save_world(self, w: World) -> None:
        self._save_json("world.json", w)

    def load_world(self) -> World | None:
        return self._load_json("world.json", World)

    def save_outline(self, o: Outline) -> None:
        self._save_json("outline.json", o)

    def load_outline(self) -> Outline | None:
        return self._load_json("outline.json", Outline)

    # ----- continuity (resumable per-chapter summaries) --------------------------

    @property
    def _continuity_path(self) -> Path:
        return self.memory / "continuity.json"

    def load_continuity(self) -> dict[int, str]:
        if not self._continuity_path.exists():
            return {}
        raw = json.loads(self._continuity_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    def save_continuity(self, summaries: dict[int, str]) -> None:
        ordered = {str(k): summaries[k] for k in sorted(summaries)}
        self._continuity_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----- chapters --------------------------------------------------------------

    def chapter_path(self, number: int) -> Path:
        return self.chapters / f"ch{number:02d}.md"

    def has_chapter(self, number: int) -> bool:
        return self.chapter_path(number).exists()

    def save_chapter(self, number: int, title: str, body: str) -> None:
        text = f"# Chapter {number}: {title}\n\n{body.strip()}\n"
        self.chapter_path(number).write_text(text, encoding="utf-8")

    def read_chapter(self, number: int) -> str:
        text = self.chapter_path(number).read_text(encoding="utf-8")
        return re.sub(r"^#\s+Chapter\s+\d+:.*?\n+", "", text, count=1).strip()

    # ----- outputs ---------------------------------------------------------------

    @property
    def novel_path(self) -> Path:
        return self.dir / "novel.md"

    @property
    def critique_path(self) -> Path:
        return self.dir / "critique.md"
