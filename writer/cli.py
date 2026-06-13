"""Project-based, resumable CLI for the agentic novel writer (tyro subcommands).

Workflow:

    writer init <name>              create a project (asks language/length/prompt)
    writer outline --project <name> concept + characters + world + outline
    writer draft   --project <name> [--chapter N]   write chapters (all or one)
    writer revise  --project <name> [--chapter N]   polish + consistency pass
    writer critic  --project <name> review the finished novel

Every stage reads/writes files under the project folder, so any stage can be
re-run, resumed, or hand-edited between runs. Provider config comes from
./secret.yaml; a project remembers its providers, overridable per command.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Annotated, Optional, Union

import tyro

from .config import get_provider, load_providers
from .engine import StageError, Writer
from .project import DEFAULT_ROOT, Project, ProjectConfig


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        ans = input(f"{prompt}{suffix}: ").strip()
        if ans:
            return ans
        if default is not None:
            return default


def _build_writer(project: Project, provider: str | None, draft_provider: str | None) -> Writer:
    cfg = project.load_config()
    planner = get_provider(provider or cfg.planner)
    dname = draft_provider or cfg.drafter
    drafter = get_provider(dname) if dname else planner
    return Writer(project, planner, drafter)


def _run_stage(project_name: str, root: str, provider, draft_provider, fn) -> int:
    """Shared wrapper: open project, build writer, run `fn(writer)`, handle errors."""
    project = Project(project_name, root)
    if not project.exists():
        print(
            f"ERROR: project {project_name!r} not found under {root}/. "
            f"Run: writer init {project_name}",
            file=sys.stderr,
        )
        return 1
    try:
        writer = _build_writer(project, provider, draft_provider)
        fn(writer)
    except (StageError, KeyError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


@dataclass
class Init:
    """Create a new project (asks for anything not given on the command line)."""

    project: tyro.conf.Positional[str]
    language: Optional[str] = None
    length: Optional[int] = None
    prompt: Optional[str] = None
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        proj = Project(self.project, self.root)
        if proj.exists():
            print(f"ERROR: project {self.project!r} already exists.", file=sys.stderr)
            return 1
        try:
            known = list(load_providers())
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        language = self.language or _ask("Creation language", "English")
        length = self.length if self.length is not None else int(
            _ask("Target length (total words)", "40000")
        )
        prompt = self.prompt
        if prompt is None:
            raw = _ask("Instructions for the writer (optional)", "")
            prompt = raw or None
        provider = self.provider or (known[0] if known else None)

        proj.create(
            ProjectConfig(
                name=self.project,
                language=language,
                length=length,
                instructions=prompt,
                planner=provider,
                drafter=self.draft_provider,
            )
        )
        print(
            f"Created project {self.project!r} at {proj.dir}\n"
            f"  language={language}  length={length:,}  provider={provider}\n"
            f"Next: writer outline {self.project}"
        )
        return 0


@dataclass
class Outline:
    """Generate the story bible: concept, characters, world, and outline."""

    project: tyro.conf.Positional[str]
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.build_outline(),
        )


@dataclass
class Draft:
    """Draft chapters — all of them, or a single one with --chapter."""

    project: tyro.conf.Positional[str]
    chapter: Optional[int] = None
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.draft(self.chapter),
        )


@dataclass
class Revise:
    """Polish chapters and fix consistency — all, or a single one with --chapter."""

    project: tyro.conf.Positional[str]
    chapter: Optional[int] = None
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.revise(self.chapter),
        )


@dataclass
class Critic:
    """Have the LLM act as a critic and write a review of the novel."""

    project: tyro.conf.Positional[str]
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.critic(),
        )


Command = Union[
    Annotated[Init, tyro.conf.subcommand("init")],
    Annotated[Outline, tyro.conf.subcommand("outline")],
    Annotated[Draft, tyro.conf.subcommand("draft")],
    Annotated[Revise, tyro.conf.subcommand("revise")],
    Annotated[Critic, tyro.conf.subcommand("critic")],
]


def main() -> int:
    # Write UTF-8 regardless of the console codepage (Windows GBK, etc.).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cmd = tyro.cli(Command, prog="writer")
    return cmd.run()


if __name__ == "__main__":
    raise SystemExit(main())
