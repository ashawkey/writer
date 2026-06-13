"""Project-based, resumable CLI for the agentic novel writer (tyro subcommands).

Workflow:

    writer init <name>                  create a project (asks language/length/prompt)
    writer outline <name>               concept + characters + world + outline
    writer critic  <name>               developmental review of the plan
    writer revise  <name>               re-plan the bible from the critique
    writer draft   <name> [--chapter N] write chapters (all or one)
    writer polish  <name> [--chapter N] prose + consistency pass (all or one)

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


from html import escape

from prompt_toolkit import prompt as _ptk_prompt
from prompt_toolkit.formatted_text import HTML


def _ask(message: str, default: str = "", multiline: bool = False) -> str:
    """Prompt the user via prompt_toolkit. The default is shown as dimmed ghost
    text (a placeholder), so it reads clearly as a suggestion rather than typed
    input; pressing Enter on an empty line accepts it."""
    kwargs = {}
    if default:
        kwargs["placeholder"] = HTML(
            f'<style fg="ansibrightblack">{escape(default)} '
            "(default — press Enter)</style>"
        )
    if multiline:
        kwargs["multiline"] = True
        kwargs["bottom_toolbar"] = (
            " Enter = newline · Esc then Enter (or Alt+Enter) = submit "
        )
    text = _ptk_prompt(f"{message}: ", **kwargs).strip()
    return text or default


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
            raw = _ask(
                "Instructions for the writer (optional, multiline)", multiline=True
            )
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
class Polish:
    """Polish chapter prose and fix consistency — all, or one with --chapter.

    Light pass: improves language/pacing/consistency without changing the plan.
    Pass optional free-form INSTRUCTIONS to steer it, e.g.
    `writer polish mybook "avoid the word 'suddenly'; add more action detail"`.
    """

    project: tyro.conf.Positional[str]
    instructions: tyro.conf.Positional[Optional[str]] = None
    chapter: Optional[int] = None
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.polish(self.chapter, self.instructions),
        )


@dataclass
class Revise:
    """Revise the story bible (concept/characters/world/outline) from the critique.

    Run after `critic`. Rewrites only the plan in memory/ — chapters are left
    untouched; re-run `draft` afterwards to rewrite them against the new plan.
    """

    project: tyro.conf.Positional[str]
    provider: Optional[str] = None
    draft_provider: Optional[str] = None
    root: str = DEFAULT_ROOT

    def run(self) -> int:
        return _run_stage(
            self.project, self.root, self.provider, self.draft_provider,
            lambda w: w.revise(),
        )


@dataclass
class Critic:
    """Review the story PLAN (concept/characters/world/outline), before drafting.

    A developmental editor judges the plot, structure, and character arcs from
    memory/*.json and writes critique.md plus per-chapter notes. Run it after
    `outline` so you can `critic` -> `revise` the plan before spending tokens on
    `draft`."""

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
    Annotated[Critic, tyro.conf.subcommand("critic")],
    Annotated[Revise, tyro.conf.subcommand("revise")],
    Annotated[Draft, tyro.conf.subcommand("draft")],
    Annotated[Polish, tyro.conf.subcommand("polish")],
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
