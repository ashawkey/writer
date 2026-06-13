"""Agentic novel-writing engine (OpenAI-compatible / provider-agnostic).

Project-backed and resumable: every operation reads its inputs from, and writes
its outputs to, a Project on disk (see project.py). Operations:

    build_outline  -> concept, characters, world, outline (the story bible)
    draft          -> chapter prose (all, or one by index), + continuity log
    revise         -> polish/consistency pass (all, or one)
    critic         -> a critical review of the finished novel

The model layer talks to any OpenAI-compatible Chat Completions endpoint
(DeepSeek, OpenAI, local, ...) via config.Provider.
"""

from __future__ import annotations

import json
import re
import sys
import warnings

from pydantic import BaseModel, ValidationError

from .config import Provider
from .models import Characters, Concept, Outline, World
from .project import Project

PLAN_MAX_TOKENS = 8000
DRAFT_MAX_TOKENS = 8000
SUMMARY_MAX_TOKENS = 1000
CRITIC_MAX_TOKENS = 8000


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text[start:]


class StageError(RuntimeError):
    """Raised when a required prior stage's artifacts are missing."""


class Writer:
    def __init__(self, project: Project, planner: Provider, drafter: Provider):
        self.project = project
        cfg = project.load_config()
        self.language = cfg.language
        self.length = cfg.length
        self.instructions = (cfg.instructions or "").strip() or None

        self.planner = planner
        self.drafter = drafter
        self._planner_client = planner.client()
        self._drafter_client = drafter.client()
        self._json_mode_ok = True

        # Loaded lazily from the project's saved artifacts.
        self.concept: Concept | None = None
        self.characters: Characters | None = None
        self.world: World | None = None
        self.outline: Outline | None = None
        self._bible_cache: str | None = None

    # ----- low-level model calls -------------------------------------------------

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    def _system(self) -> str:
        base = (
            "You are a master novelist and story architect. You write vivid, "
            "emotionally resonant literary fiction with strong structure and "
            "consistent characterization.\n"
            f"The novel MUST be written entirely in {self.language}. All prose, "
            "dialogue, titles, names, and content you generate go in "
            f"{self.language}, regardless of the language of these instructions."
        )
        if self.instructions:
            base += (
                "\n\nThe user provided the following instructions for this novel. "
                "Honor them throughout (genre, style, content, constraints, etc.):\n"
                f"{self.instructions}"
            )
        return base

    def _plan(self, instruction: str, schema: type[BaseModel], retries: int = 3):
        """JSON-structured planning call validated against a pydantic schema."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        user = (
            f"{instruction}\n\n"
            "Respond with a SINGLE JSON object (no prose, no markdown fences) "
            f"that conforms to this JSON schema:\n{schema_json}"
        )
        messages = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": user},
        ]
        last_err = ""
        for _ in range(retries):
            kwargs = dict(
                model=self.planner.model,
                max_tokens=PLAN_MAX_TOKENS,
                messages=messages,
            )
            if self._json_mode_ok:
                try:
                    resp = self._planner_client.chat.completions.create(
                        response_format={"type": "json_object"}, **kwargs
                    )
                except Exception as e:
                    self._json_mode_ok = False
                    warnings.warn(
                        f"Provider {self.planner.name!r} rejected JSON mode "
                        f"(response_format); falling back to plain parsing. {e}",
                        stacklevel=2,
                    )
                    resp = self._planner_client.chat.completions.create(**kwargs)
            else:
                resp = self._planner_client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_extract_json(content))
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_err = str(e)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That did not validate against the schema. Error:\n"
                            f"{last_err}\nReturn corrected JSON only."
                        ),
                    }
                )
        raise RuntimeError(f"Planning failed after {retries} tries: {last_err}")

    def _stream(self, client, model: str, system: str, user: str, max_tokens: int) -> str:
        """Streaming call; echoes to stdout and returns the full text."""
        parts: list[str] = []
        stream = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return "".join(parts).strip()

    def _write_stream(self, system: str, user: str) -> str:
        return self._stream(
            self._drafter_client, self.drafter.model, system, user, DRAFT_MAX_TOKENS
        )

    # ----- outline (concept + characters + world + outline) ----------------------

    def build_outline(self) -> None:
        self._stage_concept()
        self._stage_characters()
        self._stage_world()
        self._stage_outline()
        self._log("\nOutline complete. Story bible saved under memory/.")

    def _stage_concept(self) -> None:
        self._log("\n=== Concept ===")
        guidance = (
            f"The user gave these instructions/wishes (honor them): {self.instructions!r}"
            if self.instructions
            else "The user gave no specific instructions."
        )
        instruction = (
            f"Design the concept for a {self.length:,}-word novel.\n{guidance}\n"
            "Decide the novel's topic yourself — a concise logline. If the user gave "
            "a subject or constraints, derive the topic from them; otherwise invent "
            "something compelling and original. Then decide title, genre, tone, "
            "premise, themes, and a full-arc synopsis. Make it cohesive and "
            "distinctive — avoid generic, derivative ideas."
        )
        self.concept = self._plan(instruction, Concept)
        self.project.save_concept(self.concept)
        self._log(f"  Title: {self.concept.title}  ({self.concept.genre})")
        self._log(f"  Topic: {self.concept.topic}")

    def _stage_characters(self) -> None:
        self._log("\n=== Characters ===")
        instruction = (
            "Based on the concept below, design the cast. Include the protagonist, "
            "antagonist (if any), and key supporting characters. "
            "Give each a distinct voice so their dialogue never blurs together.\n\n"
            f"{self._fmt_concept()}"
        )
        self.characters = self._plan(instruction, Characters)
        self.project.save_characters(self.characters)
        self._log(f"  {len(self.characters.characters)} characters designed.")

    def _stage_world(self) -> None:
        self._log("\n=== World & setting ===")
        instruction = (
            "Based on the concept and characters below, define the story world: "
            "setting, time period, atmosphere, the rules that constrain the plot, "
            "and the key locations.\n\n"
            f"{self._fmt_concept()}\n\n{self._fmt_characters()}"
        )
        self.world = self._plan(instruction, World)
        self.project.save_world(self.world)
        self._log(f"  Setting: {self.world.setting}")

    def _stage_outline(self) -> None:
        self._log("\n=== Outline ===")
        instruction = (
            f"Produce a chapter-by-chapter outline for this ~{self.length:,}-word "
            "novel. YOU decide how many chapters best serve the story and how long "
            "each chapter should be — assign each chapter a word_budget sized to its "
            "content (climactic or eventful chapters can be longer, quieter ones "
            "shorter). The chapters together must tell the complete arc from the "
            "synopsis with rising tension, a climax, and a resolution, and their "
            f"word_budget values should sum to roughly {self.length:,}. Number the "
            "chapters from 1.\n\n"
            f"{self._fmt_concept()}\n\n{self._fmt_characters()}\n\n{self._fmt_world()}"
        )
        self.outline = self._plan(instruction, Outline)
        for i, ch in enumerate(self.outline.chapters, start=1):
            ch.number = i
            ch.word_budget = max(300, ch.word_budget)
        self.project.save_outline(self.outline)
        total = sum(ch.word_budget for ch in self.outline.chapters)
        self._log(
            f"  {len(self.outline.chapters)} chapters planned "
            f"(~{total:,} words budgeted)."
        )

    # ----- draft -----------------------------------------------------------------

    def draft(self, chapter: int | None = None) -> None:
        self._load_bible()
        chapters = self.outline.chapters
        summaries = self.project.load_continuity()

        if chapter is None:
            targets = chapters
            summaries = {}  # full pass — rebuild continuity from scratch
        else:
            ch = self._get_chapter(chapter)
            targets = [ch]

        for ch in targets:
            self._log(f"\n--- Drafting Chapter {ch.number}: {ch.title} ---")
            continuity = self._continuity_before(ch.number, summaries)
            body = self._write_stream(self._bible_system(), self._draft_prompt(ch, continuity))
            self.project.save_chapter(ch.number, ch.title, body)
            summaries[ch.number] = self._summarize(ch.number, ch.title, body)
            self.project.save_continuity(summaries)

        self._assemble()

    def _draft_prompt(self, ch, continuity: str) -> str:
        beats = "\n".join(f"  - {b}" for b in ch.beats)
        cont_block = (
            f"STORY SO FAR (continuity — stay consistent with this):\n{continuity}\n\n"
            if continuity
            else ""
        )
        return (
            f"{cont_block}"
            f"Write Chapter {ch.number} of {len(self.outline.chapters)}, titled "
            f'"{ch.title}".\n'
            f"POV: {ch.pov}\n"
            f"Summary: {ch.summary}\n"
            f"Purpose: {ch.purpose}\n"
            f"Beats to hit, in order:\n{beats}\n\n"
            f"Target length: about {ch.word_budget:,} words. "
            "Write finished, immersive prose — scene-setting, action, and "
            "dialogue. Do NOT include the chapter number/title heading, summaries, "
            "author notes, or meta commentary. Output only the chapter's prose, in "
            f"{self.language}."
        )

    # ----- revise ----------------------------------------------------------------

    def revise(self, chapter: int | None = None) -> None:
        self._load_bible()
        summaries = self.project.load_continuity()

        if chapter is None:
            targets = [c for c in self.outline.chapters if self.project.has_chapter(c.number)]
            if not targets:
                raise StageError("No drafted chapters to revise. Run 'draft' first.")
        else:
            ch = self._get_chapter(chapter)
            if not self.project.has_chapter(chapter):
                raise StageError(f"Chapter {chapter} not drafted yet.")
            targets = [ch]

        for ch in targets:
            self._log(f"\n--- Revising Chapter {ch.number}: {ch.title} ---")
            draft = self.project.read_chapter(ch.number)
            continuity = self._continuity_before(ch.number, summaries)
            cont_block = (
                f"STORY SO FAR (for consistency):\n{continuity}\n\n" if continuity else ""
            )
            user = (
                f"{cont_block}"
                f'Revise and polish Chapter {ch.number} ("{ch.title}") below. '
                "Improve prose quality, pacing, and word choice; fix continuity or "
                "characterization slips against the story bible and the story so "
                "far; tighten weak passages. Keep the plot and structure intact and "
                "preserve the approximate length. Output ONLY the revised chapter "
                f"prose, in {self.language}, with no heading or notes.\n\n"
                f"--- CURRENT DRAFT ---\n{draft}"
            )
            revised = self._write_stream(self._bible_system(), user)
            self.project.save_chapter(ch.number, ch.title, revised)
            summaries[ch.number] = self._summarize(ch.number, ch.title, revised)
            self.project.save_continuity(summaries)

        self._assemble()

    # ----- critic ----------------------------------------------------------------

    def critic(self) -> None:
        self._load_bible()
        drafted = [c for c in self.outline.chapters if self.project.has_chapter(c.number)]
        if not drafted:
            raise StageError("No drafted chapters to critique. Run 'draft' first.")
        self._assemble()
        novel = self.project.novel_path.read_text(encoding="utf-8")

        system = (
            "You are a sharp, fair, and experienced literary critic and editor. "
            "You judge novels on their merits — prose quality, structure, pacing, "
            "characterization, theme, originality, and internal consistency — and "
            "you back every judgment with specific evidence from the text. You are "
            "candid about weaknesses and generous about genuine strengths. "
            f"Write your review in {self.language}."
        )
        user = (
            "Review the following novel. Produce a structured critique in Markdown "
            "with these sections: Overall impression; Strengths; Weaknesses; Prose & "
            "style; Plot & structure; Characters; Pacing; Consistency (flag any "
            "contradictions or continuity errors with chapter references); Concrete "
            "suggestions for revision; and a final Verdict with a score out of 10. "
            "Be specific and cite chapters.\n\n"
            f"=== STORY BIBLE (for reference) ===\n{self._bible()}\n\n"
            f"=== NOVEL ===\n{novel}"
        )
        self._log("\n=== Critic review ===\n")
        review = self._stream(
            self._planner_client, self.planner.model, system, user, CRITIC_MAX_TOKENS
        )
        self.project.critique_path.write_text(review + "\n", encoding="utf-8")
        self._log(f"\nReview saved to {self.project.critique_path}")

    # ----- assembly --------------------------------------------------------------

    def _assemble(self) -> None:
        c = self.concept
        lines = [f"# {c.title}", "", f"*{c.genre} — {c.tone}*", "", "---", ""]
        count = 0
        for ch in self.outline.chapters:
            if not self.project.has_chapter(ch.number):
                continue
            count += 1
            lines.append(f"## Chapter {ch.number}: {ch.title}")
            lines.append("")
            lines.append(self.project.read_chapter(ch.number))
            lines.append("")
        self.project.novel_path.write_text("\n".join(lines), encoding="utf-8")
        words = sum(
            len(self.project.read_chapter(ch.number).split())
            for ch in self.outline.chapters
            if self.project.has_chapter(ch.number)
        )
        self._log(
            f"\nAssembled {self.project.novel_path}  "
            f"({count}/{len(self.outline.chapters)} chapters, ~{words:,} words)"
        )

    # ----- continuity ------------------------------------------------------------

    def _continuity_before(self, number: int, summaries: dict[int, str]) -> str:
        parts = []
        titles = {ch.number: ch.title for ch in self.outline.chapters}
        for n in sorted(s for s in summaries if s < number):
            parts.append(f"Ch {n} ({titles.get(n, '')}): {summaries[n]}")
        return "\n\n".join(parts)

    def _summarize(self, number: int, title: str, body: str) -> str:
        resp = self._planner_client.chat.completions.create(
            model=self.planner.model,
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[
                {"role": "system", "content": self._system()},
                {
                    "role": "user",
                    "content": (
                        "Summarize this chapter in 120-180 words for a running "
                        "story-continuity log: key plot events, character state "
                        "changes, new facts, unresolved threads. Be concrete "
                        f"(names, places). Write in {self.language}.\n\n"
                        f"Chapter {number}: {title}\n\n{body}"
                    ),
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    # ----- loading / bible -------------------------------------------------------

    def _load_bible(self) -> None:
        self.concept = self.project.load_concept()
        self.characters = self.project.load_characters()
        self.world = self.project.load_world()
        self.outline = self.project.load_outline()
        missing = [
            name
            for name, obj in (
                ("concept", self.concept),
                ("characters", self.characters),
                ("world", self.world),
                ("outline", self.outline),
            )
            if obj is None
        ]
        if missing:
            raise StageError(
                f"Missing {', '.join(missing)}. Run 'outline' first."
            )
        self._bible_cache = None

    def _get_chapter(self, number: int):
        for ch in self.outline.chapters:
            if ch.number == number:
                return ch
        raise StageError(
            f"Chapter {number} is out of range (1..{len(self.outline.chapters)})."
        )

    def _bible_system(self) -> str:
        return f"{self._system()}\n\n{self._bible()}"

    def _bible(self) -> str:
        if self._bible_cache is None:
            self._bible_cache = (
                "=== STORY BIBLE ===\n\n"
                f"{self._fmt_concept()}\n\n"
                f"{self._fmt_characters()}\n\n"
                f"{self._fmt_world()}\n\n"
                f"{self._fmt_outline()}"
            )
        return self._bible_cache

    # ----- formatting ------------------------------------------------------------

    def _fmt_concept(self) -> str:
        c = self.concept
        return (
            f"## Concept\n"
            f"- Title: {c.title}\n"
            f"- Topic: {c.topic}\n"
            f"- Genre: {c.genre}\n"
            f"- Tone: {c.tone}\n"
            f"- Themes: {', '.join(c.themes)}\n"
            f"- Premise: {c.premise}\n\n"
            f"### Synopsis\n{c.synopsis}"
        )

    def _fmt_characters(self) -> str:
        out = ["## Characters"]
        for ch in self.characters.characters:
            out.append(
                f"### {ch.name} ({ch.role})\n"
                f"- Description: {ch.description}\n"
                f"- Motivation: {ch.motivation}\n"
                f"- Arc: {ch.arc}\n"
                f"- Voice: {ch.voice}"
            )
        return "\n\n".join(out)

    def _fmt_world(self) -> str:
        w = self.world
        locs = "\n".join(f"- {l.name}: {l.description}" for l in w.locations)
        rules = "\n".join(f"- {r}" for r in w.rules)
        return (
            f"## World\n"
            f"- Setting: {w.setting}\n"
            f"- Time period: {w.time_period}\n"
            f"- Atmosphere: {w.atmosphere}\n\n"
            f"### Rules\n{rules}\n\n"
            f"### Locations\n{locs}"
        )

    def _fmt_outline(self) -> str:
        out = ["## Outline"]
        for ch in self.outline.chapters:
            beats = "\n".join(f"  - {b}" for b in ch.beats)
            out.append(
                f"### Chapter {ch.number}: {ch.title}\n"
                f"- POV: {ch.pov}\n"
                f"- Word budget: ~{ch.word_budget:,}\n"
                f"- Summary: {ch.summary}\n"
                f"- Purpose: {ch.purpose}\n"
                f"- Beats:\n{beats}"
            )
        return "\n\n".join(out)
