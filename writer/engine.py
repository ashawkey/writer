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
from .models import ChapterReview, Characters, Concept, Outline, World
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

    def _plan(
        self,
        instruction: str,
        schema: type[BaseModel],
        retries: int = 3,
        system: str | None = None,
    ):
        """JSON-structured planning call validated against a pydantic schema."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        user = (
            f"{instruction}\n\n"
            "Respond with a SINGLE JSON object (no prose, no markdown fences) "
            f"that conforms to this JSON schema:\n{schema_json}"
        )
        messages = [
            {"role": "system", "content": system or self._system()},
            {"role": "user", "content": user},
        ]
        last_err = ""
        for _ in range(retries):
            fmt = {"type": "json_object"} if self._json_mode_ok else None
            try:
                content = self._stream_messages(
                    self._planner_client, self.planner.model, messages,
                    PLAN_MAX_TOKENS, response_format=fmt,
                )
            except Exception as e:
                if fmt is None:
                    raise
                # Provider may reject response_format — warn once, retry plain.
                self._json_mode_ok = False
                warnings.warn(
                    f"Provider {self.planner.name!r} rejected JSON mode "
                    f"(response_format); falling back to plain parsing. {e}",
                    stacklevel=2,
                )
                content = self._stream_messages(
                    self._planner_client, self.planner.model, messages, PLAN_MAX_TOKENS
                )
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

    def _stream_messages(
        self, client, model: str, messages: list, max_tokens: int, response_format=None
    ) -> str:
        """Streaming chat call; echoes to stdout and returns the full text."""
        kwargs = dict(model=model, max_tokens=max_tokens, stream=True, messages=messages)
        if response_format is not None:
            kwargs["response_format"] = response_format
        parts: list[str] = []
        for chunk in client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return "".join(parts).strip()

    def _stream(self, client, model: str, system: str, user: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._stream_messages(client, model, messages, max_tokens)

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

        if chapter is None:
            removed = self.project.prune_to({c.number for c in chapters})
            if removed:
                self._log(f"Pruned {len(removed)} stale chapter(s): {removed}")

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

    # ----- polish (per-chapter language + consistency) ---------------------------

    def polish(self, chapter: int | None = None, instructions: str | None = None) -> None:
        self._load_bible()
        summaries = self.project.load_continuity()
        instructions = (instructions or "").strip() or None
        if instructions:
            self._log(f"Polish focus: {instructions}")

        if chapter is None:
            targets = [c for c in self.outline.chapters if self.project.has_chapter(c.number)]
            if not targets:
                raise StageError("No drafted chapters to revise. Run 'draft' first.")
        else:
            ch = self._get_chapter(chapter)
            if not self.project.has_chapter(chapter):
                raise StageError(f"Chapter {chapter} not drafted yet.")
            targets = [ch]

        focus_block = (
            "PRIORITIZE THIS REVISION INSTRUCTION above the general goals below:\n"
            f"{instructions}\n\n"
            if instructions
            else ""
        )
        for ch in targets:
            self._log(f"\n--- Revising Chapter {ch.number}: {ch.title} ---")
            draft = self.project.read_chapter(ch.number)
            continuity = self._continuity_before(ch.number, summaries)
            cont_block = (
                f"STORY SO FAR (for consistency):\n{continuity}\n\n" if continuity else ""
            )
            user = (
                f"{cont_block}{focus_block}"
                f'Polish Chapter {ch.number} ("{ch.title}") below. '
                "Improve prose quality, pacing, and word choice; fix continuity or "
                "characterization slips against the story bible and the story so "
                "far; tighten weak passages. Keep the plot and structure intact and "
                "preserve the approximate length. Output ONLY the polished chapter "
                f"prose, in {self.language}, with no heading or notes.\n\n"
                f"--- CURRENT DRAFT ---\n{draft}"
            )
            revised = self._write_stream(self._bible_system(), user)
            self.project.save_chapter(ch.number, ch.title, revised)
            summaries[ch.number] = self._summarize(ch.number, ch.title, revised)
            self.project.save_continuity(summaries)

        self._assemble()

    # ----- revise (re-plan the story bible from the critique) --------------------

    def revise(self) -> None:
        """Revise the story bible (concept/characters/world/outline) using the
        critic's review. Chapters are NOT touched — re-run 'draft' afterwards to
        rewrite them against the revised bible."""
        if not self.project.has_critique():
            raise StageError("No critique found. Run 'critic' first.")
        self._load_bible()
        crit = (
            "=== CRITIC REVIEW ===\n"
            f"{self.project.read_critique()}\n\n"
            "=== PER-CHAPTER ISSUES ===\n"
            f"{self.project.read_critique_issues()}"
        )
        guide = (
            "A critic reviewed the drafted novel. Revise the story plan to fix the "
            "problems they raised while preserving what already works. You may "
            "restructure, merge, split, add, or cut chapters as needed.\n\n"
            f"{crit}"
        )

        self._log("\n=== Revising concept ===")
        self.concept = self._plan(
            f"{guide}\n\nHere is the CURRENT concept:\n{self._fmt_concept()}\n\n"
            "Produce a revised concept that addresses the critique. Keep the core "
            "identity unless the critique calls for a deeper change.",
            Concept,
        )
        self.project.save_concept(self.concept)

        self._log("\n=== Revising characters ===")
        self.characters = self._plan(
            f"{guide}\n\nRevised concept:\n{self._fmt_concept()}\n\n"
            f"Current characters:\n{self._fmt_characters()}\n\n"
            "Produce a revised cast addressing the critique (fix flat or "
            "inconsistent characters, sharpen distinct voices).",
            Characters,
        )
        self.project.save_characters(self.characters)

        self._log("\n=== Revising world ===")
        self.world = self._plan(
            f"{guide}\n\nRevised concept:\n{self._fmt_concept()}\n\n"
            f"Revised characters:\n{self._fmt_characters()}\n\n"
            f"Current world:\n{self._fmt_world()}\n\n"
            "Produce a revised world that resolves any setting/logic problems the "
            "critique raised.",
            World,
        )
        self.project.save_world(self.world)

        self._log("\n=== Revising outline ===")
        self.outline = self._plan(
            f"{guide}\n\nRevised concept:\n{self._fmt_concept()}\n\n"
            f"Revised characters:\n{self._fmt_characters()}\n\n"
            f"Revised world:\n{self._fmt_world()}\n\n"
            f"Current outline:\n{self._fmt_outline()}\n\n"
            "Produce a revised chapter-by-chapter outline that fixes the plot, "
            "pacing, and logic problems from the critique. You decide the chapter "
            "count and each chapter's word_budget; budgets should sum to roughly "
            f"{self.length:,}. Number chapters from 1.",
            Outline,
        )
        for i, ch in enumerate(self.outline.chapters, start=1):
            ch.number = i
            ch.word_budget = max(300, ch.word_budget)
        self.project.save_outline(self.outline)
        self._bible_cache = None

        removed = self.project.prune_to({ch.number for ch in self.outline.chapters})
        if removed:
            self._log(f"Pruned {len(removed)} stale chapter(s): {removed}")

        self._log(
            f"\nBible revised ({len(self.outline.chapters)} chapters). "
            "Re-run 'draft' to rewrite the chapters against the revised plan:\n"
            f"  writer draft {self.project.name}"
        )

    # ----- critic ----------------------------------------------------------------

    def _critic_system(self) -> str:
        return (
            "You are an independent, rigorous developmental editor. You are "
            "reviewing a novel's PLAN — its premise, characters, world, and chapter "
            "outline — BEFORE it is written. Judge the story itself: plot logic and "
            "structure, character arcs, motivation, setup and payoff, pacing, "
            "originality, thematic coherence, and plausibility. Do NOT comment on "
            "prose or line-level writing — it does not exist yet. Be candid about "
            "weaknesses and generous about genuine strengths, and ground every "
            "judgment in the plan. "
            f"Write all output in {self.language}."
        )

    def critic(self) -> None:
        """Story-level review of the PLAN (memory/*.json), before drafting.

        Reads the concept, characters, world, and outline and judges the story —
        plot, structure, arcs, pacing — so you can critic+revise the outline
        before spending tokens on draft/polish."""
        self._load_bible()
        self.project.init_critique()
        sys_ = self._critic_system()
        context = (
            f"{self._fmt_concept()}\n\n{self._fmt_characters()}\n\n{self._fmt_world()}"
        )

        # Step 1: review each chapter's PLAN, building running notes.
        notes: list[str] = []
        reviews = []
        for ch in self.outline.chapters:
            self._log(f"\n--- Reviewing the plan for Chapter {ch.number}: {ch.title} ---")
            review = self._review_chapter_plan(ch, context, "\n\n".join(notes), sys_)
            review.number = ch.number
            self.project.save_chapter_review(review)
            notes.append(f"Ch {ch.number} ({ch.title}): {review.digest}")
            reviews.append((ch.number, ch.title, review))

        # Step 2: aggregate the actionable per-chapter issues (for revision).
        self.project.save_critique_issues(self._format_issues(reviews))

        # Step 3: synthesize the final whole-story developmental review.
        self._log("\n--- Final review ---\n")
        final = self._stream(
            self._planner_client, self.planner.model, sys_,
            "Write a structured Markdown developmental review of this novel PLAN, "
            "judging the story across these aspects, each its own section: Premise & "
            "hook; Plot & logic (call out plot holes, implausible turns, weak "
            "causality); Structure & pacing; Characters & arcs; World & setting "
            "coherence; Theme & originality; Consistency. Then a 'Top priorities "
            "before drafting' list and a final Verdict with a score out of 10. Cite "
            "chapters. Base it on the plan and your per-chapter notes below.\n\n"
            f"=== STORY PLAN ===\n{context}\n\n{self._fmt_outline()}\n\n"
            f"=== YOUR PER-CHAPTER NOTES ===\n{self._format_issues(reviews)}",
            CRITIC_MAX_TOKENS,
        )
        self.project.save_critique(final)
        self._log(
            f"\nReview saved to {self.project.critique_path}\n"
            f"Per-chapter notes under {self.project.critique}/. "
            "Run 'revise' to rework the plan, then 'draft'."
        )

    def _review_chapter_plan(self, ch, context, prior_notes, system):
        notes_block = (
            f"Your notes on earlier chapters:\n{prior_notes}\n\n" if prior_notes else ""
        )
        beats = "\n".join(f"  - {b}" for b in ch.beats)
        instruction = (
            f"Story so far (premise, characters, world):\n{context}\n\n"
            f"{notes_block}"
            f"Review the PLAN for Chapter {ch.number} (\"{ch.title}\").\n"
            f"- POV: {ch.pov}\n- Summary: {ch.summary}\n- Purpose: {ch.purpose}\n"
            f"- Beats:\n{beats}\n\n"
            "Write a one-line digest of what this chapter is meant to accomplish, "
            "then list concrete story issues (plot, logic, character, pacing, "
            "consistency, theme) with actionable fixes, and open questions about "
            "implausible or unclear developments. If the plan is strong, issues may "
            "be few or empty — do not invent problems."
        )
        return self._plan(instruction, ChapterReview, system=system)

    def _format_issues(self, reviews) -> str:
        out = []
        for number, title, review in reviews:
            lines = [f"## Chapter {number}: {title}"]
            if review.issues:
                for i in review.issues:
                    lines.append(
                        f"- [{i.aspect}/{i.severity}] {i.problem}\n  → {i.suggestion}"
                    )
            if review.questions:
                for q in review.questions:
                    lines.append(f"- [question] {q}")
            if not review.issues and not review.questions:
                lines.append("- (no significant issues)")
            out.append("\n".join(lines))
        return "\n\n".join(out)

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
