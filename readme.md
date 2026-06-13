# Writer

Agentic system that lets an LLM write long, complex novels. It simulates a
human writer's workflow as a sequence of stages backed by on-disk "memory", so
context stays bounded even for book-length output.

It is **project-based and resumable**: every stage reads its inputs from, and
writes its outputs to, a project folder. You can re-run any stage, redraft a
single chapter, hand-edit the memory files between runs, or pick up where you
left off — nothing the next command needs is held only in memory.

## Workflow

```bash
writer init <name>               # create a project (asks language/length/prompt)
writer outline <name>            # concept + characters + world + outline
writer critic  <name>            # review the PLAN (plot/structure) before drafting
writer revise  <name>            # rewrite the bible from the critique
writer draft   <name>            # draft all chapters
writer draft   <name> --chapter 3   # (re)draft just chapter 3
writer polish  <name>            # light language/consistency pass (all chapters)
writer polish  <name> --chapter 3   # polish just chapter 3
writer polish  <name> "more action detail; avoid 'suddenly'"   # one-off focus
```

- **init** — saves `project.json` (language, length, instructions, providers).
- **outline** — the story bible: topic, title, genre, themes, synopsis, characters,
  world, and a chapter outline. The model decides the chapter count and each
  chapter's word budget (chapters may differ in length) to fit the target total.
- **draft** — writes chapter prose from the bible + a rolling continuity log
  (the "story so far"). With `--chapter N`, writes only that chapter, using the
  saved summaries of earlier chapters for continuity.
- **polish** — a light pass that improves prose, pacing, and consistency
  *without changing the plan*; all chapters or one. Takes an optional one-off
  instruction (e.g. `"avoid the word 'suddenly'"`) to steer just that pass. Does
  not read the critique.
- **critic** — a developmental editor that reviews the **plan** (`memory/*.json`:
  concept, characters, world, outline) *before* drafting. It judges the story —
  plot logic, structure, character arcs, pacing, originality — chapter by
  chapter, flags issues and open questions, and writes a structured review.
  Outputs `critique.md` plus per-chapter notes and `issues.md` under `critique/`.
- **revise** — the structural fix. Run after `critic`: it rewrites the **story
  bible** (concept/characters/world/outline) in `memory/` to address the
  critique. It does not touch chapter drafts — `draft` regenerates them against
  the revised plan (and prunes any chapters the new outline dropped).

A typical loop: `outline` → `critic` → `revise` → (repeat) → `draft` → `polish`.
Because `critic`/`revise` work on the plan, you iterate the story cheaply before
spending tokens on prose.

Planning stages use JSON-structured output validated against schemas
(`writer/models.py`); chapter and review text are streamed.

## Providers

Any OpenAI-compatible Chat Completions endpoint works (DeepSeek, OpenAI,
local, ...). Configure in `secret.yaml` (copy `secret.yaml.example`):

```yaml
openai:
  deepseek:
    model: deepseek-chat
    api_key: sk-...
    base_url: https://api.deepseek.com/v1
```

A project remembers its providers (chosen at `init`). Override per command with
`--provider <name>` (planning) and, optionally, `--draft-provider <name>` to run
the heavy chapter drafting on a different model. By default drafting uses the
same provider as planning.

## Setup

```bash
pip install -e .                     # installs the `writer` command
cp secret.yaml.example secret.yaml   # then edit
```

## Example

```bash
writer init mybook --language English --length 50000 \
    --prompt "cozy mystery, first-person, no graphic violence" \
    --provider deepseek
writer outline mybook
writer draft   mybook
writer revise  mybook
writer critic  mybook
```

`init` asks interactively for anything not passed as a flag. Use `--root <dir>`
to put projects somewhere other than `projects/`.

## Project layout

```
projects/<name>/
  project.json               # config (language, length, instructions, providers)
  memory/
    concept.json               # topic, title, genre, themes, synopsis
    characters.json
    world.json
    outline.json               # chapter count + per-chapter word budgets
    continuity.json            # resumable per-chapter summaries
  chapters/
    ch01.md, ch02.md, ...
  critique/                  # critic's working notes
    chapters/chNN.json         # per-chapter digest + issues + questions
    issues.md                  # aggregated actionable issues (feeds revise)
  novel.md                   # assembled novel (kept in sync after draft/polish)
  critique.md                # developmental review of the plan
```

The `memory/*.json` files are the canonical, resumable state — edit them between
runs to steer the writer. `novel.md` and `critique.md` are the human-readable
outputs.
