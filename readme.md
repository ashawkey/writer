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
writer draft   <name>            # draft all chapters
writer draft   <name> --chapter 3   # (re)draft just chapter 3
writer revise  <name>            # polish + consistency pass (all chapters)
writer revise  <name> --chapter 3   # revise just chapter 3
writer critic  <name>            # LLM acts as a critic and reviews the novel
```

- **init** — saves `project.json` (language, length, instructions, providers).
- **outline** — the story bible: topic (always LLM-decided), title, genre,
  themes, synopsis, characters, world, and a chapter outline. The model decides
  the chapter count and each chapter's word budget (chapters may differ in
  length) to fit the target total.
- **draft** — writes chapter prose from the bible + a rolling continuity log
  (the "story so far"). With `--chapter N`, writes only that chapter, using the
  saved summaries of earlier chapters for continuity.
- **revise** — polishes prose and fixes consistency against the bible and the
  continuity log; all chapters or one.
- **critic** — switches the LLM into a critic persona and writes a structured
  review (`critique.md`) with strengths, weaknesses, consistency checks, and a
  score.

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

### As a library

```python
from writer import Project, ProjectConfig, Writer, get_provider

proj = Project("mybook")
proj.create(ProjectConfig(name="mybook", language="English", length=40000,
                          instructions="cozy mystery in a lighthouse"))
p = get_provider("deepseek")
w = Writer(proj, planner=p, drafter=p)
w.build_outline()
w.draft()            # or w.draft(chapter=3)
w.revise()
w.critic()
```

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
  novel.md                   # assembled novel (kept in sync after draft/revise)
  critique.md                # critic review
```

The `memory/*.json` files are the canonical, resumable state — edit them between
runs to steer the writer. `novel.md` and `critique.md` are the human-readable
outputs.
