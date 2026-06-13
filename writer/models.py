"""Pydantic schemas for the planning stages (structured-output targets)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Concept(BaseModel):
    title: str = Field(description="The novel's title, in the target language.")
    topic: str = Field(
        description="A concise one-line topic/logline the LLM decides for the novel, "
        "derived from any user instructions or invented if none were given."
    )
    genre: str
    tone: str = Field(description="Overall mood/voice, e.g. 'wry and elegiac'.")
    premise: str = Field(description="One-paragraph hook.")
    themes: list[str] = Field(description="3-6 core themes.")
    synopsis: str = Field(description="A 200-400 word spoiler-full synopsis covering the whole arc.")


class Character(BaseModel):
    name: str
    role: str = Field(description="protagonist / antagonist / supporting / etc.")
    description: str = Field(description="Appearance, background, defining traits.")
    motivation: str = Field(description="What they want and why.")
    arc: str = Field(description="How they change across the novel.")


class Characters(BaseModel):
    characters: list[Character]


class Location(BaseModel):
    name: str
    description: str


class World(BaseModel):
    setting: str = Field(description="Where/when the story takes place.")
    time_period: str
    atmosphere: str
    rules: list[str] = Field(description="Physical, social, or magical rules that constrain the plot.")
    locations: list[Location]


class ChapterPlan(BaseModel):
    number: int
    title: str
    pov: str = Field(description="POV character or narrator for this chapter.")
    summary: str = Field(description="2-4 sentences: what happens in this chapter.")
    beats: list[str] = Field(description="3-6 ordered scene beats to hit.")
    purpose: str = Field(description="How this chapter advances plot/character/theme.")
    word_budget: int = Field(
        description="Target word count for THIS chapter, sized to its content. "
        "Chapters may differ in length; budgets across the novel should sum to "
        "roughly the requested total."
    )


class Outline(BaseModel):
    chapters: list[ChapterPlan]


# ----- critic artifacts (built by the critic from the finished novel) ---------


class Issue(BaseModel):
    aspect: str = Field(
        description="One of: plot, logic, character, pacing, consistency, theme."
    )
    severity: str = Field(description="'minor' or 'major'.")
    problem: str = Field(description="The specific problem, with a brief quote if useful.")
    suggestion: str = Field(description="A concrete, actionable fix for revision.")


class ChapterReview(BaseModel):
    number: int = Field(description="Chapter number being reviewed.")
    digest: str = Field(
        description="100-180 words: what actually happens in this chapter, written as "
        "the critic's own memory to reason about the whole novel later."
    )
    issues: list[Issue] = Field(
        description="Problems found in THIS chapter (may be empty if the chapter is strong)."
    )
    questions: list[str] = Field(
        description="Open questions about implausible plot turns, gaps, or unclear logic."
    )
