from __future__ import annotations
from typing import Annotated, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class BulletItem(BaseModel):
    kind: Literal["arrow", "numbered", "star", "warn"] = "arrow"
    text: Annotated[str, Field(min_length=1, max_length=400)]
    number: int | None = Field(default=None)

    @field_validator("text")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def numbered_requires_number(self) -> "BulletItem":
        if self.kind == "numbered" and self.number is None:
            raise ValueError("BulletItem with kind='numbered' must supply a 'number' value.")
        return self

class Section(BaseModel):
    heading: Annotated[str, Field(min_length=1, max_length=120)]
    bullets: Annotated[List[BulletItem], Field(min_length=1)]

class ModuleNotes(BaseModel):
    subtitle: Annotated[str, Field(min_length=1, max_length=200)]
    sections: Annotated[List[Section], Field(min_length=1)]

class CourseRoadmap(BaseModel):
    title: str
    date: str
    modules: List[ModuleNotes]
    
    @field_validator("title", "date")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip()