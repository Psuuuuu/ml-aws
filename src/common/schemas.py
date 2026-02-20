from pydantic import BaseModel, Field
from typing import Literal, Annotated


FEATURE_COLUMNS: list[str] = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

TARGET_COLUMN = "exam_score"

# ==========================================================
# 1. Feature Schema (Single Data Point)
# ==========================================================


class StudentFeatures(BaseModel):
    age: Annotated[
        int,
        Field(
            ...,
            ge=0,
            le=100,
            title="Age of the student",
            description="Age of the student <=0 & >=100",
            examples=[22, 30],
            strict=True,
        ),
    ]
    gender: Annotated[
        Literal["female", "other", "male"],
        Field(
            ...,
            title="Gender",
            description="Gender of the student",
            examples=["female", "other", "male"],
            strict=True,
        ),
    ]
    course: Annotated[
        Literal["b.sc", "diploma", "bca", "b.com", "ba", "bba", "b.tech"],
        Field(
            ...,
            title="Course",
            description="Course enrolled by student",
            examples=["b.sc", "diploma", "bca", "b.com", "ba", "bba", "b.tech"],
            strict=True,
        ),
    ]
    study_hours: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=24,
            title="Study hours",
            description="Student average study hours",
            examples=[4.5, 6, 8],
            strict=True,
        ),
    ]
    class_attendance: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=100,
            title="Class attendance %",
            description="Class attendance % >=0 & <=100",
            examples=[40, 50, 100],
            strict=True,
        ),
    ]
    internet_access: Annotated[
        Literal["no", "yes"],
        Field(
            ...,
            title="Internet access",
            description="Internet access",
            examples=["no", "yes"],
            strict=True,
        ),
    ]
    sleep_hours: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=24,
            title="Sleep hours",
            description="Sleep hours",
            examples=[5, 7, 9],
            strict=True,
        ),
    ]
    sleep_quality: Annotated[
        Literal["average", "poor", "good"],
        Field(
            ...,
            title="Sleep quality",
            description="Sleep quality",
            examples=["average", "poor", "good"],
            strict=True,
        ),
    ]
    study_method: Annotated[
        Literal["online videos", "self-study", "coaching", "group study", "mixed"],
        Field(
            ...,
            title="Study method",
            description="Study method",
            examples=[
                "online videos",
                "self-study",
                "coaching",
                "group study",
                "mixed",
            ],
            strict=True,
        ),
    ]
    facility_rating: Annotated[
        Literal["low", "medium", "high"],
        Field(
            ...,
            title="Facility rating",
            description="Facility rating",
            examples=["low", "medium", "high"],
            strict=True,
        ),
    ]
    exam_difficulty: Annotated[
        Literal["easy", "moderate", "hard"],
        Field(
            ...,
            title="Exam difficulty",
            description="Exam difficulty",
            examples=["easy", "moderate", "hard"],
            strict=True,
        ),
    ]
