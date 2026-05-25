from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class UserStory(BaseModel):
    role: str
    action: str
    benefit: str


class ProductSpec(BaseModel):
    title: str
    description: str
    features: List[str]
    user_stories: List[UserStory]
    acceptance_criteria: List[str]
    tech_preferences: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    out_of_scope: List[str] = Field(default_factory=list)


class ArchFile(BaseModel):
    path: str
    file_type: str
    description: str


class TechDecision(BaseModel):
    decision: str
    rationale: str


class Architecture(BaseModel):
    overview: str
    tech_stack: List[str]
    pip_dependencies: List[str]
    dev_dependencies: List[str] = Field(default_factory=list)
    file_structure: List[ArchFile]
    key_decisions: List[TechDecision]
    api_design: Optional[str] = None
    database_schema: Optional[str] = None
    environment_variables: List[str] = Field(default_factory=list)


class CodeFile(BaseModel):
    filename: str
    content: str
    language: str
    description: str


class SetupStep(BaseModel):
    step: int
    command: str
    description: str


class CodeOutput(BaseModel):
    files: List[CodeFile]
    setup_steps: List[SetupStep]
    run_command: str
    test_command: Optional[str] = None
    notes: Optional[str] = None


class Issue(BaseModel):
    severity: IssueSeverity
    file: str
    line_range: Optional[str] = None
    description: str
    suggestion: str


class QAReport(BaseModel):
    issues: List[Issue]
    test_files: List[CodeFile]
    coverage_assessment: str
    overall_score: int = Field(..., ge=0, le=10)
    overall_assessment: str
    critical_issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)


class ReviewComment(BaseModel):
    file: str
    category: str
    comment: str
    suggestion: Optional[str] = None
    priority: str = "medium"


class ReviewReport(BaseModel):
    summary: str
    comments: List[ReviewComment]
    security_assessment: str
    performance_notes: str
    maintainability_score: int = Field(..., ge=0, le=10)
    approved: bool
    final_notes: str
