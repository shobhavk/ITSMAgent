from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

CATEGORIES: list[str] = [
    "Hardware issues",
    "Software issues",
    "Network issues",
    "Database issues",
    "Security incidents",
    "Server deployment",
    "Server configuration change",
    "Performance issue",
    "Disk/file system extension",
    "Backup related",
    "CCIR related",
    "File system cleanup",
    "Virtualisation/cloud platform issues",
    "OS upgrade/service pack upgrade",
    "Server migration",
    "Uncategorized",  # safety-net bucket, never force a bad match
]


class TicketRecord(BaseModel):
    """Normalized representation of one incident, regardless of source format."""

    ticket_id: str
    short_description: str = ""
    description: str = ""
    worklog: str = ""
    priority: Optional[str] = None
    status: Optional[str] = None
    assignment_group: Optional[str] = None
    configuration_item: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    # created_at/resolved_at are distinct from opened_at/closed_at when a
    # source system exports all four (e.g. ServiceNow's sys_created_on +
    # opened_at + resolved_at + closed_at). Kept separate rather than
    # aliased onto opened_at/closed_at - collapsing "Created"+"Opened" (or
    # "Resolved"+"Closed") onto one column name produced duplicate-named
    # columns and silently broke date parsing when a dataset had all four.
    # See trend_metrics.effective_open_resolve_times for how these
    # combine into one consistent start/end pair.
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    # Optional - only populated if the source data has a matching column
    # (see COLUMN_ALIASES in data_ingestion.py). Most ITSM exports don't
    # carry a true detection timestamp; responded_at (first response /
    # acknowledgment) is more commonly present. Resolution-metric
    # calculations in trend_metrics.py report "not available" rather than
    # guessing when these are absent - see that module's docstring.
    responded_at: Optional[datetime] = None
    detected_at: Optional[datetime] = None
    source_row: int = Field(description="Original row index for traceability")

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _stringify_id(cls, v):
        return str(v).strip() if v is not None else ""


class CategoryResult(BaseModel):
    ticket_id: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str  # "keyword_rule" | "embedding" | "llm" | "fallback"


class WorklogScore(BaseModel):
    ticket_id: str
    score: int = Field(ge=0, le=100)
    breakdown: dict[str, float]
    flags: list[str] = []


class AnalyzedTicket(BaseModel):
    ticket_id: str
    short_description: str
    description: str
    worklog: str
    priority: Optional[str]
    status: Optional[str]
    assignment_group: Optional[str]
    configuration_item: Optional[str] = None
    host: Optional[str] = None
    category: str
    category_confidence: float
    category_method: str
    worklog_score: int
    worklog_flags: list[str]
    validation_flags: list[str] = []
    # Carried through from TicketRecord (previously dropped here - trend/
    # resolution-metric analysis needs them). See TicketRecord for notes
    # on created_at/resolved_at vs opened_at/closed_at, and on
    # responded_at/detected_at's availability.
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    detected_at: Optional[datetime] = None


class AnalysisResponse(BaseModel):
    total_records: int
    valid_records: int
    rejected_records: int
    category_counts: dict[str, int]
    host_counts: dict[str, int] = {}
    average_worklog_score: float
    results: list[AnalyzedTicket]


class ChatRequest(BaseModel):
    """A management question about the most recently analyzed batch for
    this API key. `history` is the prior (question, answer) turns of this
    conversation, oldest first, for multi-turn context."""

    question: str
    history: list[tuple[str, str]] = []


class ChatResponse(BaseModel):
    answer: str
