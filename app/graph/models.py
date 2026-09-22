from pydantic import BaseModel


class ResearchTask(BaseModel):
    area: str
    objective: str


class ResearchPlan(BaseModel):
    tasks: list[ResearchTask]


class RoutingDecision(BaseModel):
    area: str
    objective: str
    agent: str


class RoutingPlan(BaseModel):
    decisions: list[RoutingDecision]


class ResearchResult(BaseModel):
    area: str
    agent: str
    findings: list[str]
    sources: list[str]

class SpecialistTask(BaseModel):
    ticker: str
    area: str
    agent: str
    objective: str

class Critique(BaseModel):
    sufficient: bool
    strengths: list[str]
    gaps: list[str]
    recommendations: list[str]


class ResearchReport(BaseModel):
    ticker: str
    executive_summary: str
    company_overview: str
    market_performance: str
    financial_performance: str
    news_and_events: str
    risks: str
    conclusion: str