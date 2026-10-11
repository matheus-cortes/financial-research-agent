from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.graph.models import (
    Critique,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    RoutingPlan,
    SpecialistTask,
)

class ResearchState(TypedDict):
    ticker: str
    messages: Annotated[list[BaseMessage], add_messages]
    logs: Annotated[list[str], add]
    research_plan: ResearchPlan
    routing_plan: RoutingPlan
    research_results: Annotated[list[ResearchResult], add]
    critique: Critique
    iteration: int
    report: ResearchReport
    
class MarketAgentState(TypedDict):
    ticker: str
    task: SpecialistTask
    messages: Annotated[list[BaseMessage], add_messages]
    result: ResearchResult | None

class NewsAgentState(TypedDict):
    ticker: str
    task: SpecialistTask
    messages: Annotated[list[BaseMessage], add_messages]
    result: ResearchResult | None