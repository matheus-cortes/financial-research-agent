from dotenv import load_dotenv
import os

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


from app.graph.nodes import (
    critic,
    financial_agent,
    market_agent,
    news_agent,
    planner,
    replanner,
    report_agent,
    risk_agent,
    supervisor,
)

from app.graph.state import ResearchState
from app.graph.models import SpecialistTask

load_dotenv()
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 2))


def route_tasks(state: ResearchState):
    return [
        Send(
            decision.agent,
            {
                "ticker": state["ticker"],
                "task": SpecialistTask(
                    ticker=state["ticker"],
                    area=decision.area,
                    objective=decision.objective,
                    agent=decision.agent,
                ),
            },
        )
        for decision in state["routing_plan"].decisions
    ]


def route_after_critic(state: ResearchState):
    if state["critique"].sufficient:
        print("[CRITIC] Research is sufficient. Proceeding to report.")
        return "report_agent"

    if state["iteration"] >= MAX_ITERATIONS:
        print(
            "[CRITIC] Maximum iterations reached. "
            "Proceeding to report with identified gaps."
        )
        return "report_agent"

    print("[CRITIC] Research is insufficient. Triggering replanner.")
    return "replanner"


builder = StateGraph(ResearchState)

builder.add_node("planner", planner)
builder.add_node("supervisor", supervisor)

builder.add_node("financial_agent", financial_agent)
builder.add_node("market_agent", market_agent)
builder.add_node("news_agent", news_agent)
builder.add_node("risk_agent", risk_agent)

builder.add_node("critic", critic)
builder.add_node("replanner", replanner)
builder.add_node("report_agent", report_agent)


builder.add_edge(START, "planner")
builder.add_edge("planner", "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_tasks,
)

builder.add_edge("financial_agent", "critic")
builder.add_edge("market_agent", "critic")
builder.add_edge("news_agent", "critic")
builder.add_edge("risk_agent", "critic")

builder.add_conditional_edges(
    "critic",
    route_after_critic,
)

builder.add_edge("replanner", "supervisor")
builder.add_edge("report_agent", END)

graph = builder.compile()