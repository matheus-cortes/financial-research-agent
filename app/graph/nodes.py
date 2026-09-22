import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.func import task

from app.graph.state import ResearchState, RoutingPlan

from app.graph.models import (
    Critique,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    RoutingPlan,
    SpecialistTask,
)

load_dotenv()


llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=1,
    max_tokens=4096,
)


planner_llm = llm.with_structured_output(ResearchPlan)
supervisor_llm = llm.with_structured_output(RoutingPlan)
financial_llm = llm.with_structured_output(ResearchResult)
market_llm = llm.with_structured_output(ResearchResult)
news_llm = llm.with_structured_output(ResearchResult)
risk_llm = llm.with_structured_output(ResearchResult)
critic_llm = llm.with_structured_output(Critique)
critic_llm = llm.with_structured_output(Critique)
replanner_llm = llm.with_structured_output(ResearchPlan)
report_llm = llm.with_structured_output(ResearchReport)


def planner(state: ResearchState):
    ticker = state["ticker"].upper()

    print(f"\n[PLANNER] Creating research plan for {ticker}...")

    plan = planner_llm.invoke(
        f"""
        Create a financial research plan for the stock ticker {ticker}.

        Identify the main areas that should be investigated to produce
        a comprehensive financial research report.

        Each task must contain:
        - area
        - objective
        """
    )

    plan = planner_llm.invoke(
        f"""
        Create a compact financial research plan for the stock ticker {ticker}.

        Create exactly 5 research tasks covering these areas:

        1. Company overview and business model
        2. Market performance
        3. Financial performance
        4. Recent news and events
        5. Risks and risk factors

        Each task must contain:
        - area
        - objective

        Keep the plan focused. Do not create additional tasks.
        """
    )

    print(f"[PLANNER] Research plan created: {len(plan.tasks)} tasks")

    return {
        "ticker": ticker,
        "research_plan": plan,
        "logs": ["Research plan created"],
    }


def supervisor(state: ResearchState):
    plan = state["research_plan"]

    print(f"\n[SUPERVISOR] Routing {len(plan.tasks)} research tasks...")

    routing_plan = supervisor_llm.invoke(
        f"""
        You are the supervisor of a financial research workflow.

        Given the research plan below, decide which specialist agent
        should investigate each task.

        Available specialist agents:
        - market_agent
        - financial_agent
        - news_agent
        - risk_agent

        Research plan:
        {plan.model_dump_json()}
        """
    )

    print(
        f"[SUPERVISOR] Routing completed: "
        f"{len(routing_plan.decisions)} tasks assigned"
    )

    return {
        "routing_plan": routing_plan,
        "logs": ["Research tasks routed"],
    }

def financial_agent(state: dict):
    task: SpecialistTask = state["task"]

    print(f"[FINANCIAL_AGENT] Starting: {task.area}")

    result = financial_llm.invoke(
        f"""
        You are the financial specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Analyze what should be investigated for this task.

        Do not invent financial data.
        Return the research findings as structured information.
        If external financial data is required, indicate what information
        should be obtained from tools or reliable sources.
        """
    )

    print(f"[FINANCIAL_AGENT] Completed: {task.area}")

    return {
        "research_results": [result],
        "logs": [f"Financial research completed: {task.area}"],
    }



def market_agent(state: dict):
    task: SpecialistTask = state["task"]

    print(f"[MARKET_AGENT] Starting: {task.area}")
    result = market_llm.invoke(
        f"""
        You are the market specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Analyze what should be investigated for this market-related task.

        Do not invent market data.
        If external market data is required, indicate what information
        should be obtained from tools or reliable sources.
        """
    )

    print(f"[MARKET_AGENT] Completed: {task.area}")

    return {
        "research_results": [result],
        "logs": [f"Market research completed: {task.area}"],
    }




def news_agent(state: dict):
    task: SpecialistTask = state["task"]

    print(f"[NEWS_AGENT] Starting: {task.area}")

    result = news_llm.invoke(
        f"""
        You are the news specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Analyze what news, events and recent developments should be investigated.

        Do not invent news or events.
        If external information is required, indicate what should be obtained
        from news or company-event tools.
        """
    )

    print(f"[NEWS_AGENT] Completed: {task.area}")

    return {
        "research_results": [result],
        "logs": [f"News research completed: {task.area}"],
    }


def risk_agent(state: dict):
    task: SpecialistTask = state["task"]

    print(f"[RISK_AGENT] Starting: {task.area}")

    result = risk_llm.invoke(
        f"""
        You are the risk specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Analyze what risks and risk-related information should be investigated.

        Do not invent facts.
        If external information is required, indicate what should be obtained
        from appropriate tools or reliable sources.
        """
    )

    print(f"[RISK_AGENT] Completed: {task.area}")

    return {
        "research_results": [result],
        "logs": [f"Risk research completed: {task.area}"],
    }



def critic(state: ResearchState):
    results = state["research_results"]

    print(f"\n[CRITIC] Evaluating {len(results)} research results...")

    critique = critic_llm.invoke(
        f"""
        You are the critic of a financial research workflow.

        Review the research results collected by the specialist agents.

        Determine whether the available research is sufficient to produce
        a comprehensive financial research report.

        Identify:
        - strengths
        - gaps
        - recommendations

        Do not invent financial facts.

        Research results:
        {results}
        """
    )

    print(
        f"[CRITIC] Evaluation completed: "
        f"sufficient={critique.sufficient}"
    )

    return {
        "critique": critique,
        "logs": ["Research results evaluated"],
    }


def replanner(state: ResearchState):
    critique = state["critique"]

    print(
        f"\n[REPLANNER] Creating additional research plan "
        f"for iteration {state['iteration'] + 1}..."
    )

    replanned = replanner_llm.invoke(
        f"""
        You are the replanner of a financial research workflow.

        The research workflow was evaluated by a critic and the research
        was considered insufficient.

        Based on the critic's assessment, create a new research plan
        containing only the additional tasks that are necessary to fill
        the identified gaps.

        Do not repeat research that has already been completed.

        Create at most 5 additional research tasks.

        If the critic identifies more than 5 gaps, prioritize the most
        important ones.

        Critique:
        {critique.model_dump_json()}

        Each task must contain:
        - area
        - objective
        """
    )

    print(f"[REPLANNER] Created {len(replanned.tasks)} additional tasks")

    return {
        "research_plan": replanned,
        "iteration": state["iteration"] + 1,
        "logs": ["Research plan replanned"],
    }


def report_agent(state: ResearchState):
    results = state["research_results"]

    print(
        f"\n[REPORT] Generating final report from "
        f"{len(results)} research results..."
    )

    report = report_llm.invoke(
        f"""
        You are the report generation agent in a financial research workflow.

        Create a comprehensive financial research report based only on the
        collected research results.

        Organize the report into:
        - executive summary
        - company overview
        - market performance
        - financial performance
        - news and events
        - risks
        - conclusion

        Do not invent facts.
        Do not provide buy, sell, or investment recommendations.

        Ticker: {state["ticker"]}

        Research results:
        {results}
        """
    )

    print("[REPORT] Final report generated")

    return {
        "report": report,
        "logs": ["Final research report generated"],
    }