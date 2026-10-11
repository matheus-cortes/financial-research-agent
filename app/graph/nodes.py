import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.func import task

from app.graph.state import MarketAgentState, ResearchState

from app.graph.models import (
    Critique,
    ResearchPlan,
    ResearchReport,
    ResearchResult,
    RoutingPlan,
    SpecialistTask,
)

from app.tools.market import (
    get_historical_prices,
    get_stock_price,
)

load_dotenv()


llm = ChatOpenAI(
    model="moonshotai/kimi-k3",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=1,
    max_tokens=4096,
)


planner_llm = llm.with_structured_output(
    ResearchPlan,
    method="json_mode",
)

supervisor_llm = llm.with_structured_output(
    RoutingPlan,
    method="json_mode",
)

financial_llm = llm.with_structured_output(
    ResearchResult,
    method="json_mode",
)

market_llm = llm.with_structured_output(
    ResearchResult,
    method="json_mode",
)

news_llm = llm.with_structured_output(
    ResearchResult,
    method="json_mode",
)

risk_llm = llm.with_structured_output(
    ResearchResult,
    method="json_mode",
)

critic_llm = llm.with_structured_output(
    Critique,
    method="json_mode",
)

replanner_llm = llm.with_structured_output(
    ResearchPlan,
    method="json_mode",
)

report_llm = llm.with_structured_output(
    ResearchReport,
    method="json_mode",
)


market_tools = [
    get_stock_price,
    get_historical_prices,
]

market_tool_llm = llm.bind_tools(market_tools)
market_tool_node = ToolNode(market_tools)


def planner(state: ResearchState):
    ticker = state["ticker"].upper()

    print(f"\n[PLANNER] Creating research plan for {ticker}...")

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

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "tasks": [
                {{
                    "area": "...",
                    "objective": "..."
                }}
            ]
        }}
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

        Given the research plan below, assign each research task to exactly
        one specialist agent.

        Available specialist agents:

        - market_agent
          Handles stock prices, historical performance, trading volume,
          valuation, dividends, and market-related analysis.

        - financial_agent
          Handles company fundamentals, financial statements, revenue,
          profitability, margins, debt, leverage, cash flow, and financial
          performance.

        - news_agent
          Handles recent news, events, announcements, regulatory events,
          earnings news, and other relevant developments.

        - risk_agent
          Handles regulatory, operational, financial, governance, and
          other company-specific risks.

        Research plan:
        {plan.model_dump_json()}

        Every research task must be assigned to exactly one agent.

        Do not create additional tasks.
        Do not remove any tasks.

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "decisions": [
                {{
                    "area": "...",
                    "objective": "...",
                    "agent": "market_agent"
                }}
            ]
        }}
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

        If external financial data is required, clearly identify the
        information that should be obtained from financial data tools
        or reliable sources.

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "area": "...",
            "agent": "financial_agent",
            "findings": [
                "..."
            ],
            "sources": [
                "..."
            ]
        }}
        """
    )

    print(f"[FINANCIAL_AGENT] Completed: {task.area}")

    return {
        "research_results": [result],
        "logs": [f"Financial research completed: {task.area}"],
    }


# -------------------------------------------------------------------
# MARKET AGENT - TOOL CALLING
# -------------------------------------------------------------------


def market_agent_start(state: MarketAgentState):
    task: SpecialistTask = state["task"]

    print(f"[MARKET_AGENT] Starting: {task.area}")

    message = HumanMessage(
        content=f"""
        You are the market specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Investigate the objective using the available market tools when
        external market data is required.

        Available tools:
        - get_stock_price
        - get_historical_prices

        Do not invent market data.

        Use the tools when they provide information necessary to investigate
        the objective.
        """
    )

    return {
        "messages": [message],
    }


def market_agent_llm_call(state: MarketAgentState):
    print("[MARKET_AGENT] Calling LLM...")

    response = market_tool_llm.invoke(state["messages"])

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"[MARKET_AGENT] Tool requested: "
                f"{tool_call['name']}"
            )

    else:
        print("[MARKET_AGENT] LLM did not request any tools")

    return {
        "messages": [response],
    }


def route_market_tools(state: MarketAgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "market_tool_node"

    return "market_finalize"


def market_finalize(state: MarketAgentState):
    task: SpecialistTask = state["task"]

    print("[MARKET_AGENT] Generating final structured result...")

    result = market_llm.invoke(
        f"""
        You are the market specialist agent in a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Based on the investigation performed by the agent and the tool
        results contained in the messages below, produce the final
        structured research result.

        Do not invent market data.

        If the available tools did not provide enough information to fully
        answer the objective, clearly state that limitation in the findings.

        Investigation messages:
        {state["messages"]}

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "area": "...",
            "agent": "market_agent",
            "findings": [
                "..."
            ],
            "sources": [
                "..."
            ]
        }}
        """
    )

    print(f"[MARKET_AGENT] Completed: {task.area}")

    return {
        "result": result,
    }


def news_agent(state: dict):
    task: SpecialistTask = state["task"]

    print(f"[NEWS_AGENT] Starting: {task.area}")

    result = news_llm.invoke(
        f"""
        You are the news specialist agent for a financial research workflow.

        Ticker: {task.ticker}
        Research area: {task.area}
        Objective: {task.objective}

        Identify the news and events that should be investigated for this task.

        Do not invent facts or news.
        If external information is required, clearly state what information
        should be obtained from reliable sources.

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "area": "{task.area}",
            "agent": "news_agent",
            "findings": [
                "..."
            ],
            "sources": [
                "..."
            ]
        }}
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

        If external information is required, clearly identify the
        information that should be obtained from appropriate tools
        or reliable sources.

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "area": "...",
            "agent": "risk_agent",
            "findings": [
                "..."
            ],
            "sources": [
                "..."
            ]
        }}
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

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "sufficient": true,
            "strengths": [
                "..."
            ],
            "gaps": [
                "..."
            ],
            "recommendations": [
                "..."
            ]
        }}
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

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "tasks": [
                {{
                    "area": "...",
                    "objective": "..."
                }}
            ]
        }}
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

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "ticker": "...",
            "executive_summary": "...",
            "company_overview": "...",
            "market_performance": "...",
            "financial_performance": "...",
            "news_and_events": "...",
            "risks": "...",
            "conclusion": "..."
        }}
        """
    )

    print("[REPORT] Final report generated")

    return {
        "report": report,
        "logs": ["Final research report generated"],
    }
