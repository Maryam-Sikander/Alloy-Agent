import asyncio
import os
import sqlite3
from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict
from psycopg_pool import ConnectionPool
from agent_workflow.database import PostgresSaverCustom
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from config.config import Config

from agent_workflow.date_worker import calculate_date
from agent_workflow.calendar_workers import calendar_workers_dict
from agent_workflow.email_workers import email_workers_dict
from agent_workflow.prompts import (
    ENTRY_PROMPT_ORCHESTRATOR,
    RESPONSE_PROMPT_ORCHESTRATOR,
    feedback_email_manager_prompt_template,
    feedback_calendar_manager_prompt_template,
    CALENDAR_MANAGER_END_PROMPT,
    EMAIL_MANAGER_END_PROMPT,
    CALENDAR_MANAGER_SYSTEM_PROMPT,
    EMAIL_MANAGER_SYSTEM_PROMPT,
)
from agent_workflow.schemas import (
    orchestrator_outputs_tuple,
    OrchestratorRouterList,
    CalendarRouterList,
    EmailRouterList,
    OrchestratorRouter,
)

# Load env
_ = load_dotenv(find_dotenv())

ENTRY_POINT_TEMPLATE = """
### The user's request is:
{user_request}"""

FINAL_ANSWER_TEMPLATE = """
### The user's request is:
{user_request}

---

### The "MANAGER OUTPUTS":
{manager_response}
"""

MANAGER_TEMPLATE = """
### The user's request is:
{user_request}

---

### Task Context:
{manager_response_context}
"""

config = Config()

class GraphState(TypedDict):
    """The state of the supervisor agents."""

    user_input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # we'll to manage how update these messages
    supervisors_messages: list[BaseMessage]
    manager_response: list[dict]
    manager_list: list[OrchestratorRouter]


llm_orchestrator =ChatOpenAI(model=config.get("configurable", "llm-model"),
                              temperature=config.get("configurable", "llm-temperature"),
                             base_url="https://api.aimlapi.com/v1")
llm = ChatOpenAI(model=config.get("configurable", "llm-model"),
                              temperature=config.get("configurable", "llm-temperature"),
                             base_url="https://api.aimlapi.com/v1")

trimmer = trim_messages(
    max_tokens=7,  # to keep the last 3 interactions messages
    strategy="last",
    token_counter=len,
    include_system=True,
    start_on="human",
    end_on=("human",),
)


async def orchestrator_input_node(
    state: GraphState,
) -> Command[Literal["orchestrator"]]:
    """An orchestrator node. Entry point of the graph."""

    # we must create the user message
    if len(state["messages"]) == 0 or state["messages"][-1].type == "ai":
        state["messages"].append(
            HumanMessage(
                content=ENTRY_POINT_TEMPLATE.format(
                    **{"user_request": state["user_input"]}
                )
            )
        )
    messages = [SystemMessage(content=ENTRY_PROMPT_ORCHESTRATOR)] + trimmer.invoke(
        state["messages"]
    )

    response: OrchestratorRouterList = llm_orchestrator.with_structured_output(
        OrchestratorRouterList
    ).invoke(messages)
    return Command(
        goto="orchestrator",
        update={"manager_list": response.managers, "manager_response": []},
    )


async def orchestrator_output_node(
    state: GraphState,
) -> Command[Literal[END]]:
    """An orchestrator node. Output point of the graph."""

    # update the user message with the manager response
    state["messages"][-1] = HumanMessage(
        content=FINAL_ANSWER_TEMPLATE.format(
            **{
                "user_request": state["user_input"],
                "manager_response": state.get("manager_response") or "NULL",
            }
        )
    )

    # timmer here, to use just the last 3 user interactions
    messages = [SystemMessage(content=RESPONSE_PROMPT_ORCHESTRATOR)] + trimmer.invoke(
        state["messages"]
    )
    ai_response = llm.invoke(messages)

    return Command(
        goto=END,
        update={
            "messages": [ai_response],
        },
    )


async def orchestrator_node(
    state: GraphState,
) -> Command[Literal[orchestrator_outputs_tuple + ("orchestrator_output",)]]:
    """An orchestrator node. Entry and exit point of the graph."""

    # no more managers to route
    if state["manager_list"] == []:
        return Command(goto="orchestrator_output")
    manager_list = state.get("manager_list")
    manager_response = state.get("manager_response", [])

    # if data_manager failed: return to make the user's answer
    if (
        len(manager_response) > 0
        and "date_manage" in manager_response[-1].get("route_manager")
        and "error" in manager_response[-1].get("answer").lower()
    ):
        return Command(goto="orchestrator_output")

    orchestrator_router = manager_list.pop(0)

    return Command(
        goto=orchestrator_router.route_manager,
        update={
            "manager_list": manager_list,
            "manager_response": manager_response + [dict(orchestrator_router)],
            # reset the supervisors messages
            "supervisors_messages": [
                HumanMessage(
                    content=MANAGER_TEMPLATE.format(
                        user_request=orchestrator_router.query,
                        manager_response_context=manager_response or "NULL",
                    )
                )
            ],
        },
    )


def date_manage_node(state: GraphState) -> Command[Literal["orchestrator"]]:
    """Date range extract in `Asia/Karachi` timezone"""
    manager_response = state["manager_response"]

    date_range_str = calculate_date(state["supervisors_messages"][-1].content)
    manager_response[-1]["answer"] = date_range_str

    return Command(
        goto="orchestrator",
        update={
            "supervisors_messages": [],
            "manager_response": manager_response,
        },
    )


async def execute_workers(data: CalendarRouterList, workers_dict):
    """Executes worker tasks asynchronously.

    Args:
        data (ManagerRouterList): An object containing a list of tasks for workers.
        workers_dict (dict): A dictionary containing worker names as keys and worker objects as values.

    Returns:
        list: A list of results from the executed calendar worker tasks.
    """
    tasks = [
        workers_dict[worker.name].ainvoke(
            {
                "workers_messages": HumanMessage(
                    content=worker.task,
                )
            }
        )
        for worker in data.workers
    ]

    results = await asyncio.gather(*tasks)
    return results


async def calendar_manage_node(
    state: GraphState,
) -> Command[Literal["feedback_synthesizer"]]:
    """An LLM-based router."""

    supervisors_messages = state["supervisors_messages"]
    messages = [
        SystemMessage(content=CALENDAR_MANAGER_SYSTEM_PROMPT)
    ] + supervisors_messages

    response: CalendarRouterList = llm.with_structured_output(
        CalendarRouterList
    ).invoke(messages)

    # in this case the manager must elaborate an answer
    # to avoid go to the feedback synthesizer with an empty ai asnwers
    if response.workers == []:
        ai_manager_answer = llm.invoke(
            [SystemMessage(content=CALENDAR_MANAGER_END_PROMPT)] + supervisors_messages
        )
        # ensure the ai answer is in the 3rd position
        supervisors_messages += supervisors_messages + [ai_manager_answer]
        return Command(
            goto="feedback_synthesizer",
            update={"supervisors_messages": supervisors_messages},
        )

    results = await execute_workers(response, calendar_workers_dict)
    for i, result in enumerate(results):
        worker = response.workers[i]
        supervisors_messages += [
            AIMessage(content=worker.task, name=worker.name),
        ]
        supervisors_messages += [
            AIMessage(content=result["workers_messages"][-1].content, name=worker.name)
        ]

    return Command(
        goto="feedback_synthesizer",
        update={"supervisors_messages": supervisors_messages},
    )


async def email_manage_node(
    state: GraphState,
) -> Command[Literal["feedback_synthesizer"]]:
    """An LLM-based router."""

    supervisors_messages = state["supervisors_messages"]
    messages = [
        SystemMessage(content=EMAIL_MANAGER_SYSTEM_PROMPT)
    ] + supervisors_messages

    response: EmailRouterList = llm.with_structured_output(EmailRouterList).invoke(
        messages
    )

    # in this case the manager must elaborate an answer
    # to avoid go to the feedback synthesizer with an empty ai asnwers
    if response.workers == []:
        ai_manager_answer = llm.invoke(
            [SystemMessage(content=EMAIL_MANAGER_END_PROMPT)] + supervisors_messages
        )
        # ensure the ai answer is in the 3rd position
        supervisors_messages += supervisors_messages + [ai_manager_answer]
        return Command(
            goto="feedback_synthesizer",
            update={"supervisors_messages": supervisors_messages},
        )

    results = await execute_workers(response, email_workers_dict)
    for i, result in enumerate(results):
        worker = response.workers[i]
        supervisors_messages += [
            AIMessage(content=worker.task, name=worker.name),
        ]
        supervisors_messages += [
            AIMessage(content=result["workers_messages"][-1].content, name=worker.name)
        ]

    return Command(
        goto="feedback_synthesizer",
        update={"supervisors_messages": supervisors_messages},
    )


def feedback_synthesizer_node(state: GraphState) -> Command[Literal["orchestrator"]]:
    """Synthesizes feedback and return to the orchestrator."""

    # state["supervisors_messages"] contains the query from the orchestrator
    orchestrator_query = state["supervisors_messages"][0].content
    manager_response = state["manager_response"]
    agents_chat_history = ""

    # we just need the agents messages to be able to synthesize the feedback
    # in 0 is the orchestrator query, in 1,3,5...(odd) is the supervisor query
    for i, mess in enumerate(state["supervisors_messages"][2::2]):
        agents_chat_history += f" ### Message {i+1} - Agent {mess.name}:\n"
        agents_chat_history += f"```{mess.content}```\n\n"

    agents_chat_history = agents_chat_history[:-2]  # remove the last \n\n

    if manager_response[-1]["route_manager"] == "calendar_manage":
        feedback_prompt_template = feedback_calendar_manager_prompt_template
    elif manager_response[-1]["route_manager"] == "email_manage":
        feedback_prompt_template = feedback_email_manager_prompt_template
    ai_response = llm.invoke(
        input=feedback_prompt_template.invoke(
            {"query": orchestrator_query, "agents_chat_history": agents_chat_history}
        ).text
    )

    # set the manager answer to the last orchestrator query
    manager_response[-1]["answer"] = ai_response.content

    return Command(
        goto="orchestrator",
        update={
            "supervisors_messages": [],
            "manager_response": manager_response,
        },
    )


# build the graph
orchestrator_builder = StateGraph(GraphState)


orchestrator_builder.add_node("orchestrator_input", orchestrator_input_node)
orchestrator_builder.add_node("orchestrator_output", orchestrator_output_node)
orchestrator_builder.add_node("orchestrator", orchestrator_node)
orchestrator_builder.add_node("date_manage", date_manage_node)
orchestrator_builder.add_node("calendar_manage", calendar_manage_node)
orchestrator_builder.add_node("email_manage", email_manage_node)
orchestrator_builder.add_node("feedback_synthesizer", feedback_synthesizer_node)
orchestrator_builder.add_edge(START, "orchestrator_input")



async def init_orchestrator():
    """Initialize the orchestrator graph with Postgres if available, else SQLite."""
    db_uri = os.getenv("POSTGRES_DB_URI")

    if db_uri:
        # Prefer PostgreSQL
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        pool = ConnectionPool(
            conninfo=db_uri,
            max_size=20,
            kwargs=connection_kwargs,
        )
        checkpointer = PostgresSaverCustom(pool)
        checkpointer.setup()  # Postgres saver uses sync setup
        orchestrator_graph = orchestrator_builder.compile(checkpointer=checkpointer)
        return orchestrator_graph

    # Fallback to SQLite (async)
    conn = await aiosqlite.connect("checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()
    orchestrator_graph = orchestrator_builder.compile(checkpointer=checkpointer)

    return orchestrator_graph