from typing import Any, Dict, Literal, Optional
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from collections.abc import AsyncIterator, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command


# Postgres Saver (async wrapper)

class PostgresSaverCustom(PostgresSaver):
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)


# SQLite Saver (async wrapper)

class SqliteSaverCustom(SqliteSaver):
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)


# -----------------------------
# Misc Helpers
# -----------------------------
def query_in_messages(query: str, name: str, messages: list[BaseMessage]) -> bool:
    """
    Checks if the query already exists in the message content and if the name is the same.
    Returns True if the query is found twice with same name.
    """
    count = 0
    for message in messages:
        if query in message.content and name == message.name:
            count += 1
    return count > 1


def create_agent_calendar_node(
    reAct_agent,
    node_name: str,
    supervisor_node_name,
):
    """Create a node for a calendar worker agent."""

    def calendar_node(state) -> Command[Literal[supervisor_node_name]]:
        result = reAct_agent.invoke(
            {
                "workers_messages": HumanMessage(
                    content=state["supervisors_messages"][-1].content
                )
            }
        )

        return Command(
            update={
                "supervisors_messages": state["supervisors_messages"]
                + [
                    AIMessage(
                        content=result["workers_messages"][-1].content, name=node_name
                    )
                ],
            },
            # Workers always report back to supervisor
            goto=supervisor_node_name,
        )

    return calendar_node


# Feedback Prompt
FEEDBACK_GENERIC_PROMPT = """You have the following conversation history of multiple AI-generated messages:
----

{agents_chat_history}

-----
Using all the information provided by these agents, produce a single, concise response to the query below:

Query: "{query}"

Instructions:
1. Do not repeat any unnecessary information.
2. Focus on the key points from each message, but retain all relevant details such as links, titles, email addresses, or calendars used. These details must be included **exactly as they were provided** by the agents, without any modification or translation.
3. Maintain all titles, subtitles, and specific details (such as event names, times, dates, and links) **in their original language and format**.
4. When appropriate, retain the reference to the entity responsible for the task without explicitly using the word 'agent'.
5. Your output must be a single message that merges all pertinent information from the conversation.
"""
