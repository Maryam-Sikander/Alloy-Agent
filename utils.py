from typing import Any, Dict, Literal, Optional
from langgraph.checkpoint.postgres import PostgresSaver
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


class PostgresSaverCustom(PostgresSaver):
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetch a checkpoint tuple using the given configuration.

        Args:
            config (RunnableConfig): Configuration specifying which checkpoint to retrieve.

        Returns:
            Optional[CheckpointTuple]: The requested checkpoint tuple, or None if not found.
        """
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints that match the given criteria.

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): List checkpoints created before this configuration.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Returns:
            AsyncIterator[CheckpointTuple]: Async iterator of matching checkpoint tuples.


        """
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config (RunnableConfig): Configuration for the checkpoint.
            checkpoint (Checkpoint): The checkpoint to store.
            metadata (CheckpointMetadata): Additional metadata for the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        return self.put_writes(config, writes, task_id, task_path)


# NOTE: UNUSED function
def query_in_messages(query: str, name: str, messages: list[BaseMessage]) -> bool:
    """
    Checks if the query already exists in the message content and if the name is the same.

    Args:
    query (str): The query to search for in the message content.
    name (str): The name to verify in the message.
    messages (list[BaseMessage]): The list of messages
    Returns:
    bool: True if the query is found twice to already exist in the message content and the name is the same, False otherwise.
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
            # We want our workers to ALWAYS "report back" to the supervisor when done
            goto=supervisor_node_name,
        )

    return calendar_node


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
