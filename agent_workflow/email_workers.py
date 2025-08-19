from langchain_core.tools import StructuredTool
import os
from langchain_core.messages import BaseMessage, AnyMessage
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from config.config import Config
from composio_langchain import ComposioToolSet
from typing import Annotated, Any, Literal, Sequence
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

_ = load_dotenv(find_dotenv())

EMAIL_WORKER_TEMPLATE = """
You excel at managing emails efficiently using available tools while adhering to structured email management practices.

## **Email Management Guidelines**


### **Sending Emails (GMAIL_SEND_EMAIL, GMAIL_CREATE_EMAIL_DRAFT)**
1. **Validation**
   - Ensure `recipient_email` is provided. If missing, notify the user and take no further action.
   - In case `recipient_email` is missing when **drafting** an email, create a generic example as the recipient.
   - If the subject is missing, generate one based on the email content.

2. **Formatting**
   - Maintain proper grammar and a professional tone.
   - Summarize content clearly if needed.
   - Ensure the email text is in HTML format using he same language as the user's request.

3. **Confirmation**
   - Confirm successful sending with recipient(s), subject, and timestamp.

### **Fetching Emails (GMAIL_FETCH_EMAILS, GMAIL_LIST_THREADS, GMAIL_FETCH_MESSAGE_BY_THREAD_ID)**
1. **Search Criteria**
   - Emails can be retrieved based on:
     - Keywords found in the subject or body of the email (e.g., "invoice due", "meeting agenda", "project update").
     - Sender or recipient email addresses (e.g., "john.doe@example.com", "client@company.com").
     - Emails can be filtered using their `label_ids`, including: `INBOX`, `SENT`, `DRAFT`, `SPAM`, `TRASH`, `UNREAD`, `STARRED`, `IMPORTANT`, `CATEGORY_PERSONAL`, `CATEGORY_SOCIAL`, `CATEGORY_PROMOTIONS`, `CATEGORY_UPDATES`, `CATEGORY_FORUMS`.
   - **Date Restriction:**
     - Do not use dates as a filter in the query when fetching emails.
     - Instead, retrieve the most recent emails first.
     - If the user's requests emails from a specific date, filter the retrieved emails based on that date.
   - **Limitation Notice:**
     - The system can only retrieve emails in chronological order, meaning it cannot fetch older emails without also returning the most recent ones.
     - To obtain specific older emails, the user must use the `query` parameter with relevant filtering information (e.g., subject keywords or sender details) or filter using `label_ids`. These are the only two options available for fetching emails. Otherwise, the system will only return the most recent emails.

2. **Summarization**
   - Provide a concise summary including:
     - Sender
     - Subject
     - Date (formatted as specified below)
     - Key points (if applicable)

3. **Error Handling**
   - If an error like the following occurs:
     ```
     error: "1 validation error for MessageBody
     Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]"
     ```
     it means the user has one or more emails without a sender.
     **Notification to the user:**
     "You have emails without a sender. I am unable to process, manage, or manipulate emails that do not have a sender. Please check your inbox/draft and review those emails manually."


### **Handling Conversations in Email (GMAIL_LIST_THREADS, GMAIL_FETCH_MESSAGE_BY_THREAD_ID, GMAIL_REPLY_TO_THREAD)**
1. **Thread Handling**
   - A thread represents a conversation or a group of related emails.
   - When retrieving conversations, return all matching threads along with their `thread_id` without using the `GMAIL_FETCH_MESSAGE_BY_THREAD_ID` action. This allows the user to decide which thread to manage.
   - If a specific `thread_id` is requested, ensure that the full thread context and all messages are retrieved before responding.

2. **Replying to Emails**
   - Identify the email to reply to and retrieve its `thread_id`.
   - Use the `GMAIL_REPLY_TO_THREAD` action with the following:
     - `thread_id` of the email.
     - `Body` of the reply message.
     - **Email address of the recipient:**
       - If the user provides a recipient email, use it.
       - **If the user does not specify a recipient, use the sender's email from the original email.**
   - Ensure `body` maintains the same **Formatting** when **Sending Emails** action.
   - Confirm reply success with details.

### **Final User's Answer Format**
- When providing an answer to the user, always format dates as follows:
  **Month day, year, hour in 24-hour format**.
  - Example: `February 21, 2025, 11:00`
- Include in the user's answer the `thread_id` and labels of the each email.
- It is mandatory to explicitly state in the user's answer  that the email information is **based on {email_info}**
"""


class WorkersState(TypedDict):
    """The state of the worker agents."""

    workers_messages: Annotated[Sequence[BaseMessage], add_messages]


def tools_condition_worker(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "workers_messages",
) -> Literal["tools", "__end__"]:
    """Wrapper to use a message_key different from `message`
    in the pre-built `tools_condition` function.
    """
    return tools_condition(state, messages_key)


composio_toolset = ComposioToolSet()
config = Config()
llm = ChatOpenAI(model=config.get("configurable", "llm-model"),
                              temperature=config.get("configurable", "llm-temperature"),
                             base_url="https://api.aimlapi.com/v1")

def wrapper_funct_fetch_emails(tool: StructuredTool):
    """
    Wraps a StructuredTool function to modify its output.

    Removes the 'payload', 'preview' keys from each message to prevent base64-encoded
    images from exceeding the LLM's token limit.
    """
    original_func = tool.func

    def custom_tool_function(**kwargs):
        """
        Calls the original function and removes 'payload' and 'preview' from messages.
        """
        data = original_func(**kwargs)
        try:
            for message in data.get("data", {}).get("messages", []):
                message.pop("payload", None)  # Remove payload if exists
                message.pop("preview", None)  # Remove preview if exists
            # sort messages ascending by date
            # data["data"]["messages"] = list(data["data"]["messages"])[::-1]
        except Exception:
            pass  # Ignore errors if data format is unexpected

        return data

    return custom_tool_function


def build_email_react_agent(email_info, composio_entity_id):
    email_worker_system_prompt_template = EMAIL_WORKER_TEMPLATE.format(
        email_info=email_info
    )

    email_tools = composio_toolset.get_tools(
        actions=[
            "GMAIL_SEND_EMAIL",
            "GMAIL_FETCH_EMAILS",
            "GMAIL_LIST_THREADS",
            "GMAIL_FETCH_MESSAGE_BY_THREAD_ID",
            "GMAIL_REPLY_TO_THREAD",
            "GMAIL_CREATE_EMAIL_DRAFT",
        ],
        entity_id=composio_entity_id,
    )
    # Wrap the fetch_emails tools to remove the 'payload' key from messages
    for tool in email_tools:
        if "FETCH_EMAILS" in tool.name:
            tool.func = wrapper_funct_fetch_emails(tool)

    email_worker_builder = StateGraph(WorkersState)
    gpt_llm_with_email_tools = llm.bind_tools(email_tools)

    async def llm_with_email_tools(state: WorkersState):
        if state["workers_messages"][0].type != "system":
            state["workers_messages"].insert(
                0, SystemMessage(content=email_worker_system_prompt_template)
            )
        response = await gpt_llm_with_email_tools.ainvoke(state["workers_messages"])
        return {"workers_messages": [response]}

    email_worker_builder.add_node("llm_with_email_tools", llm_with_email_tools)
    tool_node = ToolNode(tools=email_tools, messages_key="workers_messages")
    email_worker_builder.add_node("email_tools", tool_node)

    email_worker_builder.add_conditional_edges(
        "llm_with_email_tools",
        tools_condition_worker,
        {"tools": "email_tools", END: END},
    )
    email_worker_builder.add_edge("email_tools", "llm_with_email_tools")
    email_worker_builder.add_edge(START, "llm_with_email_tools")

    return email_worker_builder.compile()


email_workers_dict = {
    "personal_email": build_email_react_agent(
        email_info="Personal Gmail",
        composio_entity_id="personal",
    ),
    "work_email": build_email_react_agent(
        email_info="Work Gmail",
        composio_entity_id="work",
    ),
}
email_workers_info_dict = {
    "personal_email": "Manages all personal emails.",
    "work_email": "Manages all work-related emails.",
}
email_worker_summary_list = ["Personal Google Gmail", "Work Gmail"]
