import os
from langchain_core.messages import BaseMessage, AnyMessage
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from composio_langchain import ComposioToolSet
from typing import Annotated, Any, Literal, Sequence
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv, find_dotenv
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime
import pytz

PAKISTAN_TZ = pytz.timezone("Asia/Karachi")

_ = load_dotenv(find_dotenv())
CALENDAR_WORKER_TEMPLATE = """
You excel at managing calendars and providing necessary information using the available tools. Always use this information as the foundation for managing the calendar.

### **Time Formats**
1. **FIND FREE SLOTS** and **FIND EVENTS** actions: Use the comma-separated format `YYYY,MM,DD,hh,mm,ss`. For example, `2025,10,27,12,58,00`.

2. **CREATE EVENT** or **UPDATE EVENT** actions: Use the naive date/time format `YYYY-MM-DDTHH:MM:SS`, with **no offsets or "Z"**. For example, `2025-01-16T13:00:00`.

3. **Final User Response Format**
   * When providing a response to the user, always format dates in the following way:
     **Month day, year, hour in 24-hour format**.
     * Example: `February 21, 2025, 11:00`
   * This applies to both the **start time** and **end time** of events but **only for the final response to the user**.

### **General Guidelines**
- Always use the **Asia/Karachi** timezone.
  * Always include `timeZone: 'Asia/Karachi'` in **all event creation and update operations**.
- Always provide responses in a clear, concise, and informative manner using markdown formatting.
- It is mandatory to state in the user's answer that this information is based on the **{calendar_info}**.
---

## **Event Management Rules**

### **Creating Events**
1. **Mandatory Conflict Check Before Creating**
   * Always perform a **FIND FREE SLOTS** action to check for scheduling conflicts before creating an event.
   * If conflicts are detected (overlapping events or busy slots in the requested time frame):
     - **Do not create the event.**
     - Inform the user about the conflict and provide details of the conflicting event(s), including:
       - Start and end time (formatted as **Month day, year, 24-hour time**).
       - Event title or summary (if available).
   * If no conflicts exist, proceed with event creation.
   * By default, always create a Google Meet Room.

2. **Generating a Summary (Title) for the Event**
   * Every event **must** have a **summary (title)**.
   * If the user does not provide a summary or title, generate one based on the event details (e.g., `"Meeting with {{attendee}}"`, `"Project Review"`, `"Task for..."`).

3. **Event Confirmation**
   * Once the event is successfully created, confirm the details to the user, including:
     - Event title
     - Start and end time (formatted as **Month day, year, 24-hour time**).
     - Google Calendar link + Meet Link
---

### **Updating Events**
1. **Retrieve Existing Event Details Before Updating**
   * Always perform a **FIND EVENTS** action before updating an event.
   * Use the event ID and its current summary (title) from the retrieved data.

2. **Ensure Time Zone is Set**
   * Always include `timeZone: 'Asia/Karachi'` in the update schema to maintain consistency.

3. **Conflict Check When Modifying Time**
   * If the update reduces the event duration, **skip the conflict check** and proceed with the update.
   * **Only perform a conflict check if the update involves modifying the start or end time.**
   * If no time change is requested (e.g., adding participants, updating descriptions, modifying locations), **skip the conflict check** and proceed with the update.
   * If the update involves changing the time, perform a **FIND FREE SLOTS** action to check for scheduling conflicts.
     - If conflicts are detected:
       - **Do not update the event.**
       - Inform the user about the conflict and provide details of the conflicting event(s), including their **start and end times (formatted as Month day, year, 24-hour time)**.
     - If no conflicts exist, proceed with the update.

4. **Ensuring the Summary (Title) in Updates**
   * Always **retain the original event summary** unless the user explicitly requests a change.
   * Always retrieve the existing summary using **FIND EVENTS** before updating.
   * When constructing the **UPDATE EVENT** schema:
     - If the user provides a new summary, use it.
     - Otherwise, **always include the original summary** from the retrieved event data.
   * **It is strictly forbidden to generate an UPDATE EVENT schema without a `summary` key.**

5. **Update Confirmation**
   * Once the event is successfully updated, confirm the new details to the user, specifying:
     - Any modified fields (time, description, location, etc.).
     - Start and end time (formatted as **Month day, year, 24-hour time**).
     - Any additional details provided.
---

### **Current Date**
* If the user does **not** provide a specific date in their request, use the current system date: **{current_date}**.
* If the user specifies a date, always prioritize the user-provided date over the system date.
---

### **Error Prevention**
* If the user provides input in the wrong format, clarify the expected format based on the requested action and ask for corrected input.
* If the system detects conflicting events, always prioritize informing the user about the conflict over performing the requested action.
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
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)


def build_calendar_react_agent(calendar_info, composio_entity_id):
    """Build a ReAct Agent that functions as a calendar worker with
    these capabilities:
        - "GOOGLECALENDAR_CREATE_EVENT",
        - "GOOGLECALENDAR_DELETE_EVENT",
        - "GOOGLECALENDAR_FIND_EVENT",
        - "GOOGLECALENDAR_FIND_FREE_SLOTS",
        - "GOOGLECALENDAR_UPDATE_EVENT"
    """

    calendar_tools = composio_toolset.get_tools(
        actions=[
            "GOOGLECALENDAR_CREATE_EVENT",
            "GOOGLECALENDAR_DELETE_EVENT",
            "GOOGLECALENDAR_FIND_EVENT",
            "GOOGLECALENDAR_FIND_FREE_SLOTS",
            "GOOGLECALENDAR_UPDATE_EVENT",
        ],
        entity_id=composio_entity_id,
    )

    calendar_worker_builder = StateGraph(WorkersState)

    gpt_llm_with_calendar_tools = llm.bind_tools(calendar_tools)

    async def llm_with_calendar_tools(state: WorkersState):
        """Generate an AIMessage that may include a tool-call to be sent."""
        now = datetime.now(PAKISTAN_TZ)
        datatime = now.strftime("%Y-%m-%d %H:%M:00")
        dayweek = now.strftime("%A")
        current_date = f"{dayweek}, {datatime}"

        if state["workers_messages"][0].type != "system":
            state["workers_messages"].insert(
                0,
                SystemMessage(
                    content=CALENDAR_WORKER_TEMPLATE.format(
                        calendar_info=calendar_info, current_date=current_date
                    )
                ),
            )
        response = await gpt_llm_with_calendar_tools.ainvoke(state["workers_messages"])
        return {"workers_messages": [response]}

    calendar_worker_builder.add_node(
        "llm_with_calendar_tools", llm_with_calendar_tools)

    tool_node = ToolNode(tools=calendar_tools, messages_key="workers_messages")
    calendar_worker_builder.add_node("calendar_tools", tool_node)

    calendar_worker_builder.add_conditional_edges(
        "llm_with_calendar_tools",
        tools_condition_worker,
        {"tools": "calendar_tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    calendar_worker_builder.add_edge(
        "calendar_tools", "llm_with_calendar_tools")
    calendar_worker_builder.add_edge(START, "llm_with_calendar_tools")

    return calendar_worker_builder.compile()


calendar_workers_dict = {
    "personal_calendar": build_calendar_react_agent(
        calendar_info="Personal Google Calendar",
        composio_entity_id="personal",

        
    ),
    "work_calendar": build_calendar_react_agent(
        calendar_info="Work Google Calendar",
        composio_entity_id="work",
    ),
}
calendar_workers_info_dict = {
    "personal_calendar": "Manages all personal events and reminders.",
    "work_calendar": "Manages all work-related events, meetings, and tasks.",
}
calendar_worker_summary_list = [
    "Personal Google Calendar", "Work Google Calendar"]
