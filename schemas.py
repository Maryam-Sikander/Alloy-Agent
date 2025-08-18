from typing import List, Literal

from pydantic import BaseModel, Field
from app.calendar_workers import calendar_workers_dict
from app.email_workers import email_workers_dict

calendar_manager_outputs_tuple = tuple(calendar_workers_dict.keys())
email_manager_outputs_tuple = tuple(email_workers_dict.keys())

orchestrator_outputs_tuple = (
    "date_manage",
    "calendar_manage",
    "email_manage",
)


class OrchestratorRouter(BaseModel):
    """Orchestrator answer schema."""

    route_manager: Literal[orchestrator_outputs_tuple] = Field(
        ...,
        description="The route manager responsible for handling the request.",
    )
    query: str = Field(
        ...,
        description="The query for the manager.",
    )


class OrchestratorRouterList(BaseModel):
    """The managers to route tasks. If no manager needed, use a empty list `[]`."""

    managers: List[OrchestratorRouter]


class CalendarRouter(BaseModel):
    name: Literal[calendar_manager_outputs_tuple]
    task: str = Field(
        description="A concise description of the task to perform for the worker."
    )


class CalendarRouterList(BaseModel):
    """Worker to route to tasks. If no workers needed, use a empty list `[]`."""

    workers: List[CalendarRouter]


class EmailrRouter(BaseModel):
    name: Literal[email_manager_outputs_tuple]
    task: str = Field(
        description="A concise description of the task to perform for the worker."
    )


class EmailRouterList(BaseModel):
    """Worker to route to tasks. If no workers needed, use a empty list `[]`."""

    workers: List[EmailrRouter]
