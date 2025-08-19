import asyncio
from agent_workflow.orchestrator import init_orchestrator

async def main():
    orchestrator_graph = await init_orchestrator()

    config = {"configurable": {"thread_id": "testing_state_graph3.44"}}

    response = await orchestrator_graph.ainvoke(
        {"user_input": "Hello there!"},
        config
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
