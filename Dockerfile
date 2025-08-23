FROM python:3.12-slim

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# COPY ai-agents/ . # for development purposes it is better to mount the volume

# CMD ["python", "-m", "agent_workflow.discord_bot"]
