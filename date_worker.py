from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pytz
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)


PAKISTAN_TZ = pytz.timezone("Asia/Karachi")


DATE_WORKER_SYSTEM_PROMPT = """
You are an expert assistant specialized in recognizing and extracting temporal expressions from natural language input. The current date is: {current_date}.

Your task is to analyze the provided text and return a structured response with:
- 'start_datetime': The start date in ISO 8601 format (e.g., '2025-03-01T00:00:00+02:00').
- 'end_datetime': The end date in ISO 8601 format (e.g., '2025-03-01T23:59:00+02:00').
- 'description': A message explaining errors or ambiguities (only for errors).

### General Guidelines:
1. All dates must be in the Asia/Karachi time zone (UTC+5).
2. If no specific end date is mentioned, infer it based on context:
   - For single-day references (e.g., "today"), set `end_datetime` to 23:59 of the same day.
   - For multi-day ranges (e.g., "today and tomorrow"), set `start_datetime` to 00:00 of the first day and `end_datetime` to 23:59 of the last day.
3. Always ensure that `start_datetime` and `end_datetime` are calculated relative to the current time or date, and never leave them empty unless it is a clear error.

### Rules for Time References:
1. If only a time is mentioned (e.g., "at 9 AM"), assume it refers to today.
2. If a time range is mentioned (e.g., "from 2 PM to 4 PM"), use the exact times provided.
3. If an approximate time of day is mentioned:
   - "morning" → 06:00 - 12:00.
   - "afternoon" → 12:00 - 18:00.
   - "evening" → 18:00 - 22:00.
   - "night" → 22:00 - 06:00 (next day).

### Rules for Date Interpretation:
1. **Specific Dates**: Extract exact dates (e.g., "March 5th, 2024") and convert them to ISO 8601.
2. **Future Dates**:
   - If a given date has already passed this year, return the next occurrence in the following year.
   - If only a day (e.g., "the 12th") is provided, assume the next occurrence in the next month.
3. **Past Dates**:
   - If a given date has not yet occurred this year, return the last occurrence in the previous year.
   - If only a day (e.g., "the past 12th") is provided, assume the last occurrence in the current month.
4. **Relative Dates**:
   - "in X days" → Add X days to the current date.
   - "X days ago" → Subtract X days from the current date.
   - "next week" → Start on Sunday at 00:00 and end on Saturday at 23:59.
   - "next month" → Start on the 1st at 00:00 and end on the last day at 23:59.
   - If no specific time is mentioned, assume a default duration:
     - Minutes/hours → 1 hour.
     - Days/weeks/months/years → Full unit duration.

### Error Handling:
- If an invalid date is detected (e.g., "February 31") or an invalid , the `description` should provide an explanation of why the date is invalid.
- If `start_datetime` is later than `end_datetime`, the `description` should indicate that the time range is invalid (e.g., "from 10 AM to 4 AM").
- If no date or time is mentioned, the `description` should explain that no temporal reference was detected.
- Never leave `start_datetime` or `end_datetime` empty unless it is a clear error.
- **IMPORTANT**: If there is an error, the `description` must always begin with the word **"Error:"** followed by an explanation.

### Examples:
{examples}

"""


class DateExtractionResult(BaseModel):
    start_datetime: str = Field(...,
                                description="Start date in ISO 8601 format.")
    end_datetime: str = Field(..., description="End date in ISO 8601 format.")
    description: str = Field(
        default="",
        description="Message explaining errors or ambiguities (only for errors).",
    )


def generate_training_examples(current_date: datetime) -> list:
    """Generate 50 training examples based on the current date."""
    examples = []

    common_cases = [
        "tomorrow",
        "next week",
        "last month",
        "in 3 days",
        "from Monday to Friday",
        "at 9 AM",
        "this Christmas",
        "between July and August",
        "the first week of September",
        "two months ago",
        "next Friday",
        "last Monday",
        "this weekend",
        "today in the afternoon",
        "in 30 minutes",
        "tomorrow night",
        "Thursday from 5 to 9 PM",
        "the next 4 days",
        "the last 2 weeks",
        "the last week",
        "the rest of this week",
        "the rest of this month",
        "next month",
        "from the 4th to the 9th",
        "March 5th",
        "last January 3rd",
        "March 23, 2024",
        "February 4, 2026",
        "from March 6 to April 24",
        "the month of May",
        "the first days of October",
        "next summer",
        "in 2 weeks",
        "6 days ago",
        "the rest of the year",
        "next year",
        "in 3 hours",
        "yesterday at 8 PM",
        "next Monday at 10 AM",
        "last Saturday at 7 PM",
        "from 12 PM to 3 PM next Saturday",
        "in the morning",
        "this Saturday night",
        "tonight",
        "early morning",
        "today and tomorrow",
        "2 days ago",
        "the next 2 weeks",
    ]

    for input_text in common_cases:
        if "tomorrow" in input_text:
            start_datetime = current_date + timedelta(days=1)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "next week" in input_text:
            start_datetime = current_date + \
                timedelta(days=(7 - current_date.weekday()))
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + \
                timedelta(days=6, hours=23, minutes=59)
        elif "last month" in input_text:
            start_datetime = (current_date.replace(day=1) - timedelta(days=1)).replace(
                day=1
            )
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(day=28) + timedelta(days=4)
            end_datetime = (end_datetime - timedelta(days=end_datetime.day)).replace(
                hour=23, minute=59, second=0
            )
        elif "in 3 days" in input_text:
            start_datetime = current_date + timedelta(days=3)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "from Monday to Friday" in input_text:
            start_datetime = current_date + timedelta(
                days=(0 - current_date.weekday() + 7) % 7
            )
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + \
                timedelta(days=4, hours=23, minutes=59)
        elif "at 9 AM" in input_text:
            start_datetime = current_date.replace(
                hour=9, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=1)
        elif "this Christmas" in input_text:
            christmas_date = datetime(
                current_date.year, 12, 25, 0, 0, 0, tzinfo=PAKISTAN_TZ
            )
            if current_date > christmas_date:
                christmas_date = datetime(
                    current_date.year + 1, 12, 25, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            start_datetime = christmas_date
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "between July and August" in input_text:
            start_datetime = datetime(
                current_date.year, 7, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ)
            end_datetime = datetime(
                current_date.year, 8, 31, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "the first week of September" in input_text:
            start_datetime = datetime(
                current_date.year, 9, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ)
            end_datetime = start_datetime + \
                timedelta(days=6, hours=23, minutes=59)
        elif "two months ago" in input_text:
            start_datetime = (current_date.replace(day=1) - timedelta(days=1)).replace(
                day=1
            ) - timedelta(days=30)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(day=28) + timedelta(days=4)
            end_datetime = (end_datetime - timedelta(days=end_datetime.day)).replace(
                hour=23, minute=59, second=0
            )
        elif "next Friday" in input_text:
            start_datetime = current_date + timedelta(
                days=(4 - current_date.weekday() + 7) % 7
            )
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "last Monday" in input_text:
            start_datetime = current_date - \
                timedelta(days=current_date.weekday())
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "this weekend" in input_text:
            start_datetime = current_date + \
                timedelta(days=(5 - current_date.weekday()))
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + \
                timedelta(days=1, hours=23, minutes=59)
        elif "today in the afternoon" in input_text:
            start_datetime = current_date.replace(
                hour=12, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=6)
        elif "in 30 minutes" in input_text:
            start_datetime = current_date + timedelta(minutes=30)
            start_datetime = start_datetime.replace(tzinfo=PAKISTAN_TZ)
            end_datetime = start_datetime + timedelta(hours=1)
        elif "tomorrow night" in input_text:
            start_datetime = current_date + timedelta(days=1)
            start_datetime = start_datetime.replace(
                hour=20, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "Thursday from 5 to 9 PM" in input_text:
            start_datetime = current_date + timedelta(
                days=(3 - current_date.weekday() + 7) % 7
            )
            start_datetime = start_datetime.replace(
                hour=17, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=4)
        elif "the next 4 days" in input_text:
            start_datetime = current_date
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + \
                timedelta(days=4, hours=23, minutes=59)
        elif "the last 2 weeks" in input_text:
            start_datetime = current_date - timedelta(weeks=2)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = current_date.replace(hour=23, minute=59, second=0)
        elif "the last week" in input_text:
            start_datetime = current_date - timedelta(weeks=1)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = current_date.replace(hour=23, minute=59, second=0)
        elif "the rest of this week" in input_text:
            start_datetime = current_date
            start_datetime = start_datetime.replace(
                minute=0, second=0, tzinfo=PAKISTAN_TZ)
            end_datetime = current_date + \
                timedelta(days=(6 - current_date.weekday()))
            end_datetime = end_datetime.replace(hour=23, minute=59, second=0)
        elif "the rest of this month" in input_text:
            start_datetime = current_date
            start_datetime = start_datetime.replace(
                minute=0, second=0, tzinfo=PAKISTAN_TZ)
            end_datetime = current_date.replace(day=28) + timedelta(days=4)
            end_datetime = (end_datetime - timedelta(days=end_datetime.day)).replace(
                hour=23, minute=59, second=0
            )
        elif "next month" in input_text:
            start_datetime = (current_date.replace(day=28) + timedelta(days=4)).replace(
                day=1
            )
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(day=28) + timedelta(days=4)
            end_datetime = (end_datetime - timedelta(days=end_datetime.day)).replace(
                hour=23, minute=59, second=0
            )
        elif "from the 4th to the 9th" in input_text:
            start_datetime = current_date.replace(
                day=4, hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = current_date.replace(
                day=9, hour=23, minute=59, second=0)
        elif "March 5th" in input_text:
            if current_date.month > 3 or (
                current_date.month == 3 and current_date.day > 5
            ):
                start_datetime = datetime(
                    current_date.year, 3, 5, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            else:
                start_datetime = datetime(
                    current_date.year, 3, 5, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "last January 3rd" in input_text:
            if current_date.month > 1 or (
                current_date.month == 1 and current_date.day > 3
            ):
                start_datetime = datetime(
                    current_date.year, 1, 3, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            else:
                start_datetime = datetime(
                    current_date.year - 1, 1, 3, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "March 23, 2024" in input_text:
            start_datetime = datetime(2024, 3, 23, 0, 0, 0, tzinfo=PAKISTAN_TZ)
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "February 4, 2026" in input_text:
            start_datetime = datetime(2026, 2, 4, 0, 0, 0, tzinfo=PAKISTAN_TZ)
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "from March 6 to April 24" in input_text:
            if current_date.month > 3 or (
                current_date.month == 3 and current_date.day > 6
            ):
                start_datetime = datetime(
                    current_date.year, 3, 6, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            else:
                start_datetime = datetime(
                    current_date.year, 3, 6, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            end_datetime = datetime(
                current_date.year, 4, 24, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "the month of May" in input_text:
            start_datetime = datetime(
                current_date.year, 5, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ)
            end_datetime = datetime(
                current_date.year, 5, 31, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "the first days of October" in input_text:
            start_datetime = datetime(
                current_date.year, 10, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = datetime(
                current_date.year, 10, 3, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "next summer" in input_text:
            if current_date.month >= 7:
                start_datetime = datetime(
                    current_date.year + 1, 7, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            else:
                start_datetime = datetime(
                    current_date.year, 7, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ
                )
            end_datetime = datetime(
                start_datetime.year, 8, 31, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )

        elif "in 2 weeks" in input_text:
            start_datetime = current_date + timedelta(weeks=2)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "6 days ago" in input_text:
            start_datetime = current_date - timedelta(days=6)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "the rest of the year" in input_text:
            start_datetime = current_date
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = datetime(
                current_date.year, 12, 31, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "next year" in input_text:
            start_datetime = datetime(
                current_date.year + 1, 1, 1, 0, 0, 0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = datetime(
                current_date.year + 1, 12, 31, 23, 59, 0, tzinfo=PAKISTAN_TZ
            )
        elif "in 3 hours" in input_text:
            start_datetime = current_date + timedelta(hours=3)
            end_datetime = start_datetime + timedelta(hours=1)
        elif "yesterday at 8 PM" in input_text:
            start_datetime = current_date - timedelta(days=1)
            start_datetime = start_datetime.replace(
                hour=20, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=1)
        elif "next Monday at 10 AM" in input_text:
            start_datetime = current_date + timedelta(
                days=(0 - current_date.weekday() + 7) % 7
            )
            start_datetime = start_datetime.replace(
                hour=10, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=1)
        elif "last Saturday at 7 PM" in input_text:
            start_datetime = current_date - timedelta(
                days=(current_date.weekday() + 2) % 7
            )
            start_datetime = start_datetime.replace(
                hour=19, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime + timedelta(hours=1)
        elif "from 12 PM to 3 PM next Saturday" in input_text:
            start_datetime = current_date + timedelta(
                days=(5 - current_date.weekday() + 7) % 7
            )
            start_datetime = start_datetime.replace(
                hour=12, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=15, minute=0, second=0)
        elif "in the morning" in input_text:
            start_datetime = current_date.replace(
                hour=6, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=12, minute=0, second=0)
        elif "this Saturday night" in input_text:
            start_datetime = current_date + \
                timedelta(days=(5 - current_date.weekday()))
            start_datetime = start_datetime.replace(
                hour=20, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "tonight" in input_text:
            start_datetime = current_date.replace(
                hour=20, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "early morning" in input_text:
            start_datetime = current_date.replace(
                hour=6, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=9, minute=0, second=0)
        elif "today and tomorrow" in input_text:
            start_datetime = current_date.replace(
                hour=1, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = (current_date + timedelta(days=1)).replace(
                hour=23, minute=59, second=0, microsecond=0, tzinfo=PAKISTAN_TZ
            )
        elif "2 days ago" in input_text:
            start_datetime = current_date - timedelta(days=2)
            start_datetime = start_datetime.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = start_datetime.replace(hour=23, minute=59, second=0)
        elif "the next 2 weeks" in input_text:
            start_datetime = current_date.replace(
                hour=0, minute=0, second=0, tzinfo=PAKISTAN_TZ
            )
            end_datetime = current_date + timedelta(weeks=2)
            end_datetime = end_datetime.replace(
                hour=23, minute=59, second=0, tzinfo=PAKISTAN_TZ
            )
        else:
            start_datetime = current_date
            end_datetime = start_datetime + timedelta(hours=1)

        examples.append(
            {
                "input": input_text,
                "output": {
                    "start_datetime": start_datetime.isoformat(),
                    "end_datetime": end_datetime.isoformat(),
                    "description": "",
                },
            }
        )

    return examples


def get_prompt_with_examples(current_date: datetime) -> str:
    """Generate a prompt with dynamic training examples."""
    examples = generate_training_examples(current_date)

    example_text = "\n".join(
        f"- Input: {ex['input']}\n  Output: {ex['output']}" for ex in examples
    )

    return DATE_WORKER_SYSTEM_PROMPT.format(
        current_date=current_date.strftime("%A, %Y-%m-%d %H:%M:%S"),
        examples=example_text,
    )


def calculate_date(user_input: str) -> str:
    """Extract structured date information from natural language input."""
    try:
        now = datetime.now(PAKISTAN_TZ)
        prompt = get_prompt_with_examples(now)
        # print(prompt)

        response = llm.with_structured_output(
            DateExtractionResult, method="function_calling"
        ).invoke([SystemMessage(content=prompt), HumanMessage(content=user_input)])
        # print(f"Respuesta del modelo: {response}")

        if response.description.startswith("Error:"):
            return response.description

        return (
            f"The requested date range is from {response.start_datetime} "
            f"to {response.end_datetime} ('Asia/Karachi' time zone)."
        )

    except Exception as e:
        return f"Error: Could not process the request. Details: {str(e)}"
