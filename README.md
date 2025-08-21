# AI-Agents Project

AI-Agents is a modular and extensible system designed with a hierarchical architecture. An orchestrator intelligently routes user requests to specialized managers based on context. The current implementation includes a **Calendar Manager** that delegates tasks to personal and work calendar agents. The architecture supports easy expansion to additional managers for other domains. The system leverages **LangGraph**, **LangChain**, and **OpenAI's GPT-5**, enabling structured decision-making, efficient task execution, and seamless response synthesis, making it adaptable for various automation scenarios.

---

## How to Run

### 1. Composio Setup

1. Log in to [Composio](https://composio.dev/) and retrieve your API key from **Settings**.
2. Add the API key to your `.env` file:

   ```env
   COMPOSIO_API_KEY=your_api_key_here
   ```
3. Create an entity in the Composio dashboard. By default, it is `"default"`. Update `.env` accordingly:

   ```env
   COMPOSIO_ENTITY_ID=default
   ENTITY_ID_WORK_ACCOUNT=work
   ENTITY_ID_PERSONA_ACCOUNT=personal  # optional
   ```
4. Update Composio to the latest version and add tools to your entities:

   ```bash
   composio add <tool> -e <entity>
   ```

   Example for Google Calendar:

   ```bash
   composio add googlecalendar -e personal
   ```
5. Complete account authorization and verify active tools in the dashboard:

   ```
   Entity > Auth Configs Section > Active Tools
   ```

---

### 2. Setting Environment Variables

Add the following to your `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

For other APIs, update the relevant LLM scripts accordingly.

---

### 3. PostgreSQL Setup

1. Download and install PostgreSQL (optional). SQLite will be used automatically if PostgreSQL is not available.
2. If using PostgreSQL, create a database and add the URI to your `.env` file:

   ```env
   POSTGRES_DB_URI=postgresql://postgre:<password>@<host>:<port>/<dbname>
   ```

---

## Discord Bot Setup

### 1. Create a Discord Application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and give it a name.
3. In the left menu, click **Bot** → **Add Bot**.

### 2. Get the Bot Token

1. In the **Bot** tab, under **Token**, click **Reset Token** → **Copy**.
2. Create a `.env` file in your project root:

   ```env
   DISCORD_BOT_TOKEN=your_token_here
   ```

### 3. Enable Required Intents

In the **Bot** settings (Discord Developer Portal):

* Scroll to **Privileged Gateway Intents**.
* Enable:

  * **MESSAGE CONTENT INTENT** (required to read messages)
  * (Optional) **SERVER MEMBERS INTENT** and **PRESENCE INTENT** if needed

### 4. Invite the Bot to Your Server

1. Go to **OAuth2 → URL Generator** in the left menu.
2. Under **Scopes**, select:

   * `bot`
3. Under **Bot Permissions**, select at least:

   * **Read Messages / View Channels**
   * **Send Messages**
   * **Read Message History**
4. Copy the generated URL, paste it in your browser, and invite the bot to your server.
