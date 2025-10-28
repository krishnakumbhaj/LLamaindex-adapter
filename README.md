# LlamaIndex Adapter for uAgents

A seamless integration adapter that connects LlamaIndex agents with the uAgents framework, enabling your LlamaIndex workflows to communicate within the Fetch.ai agent ecosystem.

## 🌟 Features

- **Easy Integration**: Connect your existing LlamaIndex agents with minimal code changes
- **Session Management**: Built-in support for session and user context tracking
- **Agentverse Compatible**: Optional integration with Fetch.ai's Agentverse platform
- **Flexible Configuration**: Use with or without authentication tokens and context management
- **Async-First**: Built on modern async/await patterns for optimal performance


```

## 🚀 Quick Start

Here's a minimal example to get you started:

```python
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from your_agent_module import query_llamaindex_agent

async def handle_query(query: str, session_id: str, user_id: str) -> str:
    response = await query_llamaindex_agent(query)
    return str(response)

adapter = LlamaIndexAdapter(handle_query)

agent = Agent(
    name="my_llamaindex_agent",
    seed="your_secure_seed_phrase",
    port=8000,
    mailbox=True
)

agent.include(adapter.protocol, publish_manifest=True)
adapter.run(agent)
```

## 📖 Usage Scenarios

The adapter supports four main configuration patterns depending on your needs:

### 1. With Agentverse Token + Session Context

**Use Case**: Production deployment with full tracking and Agentverse integration

```python
from dataclasses import dataclass
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from your_agent_module import YourLlamaIndexAgent

AGENTVERSE_API_TOKEN = "av_xxxxxxxxxxxxxxxxxxxxx"

@dataclass
class SessionContext:
    session_id: str
    user_id: str

async def handle_query(query: str, session_id: str, user_id: str) -> str:
    """Process query with full session context"""
    context = SessionContext(session_id=session_id, user_id=user_id)
    agent = YourLlamaIndexAgent(context=context)
    response = await agent.query(query)
    return str(response)

adapter = LlamaIndexAdapter(
    handle_query,
    agentverse_api_token=AGENTVERSE_API_TOKEN,
    description="Production LlamaIndex agent with session tracking"
)

agent = Agent(
    name="production_llamaindex_agent",
    seed="your_secure_seed_phrase_production",
    port=8000,
    mailbox=True
)

agent.include(adapter.protocol, publish_manifest=True)
adapter.run(agent)
```

### 2. Session Context Only (No Agentverse)

**Use Case**: Local development or private deployments with session tracking

```python
from dataclasses import dataclass
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from your_agent_module import YourLlamaIndexAgent

@dataclass
class SessionContext:
    session_id: str
    user_id: str

async def handle_query(query: str, session_id: str, user_id: str) -> str:
    """Process query with session context, no Agentverse"""
    context = SessionContext(session_id=session_id, user_id=user_id)
    agent = YourLlamaIndexAgent(context=context)
    response = await agent.query(query)
    return str(response)

adapter = LlamaIndexAdapter(handle_query)

agent = Agent(
    name="dev_llamaindex_agent",
    seed="your_secure_seed_phrase_dev",
    port=8000,
    mailbox=True
)

agent.include(adapter.protocol, publish_manifest=True)
adapter.run(agent)
```

### 3. With Agentverse Token (Stateless)

**Use Case**: Simple production deployment without session management

```python
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from your_agent_module import YourLlamaIndexAgent

AGENTVERSE_API_TOKEN = "av_xxxxxxxxxxxxxxxxxxxxx"

async def handle_query(query: str, session_id: str, user_id: str) -> str:
    """Process query without maintaining session state"""
    agent = YourLlamaIndexAgent()
    response = await agent.query(query)
    return str(response)

adapter = LlamaIndexAdapter(
    handle_query,
    agentverse_api_token=AGENTVERSE_API_TOKEN,
    description="Stateless LlamaIndex agent for simple queries"
)

agent = Agent(
    name="stateless_llamaindex_agent",
    seed="your_secure_seed_phrase_stateless",
    port=8000,
    mailbox=True
)

agent.include(adapter.protocol, publish_manifest=True)
adapter.run(agent)
```

### 4. Minimal Configuration

**Use Case**: Quick prototyping and testing

```python
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from your_agent_module import YourLlamaIndexAgent

async def handle_query(query: str, session_id: str, user_id: str) -> str:
    """Minimal query handler for testing"""
    agent = YourLlamaIndexAgent()
    response = await agent.query(query)
    return str(response)

adapter = LlamaIndexAdapter(handle_query)

agent = Agent(
    name="test_llamaindex_agent",
    seed="test_seed_phrase",
    port=8000,
    mailbox=True
)

agent.include(adapter.protocol, publish_manifest=True)
adapter.run(agent)
```

## 🔧 Configuration Options

### LlamaIndexAdapter Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_handler` | `Callable` | Yes | Async function that processes queries |
| `agentverse_api_token` | `str` | No | Token for Agentverse platform integration |
| `description` | `str` | No | Human-readable description of your agent |

### Query Handler Signature

Your query handler must match this signature:

```python
async def query_handler(
    query: str,        # The user's query
    session_id: str,   # Unique session identifier
    user_id: str       # Unique user identifier
) -> str:              # Return response as string
    pass
```

## 🏗️ Example Implementation

Here's a complete example showing how to implement your LlamaIndex agent module:

```python

# your_agent_module.py
from llama_index.llms.google_genai import GoogleGenAI

async def create_dummy_llamaindex_agent_with_ctx(query: str, tx):
    """
    Creates a dummy LlamaIndex agent that accepts context via 'tx' parameter.
    
    Args:
        query: The user's query
        tx: Context object containing session_id and user_id (REQUIRED)
    """
    session_id = tx.session_id
    user_id = tx.user_id
    
    print(f"[dummy_agent] Query from user={user_id}, session={session_id}: {query}")
    
    # Initialize Gemini LLM
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash", 
        api_key="AIzaSyCQLXiuy8kKOaTlZRyADitrEBh9a5TKA_w"
    )
    
    # Use the LLM directly (no agent needed for testing)
    response = await llm.acomplete(query)
    
    print(f"[dummy_agent] Response: {str(response)[:100]}...")
    
    return str(response)
```

## 🔐 Security Best Practices

- **Never commit tokens**: Store `AGENTVERSE_API_TOKEN` in environment variables
- **Use strong seeds**: Generate secure seed phrases for production agents
- **Validate inputs**: Always sanitize user queries before processing
- **Rate limiting**: Implement rate limiting for production deployments

```python
import os

AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN")
AGENT_SEED = os.getenv("AGENT_SEED")
```





---

**Made with ❤️ for the Fetch.ai agent ecosystem**