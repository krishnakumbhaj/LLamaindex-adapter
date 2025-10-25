# uAgents Adapter

This package provides adapters for integrating [uAgents](https://github.com/fetchai/uAgents) with popular AI libraries:

- **LangChain Adapter**: Convert LangChain agents to uAgents
- **CrewAI Adapter**: Convert CrewAI crews to uAgents
- **MCP Server Adapter**: Integrate Model Control Protocol (MCP) servers with uAgents
- **A2A Outbound Adapter**: Bridges uAgents and A2A servers 
- **A2A Inbound Adapter**: Bridge Agentverse agents to A2A protocol for AI assistants

## Installation

```bash
# Install the base package
pip install uagents-adapter

# Install with LangChain support
pip install "uagents-adapter[langchain]"

# Install with CrewAI support
pip install "uagents-adapter[crewai]"

# Install with MCP support
pip install "uagents-adapter[mcp]"

# Install with A2A Inbound support
pip install "uagents-adapter[a2a-inbound]"

# Install with A2A Outbound support
pip install "uagents-adapter[a2a-outbound]"
```

## LangChain Adapter

The LangChain adapter allows you to convert any LangChain agent into a uAgent that can interact with other agents in the Agentverse ecosystem.

```python
from langchain_core.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from uagents_adapter import LangchainRegisterTool

# Create your LangChain agent
llm = ChatOpenAI(model_name="gpt-4")
tools = [...]  # Your tools here
agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create uAgent register tool
register_tool = LangchainRegisterTool()

# Register the agent as a uAgent
result = register_tool.invoke({
    "agent_obj": agent_executor,
    "name": "my_langchain_agent",
    "port": 8000,
    "description": "My LangChain agent as a uAgent",
    "mailbox": True,  # Use Agentverse mailbox service
    "api_token": "YOUR_AGENTVERSE_API_TOKEN",  # Optional: for Agentverse registration
    "return_dict": True  # Return a dictionary instead of a string
})

print(f"Created uAgent '{result['agent_name']}' with address {result['agent_address']} on port {result['agent_port']}")
```

## CrewAI Adapter

The CrewAI adapter allows you to convert any CrewAI crew into a uAgent.

```python
from crewai import Crew, Agent, Task
from uagents_adapter import CrewaiRegisterTool

# Define your CrewAI crew
agent1 = Agent(
    role="Researcher",
    goal="Research thoroughly",
    backstory="You are a skilled researcher",
    verbose=True,
    allow_delegation=False
)

task1 = Task(
    description="Research about a topic",
    agent=agent1
)

crew = Crew(
    agents=[agent1],
    tasks=[task1],
    verbose=True
)

# Create CrewAI register tool
register_tool = CrewaiRegisterTool()

# Register the crew as a uAgent
result = register_tool.invoke({
    "crew_obj": crew,
    "name": "my_crew_agent",
    "port": 8001,
    "description": "My CrewAI crew as a uAgent",
    "mailbox": True,  # Use Agentverse mailbox service
    "api_token": "YOUR_AGENTVERSE_API_TOKEN",  # Optional: for Agentverse registration
    "query_params": {
        "topic": {
            "type": "string",
            "description": "The topic to research",
            "required": True
        }
    },
    "example_query": "Research about artificial intelligence",
    "return_dict": True  # Return a dictionary instead of a string
})

print(f"Created uAgent '{result['agent_name']}' with address {result['agent_address']} on port {result['agent_port']}")
```

## MCP Server Adapter

The MCP Server Adapter allows you to host your MCP Servers on Agentverse and get discovered by ASI:One by enabling Chat Protocol.

First, create a FastMCP server implementation in a `server.py` file that exposes the required `list_tools` and `call_tool` async methods. Then, in the following `agent.py`, import the MCP server instance and use it with the MCPServerAdapter:

```python
from uagents import Agent
from uagents_adapter import MCPServerAdapter
from server import mcp

# Create an MCP adapter
mcp_adapter = MCPServerAdapter(
    mcp_server=mcp,
    asi1_api_key="your_asi1_api_key",
    model="asi1-mini"     # Model options: asi1-mini, asi1-extended, asi1-fast
)

# Create a uAgent
agent = Agent()

# Add the MCP adapter protocols to the agent
for protocol in mcp_adapter.protocols:
    agent.include(protocol)

# Run the MCP adapter with the agent
mcp_adapter.run(agent)
```

> **Important**: When creating MCP tools, always include detailed docstrings using triple quotes (`"""`) to describe what each tool does, when it should be used, and what parameters it expects. These descriptions are critical for ASI:One to understand when and how to use your tools.

 

## A2A Outbound Adapter

The A2A Outbound Adapter allows you to connect your uAgents with Chat Protocol to Google A2A Servers.

First, create your A2A servers in a directory named `agents` and then import the Agent Executors in your `agent.py` (uagent) file along with the SingleA2AAdapter or MultiA2AAdapter depending on whether you want to connect to a single Google A2A Server or Multiple A2A Servers. You will have to provide the Agent card to the Adapter and the Adapter will run the servers and enable the uagent with Chat Protocol so that it becomes discoverable through ASI:One and you can start interacting with any A2A Server using the A2A Outbound Adapter.

```python
from uagents_adapter import SingleA2AAdapter, A2AAgentConfig, a2a_servers

#Import A2A Server executor from your A2A Server code 
from brave.agent import BraveSearchAgentExecutor

def main():

    #Add details of your A2A Server
    agent_config = A2AAgentConfig(  
        name="brave_search_specialist",             #Name of A2A Server 
        description="AI Agent for web and news search using Brave Search API",       #Description of A2A Server 
        url="http://localhost:10020",               #Endpoint where the A2A Server should be running
        port=10020,                                 #port where the A2A Server should be running
        specialties=["web search", "news", "information retrieval", "local business", "site-specific lookup"],
        priority=3                                  
    )
    executor = BraveSearchAgentExecutor()
    a2a_servers([agent_config], {agent_config.name: executor})
    print(f"AgentCard manifest URL: http://localhost:{agent_config.port}/.well-known/agent.json")
    adapter = SingleA2AAdapter(
        agent_executor=executor,                #Your A2A Server Executor
        name="brave",                           #Name of uAgent 
        description="Routes queries to Brave Search AI specialists",    #Description of uAgent 
        port=8200,                              #Port where the uAgent should be running
        a2a_port=10020                          #Port where the A2A server should be running
    )
    adapter.run()

if __name__ == "__main__":
    main()
```

#### Notes
- Use `A2AAgentConfig` for all agent configuration
- Start A2A servers with `a2a_servers` for manifest and server management
- Use `SingleA2AAdapter` or `MultiA2AAdapter` for orchestration and chat integration
- After starting, inspect manifest URLs at `http://localhost:{port}/.well-known/agent.json`

### Payment bridging (AP2 ↔ Fetch.ai)

The outbound adapter includes payment bridging between AP2 artifacts and Fetch.ai Payment Protocol:
- AP2 `CartMandate` → Fetch.ai `RequestPayment`
- Fetch.ai `CommitPayment` → AP2 `PaymentMandate` (forwarded to producing A2A agent)
- AP2 `PaymentSuccess` → Fetch.ai `CompletePayment`
- AP2 `PaymentFailure` → Fetch.ai `CancelPayment`

Response parsing prioritizes AP2 data parts in A2A JSON-RPC responses, so carts and payment results propagate as typed objects rather than plain text.

 
### JSON-RPC message format for A2A

```json
{
  "id": "<request_id>",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [ { "type": "text", "text": "python guide" } ],
      "messageId": "<message_id>"
    }
  }
}
```

### Minimal run example (package-based)

```python
from uagents_adapter.a2a_outbound import (
    SingleA2AAdapter,
    MultiA2AAdapter,
    A2AAgentConfig,
    a2a_servers,
)

# Import your own A2A agent executors from your application
from your_project.executors import YourAgentExecutor


def run_single():
    config = A2AAgentConfig(
        name="my_specialist",
        description="Demo A2A specialist",
        url="http://localhost:10020",
        port=10020,
        specialties=["demo"],
    )
    executor = YourAgentExecutor()
    a2a_servers([config], {config.name: executor})
    adapter = SingleA2AAdapter(
        agent_executor=executor,
        name="my_uagent",
        description="Routes to a single A2A specialist",
        port=8200,
        a2a_port=10020,
    )
    adapter.run()


def run_multi():
    configs = [
        A2AAgentConfig(
            name="specialist_one",
            description="First A2A specialist",
            url="http://localhost:10020",
            port=10020,
            specialties=["one"],
            priority=2,
        ),
        A2AAgentConfig(
            name="specialist_two",
            description="Second A2A specialist",
            url="http://localhost:10022",
            port=10022,
            specialties=["two"],
        ),
    ]
    executors = {c.name: YourAgentExecutor() for c in configs}
    a2a_servers(configs, executors)
    adapter = MultiA2AAdapter(
        name="coordinator",
        description="Routes queries to multiple A2A specialists",
        llm_api_key="",  # optional; leave empty to disable LLM routing
        port=8200,
        routing_strategy="keyword_match",
    )
    for cfg in configs:
        adapter.add_agent_config(cfg)
    adapter.run()


if __name__ == "__main__":
    run_single()  # or run_multi()
```


## A2A Inbound Adapter

The A2A Inbound Adapter allows you to bridge any existing Agentverse agent to the A2A (Agent-to-Agent) ecosystem, making your uAgents accessible through the A2A protocol for AI assistants and other applications.

```python
from uagents_adapter import A2ARegisterTool

# Create A2A register tool
register_tool = A2ARegisterTool()

# Choose the agent you want to use from Agentverse.ai, copy its address in the config, add agent details or create a custom agent yourself and add its address in the config
# Configure your agent bridge
config = {
    "agent_address": "agent1qv4zyd9sta4f5ksyhjp900k8kenp9vczlwqvr00xmmqmj2yetdt4se9ypat",
    "name": "Finance Analysis Agent", 
    "description": "Financial analysis and market insights agent",
    "skill_tags": ["finance", "analysis", "markets", "investment"],
    "skill_examples": ["Analyze AAPL stock performance", "Compare crypto portfolios"],
    "port": 10000,  # A2A server port (default)
    "bridge_port": 9000,  # Optional: bridge port (auto-derived if not set)
    "host": "localhost"  # Default host
}

# Start the A2A bridge server
result = register_tool.invoke(config)

print(f"A2A server running on {config['host']}:{config['port']}")
print(f"Bridging to Agentverse agent: {config['agent_address']}")
```

For CLI usage:

```bash
# Set unique bridge seed for production
export UAGENTS_BRIDGE_SEED="your_unique_production_seed_2024"

# Start the A2A bridge
python -m uagents_adapter.a2a_inbound.cli \
  --agent-address agent1qv4zyd9sta4f5ksyhjp900k8kenp9vczlwqvr00xmmqmj2yetdt4se9ypat \
  --agent-name "Finance Agent" \
  --skill-tags "finance,analysis,markets" \
  --port 10000
```

> **Security Note**: Always set `UAGENTS_BRIDGE_SEED` environment variable for production deployments to ensure consistent bridge agent addresses across restarts and prevent conflicts.

 

## Agentverse Integration

### Mailbox Service

By default, agents are created with `mailbox=True`, which enables the agent to use the Agentverse mailbox service. This allows agents to communicate with other agents without requiring a publicly accessible endpoint.

When mailbox is enabled:
- Agents can be reached by their agent address (e.g., `agent1q...`)
- No port forwarding or public IP is required
- Messages are securely handled through the Agentverse infrastructure

### Agentverse Registration

You can optionally register your agent with the Agentverse API, which makes it discoverable and usable by other users in the Agentverse ecosystem:

1. Obtain an API token from [Agentverse.ai](https://agentverse.ai)
2. Include the token when registering your agent:
   ```python
   result = register_tool.invoke({
       # ... other parameters
       "api_token": "YOUR_AGENTVERSE_API_TOKEN"
   })
   ```

When an agent is registered with Agentverse:
- It connects to the mailbox service automatically
- It appears in the Agentverse directory
- A README with input/output models is automatically generated
- The agent gets an "innovationlab" badge
- Other users can discover and interact with it
- You can monitor its usage and performance through the Agentverse dashboard

Example of auto-generated README for LangChain agents:
```markdown
# Agent Name
Agent Description
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**Input Data Model**
```python
class QueryMessage(Model):
    query: str
```

**Output Data Model**
```python
class ResponseMessage(Model):
    response: str
```
```

Example of auto-generated README for CrewAI agents with parameters:
```markdown
# Agent Name
Agent Description
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

**Input Data Model**
```python
class ParameterMessage(Model):
    topic: str
    max_results: int | None = None
```

**Output Data Model**
```python
class ResponseMessage(Model):
    response: str
```

**Example Query**
```
Research about artificial intelligence
```
```

## License

Apache 2.0















































import asyncio
import os
import sys
from pathlib import Path

# Add parent's src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter

# Import your LlamaIndex agent
from llamaindex_dummy_agent import create_dummy_llamaindex_agent


async def main():
    print("="*70)
    print("🚀 LlamaIndex → uAgents Adapter (Clean API)")
    print("="*70)
    
    # Step 1: Create your LlamaIndex agent
    print("\n[1/5] Creating LlamaIndex agent...")
    agent_ll = create_dummy_llamaindex_agent()
    print("✅ LlamaIndex agent created")
    
    # Step 2: Create adapter with optional Agentverse token
    print("\n[2/5] Creating adapter...")
    AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN") or "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NjM5MDcwNzQsImlhdCI6MTc2MTMxNTA3NCwiaXNzIjoiZmV0Y2guYWkiLCJqdGkiOiJhNjlhMjU4MGU1MjEwOTFkYjg5OWIwYTQiLCJzY29wZSI6ImF2Iiwic3ViIjoiMjY5NDQ4MWE2Yzg2NGY4N2YwNWIxZWE3NWI3MjMyMDFiOGEyNDJjZjYyNGZjZGVlIn0.dwEGJkN2Gx8IYxIuNONXw2N_ccqwkLVW5ZTzZpnLcZ3nqIDJbNhyp8EAnl1slapTgXKkphS0pZkC6MXx6Zrz1x9aZHzKy83Qc3JtEv7MHCZbjKLHAG5oeNtr51LygCXvGO3latv9_Y8G5sBlwAYc5e3XEnieLCX-YicpF_Nt-B2W6qZ47-qMYzwbO_P_XixENbkE-exvhYCux4O9GlmqObjf2t9oibBIvgugQDVzKP98I9Ei3BqUXXTEcHQg6YIKQ3C43SrqXmxXjz08R7Zn6wXkE9JC-PMFh4rjV1FRAvKwEQyoPiHOPiPDNrm-RseAAau_Nqv6m_0EdQEdtkMn3g"
    
    adapter = LlamaIndexAdapter(
        agent_ll,
        agentverse_api_token=AGENTVERSE_API_TOKEN,
        description="LlamaIndex agent with Gemini 2.5 Flash"
    )
    print("✅ Adapter created")
    
    # Step 3: Create uAgent (YOU control this!)
    print("\n[3/5] Creating uAgent (user-controlled)...")
    agent = Agent(
        name="my_llamaindex_agent",
        seed="my_unique_seed_phrase_12345",
        port=8001,
        mailbox=True  # ✅ Just mailbox (same as LangChain!)
    )
    print(f"✅ uAgent created: {agent.address[:16]}...")
    
    # Step 4: Include protocol
    print("\n[4/5] Including protocol...")
    agent.include(adapter.protocol, publish_manifest=True)
    print("✅ Protocol included")
    
    # Step 5: Enable auto-registration (ONE LINE!)
    print("\n[5/5] Enabling Agentverse registration...")
    if adapter.agentverse_api_token:
        adapter.enable_agentverse_registration(agent, wait_seconds=30)  # ✅ Increased to 30s
        print("✅ Auto-registration enabled (adapter handles internally)")
        print("   Will attempt registration 40s after startup (waits for mailbox)")
    else:
        print("⚠️  No API token - skipping Agentverse registration")
    
    # Run
    print("\n" + "="*70)
    print("🚀 Agent Running!")
    print("="*70)
    print(f"📍 Address: {agent.address}")
    print(f"🔌 Port: 8001")
    print(f"📬 Mailbox: Enabled (same as LangChain!)")
    print(f"🌐 Agentverse: {'Will register in 40s (waits for mailbox)' if adapter.agentverse_api_token else 'Disabled'}")
    print("\n💡 First time running? Open inspector link and click 'Connect'!")
    print("\n💡 Should work just like LangChain now!")
    print("   ✅ Almanac registration")
    print("   ✅ Inspector link")
    print("   ✅ Marketplace listing")
    print("\n⚡ Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    await agent.run_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Agent stopped by user")
        print("="*70)
