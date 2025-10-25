# LlamaIndex Adapter for uAgents# LlamaIndex Adapter for uAgents



A production-ready adapter that integrates LlamaIndex query engines with uAgents using the MCP pattern, giving users full control over agent configuration while providing robust error handling and comprehensive protocol support.A flexible adapter that wraps LlamaIndex query engines as uAgents, combining the simplicity of LangChain-style API with the flexibility of MCP-style protocol customization.



## 🎯 Features## Features



- ✅ **MCP Pattern**: User controls agent creation, adapter creates protocols✅ **LangChain-style API** - Simple `invoke()` call to register agents  

- ✅ **Extreme Error Handling**: Comprehensive try/except blocks with detailed logging✅ **Custom Seed Support** - Provide your own seed for stable agent addresses  

- ✅ **Inline Documentation**: Extensive docstrings following Google style✅ **Custom Protocols** - Add your own protocols like MCP adapter  

- ✅ **Multiple Protocols**: Query protocol + Chat protocol✅ **Automatic Agentverse Integration** - Auto-registration and README updates  

- ✅ **Custom Protocol Support**: Easily add your own protocols✅ **Source Attribution** - Includes document sources in responses  

- ✅ **Source Attribution**: Automatic source document references✅ **Flexible Configuration** - Control mailbox, endpoint, and more  

- ✅ **Thread-Safe**: Safe for concurrent message processing

- ✅ **Production-Ready**: Logging, error recovery, graceful degradation## Installation



## 📦 Installation```bash

pip install uagents uagents-core llama-index llama-index-llms-openai llama-index-embeddings-openai

```bash```

pip install uagents uagents-core llama-index llama-index-llms-openai llama-index-embeddings-openai

```## Quick Start



## 🚀 Quick Start### Basic Usage (Auto Seed)



```python```python

from llama_index.core import VectorStoreIndex, SimpleDirectoryReaderfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from uagents import Agentfrom uagents_adapter.llamaindex import LlamaIndexRegisterTool

from uagents_adapter.llamaindex import LlamaIndexAdapter

# Create query engine

# 1. Create LlamaIndex query enginedocuments = SimpleDirectoryReader('data').load_data()

documents = SimpleDirectoryReader('data').load_data()index = VectorStoreIndex.from_documents(documents)

index = VectorStoreIndex.from_documents(documents)query_engine = index.as_query_engine()

query_engine = index.as_query_engine()

# Register as uAgent

# 2. Create adapter (creates protocols only)tool = LlamaIndexRegisterTool()

adapter = LlamaIndexAdapter(query_engine)agent_info = tool.invoke({

    "query_engine": query_engine,

# 3. Create agent (FULL USER CONTROL)    "name": "doc_qa_agent",

agent = Agent(    "port": 8000,

    name="My LlamaIndex Agent",    "description": "Document Q&A agent",

    seed="my_deterministic_seed",  # Same seed = same address    "api_token": "your_agentverse_api_key"

    port=8001,})

    mailbox=True

)print(agent_info)

```

# 4. Include protocols

for protocol in adapter.protocols:### With Custom Seed (Stable Address)

    agent.include(protocol, publish_manifest=True)

```python

# 5. Run agentagent_info = tool.invoke({

agent.run()    "query_engine": query_engine,

```    "name": "stable_agent",

    "port": 8000,

## 📖 Architecture    "description": "Agent with stable address",

    "api_token": "your_agentverse_api_key",

### Design Pattern: MCP Style    "seed": "my_custom_seed_phrase_12345"  # Same seed = same address

})

``````

User Code                    Adapter                    uAgent

─────────                    ───────                    ──────### With Custom Protocols (MCP-style Flexibility)

                                                        

1. Create query_engine  →                               ```python

                                                        from uagents import Protocol, Context, Model

2. Create adapter       →    Creates protocols          

                            - Query protocol           # Define custom protocol

                            - Chat protocol            class HealthCheck(Model):

                                                            status: str

3. Create agent         →                         →    Agent()

                                                        health_proto = Protocol(name="HealthProtocol")

4. Include protocols    →    Protocols provided   →    agent.include()

                                                        @health_proto.on_message(model=HealthCheck)

5. Run agent            →                         →    agent.run()async def handle_health(ctx: Context, sender: str, msg: HealthCheck):

```    await ctx.send(sender, HealthCheck(status="healthy"))



### User Responsibilities# Register with custom protocol

agent_info = tool.invoke({

- ✅ Create and configure Agent    "query_engine": query_engine,

- ✅ Choose seed (for deterministic address)    "name": "custom_agent",

- ✅ Choose port and mailbox settings    "port": 8000,

- ✅ Include protocols (all or selective)    "description": "Agent with custom protocols",

- ✅ Run agent when ready    "api_token": "your_agentverse_api_key",

- ✅ Add custom protocols (optional)    "seed": "custom_seed",

    "custom_protocols": [health_proto]  # Add your protocols!

### Adapter Responsibilities})

```

- ✅ Create protocols with handlers

- ✅ Handle query routing to LlamaIndex## API Reference

- ✅ Extract and format responses

- ✅ Handle all errors gracefully### `LlamaIndexRegisterTool.invoke()`

- ✅ Log operations with context

- ✅ Send error messages to users**Parameters:**



## 🔧 API Reference| Parameter | Type | Required | Default | Description |

|-----------|------|----------|---------|-------------|

### LlamaIndexAdapter| `query_engine` | Any | ✅ Yes | - | LlamaIndex query engine from `index.as_query_engine()` |

| `name` | str | ✅ Yes | - | Name for the uAgent |

```python| `port` | int | ✅ Yes | - | Port for the agent to run on |

class LlamaIndexAdapter:| `description` | str | ✅ Yes | - | Description of the agent's purpose |

    """Main adapter class that creates protocols for LlamaIndex integration."""| `api_token` | str | ❌ No | None | Agentverse API token for registration |

    | `seed` | str | ❌ No | Auto-generated | Custom seed for stable agent address |

    def __init__(self, query_engine: Any):| `custom_protocols` | list[Protocol] | ❌ No | None | List of custom protocols to include |

        """| `mailbox` | bool | ❌ No | True | Use Agentverse mailbox (True) or HTTP endpoint (False) |

        Initialize adapter with LlamaIndex query engine.| `return_dict` | bool | ❌ No | False | Return dict (True) or formatted string (False) |

        | `ai_agent_address` | str | ❌ No | None | Optional AI agent address for message forwarding |

        Args:

            query_engine: LlamaIndex query engine from index.as_query_engine()**Returns:**

            

        Raises:- If `return_dict=False`: Formatted string with agent info

            TypeError: If query_engine is None- If `return_dict=True`: Dictionary with detailed agent information

            AttributeError: If query_engine lacks required methods

        """```python

```{

    "name": "agent_name",

#### Properties    "address": "agent1q...",

    "port": 8000,

| Property | Type | Description |    "seed": "custom_seed",

|----------|------|-------------|    "mailbox": True,

| `protocols` | `list[Protocol]` | All protocols (query + chat) |    "query_engine": <query_engine_object>,

| `protocol` | `Protocol` | Query protocol (backward compat) |    "custom_protocols": [<protocol_objects>],

| `query_protocol` | `Protocol` | Query protocol explicitly |    "uagent": <agent_object>,

| `chat_protocol` | `Protocol` | Chat protocol explicitly |    "thread": <thread_object>

}

#### Example Usage```



```python## Usage Patterns

# Create adapter

adapter = LlamaIndexAdapter(query_engine)### Pattern 1: Simple (LangChain-style)



# Access all protocols```python

for protocol in adapter.protocols:# One call, auto-generated seed, minimal configuration

    agent.include(protocol)tool = LlamaIndexRegisterTool()

result = tool.invoke({

# Or access individually    "query_engine": query_engine,

agent.include(adapter.query_protocol)    "name": "simple_agent",

agent.include(adapter.chat_protocol)    "port": 8000,

```    "description": "Simple Q&A",

    "api_token": API_KEY

### Message Models})

```

#### QueryIndex

**Use when:** Quick prototyping, don't need stable address

```python

class QueryIndex(Model):---

    """Direct query message."""

    query: str  # Natural language query### Pattern 2: Custom Seed

```

```python

**Usage:**# Same seed = same agent address every time

```pythonresult = tool.invoke({

from uagents_adapter.llamaindex import QueryIndex    "query_engine": query_engine,

    "name": "stable_agent",

await ctx.send(    "port": 8000,

    agent_address,    "description": "Stable address agent",

    QueryIndex(query="What is the main topic?")    "api_token": API_KEY,

)    "seed": "my_stable_seed_12345"  # Custom seed

```})

```

#### QueryIndexResponse

**Use when:** Need consistent agent address across restarts

```python

class QueryIndexResponse(Model):---

    """Query response with results and sources."""

    result: str                    # Query result or error message### Pattern 3: Custom Protocols

    sources: Optional[list[str]]   # Source document excerpts

    error: Optional[str]           # Error message if failed```python

```# Add your own message handlers

custom_proto = Protocol(name="CustomProtocol")

**Example Response:**

```python@custom_proto.on_message(model=CustomMessage)

QueryIndexResponse(async def handler(ctx, sender, msg):

    result="The main topic is...",    # Your custom logic

    sources=["Source 1...", "Source 2..."],    pass

    error=None  # None on success

)result = tool.invoke({

```    "query_engine": query_engine,

    "name": "custom_agent",

### Error Handling    "port": 8000,

    "description": "Agent with custom features",

#### Custom Exceptions    "api_token": API_KEY,

    "custom_protocols": [custom_proto]

```python})

class LlamaIndexError(Exception):```

    """Base exception for all adapter errors."""

**Use when:** Need additional functionality beyond Q&A

class QueryEngineError(LlamaIndexError):

    """Raised when query engine operations fail."""---

    

    def __init__(### Pattern 4: Full Control

        self,

        message: str,```python

        original_error: Optional[Exception] = None,# Everything customized

        query: Optional[str] = Noneresult = tool.invoke({

    ):    "query_engine": query_engine,

        """    "name": "full_control_agent",

        Args:    "port": 8000,

            message: Error description    "description": "Fully customized",

            original_error: Underlying exception    "api_token": API_KEY,

            query: Query that caused error    "seed": "my_seed",

        """    "custom_protocols": [proto1, proto2],

```    "mailbox": False,  # Use HTTP endpoint

    "return_dict": True  # Get detailed info

#### Error Handling Strategy})

```

**All operations wrapped in try/except:**

**Use when:** Need complete control over configuration

1. **Query Execution**

   ```python## Message Handling

   try:

       response = query_engine.query(query_text)The agent automatically handles two types of messages:

   except Exception as e:

       # Log with context### 1. QueryMessage (Direct Queries)

       # Send user-friendly error message

       # Don't crash agent```python

   ```from uagents import Model



2. **Response Extraction**class QueryMessage(Model):

   ```python    query: str

   try:

       result = extract_response_text(response)# Send to agent address

   except QueryEngineError:await ctx.send(agent_address, QueryMessage(query="What is X?"))

       # Handle missing attributes```

       # Return error to sender

   ```### 2. ChatMessage (Agentverse Chat)



3. **Message Sending**Send via Agentverse UI or programmatically:

   ```python

   try:```python

       await ctx.send(sender, response)from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

   except Exception:

       # Log send failureawait ctx.send(

       # Don't crash handler    agent_address,

   ```    ChatMessage(

        timestamp=datetime.utcnow(),

## 📚 Usage Examples        msg_id=uuid4(),

        content=[TextContent(type="text", text="What is X?")]

### Example 1: Basic Usage    )

)

```python```

"""Minimal setup for document Q&A."""

## Response Format

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from uagents import AgentResponses include:

from uagents_adapter.llamaindex import LlamaIndexAdapter- **Answer:** The query result from LlamaIndex

- **Sources:** Top document sources (if available)

# Create index

documents = SimpleDirectoryReader('data').load_data()Example response:

index = VectorStoreIndex.from_documents(documents)```

query_engine = index.as_query_engine()The main topic is artificial intelligence and machine learning.



# Create adapter📚 Sources:

adapter = LlamaIndexAdapter(query_engine)1. Introduction to AI systems that can learn from data...

2. Machine learning algorithms enable computers to...

# Create agent```

agent = Agent(

    name="Doc Agent",## Comparison: LangChain vs LlamaIndex Adapter

    seed="my_seed_v1",

    port=8001,| Feature | LangChain Adapter | LlamaIndex Adapter |

    mailbox=True|---------|-------------------|-------------------|

)| **Input** | LangChain agent | LlamaIndex query engine |

| **Seed Control** | ❌ Auto-generated only | ✅ Auto or custom |

# Include protocols| **Custom Protocols** | ❌ Not supported | ✅ Supported |

for protocol in adapter.protocols:| **API Style** | `invoke()` | `invoke()` |

    agent.include(protocol, publish_manifest=True)| **Agentverse Integration** | ✅ Automatic | ✅ Automatic |

| **Flexibility** | Low (fixed 2 protocols) | High (add any protocols) |

# Run

agent.run()## Best Practices

```

### 1. Use Custom Seeds for Production

### Example 2: With Custom Protocols

```python

```python# ❌ Don't (address changes on restart)

"""Add custom health check protocol."""tool.invoke({...})  # Auto-generated seed



from uagents import Protocol, Model, Context# ✅ Do (stable address)

from uagents_adapter.llamaindex import LlamaIndexAdaptertool.invoke({

    ...,

# Define custom protocol    "seed": "production_seed_v1"

class HealthCheck(Model):})

    pass```



class HealthStatus(Model):### 2. Add Health Check Protocol

    status: str

```python

health_proto = Protocol(name="HealthProtocol")class HealthCheck(Model):

    timestamp: str

@health_proto.on_message(model=HealthCheck)

async def handle_health(ctx: Context, sender: str, msg: HealthCheck):health_proto = Protocol(name="HealthProtocol")

    await ctx.send(sender, HealthStatus(status="healthy"))

@health_proto.on_message(model=HealthCheck)

# Create adapterasync def health(ctx, sender, msg):

adapter = LlamaIndexAdapter(query_engine)    await ctx.send(sender, HealthCheck(timestamp=str(datetime.utcnow())))



# Create agenttool.invoke({

agent = Agent(name="Agent", seed="seed", port=8001)    ...,

    "custom_protocols": [health_proto]

# Include LlamaIndex protocols})

for protocol in adapter.protocols:```

    agent.include(protocol)

### 3. Use `return_dict=True` for Programmatic Access

# Include custom protocol

agent.include(health_proto)```python

agent_info = tool.invoke({

# Run    ...,

agent.run()    "return_dict": True

```})



### Example 3: Custom Query Engine Configuration# Access specific info

agent_address = agent_info["address"]

```pythonagent_port = agent_info["port"]

"""Advanced query engine configuration."""```



from llama_index.core import VectorStoreIndex## Advanced Example



# Create index```python

index = VectorStoreIndex.from_documents(documents)from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from llama_index.llms.openai import OpenAI

# Configure query enginefrom uagents import Protocol, Context, Model

query_engine = index.as_query_engine(from uagents_adapter.llamaindex import LlamaIndexRegisterTool

    similarity_top_k=5,           # Top 5 results

    response_mode="tree_summarize",  # Summarization mode# Configure LlamaIndex

    verbose=True                   # Enable verbose loggingSettings.llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

)

# Create custom protocols

# Create adapterclass Stats(Model):

adapter = LlamaIndexAdapter(query_engine)    queries_count: int



# Rest is the same...stats_proto = Protocol(name="StatsProtocol")

```

@stats_proto.on_message(model=Stats)

### Example 4: Selective Protocol Inclusionasync def handle_stats(ctx: Context, sender: str, msg: Stats):

    # Return query statistics

```python    await ctx.send(sender, Stats(queries_count=42))

"""Include only specific protocols."""

# Create query engine

from uagents_adapter.llamaindex import LlamaIndexAdapterdocuments = SimpleDirectoryReader('docs').load_data()

index = VectorStoreIndex.from_documents(documents)

adapter = LlamaIndexAdapter(query_engine)query_engine = index.as_query_engine(similarity_top_k=5)



# Only include query protocol (no chat)# Register with everything

agent.include(adapter.query_protocol, publish_manifest=True)tool = LlamaIndexRegisterTool()

agent_info = tool.invoke({

# Or only include chat protocol    "query_engine": query_engine,

agent.include(adapter.chat_protocol, publish_manifest=True)    "name": "production_agent",

    "port": 8000,

agent.run()    "description": "Production document Q&A with stats",

```    "api_token": AGENTVERSE_API_KEY,

    "seed": "production_seed_v1_2024",

## 🔍 Protocol Details    "custom_protocols": [stats_proto],

    "mailbox": True,

### Query Protocol    "return_dict": True

})

**Handles:** `QueryIndex` messages

print(f"Agent live at: {agent_info['address']}")

**Flow:**```

1. Receive `QueryIndex` message

2. Validate query (not empty)## Troubleshooting

3. Execute via LlamaIndex

4. Extract result and sources### Port Already in Use

5. Send `QueryIndexResponse`

```python

**Error Handling:**# The adapter automatically finds available port

- Empty query → Error responseagent_info = tool.invoke({

- Query engine failure → Error logged, response sent    ...,

- Response extraction failure → Error response    "port": 8000  # Will use 8001 if 8000 is busy

- Send failure → Logged, doesn't crash})

```

### Chat Protocol

### Agent Not Appearing on Agentverse

**Handles:** `ChatMessage` interactions

1. Check API token is correct

**Flow:**2. Ensure `mailbox=True` (required for Agentverse)

1. Receive `ChatMessage`3. Wait 10-15 seconds for registration

2. Send `ChatAcknowledgement`

3. Process each content item:### Custom Protocol Not Working

   - `StartSessionContent` → Log

   - `TextContent` → Query LlamaIndex```python

   - `EndSessionContent` → Log# ✅ Correct: Pass as list

4. Format response with sources"custom_protocols": [proto1, proto2]

5. Send `ChatMessage` response

# ❌ Wrong: Don't pass single protocol

**Error Handling:**"custom_protocols": proto1  # This won't work

- Acknowledgement failure → Logged, continue```

- Query failure → Error message sent

- Response send failure → Logged## Examples

- Unknown content types → Logged, skipped

See the `examples/` directory:

## 🏗️ Production Patterns- `llamaindex_basic.py` - Simple usage

- `llamaindex_advanced.py` - Custom seed and protocols

### Pattern 1: Health Monitoring- `llamaindex_patterns.py` - All patterns comparison



```python## License

"""Add health check for monitoring."""

Same as uagents_adapter package.

class HealthCheck(Model):
    pass

class HealthStatus(Model):
    status: str
    uptime_seconds: float

health_proto = Protocol(name="Health")
start_time = datetime.utcnow()

@health_proto.on_message(model=HealthCheck)
async def health(ctx, sender, msg):
    uptime = (datetime.utcnow() - start_time).total_seconds()
    await ctx.send(
        sender,
        HealthStatus(status="healthy", uptime_seconds=uptime)
    )

# Include with agent
agent.include(health_proto)
```

### Pattern 2: Statistics Tracking

```python
"""Track query statistics."""

class StatsTracker:
    def __init__(self):
        self.queries = 0
        self.errors = 0
    
    def record_query(self):
        self.queries += 1
    
    def record_error(self):
        self.errors += 1

tracker = StatsTracker()

# Wrap adapter methods to track stats
# (See advanced example for full implementation)
```

### Pattern 3: Graceful Shutdown

```python
"""Handle shutdown gracefully."""

import signal
import sys

def signal_handler(sig, frame):
    print("\n\nShutting down gracefully...")
    # Cleanup code here
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

agent.run()
```

## 🐛 Troubleshooting

### Issue: Import Error

```python
# Error:
ImportError: cannot import name 'LlamaIndexAdapter'

# Solution:
pip install --upgrade uagents-adapter
# Or check your import:
from uagents_adapter.llamaindex import LlamaIndexAdapter
```

### Issue: Query Engine Has No 'query' Method

```python
# Error:
AttributeError: 'X' object has no attribute 'query'

# Solution:
# Ensure you create query engine correctly:
query_engine = index.as_query_engine()  # ✅ Correct
query_engine = index                     # ❌ Wrong
```

### Issue: Agent Address Changes on Restart

```python
# Problem: New address every restart

# Solution: Use fixed seed
agent = Agent(
    name="agent",
    seed="my_fixed_seed_v1",  # ✅ Same seed = same address
    port=8001
)
```

### Issue: No Sources in Response

```python
# This is normal if:
# 1. Source nodes not available in response
# 2. Query engine config doesn't include sources
# 3. Documents don't have extractable text

# To enable sources:
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Ensure k > 0
    response_mode="compact"  # Use mode that includes sources
)
```

## 📊 Comparison: LangChain vs LlamaIndex

| Feature | LangChain Adapter | LlamaIndex Adapter |
|---------|-------------------|-------------------|
| **Pattern** | Tool (`invoke()`) | MCP (manual agent) |
| **Agent Creation** | Tool creates | User creates |
| **Seed Control** | ❌ Auto-generated | ✅ User provides |
| **Custom Protocols** | ❌ Fixed 2 | ✅ User adds any |
| **Agentverse** | ✅ Auto | Optional (user choice) |
| **Control** | Low (tool decides) | High (user decides) |
| **Flexibility** | Low | High |
| **Simplicity** | High (1 call) | Medium (5 steps) |
| **Best For** | Quick prototyping | Production use |

## 🎯 Best Practices

### 1. Use Deterministic Seeds in Production

```python
# ❌ Bad (new address every restart)
agent = Agent(name="agent", port=8001)

# ✅ Good (stable address)
agent = Agent(
    name="agent",
    seed="production_v1_20250122",
    port=8001
)
```

### 2. Always Include Both Protocols

```python
# ✅ Include all protocols
for protocol in adapter.protocols:
    agent.include(protocol, publish_manifest=True)
```

### 3. Handle Startup Errors

```python
try:
    agent.run()
except Exception as e:
    logger.error(f"Agent failed: {str(e)}")
    # Cleanup, notify monitoring, etc.
```

### 4. Use Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 5. Add Custom Protocols for Production

- Health checks
- Statistics
- Admin commands
- Cache management

## 📝 Examples Directory

- `llamaindex_basic_example.py` - Basic usage with full documentation
- `llamaindex_advanced_example.py` - Custom protocols + error handling

## 🤝 Contributing

When contributing:
1. Follow existing error handling patterns
2. Add comprehensive docstrings
3. Include logging statements
4. Add examples for new features
5. Update documentation

## 📄 License

Same as uagents-adapter package.

## 🆘 Support

- Issues: GitHub Issues
- Documentation: This README
- Examples: `examples/` directory
- Code: Fully documented inline

## ✨ Summary

The LlamaIndex adapter provides:
- ✅ Production-ready error handling
- ✅ Comprehensive inline documentation
- ✅ MCP pattern for maximum control
- ✅ Easy custom protocol integration
- ✅ Thread-safe concurrent processing
- ✅ Source attribution
- ✅ Flexible and extensible

**Perfect for production LlamaIndex deployments!** 🚀
