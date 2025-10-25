"""LlamaIndex adapter for uAgents - MCP-style pattern.

Simple adapter to wrap LlamaIndex agents/workflows/engines in uAgents.

Example:
    >>> from uagents import Agent
    >>> from uagents_adapter.llamaindex import LlamaIndexAdapter
    >>> 
    >>> # Your LlamaIndex agent/workflow/engine
    >>> agent_ll = create_your_llamaindex_agent()
    >>> 
    >>> # Create adapter
    >>> adapter = LlamaIndexAdapter(agent_ll)
    >>> 
    >>> # Create uAgent
    >>> agent = Agent(name="My Agent", seed="my_seed", port=8001, mailbox=True)
    >>> 
    >>> # Include protocol (NO try/catch needed!)
    >>> agent.include(adapter.protocol, publish_manifest=True)
    >>> 
    >>> # Run
    >>> agent.run()
"""

from .adapter import LlamaIndexAdapter

__all__ = ["LlamaIndexAdapter"]
