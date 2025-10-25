"""Adapters for uAgents to integrate with LangChain, LlamaIndex, CrewAI, and MCP."""

from importlib import metadata

# Always import these - they are core
from .common import ResponseMessage, cleanup_all_uagents, cleanup_uagent
from .llamaindex import LlamaIndexAdapter

# Lazy imports - only load when accessed
def __getattr__(name):
    """Lazy import optional modules only when accessed."""
    if name == "CrewaiRegisterTool":
        try:
            from .crewai import CrewaiRegisterTool
            return CrewaiRegisterTool
        except (ImportError, TypeError) as e:
            raise ImportError(f"CrewAI adapter not available: {e}")
    
    elif name == "LangchainRegisterTool":
        try:
            from .langchain import LangchainRegisterTool
            return LangchainRegisterTool
        except ImportError as e:
            raise ImportError(f"LangChain adapter not available: {e}")
    
    elif name == "MCPServerAdapter":
        try:
            from .mcp import MCPServerAdapter
            return MCPServerAdapter
        except ImportError as e:
            raise ImportError(f"MCP adapter not available: {e}")
    
    elif name == "A2ARegisterTool":
        try:
            from .a2a_inbound import A2ARegisterTool
            return A2ARegisterTool
        except ImportError as e:
            raise ImportError(f"A2A Inbound adapter not available: {e}")
    
    elif name in ["A2AAgentConfig", "MultiA2AAdapter", "SingleA2AAdapter", "a2a_servers"]:
        try:
            from .a2a_outbound import (
                A2AAgentConfig,
                MultiA2AAdapter,
                SingleA2AAdapter,
                a2a_servers,
            )
            return locals()[name]
        except ImportError as e:
            raise ImportError(f"A2A Outbound adapter not available: {e}")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


# Core exports
__all__ = [
    "LlamaIndexAdapter",
    "ResponseMessage",
    "cleanup_uagent",
    "cleanup_all_uagents",
    "__version__",
    # Optional modules (lazy loaded)
    "CrewaiRegisterTool",
    "LangchainRegisterTool",
    "MCPServerAdapter",
    "A2ARegisterTool",
    "A2AAgentConfig",
    "MultiA2AAdapter",
    "SingleA2AAdapter",
    "a2a_servers",
]
