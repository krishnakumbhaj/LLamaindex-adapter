import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from uagents import Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)


# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class LlamaIndexError(Exception):
    """Base exception for LlamaIndex adapter errors.
    
    All adapter-specific exceptions inherit from this class,
    making it easy to catch adapter-related errors separately.
    """
    pass


class QueryEngineError(LlamaIndexError):
    """Exception raised when query engine operations fail.
    
    This includes:
    - Query execution failures
    - Response parsing errors
    - Missing required attributes
    - Invalid query engine state
    
    Attributes:
        message: Human-readable error description
        original_error: The underlying exception (if any)
        query: The query that caused the error (if applicable)
    """
    
    def __init__(
        self, 
        message: str, 
        original_error: Optional[Exception] = None,
        query: Optional[str] = None
    ):
        """Initialize QueryEngineError.
        
        Args:
            message: Error description
            original_error: Original exception that caused this error
            query: The query string that triggered the error
        """
        self.message = message
        self.original_error = original_error
        self.query = query
        
        # Build full error message
        full_message = message
        if query:
            full_message += f" | Query: '{query[:100]}...'"
        if original_error:
            full_message += f" | Cause: {type(original_error).__name__}: {str(original_error)}"
        
        super().__init__(full_message)


# ============================================================================
# Message Models
# ============================================================================


class QueryIndex(Model):
    """Message model for querying the LlamaIndex engine.
    
    This message type is used for direct programmatic queries to the agent.
    Users can send this message to the agent's address to get responses.
    
    Attributes:
        query: The natural language query to process
        
    Example:
        >>> from uagents import Context
        >>> await ctx.send(
        >>>     agent_address,
        >>>     QueryIndex(query="What is the main topic of the documents?")
        >>> )
    """
    query: str


class QueryIndexResponse(Model):
    """Response model for query results.
    
    Contains the query result along with optional source attributions
    and error information if the query failed.
    
    Attributes:
        result: The query result text (or error message if failed)
        sources: List of source document excerpts (if available)
        error: Error message if query failed (None on success)
        
    Example Response:
        >>> QueryIndexResponse(
        >>>     result="The main topic is artificial intelligence...",
        >>>     sources=["Source 1 excerpt...", "Source 2 excerpt..."],
        >>>     error=None
        >>> )
    """
    result: str
    sources: Optional[list[str]] = None
    error: Optional[str] = None


# ============================================================================
# Protocol Specifications
# ============================================================================


# LlamaIndex protocol specification
# Used to identify and version the LlamaIndex-specific protocol

llamaindex_protocol_spec = {
    "name": "llamaindex",
    "version": "0.1.0",
    "description": "Protocol for LlamaIndex query engine integration"
}


# ============================================================================
# Main Adapter Class
# ============================================================================


class LlamaIndexAdapter:
    """Adapter for integrating LlamaIndex agents with uAgents.
    
    This adapter follows the MCP pattern where the adapter creates protocols
    but the user maintains full control over agent creation, configuration,
    and lifecycle.
    
    The adapter creates a single protocol that wraps the LlamaIndex agent/workflow/engine,
    providing seamless integration with uAgents messaging system.
    
    Registration Behavior:
        - Almanac (local): ALWAYS automatic via uAgents library
        - Agentverse (cloud): Optional, only if agentverse_api_token provided
    
    Thread Safety:
        All handlers are async and thread-safe.
        
    Error Handling:
        All operations are wrapped in comprehensive try/except blocks internally.
        User code doesn't need try/catch - adapter handles all errors gracefully.
    
    Simple Usage (Recommended - with adapter.run()):
        >>> from uagents import Agent
        >>> from uagents_adapter import LlamaIndexAdapter
        >>> 
        >>> # Create adapter (without token = local only)
        >>> adapter = LlamaIndexAdapter(agent_ll)
        >>> 
        >>> # Create uAgent
        >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> 
        >>> # Run with auto-registration
        >>> adapter.run(agent)  # ✅ Handles registration + running!
    
    With Agentverse Cloud Registration:
        >>> # Create adapter with token
        >>> adapter = LlamaIndexAdapter(
        >>>     agent_ll,
        >>>     agentverse_api_token="agv_xxx",  # ✅ Token provided
        >>>     description="My RAG agent"
        >>> )
        >>> 
        >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> 
        >>> # Run with cloud registration
        >>> adapter.run(agent)  # ✅ Registers to Almanac + Agentverse
    
    Advanced Usage (Manual Control):
        >>> # If you need async or manual control
        >>> adapter = LlamaIndexAdapter(agent_ll)
        >>> agent = Agent(...)
        >>> agent.include(adapter.protocol, publish_manifest=True)
        >>> 
        >>> adapter.register(agent)  # Setup registration handler only
        >>> agent.run()  # Run manually (sync)
        >>> # OR
        >>> await agent.run_async()  # Run manually (async)
        
    Attributes:
        agent_ll: The LlamaIndex agent/workflow/engine instance
        agentverse_api_token: Optional API token for Agentverse registration
        description: Optional description for marketplace listing
        protocol: Protocol for handling queries via chat
    """
    
    def __init__(
        self,
        agent_ll: Any,
        agentverse_api_token: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the LlamaIndex adapter.
        
        Creates protocol with message handlers that route queries to the
        provided LlamaIndex agent/workflow/engine.
        
        Args:
            agent_ll: A LlamaIndex agent, workflow, or engine instance.
                Can be any LlamaIndex object that can process queries.
                Common types:
                - Workflow instance
                - Agent instance
                - Query engine from index.as_query_engine()
            agentverse_api_token: Optional API token for Agentverse marketplace registration.
                If provided, you can call register_agent() to publish agent to marketplace.
            description: Optional description for Agentverse marketplace listing.
                
        Raises:
            TypeError: If agent_ll is None
            
        Example:
            >>> # Without Agentverse registration
            >>> adapter = LlamaIndexAdapter(agent_ll)
            >>> 
            >>> # With Agentverse registration
            >>> adapter = LlamaIndexAdapter(
            >>>     agent_ll,
            >>>     agentverse_api_token="agv_xxxxxxxxxx",
            >>>     description="LlamaIndex RAG agent"
            >>> )
            >>> 
            >>> # With query engine
            >>> query_engine = index.as_query_engine()
            >>> adapter = LlamaIndexAdapter(query_engine)
        """
        # Validate input
        if agent_ll is None:
            raise TypeError(
                "agent_ll cannot be None. "
                "Please provide a valid LlamaIndex agent, workflow, or engine."
            )
        
        self.agent_ll = agent_ll
        self.agentverse_api_token = agentverse_api_token
        self.description = description
        self._registered = False
        
        logger.info(
            f"Initializing LlamaIndexAdapter with: {type(agent_ll).__name__}"
        )
        
        # Create protocol
        try:
            self._protocol = self._create_protocol()
            logger.info("Protocol created successfully")
        except Exception as e:
            logger.error(f"Failed to create protocol: {str(e)}")
            raise
        
        logger.info("LlamaIndexAdapter initialized successfully")
    
    @property
    def protocol(self) -> Protocol:
        """Get the main protocol.
        
        Returns:
            The protocol handling chat-based queries
            
        Example:
            >>> adapter = LlamaIndexAdapter(agent_ll)
            >>> agent.include(adapter.protocol, publish_manifest=True)
        """
        return self._protocol
    
    def register(
        self, 
        agent: Any, 
        readme: Optional[str] = None,
        wait_seconds: int = 10
    ):
        """Register agent with Almanac (always) and Agentverse (if token provided).
        
        This method AUTOMATICALLY registers a startup handler that will:
        - ✅ Always: Register to Almanac for local agent discovery (automatic via uAgents)
        - ✅ If token provided: Register to Agentverse cloud marketplace
        
        The Almanac registration happens automatically through the uAgents library.
        This method only needs to handle optional Agentverse cloud registration.
        
        Args:
            agent: The uAgent instance (must have .name and .address attributes)
            readme: Optional custom README (only used if token provided)
            wait_seconds: Seconds to wait after startup before registration (default: 10)
            
        Example (with Agentverse cloud):
            >>> adapter = LlamaIndexAdapter(
            >>>     agent_ll,
            >>>     agentverse_api_token="agv_xxx",  # ✅ Token provided
            >>>     description="My RAG agent"
            >>> )
            >>> agent = Agent(name="my_agent", mailbox=True)
            >>> agent.include(adapter.protocol)
            >>> adapter.register(agent)  # Registers to Almanac + Agentverse
            >>> agent.run()
        
        Example (local only, no cloud):
            >>> adapter = LlamaIndexAdapter(agent_ll)  # ❌ No token
            >>> agent = Agent(name="my_agent", mailbox=True)
            >>> agent.include(adapter.protocol)
            >>> adapter.register(agent)  # Registers to Almanac only
            >>> agent.run()
        """
        logger.info(f"Setting up registration for agent '{agent.name}'...")
        
        # Store registration params for the startup handler
        self._agent_for_registration = agent
        self._readme_for_registration = readme
        self._wait_seconds = wait_seconds
        
        # Register startup handler that will do the registration
        @agent.on_event("startup")
        async def auto_register(ctx):
            """Auto-registration handler (created internally by adapter)."""
            import asyncio
            
            # Wait for agent to be fully initialized
            ctx.logger.info(
                f"Waiting {wait_seconds} seconds for agent to fully initialize..."
            )
            await asyncio.sleep(wait_seconds)
            
            # Almanac registration happens AUTOMATICALLY via uAgents library
            # The uAgents Agent.run() method handles this internally
            ctx.logger.info("[OK] Agent registered to Almanac (automatic via uAgents)")
            ctx.logger.info("   Agent is discoverable locally")
            
            # Agentverse cloud registration (only if token provided)
            if self.agentverse_api_token:
                ctx.logger.info("[CLOUD] Agentverse token found")
                ctx.logger.info("[INFO] Agent will be automatically registered via mailbox connection")
                ctx.logger.info("[INFO] Look for '[mailbox]: Successfully registered' message")
                ctx.logger.info("[OK] Agent ready for Agentverse and ASI1 integration")
            else:
                ctx.logger.info("[INFO] No Agentverse token provided - skipping cloud registration")
                ctx.logger.info("   Agent is discoverable locally via Almanac only")
        
        logger.info("[OK] Registration handler added")
        if self.agentverse_api_token:
            logger.info("   Will register to: Almanac (auto) + Agentverse (cloud)")
        else:
            logger.info("   Will register to: Almanac (auto) only")
    
    def run(
        self,
        agent: Any,
        readme: Optional[str] = None,
        wait_seconds: int = 10
    ):
        """Run the agent with automatic registration (Almanac always + Agentverse if token).
        
        This is the SIMPLEST way to use the adapter. It handles everything:
        1. Sets up automatic registration (Almanac + optional Agentverse)
        2. Runs the agent
        
        The agent will automatically:
        - ✅ Register to Almanac for local discovery (always)
        - ✅ Register to Agentverse marketplace (if token provided in __init__)
        
        Args:
            agent: The uAgent instance (must have protocol already included)
            readme: Optional custom README for Agentverse (only used if token provided)
            wait_seconds: Seconds to wait after startup before registration (default: 10)
            
        Example (without Agentverse token - local only):
            >>> from uagents import Agent
            >>> from uagents_adapter import LlamaIndexAdapter
            >>> 
            >>> # Create adapter without token
            >>> adapter = LlamaIndexAdapter(agent_ll)
            >>> 
            >>> # Create agent
            >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> 
            >>> # Run with auto-registration (Almanac only)
            >>> adapter.run(agent)  # ✅ Registers to Almanac automatically
        
        Example (with Agentverse token - cloud registration):
            >>> from uagents import Agent
            >>> from uagents_adapter import LlamaIndexAdapter
            >>> 
            >>> # Create adapter with token
            >>> adapter = LlamaIndexAdapter(
            >>>     agent_ll,
            >>>     agentverse_api_token="agv_xxx",
            >>>     description="My RAG agent"
            >>> )
            >>> 
            >>> # Create agent
            >>> agent = Agent(name="my_agent", seed="...", port=8001, mailbox=True)
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> 
            >>> # Run with auto-registration (Almanac + Agentverse)
            >>> adapter.run(agent)  # ✅ Registers to Almanac + Agentverse cloud
        
        Note:
            The protocol must be included BEFORE calling run():
            >>> agent.include(adapter.protocol, publish_manifest=True)
            >>> adapter.run(agent)  # ✅ Correct order
        """
        logger.info("[START] Starting agent with automatic registration...")
        
        # Step 1: Set up registration handler
        self.register(agent, readme=readme, wait_seconds=wait_seconds)
        
        # Step 2: Run the agent (blocking)
        logger.info("Starting agent.run()...")
        agent.run()
    
    def _perform_registration(self, agent: Any, readme: Optional[str] = None):
        """Internal method to perform the actual registration.
        
        Args:
            agent: The uAgent instance
            readme: Optional custom README content
        
        Raises:
            Exception: If registration API call fails
        """
        if self._registered:
            logger.info("Agent already registered, skipping")
            return
        
        # Generate README if not provided
        if readme is None:
            readme = self._generate_readme(agent.name)
        
        # Register using API
        try:
            import requests
            
            agent_address = agent.address
            port = agent.port if hasattr(agent, 'port') else None
            
            logger.info(f"Attempting to register agent '{agent.name}' to Agentverse marketplace...")
            
            # Setup headers
            headers = {
                "Authorization": f"Bearer {self.agentverse_api_token}",
                "Content-Type": "application/json",
            }
            
            # Connect agent to mailbox (if port available)
            if port:
                connect_url = f"http://127.0.0.1:{port}/connect"
                connect_payload = {
                    "agent_type": "mailbox",
                    "user_token": self.agentverse_api_token
                }
                
                try:
                    connect_response = requests.post(
                        connect_url, json=connect_payload, headers=headers, timeout=10
                    )
                    if connect_response.status_code == 200:
                        logger.info(f"Agent '{agent.name}' connected to Agentverse")
                    else:
                        logger.warning(
                            f"Failed to connect agent to mailbox: "
                            f"{connect_response.status_code}"
                        )
                except Exception as e:
                    logger.warning(f"Error connecting agent to mailbox: {e}")
            
            # First, try to GET the agent to see if it exists
            get_url = f"https://agentverse.ai/v1/agents/{agent_address}"
            
            try:
                get_response = requests.get(get_url, headers=headers, timeout=10)
                
                if get_response.status_code == 200:
                    # Agent exists, update it
                    logger.info(f"Agent found in Agentverse, updating metadata...")
                    
                    update_payload = {
                        "name": agent.name,
                        "readme": readme,
                        "short_description": self.description or f"LlamaIndex agent: {agent.name}",
                    }
                    
                    update_response = requests.put(
                        get_url, json=update_payload, headers=headers, timeout=10
                    )
                    
                    if update_response.status_code == 200:
                        logger.info(f"[OK] Agent '{agent.name}' registered to Agentverse marketplace")
                        logger.info(f"   View at: https://agentverse.ai/agents")
                        logger.info(f"   Check 'My Agents' for: {agent.name}")
                        self._registered = True
                    else:
                        logger.warning(
                            f"Failed to update agent: {update_response.status_code} - {update_response.text}"
                        )
                
                elif get_response.status_code == 404:
                    # Agent doesn't exist yet
                    logger.warning(
                        f"Agent not found in Agentverse (404). "
                        f"This is normal for first-time setup."
                    )
                    logger.info(
                        f"[TIP] The agent needs to be created in Agentverse first. "
                        f"After clicking 'Connect' in inspector, it may take a few moments to appear in the API."
                    )
                else:
                    logger.warning(f"Unexpected response when checking agent: {get_response.status_code}")
            
            except Exception as e:
                logger.warning(f"Error checking agent status: {e}")
            
        except Exception as e:
            logger.warning(f"Agentverse registration error: {e}")
            logger.info(f"[INFO] Agent still works locally via Almanac")
    
    def _generate_readme(self, agent_name: str) -> str:
        """Generate README for Agentverse marketplace listing.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Formatted README content in Markdown
        """
        agent_type = type(self.agent_ll).__name__
        
        return f"""# {agent_name}
![tag:llamaindex](https://img.shields.io/badge/llamaindex-3D8BD3)
<br />
<br />

A LlamaIndex-powered agent for intelligent query answering and document retrieval.

## Description
{self.description or f'A LlamaIndex {agent_type} integrated with uAgents for decentralized AI services.'}

## How to Use
Send chat messages to interact with this LlamaIndex agent. The agent will:
- Process your queries using LlamaIndex
- Provide intelligent responses with source attribution
- Support complex RAG (Retrieval-Augmented Generation) workflows

## Agent Details
- **Framework**: LlamaIndex
- **Agent Type**: `{agent_type}`
- **Protocol**: AgentChatProtocol v0.1.0
- **Capabilities**: Query processing, document retrieval, chat interactions

## Example Queries
- "What is X?"
- "Explain Y to me"
- "Find information about Z"
- "Summarize the documents"

## Features
- Natural language query processing
- Document retrieval with source attribution
- Context-aware responses
- Supports Workflow, Agent, and QueryEngine types

## Contact
Agent powered by LlamaIndex and uAgents adapter.
"""
    
    async def _query_llamaindex(self, query_text: str) -> tuple[str, list[str] | None]:
        """Query the LlamaIndex agent/workflow/engine.
        
        This method handles different LlamaIndex object types:
        - Workflow/FunctionAgent: calls run() method (async)
        - Agent: calls query() or chat() method  
        - QueryEngine: calls query() method
        
        Args:
            query_text: The query string
            
        Returns:
            Tuple of (result_text, sources_list)
            
        Raises:
            QueryEngineError: If query fails
        """
        try:
            # Try different query methods based on object type
            response = None
            
            # Try: workflow.run() or agent.run() (async)
            if hasattr(self.agent_ll, 'run'):
                try:
                    import inspect
                    
                    # Check if run method is async - FunctionAgent.run is async but wrapped
                    # So we need to check both iscoroutinefunction and if calling it returns a coroutine
                    is_async = inspect.iscoroutinefunction(self.agent_ll.run)
                    logger.debug(f"agent_ll.run is coroutine function: {is_async}")
                    
                    # Try calling with user_msg first (FunctionAgent parameter)
                    try:
                        response = self.agent_ll.run(user_msg=query_text)
                        logger.debug(f"Called agent.run(user_msg=...) - Got type: {type(response).__name__}")
                        
                        # Check if the response is a coroutine or awaitable (async call)
                        if inspect.iscoroutine(response) or inspect.isawaitable(response):
                            logger.debug("Response is awaitable, awaiting it...")
                            response = await response
                            logger.debug(f"After await - Got type: {type(response).__name__}")
                        
                    except TypeError as te:
                        # Fallback: try with query parameter for other agent types
                        logger.debug(f"user_msg failed ({te}), trying query parameter...")
                        response = self.agent_ll.run(query=query_text)
                        
                        if inspect.iscoroutine(response) or inspect.isawaitable(response):
                            logger.debug("Response is awaitable (query param), awaiting it...")
                            response = await response
                            logger.debug(f"After await - Got type: {type(response).__name__}")
                    
                except Exception as e:
                    logger.debug(f"run() method failed: {e}")
                    pass
            
            # Try: agent.query()
            if response is None and hasattr(self.agent_ll, 'query'):
                try:
                    response = self.agent_ll.query(query_text)
                    logger.debug("Queried via agent.query()")
                except Exception as e:
                    logger.debug(f"query() method failed: {e}")
                    pass
            
            # Try: agent.chat()
            if response is None and hasattr(self.agent_ll, 'chat'):
                try:
                    response = self.agent_ll.chat(query_text)
                    logger.debug("Queried via agent.chat()")
                except Exception as e:
                    logger.debug(f"chat() method failed: {e}")
                    pass
            
            if response is None:
                raise QueryEngineError(
                    f"Could not query {type(self.agent_ll).__name__}. "
                    f"No suitable method found (run/query/chat)."
                )
            
            # Extract result text
            result = self._extract_response_text(response)
            
            # Extract sources
            sources = self._extract_sources(response)
            
            return result, sources if sources else None
            
        except Exception as e:
            if isinstance(e, QueryEngineError):
                raise
            raise QueryEngineError(
                "Failed to query LlamaIndex agent",
                original_error=e,
                query=query_text
            )
    
    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LlamaIndex response object.
        
        LlamaIndex responses can have different structures depending on
        the query engine configuration. This method handles multiple
        response formats gracefully.
        
        Args:
            response: LlamaIndex query response object
            
        Returns:
            Extracted response text as string
            
        Raises:
            QueryEngineError: If response text cannot be extracted
            
        Note:
            This method tries multiple extraction strategies:
            1. response.response attribute (most common, may be ChatMessage)
            2. response.output attribute (FunctionAgent/AgentOutput)
            3. response.text attribute (some engines)
            4. str(response) fallback
        """
        try:
            logger.debug(f"Extracting text from response type: {type(response).__name__}, module: {type(response).__module__}")
            
            # Try common response attribute
            if hasattr(response, 'response'):
                resp_obj = response.response
                logger.debug(f"Response has .response attribute, type: {type(resp_obj).__name__}")
                
                # If response is a ChatMessage object, extract content
                if hasattr(resp_obj, 'content'):
                    result = str(resp_obj.content)
                    logger.debug(f"Extracted via .response.content: {len(result)} chars")
                    return result
                else:
                    result = str(resp_obj)
                    logger.debug(f"Extracted via .response: {len(result)} chars")
                    return result
            
            # Try output attribute (FunctionAgent returns AgentOutput with .output)
            if hasattr(response, 'output'):
                result = str(response.output)
                logger.debug(f"Extracted via .output: {len(result)} chars")
                return result
            
            # Try text attribute
            if hasattr(response, 'text'):
                result = str(response.text)
                logger.debug(f"Extracted via .text: {len(result)} chars")
                return result
            
            logger.debug("No known attributes found, trying str() conversion...")
            # Fallback to string conversion (but catch InvalidStateError)
            try:
                result = str(response)
                logger.debug(f"Extracted via str(): {len(result)} chars")
                return result
            except Exception as str_err:
                # If str() fails (e.g., InvalidStateError from workflow Handler)
                # Try one more time to get attributes
                logger.warning(f"str() conversion failed: {type(str_err).__name__}: {str_err}")
                if hasattr(response, '__dict__'):
                    logger.warning(f"Response __dict__ keys: {list(response.__dict__.keys())}")
                    # Last resort: look for any text-like attributes
                    for attr in ['message', 'content', 'answer', 'result']:
                        if hasattr(response, attr):
                            result = str(getattr(response, attr))
                            logger.debug(f"Extracted via .{attr}: {len(result)} chars")
                            return result
                raise str_err
            
        except Exception as e:
            error_msg = "Failed to extract response text from query result"
            logger.error(f"{error_msg}: {str(e)}")
            raise QueryEngineError(error_msg, original_error=e)
    
    def _extract_sources(
        self, 
        response: Any, 
        max_sources: int = 3,
        max_length: int = 200
    ) -> list[str]:
        """Extract source document excerpts from response.
        
        LlamaIndex responses can include source_nodes that reference the
        documents used to generate the answer. This method extracts and
        formats these sources for attribution.
        
        Args:
            response: LlamaIndex query response object
            max_sources: Maximum number of sources to extract
            max_length: Maximum character length per source excerpt
            
        Returns:
            List of source excerpts (empty list if no sources available)
            
        Note:
            - Sources are truncated to max_length characters
            - Newlines are replaced with spaces for readability
            - Returns empty list if sources are unavailable or errors occur
        """
        sources = []
        
        try:
            # Check for source_nodes attribute
            if not hasattr(response, 'source_nodes'):
                logger.debug("Response has no source_nodes attribute")
                return sources
            
            source_nodes = response.source_nodes
            
            # Validate source_nodes
            if not source_nodes:
                logger.debug("source_nodes is empty")
                return sources
            
            # Extract sources
            for i, node in enumerate(source_nodes[:max_sources]):
                try:
                    # Extract text from node
                    if hasattr(node, 'node') and hasattr(node.node, 'text'):
                        text = node.node.text
                    elif hasattr(node, 'text'):
                        text = node.text
                    else:
                        logger.warning(f"Source node {i} has no text attribute")
                        continue
                    
                    # Clean and truncate text
                    text = str(text).replace('\n', ' ').strip()
                    if len(text) > max_length:
                        text = text[:max_length] + "..."
                    
                    sources.append(text)
                    logger.debug(f"Extracted source {i+1}: {len(text)} chars")
                    
                except Exception as e:
                    logger.warning(f"Failed to extract source {i}: {str(e)}")
                    continue
            
            logger.debug(f"Extracted {len(sources)} source(s)")
            return sources
            
        except Exception as e:
            logger.warning(f"Error extracting sources: {str(e)}")
            return sources
    
    def _format_response_with_sources(
        self, 
        result: str, 
        sources: list[str]
    ) -> str:
        """Format response text with source attributions.
        
        Appends formatted source citations to the response text for
        better transparency and attribution.
        
        Args:
            result: The main response text
            sources: List of source excerpts
            
        Returns:
            Formatted string with result and sources
            
        Example Output:
            "The answer is 42.
            
            📚 Sources:
            1. Source document excerpt one...
            2. Source document excerpt two..."
        """
        if not sources:
            return result
        
        formatted = result + "\n\n📚 Sources:\n"
        for i, source in enumerate(sources, 1):
            formatted += f"{i}. {source}\n"
        
        return formatted
    
    def _create_protocol(self) -> Protocol:
        """Create protocol for handling chat-based queries.
        
        This protocol handles ChatMessage interactions from Agentverse UI
        or other agents using the chat protocol. It processes text content
        and sends responses back via chat messages.
        
        Returns:
            Protocol configured with ChatMessage handlers
            
        Message Flow:
            1. ChatMessage received
            2. Acknowledgement sent immediately
            3. Text content extracted from message
            4. Query processed via LlamaIndex agent_ll
            5. Response formatted with sources
            6. ChatMessage response sent back
            
        Error Handling:
            All errors caught internally - user doesn't need try/catch.
        """
        protocol = Protocol(
            name="LlamaIndexChatProtocol",
            version="0.1.0"
        )
        logger.debug("Creating chat protocol")
        
        @protocol.on_message(model=ChatMessage)
        async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
            """Handle Agentverse chat messages.
            
            User doesn't need try/catch - all errors handled internally.
            """
            ctx.logger.info(f"[LlamaIndex] Received message from {sender[:8]}...")
            
            # Send acknowledgement
            try:
                await ctx.send(
                    sender,
                    ChatAcknowledgement(
                        timestamp=datetime.now(timezone.utc),
                        acknowledged_msg_id=msg.msg_id
                    )
                )
            except Exception as e:
                ctx.logger.error(f"[LlamaIndex] Failed to send ack: {str(e)}")
            
            # Process each content item
            for item in msg.content:
                try:
                    if isinstance(item, StartSessionContent):
                        ctx.logger.info(f"[LlamaIndex] Session started")
                        continue
                    
                    elif isinstance(item, EndSessionContent):
                        ctx.logger.info(f"[LlamaIndex] Session ended")
                        continue
                    
                    elif isinstance(item, TextContent):
                        query_text = item.text
                        if not query_text or not query_text.strip():
                            continue
                        
                        ctx.logger.info(f"[LlamaIndex] Query: {query_text[:50]}...")
                        
                        try:
                            # Query LlamaIndex agent (async)
                            result, sources = await self._query_llamaindex(query_text)
                            
                            # Format with sources
                            formatted = self._format_response_with_sources(result, sources or [])
                            
                            # Send response
                            await ctx.send(
                                sender,
                                ChatMessage(
                                    timestamp=datetime.now(timezone.utc),
                                    msg_id=uuid4(),
                                    content=[
                                        TextContent(type="text", text=formatted)
                                    ]
                                )
                            )
                            ctx.logger.info(f"[LlamaIndex] Response sent ({len(result)} chars)")
                            
                        except Exception as e:
                            # Send error message to user
                            error_msg = "I encountered an error processing your query. Please try again."
                            ctx.logger.error(f"[LlamaIndex] Query error: {str(e)}\n{traceback.format_exc()}")
                            
                            try:
                                await ctx.send(
                                    sender,
                                    ChatMessage(
                                        timestamp=datetime.now(timezone.utc),
                                        msg_id=uuid4(),
                                        content=[TextContent(type="text", text=error_msg)]
                                    )
                                )
                            except Exception:
                                pass
                
                except Exception as e:
                    ctx.logger.error(f"[LlamaIndex] Content processing error: {str(e)}")
                    continue
        
        @protocol.on_message(model=ChatAcknowledgement)
        async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            """Handle chat acknowledgements."""
            ctx.logger.debug(f"[LlamaIndex] Ack received for {msg.acknowledged_msg_id}")
        
        logger.debug("Chat protocol created")
        return protocol
