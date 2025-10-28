import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataclasses import dataclass  # Import dataclass
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from llamaindex_dummy_agent import create_dummy_llamaindex_agent_with_ctx

# ... (rest of imports and setup) ...

# We can define a simple dataclass to hold the context
@dataclass
class AdapterContext:
    """A simple data container for session context."""
    session_id: str
    user_id: str
    # You could add other fields here later if needed

def main():
    print("Creating LlamaIndex agent (with context)...")
    

    # --- THIS FUNCTION NOW CREATES A CONTEXT OBJECT ---
    async def my_llamaindex_function(query: str, session_id: str, user_id: str) -> str:
        """
        This function groups context into an object and passes it
        to the agent's ainvoke method via the 'tx' parameter.
        """
        print(f"[main_with_ctx] Received query: '{query}' for session: {session_id}")
        
        # Create the context object, like in your snippet
        ctx = AdapterContext(
            session_id=session_id,
            user_id=user_id
        )
        
        # Pass the object as the 'tx' keyword argument
        # (matching the original traceback)
        response = await create_dummy_llamaindex_agent_with_ctx(
            query,
            tx=ctx
        )
        
        print(f"[main_with_ctx] Sending response: '{response}'")
        return str(response)

    # --- (rest of the file is the same) ---
    print("Initializing LlamaIndexAdapter...")
    adapter = LlamaIndexAdapter(my_llamaindex_function)
    
    agent = Agent(
        name="my_llamaindex_agent_ctx",
        seed="rl3jwenfdipljkcrjrnelrkjn3lejfdkenslrk;djs;l.wdnfsm;lsewsdkdsb32wndn3efsld",
        port=8002,
        mailbox=True
    )
    
    print(f"Including adapter protocol on agent: {agent.name}")
    agent.include(adapter.protocol, publish_manifest=True)
    
    print(f"Agent address: {agent.address}")
    
    adapter.run(agent)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Agent stopped by user")
        print("="*70)

