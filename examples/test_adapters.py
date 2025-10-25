import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from llamaindex_dummy_agent import create_dummy_llamaindex_agent


async def main(): 
    agent_ll = create_dummy_llamaindex_agent()
   
    AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN") or "agent_verse_api_key"
    
    adapter = LlamaIndexAdapter(
        agent_ll,
        agentverse_api_token=AGENTVERSE_API_TOKEN,
        description="LlamaIndex agent with Gemini 2.5 Flash"
    )
    agent = Agent(
        name="my_llamaindex_agent",
        seed="my_unique_seed_phrase_12345knedsfdzlksza",
        port=8001,
        mailbox=True  # ✅ Just mailbox (same as LangChain!)
    )
    
   
    agent.include(adapter.protocol, publish_manifest=True)
    
    adapter.enable_agentverse_registration(agent, wait_seconds=30)
    
    await agent.run_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Agent stopped by user")
        print("="*70)
