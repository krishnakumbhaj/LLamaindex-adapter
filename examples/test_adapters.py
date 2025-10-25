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
  
    
    # Step 1: Create your LlamaIndex agent
   
    agent_ll = create_dummy_llamaindex_agent()
   
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
