import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uagents import Agent
from uagents_adapter.llamaindex import LlamaIndexAdapter
from llamaindex_dummy_agent import create_dummy_llamaindex_agent


def main():

    # Step 1: Create LlamaIndex agent
    agent_ll = create_dummy_llamaindex_agent()
    
    
    AGENTVERSE_API_TOKEN = os.getenv("AGENTVERSE_API_TOKEN") or "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjE3NjQwNTgwMDAsImlhdCI6MTc2MTQ2NjAwMCwiaXNzIjoiZmV0Y2guYWkiLCJqdGkiOiI4OWIwNjQ3Mjg0ZTQ2MjA5OTU2MTg5YjYiLCJzY29wZSI6ImF2Iiwic3ViIjoiMmFjMjU3ODY4NjNkYTFiZDY5ODg3Njc4NDc4MGI4YzdjM2ZlNmE5ODc1NzBmNjA5In0.TDEXhMUnlr5djlH1gSfC5T2hu6eS52BNIMbeikTNIQIg2O3hAKdX-lekOCAtgQw4jlL5J14WZqKTKUO361zw1cefUxF8bWeqxbKMIzPCypdt0cPxbqIdcXSGiUsGYloWaKWLwZtE8T4VD0H6VmWDOFrK9kT-eevldMrN2vjMUlU9yJ9dzfEdF2-IJHdpk46avZMBg_jxat6J88gvQHyuRf9xXWEHNM9UUqRJblVE4PNtCSJNB4vWnU2wvYdBFwmKaMXeD_sQusmk45Y_MWR0OIsJDJvXl2a8tkZGoDPBa2C-29EmIc7JDUwhpXf9DbV7Dm_2of0SQzMSEcoGeUiiOQ"
    
    adapter = LlamaIndexAdapter(
    agent_ll,
    agentverse_api_token=AGENTVERSE_API_TOKEN,
    description="LlamaIndex agent with Gemini 2.5 Flash"
    )
    
    # Step 3: Create uAgent
    agent = Agent(
        name="my_llamaindex_agent",
        seed="rl3jwenfdipljkcrjfdkenslrk;djs;l.wdnfsm;lsewsdkdsb32wndn3efsld",
        port=8001,
        mailbox=True
    )
    
    # Step 4: Include protocol
    agent.include(adapter.protocol, publish_manifest=True)

    adapter.run(agent)
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Agent stopped by user")
        print("="*70)
