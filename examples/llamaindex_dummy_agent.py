
import sys
from pathlib import Path

# Add parent's src to path for importing adapter
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
import os


def create_dummy_llamaindex_agent():
    """
    Create a simple LlamaIndex FunctionAgent that responds to queries.
    
    This agent is designed to be used with the uAgents adapter.
    The adapter will handle calling the agent and extracting responses.
    
    Returns:
        LlamaIndex FunctionAgent powered by Google Gemini
    """
    # Get API key from environment or use default
    api_key = os.getenv("GOOGLE_API_KEY") or "AIzaSyCQLXiuy8kKOaTlZRyADitrEBh9a5TKA_w"
    
    # Create LLM
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.7
    )
    
    # Create agent with friendly system prompt
    agent = FunctionAgent(
        tools=[],
        llm=llm,
        system_prompt=(
            "you are the agent that deliver the roast jokes to the user in a funny way."
        ),
    )
    
    return agent


if __name__ == "__main__":
    # Simple test to verify agent creation
    print("Creating LlamaIndex FunctionAgent...")
    agent = create_dummy_llamaindex_agent()
    print(f"✓ Agent created successfully: {type(agent).__name__}")
    print("\nTo use this agent with uAgents, run: python examples/test_adapters.py")
