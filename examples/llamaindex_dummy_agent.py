from llama_index.llms.google_genai import GoogleGenAI

async def create_dummy_llamaindex_agent_with_ctx(query: str, tx):
    """
    Creates a dummy LlamaIndex agent that accepts context via 'tx' parameter.
    
    Args:
        query: The user's query
        tx: Context object containing session_id and user_id (REQUIRED)
    """
    session_id = tx.session_id
    user_id = tx.user_id
    
    print(f"[dummy_agent] Query from user={user_id}, session={session_id}: {query}")
    
    # Initialize Gemini LLM
    llm = GoogleGenAI(
        model="models/gemini-2.5-flash", 
        api_key="AIzaSyCQLXiuy8kKOaTlZRyADitrEBh9a5TKA_w"
    )
    
    # Use the LLM directly (no agent needed for testing)
    response = await llm.acomplete(query)
    
    print(f"[dummy_agent] Response: {str(response)[:100]}...")
    
    return str(response)