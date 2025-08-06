
import os
from fastapi import FastAPI
from langserve import add_routes
from dotenv import load_dotenv
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field

# Import your modules
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph
from smart_research_assistant.langsmith_module import ResearchAssistantLangSmith

# Load environment variables
load_dotenv()

# Initialize the app
app = FastAPI(
    title="Smart Research Assistant API",
    version="1.0",
    description="API for a Smart Research Assistant that can conduct comprehensive research and summarize text."
)

# Initialize our components
langchain_module = ResearchAssistantLangChain()
langgraph_module = ResearchAssistantLangGraph()
langsmith_module = ResearchAssistantLangSmith()

# Define the research chain
class ResearchInput(BaseModel):
    query: str = Field(..., description="Research question to investigate")
    enable_tracing: bool = Field(False, description="Whether to enable LangSmith tracing")

class ResearchOutput(BaseModel):
    summary: str = Field("", description="Research summary")
    research_plan: List[str] = Field(default=[], description="Steps in the research plan")
    follow_up_questions: List[str] = Field(default=[], description="Follow-up questions")
    analysis: str = Field(default="", description="Detailed analysis of findings")

# Create a simpler runnable function for research - with more error handling

def research_function(input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        query = input_data.get("query", "")
        if not query:
            return {
                "summary": "Error: No query provided",  # Return string instead of None
                "research_plan": [],  # Return empty list instead of None
                "follow_up_questions": [],  # Return empty list instead of None
                "analysis": ""  # Return empty string instead of None
            }
            
        enable_tracing = input_data.get("enable_tracing", False)
        
        # Run the research
        try:
            if enable_tracing:
                result = langsmith_module.execute_with_tracing(query)
                if "result" in result:
                    data = result["result"]
                else:
                    data = result
            else:
                data = langgraph_module.execute_research(query)
                
            # Ensure all required fields have default values (not None)
            return {
                "summary": data.get("summary") or "",  # Default to empty string
                "research_plan": data.get("research_plan") or [],  # Default to empty list
                "follow_up_questions": data.get("follow_up_questions") or [],  # Default to empty list
                "analysis": data.get("analysis") or ""  # Default to empty string
            }
        except Exception as e:
            print(f"Error in research execution: {e}")
            # Return valid default values
            return {
                "summary": f"An error occurred during research: {str(e)}",
                "research_plan": [],
                "follow_up_questions": [],
                "analysis": ""
            }
    except Exception as e:
        print(f"Error in research function: {e}")
        return {
            "summary": f"An error occurred: {str(e)}",
            "research_plan": [],
            "follow_up_questions": [],
            "analysis": ""
        }

# Create a runnable chain for research
research_chain = RunnableLambda(research_function)

# Define the summary chain
class SummaryInput(BaseModel):
    text: str = Field(..., description="Text to be summarized")

class SummaryOutput(BaseModel):
    summary: str = Field("", description="Generated summary")

# Create a simpler runnable function for summarization
def summarize_function(input_data: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
    try:
        text = input_data.get("text", "")
        if not text:
            return {"summary": "Error: No text provided"}
            
        summary = langchain_module.summarize_document(text)
        return {"summary": summary}
    except Exception as e:
        print(f"Error in summarize function: {e}")
        return {"summary": f"An error occurred: {str(e)}"}

# Create a runnable chain for summarization
summary_chain = RunnableLambda(summarize_function)




# Add routes to the app
add_routes(
    app,
    research_chain,
    path="/research",
    input_type=ResearchInput,
    output_type=ResearchOutput
)

add_routes(
    app,
    summary_chain,
    path="/summarize",
    input_type=SummaryInput,
    output_type=SummaryOutput
)

# Add a simple echo endpoint for testing
def echo_function(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return {"message": f"Received: {input_data}"}

echo_chain = RunnableLambda(echo_function)

class EchoInput(BaseModel):
    text: str = Field(..., description="Text to echo")

class EchoOutput(BaseModel):
    message: str = Field(..., description="Echo response")

add_routes(
    app,
    echo_chain,
    path="/echo",
    input_type=EchoInput,
    output_type=EchoOutput
)

# Add default routes for health check
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Smart Research Assistant API is running"}

# Run with: uvicorn langserve_app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)