# streamlit_app.py

import streamlit as st
import os
import json
from dotenv import load_dotenv
# Set the recursion limit (optional: use sys if needed)
import sys
sys.setrecursionlimit(1000)  # Adjust this value as needed
# Import our modules with correct imports
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph
from smart_research_assistant.langsmith_module import ResearchAssistantLangSmith

# Load environment variables
load_dotenv()

# Main application class (simplified
class SmartResearchAssistant:
    def __init__(self):
        """Initialize the Smart Research Assistant"""
        self.langchain_module = ResearchAssistantLangChain()
        self.langgraph_module = ResearchAssistantLangGraph()
        self.langsmith_module = ResearchAssistantLangSmith()
    
    def research(self, query, with_tracing=True):
        """Execute a complete research workflow"""
        if with_tracing:
            # Execute with LangSmith tracing
            result = self.langsmith_module.execute_with_tracing(query)
            return result
        else:
            # Execute without tracing
            result = self.langgraph_module.execute_research(query)
            return {"result": result}
    
    def generate_summary(self, text):
        """Generate a summary of a document"""
        summary = self.langchain_module.summarize_document(text)
        return summary




# Streamlit app UI
st.set_page_config(page_title="Smart Research Assistant", page_icon="🔍")
st.title("Smart Research Assistant")
st.markdown("Powered by LangChain, LangGraph, and LangSmith")

# Initialize assistant
if 'assistant' not in st.session_state:
    st.session_state.assistant = SmartResearchAssistant()

# Sidebar options
st.sidebar.header("Settings")
enable_tracing = st.sidebar.checkbox("Enable LangSmith Tracing", value=True)

# Create tabs
tab1, tab2 = st.tabs(["Research", "Text Summarization"])

# Tab 1: Research
with tab1:
    st.header("Research Query")
    query = st.text_area("Enter your research question", height=100)
    
    if st.button("Start Research"):
        if not query:
            st.error("Please enter a research query")
        else:
            with st.spinner("Researching... (this may take a while)"):
                try:
                    result = st.session_state.assistant.research(query, with_tracing=enable_tracing)
                    
                    # Extract the actual result data
                    if "result" in result:
                        data = result["result"]
                    else:
                        data = result
                    
                    # Display research plan
                    if data.get("research_plan"):
                        st.subheader("Research Plan")
                        for i, step in enumerate(data["research_plan"]):
                            st.markdown(f"**Step {i+1}:** {step}")
                    
                    # Display summary
                    if data.get("summary"):
                        st.subheader("Research Summary")
                        st.markdown(data["summary"])
                    
                    # Display follow-up questions
                    if data.get("follow_up_questions"):
                        st.subheader("Follow-up Questions")
                        for q in data["follow_up_questions"]:
                            st.markdown(f"- {q}")
                    
                    # Option to download results
                    output = {
                        "query": data.get("query", query),
                        "summary": data.get("summary", ""),
                        "analysis": data.get("analysis", ""),
                        "research_plan": data.get("research_plan", []),
                        "follow_up_questions": data.get("follow_up_questions", [])
                    }
                    
                    st.download_button(
                        label="Download Results",
                        data=json.dumps(output, indent=2),
                        file_name="research_results.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Tab 2: Text Summarization
with tab2:
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize", height=200)
    
    if st.button("Generate Summary"):
        if not text:
            st.error("Please enter text to summarize")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = st.session_state.assistant.generate_summary(text)
                    st.subheader("Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")