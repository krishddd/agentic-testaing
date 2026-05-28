
import logging
from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from src.automation.orchestrator import PipelineOrchestrator
from src.configuration import ConfigurationManager
from src.model import Model

# Setup logging
logger = logging.getLogger(__name__)

class PipelineAgent:
    """
    An agentic interface for the Pipeline Orchestrator.
    Allows users to talk to the pipeline: e.g., "Run a health check", "Ingest new files".
    """

    def __init__(self):
        # Initialize pipeline components
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.configurations()
        
        # Load LLM
        model_setup = Model(self.config)
        self.llm = model_setup.load_ollama_model()
        
        # Initialize Orchestrator
        self.orchestrator = PipelineOrchestrator(self.llm, self.config)
        
        # Create Tools
        self.tools = self._create_tools()
        
        # Create Agent
        self.agent_executor = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """Wrap orchestrator methods as tools."""
        
        def run_health_check_wrapper(*args, **kwargs):
            return str(self.orchestrator.run_health_check())

        def trigger_ingestion_wrapper(*args, **kwargs):
            # Check for new files manually or force check
            new_files = self.orchestrator._detect_new_files()
            if not new_files:
                return "No new files detected. Nothing to ingest."
            return self.orchestrator.trigger_ingestion(new_files)

        def get_status_wrapper(*args, **kwargs):
            return f"Watching: {len(self.orchestrator.known_files_cache)} files. Data Path: {self.orchestrator.data_path}"

        tools = [
            Tool(
                name="RunHealthCheck",
                func=run_health_check_wrapper,
                description="Runs a robust evaluation of the RAG pipeline's retrieval quality. Returns metrics like precision and recall."
            ),
            Tool(
                name="TriggerIngestion",
                func=trigger_ingestion_wrapper,
                description="Checks for new files in the data directory and triggers ingestion/vector store update."
            ),
            Tool(
                name="GetPipelineStatus",
                func=get_status_wrapper,
                description="Gets the current status of the pipeline, including file count and watched directory."
            )
        ]
        return tools

    def _create_agent(self) -> AgentExecutor:
        template = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        """Run the agent with a user query."""
        logger.info(f"PipelineAgent received query: {query}")
        try:
            result = self.agent_executor.invoke({"input": query})
            return result['output']
        except Exception as e:
            logger.error(f"PipelineAgent error: {e}")
            return f"Error executing pipeline agent: {e}"

if __name__ == "__main__":
    # Test run
    agent = PipelineAgent()
    print("--- Pipeline Agent Initialized ---")
    
    # Example 1: Status
    print(agent.run("What is the current status of the pipeline?"))
    
    # Example 2: Health Check
    # print(agent.run("Run a health check and tell me if everything is okay."))
