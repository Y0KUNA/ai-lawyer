import os

from dotenv import load_dotenv

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .tools.retrieval_pipeline_tool import RetrievalPipelineTool

load_dotenv()

local_llm = LLM(
    model="ollama/gemma4:12b", #ollama/gemma4:e4b
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    
)


@CrewBase
class AILawyerCrew:

   
    @agent
    def legal_researcher(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["legal_researcher"],
            
            
            tools=[RetrievalPipelineTool()],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=local_llm,
            
        )
        
    
    @agent
    def legal_request_analyzer(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["legal_request_analyzer"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=local_llm,
            
        )
        
    
    @agent
    def legal_issue_identifier(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["legal_issue_identifier"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=local_llm,
            
        )
        
    
    @agent
    def irac_legal_reasoning_specialist(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["irac_legal_reasoning_specialist"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=local_llm,
            
        )
        
    
    @agent
    def legal_verification_and_synthesis_specialist(self) -> Agent:
        
        
        return Agent(
            config=self.agents_config["legal_verification_and_synthesis_specialist"],
            
            
            tools=[],
            
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            
            max_execution_time=None,
            llm=local_llm,
            
        )
        
    

    
    @task
    def analyze_legal_request(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_legal_request"],
            markdown=False,
            
            
        )
    
    @task
    def identify_legal_issues(self) -> Task:
        return Task(
            config=self.tasks_config["identify_legal_issues"],
            markdown=False,
            
            
        )
    
    @task
    def research_relevant_laws(self) -> Task:
        return Task(
            config=self.tasks_config["research_relevant_laws"],
            markdown=False,
            
            
        )
    
    @task
    def apply_irac_legal_reasoning(self) -> Task:
        return Task(
            config=self.tasks_config["apply_irac_legal_reasoning"],
            markdown=False,
            
            
        )
    
    @task
    def verify_and_synthesize_legal_response(self) -> Task:
        return Task(
            config=self.tasks_config["verify_and_synthesize_legal_response"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the AiLegalConsultantTuVanPhapLuat crew"""

        # Custom manager agent for hierarchical process
        manager_agent = Agent(
            role="Crew Manager",
            goal="Coordinate the legal analysis workflow across all specialists in the\n    correct order: Legal Request Analyzer, then Legal Issue Identifier,\n    then Legal Researcher, then IRAC Legal Reasoning Specialist, then\n    Legal Verification and Synthesis Specialist.\n \n    Before accepting the final output, verify it independently against these\n    rejection criteria. If ANY is true, send the work back to the named\n    agent with a specific correction instruction. Do not accept partial\n    fixes silently.\n \n    Criterion 1: No direct, definitive conclusion answering the user's\n    actual question(s). If true, send back to the Verification and\n    Synthesis Specialist.\n \n    Criterion 2: Any fact in the final answer contradicts or is absent from\n    the Legal Request Analyzer's extracted facts, such as an invented\n    missing term that was actually stated. If true, send back to whichever\n    agent introduced it.\n \n    Criterion 3: Fault or causation is assigned to whichever party acted\n    last, such as the one who refused to sign, without tracing back to who\n    caused the precondition to fail. If true, send back to the IRAC Legal\n    Reasoning Specialist.\n \n    Criterion 4: A cited legal article does not match what the Legal\n    Researcher actually retrieved. If true, send back to the Legal\n    Researcher.",
            backstory="  You are the supervising partner of a law firm. Your team is fast but junior (running on a lightweight model), so you personally re-check their output against the original case facts before it goes to the client. You have seen this team make three specific mistakes before: inventing \"missing\" contract terms that were actually stated, blaming whichever party acted last instead of whoever caused the underlying problem, and hedging instead of giving a real answer. You catch all three, every time, before sign-off.\n ",
            llm=local_llm,
            allow_delegation=True,
        )

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,


            manager_agent=manager_agent,


            chat_llm=local_llm,
        )


