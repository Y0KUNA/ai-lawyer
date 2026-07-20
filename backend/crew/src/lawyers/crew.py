import os

from dotenv import load_dotenv

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import (
    EXASearchTool,
    ScrapeWebsiteTool,
)

from .tools.retrieval_pipeline_tool import RetrievalPipelineTool

load_dotenv()

local_llm = LLM(
    model="ollama/gemma4:e2b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    
)


@CrewBase
class AILawyerCrew:

    ##################################################
    # MANAGER
    ##################################################

    def create_manager(self):

        return Agent(
            config=self.agents_config["legal_manager"],
            reasoning=True,
            allow_delegation=True,
            max_reasoning_attempts=4,
            max_iter=4,
            llm=local_llm,
        )

    ##################################################
    # AGENTS
    ##################################################

    @agent
    def case_analyzer(self):

        return Agent(
            config=self.agents_config["case_analyzer"],
            reasoning=False,
            allow_delegation=False,
            max_iter=3,
            inject_date=True,
            llm=local_llm,
        )

    @agent
    def issue_identifier(self):

        return Agent(
            config=self.agents_config["issue_identifier"],
            reasoning=False,
            max_reasoning_attempts=2,
            allow_delegation=False,
            max_iter=5,
            inject_date=True,
            llm=local_llm,
        )

    @agent
    def legal_researcher(self):

        return Agent(
            config=self.agents_config["legal_researcher"],

            tools=[
                RetrievalPipelineTool(),
            ],

            reasoning=False,
            allow_delegation=False,
            max_iter=4,
            inject_date=True,
            llm=local_llm,
        )

    @agent
    def legal_reasoner(self):

        return Agent(
            config=self.agents_config["legal_reasoner"],
            reasoning=False,
            max_reasoning_attempts=3,
            allow_delegation=False,
            max_iter=5,
            inject_date=True,
            llm=local_llm,
        )

    @agent
    def legal_reviewer(self):

        return Agent(
            config=self.agents_config["legal_reviewer"],
            reasoning=False,
            max_reasoning_attempts=2,
            allow_delegation=False,
            max_iter=3,
            inject_date=True,
            llm=local_llm,
        )

    @agent
    def legal_opinion_writer(self):

        return Agent(
            config=self.agents_config["legal_opinion_writer"],
            reasoning=False,
            allow_delegation=False,
            max_iter=2,
            inject_date=True,
            llm=local_llm,
        )

    ##################################################
    # TASKS
    ##################################################

    @task
    def extract_case_facts(self):

        return Task(
            config=self.tasks_config["extract_case_facts"],
            markdown=False,
        )

    @task
    def identify_legal_issues(self):

        return Task(
            config=self.tasks_config["identify_legal_issues"],
            markdown=False,
        )

    @task
    def research_legal_authorities(self):

        return Task(
            config=self.tasks_config["research_legal_authorities"],
            markdown=False,
        )

    @task
    def apply_irac(self):

        return Task(
            config=self.tasks_config["apply_irac"],
            markdown=False,
        )

    @task
    def review_analysis(self):

        return Task(
            config=self.tasks_config["review_analysis"],
            markdown=False,
        )

    @task
    def write_legal_opinion(self):

        return Task(
            config=self.tasks_config["write_legal_opinion"],
            markdown=False,
        )

    ##################################################
    # CREW
    ##################################################

    @crew
    def crew(self):

        return Crew(

            agents=self.agents,

            tasks=self.tasks,

            process=Process.sequential,

            manager_agent=self.create_manager(),

            verbose=True,

            memory=False,

            planning=False,
            
            planning_llm=local_llm,

            llm=local_llm,
        )