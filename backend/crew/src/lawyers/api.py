from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lawyers.crew import AILawyerCrew

app = FastAPI(
    title="AI Lawyer API",
    version="1.0.0"
)


class CaseRequest(BaseModel):
    case_description: str


class CaseResponse(BaseModel):
    result: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: CaseRequest):
    crew = AILawyerCrew().crew()

    result = await crew.kickoff_async(
        inputs={
            "case_description": request.case_description
        }
    )

    return {
        "result": str(result)
    }