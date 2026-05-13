from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Banking Risk Policy Q&A API - India Edition", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ API key loaded from .env file


GROQ_URL = "https://api.groq.com/openai/v1/chat/"




SYSTEM_PROMPT = """You are an expert Banking Risk Policy Analyst specializing in both Indian and International banking regulations. You have deep knowledge in:

=== INDIAN BANKING LAWS & REGULATIONS ===
- RBI (Reserve Bank of India) Guidelines & Master Circulars
- Banking Regulation Act 1949 - core law governing Indian banks
- RBI Act 1934 - establishes RBI powers and monetary policy
- SARFAESI Act 2002 - allows banks to recover NPAs without court intervention
- IBC (Insolvency and Bankruptcy Code) 2016 - resolution of stressed assets
- PMLA (Prevention of Money Laundering Act) 2002 - Indian AML law
- KYC/AML RBI Master Direction 2016 - customer vercompletionsification rules
- FEMA (Foreign Exchange Management Act) 1999 - foreign exchange regulations
- SEBI Regulations - capital markets and investment banking rules
- NPA (Non-Performing Assets) Classification - 90-day default rule
- Priority Sector Lending (PSL) norms - 40% lending to priority sectors
- CRR (Cash Reserve Ratio) & SLR (Statutory Liquidity Ratio) requirements
- Prompt Corrective Action (PCA) Framework by RBI
- CIBIL & Credit Bureau regulations
- Digital Payment regulations - UPI, RTGS, NEFT guidelines
- RBI Integrated Ombudsman Scheme 2021
- Basel III implementation in India (RBI circular)
- NBFC regulations and shadow banking rules
- Co-operative Banking regulations in India
- Jan Dhan Yojana & Financial Inclusion norms

=== INTERNATIONAL BANKING LAWS ===
- Basel III/IV - international capital adequacy standards
- Dodd-Frank Act (USA) - post-2008 financial crisis reforms
- AML/KYC - Anti Money Laundering & Know Your Customer
- FATF (Financial Action Task Force) recommendations
- MiFID II (Europe) - financial instruments regulation
- GDPR - data protection in banking
- SWIFT compliance regulations

=== RISK DOMAINS ===
- Credit Risk (NPA, loan defaults, LTV ratios, PD/LGD/EAD models, CIBIL scores)
- Market Risk (VaR, stress testing, interest rate risk, forex risk)
- Operational Risk (fraud, cyber risk, internal controls)
- Liquidity Risk (LCR, NSFR, CRR, SLR)
- Regulatory/Compliance Risk (RBI penalties, SEBI violations)
- Counterparty Risk (CVA, DVA, exposure at default)
- Systemic Risk (too-big-to-fail, contagion risk)

IMPORTANT: When answering, always prefer Indian banking context when relevant. Cite Indian laws (RBI circulars, Banking Regulation Act, SARFAESI, IBC, PMLA etc.) along with international references.

IMPORTANT: Respond ONLY with a valid JSON object. No markdown, no code blocks, no text outside JSON.

Format exactly like this:
{
  "answer": "detailed professional answer with Indian banking context",
  "risk_level": "LOW or MEDIUM or HIGH or CRITICAL",
  "risk_score": 50,
  "risk_factors": ["factor 1", "factor 2", "factor 3"],
  "recommendations": ["rec 1", "rec 2", "rec 3"],
  "regulatory_refs": ["RBI Master Circular...", "Banking Regulation Act 1949 Section X", "Basel III..."],
  "category": "Credit Risk or Market Risk or Operational Risk or Liquidity Risk or Compliance or NPA Management or Other",
  "confidence": 85,
  "indian_context": "specific Indian banking regulation or context relevant to this query"
}"""


class QueryRequest(BaseModel):
    query: str
    context: str = ""


class QueryResponse(BaseModel):
    answer: str
    risk_level: str
    risk_score: int
    risk_factors: list[str]
    recommendations: list[str]
    regulatory_refs: list[str]
    category: str
    confidence: int
    timestamp: str
    query: str
    indian_context: str = ""


@app.get("/")
async def root():
    return {
        "message": "Banking Risk Policy Q&A API - India Edition",
        "status": "running",
        "model": "llama-3.3-70b-versatile (Groq)",
        "coverage": "Indian + International Banking Laws"
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.query}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_URL, json=payload, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq error: {response.text}")

        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

        # Strip markdown if present
        response_text = re.sub(r"```json|```", "", response_text).strip()

        # Extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {
                "answer": response_text,
                "risk_level": "MEDIUM",
                "risk_score": 50,
                "risk_factors": ["Could not parse response"],
                "recommendations": ["Please consult a risk specialist"],
                "regulatory_refs": [],
                "category": "Other",
                "confidence": 60,
                "indian_context": ""
            }

        return QueryResponse(
            answer=data.get("answer", ""),
            risk_level=data.get("risk_level", "MEDIUM"),
            risk_score=int(data.get("risk_score", 50)),
            risk_factors=data.get("risk_factors", []),
            recommendations=data.get("recommendations", []),
            regulatory_refs=data.get("regulatory_refs", []),
            category=data.get("category", "Other"),
            confidence=int(data.get("confidence", 70)),
            timestamp=datetime.utcnow().isoformat(),
            query=request.query,
            indian_context=data.get("indian_context", ""),
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out. Try again.")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model": "llama-3.3-70b-versatile",
        "provider": "Groq",
        "coverage": "Indian + International Banking Laws",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)