import json
import os
import re
import asyncio
from pathlib import Path
from urllib.parse import quote

import feedparser
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sarvamai import SarvamAI

load_dotenv()
ROOT = Path(__file__).parent
app = FastAPI(title="ProofPatch")
app.mount("/static", StaticFiles(directory=ROOT / "public"), name="static")


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "English"


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = "Untitled"
    url: str
    snippet: str = ""
    source: str = "Web"


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int = 0
    claim: str
    status: str = "unknown"
    confidence: int = Field(default=0, ge=0, le=100)
    reason: str = ""
    dependsOn: list[int] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value):
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else 0

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, value):
        if not isinstance(value, list):
            return []
        return [{"url": item} if isinstance(item, str) else item for item in value]


class ClaimsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claims: list[Claim] = Field(default_factory=list)


def split_claims(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+", text.strip()) if part.strip()][:30] or [text.strip()]


async def search_web(query: str) -> list[dict]:
    headers = {"User-Agent": "ProofPatch/1.0 evidence-research"}
    results = []
    async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=headers) as client:
        news = await client.get(f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en")
        if news.is_success:
            feed = feedparser.parse(news.text)
            for item in feed.entries[:4]:
                link = item.get("link", "")
                if link.startswith("http"):
                    results.append({
                        "title": item.get("title", "Untitled"),
                        "url": link,
                        "snippet": BeautifulSoup(item.get("summary", ""), "html.parser").get_text(" ", strip=True),
                        "source": "Google News",
                    })
        if not results:
            response = await client.get(f"https://html.duckduckgo.com/html/?q={quote(query)}")
            if response.is_success:
                soup = BeautifulSoup(response.text, "html.parser")
                for item in soup.select(".result"):
                    link = item.select_one(".result__a")
                    snippet = item.select_one(".result__snippet")
                    if link and link.get("href", "").startswith("http"):
                        results.append({"title": link.get_text(" ", strip=True), "url": link["href"], "snippet": snippet.get_text(" ", strip=True) if snippet else "", "source": "DuckDuckGo"})
                    if len(results) == 4:
                        break
    return results


def fallback_claims(claims: list[str], evidence: list[dict]) -> list[dict]:
    output = []
    for index, claim in enumerate(claims):
        sources = evidence[index]["results"] if index < len(evidence) else []
        output.append({
            "id": index + 1,
            "claim": claim,
            "status": "unknown",
            "confidence": 25 if sources else 10,
            "reason": "Web results were found, but a live adjudication key is required to compare them safely." if sources else "No web evidence was retrieved.",
            "dependsOn": [],
            "sources": sources,
            "evidenceChain": [
                {"step": "Input", "detail": "Text received and language preserved"},
                {"step": "Claim", "detail": "One atomic statement isolated"},
                {"step": "Web search", "detail": f"{len(sources)} result(s) retrieved" if sources else "No results retrieved"},
                {"step": "Decision", "detail": "Held as unknown to avoid an unsupported verdict"},
            ],
        })
    return output


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    if len(request.text.strip()) < 10:
        return {"error": "Add at least a sentence to verify."}
    claims = split_claims(request.text)
    evidence = []
    search_errors = []
    async def retrieve(claim: str) -> dict:
        try:
            return {"claim": claim, "results": await search_web(claim)}
        except Exception as error:
            print(f"Web search failed: {error}")
            search_errors.append(str(error))
            return {"claim": claim, "results": []}

    evidence = await asyncio.gather(*(retrieve(claim) for claim in claims))

    if not os.getenv("SARVAM_API_KEY") or os.getenv("DEMO_MODE") == "true":
        return {"claims": fallback_claims(claims, evidence), "demo": True, "retrieval": {"provider": "Google News RSS + DuckDuckGo", "claimsSearched": len(claims), "resultsFound": sum(len(item["results"]) for item in evidence), "errors": search_errors}}

    system = f"""You are ProofPatch, a careful multilingual fact-checking assistant.
Return ONLY compact JSON with a claims array. Use ONLY the retrieved web evidence below, never your memory.
No relevant evidence means status unknown and confidence <= 35. Use contradicted only when a retrieved
source clearly conflicts. Do not invent URLs. Preserve the original script and analyze in {request.language}.
Each claim must include id, claim, status, confidence, reason, dependsOn, and sources.
Retrieved evidence: {json.dumps(evidence, ensure_ascii=False)}"""
    messages = [SystemMessage(content=system), HumanMessage(content=request.text.strip())]
    try:
        client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-105b-conversations",
            messages=[
                {"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content}
                for message in messages
            ],
            max_tokens=2400,
        )
        content = response.choices[0].message.content
        parsed = JsonOutputParser().parse(content)
        validated = ClaimsResponse.model_validate(parsed)
        parsed_claims = []
        for index, claim in enumerate(validated.claims):
            claim.id = claim.id if claim.id > 0 else index + 1
            evidence_index = max(0, min(len(evidence) - 1, claim.id - 1)) if evidence else 0
            retrieved_sources = evidence[evidence_index]["results"] if evidence else []
            allowed = {result["url"] for result in retrieved_sources}
            attached_sources = [source for source in claim.sources if source.url in allowed]
            claim.sources = attached_sources or [Source.model_validate(source) for source in retrieved_sources[:3]]
            claim.status = claim.status if claim.status in {"verified", "contradicted", "unknown"} else "unknown"
            claim.dependsOn = [item for item in claim.dependsOn if item != claim.id]
            if not claim.sources:
                claim.status = "unknown"
                claim.confidence = min(claim.confidence, 35)
                claim.reason = "No retrieved source was attached to this claim, so it remains unknown."
            parsed_claim = claim.model_dump()
            parsed_claim["evidenceChain"] = [
                {"step": "Input", "detail": "Text received and language preserved"},
                {"step": "Claim", "detail": "Atomic factual statement extracted"},
                {"step": "Web search", "detail": f"{len(claim.sources)} retrieved result(s) attached"},
                {"step": "Decision", "detail": f"{claim.confidence}% confidence assigned against retrieved evidence"},
            ]
            parsed_claims.append(parsed_claim)
        return {"claims": parsed_claims, "retrieval": {"provider": "Google News RSS + DuckDuckGo", "claimsSearched": len(claims), "resultsFound": sum(len(item["results"]) for item in evidence), "errors": search_errors}}
    except Exception as error:
        print(f"Adjudication failed: {error}")
        return {"claims": fallback_claims(claims, evidence), "demo": True, "note": "Live adjudication failed; claims were held unknown.", "retrieval": {"provider": "Google News RSS + DuckDuckGo", "claimsSearched": len(claims), "resultsFound": sum(len(item["results"]) for item in evidence), "errors": search_errors}}


@app.get("/{path:path}")
async def frontend(path: str):
    return FileResponse(ROOT / "public" / "index.html")
