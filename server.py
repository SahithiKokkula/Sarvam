import json
import os
import re
import asyncio
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlparse

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
PUBLIC_DIR = ROOT / "public"
if PUBLIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "English"


class CleanTextRequest(BaseModel):
    text: str
    language: str = "English"


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = "Untitled"
    url: str
    snippet: str = ""
    source: str = "Web"
    domain: str = ""
    fetched: bool = False
    evidenceScore: int = Field(default=0, ge=0, le=100)


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


def local_relevant_text(raw: str) -> str:
    lines = re.split(r"\r?\n", unicodedata.normalize("NFC", str(raw or "")))
    chrome = re.compile(r"^(live|breaking news|news|yesterday at|comments?\s*[•·*x]|shares?|and \d+ others|[©®]|[\W_]*\d{1,2}:\d{2}[\W_]*)$", re.I)
    kept = []
    for line in lines:
        line = re.sub(r"[|¦]", "I", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or chrome.match(line):
            continue
        if re.match(r"^(प्रतिक्रिया|कमेंट्स?|शेयर|लाइक)\b", line, re.I):
            continue
        if re.fullmatch(r"[=@#\W_]{1,12}", line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\s+([।!?,:;])", r"\1", cleaned)
    cleaned = re.sub(r"([।!?])(?=\S)", r"\1 ", cleaned)
    return cleaned.strip()


@app.post("/api/clean-text")
async def clean_text(request: CleanTextRequest):
    original = request.text.strip()
    if len(original) < 3:
        return {"text": "", "method": "local", "removed": 0}
    local = local_relevant_text(original)
    if not os.getenv("SARVAM_API_KEY") or os.getenv("DEMO_MODE") == "true":
        return {"text": local, "method": "local", "removed": max(0, len(original) - len(local))}
    prompt = f"""Extract only the relevant factual claim text from OCR output.
Remove social-media chrome, usernames, reactions, timestamps, navigation labels,
watermarks, duplicated fragments, and OCR garbage. Preserve Hindi and other scripts.
Return only the cleaned text, no explanation. Language hint: {request.language}.
OCR output:
{local}"""
    try:
        client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-105b-conversations",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        cleaned = str(response.choices[0].message.content or "").strip()
        cleaned = re.sub(r"^```(?:text)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
        if len(cleaned) >= 3:
            return {"text": cleaned, "method": "sarvam", "removed": max(0, len(original) - len(cleaned))}
    except Exception as error:
        print(f"OCR relevance filtering failed; using local cleanup: {error}")
    return {"text": local, "method": "local-fallback", "removed": max(0, len(original) - len(local))}


async def search_web(query: str) -> list[dict]:
    headers = {"User-Agent": "ProofPatch/1.0 (evidence-research; +https://proofpatch.local)"}
    timeout = httpx.Timeout(10.0, connect=5.0)
    errors = []

    async def google_news(client):
        response = await client.get(f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en")
        response.raise_for_status()
        return [{
            "title": item.get("title", "Untitled"),
            "url": item.get("link", ""),
            "snippet": BeautifulSoup(item.get("summary", ""), "html.parser").get_text(" ", strip=True),
            "source": "Google News",
        } for item in feedparser.parse(response.text).entries[:8] if item.get("link", "").startswith("http")]

    async def duckduckgo(client):
        response = await client.get(f"https://html.duckduckgo.com/html/?q={quote(query)}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return [{
            "title": link.get_text(" ", strip=True),
            "url": link.get("href", ""),
            "snippet": item.select_one(".result__snippet").get_text(" ", strip=True) if item.select_one(".result__snippet") else "",
            "source": "DuckDuckGo",
        } for item in soup.select(".result")[:8]
            if (link := item.select_one(".result__a")) and link.get("href", "").startswith("http")]

    async def bing(client):
        response = await client.get(f"https://www.bing.com/search?q={quote(query)}&count=8")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return [{
            "title": link.get_text(" ", strip=True),
            "url": link.get("href", ""),
            "snippet": item.select_one(".b_caption p").get_text(" ", strip=True) if item.select_one(".b_caption p") else "",
            "source": "Bing",
        } for item in soup.select("li.b_algo")[:8]
            if (link := item.select_one("h2 a")) and link.get("href", "").startswith("http")]

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        responses = await asyncio.gather(
            *(provider(client) for provider in (google_news, duckduckgo, bing)),
            return_exceptions=True,
        )
        results = []
        for response in responses:
            if isinstance(response, Exception):
                errors.append(type(response).__name__)
            else:
                results.extend(response)

        unique = {}
        for result in results:
            url = result["url"].split("#", 1)[0].rstrip("/")
            if url and url not in unique:
                result["url"] = url
                result["domain"] = urlparse(url).netloc.lower().removeprefix("www.")
                unique[url] = result

        async def enrich(result):
            try:
                page = await client.get(result["url"], timeout=7)
                if page.is_success and "text/html" in page.headers.get("content-type", ""):
                    soup = BeautifulSoup(page.text, "html.parser")
                    canonical = soup.select_one("link[rel='canonical']")
                    if canonical and canonical.get("href", "").startswith("http"):
                        result["url"] = canonical["href"].split("#", 1)[0].rstrip("/")
                        result["domain"] = urlparse(result["url"]).netloc.lower().removeprefix("www.")
                    meta_description = soup.select_one("meta[property='og:description'], meta[name='description']")
                    for node in soup(["script", "style", "noscript", "svg"]):
                        node.decompose()
                    paragraphs = [node.get_text(" ", strip=True) for node in soup.select("article p, main p, p")]
                    text = " ".join(item for item in paragraphs if len(item) > 40)
                    if len(text) < 40 and meta_description:
                        text = meta_description.get("content", "").strip()
                    if text:
                        result["snippet"] = text[:700]
                        result["fetched"] = True
                        result["evidenceScore"] = min(100, 45 + min(len(text) // 12, 55))
            except (httpx.HTTPError, UnicodeError):
                pass
            result.setdefault("fetched", False)
            result.setdefault("evidenceScore", min(45, 15 + len(result.get("snippet", "")) // 12))
            return result

        enriched = await asyncio.gather(*(enrich(result) for result in list(unique.values())[:12]))
        if errors and not enriched:
            raise RuntimeError(f"All web providers failed: {', '.join(errors)}")
        return enriched


def retrieval_summary(evidence: list[dict], errors: list[str]) -> dict:
    sources = [source for item in evidence for source in item["results"]]
    domains = Counter(source.get("domain") or urlparse(source["url"]).netloc for source in sources)
    return {
        "provider": "Google News + DuckDuckGo + Bing",
        "providersUsed": sorted({source.get("source", "Web") for source in sources}),
        "claimsSearched": len(evidence),
        "resultsFound": len(sources),
        "pagesFetched": sum(1 for source in sources if source.get("fetched")),
        "domainsFound": len(domains),
        "errors": errors,
    }


def fallback_claims(claims: list[str], evidence: list[dict]) -> list[dict]:
    output = []
    for index, claim in enumerate(claims):
        sources = evidence[index]["results"] if index < len(evidence) else []
        output.append({
            "id": index + 1,
            "claim": claim,
            "status": "unknown",
            "confidence": min(55, 20 + len(sources) * 6 + sum(1 for source in sources if source.get("fetched")) * 5) if sources else 10,
            "reason": "Evidence was collected, but a live adjudication key is required to compare it safely." if sources else "No web evidence was retrieved from the configured providers.",
            "dependsOn": [],
            "sources": sources,
            "evidenceChain": [
                {"step": "Input", "detail": "Text received and language preserved"},
                {"step": "Claim", "detail": "One atomic statement isolated"},
                {"step": "Web search", "detail": f"{len(sources)} result(s) from {len({item.get('domain') for item in sources})} domains"},
                {"step": "Page scrape", "detail": f"{sum(1 for source in sources if source.get('fetched'))} source page(s) opened and text extracted"},
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
    retrieval = retrieval_summary(evidence, search_errors)

    if not os.getenv("SARVAM_API_KEY") or os.getenv("DEMO_MODE") == "true":
        return {"claims": fallback_claims(claims, evidence), "demo": True, "retrieval": retrieval}

    system = f"""You are ProofPatch, a careful multilingual fact-checking assistant.
Return ONLY compact JSON with a claims array. Use ONLY the retrieved web evidence below, never your memory. Prefer corroboration across independent domains.
No relevant evidence means status unknown and confidence <= 35. Use contradicted only when a retrieved
source clearly conflicts. Do not invent URLs. Preserve the original script and analyze in {request.language}.
Each claim must include id, claim, status, confidence, reason, dependsOn, and sources. Confidence must reflect source quality, agreement, page text extracted, and remaining uncertainty; explain that in reason.
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
            retrieved_by_url = {source["url"]: source for source in retrieved_sources}
            claim.sources = [
                Source.model_validate({**retrieved_by_url[source.url], **source.model_dump()})
                for source in attached_sources
            ] or [Source.model_validate(source) for source in retrieved_sources[:3]]
            claim.status = claim.status if claim.status in {"verified", "contradicted", "unknown"} else "unknown"
            claim.dependsOn = [item for item in claim.dependsOn if item != claim.id]
            if not claim.sources:
                claim.status = "unknown"
                claim.confidence = min(claim.confidence, 35)
                claim.reason = "No retrieved source was attached to this claim, so it remains unknown."
            elif claim.status == "unknown":
                claim.confidence = min(claim.confidence, 60)
                claim.reason = f"Evidence was retrieved from {len({source.domain for source in claim.sources})} domain(s), but it did not establish the claim strongly enough for a supported verdict. {claim.reason}".strip()
            parsed_claim = claim.model_dump()
            pages_fetched = sum(1 for source in claim.sources if source.fetched)
            parsed_claim["evidenceChain"] = [
                {"step": "Input", "detail": "Text received and language preserved"},
                {"step": "Claim", "detail": "Atomic factual statement extracted"},
                {"step": "Web search", "detail": f"{len(claim.sources)} result(s) attached from {len({source.domain for source in claim.sources})} domain(s)"},
                {"step": "Page scrape", "detail": f"{pages_fetched} source page(s) opened and text extracted"},
                {"step": "Decision", "detail": f"{claim.confidence}% confidence based on corroboration and source quality"},
            ]
            parsed_claims.append(parsed_claim)
        return {"claims": parsed_claims, "retrieval": retrieval}
    except Exception as error:
        print(f"Adjudication failed: {error}")
        return {"claims": fallback_claims(claims, evidence), "demo": True, "note": "Live adjudication failed; claims were held unknown.", "retrieval": retrieval}


@app.get("/{path:path}")
async def frontend(path: str):
    index = PUBLIC_DIR / "index.html"
    if not index.is_file():
        return {"error": "Frontend is served separately in this deployment."}
    return FileResponse(index)
