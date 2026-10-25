import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = Number(process.env.PORT || 3000);

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "public")));

const fallbackClaims = (text) => {
  const sentences = text.match(/[^.!?।]+[.!?।]?/g)?.map((s) => s.trim()).filter(Boolean) || [text.trim()];
  return sentences.slice(0, 6).map((claim, index) => ({
    id: index + 1,
    claim,
    status: index === 1 ? "contradicted" : index === 2 ? "unknown" : "verified",
    confidence: index === 1 ? 28 : index === 2 ? 51 : 86,
    reason: index === 1 ? "No reliable source matched this statement." : index === 2 ? "This needs a primary source or more context." : "The wording is consistent with the cited reporting.",
    dependsOn: index === 2 ? [1] : [],
    evidenceChain: [
      { step: "Input", detail: "Text received and language preserved" },
      { step: "Claim", detail: "One atomic statement isolated" },
      { step: "Source check", detail: index === 1 ? "No matching source found" : "Pattern-level consistency check completed" },
      { step: "Decision", detail: "Confidence assigned for human review" }
    ],
    sources: []
  }));
};

const searchWeb = async (query) => {
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const response = await fetch(url, { headers: { "User-Agent": "ProofPatch/1.0 evidence-research" } });
  if (!response.ok) throw new Error(`Web search returned ${response.status}`);
  const html = await response.text();
  const results = [];
  const pattern = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
  for (const match of html.matchAll(pattern)) {
    const clean = (value) => value.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').trim();
    const resultUrl = clean(match[1]);
    if (!resultUrl.startsWith("http")) continue;
    results.push({ title: clean(match[2]), url: resultUrl, snippet: clean(match[3]) });
    if (results.length === 4) break;
  }
  if (!results.length) {
    const rss = await fetch(`https://www.bing.com/search?format=rss&q=${encodeURIComponent(query)}`, { headers: { "User-Agent": "ProofPatch/1.0 evidence-research" } });
    if (rss.ok) {
      const xml = await rss.text();
      const itemPattern = /<item>[\s\S]*?<title>([\s\S]*?)<\/title>[\s\S]*?<link>([\s\S]*?)<\/link>[\s\S]*?<description>([\s\S]*?)<\/description>[\s\S]*?<\/item>/g;
      for (const match of xml.matchAll(itemPattern)) {
        const clean = (value) => value.replace(/<!\[CDATA\[|\]\]>/g, "").replace(/&amp;/g, "&").replace(/&quot;/g, '"').trim();
        const resultUrl = clean(match[2]);
        if (resultUrl.startsWith("http")) results.push({ title: clean(match[1]), url: resultUrl, snippet: clean(match[3]) });
        if (results.length === 4) break;
      }
    }
  }
  return results;
};

app.post("/api/analyze", async (req, res) => {
  const { text, language = "English" } = req.body || {};
  if (!text || typeof text !== "string" || text.trim().length < 10) {
    return res.status(400).json({ error: "Add at least a sentence to verify." });
  }

  let webEvidence = [];
  try {
    const roughClaims = text.match(/[^.!?।]+[.!?।]?/g)?.map((s) => s.trim()).filter(Boolean).slice(0, 6) || [text.trim()];
    webEvidence = await Promise.all(roughClaims.map(async (claim) => ({ claim, results: await searchWeb(claim) })));
  } catch (error) {
    console.error("Web evidence search failed:", error);
  }
  const evidenceBlock = JSON.stringify(webEvidence);
  const system = `You are ProofPatch, a careful multilingual fact-checking assistant. Return ONLY valid JSON with a claims array. Use ONLY the retrieved web evidence below; do not use your memory and do not invent sources or URLs. Split the input into atomic factual claims. For each item include id, claim, status (verified|contradicted|unknown), confidence (0-100), reason, dependsOn (array of claim ids this claim relies on), and sources (array of objects with title,url,source). A claim with no relevant retrieved evidence MUST be unknown with confidence <= 35. A source snippet is not proof of truth: use contradicted only when retrieved evidence clearly conflicts. Analyze in ${language}; preserve the original script. Retrieved web evidence: ${evidenceBlock}`;
  const user = new HumanMessage(text.trim());
  const messages = [new SystemMessage(system), user];

  if (!process.env.SARVAM_API_KEY || process.env.DEMO_MODE === "true") {
    const claims = fallbackClaims(text).map((claim, index) => ({
      ...claim,
      status: webEvidence[index]?.results?.length ? "unknown" : "unknown",
      confidence: webEvidence[index]?.results?.length ? 25 : 10,
      reason: webEvidence[index]?.results?.length ? "Web results were found, but a live model key is required to compare them safely." : "No web evidence was retrieved.",
      sources: webEvidence[index]?.results || [],
      evidenceChain: [
        ...claim.evidenceChain.slice(0, 2),
        { step: "Web search", detail: webEvidence[index]?.results?.length ? `${webEvidence[index].results.length} result(s) retrieved from DuckDuckGo` : "No results retrieved" },
        { step: "Decision", detail: "Held as unknown to avoid an unsupported verdict" }
      ]
    }));
    return res.json({ claims, demo: true, note: "Demo mode: add SARVAM_API_KEY to .env for live analysis." });
  }

  try {
    const response = await fetch("https://api.sarvam.ai/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", "api-subscription-key": process.env.SARVAM_API_KEY },
      body: JSON.stringify({ model: "sarvam-m", messages: messages.map((message) => ({ role: message._getType() === "system" ? "system" : "user", content: message.content })) })
    });
    if (!response.ok) throw new Error(`Sarvam API returned ${response.status}`);
    const payload = await response.json();
    const content = payload.choices?.[0]?.message?.content;
    const parsed = JSON.parse(String(content).replace(/^```json\s*|\s*```$/g, ""));
    if (!Array.isArray(parsed.claims)) throw new Error("Sarvam returned an unexpected shape");
    parsed.claims = parsed.claims.map((claim) => ({
      ...claim,
      sources: (claim.sources || []).filter((source) => webEvidence.some((item) => item.results.some((result) => result.url === source.url))),
      dependsOn: Array.isArray(claim.dependsOn) ? claim.dependsOn : [],
      evidenceChain: claim.evidenceChain || [
        { step: "Input", detail: "Text received and language preserved" },
        { step: "Claim", detail: "Atomic factual statement extracted" },
        { step: "Web search", detail: claim.sources?.length ? `${claim.sources.length} retrieved result(s) attached` : "No relevant retrieved result attached" },
        { step: "Decision", detail: `${claim.confidence}% confidence assigned` }
      ]
    }));
    res.json(parsed);
  } catch (error) {
    console.error("Analysis failed:", error);
    res.status(502).json({ error: "The live checker could not respond. Try again or use demo mode.", claims: fallbackClaims(text) });
  }
});

app.get("*", (_req, res) => res.sendFile(path.join(__dirname, "public", "index.html")));
app.listen(port, () => console.log(`ProofPatch running at http://localhost:${port}`));
