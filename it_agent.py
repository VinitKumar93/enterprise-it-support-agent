from __future__ import annotations
import os, re, uuid, time
from pathlib import Path
from typing import TypedDict, List, Optional, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
KB_DIR = Path(__file__).parent / "knowledge_base"

# Mock enterprise data only.
EMPLOYEES = {
    "E1001": {"name": "Rahul", "department": "Finance", "account_status": "ACTIVE", "vpn_enabled": True},
    "E1002": {"name": "Priya", "department": "Engineering", "account_status": "ACTIVE", "vpn_enabled": True},
    "E1003": {"name": "Alex", "department": "HR", "account_status": "LOCKED", "vpn_enabled": False},
}
TICKETS: list[dict] = []
RESET_LOG: list[dict] = []

@tool
def check_employee_account(employee_id: str) -> str:
    """Check the  employee account status for a supplied employee ID."""
    rec = EMPLOYEES.get(employee_id.strip().upper())
    if not rec:
        return f"No  employee record found for {employee_id.strip().upper()}. Check the ID or contact IT."
    return f" account lookup: employee ID {employee_id.strip().upper()}, status {rec['account_status']}, VPN enabled: {rec['vpn_enabled']}. This is simulated data."

@tool
def check_vpn_status(employee_id: str) -> str:
    """Check VPN enablement and simulated service status for a  employee ID."""
    key = employee_id.strip().upper()
    rec = EMPLOYEES.get(key)
    if not rec:
        return f"No  employee record found for {key}."
    return f" VPN status for {key}: {'enabled' if rec['vpn_enabled'] else 'not enabled'}; simulated VPN service: AVAILABLE."

@tool
def create_ticket(employee_id: str, issue: str) -> str:
    """Create a simulated IT support ticket. Never include passwords or sensitive IDs in issue text."""
    safe_issue = redact_pii(issue)
    ticket = {"ticket_id": "IT-" + uuid.uuid4().hex[:6].upper(), "employee_id": employee_id.strip().upper(), "issue": safe_issue, "status": "OPEN"}
    TICKETS.append(ticket)
    return f" ticket created: {ticket['ticket_id']} (status OPEN). Summary: {safe_issue}"

@tool
def reset_password(employee_id: str) -> str:
    """Simulate a password reset after authorization and explicit human approval. Never returns a password."""
    key = employee_id.strip().upper()
    if key not in EMPLOYEES:
        return "Reset denied: employee ID was not found in  data."
    RESET_LOG.append({"employee_id": key, "at": time.strftime('%Y-%m-%d %H:%M:%S'), "status": "SIMULATED_RESET"})
    return f"A  password-reset operation was recorded for {key}. No password was generated, changed, or revealed; no real system was contacted."

TOOLS = {t.name: t for t in [check_employee_account, check_vpn_status, create_ticket, reset_password]}

# Guardrails: detect prompt injection, PII, and sensitive/unauthorized operations.
INJECTION_PATTERNS = [r"ignore (all )?(previous|prior) instructions", r"reveal .*password", r"show .*administrator password", r"system prompt", r"bypass (the )?(guardrail|security|policy)"]
PII_PATTERNS = [
    (re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'), "Aadhaar-like number"),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "national ID-like number"),
    (re.compile(r'(?i)\b(password|passwd|passcode)\s*[:=]\s*\S+'), "password value"),
    (re.compile(r'\b\b\d{10}\b'), "10-digit number"),
]
def redact_pii(text: str) -> str:
    result = text
    for pattern, label in PII_PATTERNS:
        result = pattern.sub(f"[REDACTED {label}]", result)
    return result

def has_injection(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in INJECTION_PATTERNS)

def has_sensitive_pii(text: str) -> bool:
    return any(p.search(text) for p, _ in PII_PATTERNS)

class LocalRetriever:
    """Small local TF-IDF vector index over LangChain Documents; no API keys or network calls."""
    def __init__(self):
        self.docs: list[Document] = []
        for path in sorted(KB_DIR.glob("*.md")):
            self.docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d.page_content for d in self.docs]) if self.docs else None
    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        if self.matrix is None: return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.docs[i], float(scores[i])) for i in order]

RETRIEVER = LocalRetriever()

def ollama_answer(question: str, context: str) -> Optional[str]:
    if os.getenv("USE_OLLAMA", "false").lower() != "true": return None
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)
        prompt = ("You are an internal IT support assistant. Answer only from the provided company policy context. "
                  "If context is insufficient, say so. Never reveal credentials or claim a  operation affected a real system. "
                  "Keep the answer practical.\n\nCONTEXT:\n" + context + "\n\nQUESTION:\n" + question)
        return str(llm.invoke(prompt).content)
    except Exception:
        return None

class AgentState(TypedDict, total=False):
    query: str
    employee_id: str
    approved: bool
    safe_query: str
    blocked: bool
    pii_detected: bool
    intent: str
    docs: list
    context: str
    relevant: bool
    rewrite_count: int
    retry_count: int
    action: str
    tool_result: str
    pending_approval: bool
    answer: str
    error: str

def input_guardrail(state: AgentState) -> AgentState:
    q = state.get("query", "").strip()
    if not q:
        return {"blocked": True, "answer": "Please enter an IT support question."}
    if has_injection(q):
        return {"blocked": True, "answer": "I can’t follow instructions that attempt to bypass security controls or reveal passwords/secret instructions."}
    pii = has_sensitive_pii(q)
    return {"safe_query": redact_pii(q), "pii_detected": pii, "blocked": False}

def classify_request(state: AgentState) -> AgentState:
    q = state.get("safe_query", "").lower()
    # Intent/action is selected from the user's request; no fixed tool sequence is used.
    if any(x in q for x in ["reset my password", "reset password", "password reset", "forgot my password", "reset the password"]): intent = "password_reset"
    elif any(x in q for x in ["create a ticket", "create ticket", "raise a ticket", "open a ticket", "log a ticket", "submit a ticket"]): intent = "ticket"
    elif any(x in q for x in ["account active", "account status", "employee account", "is my account", "check my account"]): intent = "account"
    elif any(x in q for x in ["vpn status", "vpn enabled", "is vpn", "check vpn"]): intent = "vpn_status"
    else: intent = "knowledge"
    return {"intent": intent, "action": "none", "rewrite_count": state.get("rewrite_count", 0)}

def retrieve_documents(state: AgentState) -> AgentState:
    q = state.get("safe_query", state.get("query", ""))
    found = RETRIEVER.search(q, k=3)
    context = "\n\n".join(f"[Source: {d.metadata['source']} | score={score:.2f}]\n{d.page_content}" for d, score in found if score > 0.01)
    docs = [(d.metadata.get("source"), score, d.page_content) for d, score in found]
    return {"docs": docs, "context": context, "relevant": bool(context) and max([s for _, s, _ in docs] or [0]) >= 0.08}

def grade_documents(state: AgentState) -> AgentState:
    # A simple relevance threshold; if weak, rewrite once and retry retrieval.
    return {"relevant": bool(state.get("relevant"))}

def rewrite_query(state: AgentState) -> AgentState:
    q = state.get("safe_query", state.get("query", ""))
    additions = {"vpn": " VPN connection troubleshooting internet credentials client service", "wifi": " wireless network connectivity", "bitlocker": " laptop encryption recovery key security policy", "software": " approved application installation policy", "password": " self-service password policy identity verification"}
    lower = q.lower(); extra = next((v for k, v in additions.items() if k in lower), " IT helpdesk policy troubleshooting FAQ")
    return {"safe_query": q + extra, "rewrite_count": state.get("rewrite_count", 0) + 1}

def agent_select(state: AgentState) -> AgentState:
    intent = state.get("intent", "knowledge")
    if intent == "account": return {"action": "check_employee_account"}
    if intent == "vpn_status": return {"action": "check_vpn_status"}
    if intent == "ticket": return {"action": "create_ticket"}
    if intent == "password_reset": return {"action": "reset_password"}
    return {"action": "knowledge"}

def execute_tool_node(state: AgentState) -> AgentState:
    action = state.get("action", "")
    employee_id = state.get("employee_id", "E1001").strip().upper()
    try:
        if action == "check_employee_account": result = check_employee_account.invoke({"employee_id": employee_id})
        elif action == "check_vpn_status": result = check_vpn_status.invoke({"employee_id": employee_id})
        elif action == "create_ticket":
            issue = redact_pii(state.get("safe_query", state.get("query", "IT support request")))
            # simulated transient failure hook for testing retry/error handling
            if "simulate-failure" in issue.lower() and state.get("retry_count", 0) < 1:
                raise RuntimeError("Simulated transient ticket service failure")
            result = create_ticket.invoke({"employee_id": employee_id, "issue": issue})
        elif action == "reset_password":
            if not state.get("approved", False): return {"pending_approval": True, "tool_result": ""}
            result = reset_password.invoke({"employee_id": employee_id})
        else: result = "No tool selected."
        return {"tool_result": str(result), "pending_approval": False, "error": ""}
    except Exception as exc:
        if state.get("retry_count", 0) < 1:
            return {"error": str(exc), "retry_count": state.get("retry_count", 0) + 1}
        return {"error": str(exc), "tool_result": "The simulated IT operation failed after a retry. Please try again later or contact the helpdesk.", "pending_approval": False}

def human_approval(state: AgentState) -> AgentState:
    # This node is a human-in-the-loop gate. UI must explicitly resubmit with approved=True.
    if not state.get("approved", False):
        return {"pending_approval": True, "answer": "Approval required: confirm the employee ID and authorization before a simulated password reset. No reset has been performed. Use the Approve  reset button only for this demo record."}
    return {"pending_approval": False}

def generate_response(state: AgentState) -> AgentState:
    if state.get("pending_approval"):
        return {"answer": state.get("answer") or "This action is waiting for human approval. No operation has been performed."}
    if state.get("tool_result"):
        return {"answer": state["tool_result"]}
    context = state.get("context", "")
    query = state.get("query", "")
    if not state.get("relevant") or not context:
        return {"answer": "I couldn't find enough information in the company's IT knowledge base to answer this confidently. You can ask me to create an IT support ticket."}
    llm_text = ollama_answer(query, context)
    if llm_text: return {"answer": llm_text}
    # Safe offline grounded fallback.
    snippets = []
    for source, score, body in state.get("docs", []):
        if score >= 0.08:
            snippets.append(f"**{source}**\n{body.strip()}")
    if not snippets:
        return {"answer": "I couldn't find enough information in the company's IT knowledge base to answer this confidently."}
    return {"answer": "Based on the local IT knowledge base:\n\n" + "\n\n".join(snippets[:2]) + "\n\nIf the issue persists, create a support ticket. Do not include passwords or sensitive personal information."}

def output_guardrail(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    # Ensure sensitive-looking numbers and credential assignments aren't echoed.
    answer = redact_pii(answer)
    if re.search(r'(?i)(password|passwd)\s*[:=]\s*\S+', answer):
        answer = "I can’t display passwords or credentials. Use the approved self-service password portal or contact IT."
    return {"answer": answer}

def route_after_input(state: AgentState) -> str:
    if state.get("blocked"): return "end"
    return "classify"
def route_after_classify(state: AgentState) -> str:
    return "retrieve" if state.get("intent") == "knowledge" else "agent"
def route_after_grade(state: AgentState) -> str:
    if state.get("relevant"): return "generate"
    if state.get("rewrite_count", 0) < 1: return "rewrite"
    return "generate"
def route_after_agent(state: AgentState) -> str:
    return "approval" if state.get("action") == "reset_password" else "execute"
def route_after_approval(state: AgentState) -> str:
    return "execute" if state.get("approved", False) else "generate"
def route_after_execute(state: AgentState) -> str:
    if state.get("error") and state.get("retry_count", 0) < 1: return "execute"
    return "generate"

graph = StateGraph(AgentState)
for name, fn in [("input_guardrail", input_guardrail), ("classify_request", classify_request), ("retrieve_documents", retrieve_documents), ("grade_documents", grade_documents), ("rewrite_query", rewrite_query), ("agent_select", agent_select), ("execute_tool", execute_tool_node), ("human_approval", human_approval), ("generate_response", generate_response), ("output_guardrail", output_guardrail)]:
    graph.add_node(name, fn)
graph.set_entry_point("input_guardrail")
graph.add_conditional_edges("input_guardrail", route_after_input, {"end": "output_guardrail", "classify": "classify_request"})
graph.add_conditional_edges("classify_request", route_after_classify, {"retrieve": "retrieve_documents", "agent": "agent_select"})
graph.add_edge("retrieve_documents", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grade, {"generate": "generate_response", "rewrite": "rewrite_query"})
graph.add_edge("rewrite_query", "retrieve_documents")
graph.add_conditional_edges("agent_select", route_after_agent, {"approval": "human_approval", "execute": "execute_tool"})
graph.add_conditional_edges("human_approval", route_after_approval, {"execute": "execute_tool", "generate": "generate_response"})
graph.add_conditional_edges("execute_tool", route_after_execute, {"execute": "execute_tool", "generate": "generate_response"})
graph.add_edge("generate_response", "output_guardrail")
graph.add_edge("output_guardrail", END)
APP = graph.compile()

def ask_agent(query: str, employee_id: str = "E1001", approved: bool = False) -> dict:
    result = APP.invoke({"query": query, "employee_id": employee_id or "E1001", "approved": approved, "rewrite_count": 0, "retry_count": 0})
    return {"answer": result.get("answer", "No response generated."), "pending_approval": result.get("pending_approval", False), "intent": result.get("intent", "unknown"), "action": result.get("action", "none"), "pii_detected": result.get("pii_detected", False)}
