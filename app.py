import gradio as gr
from it_agent import ask_agent, TICKETS, RESET_LOG

def respond(message, history, employee_id):
    if not message or not message.strip(): return "Please enter a question."
    result = ask_agent(message, employee_id or "E1001", approved=False)
    answer = result["answer"]
    if result.get("pii_detected"):
        answer += "\n\nPrivacy note: sensitive-looking information was redacted from any simulated ticket content."
    return answer

def approve_reset(employee_id, pending_query):
    if not pending_query:
        return "No pending reset request found.", gr.update(visible=False)
    result = ask_agent(pending_query, employee_id or "E1001", approved=True)
    return result["answer"], gr.update(visible=False)

def show_demo_data():
    return "### Tickets\n" + ("\n".join(f"- **{t['ticket_id']}** — {t['employee_id']} — {t['status']} — {t['issue']}" for t in TICKETS) if TICKETS else "No tickets created yet.") + "\n\n###  reset audit\n" + ("\n".join(f"- {r['employee_id']} — {r['status']} — {r['at']}" for r in RESET_LOG) if RESET_LOG else "No reset operations recorded.")

with gr.Blocks(title="Enterprise IT Support Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛠️ Enterprise IT Support Agent\n using **LangGraph + LangChain + RAG + guardrails**. All employee records, tickets, and password-reset actions are data; ")
    with gr.Row():
        employee_id = gr.Dropdown(choices=["E1001", "E1002", "E1003"], value="E1001", label="Demo employee ID", allow_custom_value=True)
        gr.Markdown("**Try:** `How do I troubleshoot my VPN?` · `Is my employee account active?` · `My VPN is not working. Please create a support ticket.` · `Reset my password.`")
    chatbot = gr.ChatInterface(fn=respond, additional_inputs=[employee_id], type="messages", chatbot=gr.Chatbot(height=420, type="messages"), textbox=gr.Textbox(placeholder="Describe your IT issue...", container=False, scale=7), title="Chat with IT Support")
    # Provide a dedicated form for an explicit human-in-the-loop approval test.
    gr.Markdown("###  Approval \nFor the approval flow, use the form below. A reset will not execute until you explicitly click Approve.")
    with gr.Row():
        reset_request = gr.Textbox(value="Reset my password.", label="Password reset request")
        reset_employee = gr.Dropdown(choices=["E1001", "E1002", "E1003"], value="E1001", label="Employee ID")
    reset_check = gr.Button("Submit for approval")
    reset_result = gr.Markdown()
    approve_button = gr.Button("Approve and reset", variant="primary")
    approval_state = gr.State({"query": "", "employee_id": "E1001"})
    def submit_for_approval(query, emp):
        res = ask_agent(query, emp, approved=False)
        return res["answer"], {"query": query, "employee_id": emp}, gr.update(visible=bool(res.get("pending_approval")) or "Approval required" in res["answer"])
    def approve_pending(state):
        if not state or not state.get("query"): return "No pending request. Submit one first.", gr.update(visible=False)
        res = ask_agent(state["query"], state.get("employee_id", "E1001"), approved=True)
        return res["answer"], gr.update(visible=False)
    reset_check.click(submit_for_approval, [reset_request, reset_employee], [reset_result, approval_state, approve_button])
    approve_button.click(approve_pending, [approval_state], [reset_result, approve_button])
    with gr.Accordion(" Audit data", open=False):
        audit = gr.Markdown()
        refresh = gr.Button("Refresh audit his")
        refresh.click(show_demo_data, outputs=audit)
    gr.Markdown("### Safety notes\n- Prompt-injection attempts and credential requests are blocked.\n- PII-like values are redacted before simulated ticket storage.\n- Password reset is requires the explicit approval button above.\n- Knowledge answers are grounded in local Markdown documents; if retrieval is weak, the agent says it lacks enough evidence.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
