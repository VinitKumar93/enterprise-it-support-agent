from it_agent import ask_agent

def main():
    print("Enterprise IT Support Agent (/offline mode). Type 'exit' to quit.")
    employee = input("Demo employee ID [E1001]: ").strip() or "E1001"
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}: break
        res = ask_agent(q, employee, approved=False)
        print("\nAgent:", res["answer"])
        if res["pending_approval"]:
            ok = input("Approve this  password reset? Type YES to approve: ").strip()
            if ok == "YES": print("\nAgent:", ask_agent(q, employee, approved=True)["answer"])
            else: print("Approval declined; no reset performed.")
if __name__ == "__main__": main()
