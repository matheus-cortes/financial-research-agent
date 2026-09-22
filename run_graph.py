from app.graph.graph import graph


initial_state = {
    "ticker": "SAPR4",
    "iteration": 0,
}

print("\n========== FINANCIAL RESEARCH ==========")

result = graph.invoke(initial_state)

print("\n========== RESEARCH COMPLETED ==========")
print(result)