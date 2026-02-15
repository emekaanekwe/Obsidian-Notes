# LangGraph should orchestrate agent communication, not wrap a single LLM call.
class AgentState(TypedDict):
1. Reads part of state
2. Writes structured output back to state

orchestrate
1. make a plan
2. first create of graph of X, and create a graph of Y
3. test by making a jupiter notebook file
4. have orchestrator as eval or have eval as langsmith

# Langgraph common structure

![[Pasted image 20260215094323.png]]

![[Pasted image 20260215094349.png]]
