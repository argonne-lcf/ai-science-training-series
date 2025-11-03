from typing import TypedDict, Annotated

from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from inference_auth_token import get_access_token

from tools import molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation


# ============================================================
# 1. State definition
# ============================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. Routing logic
# ============================================================
def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


# ============================================================
# 3. LLM node: the "agent"
# ============================================================
def chem_agent(
    state: State,
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str = "You are an assistant that use tools to solve problems ",
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


# ============================================================
# 3*. A second agent: Handle creating structured output
# ============================================================
def structured_output_agent(
    state: State,
    llm: ChatOpenAI,
    system_prompt: str = ("You are an assistant that returns ONLY JSON. "),
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]

    result = llm.invoke(messages)
    return {"messages": [result]}


# ============================================================
# 4. LLM / tools setup
# ============================================================
# Get token for your ALCF inference endpoint
access_token = get_access_token()

# Initialize the model hosted on the ALCF endpoint
llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",
    # model_name="Qwen/Qwen3-32B",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Tool list that the LLM can call
tools = [molecule_name_to_smiles, smiles_to_coordinate_file, run_mace_calculation]

# ============================================================
# 5. Build the graph
# ============================================================
graph_builder = StateGraph(State)

# Agent node: calls LLM, which may decide to call tools
graph_builder.add_node(
    "chem_agent",
    lambda state: chem_agent(state, llm=llm, tools=tools),
)
graph_builder.add_node(
    "structured_output_agent",
    lambda state: structured_output_agent(state, llm=llm),
)

# Tool node: executes tool calls emitted by the LLM
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Graph logic
# START -> chem_agent
graph_builder.add_edge(START, "chem_agent")

# After chem_agent runs, check if we need to run tools
graph_builder.add_conditional_edges(
    "chem_agent", route_tools, {"tools": "tools", "done": "structured_output_agent"}
)

# After tools run, go back to the agent so it can use tool results
graph_builder.add_edge("tools", "chem_agent")

# After structured_output_agent, terminate the graph
graph_builder.add_edge("structured_output_agent", END)
# Compile the graph
graph = graph_builder.compile()

# ============================================================
# 6. Run / stream the graph
# ============================================================
prompt = "Optimize formic acid and acetic acid with MACE. Return the results in a JSON."
for chunk in graph.stream(
    {"messages": prompt},
    stream_mode="values",
):
    new_message = chunk["messages"][-1]
    new_message.pretty_print()
