from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import molecule_name_to_smiles, calculator
from inference_auth_token import get_access_token

# Get access token
access_token = get_access_token()

# Initialize the ALCF Inference Endpoint model
llm = ChatOpenAI(
    model_name="openai/gpt-oss-120b",
    api_key=access_token,
    base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
    temperature=0,
)

# Build a ReAct agent with the specified tools
tools = [molecule_name_to_smiles, calculator]
agent = create_agent(llm, tools=tools)

# Run the agent and display its responses
# Note: Sometimes the agent may skip tool calls and answer from its internal knowledge.
# Adjust the system prompt if consistent tool usage is required, or specifically tell the agent to "use your tools".
prompt = "What is the smiles string of methanol and the values of 3*5*5*5*5*5?"
for chunk in agent.stream(
    {"messages": prompt},
    stream_mode="values",
):
    new_message = chunk["messages"][-1]
    new_message.pretty_print()
