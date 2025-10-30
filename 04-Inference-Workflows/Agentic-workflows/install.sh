module use /soft/modulefiles/
module load conda/2025-09-25
conda activate

python3 -m venv env/_ai4s_agentic --system-site-packages

# LangChain/LangGraph
pip install langchain langgraph langchain-openai 

# globus_sdk for ALCF Inference Endpoints
pip install globus_sdk

# Pubchempy
pip install "pubchempy @ git+https://github.com/keceli/PubChemPy.git@main"