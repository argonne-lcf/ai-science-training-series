# SambaNova SN40L (Metis)


SambaNova SN40L cluster is  composed of two SambaRacks—each housing 16 Reconfigurable Dataflow Unit (RDU) processors. This dedicated setup delivers high-throughput, low-latency performance for machine learning workloads.

SambaStack is a purpose-built hardware and software platform optimized for AI inference.
Models running on the cluster are exposed through OpenAI-compatible API endpoints, with each endpoint capable of hosting multiple independently accessible models.


## Using Metis Inference Endpoints

The Sambanova SN40L cluster (Metis) is integrated as part of the ALCF inference service provided through API access to the models running on the Metis cluster. The models running on Metis can be accessed in two ways.

1. Web UI
2. API Access

### Access through Web UI
The easiest way to get started is through the web interface, accessible at https://inference.alcf.anl.gov/

The UI is based on the popular Open WebUI platform. After logging in with your ANL or ALCF credentials, you can:
1. Select a model from the dropdown menu at the top of the screen.
2. Start a conversation directly in the chat interface.

In the model selection dropdown, you can see the status of each model:

![Sambanova connection diagram](./metis_inference_gui.png)

### Access through API 
For programmatic access, you can use the API endpoints directly.

#### 1. Setup Your Environment

You can run the following setup from any internet connected machine (your local machine, or an ALCF machine).

```bash
# Create a new Conda environment
conda create -n globus_env python==3.11.9 --y
conda activate globus_env

# Install necessary packages
pip install openai globus_sdk
```
Note: A python virtual environment may be used as well.
```bash
virtualenv -p python3.10 globus_env
source globus_env/bin/activate
pip install openai globus_sdk
```

#### 2. Authenticate

To access the endpoints, you need an authentication token.

```bash
# Download the authentication helper script
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py

# Authenticate with your Globus account
python inference_auth_token.py authenticate
```

This will generate and store access and refresh tokens in your home directory. To see how much time you have left before your access token expires, type the following command (`units` can be seconds, minutes, or hours):

```bash
python inference_auth_token.py get_time_until_token_expiration --units seconds
```

!!! warning "Token Validity"
    - Access tokens are valid for 48 hours. The `get_access_token` command will automatically refresh your token if it has expired.
    - An internal policy requires re-authentication every 7 days. If you encounter permission errors, logout from Globus at [app.globus.org/logout](https://app.globus.org/logout) and re-run `python inference_auth_token.py authenticate --force`.

#### 3. Make a Test Call

Once authenticated, you can make a test call using cURL or Python.

=== "cURL"

    ```bash
    #!/bin/bash

    # Get your access token
    access_token=$(python inference_auth_token.py get_access_token)

    curl -X POST "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions" \
         -H "Authorization: Bearer ${access_token}" \
         -H "Content-Type: application/json" \
         -d '{
                "model": "gpt-oss-120b-131072",
                "messages":[{"role": "user", "content": "Explain quantum computing in simple terms."}]
             }'
    ```

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI
    from inference_auth_token import get_access_token

    # Get your access token
    access_token = get_access_token()

    client = OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    )

    response = client.chat.completions.create(
        model="gpt-oss-120b-131072",
        messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}]
    )

    print(response.choices[0].message.content)
    ```
    

!!! tip "Discovering Available Models"
The endpoint information can be accessed using the [Metis status page](https://metis.alcf.anl.gov/status). It provides the status of the endpoints and the models and the associated configurations.

The list of currently supported chat-completion models on Metis are : 
- gpt-oss-120b-131072
- Llama-4-Maverick-17B-128E-Instruct

You can programmatically query all available models and endpoints:

```bash
    access_token=$(python inference_auth_token.py get_access_token)
    curl -X GET "https://inference-api.alcf.anl.gov/resource_server/list-endpoints" \
         -H "Authorization: Bearer ${access_token}" | jq -C '.clusters.metis'
```
If you need any other models to be provisioned via these endpoints, please reach out to support[at]alcf.anl.gov.

See SambaNova's documentation for additional information to supplement the instructions below: [OpenAI compatible API](https://docs.sambanova.ai/sambastudio/latest/open-ai-api.html).



