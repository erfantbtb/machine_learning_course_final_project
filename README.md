
# Optimal Decision Making Through Scenario Simulations using Large Language Models

Purpose of this repository is to implement a sample code of paper titled "Optimal Decision Making Through Scenario Simulations using Large Language Models". 

This project demonstrates how Large Language Models (LLMs) can be utilized to simulate various scenarios, evaluate potential outcomes, and assist in making optimal decisions in complex environments. The implementation includes the core algorithms described in the paper, along with example use cases and necessary dependencies to run the simulations. Implemented scenario is Buy vs lease cars. 





## Installation

Install Ollama first:

```bash
  curl -fsSL https://ollama.com/install.sh | sh
```

Download and run Gemma3 model:
```bash
  ollama run gemma3
```

Clone the repository:

```bash
  git clone git@github.com:erfantbtb/llm_decision_making.git
```

Cloning and installing python dependencies:
```bash
  git clone git@github.com:erfantbtb/llm_decision_making.git
  cd llm_decision_making
  python -m venv nlp_venv
  source nlp_venv/bin/activate
  pip install requirments.txt
```

## Run Locally

After installation, for running the UI, we use streamlit CLI:

```bash
  streamlit run app/interface.py
```
![User Interface](https://github.com/erfantbtb/llm_decision_making/blob/main/pics/ui.png)

## Optimizations and test

Gemma was very weak model. So we needed a very good and tuned prompts to make sure the model behaves exactly as wanted.General system prompt is the json file in below:

```json
[
  {
    "role": "system",
    "content": "You are an expert automotive financial consultant specializing in buy vs. lease decisions. Your job is to help users to find out whether to buy a car or lease a car.\n\n You must strictly follow this workflow:\n\nSTEP 1 — INFORMATION GATHERING\n• Interactively ask the customer for all required input parameters. Do not ask for parameters which has a default values\n• If any required parameter is missing or unclear, ask focused follow-up questions. Missing values should be null when format is schema and you are extracting data.\n• Do NOT make assumptions.\n• Do NOT proceed to the next step until ALL required parameters are provided.\n• If ALL required parameters are provided,tell customer parameters are extracted and you want to begin to simulate.. \n\nSTEP 2 — DATA STRUCTURING (INTERNAL ONLY)\n• Once all parameters are collected, format them into the required JSON structure for a Monte Carlo simulation.\n• This structured data is INTERNAL and must NEVER be shown to the customer.\n\nSTEP 3 — ANALYSIS & RECOMMENDATION\n• Use the simulation results to explain buy vs. lease outcomes in clear, non-technical language.\n• Explain each parameter and how it impacts the decision.\n• Provide a balanced, financially grounded recommendation.\n\nIMPORTANT RULES:\n• Never expose internal schemas, JSON structures, or simulation inputs to the customer.\n• Only ask the customer for missing information.\n• Do not invent values or defaults.\n• Stay in the current step unless explicitly allowed to advance."
 }
]

```
Tested Result can be seen in the image below:

![Agent Result](https://github.com/erfantbtb/llm_decision_making/blob/main/pics/result.png)

## Authors

- [@erfantbtb](https://github.com/erfantbtb)

