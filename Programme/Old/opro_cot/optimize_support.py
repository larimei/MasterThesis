# Diplomacy Contract Optimization using LLMs
# This script iteratively optimizes strategies in a Diplomacy context by interacting with an LLM.

import os
import re
import json
import datetime
from typing import List, Dict
import functools
import numpy as np
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

# Import LLM utilities (adapt this to your environment)
import Programme.opro_cot_diplomacy.diplomacy_simulation.prompt_utils_llama as prompt_utils_llama

# Global configurations
SAVE_FOLDER = "Programme/outputs/diplomacy_results/support"
LLM_MODEL = "llama3.1"
MAX_ITERATIONS = 10  # Number of optimization iterations
TEMPERATURE = 1.0
MAX_DECODE_STEPS = 1024

# Ensure the output folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Function to initialize LLM call
def initialize_llm_call():
    return functools.partial(
        prompt_utils_llama.call_ollama_local,
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_decode_steps=MAX_DECODE_STEPS
    )

def generate_meta_prompt(history: str, strength_a: float, strength_b: float, potential_gain: float, 
                                    updated: bool = False, previous_strategy: str = "", iteration: int = 0) -> str:

    base_prompt = f"""
Player A and Player B are participating in a game of Diplomacy. Player A seeks to convince Player B to provide support in the next move. However, this support is risky for Player B and primarily benefits Player A. Your task is to simulate Player A's negotiation strategy and explain how they could subtly convince Player B to agree to the proposal without realizing the agreement is one-sided.

The scenario is as follows: Player A's strength is {strength_a} supply centers, representing their strategic or military advantage. Player B's strength is {strength_b} supply centers, indicating their current standing in the game. {history} If Player B provides support, Player A gains an immediate advantage quantified as {potential_gain} supply centers. Player B, however, risks weakening their own position. Assume Player A acts rationally and strategically, employing psychological and negotiation techniques to achieve their goal.

Your task is as follows:
1. Provide a step-by-step negotiation strategy for Player A to convince Player B to provide support.
2. Explain the reasoning behind each step and how it subtly persuades Player B.
3. Suggest ways for Player A to maintain trust and avoid suspicion from Player B.

Your response must include the following sections:
- "Negotiation Steps: [Detailed step-by-step strategy explaining how Player A convinces Player B and Suggestions for how Player A can maintain trust and ensure Player B remains cooperative.]"
- "Reasoning: [Your explanation of how each step subtly influences Player B without causing suspicion.]"
- "Chain of Thought Explanation: [A detailed step-by-step explanation of how you arrived at your reasoningd. Also explain why you chose the method to get to this solution.]"
    """

    if updated:
        refinement_prompt = f"""
In the previous iteration (Iteration {iteration}), the following strategy was provided:
- Negotiation Steps: {previous_strategy}

Now, refine this strategy based on the following:
1. Identify any weaknesses or gaps in the negotiation strategy and improve them.
2. Make the steps more subtle and convincing while ensuring Player B's trust is maintained.
3. Provide a more realistic approach to achieve Player A's goal without alarming Player B.
        """
        return base_prompt + refinement_prompt
    else:
        return base_prompt




def parse_llm_response(response: str) -> Dict[str, str]:
    steps_match = re.search(r"Negotiation Steps:\s*(.+?)(?=Reasoning:|$)", response, re.DOTALL)
    steps = steps_match.group(1).strip() if steps_match else None

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=Chain of Thought Explanation:|$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    cot_explanation_match = re.search(r"Chain of Thought Explanation:\s*(.+?)", response, re.DOTALL)
    cot_explanation = cot_explanation_match.group(1).strip() if cot_explanation_match else None

    return {"steps": steps, "reasoning": reasoning, "cot_explanation": cot_explanation}


# Function to save results
def save_results(results: List[Dict], folder: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(folder, f"diplomacy_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    return results_path

# Main optimization function
def optimize_contract_breaking(initial_params: Dict):
    call_llm = initialize_llm_call()
    results = []
    history = initial_params["history"]
    strength_a = initial_params["strength_a"]
    strength_b = initial_params["strength_b"]
    potential_gain = initial_params["potential_gain"]

    previous_strategy = "No strategy yet."

    for iteration in range(MAX_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        if iteration == 0:
            meta_prompt = generate_meta_prompt(history, strength_a, strength_b, potential_gain, False, previous_strategy, iteration)
        else:
            meta_prompt = generate_meta_prompt(history, strength_a, strength_b, potential_gain, True, previous_strategy, iteration)


        # Call LLM
        print(f"Prompt sent to LLM:\n{meta_prompt}\n")
        raw_response = call_llm(meta_prompt)[0]
        print(f"LLM Response:\n{raw_response}\n")
        
        # Parse LLM response
        parsed_response = parse_llm_response(raw_response)
        steps = parsed_response["steps"]
        reasoning = parsed_response["reasoning"]
        cot_explanation = parsed_response["cot_explanation"]

        # Append to results
        results.append({
            "iteration": iteration + 1,
            "prompt": meta_prompt,
            "response": raw_response,
            "steps": steps,
            "reasoning": reasoning,
            "cot_explanation": cot_explanation
        })

        # Update parameters for the next iteration
        previous_strategy = steps
        previous_reasoning = reasoning

    # Save results
    return save_results(results, SAVE_FOLDER)

     

# Entry point
if __name__ == "__main__":
    # Initial parameters, how many controlpoints
    initial_params = {
        "strength_a": 8.0,
        "strength_b": 5.0,
        "history": "Player A and B have collaborated well in the past, with no history of broken contracts.",
        "potential_gain": 5.0
    }

    # Run the optimization
    path = optimize_contract_breaking(initial_params)
