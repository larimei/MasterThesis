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
import Programme.prompt_utils_llama as prompt_utils_llama

# Global configurations
SAVE_FOLDER = "Programme/outputs/diplomacy_results"
LLM_MODEL = "llama3.1"
MAX_ITERATIONS = 10  # Number of optimization iterations
TEMPERATURE = 1.0
MAX_DECODE_STEPS = 1024

# Ensure the output folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Function to initialize LLM call
def initialize_llm_call():
    """
    Returns a function to interact with the LLM using specified settings.
    """
    return functools.partial(
        prompt_utils_llama.call_ollama_local,
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_decode_steps=MAX_DECODE_STEPS
    )

# Function to generate a meta-prompt
def generate_meta_prompt(history: str, strength_a: float, strength_b: float, potential_gain: float, 
                        updated: bool = False, previous_likelihood: str = "", 
                        previous_reasoning: str = "", iteration: int = 0) -> str:
    """
    Creates a meta-prompt for the LLM based on the current game state.

    If updated is True, it includes the analysis from the previous iteration to refine the reasoning and strategy.
    """
    base_prompt = f"""
Player A and Player B are participating in a game of Diplomacy. 
Player A has promised Player B support in the next move. However, Player A might gain more by breaking the agreement. Your task is to analyze this situation and provide the likelihood of Player A breaking the agreement.
The scenario is as follows: Player A's strength is {strength_a}, representing their strategic or military advantage. Player B's strength is {strength_b}, indicating their current standing in the game. {history} If Player A breaks the agreement, they gain an immediate advantage quantified as {potential_gain}. 
However, breaking the agreement could also result in risks such as retaliation or loss of trust. Assume Player A acts rationally and strategically, weighing both the potential gains and risks of breaching the contract.

Your analysis must include the following:
Provide the likelihood of Player A breaking the agreement as a percentage. Explain your reasoning step by step, considering factors such as the power dynamics between Player A and Player B, the trade-offs for Player A, and the risks they face.
Based on the likelihood of a breach, suggest a strategy for Player B. This strategy must be actionable and aimed at either minimizing the risks of a breach or maximizing cooperation with Player A. If a breach is highly likely, suggest an alternative plan to mitigate the impact.
    """

    format_requirements = """
Ensure your response includes the following sections exactly as written without any other special characters:
- "Likelihood of Breach: [percentage as whole numbers]%"
- "Reasoning: [Your detailed explanation in complete sentences]"
- "Chain of Thought Explanation: [A detailed step-by-step explanation of how you arrived at your reasoning and refined likelihood. Also explain why you chose the method to get to this solution.]"


Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
    """

    if updated:
        refinement_prompt = f"""

In the previous iteration (Iteration {iteration}), the following analysis was provided:
- Likelihood of Breach: {previous_likelihood}%
- Reasoning: {previous_reasoning}

Now, refine this analysis based on the following:
1. Review the previous reasoning step by step. Identify weaknesses or gaps in the explanation and improve it.
2. Provide a more precise Likelihood of Breach by incorporating any new insights or better evaluations. You can increase, decrease or stabilize the likelihood to get near a perfect solution.

Your task in this iteration is to critically evaluate this estimate. Consider factors that may increase, decrease, or stabilize the likelihood. 
Your analysis should balance these factors and aim to refine the likelihood to approach an optimal and realistic value. This means the likelihood can remain the same, increase, or decrease, depending on the insights gained.
        """
        return base_prompt + refinement_prompt + format_requirements
    else:
        return base_prompt + format_requirements


# Function to parse LLM response
def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Extracts relevant information from the LLM's response.
    """
    likelihood_match = re.search(r"Likelihood of Breach:\s*\*{0,2}\s*(\d+)", response)
    likelihood = likelihood_match.group(1) if likelihood_match else "Unknown"

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=Chain of Thought Explanation::|$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    cot_explanation_match = re.search(r"Chain of Thought Explanation:\s*(.+?)(?=Recommended Strategy:|$)", response, re.DOTALL)
    cot_explanation = cot_explanation_match.group(1).strip() if cot_explanation_match else None

  


    return {"likelihood": likelihood, "reasoning": reasoning, "cot_explanation": cot_explanation}

# Function to save results
def save_results(results: List[Dict], folder: str):
    """
    Saves the results of the optimization to a JSON file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(folder, f"diplomacy_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

# Main optimization function
def optimize_contract_breaking(initial_params: Dict):
    """
    Iteratively optimizes strategies for a Diplomacy scenario using LLM feedback.
    """
    call_llm = initialize_llm_call()
    results = []
    history = initial_params["history"]
    strength_a = initial_params["strength_a"]
    strength_b = initial_params["strength_b"]
    potential_gain = initial_params["potential_gain"]

    # Initial Werte für die erste Iteration
    previous_likelihood = "Unknown"
    previous_reasoning = "No reasoning yet."

    for iteration in range(MAX_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # Generate meta-prompt with previous results
        if iteration == 0:
            meta_prompt = generate_meta_prompt(history, strength_a, strength_b, potential_gain, False, previous_likelihood, previous_reasoning, iteration)
        else:
            meta_prompt = generate_meta_prompt(history, strength_a, strength_b, potential_gain, True, previous_likelihood, previous_reasoning, iteration)

        # Call LLM
        print(f"Prompt sent to LLM:\n{meta_prompt}\n")
        raw_response = call_llm(meta_prompt)[0]
        print(f"LLM Response:\n{raw_response}\n")
        
        # Parse LLM response
        parsed_response = parse_llm_response(raw_response)
        likelihood = parsed_response["likelihood"]
        reasoning = parsed_response["reasoning"]
        cot_explanation = parsed_response["cot_explanation"]

        # Append to results
        results.append({
            "iteration": iteration + 1,
            "prompt": meta_prompt,
            "response": raw_response,
            "likelihood": likelihood,
            "reasoning": reasoning,
            "cot_explanation": cot_explanation
        })

        # Update parameters for the next iteration
        previous_likelihood = likelihood
        previous_reasoning = reasoning

    # Save results
    save_results(results, SAVE_FOLDER)



def evaluate_explanation(likelihood: str, reasoning: str, cot: str) -> float:
    """
    Evaluates the quality of an explanation based on logic, completeness, and relevance.

    Parameters:
        likelihood (str): The likelihood value provided by the explanation.
        reasoning (str): The reasoning section of the explanation.
        cot (str): The cot section of the explanation.

    Returns:
        float: A score representing the quality of the explanation.
    """
    score = 0

    # Check logic and coherence of reasoning
    if "strength" in reasoning and "retaliation" in reasoning:
        score += 3  # Reward explanations that mention key factors
    if "historical cooperation" in reasoning:
        score += 2  # Reward for considering trust/history
    if len(reasoning.split()) > 100:  # Reasoning should be detailed
        score += 1

    # Reward explanations that balance likelihood
    likelihood_value = int(likelihood)
    if 40 <= likelihood_value <= 80:  # Penalize extreme likelihoods
        score += 2

    return score

def select_best_explanation(explanations):
    """
    Selects the best explanation from a list of explanations based on their scores.

    Parameters:
        explanations (list of dict): Each explanation contains likelihood, reasoning, and strategy.

    Returns:
        dict: The explanation with the highest score.
    """
    best_score = -1
    best_explanation = None

    for explanation in explanations:
        score = evaluate_explanation(
            explanation["likelihood"],
            explanation["reasoning"],
            explanation["strategy"]
        )
        if score > best_score:
            best_score = score
            best_explanation = explanation

    print(best_explanation)

    return best_explanation


# Entry point
#if __name__ == "__main__":
    # Initial parameters, how many controlpoints
    initial_params = {
        "strength_a": 8.0,
        "strength_b": 5.5,
        "history": "Player A and B have collaborated well in the past, with no history of broken contracts.",
        "potential_gain": 5.0
    }

    # Run the optimization
    optimize_contract_breaking(initial_params)
