import datetime
import functools
import json
import os

import re
import sys
from typing import Dict

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import prompt_utils_llama as prompt_utils_llama

os.chdir(os.path.dirname(os.path.realpath(__file__)))

SAVE_FOLDER = "Programme/outputs/diplomacy_results"
LLM_MODEL = "llama3.1"
TEMPERATURE = 1.0
MAX_DECODE_STEPS = 1024
ITERATION = 1

def generate_optimization_message_prompt(sender, recipient, message):
    return f"""
            You are optimizing communication in a Diplomacy game.

            Original Message:
    From: {sender}
    To: {recipient}
    Message: {message}

    Your task:
    1. Rewrite this message to maximize persuasion, trust, and cooperation.
    2. Ensure the rewritten message is subtle and strategic.
    3. Explain step-by-step why your changes improve the effectiveness of the message and how you came to this explanation.
    4. Ensure your explanation points out the risks and the enhancements you made.

    Output format:
    - Optimized Message: [Your rewritten message]
    - Reasoning: [Explanation of why this rewrite improves persuasion and cooperation]

    Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
            """

def generate_optimization_communication_prompt(name, messages):
    return f"""
       You are analyzing communication in a Diplomacy game for phase {name}. Below are the communication logs for this phase:
    {messages}

    Your task:
    1. Analyze the communication and extract key sentences or phrases that significantly influence the game's outcome.
    2. Identify patterns or strategies that improve communication and provide actionable tips.
    3. Summarize the strategic impact of the communication and suggest improvements to the messages.

    Output format:
    - Communication Tips: [List of actionable tips to improve overall communication in this phase]
    - Reasoning: [Detailed explanation of the patterns and their strategic impact, including the explanation of the improvement for the messages]
    - Optimized Messages: [Examples of improved communication for this phase]
    - Highlights: [List of key sentences or phrases that significantly influence the game's outcome. Also explain exactly and step-by-step why this is a key sentence and can change the outcome of the game]

    Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
        """

def generate_game_summary_prompt(results):
    phase_summaries = []

    print(results)

    for phase in results:
        phase_name = phase.get("phase", "Unknown Phase")
        highlights = "\n".join(phase.get("highlights", []))
        communication_tips = phase.get("communcation_tips", "")
        reasoning = phase.get("reasoning", "")

        phase_summaries.append(f"""
        Phase: {phase_name}
        Highlights:
        {highlights}

        Communication Tips:
        {communication_tips}

        Reasoning:
        {reasoning}
        """)

    consolidated_results = "\n\n".join(phase_summaries)

    return f"""
    You are summarizing a Diplomacy game based on the following phase results:

    {consolidated_results}

    Your task:
    1. Provide a summary of the overall communication and strategies across all phases.
    2. Identify key patterns or improvements in the communication process.
    3. Extract the most impactful highlights that influenced the game's outcome.
    4. Suggest how these insights could be applied to future games.

    Output format:
    - Overall Summary: [Your summary of the game's communication and strategies]
    - Key Patterns: [Patterns or improvements observed across phases]
    - Impactful Highlights: [Most impactful highlights from the game]
    - Future Suggestions: [Recommendations for improving communication in similar scenarios]
    """

def generate_moves_analysis_prompt(messages, actions):
  
    actions_summary = "\n".join([f"{player}: {action}" for player, action in actions.items()])
    messages_summary = "\n".join([f"From {msg['sender']} to {msg['recipient']}: {msg['message']}" for msg in messages])

    return f"""
    You are analyzing the communication and moves in a Diplomacy game.

    Messages:
    {messages_summary}

    Player Actions:
    {actions_summary}

    Your task:
    1. Analyze how the players' moves align or conflict with their messages.
    2. Identify discrepancies or consistencies between promises and actions.
    3. Evaluate how these moves influence trust, strategy, and cooperation for the next phase.

    Output format:
    - Analysis: [Detailed explanation of the alignment or conflict between messages and moves]
    - Trust Impact: [How the moves affect trust between players]
    - Strategic Recommendations: [Recommendations for improving communication and trust in the next phase]
    """



def optimize_messages(messages, ask_llm):
    message_results = [] 

    for iteration in range(ITERATION):
        print(f"=== Iteration {iteration + 1} ===")     

        for message in messages:
            sender = message.get("sender")
            recipient = message.get("recipient")
            content = message.get("message")

            prompt = generate_optimization_message_prompt(sender, recipient, content)
            response = ask_llm(prompt)[0]
            parsed_response = parse_llm_response_message(response)

            optimized_message = parsed_response["message"]
            reasoning = parsed_response["reasoning"]

            
            message_results.append({
                "iteration": iteration + 1,
                "original_message": content,
                "optimized_message": optimized_message,
                "reasoning": reasoning,
                "sender": sender,
                "recipient": recipient
            })

    return message_results




def optimize_communication(messages, ask_llm, phase_name):

    #message_results = optimize_messages(messages, ask_llm)

    all_messages = " ".join([
        f"From {msg['sender']} to {msg['recipient']}: {msg['message']}" 
        for msg in messages 
        if isinstance(msg, dict) and "sender" in msg and "recipient" in msg and "message" in msg
    ])

    prompt = generate_optimization_communication_prompt(phase_name, all_messages)
    response = ask_llm(prompt)[0]
    parsed_response = parse_llm_response_communication(response)

    highlights = parsed_response["highlights"]
    communcation_tips = parsed_response["communcation_tips"]
    reasoning = parsed_response["reasoning"]
    new_messages = parsed_response["new_messages"]


    return {"phase": phase_name,
            #"message_results": message_results,
            "communcation_tips": communcation_tips, 
            "reasoning": reasoning, 
            "new_messages": new_messages,
            "highlights": highlights
    }

def optimize_communication_moves(game_data, ask_llm):
    all_results = []

    for phase in game_data["phases"]:
        phase_name = phase.get("name", "Unknown Phase")
        print(f"Current Phase: {phase_name}")


        messages = phase.get("messages", [])
        actions = phase.get("orders", {})


        phase_results = optimize_communication(messages, ask_llm, phase_name)

        prompt = generate_moves_analysis_prompt(messages, actions)
        response = ask_llm(prompt)[0]

        parsed_response = parse_llm_response_moves(response)

        phase_results_moves = {
            "analysis": parsed_response["analysis"],
            "trust_impact": parsed_response["trust_impact"],
            "strategic_recommendations": parsed_response["strategic_recommendations"]
        }
        
        all_results.append({**phase_results_moves, **phase_results})


    return all_results


def parse_llm_response_message(response: str) -> Dict[str, str]:

    message_match = re.search(r"Optimized Message:\s*(.+?)(?=Reasoning:|$)", response, re.DOTALL)
    message = message_match.group(1).strip() if message_match else "Unknown"

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unknown"

    return {"message": message, "reasoning": reasoning}

def parse_llm_response_communication(response: str) -> Dict[str, str]:

    communication_tips_match = re.search(r"Communication Tips:\s*(.+?)(?=Reasoning:|$)", response, re.DOTALL)
    communication_tips = communication_tips_match.group(1).strip() if communication_tips_match else "Unknown"

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=New Messages:|$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unknown"

    new_messages_match = re.search(r"Optimized Messages:\s*(.+?)(?=Highlights:|$)", response, re.DOTALL)
    new_messages = new_messages_match.group(1).strip() if new_messages_match else "Unknown"

    highlights_match = re.search(r"Highlights:\s*(.+?)(?=$)", response, re.DOTALL)
    highlights = highlights_match.group(1).strip() if highlights_match else "Unknown"

    return {"communcation_tips": communication_tips, "reasoning": reasoning, "new_messages": new_messages, "highlights": highlights}

def parse_llm_response_summary(response: str) -> Dict[str, str]:

    overall_summary_match = re.search(r"Overall Summary:\s*(.+?)(?=Key Patterns:|$)", response, re.DOTALL)
    overall_summary = overall_summary_match.group(1).strip() if overall_summary_match else "Unknown"

    key_patterns_match = re.search(r"Key Patterns:\s*(.+?)(?=Impactful Highlights:|$)", response, re.DOTALL)
    key_patterns = key_patterns_match.group(1).strip() if key_patterns_match else "Unknown"

    impactful_highlights_match = re.search(r"Impactful Highlights:\s*(.+?)(?=Future Suggestions:|$)", response, re.DOTALL)
    impactful_highlights = impactful_highlights_match.group(1).strip() if impactful_highlights_match else "Unknown"

    future_suggestions_match = re.search(r"Future Suggestions:\s*(.+?)(?=$)", response, re.DOTALL)
    future_suggestions = future_suggestions_match.group(1).strip() if future_suggestions_match else "Unknown"

    return {
        "overall_summary": overall_summary,
        "key_patterns": key_patterns,
        "impactful_highlights": impactful_highlights,
        "future_suggestions": future_suggestions
    }

def parse_llm_response_moves(response: str) -> Dict[str, str]:

    analysis_match = re.search(r"Analysis:\s*(.+?)(?=Trust Impact:|$)", response, re.DOTALL)
    analysis = analysis_match.group(1).strip() if analysis_match else "Unknown"

    trust_impact_match = re.search(r"Trust Impact:\s*(.+?)(?=Strategic Recommendations:|$)", response, re.DOTALL)
    trust_impact = trust_impact_match.group(1).strip() if trust_impact_match else "Unknown"

    strategic_recommendations_match = re.search(r"Strategic Recommendations:\s*(.+?)(?=$)", response, re.DOTALL)
    strategic_recommendations = strategic_recommendations_match.group(1).strip() if strategic_recommendations_match else "Unknown"

    return {
        "analysis": analysis,
        "trust_impact": trust_impact,
        "strategic_recommendations": strategic_recommendations
    }


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"diplomacy_results_{timestamp}.json")

    with open(results_path, "w") as results_file:
        json.dump(results, results_file, indent=4)
    print(f"Results saved to {results_path}")


def load_data(filepath):
    with open(filepath, "r") as file:
        return json.load(file)


if __name__ == "__main__":

    ask_llm = functools.partial(
        prompt_utils_llama.call_ollama_local,
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_decode_steps=MAX_DECODE_STEPS
    )

    # ====================== test calling the servers ============================
    print("\n======== testing the optimizer server ===========")
    optimizer_test_output = ask_llm(
        "Does the sun rise from the north? Just answer yes or no."
    )
    print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the optimizer server.")
    print("\n=================================================")


    input_file = "diplomacy_short.json"
    output_dir = "outputs"

    game_data = load_data(input_file)

    all_results = optimize_communication_moves(game_data, ask_llm)

    summary_prompt = generate_game_summary_prompt(all_results)

    summary_response = ask_llm(summary_prompt)[0]

    game_summary = parse_llm_response_summary(summary_response)

    all_results.append(game_summary)

    save_results(all_results, output_dir)
