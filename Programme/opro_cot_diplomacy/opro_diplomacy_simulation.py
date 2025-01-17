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


def optimize_messages(messages, ask_llm):
    message_results = [] 

    for iteration in range(ITERATION):
        print(f"=== Iteration {iteration + 1} ===")     

        for message in messages:
            sender = message.get("sender")
            recipient = message.get("recipient")
            content = message.get("message")
            print(f"Original Message from {sender} to {recipient}: {content}")

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


def optimize_communication(game_data, ask_llm):
    all_results = []

    for phase in game_data["phases"]:
        phase_name = phase.get("name", "Unknown Phase")
        print(f"Current Phase: {phase_name}")

        messages = phase.get("messages", [])

        phase_results = optimize_messages(messages, ask_llm)

        # Extrahiere die relevanten Daten aus den Nachrichten und füge Absender und Empfänger hinzu
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


        all_results.append({
            "phase": phase_name,
            "phase_results": phase_results,
            "communcation_tips": communcation_tips, 
            "reasoning": reasoning, 
            "new_messages": new_messages,
            "highlights": highlights
        })


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

    results = optimize_communication(game_data, ask_llm)

    save_results(results, output_dir)
