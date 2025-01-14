import functools
import json
import os

import sys

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
            Rewrite this message to maximize persuasion and cooperation.

            Output format:
            - Optimized Message: [Your rewritten message]
            """

def generate_optimization_communication_prompt(name, messages):
    return f"""
        You are analyzing communication in a Diplomacy game for phase {name}. This is the communication logs:
        {messages}

        Your task:
        1. Extract key sentences or highlights that significantly influence the game's outcome.
        2. Summarize the strategic impact of these highlights.

        Output format:
        - Highlights: [List of key sentences or phrases]
        - Summary: [Summary of the strategic impact]
        """



def optimize_messages(messages, ask_llm):
    message_results = []
    highlights = []    

    for iteration in range(ITERATION):
        print(f"=== Iteration {iteration + 1} ===")     

        for message in messages:
            sender = message.get("sender")
            recipient = message.get("recipient")
            content = message.get("message")
            print(f"Original Message from {sender} to {recipient}: {content}")

            prompt = generate_optimization_message_prompt(sender, recipient, content)
            response = ask_llm(prompt)
            parsed_response = parse_llm_response_message(response)

            optimize_message = parsed_response["optimize_message"]
            reasoning = parsed_response["reasoning"]

            
            message_results.append({
                "iteration": iteration + 1,
                "original_message": content,
                "optimized_message": optimize_message,
                "reasoning": reasoning,
                "sender": sender,
                "recipient": recipient
            })

    return message_results, highlights


def optimize_communication(game_data, ask_llm):
    all_results = []
    global_highlights = []

    for phase in game_data["phases"]:
        phase_name = phase.get("name", "Unknown Phase")
        print(f"Current Phase: {phase_name}")

        messages = phase.get("messages", [])

        phase_results = optimize_messages(messages, ask_llm)

        all_messages = " ".join([msg["optimized_message"] for msg in phase_results])
        prompt = generate_optimization_communication_prompt(phase_name, all_messages)
        response = ask_llm(prompt)
        parsed_response = parse_llm_response_communication(response)

        highlights = parsed_response[highlights]
        communcation_tips = parsed_response[communcation_tips]
        reasoning = parsed_response[reasoning]
        new_messages = parsed_response[new_messages]


        all_results.append({
            "phase": phase_name,
            "phase_results": phase_results,
            "communcation_tips": communcation_tips, 
            "reasoning": reasoning, 
            "new_messages": new_messages,
            "highlights": highlights
        })

        global_highlights.extend(highlights)

    return all_results, global_highlights

def parse_llm_response_message(response: str) -> Dict[str, str]:

    return {"message": message, "reasoning": reasoning}

def parse_llm_response_communication(response: str) -> Dict[str, str]:

    return {"communcation_tips": tips, "reasoning": reasoning, "new_messages": new_messages, "highlights": highlights}


def save_results(results, highlights, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "optimized_results.json")
    highlights_path = os.path.join(output_dir, "global_highlights.json")

    with open(results_path, "w") as results_file:
        json.dump(results, results_file, indent=4)
    print(f"Results saved to {results_path}")

    with open(highlights_path, "w") as highlights_file:
        json.dump(highlights, highlights_file, indent=4)
    print(f"Highlights saved to {highlights_path}")

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


    input_file = "diplomacy.json"
    output_dir = "outputs"

    game_data = load_data(input_file)

    results, highlights = optimize_communication(game_data, ask_llm)

    save_results(results, highlights, output_dir)
