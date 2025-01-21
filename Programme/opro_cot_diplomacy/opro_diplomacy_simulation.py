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

    Your response must follow this format exactly, so the information can be parsed and processed for further analysis. Also use the colon.
            """


def generate_optimization_communication_prompt(name, messages):
    return f"""
       You are analyzing communication in a Diplomacy game for phase {name}. Below are the communication logs for this phase:
    {messages}

    Your task:
    1. Assess the clarity, persuasiveness, and strategic depth of the messages. Highlight any ambiguities, risks, or missed opportunities in the communication. 
       Suggest ways to improve trust-building and persuasion in these interactions. Provide actionable recommendations to improve the players' overall communication strategies in this phase.
    2. Analyze recurring communication strategies or negotiation techniques used by the players. Identify patterns that lead to successful alliances or conflicts. Explain why these would be successful and expound why you analyzed 
       it this way.
    3. Summarize the strategic impact of the communication and suggest improvements to the messages. Point out the messages that woul be more successfull if changed. Also explain why rhe message could be better and how you geot to 
       this conclusion.
    4. Identify key sentences, exchanges, or patterns that had a significant influence on the decision-making process during this phase. Explain why these sentences were pivotal and how they shaped the players' actions or strategies.
       Discuss any implicit or explicit agreements, power dynamics, or promises reflected in the messages.

    Output format:
 
    - Communication Tips: [List of actionable tips to improve overall communication in this phase. Detailed explanation of the patterns, their strategic impact, and how the communication can be improved.]
    - Reasoning: [Detailed explanation of the patterns and their strategic impact, including the explanation of the improvement for the messages]
    - Optimized Messages: [Examples of improved communication for this phase]
    - Highlights: [List of key sentences or phrases that significantly influence the game's outcome. Also explain exactly and step-by-step why this is a key sentence and can change the outcome of the game]
    
    Important:
    - Provide detailed step-by-step explanations for all analyses and suggestions and always explain your decisions for your solutions.
    - Ensure that all critical points are backed by reasoning, highlighting the context and implications of each message.
    - Write your response in a clear, structured, and logical format that adheres to the requested output format. Use colons where appropriate to improve readability and ensure consistency.
    - Ensure that the order is exactliy as written.
        """


def generate_game_summary_prompt(results):
    phase_summaries = []

    for phase in results[0].get("phases", "Unknown Phases"):
        phase_name = phase.get("phase", "Unknown Phase")
        highlights = phase.get("highlights", "")
        communication_tips = phase.get("communcation_tips", "")
        reasoning = phase.get("reasoning", "")

        phase_summaries.append(
            f"""
        Phase: {phase_name}
        Highlights:
        {highlights}

        Communication Tips:
        {communication_tips}

        Reasoning:
        {reasoning}
        """
        )

    consolidated_results = "".join(phase_summaries)
    print(consolidated_results)

    return f"""

    {consolidated_results}

    Your goals:
    1. Provide a comprehensive summary:
       - Summarize the overall communication and strategies across all phases.
       - Highlight critical decisions and their impact on the game's progression.
       - Use clear and concise sentences that explain the logic behind key moves and strategies.

    2. Identify key patterns and improvements:
       - Analyze the communication to uncover recurring patterns, such as alliances, betrayals, or negotiation tactics.
       - Discuss how these patterns influenced the players' decisions.
       - Suggest specific improvements in communication strategies to achieve better outcomes.

    3. Extract impactful highlights:
       - Identify the most significant highlights that directly influenced the game's outcome.
       - For each highlight, explain step-by-step why it was impactful, referencing specific moves, agreements, or conflicts.
       - Provide detailed reasoning for each highlight, ensuring full transparency.

    4. Offer actionable future suggestions:
       - Provide clear recommendations for improving communication in similar scenarios.
       - Focus on building trust, enhancing negotiation skills, and aligning communication with strategic objectives.

    Output format:
    - Overall Summary: [Your summary of the game's communication, highlights and strategies, written in clear, logical sentences. Explain all your optimization and make a transparent explanation of how you came to this decision.        Summarize the whole game and deliver clear, transparent and logical main points.]
    - Key Patterns: [Patterns or improvements observed across phases, explained with specific examples.]
    - Impactful Highlights: [List and explain the most impactful highlights step-by-step, focusing on why they were crucial and how they shaped the game's outcome.]
    - Future Suggestions: [Practical recommendations for improving communication and strategic decision-making in future games.]

    Important:
    - Provide step-by-step explanations for each analysis to ensure clarity and transparency.
    - Use precise, complete sentences to explain each point thoroughly.
    - Avoid vague statements; back up each claim with reasoning or examples from the provided phase results.
    - Ensure your response adheres to the exact output format for structured and actionable insights.
    """


def generate_moves_analysis_prompt(messages, actions):

    actions_summary = "\n".join(
        [f"{player}: {action}" for player, action in actions.items()]
    )
    messages_summary = "\n".join(
        [
            f"From {msg['sender']} to {msg['recipient']}: {msg['message']}"
            for msg in messages
        ]
    )

    return f"""
    You are analyzing the communication and moves in a Diplomacy game.

    Messages:
    {messages_summary}

    Player Actions:
    {actions_summary}

    Your task:
    1. Analyze how the players' moves align or conflict with their messages and Identify discrepancies or consistencies between promises and actions.
    2. Evaluate how these moves influence trust, strategy, and cooperation for the next phase.
    3. Explain optimization in communication, with the insighhts you gained by the orders the actions the players made step-by-step. Alos explain your decision making in the long-term-process.

    Output format:
    - Analysis: [Detailed explanation of the alignment or conflict between messages and moves]
    - Trust Impact: [How the moves affect trust between players]
    - Optimization: [Detailed Optimization of the player's communication]

    Your response must follow this format exactly, so the information can be parsed and processed for further analysis. Also use the colon.
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

            message_results.append(
                {
                    "iteration": iteration + 1,
                    "original_message": content,
                    "optimized_message": optimized_message,
                    "reasoning": reasoning,
                    "sender": sender,
                    "recipient": recipient,
                }
            )

    return message_results


def optimize_communication(messages, ask_llm, phase_name):

    # message_results = optimize_messages(messages, ask_llm)

    all_messages = " ".join(
        [
            f"From {msg['sender']} to {msg['recipient']}: {msg['message']}"
            for msg in messages
            if isinstance(msg, dict)
            and "sender" in msg
            and "recipient" in msg
            and "message" in msg
        ]
    )

    prompt = generate_optimization_communication_prompt(phase_name, all_messages)
    response = ask_llm(prompt)[0]
    print(response)
    parsed_response = parse_llm_response_communication(response)

    highlights = parsed_response["highlights"]
    communcation_tips = parsed_response["communcation_tips"]
    reasoning = parsed_response["reasoning"]
    new_messages = parsed_response["new_messages"]

    return {
        "phase": phase_name,
        # "message_results": message_results,
        "communication_tips": communcation_tips,
        "reasoning": reasoning,
        "new_messages": new_messages,
        "highlights": highlights,
    }


def optimize_communication_moves(game_data, ask_llm):

    phases = []
    for phase in game_data["phases"]:
        messages = phase.get("messages", [])

        if len(messages) != 0:
            actions = phase.get("orders", {})

            phase_name = phase.get("name", "Unknown Phase")
            print(f"Current Phase: {phase_name}")

            phase_results = optimize_communication(messages, ask_llm, phase_name)

            prompt = generate_moves_analysis_prompt(messages, actions)
            response = ask_llm(prompt)[0]

            parsed_response = parse_llm_response_moves(response)

            phase_results_moves = {
                "analysis": parsed_response["analysis"],
                "trust_impact": parsed_response["trust_impact"],
                "optimization": parsed_response["optimization"],
            }

            phases.append({**phase_results, **phase_results_moves})

    return phases


def parse_llm_response_message(response: str) -> Dict[str, str]:

    message_match = re.search(
        r"Optimized Message[\s:]*([\s\S]*?)(?=Reasoning:|$)", response, re.DOTALL
    )
    message = message_match.group(1).strip() if message_match else "Unknown"

    reasoning_match = re.search(r"Reasoning[\s:]*([\s\S]*?)(?=$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unknown"

    return {"message": message, "reasoning": reasoning}


def parse_llm_response_communication(response: str) -> Dict[str, str]:

    communication_tips_match = re.search(
        r"Communication Tips[\s:]*([\s\S]*?)(?=Reasoning:|$)", response, re.DOTALL
    )
    communication_tips = (
        communication_tips_match.group(1).strip()
        if communication_tips_match
        else "Unknown"
    )

    reasoning_match = re.search(
        r"Reasoning[\s:]*([\s\S]*?)(?=Optimized Messages:|$)", response, re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unknown"

    new_messages_match = re.search(
        r"Optimized Messages[\s:]*([\s\S]*?)(?=Highlights:|$)", response, re.DOTALL
    )
    new_messages = (
        new_messages_match.group(1).strip() if new_messages_match else "Unknown"
    )

    highlights_match = re.search(
        r"Highlights[\s:]*([\s\S]*?)(?=$)", response, re.DOTALL
    )
    highlights = highlights_match.group(1).strip() if highlights_match else "Unknown"

    return {
        "communcation_tips": communication_tips,
        "reasoning": reasoning,
        "new_messages": new_messages,
        "highlights": highlights,
    }


def parse_llm_response_summary(response: str) -> Dict[str, str]:

    overall_summary_match = re.search(
        r"Overall Summary[\s:]*([\s\S]*?)(?=Key Patterns:|$)", response, re.DOTALL
    )
    overall_summary = (
        overall_summary_match.group(1).strip() if overall_summary_match else "Unknown"
    )

    key_patterns_match = re.search(
        r"Key Patterns[\s:]*([\s\S]*?)(?=Impactful Highlights:|$)", response, re.DOTALL
    )
    key_patterns = (
        key_patterns_match.group(1).strip() if key_patterns_match else "Unknown"
    )

    impactful_highlights_match = re.search(
        r"Impactful Highlights[\s:]*([\s\S]*?)(?=Future Suggestions:|$)",
        response,
        re.DOTALL,
    )
    impactful_highlights = (
        impactful_highlights_match.group(1).strip()
        if impactful_highlights_match
        else "Unknown"
    )

    future_suggestions_match = re.search(
        r"Future Suggestions[\s:]*([\s\S]*?)(?=$)", response, re.DOTALL
    )
    future_suggestions = (
        future_suggestions_match.group(1).strip()
        if future_suggestions_match
        else "Unknown"
    )

    return {
        "overall_summary": overall_summary,
        "key_patterns": key_patterns,
        "impactful_highlights": impactful_highlights,
        "future_suggestions": future_suggestions,
    }


def parse_llm_response_moves(response: str) -> Dict[str, str]:

    analysis_match = re.search(
        r"Analysis[\s:]*([\s\S]*?)(?=Trust Impact:|$)", response, re.DOTALL
    )
    analysis = analysis_match.group(1).strip() if analysis_match else "Unknown"

    trust_impact_match = re.search(
        r"Trust Impact[\s:]*([\s\S]*?)(?=Optimization:|$)",
        response,
        re.DOTALL,
    )
    trust_impact = (
        trust_impact_match.group(1).strip() if trust_impact_match else "Unknown"
    )

    optimization_match = re.search(
        r"Optimization[\s:]*([\s\S]*?)(?=$)", response, re.DOTALL
    )
    optimization = (
        optimization_match.group(1).strip() if optimization_match else "Unknown"
    )

    return {
        "analysis": analysis,
        "trust_impact": trust_impact,
        "optimization": optimization,
    }


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")

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
        max_decode_steps=MAX_DECODE_STEPS,
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

    all_results = [{"phases": optimize_communication_moves(game_data, ask_llm)}]

    summary_prompt = generate_game_summary_prompt(all_results)

    summary_response = ask_llm(summary_prompt)[0]

    game_summary = parse_llm_response_summary(summary_response)

    all_results.append(game_summary)

    save_results(all_results, output_dir)
