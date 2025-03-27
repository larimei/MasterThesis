import datetime
import functools
import json
import os

import re
import sys
from typing import Dict, List

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from Programme.opro_cot_diplomacy import prompt_utils_gemini
import prompt_utils_gemini as prompt_utils_gemini

os.chdir(os.path.dirname(os.path.realpath(__file__)))

SAVE_FOLDER = "Programme/outputs/diplomacy_results"
LLM_MODEL = "deepseek-r1:8b"
ITERATION = 1


def parse_llm_response(response: str, pattern_dict: Dict[str, str]) -> Dict[str, str]:

    result = {}

    for section_name, next_section in pattern_dict.items():
        print(f"Looking for section: {section_name} until {next_section}")

        pattern = rf"{section_name}[\s:]*([\s\S]*?)(?={next_section}|$)"

        try:
            match = re.search(pattern, response, re.DOTALL)
            key_name = section_name.lower().replace(" ", "_").replace("-", "_")

            if match and match.group(1):
                result[key_name] = match.group(1).strip()
                print(
                    f"  Found section '{section_name}' with {len(result[key_name])} chars"
                )

        except Exception as e:
            print(f"Error parsing section '{section_name}': {str(e)}")
            result[key_name] = "Not provided"

    import os

    debug_dir = "debug_outputs"
    os.makedirs(debug_dir, exist_ok=True)

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{debug_dir}/raw_response_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(response)

    return result


def parse_llm_response_communication(response: str) -> Dict[str, str]:
    pattern_dict = {
        "DIPLOMACY-SECTION-COMMUNICATION-TIPS": "DIPLOMACY-SECTION-REASONING:",
        "DIPLOMACY-SECTION-REASONING": "DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY:",
        "DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY": "DIPLOMACY-SECTION-OPTIMIZED-MESSAGES:",
        "DIPLOMACY-SECTION-OPTIMIZED-MESSAGES": "DIPLOMACY-SECTION-HIGHLIGHTS:",
        "DIPLOMACY-SECTION-HIGHLIGHTS": "$",
    }
    result = parse_llm_response(response, pattern_dict)
    return {
        "communication_tips": result.get("diplomacy_section_communication_tips", ""),
        "reasoning": result.get("diplomacy_section_reasoning", ""),
        "simplified_summary": result.get("diplomacy_section_simplified_summary", ""),
        "optimized_messages": result.get("diplomacy_section_optimized_messages", ""),
        "highlights": result.get("diplomacy_section_highlights", ""),
    }


def parse_llm_response_moves(response: str) -> Dict[str, str]:
    """Parse moves analysis responses"""
    pattern_dict = {
        "DIPLOMACY-SECTION-ANALYSIS": "DIPLOMACY-SECTION-MOVES-SUMMARY:",  # Geändert
        "DIPLOMACY-SECTION-MOVES-SUMMARY": "DIPLOMACY-SECTION-TRUST-IMPACT:",  # Geändert
        "DIPLOMACY-SECTION-TRUST-IMPACT": "DIPLOMACY-SECTION-OPTIMIZATION:",
        "DIPLOMACY-SECTION-OPTIMIZATION": "$",
    }
    result = parse_llm_response(response, pattern_dict)
    return {
        "analysis": result.get("diplomacy_section_analysis", ""),
        "moves_summary": result.get("diplomacy_section_moves_summary", ""),  # Geändert
        "trust_impact": result.get("diplomacy_section_trust_impact", ""),
        "optimization": result.get("diplomacy_section_optimization", ""),
    }


def parse_llm_response_summary(response: str) -> Dict[str, str]:
    """Parse game summary responses"""
    pattern_dict = {
        "DIPLOMACY-SECTION-OVERALL-SUMMARY": "DIPLOMACY-SECTION-GAME-SUMMARY:",  # Geändert
        "DIPLOMACY-SECTION-GAME-SUMMARY": "DIPLOMACY-SECTION-KEY-PATTERNS:",  # Geändert
        "DIPLOMACY-SECTION-KEY-PATTERNS": "DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS:",
        "DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS": "DIPLOMACY-SECTION-FUTURE-SUGGESTIONS:",
        "DIPLOMACY-SECTION-FUTURE-SUGGESTIONS": "$",
    }
    result = parse_llm_response(response, pattern_dict)
    return {
        "overall_summary": result.get("diplomacy_section_overall_summary", ""),
        "game_summary": result.get("diplomacy_section_game_summary", ""),  # Geändert
        "key_patterns": result.get("diplomacy_section_key_patterns", ""),
        "impactful_highlights": result.get(
            "diplomacy_section_impactful_highlights", ""
        ),
        "future_suggestions": result.get("diplomacy_section_future_suggestions", ""),
    }


def generate_prompt(prompt_type: str, **kwargs) -> str:

    prompts = {
        "message_optimization": """
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
5. Provide a simple, structured summary of your reasoning with bullet points, focusing on the main improvements and why they matter.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-OPTIMIZED-MESSAGE: [Your rewritten message]
- DIPLOMACY-SECTION-REASONING: [Detailed explanation of why this rewrite improves persuasion and cooperation]
- DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY: [A bullet-point summary of the main improvements. First write a 1-2 sentence general assessment, then list 3-5 bullet points starting with '*' about specific improvements and why they matter. End with a concluding sentence about the overall impact.]

Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
""",
        "communication_optimization": """
You are analyzing communication in a Diplomacy game for phase {name}. Below are the communication logs for this phase:
{messages}

Your task:
1. Assess the clarity, persuasiveness, and strategic depth of the messages. Highlight any risks or missed opportunities in the communication. 
   Suggest ways to improve trust-building and persuasion in these interactions. Provide actionable recommendations to improve the players' overall communication strategies in this phase.
2. Analyze recurring communication strategies or negotiation techniques used by the players. Identify patterns that lead to successful alliances or conflicts. Explain why these would be successful and expound why you analyzed 
   it this way.
3. Summarize the strategic impact of the communication and suggest improvements to the messages. Point out the messages that would be more successful if changed. Also explain why the message could be better and how you got to 
   this conclusion.
4. Identify key sentences, exchanges, or patterns that had a significant influence on the decision-making process during this phase. Explain why these sentences were pivotal and how they shaped the players' actions or strategies.
   Discuss any implicit or explicit agreements, power dynamics, or promises reflected in the messages.
5. Provide a structured bullet-point summary of your analysis that is easy to understand for anyone. Start with a brief overview of the situation, then list the main issues and recommendations as bullet points, and end with a conclusion.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header)):
 
- DIPLOMACY-SECTION-COMMUNICATION-TIPS: [List of actionable tips to improve overall communication in this phase. Detailed explanation of the patterns, their strategic impact, and how the communication can be improved.]
- DIPLOMACY-SECTION-REASONING: [Detailed explanation of the patterns and their strategic impact, including the explanation of the improvement for the messages]
- DIPLOMACY-SECTION-SIMPLIFIED-SUMMARY: [Start with 1-2 sentences summarizing the key issue. Then list 3-5 bullet points starting with '*' that highlight the main problems and recommendations. End with a concluding sentence about how these changes would improve the player's position.]
- DIPLOMACY-SECTION-OPTIMIZED-MESSAGES: [Examples of improved communication for this phase]
- DIPLOMACY-SECTION-HIGHLIGHTS: [List of key sentences or phrases that significantly influence the game's outcome. Also explain exactly and step-by-step why this is a key sentence and can change the outcome of the game]

Important:
- Provide detailed step-by-step explanations for all analyses and suggestions and always explain your decisions for your solutions.
- Ensure that all critical points are backed by reasoning, highlighting the context and implications of each message.
- Write your response in a clear, structured, and logical format that adheres to the requested output format.
- In the simplified summary, use everyday language, avoid game jargon, and focus on clear actionable points.
- Ensure that the order is exactly as written.
""",
        "moves_analysis": """
You are analyzing the communication and moves in a Diplomacy game.

Messages:
{messages_summary}

Player Actions:
{actions_summary}

Your task:
1. Analyze how the players' moves align or conflict with their messages and Identify discrepancies or consistencies between promises and actions.
2. Evaluate how these moves influence trust, strategy, and cooperation for the next phase.
3. Explain optimization in communication, with the insights you gained by the orders the actions the players made step-by-step. Also explain your decision making in the long-term-process.
4. Provide a structured bullet-point summary that clearly explains your analysis in simple terms.

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-ANALYSIS: [Detailed explanation of the alignment or conflict between messages and moves]
- DIPLOMACY-SECTION-MOVES-SUMMARY: [Start with 1-2 sentences describing the situation. Then list 3-5 bullet points starting with '*' highlighting the main insights about the alignment between messages and moves. End with a sentence about the overall impact on the game.]
- DIPLOMACY-SECTION-TRUST-IMPACT: [How the moves affect trust between players]
- DIPLOMACY-SECTION-OPTIMIZATION: [Detailed Optimization of the player's communication]

Your response must follow this format exactly, so the information can be parsed and processed for further analysis.
""",
        "game_summary": """
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

5. Provide a structured bullet-point summary:
   - Create a simplified summary that captures the essence of your analysis
   - Use clear, everyday language without game jargon
   - Structure it with bullet points to highlight key issues and recommendations

Output format (IMPORTANT - always use these exact section headers without markdown or ther sybols befor or after the header):
- DIPLOMACY-SECTION-OVERALL-SUMMARY: [Your summary of the game's communication, highlights and strategies, written in clear, logical sentences. Explain all your optimization and make a transparent explanation of how you came to this decision. Summarize the whole game and deliver clear, transparent and logical main points.]
- DIPLOMACY-SECTION-GAME-SUMMARY: [Start with 1-2 sentences describing the overall game pattern. Then list 4-6 bullet points starting with '*' that highlight the main issues and recommendations from the game. End with a concluding sentence about the key lesson from this analysis.]
- DIPLOMACY-SECTION-KEY-PATTERNS: [Patterns or improvements observed across phases, explained with specific examples.]
- DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS: [List and explain the most impactful highlights step-by-step, focusing on why they were crucial and how they shaped the game's outcome.]
- DIPLOMACY-SECTION-FUTURE-SUGGESTIONS: [Practical recommendations for improving communication and strategic decision-making in future games.]

Important:
- Provide step-by-step explanations for each analysis to ensure clarity and transparency.
- Use precise, complete sentences to explain each point thoroughly.
- Avoid vague statements; back up each claim with reasoning or examples from the provided phase results.
- Make sure the simplified summary uses everyday language and focuses on actionable points.
- Ensure your response adheres to the exact output format for structured and actionable insights.
""",
    }

    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompts[prompt_type].format(**kwargs)


def optimize_communication(messages: List[Dict], ask_llm, phase_name: str) -> Dict:

    print(f"Optimizing communication for phase: {phase_name}")

    valid_messages = [
        msg
        for msg in messages
        if isinstance(msg, dict)
        and all(k in msg for k in ["sender", "recipient", "message"])
    ]

    if not valid_messages:
        print(f"Warning: No valid messages found for phase {phase_name}")
        return {
            "phase": phase_name,
            "communication_tips": "No messages to analyze",
            "reasoning": "No messages to analyze",
            "simplified_summary": "No messages were available for analysis.",
            "new_messages": "No messages to optimize",
            "highlights": "No messages to highlight",
        }

    all_messages = "\n".join(
        [
            f"From {msg['sender']} to {msg['recipient']}: {msg['message']}"
            for msg in valid_messages
        ]
    )

    prompt = generate_prompt(
        "communication_optimization", name=phase_name, messages=all_messages
    )

    try:
        response = ask_llm(prompt)[0]
        parsed_response = parse_llm_response_communication(response)

        return {
            "phase": phase_name,
            "communication_tips": parsed_response["communication_tips"],
            "reasoning": parsed_response["reasoning"],
            "simplified_summary": parsed_response["simplified_summary"],
            "new_messages": parsed_response["optimized_messages"],
            "highlights": parsed_response["highlights"],
        }
    except Exception as e:
        print(
            f"Error during communication optimization for phase {phase_name}: {str(e)}"
        )
        return {
            "phase": phase_name,
            "communication_tips": f"Error during analysis: {str(e)}",
            "reasoning": "Error during analysis",
            "simplified_summary": "An error occurred during the analysis process.",
            "new_messages": "Error during optimization",
            "highlights": "Error during highlight extraction",
        }


def generate_game_summary(results: List[Dict], ask_llm) -> Dict:

    phase_summaries = []

    for phase in results[0].get("phases", []):
        phase_name = phase.get("phase", "Unknown Phase")
        highlights = phase.get("highlights", "")
        communication_tips = phase.get("communication_tips", "")
        reasoning = phase.get("reasoning", "")
        simplified_summary = phase.get("simplified_summary", "")
        moves_summary = phase.get("moves_summary", "")  # Geändert

        phase_summaries.append(
            f"""
Phase: {phase_name}
Highlights:
{highlights}

Communication Tips:
{communication_tips}

Reasoning:
{reasoning}

Communication Summary:
{simplified_summary}

Moves Summary:
{moves_summary}
"""
        )

    consolidated_results = "".join(phase_summaries)

    prompt = generate_prompt("game_summary", consolidated_results=consolidated_results)
    response = ask_llm(prompt)[0]

    return parse_llm_response_summary(response)


def analyze_moves(messages: List[Dict], actions: Dict, ask_llm) -> Dict:

    actions_summary = "\n".join(
        [f"{player}: {action}" for player, action in actions.items()]
    )
    messages_summary = "\n".join(
        [
            f"From {msg['sender']} to {msg['recipient']}: {msg['message']}"
            for msg in messages
        ]
    )

    prompt = generate_prompt(
        "moves_analysis",
        messages_summary=messages_summary,
        actions_summary=actions_summary,
    )

    response = ask_llm(prompt)[0]
    return parse_llm_response_moves(response)


def process_game_data(game_data: Dict, ask_llm, iterations: int = 1) -> List[Dict]:

    phases = []

    for phase_idx, phase in enumerate(game_data.get("phases", [])):
        messages = phase.get("messages", [])
        if not messages:
            continue

        phase_name = phase.get("name", f"Phase {phase_idx}")
        print(
            f"\n==== Processing {phase_name} ({phase_idx + 1}/{len(game_data.get('phases', []))}) ===="
        )

        phase_results = optimize_communication(messages, ask_llm, phase_name)

        actions = phase.get("orders", {})
        if actions:
            moves_analysis = analyze_moves(messages, actions, ask_llm)
            phase_results.update(moves_analysis)

        phases.append(phase_results)

    return phases


def save_results(results: List[Dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    phases_path = os.path.join(output_dir, f"results_phases_{timestamp}.json")
    try:
        with open(phases_path, "w", encoding="utf-8") as phases_file:
            json.dump([results[0]], phases_file, indent=2, ensure_ascii=False)

        if len(results) > 1:
            summary_path = os.path.join(output_dir, f"results_summary_{timestamp}.json")
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump([results[1]], summary_file, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        # Try saving to a different location as fallback
        fallback_path = os.path.join(os.getcwd(), f"fallback_results_{timestamp}.json")
        with open(fallback_path, "w", encoding="utf-8") as fallback_file:
            json.dump(results, fallback_file, indent=2, ensure_ascii=False)
        print(f"Results saved to fallback location: {fallback_path}")
        return fallback_path

    print(f"Results successfully saved to {output_dir}")
    return output_dir


def load_data(filepath: str) -> Dict:

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading data from {filepath}: {str(e)}")
        raise


def main():
    LLM_MODEL = "deepseek-r1:8b"
    ITERATIONS = 1

    ask_llm = functools.partial(prompt_utils_gemini.call_ollama_local)

    print("\n======== Testing LLM connection ===========")
    test_output = ask_llm("Does the sun rise from the north? Just answer yes or no.")
    print(f"Test response: {test_output}")
    print("========================================\n")

    input_file = (
        input("Enter the input JSON file path (default: diplomacy.json): ")
        or "game_433761_ENGLAND_AG.json"
    )
    output_dir = input("Enter output directory (default: outputs): ") or "outputs"

    try:
        print(f"Loading data from {input_file}...")
        game_data = load_data(input_file)

        print("Processing game data...")
        phases = process_game_data(game_data, ask_llm, ITERATIONS)

        all_results = [{"phases": phases}]

        print("\nGenerating game summary...")
        game_summary = generate_game_summary(all_results, ask_llm)
        all_results.append(game_summary)

        results_path = save_results(all_results, output_dir)
        print(f"\nAll processing complete! Results saved to: {results_path}")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
