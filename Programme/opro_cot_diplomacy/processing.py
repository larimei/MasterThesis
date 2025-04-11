from typing import Dict, List, Callable, Tuple, Optional

from parsers import (
    parse_llm_response_communication,
    parse_llm_response_moves,
    parse_llm_response_summary,
    extract_thinking,
)
from prompts import generate_prompt


def optimize_communication(
    messages: List[Dict], ask_llm: Callable, phase_name: str
) -> Dict:
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
        # Use original ask_llm function but unpack the first item only
        response = ask_llm(prompt)[0]
        parsed_response = parse_llm_response_communication(response)

        result = {
            "phase": phase_name,
            "communication_tips": parsed_response["communication_tips"],
            "reasoning": parsed_response["reasoning"],
            "simplified_summary": parsed_response["simplified_summary"],
            "new_messages": parsed_response["optimized_messages"],
            "highlights": parsed_response["highlights"],
        }

        # Add thinking if present
        if "thinking" in parsed_response:
            result["thinking"] = parsed_response["thinking"]

        return result
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


def analyze_moves(messages: List[Dict], actions: Dict, ask_llm: Callable) -> Dict:
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


def generate_game_summary(results: List[Dict], ask_llm: Callable) -> Dict:
    phase_summaries = []

    for phase in results[0].get("phases", []):
        phase_name = phase.get("phase", "Unknown Phase")
        highlights = phase.get("highlights", "")
        communication_tips = phase.get("communication_tips", "")
        reasoning = phase.get("reasoning", "")
        simplified_summary = phase.get("simplified_summary", "")
        moves_summary = phase.get("moves_summary", "")

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


def process_game_data(
    game_data: Dict, ask_llm: Callable, iterations: int = 1
) -> List[Dict]:
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
