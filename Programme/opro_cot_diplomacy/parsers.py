import re
import os
import datetime
from typing import Dict, List


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

    debug_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "debug_outputs"
    )
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
    pattern_dict = {
        "DIPLOMACY-SECTION-ANALYSIS": "DIPLOMACY-SECTION-MOVES-SUMMARY:",
        "DIPLOMACY-SECTION-MOVES-SUMMARY": "DIPLOMACY-SECTION-TRUST-IMPACT:",
        "DIPLOMACY-SECTION-TRUST-IMPACT": "DIPLOMACY-SECTION-OPTIMIZATION:",
        "DIPLOMACY-SECTION-OPTIMIZATION": "$",
    }
    result = parse_llm_response(response, pattern_dict)
    return {
        "analysis": result.get("diplomacy_section_analysis", ""),
        "moves_summary": result.get("diplomacy_section_moves_summary", ""),
        "trust_impact": result.get("diplomacy_section_trust_impact", ""),
        "optimization": result.get("diplomacy_section_optimization", ""),
    }


def parse_llm_response_summary(response: str) -> Dict[str, str]:
    pattern_dict = {
        "DIPLOMACY-SECTION-OVERALL-SUMMARY": "DIPLOMACY-SECTION-GAME-SUMMARY:",
        "DIPLOMACY-SECTION-GAME-SUMMARY": "DIPLOMACY-SECTION-KEY-PATTERNS:",
        "DIPLOMACY-SECTION-KEY-PATTERNS": "DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS:",
        "DIPLOMACY-SECTION-IMPACTFUL-HIGHLIGHTS": "DIPLOMACY-SECTION-FUTURE-SUGGESTIONS:",
        "DIPLOMACY-SECTION-FUTURE-SUGGESTIONS": "$",
    }
    result = parse_llm_response(response, pattern_dict)
    return {
        "overall_summary": result.get("diplomacy_section_overall_summary", ""),
        "game_summary": result.get("diplomacy_section_game_summary", ""),
        "key_patterns": result.get("diplomacy_section_key_patterns", ""),
        "impactful_highlights": result.get(
            "diplomacy_section_impactful_highlights", ""
        ),
        "future_suggestions": result.get("diplomacy_section_future_suggestions", ""),
    }
