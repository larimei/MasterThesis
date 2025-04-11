import json
import os
import datetime
from typing import Dict, List, Any


def load_data(filepath: str) -> Dict:
    try:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            data_path = os.path.join(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                os.path.basename(filepath),
            )
            with open(data_path, "r", encoding="utf-8") as file:
                return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading data from {filepath}: {str(e)}")
        raise


def save_results(
    results: List[Dict],
    output_dir: str,
    llm_model: str,
    file: str,
    max_size_bytes: int = 10_000_000,
) -> str:

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    is_deepseek = "deepseek-r1:8b" in llm_model.lower()

    structured_results = {
        "analysis_metadata": {
            "timestamp": timestamp,
            "llm_model": llm_model,
            "input_file": file,
        },
        "phase_analyses": [],
        "game_summary": {
            "overall_summary": results[1].get("overall_summary", ""),
            "game_summary": results[1].get("game_summary", ""),
            "key_patterns": results[1].get("key_patterns", ""),
            "impactful_highlights": results[1].get("impactful_highlights", ""),
            "future_suggestions": results[1].get("future_suggestions", ""),
        },
    }

    if is_deepseek and "thinking" in results[1]:
        structured_results["game_summary"]["thinking"] = results[1].get("thinking", "")

    for phase in results[0].get("phases", []):
        phase_analysis = {
            "phase_id": phase.get("phase", "Unknown Phase"),
            "communication_analysis": {
                "communication_tips": phase.get("communication_tips", ""),
                "reasoning": phase.get("reasoning", ""),
                "simplified_summary": phase.get("simplified_summary", ""),
                "highlights": phase.get("highlights", ""),
            },
            "optimized_messages": phase.get("new_messages", ""),
        }

        if is_deepseek and "thinking" in phase:
            phase_analysis["communication_analysis"]["thinking"] = phase.get(
                "thinking", ""
            )

        if "analysis" in phase:
            phase_analysis["moves_analysis"] = {
                "analysis": phase.get("analysis", ""),
                "moves_summary": phase.get("moves_summary", ""),
                "trust_impact": phase.get("trust_impact", ""),
                "optimization": phase.get("optimization", ""),
            }

            if is_deepseek and "thinking" in phase and "analysis" in phase:
                phase_analysis["moves_analysis"]["thinking"] = phase.get("thinking", "")

        phase_analysis["transparency_metrics"] = {
            "reasoning_length": len(phase.get("reasoning", "")),
            "has_explanation": bool(phase.get("reasoning", "")),
            "has_optimization": bool(phase.get("new_messages", "")),
            "has_highlights": bool(phase.get("highlights", "")),
            "has_thinking": is_deepseek and bool(phase.get("thinking", "")),
        }

        structured_results["phase_analyses"].append(phase_analysis)

    structured_results["transparency_metrics"] = {
        "total_phases_analyzed": len(results[0].get("phases", [])),
        "has_game_summary": bool(results[1].get("overall_summary", "")),
        "has_key_patterns": bool(results[1].get("key_patterns", "")),
        "has_future_suggestions": bool(results[1].get("future_suggestions", "")),
        "has_thinking": is_deepseek
        and any("thinking" in phase for phase in results[0].get("phases", [])),
    }

    main_file_path = os.path.join(output_dir, f"results_{timestamp}.json")
    try:
        json_str = json.dumps(structured_results, indent=2, ensure_ascii=False)

        if len(json_str.encode("utf-8")) <= max_size_bytes:
            with open(main_file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"Ergebnisse gespeichert in: {main_file_path}")
            return output_dir

        print(f"JSON ist zu groß. Teile in separate Dateien auf...")

        game_summary = structured_results.pop("game_summary")
        summary_path = os.path.join(output_dir, f"game_summary_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(game_summary, f, indent=2, ensure_ascii=False)

        structured_results["game_summary_file"] = os.path.basename(summary_path)

        with open(main_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)

        print(f"Hauptdaten gespeichert in: {main_file_path}")
        print(f"Game Summary gespeichert in: {summary_path}")

    except Exception as e:
        print(f"Fehler: {str(e)}")

    return output_dir
