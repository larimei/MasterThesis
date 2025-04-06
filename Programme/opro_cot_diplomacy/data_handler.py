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


def save_results(results: List[Dict], output_dir: str) -> str:
    """Speichert Ergebnisse in strukturierter JSON-Datei"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    structured_results = {
        "analysis_metadata": {
            "timestamp": timestamp,
            # "llm_model": config.get_config().get("llm_model", "unknown"),
            "input_file": "diplomacy.json",  # Hier den tatsächlichen Dateinamen einfügen
            "version": "1.0",
        },
        "game_analysis": {
            "overall_summary": results[1].get("overall_summary", ""),
            "key_patterns": results[1].get("key_patterns", ""),
            "phase_analyses": [],
        },
    }

    for phase in results[0].get("phases", []):
        phase_analysis = {
            "phase_id": phase.get("phase", "Unknown Phase"),
            "communications": {
                # Hier könntest du weitere Metadaten zur Kommunikation hinzufügen
            },
            "optimization_results": [],
            "key_insights": [],
            "transparency_metrics": {
                # Hier könntest du Metriken zur Erklärbarkeit hinzufügen
            },
        }

        structured_results["game_analysis"]["phase_analyses"].append(phase_analysis)

    output_path = os.path.join(output_dir, f"structured_results_{timestamp}.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_results, f, indent=2, ensure_ascii=False)
        print(f"Structured results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving structured results: {str(e)}")

    original_path = os.path.join(output_dir, f"original_results_{timestamp}.json")
    with open(original_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return output_dir
