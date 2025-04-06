import traceback
import prompt_utils
import data_handler
import processing
import os

SAVE_FOLDER = "Programme/outputs/diplomacy_results"
DEBUG_DIR = "debug_outputs"


def main():
    provider_options = ["ollama", "gemini"]
    provider_input = (
        input(f"Enter LLM provider ({'/'.join(provider_options)}, default: gemini): ")
        .strip()
        .lower()
        or "gemini"
    )

    provider_models = {
        "ollama": ["llama3.1", "deepseek-r1:8b"],
        "gemini": ["models/gemini-1.5-pro", "models/gemini-1.5-flash"],
    }

    default_model = (
        provider_models[provider_input][0]
        if provider_input in provider_models
        else "llama3.1"
    )

    model_options = "/".join(provider_models.get(provider_input, ["deepseek-r1:8b"]))
    model_prompt = f"Enter {provider_input.capitalize()} model ({model_options}, default: {default_model}): "
    model_input = input(model_prompt).strip() or default_model

    ask_llm = lambda prompt: prompt_utils.call_llm(
        prompt, provider=provider_input, model=model_input
    )

    print(
        f"\n======== Testing LLM connection ({provider_input}/{model_input}) ==========="
    )
    test_output = ask_llm("Does the sun rise from the north? Just answer yes or no.")
    print(f"Test response: {test_output}")
    print("========================================\n")

    input_file = (
        input("Enter the input JSON file path (default: diplomacy.json): ")
        or "diplomacy.json"
    )

    output_dir = input("Enter output directory (default: outputs): ") or "outputs"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)

    try:
        print(f"Loading data from {input_file}...")
        game_data = data_handler.load_data(input_file)

        print("Processing game data...")
        phases = processing.process_game_data(game_data, ask_llm, 1)

        all_results = [{"phases": phases}]

        print("Generating game summary...")
        game_summary = processing.generate_game_summary(all_results, ask_llm)
        all_results.append(game_summary)

        results_path = data_handler.save_results(all_results, output_dir)
        print(f"All processing complete! Results saved to: {results_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
