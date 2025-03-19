import re
import requests
import json
import google.generativeai as genai
from typing import List, Union, Dict, Optional

# Gemini API-Konfiguration
GEMINI_API_KEY = ""
DEFAULT_MODEL = "models/gemini-1.5-pro"

_conversation_histories = {}

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API erfolgreich initialisiert")
except Exception as e:
    print(f"Fehler bei der Initialisierung der Gemini API: {str(e)}")


def call_ollama_local_single_prompt(
    prompt: str,
    model: str = DEFAULT_MODEL,
    session: bool = True,
) -> str:

    global _conversation_histories

    if model not in _conversation_histories:
        _conversation_histories[model] = []

    try:
        gemini_model_name = DEFAULT_MODEL
        if ":" in model:
            pass

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
            },
        )

        if session and _conversation_histories[model]:

            chat_session = gemini_model.start_chat(
                history=_conversation_histories[model]
            )
            response = chat_session.send_message(prompt)
        else:

            response = gemini_model.generate_content(prompt)

        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = response.parts[0].text if response.parts else ""

        if session:
            if len(_conversation_histories[model]) == 0:

                _conversation_histories[model] = [
                    {"role": "user", "parts": [prompt]},
                    {"role": "model", "parts": [response_text]},
                ]
            else:

                _conversation_histories[model].append(
                    {"role": "user", "parts": [prompt]}
                )
                _conversation_histories[model].append(
                    {"role": "model", "parts": [response_text]}
                )

        return response_text

    except Exception as e:
        error_msg = f"Error at Gemini API: {str(e)}"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"


def call_ollama_local(
    inputs: Union[str, List[str]],
    model: str = DEFAULT_MODEL,
    session: bool = True,
) -> List[str]:

    if isinstance(inputs, str):
        inputs = [inputs]

    outputs = []
    for input_str in inputs:
        output = call_ollama_local_single_prompt(
            input_str,
            model=model,
            session=session,
        )
        outputs.append(output)
    return outputs


def clear_conversation_history(model: Optional[str] = None):

    global _conversation_histories

    if model is None:
        _conversation_histories = {}
    elif model in _conversation_histories:
        _conversation_histories[model] = []


def get_conversation_history(
    model: Optional[str] = None,
) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:

    global _conversation_histories

    if model is None:
        return _conversation_histories
    elif model in _conversation_histories:
        return _conversation_histories[model]
    else:
        return []
