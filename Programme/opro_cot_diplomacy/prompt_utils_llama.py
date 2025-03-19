import re
import requests
import json
from typing import List, Union, Dict, Optional

# Globale Variable für die Konversationshistorie
_conversation_histories = {}  # Speichert Geschichte pro Modell: model_name -> history


def call_ollama_local_single_prompt(
    prompt: str,
    model: str = "deepseek-r1:8b",
    session: bool = True,
) -> str:

    global _conversation_histories

    # Initialisiere die History für dieses Modell falls nötig
    if model not in _conversation_histories:
        _conversation_histories[model] = []

    try:
        # API-URL für Ollama
        url = "http://localhost:11434/api/chat"

        # Bereite die Nachrichten vor
        messages = []

        # Füge die Konversationshistorie hinzu, wenn session=True
        if session and _conversation_histories[model]:
            messages.extend(_conversation_histories[model])

        # Füge den aktuellen Prompt hinzu
        messages.append({"role": "user", "content": prompt})

        # Bereite die Anfrage vor
        data = {"model": model, "messages": messages, "stream": False}

        # Sende die Anfrage
        response = requests.post(url, json=data, timeout=60)

        # Prüfe auf HTTP-Fehler
        response.raise_for_status()

        # Parse die Antwort
        result = response.json()

        if "message" in result and "content" in result["message"]:
            response_text = result["message"]["content"]

            # Aktualisiere die Konversationshistorie, wenn session=True
            if session:
                _conversation_histories[model].append(
                    {"role": "user", "content": prompt}
                )
                _conversation_histories[model].append(
                    {"role": "assistant", "content": response_text}
                )

            return response_text
        else:
            error_msg = f"Unerwartetes Antwortformat von der API: {json.dumps(result)}"
            print(f"Error: {error_msg}")
            return f"Error: {error_msg}"

    except requests.exceptions.ConnectionError:
        error_msg = "Verbindung zum Ollama-Server fehlgeschlagen. Läuft der Server? (ollama serve)"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Fehler bei API-Aufruf: {str(e)}"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"


def call_ollama_local(
    inputs: Union[str, List[str]],
    model: str = "deepseek-r1:8b",
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
        print("Konversationshistorie für alle Modelle gelöscht.")
    elif model in _conversation_histories:
        _conversation_histories[model] = []
        print(f"Konversationshistorie für Modell '{model}' gelöscht.")
    else:
        print(f"Keine Konversationshistorie für Modell '{model}' gefunden.")


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
