import re
import requests
import json
from typing import List, Union, Dict, Optional

# Gemini API import und Konfiguration
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
    GEMINI_API_KEY = ""
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False

# Konstanten
DEFAULT_GEMINI_MODEL = "models/gemini-1.5-pro"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"
DEFAULT_OLLAMA_MODEL_LLAMA = "llama3.1"

# Globaler Zustand für Konversationshistorien
_ollama_conversation_histories = {}
_gemini_conversation_histories = {}


def call_ollama(prompt, model=DEFAULT_OLLAMA_MODEL, session=True):
    global _ollama_conversation_histories

    if model not in _ollama_conversation_histories:
        _ollama_conversation_histories[model] = []

    try:
        url = "http://localhost:11434/api/chat"
        messages = []

        if session and _ollama_conversation_histories[model]:
            messages.extend(_ollama_conversation_histories[model])

        messages.append({"role": "user", "content": prompt})
        data = {"model": model, "messages": messages, "stream": False}
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "message" in result and "content" in result["message"]:
            response_text = result["message"]["content"]

            if session:
                _ollama_conversation_histories[model].append(
                    {"role": "user", "content": prompt}
                )
                _ollama_conversation_histories[model].append(
                    {"role": "assistant", "content": response_text}
                )

            return response_text
        else:
            error_msg = f"Unerwartetes Antwortformat von der API: {json.dumps(result)}"
            print(f"Error: {error_msg}")
            return f"Error: {error_msg}"

    except requests.exceptions.ConnectionError:
        error_msg = "Verbindung zum Ollama-Server fehlgeschlagen. Läuft der Server?"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Fehler bei API-Aufruf: {str(e)}"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"


def call_gemini(prompt, model=DEFAULT_GEMINI_MODEL, session=True):
    if not GEMINI_AVAILABLE:
        return "Fehler: Gemini API ist nicht verfügbar."

    global _gemini_conversation_histories

    if model not in _gemini_conversation_histories:
        _gemini_conversation_histories[model] = []

    try:
        gemini_model_name = model
        if ":" in model:  # Falls Ollama-Modellname
            gemini_model_name = DEFAULT_GEMINI_MODEL

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "top_p": 0.95,
                "top_k": 40,
            },
        )

        if session and _gemini_conversation_histories[model]:
            chat_session = gemini_model.start_chat(
                history=_gemini_conversation_histories[model]
            )
            response = chat_session.send_message(prompt)
        else:
            response = gemini_model.generate_content(prompt)

        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = response.parts[0].text if response.parts else ""

        if session:
            if len(_gemini_conversation_histories[model]) == 0:
                _gemini_conversation_histories[model] = [
                    {"role": "user", "parts": [prompt]},
                    {"role": "model", "parts": [response_text]},
                ]
            else:
                _gemini_conversation_histories[model].append(
                    {"role": "user", "parts": [prompt]}
                )
                _gemini_conversation_histories[model].append(
                    {"role": "model", "parts": [response_text]}
                )

        return response_text

    except Exception as e:
        error_msg = f"Fehler bei Gemini API: {str(e)}"
        print(f"Error: {error_msg}")
        return f"Error: {error_msg}"


def call_llm(inputs, provider="ollama", model=None, session=True):
    if isinstance(inputs, str):
        inputs = [inputs]

    outputs = []
    for input_str in inputs:
        if provider.lower() == "ollama":
            output = call_ollama(input_str, model or DEFAULT_OLLAMA_MODEL, session)
        elif provider.lower() == "gemini":
            output = call_gemini(input_str, model or DEFAULT_GEMINI_MODEL, session)
        else:
            output = f"Error: Unbekannter Provider '{provider}'"

        outputs.append(output)

    return outputs


def clear_history(provider=None, model=None):
    global _ollama_conversation_histories, _gemini_conversation_histories

    if provider is None or provider.lower() == "ollama":
        if model is None:
            _ollama_conversation_histories = {}
        elif model in _ollama_conversation_histories:
            _ollama_conversation_histories[model] = []

    if provider is None or provider.lower() == "gemini":
        if model is None:
            _gemini_conversation_histories = {}
        elif model in _gemini_conversation_histories:
            _gemini_conversation_histories[model] = []


def get_history(provider, model=None):
    if provider.lower() == "ollama":
        if model is None:
            return _ollama_conversation_histories
        else:
            return _ollama_conversation_histories.get(model, [])
    elif provider.lower() == "gemini":
        if model is None:
            return _gemini_conversation_histories
        else:
            return _gemini_conversation_histories.get(model, [])
    else:
        return []


def call_ollama_local(inputs, model=DEFAULT_OLLAMA_MODEL, session=True):
    return call_llm(inputs, provider="ollama", model=model, session=session)


def call_ollama_local_single_prompt(prompt, model=DEFAULT_OLLAMA_MODEL, session=True):
    return call_ollama(prompt, model=model, session=session)
