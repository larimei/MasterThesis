# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess


def call_ollama_local_single_prompt(
    prompt, model="llama3.1", max_decode_steps=1024, temperature=1.0
):
    """
    Calls the local Ollama model for a single prompt.

    Args:
        prompt (str): The input string to be processed by the model.
        model (str): The local model name (default: "llama3.1").
        max_decode_steps (int): Maximum number of tokens to decode.
        temperature (float): Sampling temperature for response generation.

    Returns:
        str: Model's response.
    """
    try:
        command = ["ollama", "run", "llama3.1", prompt]

        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        output = process.stdout.decode("utf-8")

        output_filtered = "\n".join(
            [
                line
                for line in output.splitlines()
                if "Das Handle ist ungültig" not in line
            ]
        )

        return output_filtered

    except subprocess.CalledProcessError as e:
        print(f"Error calling Ollama model: {e.stderr}")
        return ""

    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""


def call_ollama_local(inputs, model="llama3.1", max_decode_steps=1024, temperature=1.0):
    """
    Calls the local Ollama model for a list of input prompts.

    Args:
        inputs (list or str): The input string or list of strings to be processed by the model.
        model (str): The local model name (default: "llama3.1").
        max_decode_steps (int): Maximum number of tokens to decode.
        temperature (float): Sampling temperature for response generation.

    Returns:
        list: List of responses for each input prompt.
    """

    if isinstance(inputs, str):
        inputs = [inputs]

    outputs = []
    for input_str in inputs:
        output = call_ollama_local_single_prompt(
            input_str,
            model=model,
            max_decode_steps=max_decode_steps,
            temperature=temperature,
        )
        outputs.append(output)
    return outputs
