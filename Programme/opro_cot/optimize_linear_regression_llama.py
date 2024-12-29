# Diplomacy Contract Optimization using LLMs
# This script iteratively optimizes strategies in a Diplomacy context by interacting with an LLM.

import os
import re
import json
import datetime
from typing import List, Dict
import functools
import numpy as np



# Global configurations
SAVE_FOLDER = "outputs/diplomacy_results"
LLM_MODEL = "llama3.1"
MAX_ITERATIONS = 5  # Number of optimization iterations
TEMPERATURE = 1.0
MAX_DECODE_STEPS = 1024

# Ensure the output folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

import datetime
import functools
import json
import os
import re
import sys

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

# Import LLM utilities (adapt this to your environment)
import Programme.prompt_utils_llama as prompt_utils_llama
from absl import app
from absl import flags
import numpy as np

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "ollama-local", "The name of the optimizer LLM."
)


def main(_):
  # ============== set optimization experiment configurations ================
  num_points = 50  # number of points in linear regression
  w_true = 15  # the true w
  b_true = 14  # the true b
  max_num_steps = 5  # the number of optimization steps
  num_reps = 5  # the number of repeated runs
  max_num_pairs = 20  # the maximum number of input-output pairs in meta-prompt
  num_input_decimals = 0  # num of decimals for input values in meta-prompt
  num_output_decimals = 0  # num of decimals for output values in meta-prompt
  num_generated_points_in_each_step = 8

  # ================ load LLM settings ===================
  optimizer_llm_name = _OPTIMIZER.value
  optimizer_llm_name == "llama3.1"

  # =================== create the result directory ==========================
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      OPRO_ROOT_PATH,
      "outputs",
      "optimization-results",
      f"linear_regression-o-{optimizer_llm_name}-{datetime_str}/",
  )
  os.makedirs(save_folder)
  print(f"result directory:\n{save_folder}")

  # ====================== optimizer model configs ============================
  optimizer_llm_dict = {
      "model_type": optimizer_llm_name.lower(),
      "max_decode_steps": 1024,
      "temperature": 1.0,
      "batch_size": 1
  }

  call_optimizer_server_func = functools.partial(
      prompt_utils_llama.call_ollama_local,
      model="llama3.1",
      temperature=optimizer_llm_dict["temperature"],
      max_decode_steps=optimizer_llm_dict["max_decode_steps"]
  )

  # ====================== try calling the servers ============================
  print("\n======== testing the optimizer server ===========")
  optimizer_test_output = call_optimizer_server_func(
      "Does the sun rise from the north? Just answer yes or no."
  )
  print(f"optimizer test output: {optimizer_test_output}")
  print("Finished testing the optimizer server.")
  print("\n=================================================")


  # ====================== utility functions ============================
  def evaluate_loss(X, y, w, b):  # pylint: disable=invalid-name
    residual = y - (X * w + b)
    return np.linalg.norm(residual) ** 2

  def gen_meta_prompt(
      old_value_pairs_set,
      X,  # pylint: disable=invalid-name, unused-argument
      y,  # pylint: disable=unused-argument
      num_input_decimals=5,
      num_output_decimals=5,
      max_num_pairs=100,
  ):
    """Generate the meta-prompt for optimization.

    Args:
     old_value_pairs_set (set): the set of old (w, b, z) pairs.
     X (np.array): the 1D array of x values.
     y (np.array): the 1D array of y values.
     num_input_decimals (int): the number of decimals for (w, b) in the
       meta-prompt.
     num_output_decimals (int): the number of decimals for z in the meta-prompt.
     max_num_pairs (int): the maximum number of exemplars in the meta-prompt.

    Returns:
      meta_prompt (str): the generated meta-prompt.
    """
    old_value_pairs_set = set(
        [  # pylint: disable=g-complex-comprehension
            (
                np.round(w, num_input_decimals)
                if num_input_decimals > 0
                else int(w),
                np.round(b, num_input_decimals)
                if num_input_decimals > 0
                else int(b),
                np.round(z, num_output_decimals)
                if num_output_decimals > 0
                else int(z),
            )
            for w, b, z in old_value_pairs_set
        ]
    )
    old_value_pairs = list(old_value_pairs_set)
    old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[2])[
        -max_num_pairs:
    ]
    old_value_pairs_substr = ""
    for w, b, z in old_value_pairs:
      old_value_pairs_substr += f"\ninput:\nw={w}, b={b}\nvalue:\n{z}\n"
    meta_prompt = """
  Now you will help me minimize a function with two input variables w, b. I have some (w, b) pairs and the function values at those points. The pairs are arranged in descending order based on their function values, where lower values are better.
    """.strip()
    meta_prompt += "\n\n"
    meta_prompt += old_value_pairs_substr.strip()
    meta_prompt += "\n\n"
    # function_analytic_form = ""
    # for xi, yi in zip(X, y):
    #   function_analytic_form += f"({yi:.4f} - ({xi:.4f} * w + b)) ** 2 + "
    # function_analytic_form = function_analytic_form[:-3]
    # meta_prompt += (
    #     "The function has the analytic form f(w, b) ="
    #     f" {function_analytic_form}. When evaluating the value of a (w, b)"
    #     " pair, you should replace the w and b in the analytic form with your"
    #     " values and do the computation."
    # )
    # meta_prompt += "\n\n"

    meta_prompt += """
    Additionally, explain your reasoning step-by-step when generating the new (w, b) pair. Provide the reasoning before giving the pair, and ensure the reasoning is logical and coherent. End your explanation with the pair [w, b].
    """.strip()
    meta_prompt += """Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than any of the above. Do not write code. The output must end with a pair [w, b], where w and b are numerical values.
    """.strip()
    
    return meta_prompt

  def extract_string_in_square_brackets(input_string):
    raw_result = re.findall(r"\[.*?\]", input_string)
    if raw_result:
      for pair in raw_result[::-1]:
        if "=" not in pair and ("w" in pair or "b" in pair):
          continue
        return pair[1:-1]
      return ""
    else:
      return ""
    
  
  def extract_cot_and_output(raw_response):
    """
    Extract the Chain of Thought (CoT) explanation and the resulting [w, b] pair.
    """
    cot_explanation = re.split(r"\[.*?\]", raw_response, maxsplit=1)[0].strip()
    extracted_output = extract_string_in_square_brackets(raw_response)
    return cot_explanation, extracted_output

  def parse_output(extracted_output):
    """Parse the extracted output 'w, b' string to np.array([w, b]).

    Args:
      extracted_output (str): the extracted output string, like '1.5, 2.5'.

    Returns:
      parsed_output (np.array): the parsed output in a numpy array, like [1.5,
      2.5].
    """
    if not extracted_output:
      return
    extracted_values = []
    for item in extracted_output.split(","):
      if "=" in item:
        item = item[item.index("=") + 1 :]
      extracted_values.append(item.strip())
    parsed_output = np.array(extracted_values).astype(float)
    return parsed_output

  configs_dict = dict()
  results_dict = dict()
  num_convergence_steps = []
  for i_rep in range(num_reps):
    found_optimal = False
    print(f"\nRep {i_rep}:")

    # ================= generate the ground truth X, y =====================
    X = np.arange(num_points).astype(float) + 1  # pylint: disable=invalid-name
    np.random.seed(i_rep + 1)
    y = X * w_true + b_true + np.random.randn(num_points)
    loss_at_true_values = evaluate_loss(X, y, w_true, b_true)
    print(f"value at (w_true, b_true): {loss_at_true_values}")

    # ================= generate the starting points =====================
    num_starting_points = 5  # the number of initial points for optimization
    np.random.seed((i_rep + 1) * 10)
    init_w = np.random.uniform(low=10, high=20, size=num_starting_points)
    np.random.seed((i_rep + 1) * 100)
    init_b = np.random.uniform(low=10, high=20, size=num_starting_points)

    # ====================== run optimization ============================
    configs_dict_single_rep = {
        "optimizer_llm_configs": optimizer_llm_dict,
        "data": {
            "num_points": num_points,
            "w_true": w_true,
            "b_true": b_true,
            "loss_at_true_values": loss_at_true_values,
            "X": list(X),
            "y": list(y),
        },
        "init_w": list(init_w),
        "init_b": list(init_b),
        "max_num_steps": max_num_steps,
        "max_num_pairs": max_num_pairs,
        "num_input_decimals": num_input_decimals,
        "num_output_decimals": num_output_decimals,
        "num_generated_points_in_each_step": num_generated_points_in_each_step,
    }
    configs_dict[i_rep] = configs_dict_single_rep
    configs_json_path = os.path.join(save_folder, "configs.json")
    print(f"saving configs to\n{configs_json_path}")
    with open(configs_json_path, "w") as f:
      json.dump(configs_dict, f, indent=4)

    old_value_pairs_set = set()
    old_value_pairs_with_i_step = []  # format: [(w, b, z = f(w, b), i_step)]
    meta_prompts_dict = dict()  # format: {i_step: meta_prompt}
    raw_outputs_dict = dict()  # format: {i_step: raw_outputs}
    cot_explanations_dict = dict()

    rounded_inits = [
        (np.round(w, num_input_decimals), np.round(b, num_input_decimals))
        for w, b in zip(init_w, init_b)
    ]
    rounded_inits = [
        tuple(item) for item in list(np.unique(rounded_inits, axis=0))
    ]
    for w, b in rounded_inits:
      z = evaluate_loss(X, y, w, b)
      old_value_pairs_set.add((w, b, z))
      old_value_pairs_with_i_step.append((w, b, z, -1))

    print("\n================ run optimization ==============")
    print(
        f"initial points: {[tuple(item[:2]) for item in old_value_pairs_set]}"
    )
    print(f"initial values: {[item[-1] for item in old_value_pairs_set]}")
    results_json_path = os.path.join(save_folder, "results.json")
    print(f"saving results to\n{results_json_path}")

    for i_step in range(max_num_steps):
      print(f"\nStep {i_step}:")
      meta_prompt = gen_meta_prompt(
          old_value_pairs_set,
          X,
          y,
          num_input_decimals=num_input_decimals,
          num_output_decimals=num_output_decimals,
          max_num_pairs=max_num_pairs,
      )
      if not i_step % 5:
        print("\n=================================================")
        print(f"meta_prompt:\n{meta_prompt}")
      meta_prompts_dict[i_step] = meta_prompt

      # generate a maximum of the given number of points in each step
      remaining_num_points_to_generate = num_generated_points_in_each_step
      raw_outputs = []
      while remaining_num_points_to_generate > 0:
        raw_outputs += call_optimizer_server_func(meta_prompt)
        remaining_num_points_to_generate -= optimizer_llm_dict["batch_size"]
      raw_outputs = raw_outputs[:num_generated_points_in_each_step]

      parsed_outputs = []
      cot_explanations = []

      for raw_response in raw_outputs:
        cot_explanation, extracted_output = extract_cot_and_output(raw_response)
        cot_explanations.append(cot_explanation)
        
        try:
            parsed_output = parse_output(extracted_output)
            if parsed_output is not None and len(parsed_output) == 2:
                parsed_outputs.append(parsed_output)
        except ValueError:
            pass

      raw_outputs_dict[i_step] = raw_outputs
      meta_prompts_dict[i_step] = meta_prompt
      cot_explanations_dict[i_step] = cot_explanations

      for string in raw_outputs:
        if not i_step % 5:
          print("\n=================================================")
          print("raw output:\n", string)
          print("\n=================================================")
        try:
          parsed_output = parse_output(
              extract_string_in_square_brackets(string)
          )
          if parsed_output is not None and len(parsed_output) == 2:
            parsed_outputs.append(parsed_output)
        except ValueError:
          pass
      parsed_outputs = [tuple(item) for item in parsed_outputs]
      print(f"proposed points before rounding: {parsed_outputs}")

      # round the proposed points to the number of decimals in meta-prompt
      rounded_outputs = [
          (np.round(w, num_input_decimals), np.round(b, num_input_decimals))
          for w, b in parsed_outputs
      ]
      rounded_outputs = [
          tuple(item) for item in list(np.unique(rounded_outputs, axis=0))
      ]
      print(f"proposed points after rounding: {rounded_outputs}")

      # evaluate the values of proposed and rounded outputs
      single_step_values = []
      for w, b in rounded_outputs:
        if w == w_true and b == b_true:
          found_optimal = True
        z = evaluate_loss(X, y, w, b)
        single_step_values.append(z)
        old_value_pairs_set.add((w, b, z))
        old_value_pairs_with_i_step.append((w, b, z, i_step))
      print(f"single_step_values: {single_step_values}")

      # ====================== save results ============================
      results_dict_single_rep = {
          "meta_prompts": meta_prompts_dict,
          "raw_outputs": raw_outputs_dict,
          "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
          "cot_explanations": cot_explanations_dict,
      }
      results_dict[i_rep] = results_dict_single_rep
      with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
      if found_optimal:
        print(
            f"Repetition {i_rep+1}, optimal found at Step {i_step+1}, saving"
            f" final results to\n{save_folder}"
        )
        num_convergence_steps.append(i_step + 1)
        break
  print(f"num_convergence_steps: {num_convergence_steps}")


if __name__ == "__main__":
  app.run(main)
