# Instantiate agents and run the CoT process
from Programme.CoT_Opro import ChainOfThoughtManager, VerifierAgent, prompt_utils
from Programme.CoT_Opro.OptimizerAgent import OptimizationAgent
from Programme.CoT_Opro.ReasonerAgent import ReasonerAgent


import os
import sys

ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ROOT_PATH)

import datetime
import functools

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

    # ================ load LLM settings =================== //TODO mehr Settings
    optimizer_llm_name = "llama3.1"  

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        ROOT_PATH,
        "outputs",
        "optimization-results",
        f"linear_regression-o-{optimizer_llm_name}-{datetime_str}/",
    )
    os.makedirs(save_folder)
    print(f"result directory:\n{save_folder}")

    
    # ====================== optimizer model configs ============================ //TODO model configs in lokales llm
    optimizer_llm_dict = {
        "model_type": optimizer_llm_name.lower(),
        "max_decode_steps": 1024,
        "temperature": 1.0,
        "batch_size": 1
    }

    call_optimizer_server_func = functools.partial(
        prompt_utils.call_ollama_local,
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

    
    agents = {
        'OptimizationAgent': OptimizationAgent(),
        'VerifierAgent': VerifierAgent(),
        'ReasonerAgent': ReasonerAgent()
    }
    chain_manager = ChainOfThoughtManager(agents)
    chain_manager.run_chain_of_thought()


if __name__ == "__main__":
    main()
