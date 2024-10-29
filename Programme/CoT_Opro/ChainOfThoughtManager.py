import numpy as np
import os
import sys
import datetime
import json

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

class ChainOfThoughtManager:
    def __init__(self, agents, configuration, optimizer_llm_dict, call_optimizer_server_func):
        self.agents = agents  # OptimizationAgent, VerifierAgent, ReasonerAgent
        self.num_reps = configuration["num_reps"]
        self.num_points = configuration["num_points"]
        self.w_true = configuration["w_true"]
        self.b_true = configuration["b_true"]
        self.max_num_steps = configuration["max_num_steps"]
        self.max_num_pairs = configuration["max_num_pairs"]
        self.num_input_decimals = configuration["num_input_decimals"]
        self.num_output_decimals = configuration["num_output_decimals"]
        self.num_generated_points_in_each_step = configuration["num_generated_points_in_each_step"]
        self.optimizer_llm_dict = optimizer_llm_dict
        self.call_optimizer_server_func = call_optimizer_server_func

    def run_chain_of_thought(self):

        # =================== create the result directory ==========================
        datetime_str = (
            str(datetime.datetime.now().replace(microsecond=0))
            .replace(" ", "-")
            .replace(":", "-")
        )

        optimizer_llm_name = "llama3.1"

        save_folder = os.path.join(
            OPRO_ROOT_PATH,
            "outputs",
            "optimization-results",
            f"linear_regression-o-{optimizer_llm_name}-{datetime_str}/",
        )
        os.makedirs(save_folder)
        print(f"result directory:\n{save_folder}")

        configs_dict = dict()
        results_dict = dict()
        num_convergence_steps = []
        for i_rep in range(self.num_reps):
            found_optimal = False
            print(f"\nRep {i_rep}:")

             # ================= generate the ground truth X, y =====================
            X = np.arange(self.num_points).astype(float) + 1  # pylint: disable=invalid-name
            np.random.seed(i_rep + 1)
            y = X * self.w_true + self.b_true + np.random.randn(self.num_points)
            loss_at_true_values = self.agents['OptimizationAgent'].evaluate_loss(X, y, self.w_true, self.b_true)
            print(f"value at (w_true, b_true): {loss_at_true_values}")

            # ================= generate the starting points =====================
            num_starting_points = 5  # the number of initial points for optimization
            np.random.seed((i_rep + 1) * 10)
            init_w = np.random.uniform(low=10, high=20, size=num_starting_points)
            np.random.seed((i_rep + 1) * 100)
            init_b = np.random.uniform(low=10, high=20, size=num_starting_points)

            # ====================== run optimization ============================
            configs_dict_single_rep = {
                "optimizer_llm_configs": self.optimizer_llm_dict,
                "data": {
                    "num_points": self.num_points,
                    "w_true": self.w_true,
                    "b_true": self.b_true,
                    "loss_at_true_values": loss_at_true_values,
                    "X": list(X),
                    "y": list(y),
                },
                "init_w": list(init_w),
                "init_b": list(init_b),
                "max_num_steps": self.max_num_steps,
                "max_num_pairs": self.max_num_pairs,
                "num_input_decimals": self.num_input_decimals,
                "num_output_decimals": self.num_output_decimals,
                "num_generated_points_in_each_step": self.num_generated_points_in_each_step,
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

            rounded_inits = [
                (np.round(w, self.num_input_decimals), np.round(b, self.num_input_decimals))
                for w, b in zip(init_w, init_b)
            ]
            rounded_inits = [
                tuple(item) for item in list(np.unique(rounded_inits, axis=0))
            ]
            for w, b in rounded_inits:
                z = self.agents['OptimizationAgent'].evaluate_loss(X, y, w, b)
                old_value_pairs_set.add((w, b, z))
                old_value_pairs_with_i_step.append((w, b, z, -1))

                print("\n================ run optimization ==============")
            print(
                f"initial points: {[tuple(item[:2]) for item in old_value_pairs_set]}"
            )
            print(f"initial values: {[item[-1] for item in old_value_pairs_set]}")
            results_json_path = os.path.join(save_folder, "results.json")
            print(f"saving results to\n{results_json_path}")
      

    def log_explanation(self, step, proposal, explanation):
        # Log each step's explanation for transparency
        pass