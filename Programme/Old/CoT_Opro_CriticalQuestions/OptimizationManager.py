import numpy as np
import os
import sys
import datetime
import json

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

class OptimizationManager:
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

    def run_optimization(self):

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

            for i_step in range(self.max_num_steps):
                print(f"\nStep {i_step}:")
                meta_prompt = self.agents['OptimizationAgent'].gen_meta_prompt(
                    old_value_pairs_set,
                    num_input_decimals=self.num_input_decimals,
                    num_output_decimals=self.num_output_decimals,
                    max_num_pairs=self.max_num_pairs,
                )
                if not i_step % 5:
                    print("\n=================================================")
                    print(f"meta_prompt:\n{meta_prompt}")
                meta_prompts_dict[i_step] = meta_prompt

                # generate a maximum of the given number of points in each step
                remaining_num_points_to_generate = self.num_generated_points_in_each_step
                raw_outputs = []
                while remaining_num_points_to_generate > 0:
                    raw_outputs += self.call_optimizer_server_func(meta_prompt)
                    remaining_num_points_to_generate -= self.optimizer_llm_dict["batch_size"]
                raw_outputs = raw_outputs[:self.num_generated_points_in_each_step]

                raw_outputs_dict[i_step] = raw_outputs
                parsed_outputs = []
                for string in raw_outputs:
                    if not i_step % 5:
                        print("\n=================================================")
                        print("raw output:\n", string)
                        print("\n=================================================")
                    try:
                        parsed_output = self.agents['OptimizationAgent'].parse_output(
                            self.agents['OptimizationAgent'].extract_string_in_square_brackets(string)
                        )
                        if parsed_output is not None and len(parsed_output) == 2:
                            parsed_outputs.append(parsed_output)
                    except ValueError:
                        pass
                parsed_outputs = [tuple(item) for item in parsed_outputs]
                print(f"proposed points before rounding: {parsed_outputs}")

                # round the proposed points to the number of decimals in meta-prompt
                rounded_outputs = [
                    (np.round(w, self.num_input_decimals), np.round(b, self.num_input_decimals))
                    for w, b in parsed_outputs
                ]
                rounded_outputs = [
                    tuple(item) for item in list(np.unique(rounded_outputs, axis=0))
                ]
                print(f"proposed points after rounding: {rounded_outputs}")

                # evaluate the values of proposed and rounded outputs
                single_step_values = []
                for w, b in rounded_outputs:

                     # Schritt 2: Kritische Fragen stellen
                    for proposal in parsed_outputs:
                        critical_questions = self.agents['VerifierAgent'].ask_critical_questions(proposal)
                        critical_answers = [self.call_optimizer_server_func(q) for q in critical_questions]

                        # Schritt 3: Entscheidung treffen
                        decision = self.agents['VerifierAgent'].reject_or_accept(proposal, critical_answers)
                        explanation = self.agents['ReasonerAgent'].resolve_conflicts(critical_questions, critical_answers)
                        decision_explanation = self.agents['ReasonerAgent'].explain_decision(explanation, proposal)

                        # Schritt 4: Speichern der Erklärung
                        self.log_explanation(i_step, proposal, decision_explanation)

                        if decision == "accept":
                            # Bewerten und zum Wertpaarsatz hinzufügen
                            w, b = proposal
                            z = self.agents['OptimizationAgent'].evaluate_loss(X, y, w, b)
                            old_value_pairs_set.add((w, b, z))
                            if w == self.w_true and b == self.b_true:
                                found_optimal = True
                            z = self.agents['OptimizationAgent'].evaluate_loss(X, y, w, b)
                            single_step_values.append(z)
                            old_value_pairs_set.add((w, b, z))
                            old_value_pairs_with_i_step.append((w, b, z, i_step))
                        print(f"single_step_values: {single_step_values}")


                # ====================== save results ============================
                results_dict_single_rep = {
                    "meta_prompts": meta_prompts_dict,
                    "raw_outputs": raw_outputs_dict,
                    "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
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


      

    def log_explanation(self, step, proposal, explanation):
        log_entry = {
            "step": step,
            "proposal": proposal,
            "explanation": explanation
        }
        with open("explanations.json", "a") as f:
            f.write(json.dumps(log_entry, indent=4) + "\n")