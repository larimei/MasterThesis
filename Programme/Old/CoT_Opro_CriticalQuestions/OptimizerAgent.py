import re
import numpy as np

class OptimizationAgent:

    def evaluate_loss(self, X, y, w, b):
        residual = y - (X * w + b)
        return np.linalg.norm(residual) ** 2
    
    def gen_meta_prompt(self, old_value_pairs_set, num_input_decimals=5, num_output_decimals=5, max_num_pairs=100):
        old_value_pairs_set = set(
            [
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

        meta_prompt += """Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than any of the above. Do not write code. The output must end with a pair [w, b], where w and b are numerical values.
        """.strip()
        return meta_prompt
    
    def parse_output(self, extracted_output):

        if not extracted_output:
            return
        extracted_values = []
        for item in extracted_output.split(","):
            if "=" in item:
                item = item[item.index("=") + 1 :]
            extracted_values.append(item.strip())
        parsed_output = np.array(extracted_values).astype(float)
        return parsed_output
    
    def extract_string_in_square_brackets(self, input_string):
        raw_result = re.findall(r"\[.*?\]", input_string)
        if raw_result:
            for pair in raw_result[::-1]:
                if "=" not in pair and ("w" in pair or "b" in pair):
                    continue
                return pair[1:-1]
            return ""
        else:
            return ""


    def verify_proposal(self, proposal):
        # Verify if the new proposal is valid by calculating the loss
        pass

    def explain_proposal(self, proposal):
        # Explain why this proposal was generated
        return f"Generated (w, b) = {proposal} based on the loss function minimization."
