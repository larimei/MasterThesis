from graphviz import Digraph


uml_cot_opro = Digraph('UML_Diagram_ChainOfThought_Opro', filename='uml_diagram_cot_opro', format='png')
uml_cot_opro.attr(label="UML Chain of Thought Diagram - Opro", labelloc="t", fontname="Helvetica,Arial,sans-serif")


uml_cot_opro.node('OptimizationAgent', '''<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
<tr><td><b>OptimizationAgent</b></td></tr>
<tr><td align="left">+ evaluate_loss(X, y, w, b): float</td></tr>
<tr><td align="left">+ gen_meta_prompt(old_value_pairs_set, X, y, num_input_decimals, num_output_decimals, max_num_pairs): str</td></tr>
<tr><td align="left">+ parse_output(extracted_output): np.array</td></tr>
</table>>''', shape="plaintext")

uml_cot_opro.node('VerifierAgent', '''<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
<tr><td><b>VerifierAgent</b></td></tr>
<tr><td align="left">+ extract_string_in_square_brackets(input_string): str</td></tr>
</table>>''', shape="plaintext")

uml_cot_opro.node('ReasonerAgent', '''<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
<tr><td><b>ReasonerAgent</b></td></tr>
<tr><td align="left">+ resolve_conflicts(proposals): list</td></tr>
<tr><td align="left">+ select_best_proposal(proposals): tuple</td></tr>
<tr><td align="left">+ explain_decision(best_proposal, all_proposals): str</td></tr>
<tr><td align="left">+ log_decision_process(log_path, best_proposal, explanations): None</td></tr>
</table>>''', shape="plaintext")

uml_cot_opro.node('ChainOfThoughtManager', '''<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
<tr><td><b>ChainOfThoughtManager</b></td></tr>
<tr><td align="left">+ run_chain_of_thought(): None</td></tr>
<tr><td align="left">+ log_explanation(step, proposal, explanation): None</td></tr>
</table>>''', shape="plaintext")


uml_cot_opro.edge('ChainOfThoughtManager', 'OptimizationAgent', label="Manages")
uml_cot_opro.edge('ChainOfThoughtManager', 'VerifierAgent', label="Manages")
uml_cot_opro.edge('ChainOfThoughtManager', 'ReasonerAgent', label="Manages")


uml_cot_opro.render('Files/Diagrams/uml_diagram_cot_opro')
