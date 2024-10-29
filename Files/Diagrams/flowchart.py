from graphviz import Digraph

flowchart_cot_opro = Digraph('Flowchart', filename='flowchart_cot_opro', format='png')

flowchart_cot_opro.node('Start', 'Start Chain of Thought Process', shape='ellipse')
flowchart_cot_opro.node('GenProposal', 'OptimizationAgent: gen_meta_prompt()\nGenerates initial (w, b) pairs as proposals', shape='box')
flowchart_cot_opro.node('EvalLoss', 'OptimizationAgent: evaluate_loss(X, y, w, b)\nCalculates loss for each (w, b)', shape='box')
flowchart_cot_opro.node('ParseOutput', 'OptimizationAgent: parse_output()\nParses output from proposals', shape='box')

flowchart_cot_opro.node('Extract', 'VerifierAgent: extract_string_in_square_brackets()\nExtracts (w, b) from LLM output', shape='box')
flowchart_cot_opro.node('VerifyLoop', 'Verifier Loop\nValidates proposal (w, b) pairs', shape='diamond')

flowchart_cot_opro.node('ConflictRes', 'ReasonerAgent: resolve_conflicts(proposals)\nResolves conflicts in proposals', shape='box')
flowchart_cot_opro.node('SelectBest', 'ReasonerAgent: select_best_proposal(proposals)\nSelects optimal (w, b) based on minimum loss', shape='box')
flowchart_cot_opro.node('Explain', 'ReasonerAgent: explain_decision()\nExplains decision for final (w, b)', shape='box')
flowchart_cot_opro.node('LogProcess', 'ReasonerAgent: log_decision_process()\nLogs process and final decision', shape='box')

flowchart_cot_opro.node('CheckLoop', 'Continue Process?\nAdditional gen_meta_prompt required?', shape='diamond')
flowchart_cot_opro.node('End', 'End Chain of Thought Process', shape='ellipse')

flowchart_cot_opro.edge('Start', 'GenProposal')
flowchart_cot_opro.edge('GenProposal', 'EvalLoss')
flowchart_cot_opro.edge('EvalLoss', 'ParseOutput')
flowchart_cot_opro.edge('ParseOutput', 'Extract')
flowchart_cot_opro.edge('Extract', 'VerifyLoop')
flowchart_cot_opro.edge('VerifyLoop', 'GenProposal', label='Proposal Rejected')
flowchart_cot_opro.edge('VerifyLoop', 'ConflictRes', label='Proposal Accepted')
flowchart_cot_opro.edge('ConflictRes', 'SelectBest')
flowchart_cot_opro.edge('SelectBest', 'Explain')
flowchart_cot_opro.edge('Explain', 'LogProcess')
flowchart_cot_opro.edge('LogProcess', 'CheckLoop')
flowchart_cot_opro.edge('CheckLoop', 'GenProposal', label='Yes')
flowchart_cot_opro.edge('CheckLoop', 'End', label='No')

flowchart_cot_opro.render('Files/Diagrams/flowchart_cot_opro')
