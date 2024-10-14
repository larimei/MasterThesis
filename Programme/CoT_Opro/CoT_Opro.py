# Instantiate agents and run the CoT process
from Programme.CoT_Opro import ChainOfThoughtManager, VerifierAgent
from Programme.CoT_Opro.OptimizerAgent import OptimizationAgent
from Programme.CoT_Opro.ReasonerAgent import ReasonerAgent

def main(_):
    agents = {
        'OptimizationAgent': OptimizationAgent(),
        'VerifierAgent': VerifierAgent(),
        'ReasonerAgent': ReasonerAgent()
    }
    chain_manager = ChainOfThoughtManager(agents)
    chain_manager.run_chain_of_thought()


if __name__ == "__main__":
    main()
