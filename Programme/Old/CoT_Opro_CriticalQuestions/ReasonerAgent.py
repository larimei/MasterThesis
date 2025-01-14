class ReasonerAgent:
    def resolve_conflicts(self, critical_questions, critical_answers):
        """
        Resolves conflicts in the decision-making process.
        """
        explanation = []
        for question, answer in zip(critical_questions, critical_answers):
            explanation.append(f"Question: {question} - Answer: {answer}")
        return explanation

    def explain_decision(self, explanation, proposal):
        """
        Generate a final explanation based on the evaluation.
        """
        if "reject" in explanation:
            return f"Proposal {proposal} was rejected because of these reasons: {explanation}"
        return f"Proposal {proposal} was accepted based on the evaluation: {explanation}"
