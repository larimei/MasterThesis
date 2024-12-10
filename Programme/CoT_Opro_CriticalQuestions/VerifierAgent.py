class VerifierAgent:
    def ask_critical_questions(self, proposal):
        questions = [
            f"Does the proposal {proposal} respect the constraints of the problem?",
            f"Is the function value at {proposal} better than the current minimum?",
            f"Could there be numerical instabilities or rounding errors affecting {proposal}?"
        ]
        return questions

    def reject_or_accept(self, proposal, critical_answers):
        """
        Evaluate the answers to critical questions and decide to accept or reject.
        """
        for answer in critical_answers:
            print(answer)
            if "no" in answer.lower():
                return "reject"
        return "accept"