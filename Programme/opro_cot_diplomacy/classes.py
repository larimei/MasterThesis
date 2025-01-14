from typing import Dict, List


class Message:
    def __init__(self, sender: str, recipient: str, message: str):
        self.sender = sender
        self.recipient = recipient
        self.message = message

class Phase:
    def __init__(self, name: str, messages: List[Message]):
        self.name = name
        self.messages = messages

class Game:
    def __init__(self, phases: List[Phase]):
        self.phases = phases

    def from_json(data: Dict):
        phases = []
        for phase_data in data.get("phases", []):
            name = phase_data.get("name", "Unknown Phase")
            messages = [Message(m.get("sender"), m.get("recipient"), m.get("message")) 
                        for m in phase_data.get("messages", [])]
            phases.append(Phase(name=name, messages=messages))
        return Game(phases=phases)