
class ProviderAgent:
    """Simple scripted provider that returns the gold answer passed to it."""
    def reply(self, gold_answer):
        # return as-is; in future could be more complex
        return gold_answer
