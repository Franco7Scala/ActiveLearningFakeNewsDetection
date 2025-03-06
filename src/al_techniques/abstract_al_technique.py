

class AbstractALTechnique:

    def __init__(self, model, target_type="Not defined"):
        self.model = model
        self.target_type = target_type

    def get_score(self, sample, target_type):
        pass
