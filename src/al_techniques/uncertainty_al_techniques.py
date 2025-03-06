import torch
import numpy

from src.al_techniques.abstract_al_technique import AbstractALTechnique


class MarginALTechnique(AbstractALTechnique):

    def get_score(self, sample, target_type):
        preds = torch.nn.functional.softmax(self.model(sample.x_dict, sample.edge_index_dict)[0][self.target_type], dim=1)
        preds_argmax = torch.argmax(preds, dim=1)
        max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
        preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
        preds_sub_argmax = torch.argmax(preds, dim=1)
        return (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().detach().numpy() * -1 #TODO to remove -1 made for sociologi


class LCSALTechnique(AbstractALTechnique):

    def get_score(self, sample, target_type):
        return self.model(sample.x_dict, sample.edge_index_dict)[0][self.target_type].max(axis=1).values.cpu().detach().numpy() * -1 #TODO to remove -1 made for sociologi


class EntropyALTechnique(AbstractALTechnique):

    def get_score(self, sample, target_type):
        preds = torch.nn.functional.softmax(self.model(sample.x_dict, sample.edge_index_dict)[0][self.target_type], dim=1).detach().cpu().numpy()
        return (numpy.log(preds + 1e-6) * preds).sum(axis=1) * -1 #TODO to remove -1 made for sociologi
