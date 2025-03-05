import torch
from torch_geometric.explain import Explainer, CaptumExplainer



def get_explanation(data, model, target_type, indices=None):

    if indices is None:
        indices = torch.arange(data.y_dict[target_type].shape[0])

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),  # InputXGradient
        explanation_type='phenomenon', # model's beahviour (model) vs individual predictions (phenomenon)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 menon',  # model's beahviour (model) vs individual predictions (phenomenon)
        node_mask_type='attributes',
        edge_mask_type='object', #None
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='probs', #probability of the positive class. ------  other types: log_probs, raw
        )
    )

    with torch.no_grad():
        explanation = explainer(
            data.x_dict,
            data.edge_index_dict,
            target=data.y_dict[target_type],
            index=indices
        )
        return explanation


def get_topk_nodes(explanation, node_type, topk):
    values, indices = torch.topk(explanation.node_mask_dict[node_type].sum(-1), k=topk)
    return indices.cpu().numpy().tolist()
