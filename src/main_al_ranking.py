import os
import sys
import importlib
import torch

from src import utils
from src.data_loading.mumin import load_mumin_heterodata
from src.data_loading.politifact import load_politifact_heterodata
from src.data_utils import create_nodes_dict_empty, create_nodes_dict_full
from src.models.GAT_enhanced import GAT_enhanced
from src.sampling_strategies.active_ers2 import ActiveERS2
from src.utils import set_random_seed, training_seeds, compute_weights, cprint, Color, send_telegram_notification
from src.trainer import evaluate
from data_utils import get_base_dir
from torch_geometric.nn import to_hetero
from trainer import train


if __name__ == "__main__":
    if len(sys.argv) < 6:
        cprint(f"You should put these parameters:\n"
               f"- Uncertainty sampling technique: LCSALTechnique, EntropyALTechnique or MarginALTechnique\n"
               f"- Ranking node type: the type of notes to rank\n"
               f"- Dataset name: Politifact or Mumin\n"
               f"- Force retrain: True if you want to train the model, even if it has already been created\n"
               f"- Quantity training epochs: 200 should be a nice value\n", Color.WARNING)
        sys.exit(1)

    uncertainty_sampling_technique = getattr(importlib.import_module("src.al_techniques.uncertainty_al_techniques"), sys.argv[1])
    ranking_type = sys.argv[2]
    dataset_name = sys.argv[3]
    force_retrain = bool(sys.argv[4])
    n_epochs = int(sys.argv[5])

    ranking_strategy = ActiveERS2
    device = utils.get_device()
    base_dir = get_base_dir(dataset_name)

    cprint(f"Building dataset {dataset_name}...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    if dataset_name == "politifact":
        data = load_politifact_heterodata(base_dir)
        target_type = "news"
        num_layers = 2

    elif dataset_name == "mumin":
        data = load_mumin_heterodata(base_dir)
        target_type = "claim"
        num_layers = 3

    else:
        raise ValueError(f"Unknown dataset named {dataset_name}...")

    path_model = os.path.join(base_dir, f"model_{dataset_name}_{n_epochs}_epochs_{num_layers}_layers.pth")
    num_classes = len(torch.unique(data[target_type].y))

    training_seed = training_seeds[0]
    cprint(f"Performing run with seed '{training_seed}' on {dataset_name} dataset...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    set_random_seed(training_seed)

    cprint(f"Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = GAT_enhanced(hidden_channels=64, out_channels=num_classes, dropout=0.4, num_layers=num_layers)
    model = to_hetero(model, data.metadata(), aggr="sum").to(device)

    if os.path.isfile(path_model) and not force_retrain:
        cprint(f"Loading pre-trained model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        model.load_state_dict(torch.load(path_model, weights_only=True))
        model.eval()

    else:
        cprint(f"Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss(compute_weights(data[target_type].y).float().to(device))
        train(model, data, optimizer, criterion, target_type, ranking_type, directory=base_dir, n_epochs=n_epochs)
        torch.save(model.state_dict(), path_model)

    cprint("Looking for the best samples...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    strategy = ranking_strategy(uncertainty_sampling_technique(model, target_type))
    strategy.sample(data, create_nodes_dict_empty(data), {}, create_nodes_dict_full(data), {}, target_type, ranking_type, save=True, directory=base_dir)

    f1_micro, f1_macro, f1_weigh, auc = evaluate(model, data, target_type, directory=base_dir)
    cprint(f"f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, f1-weighted: {f1_weigh:.3f}, roc-auc: {auc:.3f}", Color.EXPERIMENT_OUTPUT)

    send_telegram_notification(f"Research finished:\n"
                               f"- Sampling Technique: {uncertainty_sampling_technique.__name__}\n"
                               f"- Dataset: {dataset_name}\n"
                               f"- F1 micro: {f1_micro:.3f}\n"
                               f"- F1 macro: {f1_macro:.3f}\n"
                               f"- F1 weighted: {f1_weigh:.3f}\n"
                               f"- ROC-AUC: {auc:.3f}\n"
                               f"- Random seed: '{training_seed}'\n"
                               f"- Model saved in: '{path_model}'\n"
                               f"- Results saved in: '{base_dir}/selected/...'")

    cprint(f"Completed!", Color.OTHER)
