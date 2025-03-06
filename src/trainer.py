import os
import torch
import numpy as np

from sklearn.metrics import f1_score
from src.data_utils import save_dict_to_pickle


def train(model, data, optimizer, criterion, target_type, ranking_type, directory, n_epochs=200):
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()
        f1_micro, f1_macro, f1_weigh, auc = evaluate(model, data, target_type, directory)
        print(f"Epoch: {epoch + 1:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}")

    save_dict_to_pickle(embeddings, f"{directory}/embeddings_after_training.pkl")
    user_np = embeddings[ranking_type].cpu().detach().numpy()
    news_np = embeddings[target_type].cpu().detach().numpy()
    np.save(f"{directory}/embeddings_{target_type}_np.npy", user_np)
    np.save(f"{directory}/embeddings_{target_type}_np.npy", news_np)


def evaluate(model, data, target_type, directory):
    model.eval()
    with torch.no_grad():
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        prediction = out[target_type].argmax(dim=-1)
        f1_micro = f1_score(data[target_type].y.cpu(), prediction.cpu(), average="micro")
        f1_macro = f1_score(data[target_type].y.cpu(), prediction.cpu(), average="macro")
        f1_weigh = f1_score(data[target_type].y.cpu(), prediction.cpu(), average="weighted")
        # Save embeddings for validation set
        val_embeddings = embeddings[target_type].cpu().numpy()
        np.save(os.path.join(directory, "embeddings.npy"), val_embeddings)

        return f1_micro, f1_macro, f1_weigh, 0
