# Who Drives Misinformation? Key Node Detection with Heterogeneous Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the research paper *"Who Drives Misinformation? Key Node Detection with Heterogeneous Graph Neural Networks"*. The framework proposed in this work enables the detection of key influencers in misinformation networks through a novel combination of **Graph Attention Networks (GATs)** and **post-hoc analytical techniques** including uncertainty-based Active Learning-like methods.

## Key Features

* **Heterogeneous GNNs for Fake News Detection:** Utilizes meta-path-enhanced Graph Attention Networks to perform binary classification of news as real or fake;
* **Key Node Identification:** Identifies influential user nodes responsible for the propagation of misinformation.
* **Post-hoc Explainability Techniques:**

  * **Uncertainty-based AL-like methods** (Least Confidence, Margin, and Entropy-based ranking);
  * **GNN-Explainer** for feature- and relation-based node attribution;
* **Comparative Evaluation:** Benchmarks the approach against traditional centrality measures (e.g., PageRank, Betweenness, VoteRank) on real-world datasets;
* **Scalability:** Designed for large-scale and densely connected social media graphs.

## Repository Structure

```
.
├── data/
│   └── [Preprocessed datasets: MuMiN and PolitiFact]
├── models/
│   └── gat_model.py
├── explainer/
│   └── gnn_explainer.py
├── al_like/
│   ├── entropy_sampling.py
│   ├── margin_sampling.py
│   └── least_confidence.py
├── baseline/
│   └── centrality_measures.py
├── utils/
│   └── graph_utils.py
├── main.py
└── README.md
```

## Requirements

* Python (>=3.10.6)
* PyTorch (>=2.2.1)
* Torchvision (>=0.17.1)
* NumPy (>=1.26.4)
* Scikit-learn (>=1.4.1.post1)
* CodeCarbon (>=2.3.4)
* ptflops (>=0.7.3)
* Captum (for GNNExplainer)

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## Running the Code

1. **Preprocess the datasets** (MuMiN, PolitiFact) using provided scripts under `data/`.
2. **Train the GAT model:**

```bash
python main.py --dataset mumin --train
```

3. **Apply post-hoc analysis:**

   * AL-like ranking:

     ```bash
     python al_like/entropy_sampling.py --dataset mumin
     ```
   * GNN-Explainer:

     ```bash
     python explainer/gnn_explainer.py --dataset mumin
     ```

4. **Evaluate rankings vs baselines:**

```bash
python baseline/centrality_measures.py --dataset mumin
```

## Datasets

* **MuMiN:** Multilingual, multimodal misinformation dataset including claims, tweets, users, and hashtags.
* **PolitiFact:** Political news fact-checked dataset from FakeNewsNet.

Each is modeled as a heterogeneous information network with multiple node and edge types.

## Performance

| Dataset    | F1-micro | F1-macro | Time (s) |
| ---------- | -------- | -------- | -------- |
| MuMiN      | 0.954    | 0.788    | \~189    |
| PolitiFact | 0.859    | 0.845    | \~332    |

Post-hoc methods outperform traditional centrality metrics in both **influence reachability** and **coverage**, especially in high-connectivity graphs.

## Citation

```
@InProceedings{10.1007/978-3-032-05461-6_4,
 author="Martirano, Liliana
 and Scala, Francesco
 and Comito, Carmela
 and Pontieri, Luigi",
 editor="D{\v{z}}eroski, Sa{\v{s}}o
 and Levati{\'{c}}, Jurica
 and Pio, Gianvito
 and Simidjievski, Nikola",
 title="Who Drives Misinformation? Key Node Detection with Heterogeneous Graph Neural Networks",
 booktitle="Discovery Science",
 year="2025",
 publisher="Springer Nature Switzerland",
 address="Cham",
 pages="47--62",
 abstract="Misinformation propagation in online networks involves multifaceted interactions between users, contents, and engagement mechanisms (likes, shares, comments). Addressing this issue entails both understanding how information spreads and identifying influential users driving the dissemination process. To tackle these challenges, this paper proposes a framework based on a Graph Attention Network model, applied to a heterogeneous graph representing social interactions and context-aware dynamics. Targeting the binary classification of real vs fake news, it offers insights into both propagation patterns and influential users in the dissemination process. A core contribution is the adoption of two post-hoc mechanisms for uncovering such users: uncertainty-based Active learning-like and GNN-Explainer. A detailed comparative analysis reveals that nodes where the model exhibits the highest confidence often lack rich content information; nevertheless, combining both high-confidence and content-rich nodes grasps complementary aspects and better aligns with influential users in information propagation. The framework is benchmarked against traditional centrality measures, widely used to identify influential users in social networks. A comparative evaluation on two heterogeneous, real-world, social networks confirms that the proposed method both achieves compelling accuracy in finding influential nodes and shows a potential to scale-up to densely-connected graphs on which classic approaches may fail.",
 isbn="978-3-032-05461-6"
}


```

## License

This project is licensed under the MIT License.
