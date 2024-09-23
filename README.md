# IMGA-Net
 
# Fake News Detection with BERT and Graph Neural Networks

This project focuses on detecting fake news using a combination of BERT embeddings and graph neural networks (GAT and GCN). The model leverages textual features, skipgram-based co-occurrence graphs (PMI), dependency graphs, and GCN-based node embeddings to predict whether a given news is fake or real.

Overview:

The model integrates three main components:
1. BERT Embeddings: Extracts embeddings from raw textual data.
2. Graph Attention Networks (GAT): Used to process both PMI and dependency graphs.
3. Graph Convolutional Networks (GCN): Used to propagate features from node embeddings.
4. Inconsistency Learning: Identifies inconsistencies between different types of embeddings and graph information.

This approach combines statistical relationships (PMI), syntactic structures (dependency graphs), and node propagation (GCN) to improve the accuracy of fake news detection.


Datasets and Evaluation Metrics:

This project utilizes three widely used benchmark datasets sourced from social media platforms, specifically Twitter and LIAR. These datasets are:

1. Twitter15: This dataset is collected from Twitter and contains tweets that are labeled as either fake or real news. It serves as a reliable benchmark for evaluating fake news detection algorithms.

2. Twitter16: Similar to Twitter15, this dataset is also sourced from Twitter, with slight differences in structure and data types. It is often used for evaluating and improving the performance of fake news detection systems.

3. LIAR: This dataset is collected from the Politifact website, consisting of statements that have been fact-checked and classified into categories ranging from completely true to completely false. LIAR provides a valuable benchmark for evaluating fake news detection systems on a more diverse set of claims.



 Model Architecture:

The model architecture consists of the following components:
- BERT: Tokenizes and generates embeddings for each tweet.
- GAT: Processes PMI and Dependency matrices.
- GCN: Processes node embeddings and propagates features through the graph.
- Inconsistency Learning Module: Self-attention and cross-attention layers detect inconsistencies between embeddings.
- Classifier: Final binary classification of fake or real news.

 Installation:

To run this project, ensure you have the following dependencies installed:

pip install torch
pip install torch-geometric
pip install transformers
pip install pandas
pip install numpy
pip install h5py
pip install tqdm
pip install scikit-learn


Training:

The training script uses the following steps:

1. Load BERT embeddings for tweets.
2. Generate graphs for PMI and dependency relationships.
3. Train GAT and GCN models to capture important relationships between words in the text.
4. Apply the Inconsistency Module to detect mismatches between different embedding types.
5. Classify tweets as fake or real.
