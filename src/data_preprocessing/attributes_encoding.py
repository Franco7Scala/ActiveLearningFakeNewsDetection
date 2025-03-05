from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def load_model():
    """Load and return the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def convert_categorical_to_string(df, col):
    """Convert a categorical column to string if necessary."""
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        df[col] = df[col].astype(str)
    return df


def get_embedding(model, text_or_list):
    """Generate embeddings for a string or list of strings."""
    if isinstance(text_or_list, str):
        if text_or_list == "":
            return None
        return model.encode(text_or_list)
    if isinstance(text_or_list, list):
        embeddings = [model.encode(text) for text in text_or_list if text != ""]
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
    return None


def replace_missing_embeddings(embeddings, mean_embedding):
    """Replace missing embeddings with the mean embedding."""
    return [embedding if embedding is not None else mean_embedding for embedding in embeddings]


def reduce_dimensionality(embeddings_matrix, target_dim):
    """Perform PCA to reduce dimensionality of embeddings."""
    pca = PCA(n_components=target_dim)
    return pca.fit_transform(embeddings_matrix)


def encoding_short_text(df, col, target_dim=28):
    """
    Main function to encode text in a DataFrame column, reduce dimensionality,
    and save the results back into the DataFrame.
    """
    # Load the model
    model = load_model()

    # Convert categorical column to string if necessary
    df = convert_categorical_to_string(df, col)

    # Generate embeddings
    embeddings = df[col].apply(lambda x: get_embedding(model, x))
    non_nan_embeddings = [e for e in embeddings if e is not None]

    # Compute mean embedding for replacement
    if len(non_nan_embeddings) > 0:
        embeddings_matrix = np.vstack(non_nan_embeddings)
        mean_embedding = np.mean(embeddings_matrix, axis=0)

        # Replace missing embeddings
        embeddings = replace_missing_embeddings(embeddings, mean_embedding)

        # Reduce dimensionality
        embeddings_matrix = np.vstack(embeddings)
        reduced_embeddings = reduce_dimensionality(embeddings_matrix, target_dim)

        # Save reduced embeddings back to DataFrame
        #df[col + '_encoded'] = [embedding.tolist() for embedding in reduced_embeddings]
        df[col + '_encoded'] = [
            np.array2string(embedding, separator=' ').replace('\n', '')
            for embedding in reduced_embeddings
        ]

        # Drop the original column
        df.drop([col], axis=1, inplace=True)

    return df
