from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_embedding(X, labels = None):
    """Visualize the embedding using t-SNE.
    Args:
        X (np.ndarray): The embedding matrix of shape (N (embeddings), D (feature dimension)).
        labels (np.ndarray): The labels of shape (N (embeddings)).
    """
    
    tsne = TSNE()
    X_reduced = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 10))
    if labels is None:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.show()
    

if __name__ == '__main__':
    
    X1 = np.random.rand(25, 128) + 1
    X2 = np.random.rand(25, 128) -1
    X3 = np.random.rand(25, 128) + 0.5
    X4 = np.random.rand(25, 128) - 0.5
    X = np.concatenate((X1, X2, X3, X4), axis=0)
    labels = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)
    
    visualize_embedding(X, labels)
    
    d = {}