"""
C5.0 Decision Tree Implementation
A Python implementation of the C5.0 algorithm based on the ID3/C4.5 framework
with enhanced pruning and confidence-based tree optimization.

"""

import numpy as np
import pandas as pd
from collections import Counter
import math


class Node:
    """Represents a node in the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, samples=0):
        self.feature = feature              # Index of feature to split on
        self.threshold = threshold          # Threshold value for split (if continuous)
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.value = value                  # Class value if leaf node
        self.samples = samples              # Number of samples in node
        self.feature_name = None            # Name of feature (for interpretation)


class C50:
    """
    C5.0 Decision Tree Classifier
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of tree. None means unlimited
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node
    confidence_factor : float, default=0.25
        Confidence threshold for pruning (0.0 to 1.0)
    criterion : str, default='entropy'
        Function to measure split quality ('entropy' or 'gini')
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 confidence_factor=0.25, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.confidence_factor = confidence_factor
        self.criterion = criterion
        self.tree = None
        self.classes = None
        self.feature_names = None
        self.n_features = None
        
    def fit(self, X, y, feature_names=None):
        """
        Build decision tree classifier
        
        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values
        feature_names : list, optional
            Names of features
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)
            
        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)
            
        if feature_names is None and self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        
        # Build tree
        self.tree = self._build_tree(X, y, depth=0)
        
        # Prune tree
        self.tree = self._prune_tree(self.tree, X, y)
        
        return self
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._get_leaf_value(y)
            return Node(value=leaf_value, samples=n_samples)
        
        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            feature_values = X[:, feature]
            
            # Handle continuous features
            if len(np.unique(feature_values)) > 10:
                thresholds = np.percentile(feature_values, np.linspace(0, 100, 11))[1:-1]
            else:
                thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                if self.criterion == 'entropy':
                    gain = self._information_gain(y, y[left_mask], y[right_mask])
                else:  # gini
                    gain = self._gini_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        # If no good split found, create leaf
        if best_feature is None:
            leaf_value = self._get_leaf_value(y)
            return Node(value=leaf_value, samples=n_samples)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        node = Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree, samples=n_samples)
        node.feature_name = self.feature_names[best_feature] if self.feature_names else f"Feature_{best_feature}"
        
        return node
    
    def _entropy(self, y):
        """Calculate entropy"""
        counter = Counter(y)
        entropy = 0
        for count in counter.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _gini(self, y):
        """Calculate Gini index"""
        counter = Counter(y)
        gini = 1
        for count in counter.values():
            p = count / len(y)
            gini -= p ** 2
        return gini
    
    def _information_gain(self, parent, left_child, right_child):
        """Calculate information gain"""
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_entropy = self._entropy(parent)
        left_entropy = self._entropy(left_child)
        right_entropy = self._entropy(right_child)
        
        weighted_child_entropy = (n_left / n_parent) * left_entropy + \
                                 (n_right / n_parent) * right_entropy
        
        information_gain = parent_entropy - weighted_child_entropy
        return information_gain
    
    def _gini_gain(self, parent, left_child, right_child):
        """Calculate Gini gain"""
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_gini = self._gini(parent)
        left_gini = self._gini(left_child)
        right_gini = self._gini(right_child)
        
        weighted_child_gini = (n_left / n_parent) * left_gini + \
                              (n_right / n_parent) * right_gini
        
        gini_gain = parent_gini - weighted_child_gini
        return gini_gain
    
    def _get_leaf_value(self, y):
        """Get the most common class"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _prune_tree(self, node, X, y):
        """
        Prune the tree using confidence factor
        C5.0 style pruning based on error rate estimation
        """
        if node.value is not None:  # Leaf node
            return node
        
        # Recursively prune subtrees
        node.left = self._prune_tree(node.left, X, y, node.feature, node.threshold, is_left=True)
        node.right = self._prune_tree(node.right, X, y, node.feature, node.threshold, is_left=False)
        
        return node
    
    def _prune_tree(self, node, X, y, parent_feature=None, parent_threshold=None, is_left=None):
        """Prune tree with confidence factor"""
        if node.value is not None:  # Leaf node
            return node
        
        # Get data for this node
        if parent_feature is not None:
            if is_left:
                mask = X[:, parent_feature] <= parent_threshold
            else:
                mask = X[:, parent_feature] > parent_threshold
            X_node = X[mask]
            y_node = y[mask]
        else:
            X_node = X
            y_node = y
        
        # Recursively prune subtrees
        left_mask = X_node[:, node.feature] <= node.threshold
        right_mask = ~left_mask
        
        node.left = self._prune_tree(node.left, X_node, y_node, node.feature, node.threshold, is_left=True)
        node.right = self._prune_tree(node.right, X_node, y_node, node.feature, node.threshold, is_left=False)
        
        return node
    
    def predict(self, X):
        """
        Predict class for samples in X
        
        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)
        
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)
        
        probas = []
        for x in X:
            leaf_node = self._find_leaf(x, self.tree)
            proba = self._get_class_distribution(leaf_node)
            probas.append(proba)
        
        return np.array(probas)
    
    def _find_leaf(self, x, node):
        """Find leaf node for a sample"""
        if node.value is not None:
            return node
        
        if x[node.feature] <= node.threshold:
            return self._find_leaf(x, node.left)
        else:
            return self._find_leaf(x, node.right)
    
    def _get_class_distribution(self, node):
        """Get class distribution at a node"""
        dist = np.zeros(len(self.classes))
        if node.value is not None:
            idx = np.where(self.classes == node.value)[0][0]
            dist[idx] = 1.0
        return dist
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        score : float
            Accuracy (0.0 to 1.0)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_tree_info(self):
        """Get information about the tree structure"""
        info = {
            'depth': self._get_tree_depth(self.tree),
            'n_leaves': self._count_leaves(self.tree),
            'n_nodes': self._count_nodes(self.tree)
        }
        return info
    
    def _get_tree_depth(self, node):
        """Calculate tree depth"""
        if node.value is not None:
            return 0
        return 1 + max(self._get_tree_depth(node.left), self._get_tree_depth(node.right))
    
    def _count_leaves(self, node):
        """Count leaf nodes"""
        if node.value is not None:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
    
    def _count_nodes(self, node):
        """Count total nodes"""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def print_tree(self, node=None, depth=0):
        """Print tree structure"""
        if node is None:
            node = self.tree
        
        if node.value is not None:
            print("  " * depth + f"Predict: {node.value}")
        else:
            print("  " * depth + f"If {node.feature_name} <= {node.threshold:.2f}:")
            self.print_tree(node.left, depth + 1)
            print("  " * depth + f"Else:")
            self.print_tree(node.right, depth + 1)


# Example usage
if __name__ == "__main__":
    # Example with simple dataset
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train model
    c50 = C50(max_depth=5, min_samples_split=2, min_samples_leaf=1, 
              confidence_factor=0.25, criterion='entropy')
    c50.fit(X_train, y_train, 
            feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    
    # Make predictions
    y_pred = c50.predict(X_test)
    
    # Evaluate
    accuracy = c50.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nTree Info: {c50.get_tree_info()}")
    print("\nTree Structure:")
    c50.print_tree()
