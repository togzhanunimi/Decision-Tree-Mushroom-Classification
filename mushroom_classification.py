import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, brier_score_loss
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", 
               "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", 
               "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
               "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", 
               "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    df = pd.read_csv(url, header=None, names=columns)
    df = df.apply(LabelEncoder().fit_transform)
    return df

df = load_data()
X = df.drop("class", axis=1)
y = df["class"]

plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None,
                 class_counts=None, samples_count=None):
        self.feature_index = feature_index  
        self.threshold = threshold  
        self.left = left  
        self.right = right  
        self.value = value  
        self.class_counts = class_counts  
        self.samples_count = samples_count

    def is_leaf_node(self):
        return self.value is not None

class CustomDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                 min_impurity_decrease=0.0, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.root = None

    def _impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / m
        if self.criterion == 'gini':
            return 1 - np.sum(probs ** 2)
        elif self.criterion == 'entropy':
            return -np.sum([p * np.log2(p) for p in probs if p > 0])
        elif self.criterion == 'misclassification':
            return 1 - np.max(probs)
        else:
            raise ValueError("Unknown criterion.")

    def _best_split(self, X, y):
        m, n_features = X.shape
        if m < self.min_samples_split:
            return None, None, 0

        parent_impurity = self._impurity(y)
        best_gain = 0
        best_feature, best_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]
                impurity_left = self._impurity(y_left)
                impurity_right = self._impurity(y_right)
                n_left, n_right = len(y_left), len(y_right)
                weighted_impurity = (n_left / m) * impurity_left + (n_right / m) * impurity_right
                gain = parent_impurity - weighted_impurity

                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        m = len(y)
        num_labels = len(np.unique(y))
        counts = np.bincount(y, minlength=2)

        if depth >= self.max_depth or m < self.min_samples_split or num_labels == 1:
            majority_class = np.argmax(counts)
            return Node(value=majority_class, class_counts=counts, samples_count=m)

        feature_index, threshold, gain = self._best_split(X, y)
        if feature_index is None:
            majority_class = np.argmax(counts)
            return Node(value=majority_class, class_counts=counts, samples_count=m)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_child, right=right_child,
                    class_counts=counts, samples_count=m)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'min_impurity_decrease': [0.0, 0.01],
    'criterion': ['gini', 'entropy', 'misclassification']
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(CustomDecisionTree(), param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_tree = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)
print(f"Cross-validated training accuracy: {grid_search.best_score_:.4f}")

y_pred = best_tree.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Custom Decision Tree Confusion Matrix")
plt.show()
