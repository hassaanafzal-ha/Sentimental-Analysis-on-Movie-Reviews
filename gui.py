# Import necessary libraries
import tkinter as tk
from tkinter import messagebox
import pickle
import re
import numpy as np
from collections import Counter

 #Define custom stopwords
custom_stopwords = set(["the", "and", "is", "of", "in", "to", "a", "an", "and", "are", "as", "at", "be", "but", "by", 
                        "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
                        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])
import numpy as np
from scipy.special import expit  # Sigmoid function

class DecisionTree:
    def __init__(self, depth=10):
        self.depth = depth
        self.tree = None
        self.feature_importance_ = None  # To store feature importances

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        self.feature_importance_ = self._calculate_feature_importance(X)  # Calculate feature importance after training

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if depth >= self.depth or num_labels == 1 or num_samples < 2:
            return {"leaf": True, "value": self._most_common_label(y)}

        feat_idxs = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Get left and right indices, and ensure they are integer arrays
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Ensure indexing arrays are integers
        left_idxs = left_idxs.astype(int)
        right_idxs = right_idxs.astype(int)

        left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {"leaf": False, "feature": best_feat, "threshold": best_thresh, "left": left_tree, "right": right_tree}

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.percentile(X_column, np.linspace(0, 100, 10))  # Use quantiles for efficiency
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0
        child_entropy = (n_l / n) * self._entropy(y[left_idxs]) + (n_r / n) * self._entropy(y[right_idxs])
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0].astype(int)  # Explicitly cast to integers
        right_idxs = np.where(X_column > split_thresh)[0].astype(int)  # Explicitly cast to integers
        # Ensure no empty splits
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return np.arange(len(X_column)), np.array([])  # All to one side, empty other side
        return left_idxs, right_idxs

    def _entropy(self, y):
        if y.dtype.kind == 'f':  # For regression tasks
            return np.var(y)  # Variance as a proxy for regression quality
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:  # Check for empty array
            return 0  # Default to 0 or another reasonable default
        if y.dtype.kind == 'f':  # For regression tasks
            return np.mean(y)
        return np.bincount(y).argmax()

    def _calculate_feature_importance(self, X):
        # Calculate feature importances based on the number of times a feature was used in a split
        feature_importance = np.zeros(X.shape[1])
        self._accumulate_feature_importance(self.tree, feature_importance)
        return feature_importance

    def _accumulate_feature_importance(self, node, feature_importance):
        if node["leaf"]:
            return
        feature_importance[node["feature"]] += 1  # Increment importance for the feature used in the split
        self._accumulate_feature_importance(node["left"], feature_importance)
        self._accumulate_feature_importance(node["right"], feature_importance)

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample  # Add subsample for stochastic gradient boosting
        self.trees = []
        self.initial_prediction = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        # Ensure y is a numpy array
        y = np.array(y)  # Convert y to numpy array if it's not already
        
        # Initialize the model with the mean of y (for regression)
        self.initial_prediction = np.mean(y)
        current_prediction = self.initial_prediction * np.ones_like(y)
        feature_importance = np.zeros(X.shape[1])  # Initialize feature importance array
        
        for _ in range(self.n_estimators):
            # Sample the data for stochastic gradient boosting if subsample < 1.0
            if self.subsample < 1.0:
                sample_idxs = np.random.choice(np.arange(len(y)), size=int(self.subsample * len(y)), replace=False)
                X_sample = X[sample_idxs, :]  # Ensure proper 2D indexing for X
                y_sample = y[sample_idxs]      # 1D array for y
            else:
                X_sample = X
                y_sample = y

            # Compute residuals (negative gradient)
            residual = y_sample - current_prediction[sample_idxs]
            if len(residual) == 0:  # Handle empty residuals
                residual = np.zeros_like(y_sample)

            # Fit a tree on residuals
            tree = DecisionTree(depth=self.max_depth)
            tree.fit(X_sample, residual)
            self.trees.append(tree)

            # Update predictions
            current_prediction[sample_idxs] += self.learning_rate * tree.predict(X_sample)
            
            # Calculate and accumulate feature importance for this tree
            feature_importance += tree.feature_importance_  # Assuming the tree stores feature importances

        # Normalize feature importances
        self.feature_importances_ = feature_importance / self.n_estimators

    def predict(self, X):
        # Start with the initial prediction
        prediction = self.initial_prediction * np.ones((X.shape[0],))
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return np.round(prediction).astype(int)  # For classification
    
    def predict_proba(self, X):
        # Start with the initial prediction
        prediction = self.initial_prediction * np.ones((X.shape[0],))
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)

        # Apply sigmoid function to convert predictions to probabilities
        probabilities = expit(prediction)  # Sigmoid function for probability
        return probabilities

# Logistic Regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# Define the SimpleVectorizer class
class SimpleVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}
        self.vocab_size = 0
        self.stopwords = custom_stopwords

    def fit_transform(self, texts):
        word_counts = Counter()
        for text in texts:
            cleaned_text = self.clean_text(text)
            word_counts.update(cleaned_text.split())

        if self.max_features is not None:
            top_n_words = word_counts.most_common(self.max_features)
            self.vocab = {word: i for i, (word, _) in enumerate(top_n_words)}
            self.vocab_size = len(top_n_words)
        else:
            for word in word_counts:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1

        X = np.zeros((len(texts), self.vocab_size))
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            for word in cleaned_text.split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def transform(self, texts):
        X = np.zeros((len(texts), self.vocab_size))
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            for word in cleaned_text.split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove custom stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        # Join tokens back into text
        cleaned_text = ' '.join(tokens)
        return cleaned_text

# Global variables for the selected model and vectorizer
model = None
vectorizer = None

# Function to load the selected model
def load_model(selected_model):
    global model, vectorizer

    if selected_model == "Logistic Regression":
        model_filename = "logistic_regression_model.pkl"
    elif selected_model == "Gradient Boosting":
        model_filename = "gradient_boosting_model.pkl"
    else:
        messagebox.showerror("Model Selection Error", "Unknown model selected.")
        return

    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)

        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        messagebox.showinfo("Model Loaded", f"{selected_model} model loaded successfully!")

    except FileNotFoundError:
        messagebox.showerror("File Error", f"{model_filename} or vectorizer.pkl not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to classify review
def classify_review():
    if model is None or vectorizer is None:
        messagebox.showerror("Model Error", "Please load a model first.")
        return

    review_text = review_input.get("1.0", tk.END).strip()
    if not review_text:
        messagebox.showerror("Input Error", "Please enter a review.")
        return

    # Transform the review using the vectorizer
    transformed_review = vectorizer.transform([review_text])

    # Predict sentiment
    prediction = model.predict(transformed_review)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Display the result
    result_label.config(text=f"Predicted Sentiment: {sentiment}")



# Create the GUI window
window = tk.Tk()
window.title("Sentiment Classifier")
window.geometry("700x400")  # Set size
window.configure(bg='darkRed')  # Set background color
model_selection = tk.StringVar(value="Logistic Regression")
# Add GUI components with custom colors and fonts
tk.Label(window, text="Select Model:", bg="lightblue", fg="darkblue", font=("Arial", 12)).pack(pady=5)
model_menu = tk.OptionMenu(window, model_selection, "Logistic Regression", "Gradient Boosting")
model_menu.config(bg="white", fg="black", font=("Arial", 10))
model_menu.pack(pady=5)

load_button = tk.Button(window, text="Load Model", command=lambda: load_model(model_selection.get()), bg="darkgreen", fg="white", font=("Arial", 12))
load_button.pack(pady=5)

tk.Label(window, text="Enter a Review:", bg="lightblue", fg="darkblue", font=("Arial", 12)).pack(pady=5)
review_input = tk.Text(window, height=5, width=50, bg="lightyellow", font=("Arial", 10))
review_input.pack(pady=5)

classify_button = tk.Button(window, text="Classify", command=classify_review, bg="blue", fg="white", font=("Arial", 12))
classify_button.pack(pady=5)

result_label = tk.Label(window, text="", font=("Helvetica", 14, "bold"), bg="lightblue", fg="darkred")
result_label.pack(pady=10)

# Run the GUI event loop
window.mainloop()

