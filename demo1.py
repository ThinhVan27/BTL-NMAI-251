import numpy as np
import pandas as pd

def entropy(p, n):
    total = p + n
    if total == 0:
        return 0.0
    prob_p = p / total
    prob_n = n / total
    if prob_p == 0 or prob_n == 0:
        return 0.0
    return - (prob_p * np.log2(prob_p) + prob_n * np.log2(prob_n))

def expected_information(df, attribute, target):
    p = (df[target] == 'P').sum()
    n = (df[target] == 'N').sum()
    total = p + n
    e_a = 0.0
    values = df[attribute].unique()
    for value in values:
        sub_df = df[df[attribute] == value]
        sub_p = (sub_df[target] == 'P').sum()
        sub_n = (sub_df[target] == 'N').sum()
        weight = (sub_p + sub_n) / total
        e_a += weight * entropy(sub_p, sub_n)
    return e_a

def information_gain(df, attribute, target):
    p = (df[target] == 'P').sum()
    n = (df[target] == 'N').sum()
    return entropy(p, n) - expected_information(df, attribute, target)

def gain_ratio(df, attribute, target):
    gain = information_gain(df, attribute, target)
    values = df[attribute].unique()
    split_info = -sum((df[attribute] == value).sum() / len(df) * np.log2((df[attribute] == value).sum() / len(df)) for value in values)
    return gain / split_info if split_info > 0 else 0

def discretize_df(df, columns):
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            valid_data = df[col].dropna()
            if len(valid_data) == 0:
                raise ValueError(f"No valid data in column {col} for discretization")
            min_val, max_val = valid_data.min(), valid_data.max()
            if min_val >= max_val:
                raise ValueError(f"Invalid range in column {col}: min_val ({min_val}) >= max_val ({max_val})")
            
            bins = [min_val, (min_val + max_val) / 3, 2 * (min_val + max_val) / 3, max_val]
            bins = sorted(bins)  
            labels = ['low', 'medium', 'high']
            df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

def form_tree(df, target, attributes, use_gain_ratio=False, max_depth=None, current_depth=0):
    p = (df[target] == 'P').sum()
    n = (df[target] == 'N').sum()
    if p == 0 or n == 0 or (max_depth and current_depth >= max_depth):
        return Node(label='P' if p > 0 else 'N')
    if len(attributes) == 0:
        return Node(label='P' if p > n else 'N')
    if use_gain_ratio:
        gains = [gain_ratio(df, attr, target) for attr in attributes]
    else:
        gains = [information_gain(df, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]
    tree = Node(attribute=best_attr)
    values = df[best_attr].unique()
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    for value in values:
        sub_df = df[df[best_attr] == value]
        if sub_df.empty:
            tree.children[value] = Node(label='P' if p > n else 'N')
        else:
            tree.children[value] = form_tree(sub_df, target, remaining_attrs, use_gain_ratio, max_depth, current_depth + 1)
    return tree

def id3_iterative(df, target, attributes, window_fraction=0.5, max_iterations=100, use_gain_ratio=False, max_depth=None):
    n = len(df)
    window_size = max(1, int(n * window_fraction))
    window = df.sample(n=window_size)
    for iteration in range(max_iterations):
        tree = form_tree(window, target, attributes, use_gain_ratio, max_depth)
        predictions = predict_df(tree, df)
        misclassified = df[predictions != df[target]]
        if len(misclassified) == 0:
            print(f"Converged after {iteration + 1} iterations.")
            return tree
        add_size = min(len(misclassified), window_size)
        to_add = misclassified.sample(n=add_size)
        window = pd.concat([window, to_add]).drop_duplicates()
    print("Max iterations reached without full convergence.")
    return tree

def predict(tree, instance):
    if tree.label is not None:
        return tree.label
    value = instance.get(tree.attribute)
    if pd.isna(value): 
        labels = [child.label for child in tree.children.values() if child.label]
        return max(set(labels), key=labels.count, default=None)
    if value in tree.children:
        return predict(tree.children[value], instance)
    return None 

def predict_df(tree, df):
    return df.apply(lambda row: predict(tree, row), axis=1)

def prune_tree(tree, df, target, min_samples=2):
    if tree.label is not None:
        return
    
    for value, child in list(tree.children.items()):
        prune_tree(child, df, target, min_samples)
    
    if len(tree.children) < min_samples:
        predictions_before = predict_df(tree, df)
        accuracy_before = (predictions_before == df[target]).mean()
        print(f"Pruning node {tree.attribute}, accuracy before: {accuracy_before:.2f}")
        
        majority_class = df[target].mode()[0]
        children = {k: v for k, v in tree.children.items()}
        tree.children = {}
        tree.label = majority_class
        
        predictions_after = predict_df(tree, df)
        accuracy_after = (predictions_after == df[target]).mean()
        print(f"Accuracy after: {accuracy_after:.2f}")
        
        if accuracy_after < accuracy_before:
            tree.label = None
            tree.children = children
            print("Reverted pruning due to accuracy drop.")

def handle_incomplete(df, attribute, target):
    missing_mask = df[attribute].isna()
    if not missing_mask.any():
        return df
    non_missing = df[~missing_mask]
    value_counts = non_missing[attribute].value_counts(normalize=True)
    new_rows = []
    for idx in df.index[missing_mask]:
        row = df.loc[idx].copy()
        for value, prob in value_counts.items():
            new_row = row.copy()
            new_row[attribute] = value
            new_row['__weight__'] = prob
            new_rows.append(new_row)
    df = df.drop(df.index[missing_mask])
    return pd.concat([df] + [pd.DataFrame([r]) for r in new_rows], ignore_index=True)

def calculate_accuracy(tree, df, target):
    predictions = predict_df(tree, df)
    correct = (predictions == df[target]).sum()
    total = len(df)
    return correct / total if total > 0 else 0.0

def print_tree(node, indent=""):
    if node.label is not None:
        print(indent + "Label: " + str(node.label))
    else:
        print(indent + "Attribute: " + node.attribute)
        for value, child in node.children.items():
            print(indent + "  Value: " + str(value))
            print_tree(child, indent + "    ")

if __name__ == "__main__":
    data5 = {
        'temperature': [30, 25, 28, 22, 35, 20, 27, 33, 24, 29,
                       31, 26, 23, 34, 21, 32, 28, 25, 30, 27],
        'humidity': [85, 70, 90, 65, 95, 60, 80, 88, 72, 87,
                     83, 68, 91, 89, 63, 86, 75, 92, 84, 77],
        'windy': ['false', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true',
                  'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true', 'false', 'true'],
        'class': ['N', 'P', 'N', 'P', 'N', 'P', 'N', 'N', 'P', 'N',
                  'N', 'P', 'P', 'N', 'P', 'N', 'P', 'N', 'P', 'P']
    }
    df = pd.DataFrame(data5)
    
    df = discretize_df(df, ['temperature', 'humidity'])
    attributes = ['temperature', 'humidity', 'windy']
    target = 'class'
    
    print(f"Number of rows: {len(df)}")
    
    tree = id3_iterative(df, target, attributes, window_fraction=0.5, use_gain_ratio=False, max_depth=5)
    
    prune_tree(tree, df, target, min_samples=2)
    
    print_tree(tree)
    
    accuracy = calculate_accuracy(tree, df, target)
    print(f"Accuracy on training set: {accuracy:.2f}")
    