from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import csv
import time
import json
import os  # For file operations
import pickle


app = Flask(__name__)

# Data Preprocessing function (modify return statement)
def data_preprocess():
    df = pd.read_csv('Train_1.txt', sep = ',', names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"])

    df_test = pd.read_csv('Test_1.txt', sep = ',', names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"])

    df.head()

    df.groupby(['protocol_type']).describe()


    df_test.groupby(['attack']).describe()

    df.loc[df['protocol_type'] == 'icmp', "protocol_type"] = 3
    df.loc[df['protocol_type'] == 'tcp', "protocol_type"] = 1
    df.loc[df['protocol_type'] == 'udp', "protocol_type"] = 2

    df_test.loc[df_test['protocol_type'] == 'icmp', "protocol_type"] = 3
    df_test.loc[df_test['protocol_type'] == 'tcp', "protocol_type"] = 1
    df_test.loc[df_test['protocol_type'] == 'udp', "protocol_type"] = 2

    df.head()

    df_test.groupby(['service']).describe()

    df = df.drop('service', axis = 1)
    df_test = df_test.drop('service', axis = 1)

    df.groupby(['flag']).describe()


    df.loc[df['flag'] == 'REJ', "flag"] = 1
    df.loc[df['flag'] == 'SF', "flag"] = 2
    df.loc[df['flag'] == 'S0', "flag"] = 3
    df.loc[df['flag'] == 'RSTR', "flag"] = 4
    df.loc[df['flag'] == 'RSTOS0', "flag"] = 5
    df.loc[df['flag'] == 'RSTO', "flag"] = 6
    df.loc[df['flag'] == 'SH', "flag"] = 7
    df.loc[df['flag'] == 'S1', "flag"] = 8
    df.loc[df['flag'] == 'S2', "flag"] = 9
    df.loc[df['flag'] == 'S3', "flag"] = 10
    df.loc[df['flag'] == 'OTH', "flag"] = 11

    df_test.loc[df_test['flag'] == 'REJ', "flag"] = 1
    df_test.loc[df_test['flag'] == 'SF', "flag"] = 2
    df_test.loc[df_test['flag'] == 'S0', "flag"] = 3
    df_test.loc[df_test['flag'] == 'RSTR', "flag"] = 4
    df_test.loc[df_test['flag'] == 'RSTOS0', "flag"] = 5
    df_test.loc[df_test['flag'] == 'RSTO', "flag"] = 6
    df_test.loc[df_test['flag'] == 'SH', "flag"] = 7
    df_test.loc[df_test['flag'] == 'S1', "flag"] = 8
    df_test.loc[df_test['flag'] == 'S2', "flag"] = 9
    df_test.loc[df_test['flag'] == 'S3', "flag"] = 10
    df_test.loc[df_test['flag'] == 'OTH', "flag"] = 11

    df.head()

    df.groupby(['attack']).describe()

    # Convert 'normal' and 'intrusion' strings to 1 and 0
    # Ensure all values are mapped before converting to int
    # Check unique values before mapping to identify any unexpected strings
    # print("Unique attack values before mapping:", df['attack'].unique())
    # print("Unique attack values before mapping (test):", df_test['attack'].unique())

    df['attack'] = df['attack'].replace({'normal': 1, 'intrusion': 0})
    # Handle potential typos or other strings in the test set
    df_test['attack'] = df_test['attack'].replace({'normal': 1, 'intrusion': 0, 'iintrusionweep': 0, 'udintrusiontorm': 0}) # Assuming typos should be treated as intrusion
    # Convert the column to integer type explicitly
    df['attack'] = df['attack'].astype(int)
    df_test['attack'] = df_test['attack'].astype(int)


    df_test_nb = df_test
    df_train_nb = df

    df.head()

    df_x = df.drop('attack', axis = 1)
    df_y = df['attack']

    df_x_test = df_test.drop('attack', axis = 1)
    df_y_test = df_test['attack']

    scaler = StandardScaler()
    df_x_scaled = pd.DataFrame(scaler.fit_transform(df_x))
    df_x_test_scaled = pd.DataFrame(scaler.transform(df_x_test)) # Use transform, not fit_transform for test data

    df_y.head()

    df_x_scaled.head()

    df_x_test_scaled.head()

    # Convert to PyTorch tensors
    # df_y.values and df_y_test.values should now be numerical (int) arrays
    df_tensor = torch.tensor(df_x_scaled.values).float() # Cast features to float
    df_tensor_y = torch.tensor(df_y.values).long() # Target for NLLLoss should be LongTensor

    df_tensor_test = torch.tensor(df_x_test_scaled.values).float() # Cast features to float
    df_tensor_y_test = torch.tensor(df_y_test.values).float() # Target for BCE loss should be FloatTensor

    # Return the scaler along with the tensors
    return df_tensor, df_tensor_y, df_tensor_test, df_tensor_y_test, scaler


# Neural Network Model
class IDSModel(nn.Module):
    def __init__(self, input_size):
        super(IDSModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class FeatureImportanceModel:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def compute_importance(self, X_test, y_test, n_samples=100):
        """Compute feature importance using feature permutation"""
        importances = {}

        # X_test and y_test are now expected to be NumPy arrays (as passed from evaluate_models)
        # Convert X_test NumPy array to PyTorch tensor for model inference
        X_tensor = torch.FloatTensor(X_test)

        # Get baseline accuracy
        self.model.eval()
        with torch.no_grad():
            # Ensure the model output is flattened and converted to int predictions
            baseline_pred = (self.model(X_tensor).cpu().numpy().flatten() > 0.5).astype(int)
        baseline_acc = accuracy_score(y_test, baseline_pred)

        # Permute each feature and measure drop in accuracy
        for i in range(X_test.shape[1]):
            acc_drops = []

            for _ in range(n_samples):
                # Work with the NumPy array for shuffling
                X_permuted_np = X_test.copy() # Use NumPy copy here

                # Shuffle the values in column i of the NumPy array
                np.random.shuffle(X_permuted_np[:, i])

                # Convert the permuted NumPy array back to a PyTorch tensor
                X_permuted_tensor = torch.FloatTensor(X_permuted_np)

                # Get predictions using the model (expects FloatTensor)
                with torch.no_grad():
                    # Ensure model output is flattened and converted to int predictions
                    perm_pred = (self.model(X_permuted_tensor).cpu().numpy().flatten() > 0.5).astype(int)

                # Compute accuracy drop
                perm_acc = accuracy_score(y_test, perm_pred)
                acc_drop = baseline_acc - perm_acc
                acc_drops.append(acc_drop)

            # Average accuracy drop
            importance = max(0, np.mean(acc_drops))

            # Store feature importance
            # Ensure feature_names list has enough elements or use default names
            feature_name = f"feature_{i}" if i >= len(self.feature_names) else self.feature_names[i]
            importances[feature_name] = importance

        # Normalize importances
        total_importance = sum(importances.values())
        if total_importance > 0:
            for feature in importances:
                importances[feature] /= total_importance

        return importances

# Evaluate models and compute metrics (Modification to pass NumPy arrays to FeatureImportanceModel)
def evaluate_models():
    # Get data from global data dictionary
    X_test_tensor = data['X_test']  # This is a FloatTensor
    y_test_tensor_float = data['y_test'] # This is a FloatTensor (for BCE evaluation)

    # Convert y_test (float tensor) to integer NumPy array for metrics and feature importance
    # This conversion should happen here once
    y_test_int_np = y_test_tensor_float.cpu().numpy().astype(int)

    # Evaluate standard model
    models['standard'].eval()
    with torch.no_grad():
        # Use X_test_tensor for model inference
        # Output is probability (sigmoid), compare directly to 0.5
        y_pred_standard = (models['standard'](X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)

    # Compute metrics for standard model
    # Use the integer NumPy array y_test_int_np
    acc_standard = accuracy_score(y_test_int_np, y_pred_standard)
    precision_standard = precision_score(y_test_int_np, y_pred_standard, zero_division=0)
    recall_standard = recall_score(y_test_int_np, y_pred_standard, zero_division=0)
    f1_standard = f1_score(y_test_int_np, y_pred_standard, zero_division=0)
    cm_standard = confusion_matrix(y_test_int_np, y_pred_standard)

    # Store metrics
    models['standard_metrics'] = {
        'accuracy': acc_standard,
        'precision': precision_standard,
        'recall': recall_standard,
        'f1': f1_standard,
        'confusion_matrix': cm_standard.tolist()
    }
    print(f"Standard Model Metrics: Accuracy={acc_standard:.4f}, Precision={precision_standard:.4f}, Recall={recall_standard:.4f}, F1={f1_standard:.4f}")


    # Evaluate robust model
    models['robust'].eval()
    with torch.no_grad():
        # Use X_test_tensor for model inference
        # Output is probability (sigmoid), compare directly to 0.5
        y_pred_robust = (models['robust'](X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)

    # Compute metrics for robust model
    # Use the integer NumPy array y_test_int_np
    acc_robust = accuracy_score(y_test_int_np, y_pred_robust)
    precision_robust = precision_score(y_test_int_np, y_pred_robust, zero_division=0)
    recall_robust = recall_score(y_test_int_np, y_pred_robust, zero_division=0)
    f1_robust = f1_score(y_test_int_np, y_pred_robust, zero_division=0)
    cm_robust = confusion_matrix(y_test_int_np, y_pred_robust)

    # Store metrics
    models['robust_metrics'] = {
        'accuracy': acc_robust,
        'precision': precision_robust,
        'recall': recall_robust,
        'f1': f1_robust,
        'confusion_matrix': cm_robust.tolist()
    }
    print(f"Robust Model Metrics: Accuracy={acc_robust:.4f}, Precision={precision_robust:.4f}, Recall={recall_robust:.4f}, F1={f1_robust:.4f}")


    # Compute feature importance for standard model
    print("Computing feature importance...")
    # Use a smaller subset for feature importance for efficiency
    # Get the subset indices
    indices = np.random.choice(len(X_test_tensor), min(1000, len(X_test_tensor)), replace=False)

    # Extract the subset as NumPy arrays for FeatureImportanceModel
    X_subset_np = X_test_tensor[indices].cpu().numpy()
    y_subset_int_np = y_test_int_np[indices] # Use integer NumPy array subset

    feature_importance_model = FeatureImportanceModel(models['standard'], data['feature_names'])
    # Pass NumPy arrays to compute_importance
    feature_importance = feature_importance_model.compute_importance(X_subset_np, y_subset_int_np, n_samples=10)
    models['feature_importance'] = feature_importance
    print("Feature Importance:", feature_importance)

# FGSM attack implementation
def fgsm_attack(model, X, y, epsilon):
    """
    Fast Gradient Sign Method attack
    Args:
        model: PyTorch neural network model
        X: Input tensor
        y: Target tensor
        epsilon: Attack strength parameter
    """
    # Create a copy of the input that requires gradient
    X_adv = X.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(X_adv)

    # Calculate loss
    if y.dim() == 1:
        y = y.unsqueeze(1).float()
    loss = nn.BCELoss()(outputs, y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    X_adv = X_adv + epsilon * X_adv.grad.sign()

    # Ensure adversarial examples stay within valid range
    X_adv = torch.clamp(X_adv, 0, 1)

    return X_adv.detach()

# PGD attack implementation
def pgd_attack(model, X, y, epsilon, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent attack
    Args:
        model: PyTorch neural network model
        X: Input tensor
        y: Target tensor
        epsilon: Attack strength parameter
        alpha: Step size
        num_iter: Number of iterations
    """
    # Create a copy of the input
    X_adv = X.clone().detach()

    # PGD iterations
    for _ in range(num_iter):
        X_adv.requires_grad = True

        # Forward pass
        outputs = model(X_adv)

        # Calculate loss
        if y.dim() == 1:
            y = y.unsqueeze(1).float()
        loss = nn.BCELoss()(outputs, y)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update adversarial examples
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(X_adv - X, -epsilon, epsilon)
            X_adv = X + delta

            # Ensure adversarial examples stay within valid range
            X_adv = torch.clamp(X_adv, 0, 1)

    return X_adv.detach()

# DeepFool attack implementation
def deepfool_attack(model, X, y, max_iter=10, epsilon=0.02):
    """
    DeepFool attack
    Args:
        model: PyTorch neural network model
        X: Input tensor (single example)
        y: Target tensor (single example)
        max_iter: Maximum number of iterations
        epsilon: Small constant to ensure progress
    """
    # Only support single examples for simplicity
    if X.shape[0] > 1:
        X_adv = X.clone().detach()
        for i in range(X.shape[0]):
            X_adv[i] = deepfool_attack(model, X[i:i+1], y[i:i+1], max_iter, epsilon)
        return X_adv

    model.eval()
    X_adv = X.clone().detach()

    with torch.no_grad():
        original_output = model(X_adv)

    # Binary classification: perturb toward decision boundary
    for i in range(max_iter):
        X_adv.requires_grad = True

        # Forward pass
        output = model(X_adv)

        # Binary classification: goal is to cross the decision boundary (0.5)
        target_output = 0.5

        # Calculate gradient
        loss = (output - target_output) ** 2
        model.zero_grad()
        loss.backward()

        # Get gradient
        grad = X_adv.grad.clone()
        X_adv.grad.zero_()

        # Calculate perturbation
        with torch.no_grad():
            # Normalize gradient
            grad_norm = torch.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm

            # Calculate distance to decision boundary
            output_diff = abs(output.item() - target_output)

            # Update adversarial example
            perturbation = (output_diff + epsilon) * grad
            X_adv = X_adv - perturbation

            # Ensure adversarial examples stay within valid range
            X_adv = torch.clamp(X_adv, 0, 1)

            # Check if we've crossed the decision boundary
            new_output = model(X_adv)
            if ((y > 0.5) and (new_output < 0.5)) or ((y < 0.5) and (new_output > 0.5)):
                break

    return X_adv.detach()


def train_model(X_train, y_train, X_test, y_test):
    # input_size needs to be determined from the data
    input_size = X_train.shape[1]
    model = IDSModel(input_size) # Pass input_size to the model constructor

    optimizer = optim.Adam(model.parameters(), lr = 0.001) # Using Adam for better convergence
    criterion = nn.BCELoss() # Use BCELoss for binary classification with Sigmoid output

    epochs = 10
    training_loss = []
    model.train()

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train.float().unsqueeze(1)) # y needs to be float and have shape (batch_size, 1) for BCELoss
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for e in range(epochs):
        running_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model.forward(batch_X)

            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader) # Average loss per batch
        training_loss.append(avg_epoch_loss)
        print(f"Epoch {e+1}/{epochs}, Training loss: {avg_epoch_loss:.4f}")


    # Evaluation
    model.eval()
    with torch.no_grad():
        test_output = model.forward(X_test.type(torch.FloatTensor)) # Ensure test data is FloatTensor
        # Convert probabilities to binary predictions
        predicted_labels = (test_output > 0.5).int().flatten().cpu().numpy()
        actual_labels = y_test.cpu().numpy()

    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print("accuracy_score:", accuracy * 100)
    print("Saving model...")
    path = 'standard_model.pth'
    # Save the model's state_dict
    torch.save(model.state_dict(), path)

    return model

def train_robust_model(X_train, y_train, input_size):
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train.float()).unsqueeze(1)  # Add dimension for BCE loss

    # Initialize model
    model = IDSModel(input_size)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train model with adversarial examples
    model.train()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Adversarial training loop
    epochs = 10 # Increased epochs for better robustness
    for epoch in range(epochs):
        print(f"Training robust model - Epoch {epoch+1}/{epochs}...")
        for batch_X, batch_y in loader:
            # Train on clean data
            optimizer.zero_grad()
            outputs_clean = model(batch_X)
            loss_clean = criterion(outputs_clean, batch_y)
            loss_clean.backward()
            optimizer.step()

            # Generate FGSM adversarial examples and train
            adv_X_fgsm = fgsm_attack(model, batch_X, batch_y, epsilon=0.1)
            optimizer.zero_grad()
            outputs_adv_fgsm = model(adv_X_fgsm)
            loss_adv_fgsm = criterion(outputs_adv_fgsm, batch_y)
            loss_adv_fgsm.backward()
            optimizer.step()

            # Generate PGD adversarial examples and train
            adv_X_pgd = pgd_attack(model, batch_X, batch_y, epsilon=0.1, alpha=0.01, num_iter=10)
            optimizer.zero_grad()
            outputs_adv_pgd = model(adv_X_pgd)
            loss_adv_pgd = criterion(outputs_adv_pgd, batch_y)
            loss_adv_pgd.backward()
            optimizer.step()

            # Combine losses (optional, can also train sequentially)
            # loss = 0.5 * loss_clean + 0.25 * loss_adv_fgsm + 0.25 * loss_adv_pgd
            # loss.backward()
            # optimizer.step()

    model.eval()
    print("Saving robust model...")
    torch.save(model.state_dict(), "robust_model.pth")  # Save the model
    return model

# --- Model Loading and Initialization ---
models = {}
data = {}

import warnings
from sklearn.exceptions import InconsistentVersionWarning

def load_scaler(filepath="scaler.pkl"):
    try:
        with open(filepath, "rb") as f:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always") # Always show warnings
                scaler = pickle.load(f)
                for warning_message in w:
                    if issubclass(warning_message.category, InconsistentVersionWarning):
                        print(f"Caught InconsistentVersionWarning: {warning_message.message}")
                        print("Returning None to signal scaler needs re-creation.")
                        return None # Signal that scaler needs to be re-created
        print(f"Scaler loaded successfully from {filepath}")
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {filepath}")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error loading scaler: {e}")
        print("This might be due to a change in pickle protocols or security settings.")
        print("Ensure the scaler file was saved with a compatible pickle version.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading scaler: {e}")
        return None

def initialize():
    global models, data
    scaler_filepath = "scaler.pkl"
    try:
        # Try to load the scaler and data if they exist
        # You would also need to save/load the preprocessed data (X_train, y_train, etc.)
        # for a full persistence solution. For simplicity, let's just handle the scaler for now.

        # For a more robust solution, save and load the processed tensors as well
        # For this example, we will re-preprocess if the scaler is not found.
        loaded_scaler = load_scaler(scaler_filepath)

        if loaded_scaler is not None:
            print("Scaler loaded, proceeding with data preprocessing using loaded scaler.")
            X_train, y_train, X_test, y_test, _ = data_preprocess() # Ignore the scaler returned by preprocess
            input_size = X_train.shape[1]

            data['X_train'] = X_train
            data['y_train'] = y_train
            data['X_test'] = X_test
            data['y_test'] = y_test
            data['scaler'] = loaded_scaler # Use the loaded scaler
            # Re-define feature names
            original_feature_names = ["duration","protocol_type","flag","src_bytes","dst_bytes","land",
            "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
            "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
            "num_shells","num_access_files","num_outbound_cmds","is_host_login",
            "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
            "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
            "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate", "last_flag"] # Exclude 'service' and 'attack'

            data['feature_names'] = original_feature_names

        else:
             print("Scaler file not found or loading failed, performing full data preprocessing.")
             # Perform data preprocessing as before if loading fails
             X_train, y_train, X_test, y_test, scaler = data_preprocess()
             input_size = X_train.shape[1]

             # Store data and scaler globally
             data['X_train'] = X_train
             data['y_train'] = y_train
             data['X_test'] = X_test
             data['y_test'] = y_test
             data['scaler'] = scaler # Store the newly created scaler
             # Re-define feature names
             original_feature_names = ["duration","protocol_type","flag","src_bytes","dst_bytes","land",
             "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
             "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
             "num_shells","num_access_files","num_outbound_cmds","is_host_login",
             "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
             "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
             "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate", "last_flag"] # Exclude 'service' and 'attack'

             data['feature_names'] = original_feature_names

             # Save the newly created scaler
             with open(scaler_filepath, "wb") as f:
                 pickle.dump(data['scaler'], f)
             print(f"Scaler saved to {scaler_filepath}")

        # Load or train models
        if os.path.exists("standard_model.pth") and os.path.exists("robust_model.pth"):
            standard_model = IDSModel(input_size)
            standard_model.load_state_dict(torch.load("standard_model.pth"))
            models['standard'] = standard_model
            robust_model = IDSModel(input_size)
            robust_model.load_state_dict(torch.load("robust_model.pth"))
            models['robust'] = robust_model
        else:
            standard_model = train_model(X_train, y_train, X_test, y_test, input_size)
            torch.save(standard_model.state_dict(), "standard_model.pth")  # Save standard model
            models['standard'] = standard_model
            robust_model = train_robust_model(X_train, y_train, input_size)
            models['robust'] = robust_model

        print("Evaluating models...")
        evaluate_models()
        print("Initialization complete")
    except Exception as e:
        print(f"An error occurred during initialization: {e}")

# Evaluate models and compute metrics
def evaluate_models():
    # Get data from global data dictionary
    X_test_tensor = data['X_test']  # This is a FloatTensor
    y_test_tensor_float = data['y_test'] # This is a FloatTensor (for BCE evaluation)

    # Convert y_test (float tensor) to integer NumPy array for metrics and feature importance
    # This conversion should happen here once
    y_test_int_np = y_test_tensor_float.cpu().numpy().astype(int)

    # Evaluate standard model
    models['standard'].eval()
    with torch.no_grad():
        # Use X_test_tensor for model inference
        # Output is probability (sigmoid), compare directly to 0.5
        y_pred_standard = (models['standard'](X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)

    # Compute metrics for standard model
    # Use the integer NumPy array y_test_int_np
    acc_standard = accuracy_score(y_test_int_np, y_pred_standard)
    precision_standard = precision_score(y_test_int_np, y_pred_standard, zero_division=0)
    recall_standard = recall_score(y_test_int_np, y_pred_standard, zero_division=0)
    f1_standard = f1_score(y_test_int_np, y_pred_standard, zero_division=0)
    cm_standard = confusion_matrix(y_test_int_np, y_pred_standard)

    # Store metrics
    models['standard_metrics'] = {
        'accuracy': acc_standard,
        'precision': precision_standard,
        'recall': recall_standard,
        'f1': f1_standard,
        'confusion_matrix': cm_standard.tolist()
    }

    # Evaluate robust model
    models['robust'].eval()
    with torch.no_grad():
        # Use X_test_tensor for model inference
        # Output is probability (sigmoid), compare directly to 0.5
        y_pred_robust = (models['robust'](X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)

    # Compute metrics for robust model
    # Use the integer NumPy array y_test_int_np
    acc_robust = accuracy_score(y_test_int_np, y_pred_robust)
    precision_robust = precision_score(y_test_int_np, y_pred_robust, zero_division=0)
    recall_robust = recall_score(y_test_int_np, y_pred_robust, zero_division=0)
    f1_robust = f1_score(y_test_int_np, y_pred_robust, zero_division=0)
    cm_robust = confusion_matrix(y_test_int_np, y_pred_robust)

    # Store metrics
    models['robust_metrics'] = {
        'accuracy': acc_robust,
        'precision': precision_robust,
        'recall': recall_robust,
        'f1': f1_robust,
        'confusion_matrix': cm_robust.tolist()
    }

    # Compute feature importance for standard model
    feature_importance_model = FeatureImportanceModel(models['standard'], data['feature_names'])
    # Pass NumPy arrays to compute_importance
    feature_importance = feature_importance_model.compute_importance(X_test_tensor.cpu().numpy(), y_test_int_np, n_samples=10)
    models['feature_importance'] = feature_importance


# --- Attack and Robustness Evaluation ---
def format_sample_for_display(sample_np, is_intrusion):
    # This is a placeholder. You might want to format it more nicely for display.
    # For example, mapping numerical protocol_type back to string, etc.
    # For now, just return a comma-separated string.
    return ",".join(map(str, sample_np.round(4).tolist()))

def evaluate_attack_robustness_full(model_to_evaluate, X_data_tensor, y_data_tensor_float, attack_type, epsilon, alpha=0.01, num_iter=10):
    """
    Evaluates a given model's robustness against a specified adversarial attack on the *full* dataset.

    Args:
        model_to_evaluate: The PyTorch IDSModel object.
        X_data_tensor: Full input features dataset (PyTorch Tensor, Float).
        y_data_tensor_float: Full true labels dataset (PyTorch Tensor, Float).
        attack_type: Type of attack ('fgsm', 'pgd', 'deepfool').
        epsilon: Attack strength (for FGSM/PGD).
        alpha: Step size for PGD.
        num_iter: Number of iterations for PGD/DeepFool.

    Returns:
        A dictionary containing evaluation metrics on adversarial examples.
    """
    model_to_evaluate.eval() # Set the model to evaluation mode

    print(f"Generating adversarial examples ({attack_type}) for full dataset evaluation...")

    X_adv_tensor = None
    current_y_data_tensor_float = y_data_tensor_float # Labels corresponding to X_adv

    if attack_type == 'fgsm':
        X_adv_tensor = fgsm_attack(model_to_evaluate, X_data_tensor, current_y_data_tensor_float, epsilon)
    elif attack_type == 'pgd':
        X_adv_tensor = pgd_attack(model_to_evaluate, X_data_tensor, current_y_data_tensor_float, epsilon, alpha, num_iter)
    elif attack_type == 'deepfool':
        # DeepFool on the full dataset is very slow. It's usually evaluated on a subset.
        # If you uncomment this, be prepared for very long execution times.
        # As implemented, deepfool_attack handles batches iteratively, but it's still slow.
        # print("Warning: Running DeepFool on the full dataset will be very slow.")
        # X_adv_tensor = deepfool_attack(model_to_evaluate, X_data_tensor, current_y_data_tensor_float, epsilon)
        print("DeepFool full dataset evaluation skipped due to computational cost.")
        return {'error': 'DeepFool full dataset evaluation skipped'}

    else:
        print(f"Error: Invalid attack type '{attack_type}' for full evaluation.")
        return {'error': f'Invalid attack type {attack_type}'}


    # --- Evaluate Model on Adversarial Examples ---
    with torch.no_grad():
        # Get predictions on adversarial examples
        adv_outputs = model_to_evaluate(X_adv_tensor)
        y_pred_adv_int = (adv_outputs.cpu().numpy().flatten() > 0.5).astype(int)

        # Get predictions on original examples (baseline)
        original_outputs = model_to_evaluate(X_data_tensor)
        y_pred_original_int = (original_outputs.cpu().numpy().flatten() > 0.5).astype(int)


    # --- Calculate Metrics ---
    # Convert true labels to integer NumPy array for metric calculations
    y_data_int_np = current_y_data_tensor_float.cpu().numpy().astype(int)

    accuracy = accuracy_score(y_data_int_np, y_pred_adv_int)
    precision = precision_score(y_data_int_np, y_pred_adv_int, zero_division=0)
    recall = recall_score(y_data_int_np, y_pred_adv_int, zero_division=0)
    f1 = f1_score(y_data_int_np, y_pred_adv_int, zero_division=0)
    cm = confusion_matrix(y_data_int_np, y_pred_adv_int)

    # Attack success rate: percentage of samples where the prediction changed from original to adversarial
    attack_success_rate = np.mean(y_pred_original_int != y_pred_adv_int)

    # Average perturbation (L2 norm or L-inf norm could also be used)
    # Using L1 norm for simplicity here, matching the sample perturbation
    avg_perturbation = torch.mean(torch.abs(X_adv_tensor - X_data_tensor)).item()


    print(f"Full Dataset Evaluation on Adversarial Examples ({attack_type}, epsilon={epsilon}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Attack Success Rate: {attack_success_rate:.4f}")
    print(f"Average Perturbation (L1): {avg_perturbation:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")


    # For confidence comparison, let's get confidences for a small subset of samples
    # This is a placeholder; in a real scenario, you'd want to select samples
    # that are correctly classified by the clean model but misclassified by the adversarial.
    num_samples_for_confidence = min(10, len(X_data_tensor))
    sample_indices = np.random.choice(len(X_data_tensor), num_samples_for_confidence, replace=False)

    standard_confidences = []
    robust_confidences = []

    # Ensure models are available for confidence comparison
    if 'standard' in models and 'robust' in models:
        with torch.no_grad():
            for idx in sample_indices:
                # Get original sample and its adversarial counterpart
                original_sample = X_data_tensor[idx:idx+1]
                adversarial_sample = X_adv_tensor[idx:idx+1]

                # Get confidences from both models on the adversarial sample
                standard_conf_adv = models['standard'](adversarial_sample).item()
                robust_conf_adv = models['robust'](adversarial_sample).item()

                standard_confidences.append(standard_conf_adv)
                robust_confidences.append(robust_conf_adv)

    confidence_comparison_data = {
        'sample_indices': sample_indices.tolist(),
        'standard_confidences': standard_confidences,
        'robust_confidences': robust_confidences
    }

    return {
        'attack_type': attack_type,
        'epsilon': epsilon,
        'accuracy': float(accuracy),
        'attack_success_rate': float(attack_success_rate),
        'avg_perturbation': float(avg_perturbation),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'confidence_comparison': confidence_comparison_data
    }

# --- Flask Routes ---
@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

@app.route('/sample_data')
def get_sample_data():
    with open('Train_1.txt', 'r') as f:
        samples = [next(f) for _ in range(5)]
    return jsonify({
        'samples': samples
    })

@app.route('/model_info')
def get_model_info():
    # If models and data not initialized, do it now
    if not models or not data:
        initialize()

    # Get standard model metrics
    metrics = models.get('standard_metrics') # Use .get() for safer access

    # Get model architecture (handle case where standard model might not be available)
    model_architecture = {}
    if 'standard' in models:
        model_architecture = {
            'input_size': models['standard'].layer1.in_features,
            'hidden_layers': [
                models['standard'].layer1.out_features,
                models['standard'].layer2.out_features,
                models['standard'].layer3.out_features
            ],
            'output_size': 1
        }

    # Return model info
    return jsonify({
        'accuracy': metrics.get('accuracy'), # Use .get()
        'precision': metrics.get('precision'), # Use .get()
        'recall': metrics.get('recall'), # Use .get()
        'f1': metrics.get('f1'), # Use .get()
        'model_architecture': model_architecture,
        'feature_importance': models.get('feature_importance'), # Use .get()
        'confusion_matrix': metrics.get('confusion_matrix'), # Use .get()
        'comparison_data': models.get('comparison_metrics') # Add comparison data here
    })

@app.route('/run_attack', methods=['POST'])
def run_attack():
    # Check if the request content type is JSON
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type", "message": "Request must be in JSON format"}), 415

    global models, data
    # Get attack parameters from the request
    params = request.json
    attack_type = params.get('attack_type', 'fgsm')
    epsilon = float(params.get('epsilon', 0.1))
    # Add alpha and num_iter for PGD, default if not provided
    alpha = float(params.get('alpha', 0.01))
    num_iter = int(params.get('num_iter', 10))


    # Get test data (as Tensors)
    # Ensure models and data are initialized before accessing
    if not models or not data:
        print("Models/Data not initialized. Initializing...")
        try:
            initialize()
        except Exception as e:
             return jsonify({"error": "Initialization Failed", "message": str(e)}), 500

    # Re-get data tensors after potential initialization
    if 'X_test' not in data or 'y_test' not in data:
         return jsonify({"error": "Data not available", "message": "Data preprocessing failed"}), 500

    X_test_tensor = data['X_test']
    y_test_tensor_float = data['y_test'] # Float tensor for attack functions
    # Get integer NumPy array for true labels for comparison
    y_test_int_np = y_test_tensor_float.cpu().numpy().astype(int)

    # Select the standard model to attack
    if 'standard' not in models:
         return jsonify({"error": "Model not available", "message": "Standard model not found"}), 404


    standard_model = models['standard']
    standard_model.eval()

    # Get predictions on clean data to find correctly classified examples
    with torch.no_grad():
        y_pred_standard_clean = (standard_model(X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)

    # Find correctly classified examples by the standard model
    # Use the integer NumPy array for comparison
    correct_indices = np.where(y_pred_standard_clean == y_test_int_np)[0]

    if len(correct_indices) == 0:
        return jsonify({'error': 'No correctly classified examples found by the standard model to attack.'}), 400

    # Select a random correctly classified example
    idx = np.random.choice(correct_indices)
    X_sample_tensor = X_test_tensor[idx:idx+1] # Keep as tensor for attack functions
    y_sample_tensor_float = y_test_tensor_float[idx:idx+1] # Keep as float tensor for attack functions
    y_sample_int = y_test_int_np[idx] # Get the integer label for comparison


    # --- Generate Adversarial Example for the Single Sample ---
    X_adv_sample_tensor = None
    try:
        if attack_type == 'fgsm':
            X_adv_sample_tensor = fgsm_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon)
        elif attack_type == 'pgd':
            X_adv_sample_tensor = pgd_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon, alpha, num_iter)
        elif attack_type == 'deepfool':
            # DeepFool works on single samples as well
            # Pass the specific sample
            X_adv_sample_tensor = deepfool_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon)
        else:
            return jsonify({'error': f'Invalid attack type {attack_type}'}), 400
    except Exception as e:
         return jsonify({"error": "Attack Generation Failed", "message": str(e)}), 500


    # --- Get Predictions and Confidences for the Sample ---
    with torch.no_grad():
        # Get original and adversarial outputs (confidences) for the single sample
        original_output = standard_model(X_sample_tensor).item() # Use .item() for single value tensor
        adversarial_output = standard_model(X_adv_sample_tensor).item() # Use .item()


    # Get binary predictions for the single sample
    original_prediction_int = int(original_output > 0.5)
    adversarial_prediction_int = int(adversarial_output > 0.5)

    original_prediction_label = 'intrusion' if original_prediction_int > 0.5 else 'normal'
    adversarial_prediction_label = 'intrusion' if adversarial_prediction_int > 0.5 else 'normal'
    true_label_label = 'intrusion' if y_sample_int > 0.5 else 'normal'


    # --- Calculate Sample-Specific Metrics ---
    # Check if the attack was successful for this sample
    attack_successful_for_sample = (original_prediction_int == y_sample_int) and (adversarial_prediction_int != y_sample_int)

    # Average perturbation for this sample
    sample_perturbation = torch.mean(torch.abs(X_adv_sample_tensor - X_sample_tensor)).item()


    # Convert sample tensors back to NumPy arrays for display formatting if needed
    X_sample_np = X_sample_tensor.cpu().numpy()[0]
    X_adv_sample_np = X_adv_sample_tensor.cpu().numpy()[0]


    # Format samples for display (using the scaler if available)
    original_sample_str = format_sample_for_display(X_sample_np, y_sample_int > 0.5)
    adversarial_sample_str = format_sample_for_display(X_adv_sample_np, y_sample_int > 0.5)


    # Calculate *overall* attack success rate and avg perturbation across test set
    # This requires running the attack on the full test set
    # Let's calculate the overall metrics for the standard model here for context

    overall_attack_success_rate = None
    overall_avg_perturbation = None

    # Only calculate overall metrics for attacks where it's computationally feasible on the full dataset
    if attack_type in ['fgsm', 'pgd']:
        try:
            print(f"Calculating overall {attack_type} attack metrics for standard model...")
            # Generate adversarial examples for the full test set using the standard model
            if attack_type == 'fgsm':
                X_adv_full = fgsm_attack(standard_model, X_test_tensor, y_test_tensor_float, epsilon)
            elif attack_type == 'pgd':
                 X_adv_full = pgd_attack(standard_model, X_test_tensor, y_test_tensor_float, epsilon, alpha, num_iter)

            with torch.no_grad():
                y_pred_standard_adv_full = (standard_model(X_adv_full).cpu().numpy().flatten() > 0.5).astype(int)

            # Calculate overall attack success rate on the test set
            # Percentage of samples where prediction changed from clean prediction
            overall_attack_success_rate = np.mean(y_pred_standard_clean != y_pred_standard_adv_full)
            # Calculate overall average perturbation (L1 norm)
            overall_avg_perturbation = torch.mean(torch.abs(X_adv_full - X_test_tensor)).item()

        except Exception as e:
             print(f"Error calculating overall metrics: {e}")
             # Optionally return an error for overall metrics but still return sample data
             pass # Continue to return sample-specific data


    return jsonify({
        'attack_type': attack_type,
        'epsilon': epsilon,
        'alpha': alpha, # Include PGD parameters
        'num_iter': num_iter, # Include PGD parameters
        'sample_index': int(idx), # Return the index of the chosen sample
        'original_confidence': float(original_output),
        'adversarial_confidence': float(adversarial_output),
        'original_prediction': original_prediction_label,
        'adversarial_prediction': adversarial_prediction_label,
        'true_label': true_label_label,
        'attack_successful_for_sample': bool(attack_successful_for_sample), # Whether this specific attack succeeded
        'sample_perturbation': float(sample_perturbation), # Perturbation for this sample
        'original_sample': original_sample_str,
        'adversarial_sample': adversarial_sample_str,
        'original_sample_features': X_sample_np.tolist(), # Return features as list
        'adversarial_sample_features': X_adv_sample_np.tolist(), # Return adversarial features as list
        'overall_attack_success_rate_standard': float(overall_attack_success_rate) if overall_attack_success_rate is not None else None,
        'overall_avg_perturbation_standard': float(overall_avg_perturbation) if overall_avg_perturbation is not None else None
    })


@app.route('/test_robust_model', methods=['POST'])
def robustness_check():
    # Check if the request content type is JSON
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type", "message": "Request must be in JSON format"}), 415

    global models, data
     # Ensure models and data are initialized
    if not models or not data:
        print("Models/Data not initialized. Initializing...")
        try:
            initialize()
        except Exception as e:
             return jsonify({"error": "Initialization Failed", "message": str(e)}), 500

    # Get attack parameters from the request
    params = request.json
    attack_type = params.get('attack_type', 'fgsm')
    epsilon = float(params.get('epsilon', 0.1))
    # Add alpha and num_iter for PGD, default if not provided
    alpha = float(params.get('alpha', 0.01))
    num_iter = int(params.get('num_iter', 10))

    # Get test data (as Tensors) from global data
    if 'X_test' not in data or 'y_test' not in data:
         return jsonify({"error": "Data not available", "message": "Data preprocessing failed"}), 500

    X_test_tensor = data['X_test']
    y_test_tensor_float = data['y_test'] # Float tensor for attack functions


    robustness_results = {}

    # Evaluate the standard model
    if 'standard' in models:
        print(f"Evaluating standard model robustness against {attack_type} (epsilon={epsilon})...")
        try:
            # Call evaluate_attack_robustness_full with all required arguments
            standard_model_results = evaluate_attack_robustness_full(
                models['standard'],         # model_to_evaluate
                X_test_tensor,              # X_data_tensor
                y_test_tensor_float,        # y_data_tensor_float
                attack_type,                # attack_type
                epsilon,                    # epsilon
                alpha=alpha,                # alpha (for PGD)
                num_iter=num_iter           # num_iter (for PGD/DeepFool)
            )
            robustness_results['standard'] = standard_model_results
        except Exception as e:
            robustness_results['standard'] = {'error': 'Evaluation Failed', 'message': str(e)}
            print(f"Error evaluating standard model: {e}")

    else:
        robustness_results['standard'] = {'error': 'Standard model not available'}
        print("Standard model not found in global models dictionary.")


    # Evaluate the robust model
    if 'robust' in models:
        print(f"Evaluating robust model robustness against {attack_type} (epsilon={epsilon})...")
        try:
            # Call evaluate_attack_robustness_full with all required arguments
            robust_model_results = evaluate_attack_robustness_full(
                models['robust'],           # model_to_evaluate
                X_test_tensor,              # X_data_tensor
                y_test_tensor_float,        # y_data_tensor_float
                attack_type,                # attack_type
                epsilon,                    # epsilon
                alpha=alpha,                # alpha (for PGD)
                num_iter=num_iter           # num_iter (for PGD/DeepFool)
            )
            robustness_results['robust'] = robust_model_results
        except Exception as e:
            robustness_results['robust'] = {'error': 'Evaluation Failed', 'message': str(e)}
            print(f"Error evaluating robust model: {e}")

    else:
        robustness_results['robust'] = {'error': 'Robust model not available'}
        print("Robust model not found in global models dictionary.")


    # You can add more models here if needed

    # Extract results for standard and robust models
    standard_results = robustness_results.get('standard', {})
    robust_results = robustness_results.get('robust', {})

    # Prepare data for comparison table
    # Store results in a more structured way for comparison tab
    if 'comparison_metrics' not in models:
        models['comparison_metrics'] = {
            'standard': {'clean_accuracy': 0, 'fgsm_accuracy': 0, 'pgd_accuracy': 0, 'deepfool_accuracy': 0},
            'robust': {'clean_accuracy': 0, 'fgsm_accuracy': 0, 'pgd_accuracy': 0, 'deepfool_accuracy': 0}
        }

    # Update clean accuracies (these are constant once models are trained)
    models['comparison_metrics']['standard']['clean_accuracy'] = models['standard_metrics']['accuracy']
    models['comparison_metrics']['robust']['clean_accuracy'] = models['robust_metrics']['accuracy']

    # Update attack-specific accuracies
    if standard_results.get('attack_type') == 'fgsm':
        models['comparison_metrics']['standard']['fgsm_accuracy'] = standard_results.get('accuracy', 0)
        models['comparison_metrics']['robust']['fgsm_accuracy'] = robust_results.get('accuracy', 0)
    elif standard_results.get('attack_type') == 'pgd':
        models['comparison_metrics']['standard']['pgd_accuracy'] = standard_results.get('accuracy', 0)
        models['comparison_metrics']['robust']['pgd_accuracy'] = robust_results.get('accuracy', 0)
    elif standard_results.get('attack_type') == 'deepfool':
        models['comparison_metrics']['standard']['deepfool_accuracy'] = standard_results.get('accuracy', 0)
        models['comparison_metrics']['robust']['deepfool_accuracy'] = robust_results.get('accuracy', 0)

    comparison_data = models['comparison_metrics']

    # Select a random correctly classified example from the test set to generate an adversarial sample for display
    # This is for the example comparison in the UI, not for the full robustness evaluation
    y_test_int_np = y_test_tensor_float.cpu().numpy().astype(int)
    standard_model = models['standard']
    standard_model.eval()
    with torch.no_grad():
        y_pred_standard_clean = (standard_model(X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)
    correct_indices = np.where(y_pred_standard_clean == y_test_int_np)[0]

    adversarial_sample_str = "N/A"
    true_label = "N/A"
    standard_prediction_on_adv = "N/A"
    robust_prediction_on_adv = "N/A"
    standard_confidence_on_adv = 0.0
    robust_confidence_on_adv = 0.0
    confidence_comparison_data = {}

    if len(correct_indices) > 0:
        idx = np.random.choice(correct_indices)
        X_sample_tensor = X_test_tensor[idx:idx+1]
        y_sample_tensor_float = y_test_tensor_float[idx:idx+1]
        y_sample_int = y_test_int_np[idx]

        # Generate adversarial example for this single sample
        X_adv_sample_tensor = None
        try:
            if attack_type == 'fgsm':
                X_adv_sample_tensor = fgsm_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon)
            elif attack_type == 'pgd':
                X_adv_sample_tensor = pgd_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon, alpha, num_iter)
            elif attack_type == 'deepfool':
                X_adv_sample_tensor = deepfool_attack(standard_model, X_sample_tensor, y_sample_tensor_float, epsilon)
        except Exception as e:
            print(f"Error generating single adversarial sample: {e}")

        if X_adv_sample_tensor is not None:
            # Get predictions and confidences for the single adversarial sample
            with torch.no_grad():
                standard_output_adv = models['standard'](X_adv_sample_tensor).item()
                robust_output_adv = models['robust'](X_adv_sample_tensor).item()

            standard_prediction_on_adv = 'intrusion' if standard_output_adv > 0.5 else 'normal'
            robust_prediction_on_adv = 'intrusion' if robust_output_adv > 0.5 else 'normal'
            standard_confidence_on_adv = standard_output_adv
            robust_confidence_on_adv = robust_output_adv
            true_label = 'intrusion' if y_sample_int > 0.5 else 'normal'

            X_adv_sample_np = X_adv_sample_tensor.cpu().numpy()[0]
            adversarial_sample_str = format_sample_for_display(X_adv_sample_np, y_sample_int > 0.5)

    # Extract confidence comparison data from one of the full evaluation results
    # Assuming standard_model_results will always be present if evaluation was successful
    if 'standard' in robustness_results and 'confidence_comparison' in robustness_results['standard']:
        confidence_comparison_data = robustness_results['standard']['confidence_comparison']

    return jsonify({
        'standard': standard_results,
        'robust': robust_results,
        'adversarial_sample': adversarial_sample_str,
        'true_label': true_label,
        'standard_prediction_on_adv': standard_prediction_on_adv,
        'robust_prediction_on_adv': robust_prediction_on_adv,
        'standard_confidence_on_adv': standard_confidence_on_adv,
        'robust_confidence_on_adv': robust_confidence_on_adv,
        'confidence_comparison': confidence_comparison_data,
        'comparison_data': models['comparison_metrics'] # Return the full comparison_metrics
    })

# Keep the evaluate_attack_robustness_full function as defined previously.

@app.route('/run_inference', methods=['POST'])
def run_inference():
    global models, data
    if not models or not data:
        try:
            initialize()
        except Exception as e:
            return jsonify({"error": "Initialization Failed", "message": str(e)}), 500

    try:
        # Read all lines from Test_1.txt
        with open('Test_1.txt', 'r') as f:
            lines = f.readlines()

        if not lines:
            return jsonify({"error": "Test data not available", "message": "Test_1.txt is empty."}), 500

        # Select a random line
        random_line = np.random.choice(lines)
        
        # Extract features (all but the last two columns)
        # Assuming the last column is 'last_flag' and the second to last is 'attack'
        # We need to drop 'service' column as well, which is done in data_preprocess
        # For inference, we'll assume the input format matches the preprocessed features
        
        # Temporarily parse the line to extract features, ignoring 'service' and 'attack'
        # This is a simplified parsing. A more robust solution would re-use data_preprocess logic.
        parts = random_line.strip().split(',')
        # Assuming 'service' is the 3rd column (index 2) and 'attack' is the second to last
        # We need to reconstruct the feature list as per original_feature_names
        
        # For simplicity, let's assume the random_line contains all features except 'service' and 'attack'
        # and is already in the order expected by the model after preprocessing.
        # This is a critical assumption. If Test_1.txt is raw, it needs full preprocessing.
        
        # Given the data_preprocess function, Test_1.txt is raw.
        # We need to simulate the preprocessing for a single line.
        
        # A more robust way: re-use data_preprocess for a single sample
        # For now, let's manually map based on the column names in data_preprocess
        
        # Define column names as in data_preprocess, excluding 'service' and 'attack'
        feature_columns = [
            "duration", "protocol_type", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "last_flag"
        ]
        
        # Create a temporary DataFrame for the single sample to apply preprocessing
        # This requires knowing the original column names and their order in Test_1.txt
        df_single_test = pd.read_csv(io.StringIO(random_line), sep=',', names=[
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
            "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
        ])

        # Apply the same transformations as in data_preprocess
        df_single_test.loc[df_single_test['protocol_type'] == 'icmp', "protocol_type"] = 3
        df_single_test.loc[df_single_test['protocol_type'] == 'tcp', "protocol_type"] = 1
        df_single_test.loc[df_single_test['protocol_type'] == 'udp', "protocol_type"] = 2
        
        df_single_test = df_single_test.drop('service', axis=1)

        df_single_test.loc[df_single_test['flag'] == 'REJ', "flag"] = 1
        df_single_test.loc[df_single_test['flag'] == 'SF', "flag"] = 2
        df_single_test.loc[df_single_test['flag'] == 'S0', "flag"] = 3
        df_single_test.loc[df_single_test['flag'] == 'RSTR', "flag"] = 4
        df_single_test.loc[df_single_test['flag'] == 'RSTOS0', "flag"] = 5
        df_single_test.loc[df_single_test['flag'] == 'RSTO', "flag"] = 6
        df_single_test.loc[df_single_test['flag'] == 'SH', "flag"] = 7
        df_single_test.loc[df_single_test['flag'] == 'S1', "flag"] = 8
        df_single_test.loc[df_single_test['flag'] == 'S2', "flag"] = 9
        df_single_test.loc[df_single_test['flag'] == 'S3', "flag"] = 10
        df_single_test.loc[df_single_test['flag'] == 'OTH', "flag"] = 11

        # Drop the 'attack' column for features
        df_single_test_features = df_single_test.drop('attack', axis=1)
        
        # Ensure the order of columns matches the training data
        # This is crucial. The scaler expects features in the same order it was fitted on.
        # The 'original_feature_names' in initialize() defines this order.
        if 'feature_names' not in data or not data['feature_names']:
            return jsonify({"error": "Feature names not available", "message": "Model initialization failed to set feature names."}), 500
        
        # Reorder columns to match the feature_names used during training
        df_single_test_features = df_single_test_features[data['feature_names']]

        # Convert to numpy array for scaling
        input_features_np = df_single_test_features.values.astype(float)
        
        # Scale the input data using the loaded scaler
        if 'scaler' not in data or data['scaler'] is None:
            return jsonify({"error": "Scaler not available", "message": "Scaler not loaded or trained."}), 500

        scaled_input_np = data['scaler'].transform(input_features_np)
        scaled_input_tensor = torch.FloatTensor(scaled_input_np).unsqueeze(0) # Add batch dimension

        # Run inference on standard model
        models['standard'].eval()
        with torch.no_grad():
            standard_output = models['standard'](scaled_input_tensor).item()
        standard_prediction = 'intrusion' if standard_output > 0.5 else 'normal'

        # Run inference on robust model
        models['robust'].eval()
        with torch.no_grad():
            robust_output = models['robust'](scaled_input_tensor).item()
        robust_prediction = 'intrusion' if robust_output > 0.5 else 'normal'

        return jsonify({
            'original_sample_raw': random_line.strip(), # Return the raw line from Test.txt
            'standard_prediction': standard_prediction,
            'standard_confidence': standard_output,
            'robust_prediction': robust_prediction,
            'robust_confidence': robust_output
        })

    except ValueError as ve:
        return jsonify({"error": "Data parsing error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Inference failed", "message": str(e)}), 500

if __name__ == '__main__':
    initialize()  # Initialize models and data
    app.run(debug=True, port=5001)
