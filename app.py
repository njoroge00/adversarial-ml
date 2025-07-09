from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import os
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)

# --- Data Preprocessing ---
def data_preprocess():
    # Load data
    df_train = pd.read_csv('Train_1.txt', sep=',', header=None)
    df_test = pd.read_csv('Test_1.txt', sep=',', header=None)

    # Assign column names
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate", "srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack", "last_flag"]
    df_train.columns = columns
    df_test.columns = columns

    # Combine train and test sets for consistent preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)

    # --- Feature Engineering & Cleaning ---
    # Drop 'service' column
    df = df.drop('service', axis=1)

    # Encode 'protocol_type'
    df['protocol_type'] = df['protocol_type'].map({'icmp': 3, 'tcp': 1, 'udp': 2}).fillna(0)

    # Encode 'flag'
    flag_mapping = {'REJ': 1, 'SF': 2, 'S0': 3, 'RSTR': 4, 'RSTOS0': 5, 'RSTO': 6, 'SH': 7, 'S1': 8, 'S2': 9, 'S3': 10, 'OTH': 11}
    df['flag'] = df['flag'].map(flag_mapping).fillna(0)

    # Encode 'attack' label: 1 for normal, 0 for intrusion
    df['attack'] = df['attack'].apply(lambda x: 1 if 'normal' in str(x) else 0)

    # Separate features (X) and target (y)
    X = df.drop('attack', axis=1)
    y = df['attack']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled).float()
    y_train_tensor = torch.tensor(y_train.values).float()
    X_test_tensor = torch.tensor(X_test_scaled).float()
    y_test_tensor = torch.tensor(y_test.values).float()

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler, X.columns.tolist()


# --- Model Definition ---
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

# --- Feature Importance ---
class FeatureImportanceModel:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def compute_importance(self, X_test, y_test, n_samples=100):
        importances = {}
        X_tensor = torch.FloatTensor(X_test)
        self.model.eval()
        with torch.no_grad():
            baseline_pred = (self.model(X_tensor).cpu().numpy().flatten() > 0.5).astype(int)
        baseline_acc = accuracy_score(y_test, baseline_pred)

        for i in range(X_test.shape[1]):
            acc_drops = []
            for _ in range(n_samples):
                X_permuted_np = X_test.copy()
                np.random.shuffle(X_permuted_np[:, i])
                X_permuted_tensor = torch.FloatTensor(X_permuted_np)
                with torch.no_grad():
                    perm_pred = (self.model(X_permuted_tensor).cpu().numpy().flatten() > 0.5).astype(int)
                perm_acc = accuracy_score(y_test, perm_pred)
                acc_drop = baseline_acc - perm_acc
                acc_drops.append(acc_drop)
            importance = max(0, np.mean(acc_drops))
            feature_name = f"feature_{i}" if i >= len(self.feature_names) else self.feature_names[i]
            importances[feature_name] = importance

        total_importance = sum(importances.values())
        if total_importance > 0:
            for feature in importances:
                importances[feature] /= total_importance
        return importances

# --- Attacks ---
def fgsm_attack(model, X, y, epsilon):
    X_adv = X.clone().detach().requires_grad_(True)
    outputs = model(X_adv)
    if y.dim() == 1:
        y = y.unsqueeze(1).float()
    loss = nn.BCELoss()(outputs, y)
    model.zero_grad()
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    X_adv = torch.clamp(X_adv, 0, 1)
    return X_adv.detach()

def pgd_attack(model, X, y, epsilon, alpha=0.01, num_iter=10):
    X_adv = X.clone().detach()
    for _ in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        if y.dim() == 1:
            y = y.unsqueeze(1).float()
        loss = nn.BCELoss()(outputs, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            delta = torch.clamp(X_adv - X, -epsilon, epsilon)
            X_adv = X + delta
            X_adv = torch.clamp(X_adv, 0, 1)
    return X_adv.detach()

def deepfool_attack(model, X, y, max_iter=50, epsilon=0.02):
    if X.shape[0] > 1:
        return torch.cat([deepfool_attack(model, X[i:i+1], y[i:i+1], max_iter, epsilon) for i in range(X.shape[0])])

    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)
    output = model(X_adv)
    initial_label = (output.item() > 0.5)

    for _ in range(max_iter):
        output = model(X_adv)
        current_label = (output.item() > 0.5)

        if current_label != initial_label:
            break

        # Gradient of the output with respect to the input
        output.backward()
        grad = X_adv.grad.data

        # Perturbation calculation
        f_prime = output.item()
        w_prime = grad
        
        # Perturbation to push to the boundary (0.5)
        # For sigmoid, boundary is where output is 0.5
        # The value f_prime is the current output
        perturbation = abs(f_prime - 0.5) * w_prime / (torch.norm(w_prime)**2)
        
        # Apply perturbation and a small overshoot
        X_adv = X_adv - (perturbation + epsilon * w_prime)
        X_adv = torch.clamp(X_adv, 0, 1).detach().requires_grad_(True)

    return X_adv.detach()


# --- Model Training ---
def train_model(X_train, y_train, input_size):
    model = IDSModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model.train()
    for e in range(10):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model.forward(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model

def train_robust_model(X_train, y_train, input_size):
    model = IDSModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for epoch in range(10):
        for batch_X, batch_y in loader:
            # Generate and train on FGSM examples
            adv_X_fgsm = fgsm_attack(model, batch_X, batch_y, epsilon=0.1)
            optimizer.zero_grad()
            outputs_adv_fgsm = model(adv_X_fgsm)
            loss_adv_fgsm = criterion(outputs_adv_fgsm, batch_y)
            loss_adv_fgsm.backward()
            optimizer.step()
    return model

# --- Globals and Initialization ---
models = {}
data = {}

def initialize():
    global models, data
    scaler_filepath = "scaler.pkl"
    
    X_train, y_train, X_test, y_test, scaler, feature_names = data_preprocess()
    input_size = X_train.shape[1]

    data.update({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'scaler': scaler, 'feature_names': feature_names})

    with open(scaler_filepath, "wb") as f:
        pickle.dump(scaler, f)

    if os.path.exists("standard_model.pth") and os.path.exists("robust_model.pth"):
        standard_model = IDSModel(input_size)
        standard_model.load_state_dict(torch.load("standard_model.pth"))
        robust_model = IDSModel(input_size)
        robust_model.load_state_dict(torch.load("robust_model.pth"))
    else:
        print("Training models...")
        standard_model = train_model(X_train, y_train, input_size)
        torch.save(standard_model.state_dict(), "standard_model.pth")
        robust_model = train_robust_model(X_train, y_train, input_size)
        torch.save(robust_model.state_dict(), "robust_model.pth")

    models.update({'standard': standard_model, 'robust': robust_model})
    evaluate_models()
    print("Initialization complete")

# --- Evaluation ---
def evaluate_models():
    X_test_tensor = data['X_test']
    y_test_int_np = data['y_test'].cpu().numpy().astype(int)

    for model_name in ['standard', 'robust']:
        model = models[model_name]
        model.eval()
        with torch.no_grad():
            y_pred = (model(X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test_int_np, y_pred),
            'precision': precision_score(y_test_int_np, y_pred, zero_division=0),
            'recall': recall_score(y_test_int_np, y_pred, zero_division=0),
            'f1': f1_score(y_test_int_np, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test_int_np, y_pred).tolist()
        }
        models[f'{model_name}_metrics'] = metrics

    # Feature importance for standard model
    feature_importance_model = FeatureImportanceModel(models['standard'], data['feature_names'])
    feature_importance = feature_importance_model.compute_importance(X_test_tensor.cpu().numpy(), y_test_int_np, n_samples=10)
    models['feature_importance'] = feature_importance

def evaluate_attack_robustness_full(model, X_data, y_data, attack_type, epsilon, alpha, num_iter):
    model.eval()
    if attack_type == 'fgsm':
        X_adv = fgsm_attack(model, X_data, y_data, epsilon)
    elif attack_type == 'pgd':
        X_adv = pgd_attack(model, X_data, y_data, epsilon, alpha, num_iter)
    elif attack_type == 'deepfool':
        X_adv = deepfool_attack(model, X_data, y_data, max_iter=num_iter, epsilon=epsilon)
    else:
        return {'error': f'Invalid attack type {attack_type}'}

    with torch.no_grad():
        y_pred_adv = (model(X_adv).cpu().numpy().flatten() > 0.5).astype(int)
    
    y_true = y_data.cpu().numpy().astype(int)
    accuracy = accuracy_score(y_true, y_pred_adv)
    
    return {'attack_type': attack_type, 'epsilon': epsilon, 'accuracy': float(accuracy)}

def format_sample_data(sample_array, features_per_line=10):
    formatted_lines = []
    for i in range(0, len(sample_array), features_per_line):
        formatted_lines.append(",".join(map(str, sample_array[i:i+features_per_line])))
    return "\n".join(formatted_lines)

# --- Flask Routes ---
@app.route('/')
def index():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

@app.route('/sample_data')
def get_sample_data():
    with open('Train_1.txt', 'r') as f:
        samples = [next(f) for _ in range(5)]
    return jsonify({'samples': samples})

@app.route('/model_info')
def get_model_info():
    if not models or not data:
        initialize()
    metrics = models.get('standard_metrics', {})
    # Get a random sample and its features
    idx = np.random.randint(0, len(data['X_test']))
    random_sample_features = data['X_test'][idx:idx+1].cpu().numpy().flatten().round(4).tolist()
    
    return jsonify({
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1': metrics.get('f1'),
        'feature_importance': models.get('feature_importance'),
        'confusion_matrix': metrics.get('confusion_matrix'),
        'sample_features': random_sample_features,
        'feature_names': data.get('feature_names')
    })

@app.route('/run_attack', methods=['POST'])
def run_attack():
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type"}), 415
    if not models or not data:
        initialize()

    params = request.json
    attack_type = params.get('attack_type', 'fgsm')
    epsilon = float(params.get('epsilon', 0.1))
    alpha = float(params.get('alpha', 0.01))
    num_iter = int(params.get('num_iter', 20))

    X_test, y_test = data['X_test'], data['y_test']
    y_test_int = y_test.cpu().numpy().astype(int)
    
    standard_model = models['standard']
    standard_model.eval()
    with torch.no_grad():
        y_pred_clean = (standard_model(X_test).cpu().numpy().flatten() > 0.5).astype(int)
    
    # Find a sample that is predicted as intrusion on the original sample and misclassified as normal on the adversarial sample
    X_sample = None
    y_sample = None
    y_sample_int = None
    max_attempts = 100  # Limit attempts to find a suitable sample

    for _ in range(max_attempts):
        idx = np.random.randint(0, len(X_test))
        temp_X_sample, temp_y_sample = X_test[idx:idx+1], y_test[idx:idx+1]
        temp_y_sample_int = y_test_int[idx]

        # Only consider samples that are truly 'intrusion' (label 0)
        if temp_y_sample_int == 1: # 1 means normal, 0 means intrusion
            continue

        with torch.no_grad():
            original_output_temp = standard_model(temp_X_sample).item()
        
        # Check if original sample is predicted as intrusion (output > 0.5 for normal, so < 0.5 for intrusion)
        if original_output_temp < 0.5: # Predicted as intrusion
            # Generate adversarial sample
            if attack_type == 'fgsm':
                X_adv_sample_temp = fgsm_attack(standard_model, temp_X_sample, temp_y_sample, epsilon)
            elif attack_type == 'pgd':
                X_adv_sample_temp = pgd_attack(standard_model, temp_X_sample, temp_y_sample, epsilon, alpha, num_iter)
            elif attack_type == 'deepfool':
                X_adv_sample_temp = deepfool_attack(standard_model, temp_X_sample, temp_y_sample, max_iter=num_iter, epsilon=epsilon)
            else:
                continue # Should not happen with valid attack_type

            with torch.no_grad():
                adversarial_output_temp = standard_model(X_adv_sample_temp).item()
            
            # Check if adversarial sample is misclassified as normal (output > 0.5 for normal)
            if adversarial_output_temp > 0.5: # Misclassified as normal
                X_sample = temp_X_sample
                y_sample = temp_y_sample
                y_sample_int = temp_y_sample_int
                X_adv_sample = X_adv_sample_temp
                original_output = original_output_temp
                adversarial_output = adversarial_output_temp
                break
    
    if X_sample is None:
        return jsonify({'error': 'Could not find a suitable sample for this attack (intrusion -> normal misclassification). Try different parameters or attack type.'}), 400

    if attack_type == 'fgsm':
        X_adv_sample = fgsm_attack(standard_model, X_sample, y_sample, epsilon)
    elif attack_type == 'pgd':
        X_adv_sample = pgd_attack(standard_model, X_sample, y_sample, epsilon, alpha, num_iter)
    elif attack_type == 'deepfool':
        X_adv_sample = deepfool_attack(standard_model, X_sample, y_sample, max_iter=num_iter, epsilon=epsilon)
    else:
        return jsonify({'error': f'Invalid attack type {attack_type}'}), 400

    with torch.no_grad():
        original_output = standard_model(X_sample).item()
        adversarial_output = standard_model(X_adv_sample).item()

    original_prediction_label = 'intrusion' if original_output < 0.5 else 'normal'
    adversarial_prediction_label = 'intrusion' if adversarial_output < 0.5 else 'normal'
    true_label = 'intrusion' if y_sample_int == 0 else 'normal'

    return jsonify({
        'attack_type': attack_type,
        'epsilon': epsilon,
        'original_confidence': float(original_output),
        'adversarial_confidence': float(adversarial_output),
        'original_prediction': original_prediction_label,
        'adversarial_prediction': adversarial_prediction_label,
        'true_label': true_label,
        'original_sample': format_sample_data(X_sample.cpu().numpy().flatten().round(4)),
        'adversarial_sample': format_sample_data(X_adv_sample.cpu().numpy().flatten().round(4)),
    })

@app.route('/test_robust_model', methods=['POST'])
def robustness_check():
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type"}), 415
    if not models or not data:
        initialize()

    params = request.json
    attack_type = params.get('attack_type', 'fgsm')
    epsilon = float(params.get('epsilon', 0.1))
    alpha = float(params.get('alpha', 0.01))
    num_iter = int(params.get('num_iter', 20))

    X_test, y_test = data['X_test'], data['y_test']
    
    standard_results = evaluate_attack_robustness_full(models['standard'], X_test, y_test, attack_type, epsilon, alpha, num_iter)
    robust_results = evaluate_attack_robustness_full(models['robust'], X_test, y_test, attack_type, epsilon, alpha, num_iter)

    # Update comparison metrics
    if 'comparison_metrics' not in models:
        models['comparison_metrics'] = {'standard': {}, 'robust': {}}
    
    models['comparison_metrics']['standard'][f'{attack_type}_accuracy'] = standard_results.get('accuracy', 0)
    models['comparison_metrics']['robust'][f'{attack_type}_accuracy'] = robust_results.get('accuracy', 0)
    models['comparison_metrics']['standard'][f'{attack_type}_epsilon'] = epsilon
    models['comparison_metrics']['robust'][f'{attack_type}_epsilon'] = epsilon

    # Find a specific adversarial sample that fools the standard model but not the robust model
    robust_example = {
        'original_sample': None,
        'adversarial_sample': None,
        'original_prediction': None,
        'adversarial_standard_prediction': None,
        'adversarial_robust_prediction': None,
        'true_label': None
    }
    
    max_robust_attempts = 100
    for _ in range(max_robust_attempts):
        idx = np.random.randint(0, len(X_test))
        temp_X_sample, temp_y_sample = X_test[idx:idx+1], y_test[idx:idx+1]
        temp_y_sample_int = y_test.cpu().numpy().astype(int)[idx]

        # We are looking for an intrusion sample (true label 0)
        if temp_y_sample_int == 1: # 1 means normal, 0 means intrusion
            continue

        # Generate adversarial sample for standard model
        if attack_type == 'fgsm':
            X_adv_sample_temp = fgsm_attack(models['standard'], temp_X_sample, temp_y_sample, epsilon)
        elif attack_type == 'pgd':
            X_adv_sample_temp = pgd_attack(models['standard'], temp_X_sample, temp_y_sample, epsilon, alpha, num_iter)
        elif attack_type == 'deepfool':
            X_adv_sample_temp = deepfool_attack(models['standard'], temp_X_sample, temp_y_sample, max_iter=num_iter, epsilon=epsilon)
        else:
            continue

        with torch.no_grad():
            original_standard_output = models['standard'](temp_X_sample).item()
            adversarial_standard_output = models['standard'](X_adv_sample_temp).item()
            adversarial_robust_output = models['robust'](X_adv_sample_temp).item()

        original_standard_pred_label = 'intrusion' if original_standard_output < 0.5 else 'normal'
        adversarial_standard_pred_label = 'intrusion' if adversarial_standard_output < 0.5 else 'normal'
        adversarial_robust_pred_label = 'intrusion' if adversarial_robust_output < 0.5 else 'normal'
        true_label_str = 'intrusion' if temp_y_sample_int == 0 else 'normal'

        # Condition: standard model predicts intrusion on original, normal on adversarial
        # AND robust model predicts intrusion on adversarial
        if (original_standard_pred_label == 'intrusion' and 
            adversarial_standard_pred_label == 'normal' and 
            adversarial_robust_pred_label == 'intrusion'):
            
            robust_example['original_sample'] = format_sample_data(temp_X_sample.cpu().numpy().flatten().round(4))
            robust_example['adversarial_sample'] = format_sample_data(X_adv_sample_temp.cpu().numpy().flatten().round(4))
            robust_example['original_prediction'] = original_standard_pred_label
            robust_example['adversarial_standard_prediction'] = adversarial_standard_pred_label
            robust_example['adversarial_robust_prediction'] = adversarial_robust_pred_label
            robust_example['true_label'] = true_label_str
            break

    return jsonify({
        'standard': standard_results,
        'robust': robust_results,
        'comparison_data': models['comparison_metrics'],
        'robust_example': robust_example
    })

@app.route('/run_inference', methods=['POST'])
def run_inference():
    if not models or not data:
        initialize()
    
    idx = np.random.randint(0, len(data['X_test']))
    sample_tensor = data['X_test'][idx:idx+1]
    
    with torch.no_grad():
        standard_output = models['standard'](sample_tensor).item()
        robust_output = models['robust'](sample_tensor).item()

    standard_prediction = 'intrusion' if standard_output < 0.5 else 'normal'
    robust_prediction = 'intrusion' if robust_output < 0.5 else 'normal'

    return jsonify({
        'original_sample_raw': format_sample_data(sample_tensor.cpu().numpy().flatten().round(4)),
        'standard_prediction': standard_prediction,
        'standard_confidence': standard_output,
        'robust_prediction': robust_prediction,
        'robust_confidence': robust_output
    })

if __name__ == '__main__':
    initialize()
    app.run(debug=True, port=5001)