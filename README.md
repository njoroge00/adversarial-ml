# IDS Adversarial Attack Demo

This project demonstrates the vulnerability of a simple Intrusion Detection System (IDS) to adversarial attacks and the effectiveness of a robustly trained model in defending against them.

## Features

-   **Data Preprocessing:** Loads and preprocesses network traffic data.
-   **IDS Models:** Trains a standard neural network and a robustly trained neural network for intrusion detection.
-   **Adversarial Attacks:** Implements FGSM, PGD, and DeepFool attacks to generate adversarial examples.
-   **Model Evaluation:** Provides performance metrics (accuracy, precision, recall, F1-score) and feature importance.
-   **Interactive Web Interface:** A Flask-based web application to visualize:
    -   Model performance and confusion matrix.
    -   How adversarial attacks can fool the standard model.
    -   The robustness of the robustly trained model against these attacks.
    -   Comparison of standard and robust model performance under attack.
    -   Real-time inference on random samples.

## Getting Started

### Prerequisites

-   Python 3.8+
-   pip (Python package installer)
-   `Train_1.txt` and `Test_1.txt` data files (should be in the project root directory).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd adversarial-ml
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, you can generate one or install manually:
    ```bash
    pip install Flask numpy pandas torch scikit-learn
    ```

### Running the Application

1.  **Start the Flask server:**
    ```bash
    python3 app.py
    ```
    The application will typically run on `http://127.0.0.1:5001/`.

2.  **Open your web browser** and navigate to the address provided in the console.

### Running Tests

To run the backend unit tests:

```bash
python3 -m unittest test_app.py
```

## Project Structure

```
.gitignore
app.py
index.html
my_mlp_model.pth
README.md
robust_model.pth
scaler.pkl
standard_model.pth
Test_1.txt
test_app.py
Train_1.txt
venv/
```

## Models

-   `standard_model.pth`: The trained standard neural network model.
-   `robust_model.pth`: The trained robust neural network model (trained with adversarial examples).
-   `scaler.pkl`: The StandardScaler object used for feature scaling.

These files will be generated upon the first run of `app.py` if they do not exist.

## Contributing

Feel free to fork the repository and contribute. Pull requests are welcome.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).