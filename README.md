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


## Models

-   `standard_model.pth`: The trained standard neural network model.
-   `robust_model.pth`: The trained robust neural network model (trained with adversarial examples).
-   `scaler.pkl`: The StandardScaler object used for feature scaling.


## Contributing

Feel free to fork the repository and contribute. Pull requests are welcome.
