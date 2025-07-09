import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import functions and classes from app.py
from app import app, data_preprocess, IDSModel, fgsm_attack, pgd_attack, deepfool_attack, initialize, models, data

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure models and data are initialized before running tests
        # This might take some time as it trains models if not already saved
        with patch('app.evaluate_models'): # Prevent evaluate_models from running during init
            initialize()

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_data_preprocess(self):
        X_train, y_train, X_test, y_test, scaler, feature_names = data_preprocess()
        self.assertIsInstance(X_train, torch.Tensor)
        self.assertIsInstance(y_train, torch.Tensor)
        self.assertIsInstance(X_test, torch.Tensor)
        self.assertIsInstance(y_test, torch.Tensor)
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(X_train.shape[1], len(feature_names))

    def test_ids_model_forward(self):
        input_size = data['X_train'].shape[1]
        model = IDSModel(input_size)
        # Create a dummy input tensor
        dummy_input = torch.randn(1, input_size).float()
        output = model(dummy_input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 1))
        self.assertGreaterEqual(output.item(), 0.0)
        self.assertLessEqual(output.item(), 1.0)

    def test_fgsm_attack(self):
        input_size = data['X_train'].shape[1]
        model = IDSModel(input_size)
        # Dummy data
        X = torch.randn(1, input_size).float()
        y = torch.tensor([0.0]).float()
        epsilon = 0.1
        X_adv = fgsm_attack(model, X, y, epsilon)
        self.assertIsInstance(X_adv, torch.Tensor)
        self.assertEqual(X_adv.shape, X.shape)
        # Check if perturbation is applied (simple check)
        self.assertFalse(torch.equal(X, X_adv))

    def test_pgd_attack(self):
        input_size = data['X_train'].shape[1]
        model = IDSModel(input_size)
        # Dummy data
        X = torch.randn(1, input_size).float()
        y = torch.tensor([0.0]).float()
        epsilon = 0.1
        alpha = 0.01
        num_iter = 5
        X_adv = pgd_attack(model, X, y, epsilon, alpha, num_iter)
        self.assertIsInstance(X_adv, torch.Tensor)
        self.assertEqual(X_adv.shape, X.shape)
        self.assertFalse(torch.equal(X, X_adv))

    def test_deepfool_attack(self):
        input_size = data['X_train'].shape[1]
        model = IDSModel(input_size)
        # Dummy data
        X = torch.randn(1, input_size).float()
        y = torch.tensor([0.0]).float()
        epsilon = 0.02
        X_adv = deepfool_attack(model, X, y, epsilon=epsilon)
        self.assertIsInstance(X_adv, torch.Tensor)
        self.assertEqual(X_adv.shape, X.shape)
        self.assertFalse(torch.equal(X, X_adv))

    def test_model_info_endpoint(self):
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('accuracy', data)
        self.assertIn('feature_importance', data)
        self.assertIn('confusion_matrix', data)
        self.assertIn('sample_features', data)
        self.assertIn('feature_names', data)
        self.assertIsInstance(data['sample_features'], list)
        self.assertIsInstance(data['feature_names'], list)

    def test_run_attack_endpoint(self):
        # Mock the attack functions to control output for testing specific scenarios
        with patch('app.fgsm_attack') as mock_fgsm:
            with patch('app.pgd_attack') as mock_pgd:
                with patch('app.deepfool_attack') as mock_deepfool:
                    # Simulate a scenario where an intrusion sample is found and misclassified
                    # Mock fgsm_attack to return a perturbed sample that will be misclassified
                    mock_fgsm.return_value = data['X_test'][0:1] + 0.5 # Large perturbation to force misclassification
                    mock_pgd.return_value = data['X_test'][0:1] + 0.5
                    mock_deepfool.return_value = data['X_test'][0:1] + 0.5

                    # Find an actual intrusion sample from the test set for the mock
                    intrusion_idx = -1
                    for i, label in enumerate(data['y_test']):
                        if label.item() == 0: # 0 is intrusion
                            intrusion_idx = i
                            break
                    
                    if intrusion_idx == -1:
                        self.skipTest("No intrusion samples found in test data for this test.")

                    # Temporarily replace X_test and y_test in app.data to control the sample picked
                    original_X_test = data['X_test']
                    original_y_test = data['y_test']
                    data['X_test'] = original_X_test[intrusion_idx:intrusion_idx+1]
                    data['y_test'] = original_y_test[intrusion_idx:intrusion_idx+1]

                    response = self.app.post('/run_attack', json={'attack_type': 'fgsm', 'epsilon': 0.1})
                    self.assertEqual(response.status_code, 200)
                    result = response.get_json()
                    self.assertIn('original_prediction', result)
                    self.assertIn('adversarial_prediction', result)
                    self.assertIn('true_label', result)
                    self.assertEqual(result['true_label'], 'intrusion') # Ensure we picked an intrusion sample
                    self.assertEqual(result['original_prediction'], 'intrusion') # Should be intrusion initially
                    self.assertEqual(result['adversarial_prediction'], 'normal') # Should be misclassified as normal

                    # Restore original data
                    data['X_test'] = original_X_test
                    data['y_test'] = original_y_test

    def test_robustness_check_endpoint(self):
        response = self.app.post('/test_robust_model', json={'attack_type': 'fgsm', 'epsilon': 0.1})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('standard', data)
        self.assertIn('robust', data)
        self.assertIn('robust_example', data)
        self.assertIn('accuracy', data['standard'])
        self.assertIn('accuracy', data['robust'])

    def test_run_inference_endpoint(self):
        response = self.app.post('/run_inference')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('original_sample_raw', data)
        self.assertIn('standard_prediction', data)
        self.assertIn('robust_prediction', data)

if __name__ == '__main__':
    unittest.main()