import unittest
import json
import torch
import numpy as np
from app import app, initialize, data, models

# Access global data and models directly from the app module
# This is necessary because app.data and app.models are not directly accessible
# when running tests, as they are global variables within the app module.
# We need to ensure they are initialized before tests run.


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # Ensure the environment is initialized before running tests
        initialize()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'IDS Adversarial Attack Demo', response.data)

    def test_model_info_endpoint(self):
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('accuracy', response_data)
        self.assertIn('precision', response_data)
        self.assertIn('recall', response_data)
        self.assertIn('f1', response_data)
        self.assertIn('model_architecture', response_data)
        self.assertIn('feature_importance', response_data)
        self.assertIn('confusion_matrix', response_data)

    def test_run_attack_endpoint(self):
        # Test FGSM attack
        response = self.app.post('/run_attack',
                                 data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.1}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('original_confidence', data)
        self.assertIn('adversarial_confidence', data)
        self.assertIn('original_prediction', data)
        self.assertIn('adversarial_prediction', data)

        # Test PGD attack
        response = self.app.post('/run_attack',
                                 data=json.dumps({'attack_type': 'pgd', 'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('original_confidence', data)
        self.assertIn('adversarial_confidence', data)

    def test_test_robust_model_endpoint(self):
        # Test with FGSM
        response = self.app.post('/test_robust_model',
                                 data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.1}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('standard', data)
        self.assertIn('robust', data)
        self.assertIn('accuracy', data['standard'])
        self.assertIn('accuracy', data['robust'])

        # Test with PGD
        response = self.app.post('/test_robust_model',
                                 data=json.dumps({'attack_type': 'pgd', 'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('standard', data)
        self.assertIn('robust', data)
        self.assertIn('accuracy', data['standard'])
        self.assertIn('accuracy', data['robust'])

    def test_run_inference_endpoint(self):
        # Create a sample input with the correct number of features
        input_features = np.random.rand(data['X_test'].shape[1]).tolist()
        input_data_str = ','.join(map(str, input_features))

        response = self.app.post('/run_inference',
                                 data=json.dumps({'input_data': input_data_str}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('standard_prediction', response_data)
        self.assertIn('standard_confidence', response_data)
        self.assertIn('robust_prediction', response_data)
        self.assertIn('robust_confidence', response_data)

    def test_subsequent_button_presses(self):
        # Simulate first click
        response1 = self.app.post('/run_attack',
                                  data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.1}),
                                  content_type='application/json')
        self.assertEqual(response1.status_code, 200)
        data1 = json.loads(response1.data)
        self.assertIn('adversarial_confidence', data1)

        # Simulate second click with different epsilon
        response2 = self.app.post('/run_attack',
                                  data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.2}),
                                  content_type='application/json')
        self.assertEqual(response2.status_code, 200)
        data2 = json.loads(response2.data)
        self.assertIn('adversarial_confidence', data2)

        # Check that the results are different
        self.assertNotEqual(data1['adversarial_confidence'], data2['adversarial_confidence'])

    def test_sample_data_styling(self):
        response = self.app.get('/')
        self.assertIn(b'word-wrap: break-word;', response.data)
        self.assertIn(b'white-space: pre-wrap;', response.data)

if __name__ == '__main__':
    unittest.main()