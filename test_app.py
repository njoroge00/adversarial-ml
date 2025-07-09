import unittest
import json
from app import app, initialize

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        with app.app_context():
            initialize()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'IDS Adversarial Attack Demo', response.data)

    def test_model_info_endpoint(self):
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('accuracy', data)
        self.assertIn('precision', data)
        self.assertIn('recall', data)
        self.assertIn('f1', data)
        self.assertIn('feature_importance', data)
        self.assertIn('confusion_matrix', data)

    def test_run_attack_endpoint(self):
        response = self.app.post('/run_attack',
                                 data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.1}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('original_confidence', data)
        self.assertIn('adversarial_confidence', data)

    def test_test_robust_model_endpoint(self):
        response = self.app.post('/test_robust_model',
                                 data=json.dumps({'attack_type': 'fgsm', 'epsilon': 0.1}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('standard', data)
        self.assertIn('robust', data)
        self.assertIn('accuracy', data['standard'])
        self.assertIn('accuracy', data['robust'])

    def test_run_inference_endpoint(self):
        response = self.app.post('/run_inference', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('standard_prediction', data)
        self.assertIn('robust_prediction', data)

if __name__ == '__main__':
    unittest.main()