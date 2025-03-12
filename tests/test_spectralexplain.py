import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_breast_cancer

import spectralexplain as spex

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TestSpectralExplain(unittest.TestCase):    
    def test_tabular_interactions(self):
        data, target = load_breast_cancer(return_X_y=True)
        test_point, data, target = data[0], data[1:], target[1:]
        
        model = RandomForestRegressor().fit(data, target)
        
        def tabular_masking(X):
            return model.predict(np.where(X, test_point, data.mean(axis=0)))
        
        explainer = spex.Explainer(
            value_function=tabular_masking,
            features=range(len(test_point)),
            sample_budget=1000
        )
        
        interactions = explainer.interactions(index="fbii")
        self.assertIsNotNone(interactions)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Skipping sentiment analysis test because transformers is not installed.")
    def test_sentiment_interactions(self):
        review = "Her acting never fails to impress".split()
        sentiment_pipeline = pipeline("sentiment-analysis")
        
        def sentiment_masking(X):
            masked_reviews = [" ".join([review[i] if x[i] == 1 else "[MASK]" for i in range(len(review))]) for x in X]
            return [outputs['score'] if outputs['label'] == 'POSITIVE' else 1 - outputs['score'] for outputs in sentiment_pipeline(masked_reviews)]
        
        explainer = spex.Explainer(value_function=sentiment_masking,
                                   features=review,
                                   sample_budget=1000)
        
        interactions = explainer.interactions(index="stii")
        self.assertIsNotNone(interactions)

if __name__ == "__main__":
    unittest.main()
