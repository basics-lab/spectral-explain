
<h2 id="examples">Examples</h2>
<h3>Tabular</h3>

```python
import spectralexplain as spex
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_breast_cancer

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

print(explainer.interactions(index="fbii"))

>> Interactions(
>>   index=FBII, max_order=4, baseline_value=0.626
>>   sample_budget=1000, num_features=30,
>>   Top Interactions:
>>     (27,): -0.295
>>     (22,): -0.189
>>     (3, 6, 8, 22): 0.188
>>     (6, 10, 14, 28): 0.176
>>     (23,): -0.145
>> )
```
<h3>Sentiment Analysis</h3>

```python
import spectralexplain as spex
from transformers import pipeline

review = "Her acting never fails to impress".split()
sentiment_pipeline = pipeline("sentiment-analysis")

def sentiment_masking(X):
    masked_reviews = [" ".join([review[i] if x[i] == 1 else "[MASK]" for i in range(len(review))]) for x in X]
    return [outputs['score'] if outputs['label'] == 'POSITIVE' else 1-outputs['score'] for outputs in sentiment_pipeline(masked_reviews)]

explainer = spex.Explainer(value_function=sentiment_masking,
                           features=review,
                           sample_budget=1000)

print(explainer.interactions(index="stii"))

>> Interactions(
>>   index=STII, max_order=5, baseline_value=-0.63
>>   sample_budget=1000, num_features=6,
>>   Top Interactions:
>>     ('never', 'fails'): 2.173
>>     ('fails', 'impress'): -1.615
>>     ('never', 'fails', 'impress'): 1.592
>>     ('fails', 'to'): -1.505
>>     ('impress',): 1.436
>> )
```
