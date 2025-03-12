
<h1 align="center">
  <br>
  <img src="https://github.com/landonbutler/landonbutler.github.io/blob/master/imgs/spex.png?raw=True" width="200">
  <br>

</h1>

<h4 align="center">Spectral Explainer: Scalable Feature Interaction Attribution</h4>


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#examples">Examples</a> •
  <a href="#citation">Citation</a>
</p>

<h2 id="installation">Installation</h2>

To install the core `spectralexplain` package via PyPI, run:
```
pip install spectralexplain
```

To replicate the experiments in this repository, you need to install additional dependencies. To install `spectralexplain` with these optional dependencies, run:
```
git clone git@github.com:basics-lab/spectral-explain.git
cd spectral-explain
pip install -e .[dev]
```

<h2 id="quickstart">Quickstart</h2>

`spectralexplain` can be used to quickly compute feature interactions for your models and datasets. Simply define a `value_function` which takes in a matrix of masking patterns and returns the model's outputs to masked inputs.

Upon passing this function to the `Explainer` class, alongside the number of features in your dataset, `spectralexplain` will discover feature interactions.

Calling `explainer.interactions`, alongside a choice of interaction index, will return an `Interactions` object for any of the following interaction types:
- `fbii` Faith-Banzhaf Interaction Index, `fsii` Faith-Shapley Interaction Index, `stii` Shapley-Taylor Interaction Index, `bii` Banzhaf Interaction Index, `sii` Shapley Interaction Index, `fourier` Fourier Interactions, `mobius` Mobius Interactions

```python
import spectralexplain as spex

# X is a (num_samples x num_features) binary masking matrix
def value_function(X):
    return ...

explainer = spex.Explainer(
    value_function=value_function,
    features=num_features,
)

print(explainer.interactions(index="fbii"))
```
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

<h2 id="citation">Citation</h2>

```bibtex
@misc{kang2025spex,
      title={SPEX: Scaling Feature Interaction Explanations for LLMs}, 
      author={Justin Singh Kang and Landon Butler and Abhineet Agarwal and Yigit Efe Erginbas and Ramtin Pedarsani and Kannan Ramchandran and Bin Yu},
      year={2025},
      eprint={2502.13870},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.13870}, 
}
```
