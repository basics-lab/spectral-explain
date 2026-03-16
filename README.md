<p align="center">
  <b>⚠️ NOTE: We encourage using the implementation of SPEX and ProxySPEX within <a href="https://github.com/mmschlk/shapiq">shapiq</a>, which receives much more frequent maintenance. ⚠️</b>
</p>
<h1 align="center">
<p align="center">
  <img src="https://github.com/landonbutler/landonbutler.github.io/blob/master/imgs/spex.png?raw=True" width="200" style="vertical-align: middle;">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/landonbutler/landonbutler.github.io/blob/master/imgs/ProxySPEX.png?raw=True" width="260" style="vertical-align: middle;">
</p>

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

### Requirements
To replicate the experiments in this repository, you need to install additional dependencies. To install `spectralexplain` with these optional dependencies, run:
```
git clone git@github.com:basics-lab/spectral-explain.git
cd spectral-explain
pip install -e .[dev]
```

To use the `ExactSolver` for finding the optimal value function, you will additionally need a valid [Gurobi License](https://www.gurobi.com/) configured on your machine.

For Hugging Face models, you must have your Hugging Face API Token configured as an environment variable in your terminal:
```bash
export HF_TOKEN="your_hf_token_here"
```



<h2 id="quickstart">Quickstart</h2>

`spectralexplain` can be used to quickly compute feature interactions for your models and datasets. Simply define a `value_function` which takes in a matrix of masking patterns and returns the model's outputs to masked inputs.

Upon passing this function to the `Explainer` class, alongside the number of features in your dataset, `spectralexplain` will discover feature interactions. You can specify `algorithm="proxyspex"` to use the recent [ProxySPEX](https://openreview.net/forum?id=KI8qan2EA7) algorithm, or use the default [SPEX](https://openreview.net/forum?id=pRlKbAwczl) algorithm.

Calling `explainer.interactions`, alongside a choice of interaction index, will return an `Interactions` object for any of the following interaction types:

<div align="center">
  
| Index | Full Name | Citation |
| :--- | :--- | :--- | 
| **`fourier`** | Fourier Interactions | [Ahmed et al. (1975)](https://www.researchgate.net/publication/3115888_Orthogonal_Transform_for_Digital_Signal_Processing) | 
| **`mobius`** | Möbius Interactions (Harsanyi Dividends) | [Harsanyi (1959)](https://doi.org/10.2307/2525487), [Grabisch et al. (2000)](https://www.jstor.org/stable/3690575)|
| **`bii`** | Banzhaf Interaction Index | [Grabisch et al. (2000)](https://www.jstor.org/stable/3690575) |
| **`sii`** | Shapley Interaction Index | [Grabisch et al. (2000)](https://www.jstor.org/stable/3690575) |
| **`fbii`** | Faith-Banzhaf Interaction Index | [Tsai et al. (2023)](https://jmlr.org/papers/v24/22-0202.html) |
| **`fsii`** | Faith-Shapley Interaction Index | [Tsai et al. (2023)](https://jmlr.org/papers/v24/22-0202.html) | 
| **`stii`** | Shapley-Taylor Interaction Index | [Sundararajan et al. (2020)](https://proceedings.mlr.press/v119/sundararajan20a.html) |


</div>

```python
import spectralexplain as spex

# X is a (num_samples x num_features) binary masking matrix
def value_function(X):
    return ...

explainer = spex.Explainer(
    value_function=value_function,
    features=num_features,
    algorithm="proxyspex" # Optional: defaults to "spex"
)

print(explainer.interactions(index="fbii"))
```

First, a sparse Fourier representation is learned. Then, the representation is converted to your index of choice using the conversions in [Appendix C](https://openreview.net/forum?id=pRlKbAwczl) of our paper.
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
    sample_budget=1000,
    algorithm="proxyspex"
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

<h3>Optimizing the Value Function</h3>

```python
import spectralexplain as spex

# A basic example of finding the optimal feature perturbations to maximize the value function
# given a sparse Fourier interaction representation.

solver = spex.utils.ExactSolver(
    fourier_dictionary=explainer.fourier_transform,
    maximize=True, 
    exact_solution_order=5 # Optional: specify exact number of features to select
)
optimal_features = solver.solve()
print("Optimal feature selection:", optimal_features)
```

<h2 id="citation">Citation</h2>

```bibtex
@inproceedings{
  kang2025spex,
  title={{SPEX}: Scaling Feature Interaction Explanations for {LLM}s},
  author={Justin Singh Kang and Landon Butler and Abhineet Agarwal and Yigit Efe Erginbas and Ramtin Pedarsani and Bin Yu and Kannan Ramchandran},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=pRlKbAwczl}
}

@inproceedings{
  butler2025proxyspex,
  title={ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs},
  author={Landon Butler and Abhineet Agarwal and Justin Singh Kang and Yigit Efe Erginbas and Bin Yu and Kannan Ramchandran},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=KI8qan2EA7}
}
```
