
<h1 align="center">
  <br>
  <img src="temp_logo.png" width="200">
  <br>

</h1>

<h4 align="center">Spectral Explainer: Scalable Feature Interaction Attribution</h4>


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#dependencies">Dependencies</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#citation">Citation</a>
</p>

<h2 id="installation">Installation</h2>
```
pip install spex
```

<h2 id="dependencies">Dependencies</h2>
```
pip install sparse-transform
pip install torch
pip install transformers
pip install scikit-learn
pip install openai
pip install numpy
pip install shapiq
pip install openml
pip install lime
pip install pandas
pip install pickle
pip install numba
pip install tqdm
```

<h2 id="quickstart">Quickstart</h2>

```
import spex

explicands, model = spex.modelloader.get_model("sentiment")
for explicand in explicands:
    model.set_explicand(explicand)
    interaction_values = spex.Explainer(function=model.sampling_function,
                                        index="FBII")
    print(interaction_values)
```

<h2 id="citation">Citation</h2>

```bibtex
@inproceedings{TBD,
  title     = {TBD},
  author    = {TBD},
  booktitle = {TBD},
  year      = {TBD},
  url       = {TBD}
}
```