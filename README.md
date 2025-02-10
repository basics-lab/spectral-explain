
<h1 align="center">
  <br>
  <img src="temp_logo.png" width="200">
  <br>

</h1>

<h4 align="center">Spectral Explainer: Scalable Feature Interaction Attribution</h4>


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#citation">Citation</a>
</p>

<h2 id="installation">Installation</h2>

To install the core `spex` package via PyPI, run:

```
pip install spex
```

To replicate the experiments in this repository, you need to install additional dependencies. To install `spex` with these optional dependencies, run:


```
git clone git@github.com:basics-lab/spectral-explain.git
cd spectral-explain
pip install -e .[dev]
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