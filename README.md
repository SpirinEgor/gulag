# GULAG: GUessing LAnGuages with neural networks

[![Main](https://github.com/SpirinEgor/gulag/actions/workflows/main.yaml/badge.svg)](https://github.com/SpirinEgor/gulag/actions/workflows/main.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![GitHub license](https://img.shields.io/github/license/SpirinEgor/gulag)](https://github.com/SpirinEgor/gulag/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SpirinEgor/gulag?style=social)](https://github.com/SpirinEgor/gulag/stargazers)

<p align="center">
  <img src="https://i.ibb.co/Y81PByz/htmlconvd-JIAk6-X31x1.jpg" alt="cannon on sparrows"/>
</p>

Classify languages in text via neural networks.

```
> Привет! My name is Egor. Was für ein herrliches Frühlingswetter, хутка расцвітуць дрэвы.
ru -- Привет
en -- My name is Egor
de -- Was für ein herrliches Frühlingswetter
be -- хутка расцвітуць дрэвы
```

# Usage

Use [`requirements.txt`](./requirements.txt) to install necessary dependencies:
```shell
pip install -r requirements.txt
```

After that you can either train model:
```shell
python -m src.main train --gin-file config/train.gin
```
Or run inference:
```shell
python -m src.main infer
```

# Training

All training details are covered by [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/stable/).
There are:
- [`MultiLangugeClassifier`](./src/model/multilanguage_classifier.py): lightning module that encapsulates model details
- [`MultiLanguageClassificationDataModule`](./src/data/data_module.py): lightning data module that encapsulates data details

Both modules have explicit documentation, see source files for usage details.

## Dataset

Since extracting languages from a text is a kind of synthetic task, then there is no exact dataset of that.
A possible approach to handle this is to use general multilingual corpses to create a synthetic dataset with multiple languages per one text.
Although there is a popular [mC4](https://huggingface.co/datasets/mc4) dataset with large texts in over 100 languages.
It is too large for this pet project.
Therefore, I used [wikiann](https://huggingface.co/datasets/wikiann) dataset that also supports over 100 languages including
Russian, Ukrainian, Belarusian, Kazakh, Azerbaijani, Armenian, Georgian, Hebrew, English, and German.
But this dataset consists of only small sentences for NER classification that make it more unnatural.

### Synthetic data

To create a dataset with multiple languages per example, I use the following sampling strategy:
1. Select number of languages in next example
2. Select number of sentences for each language
3. Sample sentences, shuffle them and concatenate into single text

For exact details about sampling algorithm see [`generate_example`](./src/data/dataset.py) method.

This strategy allows training on a large non-repeating corpus.
But for proper evaluation during training, we need a deterministic subset of data.
For that, we can pre-generate a bunch of texts and then reuse them on each validation.

## Model

As a training objective, I selected per-token classification.
This automatically allows not only classifying languages in the text, but also specifying their ranges.

The model consists of two parts:
1. The backbone model that embeds tokens into vectors
2. Head classifier that predicts classes by embedding vector

As backbone model I selected vanilla [BERT](https://huggingface.co/bert-base-multilingual-cased).
This model already pretrained on large multilingual corpora including non-popular languages.
During training on a target task, weights of BERT were frozen to enhance speed.

Head classifier is a simple MLP, see [`TokenClassifier`](./src/model/token_classifier.py) for details.

## Configuration

To handle big various of parameters, I used [`gin-config`](https://github.com/google/gin-config/).
[`config`](./config) folder contains all configurations split by modules that used them.

Use `--gin-file` CLI argument to specify config file and `--gin-param` to manually overwrite some values.
For example, to run debug mode on a small subset with a tiny model for 10 steps use
```shell
python -m src.main train --gin-file config/debug.gin --gin-param="train.n_steps = 10"
```

You can also use jupyter notebook to run training, this is a convenient way to train with Google Colab.
See [`train.ipynb`](./notebooks/train.ipynb).

## Artifacts

All training logs and artifacts are stored on [W&B](https://wandb.ai/).
See [voudy/gulag](https://wandb.ai/voudy/gulag?workspace=user-voudy) for information about current runs, their losses and metrics.
Any of the presented models may be used on inference.

# Inference

In inference mode, you may play with the model to see whether it is good or not.
This script requires a W&B run path where checkpoint is stored and checkpoint name.
After that, you can interact with a model in a loop.

```shell
$ python -m src.main infer --wandb "voudy/gulag/1ykm0l2n" --ckpt "step_20000.ckpt.ckpt"
...
Enter text to classify languages (Ctrl-C to exit):
> İrəli! Вперёд! Nach vorne!
az -- İrəli
ru -- Вперёд
de -- Nach vorne
Enter text to classify languages (Ctrl-C to exit):
> ...
```

For now, text preprocessing removes all punctuation and digits.
It makes the data more robust.
But restoring them back is a straightforward technical work that I was lazy to do.

Of course, you can use model from the Jupyter Notebooks, see [`infer.ipynb`](./notebooks/infer.ipynb)

# Further work

Next steps may include:
- Improved dataset with more natural examples, e.g. adopt mC4.
- Better tokenization to handle rare languages, this should help with problems on the bounds of similar texts.
- Experiments with another embedders,
e.g. [mGPT-3](https://huggingface.co/sberbank-ai/mGPT) from Sber covers all interesting languages, but requires technical work to adopt for classification task.
