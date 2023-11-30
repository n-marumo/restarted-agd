This repository provides codes and data for numerical experiments of our paper: https://arxiv.org/abs/2212.06410.


# Execution
To generate numerical results, run [main.py](main.py).
At the bottom of [main.py](main.py), four functions are called:
- `execute_rosenbrock`
- `execute_classification_mnist`
- `execute_ae_mnist`
- `execute_mf_movielens`

They correspond to each problem instance.
Before executing [main.py](main.py), comment out some of these function calls if necessary.
The time limit and other parameters can be changed using options.

For the execution of function `execute_mf_movielens`, the MovieLens dataset should be placed in the appropriate folder.
The default relative paths are as follows:
- `../dataset/ml-100k/u.data`
- `../dataset/ml-1m/ratings.dat`

This can be changed by editing the corresponding section of [mf_movielens.py](problem/mf_movielens.py).
The dataset can be downloaded from https://grouplens.org/datasets/movielens/.


# Result
The resulting CSV files of [main.py](main.py) are stored in [result/](result/).


# Plot
To plot the results, run the following:
- [visualizer/compare_methods_intro.py](visualizer/compare_methods_intro.py)
- [visualizer/compare_methods.py](visualizer/compare_methods.py)
- [visualizer/objlm.py](visualizer/objlm.py)

The resulting PDF files are stored in [result/](result/).


# Software & package information
We ran the code with Python 3.10.13 and the following packages.

~~~
Package             Version
------------------- ------------
absl-py             2.0.0
certifi             2023.7.22
charset-normalizer  3.3.2
chex                0.1.84
click               8.1.7
contourpy           1.2.0
cycler              0.12.1
etils               1.5.2
filelock            3.13.1
flax                0.7.5
fonttools           4.44.0
fsspec              2023.10.0
idna                3.4
importlib-resources 6.1.1
jax                 0.4.20
jaxlib              0.4.20
Jinja2              3.1.2
joblib              1.3.2
kiwisolver          1.4.5
markdown-it-py      3.0.0
MarkupSafe          2.1.3
matplotlib          3.8.1
mccabe              0.7.0
mdurl               0.1.2
ml-dtypes           0.3.1
mpmath              1.3.0
msgpack             1.0.7
mypy-extensions     1.0.0
nest-asyncio        1.5.8
networkx            3.2.1
numpy               1.26.1
opt-einsum          3.3.0
optax               0.1.7
orbax-checkpoint    0.4.2
packaging           23.2
pandas              2.1.2
pathspec            0.11.2
Pillow              10.1.0
pip                 23.3.1
platformdirs        3.11.0
protobuf            4.25.0
pycodestyle         2.11.1
pycutest            1.5.1
pyflakes            3.1.0
Pygments            2.16.1
pyparsing           3.1.1
python-dateutil     2.8.2
pytz                2023.3.post1
PyYAML              6.0.1
requests            2.31.0
rich                13.6.0
scikit-learn        1.3.2
scipy               1.11.3
seaborn             0.13.0
setuptools          65.5.0
six                 1.16.0
sympy               1.12
tensorstore         0.1.47
threadpoolctl       3.2.0
tomli               2.0.1
toolz               0.12.0
torch               2.1.0
torchvision         0.16.0
typing_extensions   4.8.0
tzdata              2023.3
urllib3             2.0.7
zipp                3.17.0
~~~