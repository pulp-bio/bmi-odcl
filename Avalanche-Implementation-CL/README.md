# Avalanche-Implementation-CL

This repository contains implementation of continual learning (CL) framework with **[Avalanche](https://avalanche.continualai.org/)** using PyTorch.

Please cite the following publication if you use these implementations:

```
@misc{mei2024BMIODCL,
      title={An Ultra-Low Power Wearable BMI System with Continual Learning Capabilities}, 
      author={Lan Mei, Thorir Mar Ingolfsson, Cristian Cioflan, Victor Kartsch, Andrea Cossettini, Xiaying Wang, Luca Benini},
      journal={IEEE Transactions on Biomedical Circuits and Systems}
      year={2024},
      doi={10.1109/TBCAS.2024.3457522}
}
```

## Requirements
To install the prerequisites, create a conda environment with:

```
$ conda create --name avalanche
$ conda activate avalanche
(avalanche) $ pip install avalanche-lib
(avalanche) $ conda install pandas
(avalanche) $ pip install ipykernel
(avalanche) $ python -m ipykernel install --user --name=my-python3-kernel
(avalanche) $ jupyter notebook
```
The Jupyter Notebook files can then run on the newly registered kernel of the environment. Note that the Avalanche library is needed for running the codes, where PyTorch will be installed automatically with Avalanche.

## Datasets and Usage
### Datasets
The CL implementation with Avalanche is based on two in-house EEG datasets, [Dataset A](https://iis-people.ee.ethz.ch/~datasets/Datasets-ODCL/DatasetA/) and [Dataset B](https://iis-people.ee.ethz.ch/~datasets/Datasets-ODCL/DatasetB/).

### Usage
* **CL-Metrics-Avalanche-DatasetA.ipynb**: Run TL/CL algorithms (naive TL, ER-buffer-20, ER-buffer-200, LwF, EWC) on seven sessions of Dataset A. Compute metrics including accuracy, precision, recall, and specificity.
* **CL-Metrics-Avalanche-DatasetB-2classes.ipynb**: Run TL/CL algorithms (naive TL, ER-buffer-20, ER-buffer-200, LwF, EWC) on four MM sessions of Dataset B, subject A. Compute metrics including accuracy, precision, recall, and specificity.
* **CL-Metrics-Avalanche-DatasetB-4classes.ipynb**: Run TL/CL algorithms (naive TL, ER-buffer-20, ER-buffer-200, LwF, EWC) on four sessions of Dataset B, subject A. Compute accuracy.
Execute the code blocks in Jupyter Notebook files to reproduce these workflows.

## Contributors
* Lan Mei, ETH Zurich lanmei@student.ethz.ch
* Thorir Mar Ingolfsson, ETH Zurich thoriri@iis.ee.ethz.ch
* Cristian Cioflan, ETH Zurich cioflanc@iis.ee.ethz.ch
* Victor Kartsch, ETH Zurich victor.kartsch@iis.ee.ethz.ch
* Andrea Cossettini, ETH Zurich cossettini.andrea@iis.ee.ethz.ch
* Xiaying Wang, ETH Zurich xiaywang@iis.ee.ethz.ch

## License
Unless explicitly stated otherwise, the code is released under Apache 2.0. Please see the LICENSE file in the root of this repository for details. Note that the license under which the current repository is released might differ from the license of each individual package:

* Avalanche - [MIT License](https://github.com/ContinualAI/avalanche/blob/master/LICENSE);
* PyTorch - a [mix of licenses](https://github.com/pytorch/pytorch/blob/master/NOTICE), including the Apache 2.0 License and the 3-Clause BSD License;
* TensorBoard - [Apache 2.0 License](https://github.com/tensorflow/tensorboard/blob/master/LICENSE);
* NetworkX - [3-Clause BSD License](https://github.com/networkx/networkx/blob/main/LICENSE.txt);
* GraphViz - [MIT License](https://github.com/graphp/graphviz/blob/master/LICENSE);
* matplotlib - a [custom license](https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE);
* NumPy - [3-Clause BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt);
* SciPy - [3-Clause BSD License](https://github.com/scipy/scipy/blob/master/LICENSE.txt);
* Mako - [MIT License](https://github.com/sqlalchemy/mako/blob/master/LICENSE);
* Jupyter - [3-Clause BSD License](https://github.com/jupyter/notebook/blob/master/LICENSE);
* Pandas - [3-Clause BSD License](https://github.com/pandas-dev/pandas/blob/main/LICENSE);
* early-stopping-pytorch - [MIT License](https://github.com/Bjarten/early-stopping-pytorch/blob/master/LICENSE).