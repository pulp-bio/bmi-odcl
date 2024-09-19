# BMI-ODCL

This repository contains implementation of continual learning (CL) framework and Train-On-Request (TOR) workflow using PyTorch and on-device deployment of CL on the ultra-low power GAP9 microcontroller.

Please cite the following publication if you use our implementation of CL or on-device deployment:

```
@misc{mei2024BMIODCL,
      title={An Ultra-Low Power Wearable BMI System with Continual Learning Capabilities}, 
      author={Lan Mei, Thorir Mar Ingolfsson, Cristian Cioflan, Victor Kartsch, Andrea Cossettini, Xiaying Wang, Luca Benini},
      journal={IEEE Transactions on Biomedical Circuits and Systems}
      year={2024},
      doi={10.1109/TBCAS.2024.3457522}
}
```

Please cite the following publication if you use our TOR workflow:

```
@misc{mei2024trainonrequestondevicecontinuallearning,
      title={Train-On-Request: An On-Device Continual Learning Workflow for Adaptive Real-World Brain Machine Interfaces}, 
      author={Lan Mei, Cristian Cioflan, Thorir Mar Ingolfsson, Victor Kartsch, Andrea Cossettini, Xiaying Wang, Luca Benini},
      year={2024},
      eprint={2409.09161},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2409.09161}, 
}
```

## Requirements

This is an environment derived from [QuantLab](https://github.com/pulp-platform/quantlab) based on PyTorch 1.13.1. To install the prerequisites, create a conda environment with:
```sh
# PyTorch 1.13.1 (Recommended)
$> conda create --name pytorch-1.13
$> conda activate pytorch-1.13
$> conda config --env --add channels conda-forge
$> conda config --env --add channels pytorch 
$> conda install python=3.8 pytorch=1.13.1 pytorch-gpu torchvision=0.14.1 torchtext=0.14.1 torchaudio=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
$> conda install ipython packaging parse setuptools tensorboard tqdm networkx python-graphviz scipy pandas ipdb onnx onnxruntime einops yapf tabulate
$> pip install setuptools==59.5.0 torchsummary parse coloredlogs netron
```

For converting quantized networks from QuantLab to codes for on-device deployments with DORY, the [quantlib](https://github.com/pulp-platform/quantlib/tree/ba13b4957bd23c54d94b5aae78457b78341e76bf) quantization library is needed in the conda environment:

```
$ cd Offline-Training/quantlab-cl
$ pip install -e quantlib
```

The current `quantlib` in this repository is based on `commit:ba13b4957bd23c54d94b5aae78457b78341e76bf` with modifications on the quantization workflow to export our models correctly. 

For generating and running on-device implementation:
#### [GAP9-SDK](https://github.com/GreenWaves-Technologies/gap_sdk_private) 
Install GAP9-SDK under `BMI-ODCL/` as indicated in the associated repository. Access to this repository can be granted by [GreenWaves Technologies](https://greenwaves-technologies.com/), 

#### [DORY](https://github.com/pulp-platform/dory) 
Install Dory under `BMI-ODCL/Offline-Training/` as indicated in the associated repository. Check branch_id.log to match our working branch. Replace `dory/Parsers/Parser_ONNX_to_DORY.py` with the file already present in the current repository under `BMI-ODCL/Offline-Training/dory/Parser_ONNX_to_DORY.py`.

#### [PULP-TrainLib](https://github.com/pulp-platform/pulp-trainlib/tree/main) 
Install PULP-Trainlib under `BMI-ODCL/On-Device-Implementation/` as indicated in the associated repository. The current repository was tested with commit `6615e084738958890ea9dd10195f8bbfe089ceb7`.


## Datasets and Usage
### Datasets
This work uses two in-house EEG datasets for BMI. The datasets can be downloaded from this link: https://iis-people.ee.ethz.ch/~datasets/Datasets-ODCL/.
* **Dataset A**: An in-house EEG MM dataset for classifying left hand and right hand movements. This dataset contains seven data sessions from one subject. The stored data files are csv files.
* **Dataset B**: An in-house EEG MM/MI dataset for classifying left hand, right hand, tongue, and rest. This dataset contains data from five subjects and four sessions for each subject. The stored data files are binary files, which can be converted to csv files with `run_conversion.m` from `BMI-ODCL/Preprocessing`. The lists of file paths to be converted can be modified or added in `run_conversion.m`. The corresponding csv files will be stored in a newly created folder: `DatasetB/SubjectX_XXXX_SX/MM/csv/`.

Note that only csv files will be used in classification and the conversion of all files in Dataset B should be treated as a preprocessing step. 

### Usage
This repository contains five folders:
* **Preprocessing**:  Preprocessing codes for converting binary files of Dataset B to csv files.
* **Avalanche-Implementation-CL**: Implementation of CL algorithms on Dataset A using Avalanche in Python.
* **Offline-Training**: Offline implementations of within-session classification, transfer learning (TL) and CL workflow, and quantization using QuantLab in Python.
* **On-Device-Implementation**: Implementation of on-device TL/CL on GAP9.
* **BMI-TOR**:  Implementation of the Train-On-Request (TOR) workflow with continual learning capabilities.

Detailed descriptions and instructions of each component can be found in their respective README files. 

## Contributors
* Lan Mei, ETH Zurich lanmei@student.ethz.ch
* Thorir Mar Ingolfsson, ETH Zurich thoriri@iis.ee.ethz.ch
* Cristian Cioflan, ETH Zurich cioflanc@iis.ee.ethz.ch
* Victor Kartsch, ETH Zurich victor.kartsch@iis.ee.ethz.ch
* Andrea Cossettini, ETH Zurich cossettini.andrea@iis.ee.ethz.ch
* Xiaying Wang, ETH Zurich xiaywang@iis.ee.ethz.ch

## License
Unless explicitly stated otherwise, the code is released under Apache 2.0. Please see the LICENSE file in the root of this repository for details. 

As an exception, the weights:
* `./On-Device-Implementation/Backbone-Example/model.onnx`
* `./On-Device-Implementation/Backbone-Example/hex/*weights.hex`
* `./On-Device-Implementation/Classifier-Example/linear-data.h`
* `./On-Device-Implementation/Classifier-Example/weights_fc.npy` and `./On-Device-Implementation/Classifier-Example/bias_fc.npy`
* `./On-Device-Implementation/Classifier-Example-LwF/linear-data.h` 
* `./On-Device-Implementation/Classifier-Example-LwF/weights_fc.npy` and `./On-Device-Implementation/Classifier-Example-LwF/bias_fc.npy`

and the inputs:
* `./On-Device-Implementation/Backbone-Example/hex/*inputs.hex`
* `./On-Device-Implementation/Classifier-Example/inputs/`
* `./On-Device-Implementation/Classifier-Example-LwF/inputs/`

are released under Creative Commons Attribution-NoDerivatives 4.0 International. Please see the LICENSE file in their respective directories. 

Note that the license under which the current repository is released might differ from the license of each individual package:

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