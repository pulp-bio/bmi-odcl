# Offline-Training

This repository contains implementation of within-session classification, transfer learning (TL) and continual learning (CL) workflow, and quantization based on **[QuantLab]((https://github.com/pulp-platform/quantlab/tree/main))**. 

Please cite the following publication if you use our implementation(s):

```
@misc{mei2024BMIODCL,
      title={An Ultra-Low Power Wearable BMI System with Continual Learning Capabilities}, 
      author={Lan Mei, Thorir Mar Ingolfsson, Cristian Cioflan, Victor Kartsch, Andrea Cossettini, Xiaying Wang, Luca Benini},
      journal={IEEE Transactions on Biomedical Circuits and Systems}
      year={2024},
      doi={10.1109/TBCAS.2024.3457522}
}
```

## Dataset preparation
The CL implementation with Avalanche is based on two in-house EEG datasets, [Dataset A](https://iis-people.ee.ethz.ch/~datasets/Datasets-ODCL/DatasetA/) and [Dataset B](https://iis-people.ee.ethz.ch/~datasets/Datasets-ODCL/DatasetB/). 

### **N-fold within-session classification**:
To run the N-fold within-session classification experiments, put the folder containing csv files of the session to be used into the `quantlab-cl/systems/BCI_Drone_4classes/data` folder. Then, modify the corresponding path in `quantlab-cl/systems/BCI_Drone_4classes/utils/data/data.py` as the name of the data session to be classified.

For Dataset A, use the following lines and select a session folder:
```
# ============= N-fold Within-session Classification - Dataset A =============
# This name can be modified to the folder name of the data session to be classified in 5-fold CV experiments
dir_data = os.path.join(path_data, 'DatasetA_S1_1127') # 'DatasetA_S2_1130', 'DatasetA_S3_1205', ...
```

For Dataset B, uncomment the following lines and select a session folder:
```
# ============= N-fold Within-session Classification - Dataset B =============
# This name can be modified to the folder name of the data session to be classified in 5-fold CV experiments
dir_data = os.path.join(path_data, 'SubjectA_1129_S1') # 'SubjectA_1206_S2', 'SubjectA_1213_S3', ...
```

### **TL/CL workflow**:
To run the TL/CL experiments, put the folders containing csv files of the sessions to be used into the `quantlab-cl/systems/BCI_Drone_4classes/data` folder. Then for each session, copy and put the first 3 runs into a newly created train folder and the remaining 2 runs into a newly created test folder. 

As shown in `quantlab-cl/systems/BCI_Drone_4classes/utils/data/data.py`, if we conduct TL/CL on a chain of 4 sessions: SubjectA_1129_S1, SubjectA_1206_S2, SubjectA_1213_S3, SubjectA_1220_S4 of dataset B, first create two folders in each session folder, e.g., for `SubjectA_1129_S1`, create `SubjectA_1129_S1_t3v2/SubjectA_1129_S1_train` and paste the first 3 csv files inside, and create `SubjectA_1129_S1_t3v2/SubjectA_1129_S1_test` and paste the remaining 2 csv files inside. 

Similarly, for `SubjectA_1206_S2`, create `SubjectA_1206_S2_t3v2/SubjectA_1206_S2_train` and paste the first 3 csv files inside, and create `SubjectA_1206_S2_t3v2/SubjectA_1206_S2_test` and paste the remaining 2 csv files inside. For more details, please refer to the parts of codes labelled with:

```
# ============= N-repetition TL/LwF - Dataset B =============
```

## Usage
#### Configuration and Setup

To configure QuantLab, first configure where the data for and logs of your experiments should be physically stored by editing `storage_cfg.json`:
```
(pytorch-1.13) $ vim BMI-TOR/configure/storage_cfg.json
$ 
{
    'data': '/scratchb',
    'logs': '/scratcha'
}
:wq
(pytorch-1.13) $ 
```
In this case, the tool will fetch your data from the SSD drive and write logs to the HDD drive. Note that you must specify absolute (not relative) paths.
Running the `storage.sh` script creates *mock-up QuantLab homes* under both folders:
```
(pytorch-1.13) $ bash configure/storage.sh
```

QuantLab is shipped with example *problem packages* (in our case, `BCI_Drone_4classes`). Our problem package `BCI_Drone_4classes` contains a *topology sub-package*, i.e., `MIBMINet`. For this implementation, we use MI-BMINet for classifying BCI Dataset A. 
When running an experiment in QuantLab, its abstractions will look for data in a `data` sub-folder under the chosen problem package, independently of the chosen network topology.
Instead, they will log results in a `logs` sub-folder under the chosen topology sub-package.
The `problem.sh` and `topology.sh` scripts actually create such folders on the devices specified at configuration time, and then create links to these folders under the problem and topology sub-package:
```
(pytorch-1.13) $ bash configure/problem.sh BCI_Drone_4classes       
(pytorch-1.13) $ bash configure/topology.sh BCI_Drone_4classes MIBMINet  
```

### Code Preparation
The default codes conduct 2-class classification between left and right MM on Dataset A.

For 2-class classification (tongue MM / rest) on Dataset B, in `quantlab-cl/systems/BCI_Drone_4classes/utils/data/data.py`, comment out the parts of `def get_data(directory, n_files, fs, filter_config, ds=1)` labelled with:
```
# ===================== Dataset A, left / right MM - START =====================
...
# ===================== Dataset A, left / right MM -  END  =====================
```
And uncomment the parts:
```
# ===================== Dataset B, tongue MM / rest - START =====================
...
# ===================== Dataset B, tongue MM / rest -  END  =====================
```

For 4-class classification on Dataset B, uncomment the parts:
```
# ===================== Dataset B, left / right / tongue MM / rest - START =====================
...
# ===================== Dataset B, left / right / tongue MM / rest -  END  =====================
```

#### Running Experiments
To obtain within-session 5-fold CV results without quantization, go to `quantlab-cl`, and train the network with:
```
(pytorch-1.13) $ python main.py --problem=BCI_Drone_4classes --topology=MIBMINet train --exp_id=1
```
The results can be analyzed with `Offline-Training/Within-Session-Analysis.ipynb`.

To obtain chain-TL results without quantization, train the network with:
```
(pytorch-1.13) $ python main.py --problem=BCI_Drone_4classes --topology=MIBMINet train_TL --exp_id=2
```

To obtain quantized chain-TL results, train the network with:
```
(pytorch-1.13) $ python main.py --problem=BCI_Drone_4classes --topology=MIBMINet train_TL --exp_id=3
```

To obtain results by finetuning network with layers frozen except the last FCL, train the network with: 
```
(pytorch-1.13) $ python main.py --problem=BCI_Drone_4classes --topology=MIBMINet train_TL --exp_id=4 --freeze_except_last
```
Different network adaptation depth can be modified under `quantlab-cl/manager/flows/train.py` accordingly.

Hyperparameters, e.g., learning rate and number of epochs, can be adjusted by modifying `config.json` in each log folder of the experiment.

To create new experiments, replace `quantlab-cl/systems/BCI_Drone_4classes/MIBMINet/config.json` with the corresponding `quantlab-cl/systems/BCI_Drone_4classes/MIBMINet/logs/expXXXX/config.json` of the specific type of experiment, then configure the experiment with:

```
(pytorch-1.13) $ python main.py --problem=BCI_Drone_4classes --topology=MIBMINet configure
...
```
and run the experiment afterwards.

QuantLab depends on [TensorBoard](https://www.tensorflow.org/tensorboard) for enabling useful analysis and data visualisations.
After a training run has reached completion, you can inspect the logged statistics by issuing the following command:
```
(pytorch-1.13) $ tensorboard --log_dir=~/quantlab-cl/systems/BCI_Drone_4classes/MIBMINet/logs/exp0001 --port=6006
```

#### Exporting and Converting Quantized Network
To integerize and export the quantized model, go to `quantlab-cl/examples/fx_integerization`, and run `integerize_pactnets.py`. For example, to integerize and export experiment 4 (exp0004) and checkpoint 39 (epoch039.ckpt), first train and obtain the model, then run:
```
(pytorch-1.13) $ python integerize_pactnets.py --net MIBMINet --exp_id 4 --ckpt_id 39 --validate_fq --validate_tq --export_dir export_to_folder_name
```
The exported files can be found at: `quantlab-cl/examples/fx_integerization/export_to_folder_name`. Note that only quantized experiments can be integerized and exported in this way.

Afterwards, the exported files can be converted to deployable codes using DORY. To generate the C codes of the network and run the codes on gvsoc or the microcontroller, run `script_run_network_withGenerate.sh`. Note that the name of folder containing `config_MIBMINet.json` should be modified as the corresponding `export_to_folder_name`. 

## Contributors
* Lan Mei, ETH Zurich lanmei@student.ethz.ch
* Thorir Mar Ingolfsson, ETH Zurich thoriri@iis.ee.ethz.ch
* Cristian Cioflan, ETH Zurich cioflanc@iis.ee.ethz.ch
* Victor Kartsch, ETH Zurich victor.kartsch@iis.ee.ethz.ch
* Andrea Cossettini, ETH Zurich cossettini.andrea@iis.ee.ethz.ch
* Xiaying Wang, ETH Zurich xiaywang@iis.ee.ethz.ch

## Acknowledgement
The early-stopping implementation is based on: [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch/tree/master?tab=MIT-1-ov-file) with the [MIT License](https://github.com/Bjarten/early-stopping-pytorch/blob/master/LICENSE).

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