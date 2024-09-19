# FX Integerization Example Script
The `integerize_pactnets.py` script allows you to load checkpoints, quantize, evaluate, integerize and export integerized ONNX graphs of the following topologies/problems:
* BCI_Drone_4classes/MIBMINet

You can specify the network (the problem is implicit as each network only targets 1 problem), QuantLab experiment ID, export directory and name and whether to validate the chosen network before and/or after integerization. For example, to integerize and export experiment 4 (exp0004) and checkpoint 39 (epoch039.ckpt), first train and obtain the model, then run:
```
(pytorch-1.13) $ python integerize_pactnets.py --net MIBMINet --exp_id 4 --ckpt_id 39 --validate_fq --validate_tq --export_dir export_to_folder_name
```
For exporting more than one input trial, for example 100 trials, run:
```
(pytorch-1.13) $ python integerize_pactnets.py --net MIBMINet --exp_id 4 --ckpt_id 39 --validate_fq --validate_tq --num_inputs_exported 100 --export_dir export_to_folder_name
```
The exported files can be found at: `quantlab-cl/examples/fx_integerization/export_to_folder_name`. Note that only quantized experiments can be integerized and exported in this way.

Afterwards, the exported files can be converted to deployable codes using DORY. Please refer to `BMI-ODCL/Offline-Training/README.md` for more details.

For additional information on the command line flags, run
```
(pytorch-1.13) $ python integerize_pactnets.py --help
```
The script should run without issues in the Conda environment of QuantLab.