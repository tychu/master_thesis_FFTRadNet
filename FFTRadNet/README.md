# Raw High-Definition Radar for Multi-Task Learning

## Paper
![](./FFT-RadNetArchi.png)

[Raw High-Definition Radar for Multi-Task Learning](https://arxiv.org/pdf/2112.10646)  
 [Julien Rebut](https://www.linkedin.com/in/julien-rebut-9803887b),  [Arthur Ouaknine](https://arthurouaknine.github.io/), [Waqas Malik](https://www.linkedin.com/in/waqas-malik-2070012b/), [Patrick Pérez](https://ptrckprz.github.io/)  
valeo.ai, France  

## Abstract
With their robustness to adverse weather conditions and ability to measure speeds, radar sensors have been part of the automotive landscape for more than two decades. Recent progress toward High Definition (HD) Imaging radar has driven the angular resolution below the degree, thus approaching laser scanning performance. However, the amount of data a HD radar delivers and the computational cost to estimate the angular positions remain a challenge. In this paper, we propose a novel HD radar sensing model, FFT-RadNet, that eliminates the overhead of computing the Range-Azimuth-Doppler 3D tensor, learning instead to recover angles from a Range-Doppler spectrum. FFT-RadNet is trained both to detect vehicles and to segment free driving space. On both tasks, it competes with the most recent radar-based models while requiring less compute and memory. Also, we collected and annotated 2-hour worth of raw data from synchronized automotive-grade sensors (camera, laser, HD radar) in various environments (city street, highway, countryside road). This unique dataset, nick-named RADIal for "Radar, Lidar et al.", is available at this https URL.

## Demo
[![](./FFTRadNet.png)](https://www.youtube.com/watch?v=t9WNLUiWDFE "")


### Datasets
Download the RADIal dataset as explained [here](https://github.com/valeoai/RADIal)

### Pre-trained models
Pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1qh_ixfiDRUAiXg0d0SIBv0wj2L_DxTIS?usp=share_link)

### Config file
The config files provided include normalization constants for each of the differing input types. To obtain your own normalization constants you can run the following command within the dataset folder:

`$ python print_dataset_statistics.py`

### Training
For training, execute:

- paper's config: for MATLAB simulation dataset (random targets), 3 targets
```
$ python 1-Train.py --config config/config_FFTRadNet_matlab.json
```
- paper's config: for MATLAB simulation dataset (sequential targets), 3 targets
```
$ python 1-Train.py --config config/config_FFTRadNet_matlab_seq.json
```
- paper's config: for MATLAB simulation dataset (sequential targets and limit range values), 3 targets
```
$ python 1-Train.py --config config/config_FFTRadNet_matlab_seqtrace.json
```
- paper's config: for MATLAB simulation dataset (random targets), 1 targets
```
$ python 1-Train.py --config config/config_FFTRadNet_matlab_onetarget.json
```
- paper's config: for RadIal dataset
```
$ python 1-Train.py --config config/config_FFTRadNet_192_56.json
```
In each case, training can be resumed via the command:

```
$ python Train.py --config /path/to/config.json --resume /path/to/previous_model.pth
```

### Testing
For testing, execute:
```
$ python 2-Test.py --config config/config_FFTRadNet_matlab_seq.json --checkpoint /path/to/my_experiment/model.pth
```

### Performance evaluation
To evaluate performances, run the following script:
```
$ python 3-Evaluation.py --config config/config_FFTRadNet_matlab_seq.json --checkpoint /path/to/my_experiment/model.pth
```
path to .pth file example: FFTRadNet_matlab_Jul-05-2024_16rx_3targets_shuffle_seqdata/FFTRadNet_matlab_epoch499_loss_12.1030_AP_0.0000_AR_0.0000.pth

### Additional code
- **4-Data_checking.py :** check RadIal input files (all the input .npy files are identical)
- **5-Model_result.py :** check model training and validation loss and plot the loss for every epoch (compute validation loss without eval() mode )
PLot the loss for all checkpoint file's epoch in the folder (/path/to/my_experiment/)
```
$ python 5-Model_result.py --config config/config_FFTRadNet_matlab_seq.json --checkpointdir /path/to/my_experiment/
```
PLot the loss from epoch 0 to the checkpoint file's epoch
```
$ python 5-Model_result.py --config config/config_FFTRadNet_matlab_seq.json --evalmmode /path/to/my_experiment/model.pth
```

## License
FFTRadNet is released under the [Apache 2.0 license](./LICENSE).

## checkpoint files
### Using random targets dataset
|                Parameters             | checkpoint (file path /imec/other/dl4ms/chu06/RADIal/) |  Notes |
|---------------------------------------|---------------|-----------------|
| rx=16, data size=500, # targets=1     | FFTRadNet_matlab_Jun-12-2024_16rx_detection_regression/         |  Not shuffle training dataset          |
| rx=2, data size=500, # targets=1      | FFTRadNet_matlab_Jun-12-2024_2rx_detection_regression/         |  Not shuffle training dataset          |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jun-17-2024_16rx_detection_regression_3targets/         |  Not shuffle training dataset          |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jun-18-2024_detreg_3targets_not-runningstat/         |   "BatchNormlayer, running_stat=False" ; Not shuffle training dataset       |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jun-19-2024_shuffledata/         |  Shuffle training dataset          |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jun-20-2024_detreg_3tragets_momentum0.5/         |  Shuffle training dataset; Momentum=0.5 (didn't finish it since the result looks not good)          |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jun-21-2024_16rx_detreg_3targets_batch8/         |  Shuffle training dataset; Batch size=8 (original batch size=4)          |
| rx=2, data size=1000, # targets=3     | FFTRadNet_matlab_Jun-27-2024_2rx_detreg_3targets_batch16/         |  Shuffle training dataset; Batch size=16 (original batch size=4) 1000 samples          |
| rx=2, data size=1000, # targets=3     | FFTRadNet_matlab___Jun-27-2024___23:01:23/         |  Shuffle training dataset; Batch size=8 (original batch size=4) 1000 samples          |

#### Using sequential targets dataset
|                Parameters             | checkpoint (file path /imec/other/dl4ms/chu06/RADIal/) |  Notes |
|---------------------------------------|---------------|-----------------|
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab_Jul-05-2024_16rx_3targets_shuffle_seqdata/         |  Shuffle training dataset        |
| rx=16, data size=500, # targets=3     | FFTRadNet_matlab___Jul-06-2024___08:31:59/         |    Shuffle training dataset; Limit range:0-100        |