## Multi-modal Speech Recognition for ABCS Corpus

 This respository is the official implementation of "[End-to-End Multi-Modal Speech Recognition on an Air and Bone Conducted Speech Corpus](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9961873)" for TASLP 2023. 



### Installation

1. If you just need the module only, run

   ```
   pip install espnet
   ```

   first, and you can use the modules in `model` package.

2. If you want to do full experiments, you need to correctly install espnet and kaldi first. See [Installation](https://espnet.github.io/espnet/installation.html).

   Next, run

   ```
   pip install -r requirements.txt
   ```

   to install the required packages.



### Data Preparation

1. Download dataset.

   Download ABCS Corpus here: [Links](https://github.com/wangmou21/abcs?tab=readme-ov-file).

   Download noisy air conducted data (ns_air_data.zip) here: [[Onedrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/jqchen_mail_nwpu_edu_cn/EZShUMFoYnlPo-WwZ2DXZbgB-3gmW7sUoRLdCR7Z4vuF1A?e=zuH0L4)] or [Baidu Cloud]

   Unzip the noisy data to ABCS's directory:

   ```
   unzip -d <ABCS dir>/Audio/ ns_air_data.zip
   ```

2. Run data preparation code.

   For inference only:

   ```
   python3 data_prep --dataset_root <ABCS dir> --test
   ```

   For full experiments:

   ```
   python3 data_prep --dataset_root <ABCS dir>
   ```

   

### Inference

1. Make sure you have correctly installed kaldi and espnet.  Then modify the 3 line in `test.sh`:

    ```
    export ESPNETROOT=<Your Espnet Root>
    ```

2. Download model parameters file here [[Onedrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/jqchen_mail_nwpu_edu_cn/EWmrzNQfDf1FmZ7JIyx9xqsBaTOcgVR8Zou2x3hAvqWH2g)] or [Baidu Cloud]

3. Run

    ```
    bash test.sh
    ```



### TODO

The training pipeline.



### Citing

If you found this code useful, please cite this:

```
@ARTICLE{9961873,
  author={Wang, Mou and Chen, Junqi and Zhang, Xiao-Lei and Rahardja, Susanto},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={End-to-End Multi-Modal Speech Recognition on an Air and Bone Conducted Speech Corpus}, 
  year={2023},
  volume={31},
  number={},
  pages={513-524},
  keywords={Speech recognition;Speech processing;Signal to noise ratio;Spectrogram;Headphones;Microphones;Synchronization;Speech recognition;multi-modal speech processing;bone conduction;air- and bone-conducted speech corpus},
  doi={10.1109/TASLP.2022.3224305}}
```



