# SmokeSeger: A Transformer-CNN coupled model for urban scene smoke segmentation

+ Implementation of `SmokeSeger: A Transformer-CNN coupled model for urban scene smoke segmentation` based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).

+ Also using [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) as the codebase.



# Descriptions

## Requirements

+ PaddleSeg 2.4.2
  + Please refer to the guidelines in [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).
  

## Configurations

+ Training and evaluation settings are defined in a yaml file (refer to [pre_config](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/config/pre_config.md)).
+ The yaml file of SmokeSeger is in `local_data/yml/smoke_seger.yml`.



# USS Dataset

+ An urban-scene smoke segmentation dataset. Some image examples:

<img src="local_data\pics\fig_examples_of_the_uss_dataset.jpg" style="zoom:70%" />

+ Descriptions
  + Images and masks are packaged into a h5 file.
  + Dataset split by train.list and valid.list.
  + Put the h5 file and the list files in the `local_data` under the project directory.
+ Downloads.
  + URL: [Baidu Disk](https://pan.baidu.com/s/1ffXN0KRt0qvcS_Ela3a6lA)
  + Code: sspm



# Evaluation and Training 

## Evaluation

+ Download dataset files and put into the `local_data`.
+ Download `trained weights` ([Baidu](https://pan.baidu.com/s/1--vmsy9qapiE8GptW3zuNA), Code: sspm).
+ In val.py

```Python
# Input the trained model dir
model_dir = r'Trained model file dir'
# Input the trained model name
model_path = os.path.join(model_dir, 'Trained model file name')
```

+ Run val.py

## Training

+ Download dataset files and put into the `local_data`.
+ In train.py

```Python
# Define the yml file used for training
yml_path = os.path.join('local_data', 'yml', 'smoke_seger.yml')
```

+ Run train.py



# Discussions

+ To be added.



# Acknowledgement

Thanks to [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [SegFormer](https://github.com/NVlabs/SegFormer) and [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) for the help.



# Citation

```
@article{smoke_seger,  
    author={Tao, Jing and Meng, Qing-Hao and Hou, Huirang},  
    journal={IEEE Transactions on Industrial Informatics},   
    title={SmokeSeger: A Transformer-CNN coupled model for urban scene smoke segmentation},   
    year={2023},  
    volume={},  
    number={},  
    pages={1-1},  
    doi={}
}
```

