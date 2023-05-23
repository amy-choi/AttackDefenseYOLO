# [Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios (IV 2022)](https://arxiv.org/abs/2202.04781)

This repository contains the adversarial attack/defense implementation for the paper:
> Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios  
> Jung Im Choi, Qing Tian   
> Bowling Green State University  
> IV 2022

<p align="center">
<img src="resources/overview.PNG" height = "50%" width = "60%"">
</p>

                                                            
## Model Training 
**1. Convert your datasets to VOC format for training.**                                                                      
   * Put the label file in the Annotation under the data folder.                                                                   
   * Put the picture file in JPEGImages under the data folder.      
                                                               
**2. Create .txt file by using voc_annotation.py for training.**                                                                    
   * Create a your_classes.txt under the model_data folder and write the categories you need to classifiy in it.      
   * Modify the class_path in kitti_annotation.py to model_data/your_cls_classes.txt.        
                                                               
**3. Modify the classes_path in adv_training.py and run it to start adversarial training.**                                                              

## Citation
If you find it helpful in your research, please consider citeing our paper: 

```
@InProceedings{choi2022advYOLO,
  title = {Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios},
  author = {Choi, Jung Im and Tian, Qing},
  booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)},
  year = {2022},
  pages = {1011-1017},
  doi={10.1109/IV51971.2022.9827222},
}
```

## References
Any pretrained weights and other codes needed for YOLOv4 can be founded on [link](https://github.com/bubbliiiing/yolov4-pytorch).

## Contact
If you have any questions or suggestions, feel free to contact us. (<a>choij@bgsu.edu</a>) 
