# RealTimeActionRecongtionTools
Real-time and Stable Skeleton-based models and tools 

ReadMe will update.

1. ### HRNet weights in BaiduPans 

   links：https://pan.baidu.com/s/1kA5YnB6ufaDxGnYoOyvujg 
   codes：0my1
   
   put it in simple-HRNet/weights for demo
   put it in preprocess_data/simple-HRNet/weights for generate your own skeleton-based dataset.

2. ### Yolo weights in BaiduPans

   links: https://pan.baidu.com/s/1wWg1Zl35TaYKYNdXDkeaYw

   codes: d9vk

   put it in  simple-HRNet/models/detectors/yolo/weights
   put it in preprocess_data/simple-HRNet/models/detectors/yolo/weights for generate your own skeleton-based dataset.

### 3、generate your own skeleton-based dataset:

   1. python 1video_to_image.py 
      to get frames from videos
   2. split dun frames into 5 parts 
      python split5.py
    3. cd simple-HRNet, 
        1. **python process_by_hrnet1-5** to generate json files from frames using hrnet(parallel). 
        2. **python generate_label**, 
           **python kinetic_gendata.py** to generate kinetic format custom dataset

### 4、training MSG3D and get pytorch model
   code here:https://github.com/kenziyuliu/MS-G3D

### 5、convert torch model to onnx model using python torch2onnx.py

### 6、using msg3d onnx models in Demo/onnx_models/ to classify

### 7、run demo2D.sh and result in Demo/result



