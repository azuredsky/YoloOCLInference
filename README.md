# YoloOCLInference

A very light weight inference engine of 'tiny-yolo' object detection system for OpenCL based graphics cards/devices. For original work, ideas and implementation, refer to main author's page [here](https://pjreddie.com/darknet/)

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
:![alt text](https://github.com/sat8/YoloOCLInference/blob/master/frame_000006.jpg) **Example output with display enabled**:|:
![alt text](https://github.com/sat8/YoloOCLInference/blob/master/Capture.PNG) **Command line output with display disabled**:

## Background
This project has been deveoped to adapt 'tiny-YOlO' object detection system for OpenCL enabled hardware and extend the inference engine to support FP16 and INT8 versions of target models in the future. The original implementation of YOLO & its variants specifically target NVIDIA cards and there have been many efforts to [make YOLO a part of well known deep neural network frameworks](https://www.google.co.uk/search?rlz=1C1CHBD_en-GBGB743GB743&q=tensorflow+yolo&spell=1&sa=X&ved=0ahUKEwip3fKSkN_WAhXmBsAKHX7oB-EQvwUIJSgA&biw=1920&bih=974) so that OpenCL enabled devices can be supported. The [original implementation](https://github.com/pjreddie/darknet) is an excellent reference point for someone to get started but both training and inference are inter-twined there by creating some room for optimization in inference logic, especially for 'tiny-yolo' variant. Additionally, support for tiny-YOLO in OpenCL was also something many folks would be interested in and hence YoloOCLInference is mainly written to detach the inference engine from the training logic and thereby making the engine simple, more focused towards performance & optimization.  

## Key features
1. Achieves run time speeds of **~208 frames per second** on GTX 1080 Ti & **~73** frames per second on AMD RX580.
2. Implemented entirely in C++. No depenceny on exsting DNN frameworks.
3. Optimizations include 
  * Use of Swap buffers in convolutional layers to clear outputs in activation stages.
  * Addition of two im2col kernels for 3x3 & 1x1 variants
  * Folding batch normalization into convolutional layer output calculations. See http://machinethink.net/blog/object-detection-with-yolo/ 
  * Far less branched calculations in kernels comapred to original CUDA version
  * Linear activation has been altogether disabled. Output stays untouched once bias is added to GEMM output.
  * Loop unrolling where ever applicable. 
4. Uses [cairo graphics](https://wiki.gnome.org/Projects/gtkmm/MSWindows) to render overlay of text
5. Far less memory footprint compared to original CUDA version. 

## Dependencies
1. VC++ 2015
2. OpenCV 2.4.9
3. Gtkmm
4. NVIDIA Computing tool kit v8.0 (OpenCL libraries references)
5. [CLBLast](https://github.com/CNugteren/CLBlast) - Excellent BLAS library for OpenCL enabled hardware.

### Folder structure:

      - YoloOCLInference\
        -3rdparty\
            -cairo\
            -CLBlast\
            -opencv\
        -data\
            -coco.names
            -pedestrians.jpg
            -tiny-yolo.cfg
            -tiny-yolo.weights
        -project
            -YoloOCLInference.sln
            ...
        -src
            -Kernels\
                -DeepNNFP32.cl
                -DeepNNFP16.cl
          -OpenCL
                -OCLManager.cpp
                ...
          -Utilities
                ....
          -VideoProcessing
                -YoloOCLDNN.cpp
          -YoloOCLMain.cpp
           
The **3rdparty** folder contains readily usable headers & libraries hatare required by the project to compile while the **data** folder contains tiny-yolo model configuration, weights & labels. The rest of the folders are self-explanatory with relvant solution files and source code. Upon successful compilation (either in Release or Debug mode), **build** directory will be created and all the necessary dependencies and data will be copied into the directory. It must be noted that **DeepNNFP32.cl** will need to be deleted in the **build** directory if you are hand-optimizing the kernel code. Once you are done with the changes to kernel code, build the project which will put the updated kernel code so the executable will use it from then on.

## Usage
  Open command prompt & browse to the build directory
  Type 
  
  > YoloOCLInference.exe -input <image> -display <0/1> - save <0/1>
    
 eg:  > YoloOCLInference.exe -input pedestrians.jpg -display 1 -save 0
 The above command will make the executable load 'pedestrians.jpg' image available in build directory and run inference on it. Here, the output display is enabled with option **-display 1**, so an OpenCV window will display the detected output as below. Notice that **-save** option is set to 0, meaning the detected output frames will not be saved to fisk. In case the **-save** option is set to 1, the frames with detections will be written to disk in a folder named **output** within build directory.
 
If you want to let the engine free-flow without any display or saving options, the benchmarks reported here can be reproduced. 
Command: The relvant command in this case is
> YoloOCLInference.exe -input pedestrians.jpg -display 0 -save 0
 
 ![alt text](https://github.com/sat8/YoloOCLInference/blob/master/Capture.PNG)

## Limitations

## Future work

## Acknowledgements






