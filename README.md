# YOLO OpenCL Inference Engine

A very light weight inference engine of **tiny-YOLO** object detection system for OpenCL based graphics cards/devices. For original work and CUDA implementation, refer to main author's page [here](https://pjreddie.com/darknet/).

Example output with display enabled|
:-------------------------:|
:![](https://github.com/sat8/YoloOCLInference/blob/master/frame_000006.jpg)|


## Background
This project has been deveoped to adapt **tiny-YOLO** object detection system for OpenCL enabled hardware and extend the inference engine to support FP16 and INT8 versions of target models in the future. The original implementation of YOLO & its variants specifically target NVIDIA cards and there have been many efforts to [make YOLO a part of well known deep neural network frameworks](https://www.google.co.uk/search?rlz=1C1CHBD_en-GBGB743GB743&q=tensorflow+yolo&spell=1&sa=X&ved=0ahUKEwip3fKSkN_WAhXmBsAKHX7oB-EQvwUIJSgA&biw=1920&bih=974) so that OpenCL enabled devices can be supported. The [original implementation](https://github.com/pjreddie/darknet) is an excellent reference point for someone to get started but both training and inference are tightly coupled there by leaving some room for simplifying inference engine, especially for 'tiny-yolo' variant. Additionally, support for tiny-YOLO in OpenCL would also be something many folks would be interested in and hence YoloOCLInference is written to detach the inference engine from the training logic and thereby making the engine simple, more focused towards performance & optimization.  

## Key features
1. Achieves run time speeds of **~208 frames per second** on GTX 1080 Ti & **~73** frames per second on AMD RX580.

Inference speed - Display disabled|
:-------------------------:|
:![](https://github.com/sat8/YoloOCLInference/blob/master/Capture.jpg)|

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
           
The **3rdparty** folder contains readily usable headers & libraries that are required by the project to compile while the **data** folder contains tiny-yolo model configuration, weights & labels. The rest of the folders are self-explanatory with relvant solution files and source code. Upon successful compilation (either in Release or Debug mode), **build** directory will be created and all the necessary dependencies and data will be copied into the directory. It must be noted that **DeepNNFP32.cl** will need to be deleted in the **build** directory if you are hand-optimizing the kernel code. Once you are done with the changes to kernel code, building the project will copy  the updated kernel code so the executable will be using it from then on during runtime.

## Usage
  Open command prompt & browse to the build directory
  Type 
  
 > YoloOCLInference.exe -input <image> -display <0/1> - save <0/1>
    
 eg:  
 > YoloOCLInference.exe -input pedestrians.jpg -display 1 -save 0
 
The above command will make the executable load 'pedestrians.jpg' image available in build directory and runs inference on it. Here, the output display is enabled with option **-display 1**, so an OpenCV window will display the detected output as above. In case the **-save** option is set to 1, the frames with detections will be written to disk in a folder named **output** within build directory. If **-save** option is set to 0, the detection output frames with overlay will not be saved to disk. 

Supposing display option is enabled, the OpenCV output window will pause for user input before proceeding onto running inference in next iteration.
 
If you prefer to let the engine free-flow without any display or saving options, the benchmarks reported here can be reproduced. 
The relvant command in this case would be 

> YoloOCLInference.exe -input pedestrians.jpg -display 0 -save 0
 
## Limitations

Following are some of the limitations in YoloOCLInference application
* Presently, only windows environment is supported. 
* Only single image input is supported. Neither batch input nor video file input is possible.
* The application runs inference on the same image in a sequence for 1000 iterations. There are no control settings available.
* Object labels/classes are not displayed near the detections overlayed
* Sometimes, the application crashes on termination after completing execution and finally printing the inference statistics (time & speed). 
* Sometimes, there is a scaling/shift error on the bounding boxes overlaid around objects.

## Future work
* Support cross compilation in Linux & Windows using CMake
* Support Video file and folder/batch input of images
* Support storing output video to disk
* Build a reusable API that supports RAW image input and file input (both video & image). The RAW image input is expected to be very useful in feeding hardware accelerated decoder output from either the same GPU (NVIDIA, AMD, ARM chips) or host CPU (Intel Quick Sync).

## Acknowledgements
Thanks to the authors of following repos.
* https://github.com/pjreddie/darknet
* https://github.com/CNugteren/CLBlast
* https://github.com/AlexeyAB/darknet






