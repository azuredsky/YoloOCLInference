# YOLO OpenCL Inference Engine (Linux & Windows)

A very light weight inference engine of **tiny-YOLO** object detection system for OpenCL based graphics cards/devices operating in Linux and Windows environments.

Example output with display enabled|
:-------------------------:|
:![](https://github.com/sat8/YoloOCLInference/blob/master/frame_000006.jpg)|


## Background
This project has been deveoped to adapt **tiny-YOLO** object detection system for OpenCL enabled hardware and extend the inference engine to support FP16 and INT8 versions of target models in future. The [original implementation based on CUDA](https://github.com/pjreddie/darknet) is an excellent reference point for someone to get started but both training and inference are tightly coupled there by leaving some room for simplifying inference engine, especially the 'tiny-YOLO' variant. Additionally, support for tiny-YOLO in OpenCL would also be something many folks would be interested in and hence YoloOCLInference is written to detach the inference engine from the training logic and thereby making the engine simple, more focused & optimized towards performance.  

## Key features
1. Achieves run time speeds of **~227 Fps** on GTX 1080 Ti in Linux & **~73 Fps** on AMD RX580.

Inference speed - Display disabled|
:-------------------------:|
:![](https://github.com/sat8/YoloOCLInference/blob/master/Capture.jpg)|

2. Implemented entirely in C++. No dependency on existing DNN frameworks.
3. Optimizations include 
   * Use of Swap buffers in convolutional layers to clear outputs in activation stages.
   * Addition of two im2col kernels, one each for 3x3 & 1x1 variants
   * Folding batch normalization into convolutional layer output calculations. See http://machinethink.net/blog/object-detection-with-yolo/ 
   * Branched calculations are optimized with fixed number of instructions.
   * Linear activation has been altogether disabled. Output stays untouched once bias is added to GEMM output.
   * Loop unrolling where ever applicable. 
4. Uses [cairo graphics](https://wiki.gnome.org/Projects/gtkmm/MSWindows) to render overlay of text
5. Far less memory footprint compared to original CUDA version. 

## Dependencies
1. CMake 3.8.11 (May work with older versions)
2. VC++ 2015 (Windows only)
3. OpenCV 3.3.0
4. Gtkmm (Windows only). Download installer from [here](http://ftp.gnome.org/pub/GNOME/binaries/win64/gtkmm/2.22/gtkmm-win64-devel-2.22.0-2.exe)
5. NVIDIA Computing tool kit v9.0 (OpenCL library references)
6. [CLBLast](https://github.com/CNugteren/CLBlast) - Excellent BLAS library for OpenCL enabled hardware.

## Compilation in Ubuntu Linux

Ensure that OpenCV is built with GTK+ 2.x and C++11 support.

	mkdir build
	cd build
	cmake .. 
	make -j 4

## Compilation in Windows

	mkdir build
	cd build
	cmake .. -G "Visual Studio 14 2015 Win64"
	cmake --build . --target ALL_BUILD --config Release

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
* Only single image input is supported. Neither batch input nor video file input is possible.
* The application runs inference on the same image in a sequence for 1000 iterations. There are no control settings available.
* Object labels/classes are not displayed near the detections overlayed

## Future work
* Support Video file and folder/batch input of images
* Support storing output video to disk
* Build a reusable API that supports RAW image input and file input (both video & image). The RAW image input is expected to be very useful in feeding hardware accelerated decoder output from either the same GPU (NVIDIA, AMD, ARM chips) or host CPU (Intel Quick Sync).

## Acknowledgements
Thanks to the authors of following repos.
* https://github.com/pjreddie/darknet
* https://github.com/CNugteren/CLBlast
* https://github.com/AlexeyAB/darknet






