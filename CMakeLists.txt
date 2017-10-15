
# ==================================================================================================
# This file is part of the YoloOCLInference project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
# width of 100 characters per line.
#
# Author(s):
#   Sateesh Pedagadi
#
# ==================================================================================================

cmake_minimum_required(VERSION 2.8.11)

# CMake project details
project("YoloOCLInference" C CXX)
set(YoloOCLInference_VERSION_MAJOR 0)
set(YoloOCLInference_VERSION_MINOR 1)
set(YoloOCLInference_VERSION_PATCH 0)

# Compiler-version check (requires at least CMake 2.8.10)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif(MSVC)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ot")
  set(FLAGS "${FLAGS} /wd4715 /D_CRT_SECURE_NO_WARNINGS")
else()
  set(FLAGS "-std=gnu++11")
  if(VERBOSE)
    set(FLAGS "${FLAGS} -O1 -g")
  else()
    set(FLAGS "${FLAGS} -O2")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(FLAGS "${FLAGS} -Wall -Wno-comment -Wno-return-type -Wno-switch -Wno-missing-noreturn")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
      set(FLAGS "${FLAGS} -Wno-attributes -Wno-unused-variable")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0.0)
      # GCC does not support attributes on template arguments
      # in particular we hit this with the alignment attributes on cl_XXX types
      # which are then used to instantiate various templates in CLBlast
      set(FLAGS "${FLAGS} -Wno-ignored-attributes")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
    set(FLAGS "${FLAGS} -Wno-missing-prototypes -Wno-float-equal -Wno-switch-enum -Wno-switch")
    set(FLAGS "${FLAGS} -Wno-exit-time-destructors -Wno-global-constructors -Wno-missing-noreturn")
    set(FLAGS "${FLAGS} -Wno-deprecated-declarations")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.9.0)  # clang 4.0 or higher
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.0)  # but not for AppleClang
          set(FLAGS "${FLAGS} -Wno-undefined-var-template")
        endif()
    endif()
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

# C compiler settings (for the sample)
if(MSVC)
  set(CFLAGS "/Ot")
else()
  set(CFLAGS "-O2 -std=c99")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CFLAGS}")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${YoloOCLInference_SOURCE_DIR}/cmake/Modules/")

# Requires OpenCL. It is found through the included "FindOpenCL.cmake" in CMAKE_MODULE_PATH.
find_package(OpenCL REQUIRED)

if(!MSVC)
	find_package(OpenCV REQUIRED)
	message(STATUS "OpenCV library status:")
	message(STATUS "    version: ${OpenCV_VERSION}")
	message(STATUS "    libraries: ${OpenCV_LIBS}")
	message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif()

# ==================================================================================================

# Gathers all source-files (required for the compiler) and header-files (for IDEs only)
set(SOURCES
  src/YoloOCLMain.cpp
  src/VideoProcessing/YoloOCLDNN.cpp
  src/OpenCL/buffer.cpp
  src/OpenCL/cl_wrapper.cpp
  src/OpenCL/clutils.cpp
  src/OpenCL/helpers.cpp
  src/OpenCL/iv_common.cpp
  src/OpenCL/kernel_launcher.cpp
  src/OpenCL/OCLManager.cpp
  src/OpenCL/oclUtils.cpp
  src/OpenCL/program.cpp
  src/OpenCL/shrUtils.cpp
)
set(HEADERS  # such that they can be discovered by IDEs such as CLion and Visual Studio
  src/VideoProcessing/include/YoloOCLDNN.h
  src/Utilities/include/GlobalDefines.h
  src/Utilities/include/SimpleIni.h
  src/Utilities/include/ConvertUTF.h
  src/OpenCL/include/cl_wrapper.hpp
  src/OpenCL/include/buffer.h
  src/OpenCL/include/clutils.h
  src/OpenCL/include/helpers.h
  src/OpenCL/include/iv_common.h
  src/OpenCL/include/kernel_launcher.h
  src/OpenCL/include/OCLManager.h
  src/OpenCL/include/oclUtils.h
  src/OpenCL/include/program.h
)

if(MSVC)
	include_directories(3rdparty/opencv/include/ 3rdparty/clblast/include/ 3rdparty/cairo/include/ src/OpenCL/include/ src/Utilities/include/ src/VideoProcessing/include/ ${OpenCL_INCLUDE_DIRS})
	link_directories(3rdparty/opencv/lib/ 3rdparty/clblast/Release 3rdparty/cairo/lib/)
else()
	include_directories(/usr/include/cairo src/OpenCL/include/ src/Utilities/include/ src/VideoProcessing/include/ ${OPENCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
endif()

# Creates and links the library
add_executable(YoloOCLInference ${SOURCES} ${HEADERS})

if(MSVC)
	target_link_libraries(YoloOCLInference ${OpenCL_LIBRARIES} clblast opencv_core330 opencv_highgui330 opencv_imgcodecs330 opencv_imgproc330 cairo)
else()
	target_link_libraries(YoloOCLInference opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs ${OPENCL_LIBRARIES} clblast cairo )
endif()

add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:YoloOCLInference>)
add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/src/Kernels $<TARGET_FILE_DIR:YoloOCLInference>)

if(MSVC)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/clblast/Release/clblast.dll $<TARGET_FILE_DIR:YoloOCLInference>)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/opencv/bin/opencv_core330.dll $<TARGET_FILE_DIR:YoloOCLInference>)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/opencv/bin/opencv_highgui330.dll $<TARGET_FILE_DIR:YoloOCLInference>)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/opencv/bin/opencv_imgproc330.dll $<TARGET_FILE_DIR:YoloOCLInference>)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/opencv/bin/opencv_imgcodecs330.dll $<TARGET_FILE_DIR:YoloOCLInference>)
	add_custom_command(TARGET YoloOCLInference PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/opencv/bin/opencv_videoio330.dll $<TARGET_FILE_DIR:YoloOCLInference>)
endif()