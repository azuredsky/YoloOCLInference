
/*Copyright 2017 Sateesh Pedagadi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/


#ifndef OCLWRAPPER_H_
#define OCLWRAPPER_H_

#include <chrono>
#include <algorithm>
//#include <windows.h>
#include <string>
#include <iostream>
#include "CL/cl.h"
#include "cl_wrapper.hpp"
#include "GlobalDefines.h"
#include "clblast.h"
//#include "clblast_half.h"

#define PINNED_MEM_OUTPUT

#define PROFILE_KERNELS		0
#define BLOCK_KERNEL_EXEC	0


using namespace std;;

string ExePath();


typedef struct {

	void			*m_PinnedMemory;
	OCLBuffer		*m_OCLBuffer;

}StructPinnedOCLBuffer;

float sec(clock_t clocks);

static const char* NN_KERNEL_NAMES[NN_MAX_KERNEL_COUNT] = {

	"image2columarray3x3",
	"image2columarray1x1",
	"normalizearray",
	"scalebias",
	"addbias",
	"scaleaddbias",
	"normscaleaddbias",
	"leakyactivatearray",
	"linearactivatearray",
	"flattenarray",
	"softmax",
	"maxpool",
	"resetarray"
};

class OCLManager {

public:

	OCLManager();
	~OCLManager();
	int Initialize();
	int Finalize();
	void ReleaseLock();
	void SetLock();

	void TestCLBlastLib();
	void TestCLBlastLibHalfFloat();


	OCLBuffer* InitializeFloatArray(float *inArray, size_t numItems);
	OCLBuffer* InitializeIntArray(size_t numItems);
	void FinalizeFloatArray(OCLBuffer *oclBuffer);
	void FinalizeIntArray(OCLBuffer *oclBuffer);
	void WriteFloatArray(OCLBuffer *oclBuffer, float *cpuArray, size_t numItems);
	void WriteIntArray(OCLBuffer *oclBuffer, int *cpuArray, size_t numItems);
	void ReadFloatArray(float *cpuArray, OCLBuffer *oclBuffer, size_t numItems);
	void ReadIntArray(int *cpuArray, OCLBuffer *oclBuffer, size_t numItems);
	float ConvertImageToColumnArray(OCLBuffer *im, int channels, int height, int width,
		int ksize, int stride, int pad, OCLBuffer *data_col);
	float ComputeGEMM(bool isATransponsed, bool isBTransposed, const size_t m, const size_t n, const size_t k,
		const float alpha, OCLBuffer *bufferA, const size_t offsetA, const size_t ldA, OCLBuffer *bufferB, const size_t offsetB, const size_t ldB,
		const float beta, OCLBuffer *bufferC, const size_t offsetC, const size_t ldC);
	float BatchNorm(OCLBuffer *x, OCLBuffer *mean, OCLBuffer *variance, int batch, int filters, int spatial);
	float ScaleBias(OCLBuffer *output, OCLBuffer *biases, int batch, int n, int size);
	float AddBias(OCLBuffer *output, OCLBuffer *biases, int batch, int n, int size);// , int activationType);
	float ScaleAddBias(OCLBuffer *output, OCLBuffer *scales, OCLBuffer *biases, int batch, int n, int size);
	float NormScaleAddBias(OCLBuffer *output, OCLBuffer *scales, OCLBuffer *biases, OCLBuffer *mean, OCLBuffer *variance, int filters, int spatial);
	float ActivateInputs(OCLBuffer *x, OCLBuffer *swapBuffer, int n, int actType);
	float FlattenArray(OCLBuffer *x, int spatial, int layers, int batch, int forward, OCLBuffer *out);
	float SoftMax(OCLBuffer *input, int n, int offset, int groups, float temp, OCLBuffer *output, int base);
	float MaxPool(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, OCLBuffer *input, OCLBuffer *output); // , OCLBuffer *indexes);
	//float ResetArray(int N, OCLBuffer * Y);
	float ResetArray(int N, OCLBuffer *inArray, OCLBuffer *biasArray, int filtSize);
	void  FillBuffer(OCLBuffer *inBuffer, int value, int numItems);

	StructPinnedOCLBuffer* InitializePinnedFloatArray(size_t numItems);
	void FinalizePinnedFloatArray(StructPinnedOCLBuffer*);

	const char *GetDeviceName() { return m_DeviceName; };

private:

	Program*			m_OpenCLProgram;
	void				*m_RefObject;
	int					m_Status;
	int					m_LockStatus;
	int					m_CallerId;
	//HANDLE				m_LockMutex;
	CLSetup				m_OpenCLSetup;
	KernelLauncher*		m_OpenCLKernels[NN_MAX_KERNEL_COUNT];
	char				m_DeviceName[256];
};



#endif /* OCLWRAPPER_H_ */

