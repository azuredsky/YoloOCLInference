
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

#include "OCLManager.h"

#define BLOCK 8

#ifdef WIN32

#include <Windows.h>
string ExePath() {

	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	string::size_type pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}

#elif __linux__

string ExePath() {

	return "";
}

#endif

float sec(clock_t clocks) {

	return (float)clocks / CLOCKS_PER_SEC;
}

typedef struct {

	int m_NumX;
	int m_NumY;
	int m_NumZ;
}StructOCLDims;

OCLManager::OCLManager() {

	m_OpenCLProgram = NULL;

	for(int i = 0; i < NN_MAX_KERNEL_COUNT; i++)
		m_OpenCLKernels[i] = 0;

	//m_LockMutex = NULL;
	//cl_int err = clblasSetup();

	m_Status = OCL_STATUS_INITIALIZED;
}

OCLManager::~OCLManager() {

	//clblasTeardown();
	for( int i = 0; i < NN_MAX_KERNEL_COUNT; i++ )
		delete m_OpenCLKernels[i];

	delete m_OpenCLProgram;

	
	m_Status = OCL_STATUS_FINALIZED;
}

int OCLManager::Initialize() {

	std::string file;
	
#ifdef WIN32
	file = ExePath() + "\\DeepNNFP32.cl";
#elif __linux__
	file = "DeepNNFP32.cl";
#endif

	std::vector<std::string> kernelFiles;
	kernelFiles.push_back(file);

	m_OpenCLSetup.init(m_DeviceName);
	m_OpenCLProgram = m_OpenCLSetup.createProgram(kernelFiles);
	m_OpenCLProgram->buildProgram();

	for( int i = 0; i < NN_MAX_KERNEL_COUNT; i++ )
		m_OpenCLKernels[i] = m_OpenCLProgram->createKernelLauncher(NN_KERNEL_NAMES[i]);

	//m_LockMutex = CreateMutex(NULL, FALSE, NULL);

	/*if( pthread_mutex_init(&m_LockMutex, NULL) != 0 ) {
	    printf("ERROR - OCLWrapper::Initialize() Mutex initialization error \n");
	    m_Status = OCL_STATUS_MUTEX_ERROR;
	    return m_Status;
	}*/


	m_LockStatus = OCL_LOCK_RELEASE;
	m_Status = OCL_STATUS_READY;
	return m_Status;
}

int OCLManager::Finalize() {

	//if(m_LockMutex != NULL)
	//if (m_Status != OCL_STATUS_MUTEX_ERROR) {
	
		//pthread_mutex_destroy(&m_LockMutex);
		//CloseHandle(m_LockMutex);
	//}

	m_Status = OCL_STATUS_FINALIZED;
	return m_Status;
}

void OCLManager::SetLock() {

	//pthread_mutex_lock(&m_LockMutex);
	//WaitForSingleObject(m_LockMutex, INFINITE);
}

void OCLManager::ReleaseLock() {

	//pthread_mutex_unlock(&m_LockMutex);
	//ReleaseMutex(m_LockMutex);
}

StructPinnedOCLBuffer* OCLManager::InitializePinnedFloatArray(size_t numItems) {

	StructPinnedOCLBuffer *pinnedMemBuffer = new StructPinnedOCLBuffer;

	size_t totalSize = sizeof(float) * numItems;
	pinnedMemBuffer->m_OCLBuffer = m_OpenCLSetup.createBuffer(totalSize, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NULL);
	pinnedMemBuffer->m_PinnedMemory = pinnedMemBuffer->m_OCLBuffer->map(CL_MAP_READ, totalSize, 0, CL_TRUE);

	return pinnedMemBuffer;
}

void OCLManager::FinalizePinnedFloatArray(StructPinnedOCLBuffer *oclPinnedBuffer) {

	oclPinnedBuffer->m_OCLBuffer->unmap(oclPinnedBuffer->m_PinnedMemory);
	delete oclPinnedBuffer;
}

OCLBuffer* OCLManager::InitializeFloatArray(float *inArray, size_t numItems) {

	size_t totalSize = sizeof(float) * numItems;
	OCLBuffer* outBuffer = m_OpenCLSetup.createBuffer(totalSize, CL_MEM_READ_WRITE, NULL);

	if (inArray != NULL)
		outBuffer->write(inArray, totalSize, 0, CL_TRUE);

	return outBuffer;
}

OCLBuffer* OCLManager::InitializeIntArray(size_t numItems) {

	size_t totalSize = sizeof(int) * numItems;
	OCLBuffer* outBuffer = m_OpenCLSetup.createBuffer(totalSize, CL_MEM_READ_WRITE, NULL);
	return outBuffer;
}

void OCLManager::FinalizeFloatArray(OCLBuffer *oclBuffer) {

	delete oclBuffer;
}

void OCLManager::FinalizeIntArray(OCLBuffer *oclBuffer) {

	delete oclBuffer;
}

void OCLManager::WriteFloatArray(OCLBuffer *oclBuffer, float *cpuArray, size_t numItems) {

	oclBuffer->write(cpuArray, numItems * sizeof(float), 0, CL_TRUE);
}

void  OCLManager::FillBuffer(OCLBuffer *inBuffer, int value, int numItems) {

	int inValue = value;
	size_t totalSize = sizeof(float) * numItems;
	inBuffer->FillBuffer(&inValue, totalSize, 0);
}

void OCLManager::WriteIntArray(OCLBuffer *oclBuffer, int *cpuArray, size_t numItems) {

	oclBuffer->write(cpuArray, numItems * sizeof(int), 0, CL_TRUE);
}

void OCLManager::ReadFloatArray(float *cpuArray, OCLBuffer *oclBuffer, size_t numItems) {

#ifndef PINNED_MEM_OUTPUT
	oclBuffer->read(cpuArray, numItems * sizeof(float), 0, CL_TRUE);
#else
	oclBuffer->read(cpuArray, numItems * sizeof(float), 0, CL_FALSE);
#endif
}

void OCLManager::ReadIntArray(int *cpuArray, OCLBuffer *oclBuffer, size_t numItems) {

	oclBuffer->read(cpuArray, numItems * sizeof(int), 0, CL_TRUE);
}

float OCLManager::ResetArray(int N, OCLBuffer *inArray, OCLBuffer *biasArray, int filtSize) {

	int globalDimX = N / BLOCK;
	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(0, inArray->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(1, biasArray);
	m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->arg(2, filtSize);
	return m_OpenCLKernels[NN_KERNEL_IDX_RESETARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::ConvertImageToColumnArray(OCLBuffer *im, int channels, int height, int width,
				int ksize, int stride, int pad, OCLBuffer *data_col) {

	int LOCAL_BLOCK = BLOCK;// / 2;

	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;

	int globalDimX = ((num_kernels + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;

	int KERNEL_IDX = (ksize == 3) ? NN_KERNEL_IDX_IM2COL3X3 : NN_KERNEL_IDX_IM2COL1X1;

	m_OpenCLKernels[KERNEL_IDX]->pGlobal(globalDimX)->pLocal(LOCAL_BLOCK);
	//m_OpenCLKernels[KERNEL_IDX]->pGlobal(globalDimX, filtSize)->pLocal(LOCAL_BLOCK, 1);
	m_OpenCLKernels[KERNEL_IDX]->arg(0, num_kernels);
	m_OpenCLKernels[KERNEL_IDX]->arg(1, im->getMem());
	m_OpenCLKernels[KERNEL_IDX]->arg(2, height);
	m_OpenCLKernels[KERNEL_IDX]->arg(3, width);
	m_OpenCLKernels[KERNEL_IDX]->arg(4, ksize);
	m_OpenCLKernels[KERNEL_IDX]->arg(5, pad);
	m_OpenCLKernels[KERNEL_IDX]->arg(6, stride);
	m_OpenCLKernels[KERNEL_IDX]->arg(7, height_col);
	m_OpenCLKernels[KERNEL_IDX]->arg(8, width_col);
	m_OpenCLKernels[KERNEL_IDX]->arg(9, data_col->getMem());
	return m_OpenCLKernels[KERNEL_IDX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::ComputeGEMM(bool isATransponsed, bool isBTransposed, const size_t m, const size_t n, const size_t k,
	const float alpha, OCLBuffer *bufferA, const size_t offsetA, const size_t ldA, OCLBuffer *bufferB, const size_t offsetB, const size_t ldB,
	const float beta, OCLBuffer *bufferC, const size_t offsetC, const size_t ldC) {

	//return 0.0f; 

	cl_event execEvent = NULL;
	double total = 0;

	clblast::StatusCode status = clblast::Gemm(clblast::Layout::kRowMajor,	clblast::Transpose::kNo, clblast::Transpose::kNo, m, n, k, alpha, bufferA->getMem(), 0, ldA,
		bufferB->getMem(), 0, ldB, beta, bufferC->getMem(), 0, ldC, m_OpenCLSetup.getQueue(), (PROFILE_KERNELS == 1) ? &execEvent : NULL);

	if (status == clblast::StatusCode::kSuccess) {

		if (PROFILE_KERNELS == 1) {

			clWaitForEvents(1, &execEvent);
			long long start, end;
			cl_int status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_START,
				sizeof(start), &start, NULL);
			status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_END,
				sizeof(end), &end, NULL);

			total = (double)(end - start) / 1e6;
			printf("Total kernel time was {%5.2f} msecs - ComputeGEMM() \n", total);
			
			clReleaseEvent(execEvent);
		}
	}

	return (float)total;
}

float OCLManager::BatchNorm(OCLBuffer *x, OCLBuffer *mean, OCLBuffer *variance, int batch, int filters, int spatial) {

	int N = batch * filters * spatial;

	int globalDimX = N / BLOCK;
	if(globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(0, N);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(1, x->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(2, mean->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(3, variance->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(4, batch);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(5, filters);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->arg(6, spatial);
	execTime += m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

	return execTime;
}

float OCLManager::ScaleBias(OCLBuffer *output, OCLBuffer *biases, int batch, int n, int size) {

	float execTime = 0.0f;
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->pGlobal(size, n)->pLocal(1, 1);
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->arg(0, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->arg(1, biases->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->arg(2, n);
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->arg(3, size);
	execTime += m_OpenCLKernels[NN_KERNEL_IDX_SCALEBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

	return execTime;
}

float OCLManager::AddBias(OCLBuffer *output, OCLBuffer *biases, int batch, int n, int size) { //, int activationType) {

	int LOCAL_BLOCK = BLOCK;
	int globalDimX = size / LOCAL_BLOCK;
	if (globalDimX % LOCAL_BLOCK != 0)
		globalDimX = ((globalDimX + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;

	m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->pGlobal(globalDimX, n)->pLocal(LOCAL_BLOCK, 1);
	m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(0, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(1, biases->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(2, n);
	m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->arg(3, size);
	return m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::ScaleAddBias(OCLBuffer *output, OCLBuffer *scales, OCLBuffer *biases, int batch, int n, int size) {

	int globalDimX = size / BLOCK;
	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->pGlobal(globalDimX, n)->pLocal(BLOCK, 1);
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->arg(0, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->arg(1, scales->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->arg(2, biases->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->arg(3, n);
	m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->arg(4, size);
	return m_OpenCLKernels[NN_KERNEL_IDX_SCALEADDBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

}

float OCLManager::NormScaleAddBias(OCLBuffer *output, OCLBuffer *scales, OCLBuffer *biases, OCLBuffer *mean,
	OCLBuffer *variance, int filters, int spatial) {

	int N = spatial;
	int LOCAL_BLOCK = BLOCK;
	int globalDimX = N / LOCAL_BLOCK;
	if (globalDimX % LOCAL_BLOCK != 0)
		globalDimX = ((globalDimX + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;

	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->pGlobal(globalDimX, filters)->pLocal(LOCAL_BLOCK, 1);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(0, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(1, scales->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(2, biases->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(3, mean->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(4, variance->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(5, filters);
	m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->arg(6, spatial);

	return m_OpenCLKernels[NN_KERNEL_IDX_NORMSCALEADDBIAS]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::ActivateInputs(OCLBuffer *x, OCLBuffer *swapBuffer, int n, int actType) {

	int globalDimX = n / BLOCK;

	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	int KERNEL_IDX = (actType == 7) ? NN_KERNEL_IDX_LEAKY_ACTIVATE : NN_KERNEL_IDX_LINEAR_ACTIVATE;

	m_OpenCLKernels[KERNEL_IDX]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[KERNEL_IDX]->arg(0, x->getMem());
	m_OpenCLKernels[KERNEL_IDX]->arg(1, swapBuffer->getMem());
	m_OpenCLKernels[KERNEL_IDX]->arg(2, n);
	m_OpenCLKernels[KERNEL_IDX]->arg(3, actType);
	return m_OpenCLKernels[KERNEL_IDX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

}

float OCLManager::FlattenArray(OCLBuffer *x, int spatial, int layers, int batch, int forward, OCLBuffer *out) {

	int size = spatial * batch * layers;

	int globalDimX = size / BLOCK;

	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(0, size);
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(1, x->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(2, spatial);
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(3, layers);
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(4, batch);
	m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->arg(5, out->getMem());
	return m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::SoftMax(OCLBuffer *input, int n, int offset, int groups, float temp, OCLBuffer *output, int base) {

	int inputs = n;
	int batch = groups;
	int LOCAL_BLOCK = 2;// BLOCK;

	int globalDimX = batch / LOCAL_BLOCK;

	if (globalDimX % LOCAL_BLOCK != 0)
		globalDimX = ((globalDimX + LOCAL_BLOCK) / LOCAL_BLOCK) * LOCAL_BLOCK;
	
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->pGlobal(globalDimX)->pLocal(LOCAL_BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(0, inputs);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(1, offset);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(2, batch);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(3, input->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(4, temp);
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(5, output->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->arg(6, base);
	return m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
}

float OCLManager::MaxPool(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, OCLBuffer *input, OCLBuffer *output) { 

	int globalDimX = n / BLOCK;

	if (globalDimX % BLOCK != 0)
		globalDimX = ((globalDimX + BLOCK) / BLOCK) * BLOCK;

	float execTime = 0.0f;

	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(globalDimX)->pLocal(BLOCK);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(0, n);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(1, in_h);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(2, in_w);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(3, in_c);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(4, stride);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(5, size);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(6, pad);
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(7, input->getMem());
	m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->arg(8, output->getMem());
	execTime += m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);

	return execTime;
}


void OCLManager::TestCLBlastLibHalfFloat() {


}



void OCLManager::TestCLBlastLib() {

	const size_t m = 32;// 128;
	const size_t n = 43264;// 64;
	const size_t k = 144;// 512;
	const float alpha = 0.7f;
	const float beta = 1.0f;
	const size_t a_ld = k;
	const size_t b_ld = n;
	const size_t c_ld = n;
	
	/*const cl_half alpha = FloatToHalf(0.5f);

	cl_half* host_a = (cl_half*)malloc(sizeof(cl_half)*n);
	cl_half* host_b = (cl_half*)malloc(sizeof(cl_half)*n);
	for (size_t i = 0; i<n; ++i) { host_a[i] = FloatToHalf(2.2f); }
	for (size_t i = 0; i<n; ++i) { host_b[i] = FloatToHalf(0.4f); }
	printf("Input values at index 0: alpha * a[0] + b[0] == %.3lf * %.3lf + %.3lf\n",
		HalfToFloat(alpha), HalfToFloat(host_a[0]), HalfToFloat(host_b[0]));
		*/

	float* host_a = (float*)malloc(sizeof(float)*m*k);
	float* host_b = (float*)malloc(sizeof(float)*n*k);
	float* host_c = (float*)malloc(sizeof(float)*m*n);
	
	for (size_t i = 0; i<m*k; ++i)
		host_a[i] = 12.193f; 
	
	for (size_t i = 0; i<n*k; ++i)
		host_b[i] = -8.199f; 
	
	for (size_t i = 0; i<m*n; ++i)
		host_c[i] = 0.0f; 

	int totalSize_a = m * k * sizeof(float);
	OCLBuffer* device_a = m_OpenCLSetup.createBuffer(totalSize_a, CL_MEM_READ_WRITE, NULL);
	int totalSize_b = n * k * sizeof(float);
	OCLBuffer* device_b = m_OpenCLSetup.createBuffer(totalSize_b, CL_MEM_READ_WRITE, NULL);
	int totalSize_c = m * n * sizeof(float);
	OCLBuffer* device_c = m_OpenCLSetup.createBuffer(totalSize_c, CL_MEM_READ_WRITE, NULL);

	device_a->write(host_a, totalSize_a, 0, CL_TRUE);
	device_b->write(host_b, totalSize_b, 0, CL_TRUE);
	device_c->write(host_c, totalSize_c, 0, CL_TRUE);

	cl_event event = NULL;

	auto timings = std::vector<double>(1);

	
	/*for (auto &timing : timings) {

		const auto start_time = std::chrono::steady_clock::now();

		CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
			m, n, k, alpha, device_a->getMem(), 0, a_ld,
			device_b->getMem(), 0, b_ld, beta, device_c->getMem(), 0, c_ld, m_OpenCLSetup.getQueue(), &event);

		if (status == CLBlastSuccess) {

			clWaitForEvents(1, &event);
			clReleaseEvent(event);
		}

		
		const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		
		timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
		printf("CLBlastSgemm() Execution Time : %f\n\n", timing);
	}*/

	printf("CLBlastSgemm() Execution Time : %f\n\n", *std::min_element(timings.begin(), timings.end()));
	free(host_c);
	delete device_a;
	delete device_b;
	delete device_c;

	

	free(host_a);
	free(host_b);
}


/*int inBlock = 1024;
int globalWorkGsize = (N + inBlock - 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = N / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;

inBlock = 1;
globalWorkGsize = N / inBlock;*/

//m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_NORMARR]->pGlobal(N, 1, 1)->pLocal(1, 1, 1);



/*int inBlock = 1024;
int globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/

//m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->pGlobal((size - 1) / BLOCK + 1, n, batch)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_ADDBIAS]->pGlobal(size, n, batch)->pLocal(1, 1, 1);

/*int inBlock = 1024;
int globalWorkGsize = ((n - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((n - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/

//m_OpenCLKernels[NN_KERNEL_IDX_ACTIVATE]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_ACTIVATE]->pGlobal(n, 1, 1)->pLocal(1, 1, 1);

/*int inBlock = 1024;
int globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/

//m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_FLATARR]->pGlobal(size, 1, 1)->pLocal(1, 1, 1);

/*int inBlock = 1024;
int globalWorkGsize = ((batch - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((batch - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/

//m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->pGlobal(batch, 1, 1)->pLocal(1, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_SOFTMAX]->pGlobal(globalWorkGsize, 1, 1)->pLocal(inBlock, 1, 1);

/*int inBlock = 1024;
int globalWorkGsize = (n + inBlock - 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = n / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;

if(globalDims.m_NumX % BLOCK != 0)
globalDims.m_NumX = ((globalDims.m_NumX + BLOCK) / BLOCK) * BLOCK;*/

//m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);

//m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(n, 1, 1)->pLocal(1, 1, 1);

//m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(globalWorkGsize, 1, 1)->pLocal(inBlock, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_MAXPOOL]->pGlobal(globalDims.m_NumX)->pLocal(BLOCK);

//m_OpenCLKernels[NN_KERNEL_IDX_REORG]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_REORG]->pGlobal(size, 1, 1)->pLocal(1, 1, 1);

/*int inBlock = 1024;
int globalWorkGsize = ((N - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((N - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/
//if (globalDims.m_NumX % BLOCK != 0)
//	globalDims.m_NumX = ((globalDims.m_NumX + BLOCK) / BLOCK) * BLOCK;

//m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->pGlobal(globalDims.m_NumX, globalDims.m_NumY, globalDims.m_NumZ)->pLocal(BLOCK, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->pGlobal(N, 1, 1)->pLocal(1, 1, 1);
//m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->pGlobal(globalWorkGsize)->pLocal(inBlock);

/*
cl_event execEvent = NULL;

CLBlastStatusCode status = CLBlastSim2col(channels, height, width, ksize, ksize, pad, pad,
	stride, stride, 0, 0, im->getMem(), 0, data_col->getMem(), 0, m_OpenCLSetup.getQueue(), &execEvent);

if (status == CLBlastSuccess) {

	clWaitForEvents(1, &execEvent);
	clReleaseEvent(execEvent);
}
*/


/*int inBlock = m_OpenCLKernels[NN_KERNEL_IDX_IM2COL]->GetOptimalLWGSize();
int globalWorkGsize = (num_kernels + inBlock - 1) / inBlock;

while(globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = num_kernels / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;*/




//Call clblas extended function. Perform gemm for the lower right sub-matrices 
/*cl_int err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alpha, bufferA->getMem(), 0, ldA,
bufferB->getMem(), 0, ldB, beta, bufferC->getMem(), 0, ldC,	1, m_OpenCLSetup.getQueue(), 0, NULL, &execEvent);

if (err == CL_SUCCESS) {

if (PROFILE_KERNELS == 1 || INTERNAL_PROFILE) {

clWaitForEvents(1, &execEvent);
//clFinish(*m_OpenCLSetup.getQueue());
//clReleaseEvent(execEvent);

long long start, end;


cl_int status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_START,
sizeof(start), &start, NULL);
status = clGetEventProfilingInfo(execEvent, CL_PROFILING_COMMAND_END,
sizeof(end), &end, NULL);

total = (double)(end - start) / 1e6;
printf("Total kernel time was {%5.2f} msecs - ComputeGEMM() \n", total);

clReleaseEvent(execEvent);
}
}*/

/*
float OCLManager::ReOrgOutput(OCLBuffer *x, int w, int h, int c, int batch, int stride, int forward, OCLBuffer *out) {

int size = w * h * c * batch;

StructOCLDims globalDims = getOCLGridDims(size);
int inBlock = m_OpenCLKernels[NN_KERNEL_IDX_IM2COL]->GetOptimalLWGSize();
int globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;

while (globalWorkGsize < inBlock) {

inBlock = inBlock / 2;
globalWorkGsize = ((size - 1) / inBlock + 1) / inBlock;
}

if (globalWorkGsize % inBlock != 0)
globalWorkGsize = ((globalWorkGsize + inBlock) / inBlock) * inBlock;


//m_OpenCLKernels[NN_KERNEL_IDX_REORG]->pGlobal(globalWorkGsize)->pLocal(inBlock);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->pGlobal(size)->pLocal(1);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(0, size);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(1, x->getMem());
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(2, w);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(3, h);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(4, c);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(5, batch);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(6, stride);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(7, forward);
m_OpenCLKernels[NN_KERNEL_IDX_REORG]->arg(8, out->getMem());
return m_OpenCLKernels[NN_KERNEL_IDX_REORG]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
//printf("ReOrgOutput speed is %f : \n", sec(clock() - time));
}

float OCLManager::CopyArray(int N, OCLBuffer * X, int xOffset, int INCX, OCLBuffer * Y, int yOffset, int INCY) {

StructOCLDims globalDims = getOCLGridDims(N);

m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->pGlobal(N)->pLocal(1);
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(0, N);
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(1, X->getMem());
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(2, xOffset);
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(3, INCX);
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(4, Y->getMem());
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(5, yOffset);
m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->arg(6, INCY);
return m_OpenCLKernels[NN_KERNEL_IDX_COPYARR]->run(PROFILE_KERNELS, BLOCK_KERNEL_EXEC);
//printf("CopyArray speed is %f : \n", sec(clock() - time));
}
*/
