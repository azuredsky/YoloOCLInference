
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

#include "YoloOCLDNN.h"

#define strcpy strcpy_s

float BBOX_COLORS[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

float rand_uniform(float minVal, float maxVal)
{
	if (maxVal < minVal) {

		float swap = minVal;
		minVal = maxVal;
		maxVal = swap;
	}
	return ((float)rand() / RAND_MAX * (maxVal - minVal)) + minVal;
}

void ParseDelimitedStrToIntVec(std::string instr, std::vector<int> &outVals) {

	std::stringstream ss(instr);

	int i;

	while (ss >> i){

		outVals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}
}

void ParseDelimitedStrToFloatVec(std::string instr, std::vector<float> &outVals) {

	std::stringstream ss(instr);

	float i;

	while (ss >> i) {

		outVals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}
}

EnumYOLODeepNNLayerType MapNNLayerTypeStr(std::string inStr) {

	if (inStr.find("region") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION;
	else if (inStr.find("convolutional") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;
	else if (inStr.find("maxpool") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL;
	
	return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;

}

EnumYOLODeepNNActivationType MapNNLayerActivationStr(char *activationStr) {

	if (strcmp(activationStr, "logistic") == 0) 
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LOGISTIC;
	if (strcmp(activationStr, "loggy") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LOGGY;
	if (strcmp(activationStr, "relu") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELU;
	if (strcmp(activationStr, "elu") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_ELU;
	if (strcmp(activationStr, "relie") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELIE;
	if (strcmp(activationStr, "plse") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_PLSE;
	if (strcmp(activationStr, "hardtan") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_HARDTAN;
	if (strcmp(activationStr, "lhtan") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LHTAN;
	if (strcmp(activationStr, "linear") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LINEAR;
	if (strcmp(activationStr, "ramp") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RAMP;
	if (strcmp(activationStr, "leaky") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LEAKY;
	if (strcmp(activationStr, "tanh") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_TANH;
	if (strcmp(activationStr, "stair") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_STAIR;

	printf("ERROR : Couldn't find activation function %s, going with ReLU\n", activationStr);

	return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELU;
}


YOLONeuralNet::YOLONeuralNet(char* classLabelsFile, char *networkConfigFile,  char *weightsFile, bool display, bool saveOutput) {

	strcpy(m_ClassLabelsFile, classLabelsFile);
	strcpy(m_NetworkConfigFile, networkConfigFile);
	strcpy(m_WeightsFile, weightsFile);
	m_InImage = NULL;
	m_ResizedImage = NULL;
	m_TempImage = NULL;
	m_CairoSurface = NULL;
	m_EnableDisplay = display;
	m_SaveOutput = saveOutput;

	//ENABLE_DISPLAY = (display?1:0);

}

YOLONeuralNet::~YOLONeuralNet() {

}

bool YOLONeuralNet::ParseNetworkConfiguration() {


	CSimpleIniA::TNamesDepend sections;
	m_IniReader->GetAllSections(sections);
	sections.sort(CSimpleIniA::Entry::LoadOrder());

	if(sections.size() == 0) {

		//LOG error
		return false;
	}

	CSimpleIniA::TNamesDepend::const_iterator i;
	for (i = sections.begin(); i != sections.end(); ++i)
		m_LayerNames.push_back(i->pItem);

		
	memset(m_YOLODeepNN, 0, sizeof(StructYOLODeepNN));
	m_YOLODeepNN->m_TotalLayers = (int)sections.size() - 1;
	m_YOLODeepNN->m_Layers = (StructYOLODeepNNLayer*)calloc(m_YOLODeepNN->m_TotalLayers, sizeof(StructYOLODeepNNLayer));

	m_YOLODeepNN->m_GpuIndex = 0; // TODO : Pass this as part of configuration
	m_YOLODeepNN->m_BatchSize = (int)m_IniReader->GetDoubleValue("net", "batch", 1);
	int subDivs = (int)m_IniReader->GetDoubleValue("net", "subdivisions", 1);
	m_YOLODeepNN->m_TimeSteps = (int)m_IniReader->GetDoubleValue("net", "time_steps", 1);
	m_YOLODeepNN->m_H = (int)m_IniReader->GetDoubleValue("net", "height", 0);
	m_YOLODeepNN->m_W = (int)m_IniReader->GetDoubleValue("net", "width", 0);
	m_YOLODeepNN->m_C = (int)m_IniReader->GetDoubleValue("net", "channels", 0);
	m_YOLODeepNN->m_Inputs = (int)m_IniReader->GetDoubleValue("net", "inputs", m_YOLODeepNN->m_H * m_YOLODeepNN->m_W * m_YOLODeepNN->m_C);
	
	if (!m_YOLODeepNN->m_Inputs && !(m_YOLODeepNN->m_H && m_YOLODeepNN->m_W && m_YOLODeepNN->m_C)) {

		//LOG Error
		return false;
	}


	return true;
}

bool YOLONeuralNet::PrepareConvolutionalTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int pad = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "pad", 0);
	int padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "padding", 0);
	if (pad)
		padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", 1) / 2;


	char activation_s[512];
	strcpy(activation_s, m_IniReader->GetValue((char*)m_LayerNames[sectionIdx].c_str(), "activation", "logistic"));//, activation_s);

	EnumYOLODeepNNActivationType activation = MapNNLayerActivationStr(activation_s);

	m_YOLODeepNN->m_Layers[layerIdx].m_Flipped = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "flipped", 0);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;

	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_C = layerFeedParams->m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_N = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "filters", 1);
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_Stride = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "stride", 1); ;
	m_YOLODeepNN->m_Layers[layerIdx].m_Size = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", 1); ;
	m_YOLODeepNN->m_Layers[layerIdx].m_Pad = padding;
	m_YOLODeepNN->m_Layers[layerIdx].m_BatchNormalize = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "batch_normalize", 0);

	int weightsLength = m_YOLODeepNN->m_Layers[layerIdx].m_C *
		m_YOLODeepNN->m_Layers[layerIdx].m_N * m_YOLODeepNN->m_Layers[layerIdx].m_Size *
		m_YOLODeepNN->m_Layers[layerIdx].m_Size;

	m_YOLODeepNN->m_Layers[layerIdx].m_Weights = (float*)calloc(weightsLength, sizeof(float));

	m_YOLODeepNN->m_Layers[layerIdx].m_Biases = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));

	float scale = (float)sqrt(2. / (m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_C));

	for (int i = 0; i < weightsLength; ++i)
		m_YOLODeepNN->m_Layers[layerIdx].m_Weights[i] = scale * rand_uniform(-1, 1);

	m_YOLODeepNN->m_Layers[layerIdx].m_OutH = (m_YOLODeepNN->m_Layers[layerIdx].m_H + 2 * m_YOLODeepNN->m_Layers[layerIdx].m_Pad - m_YOLODeepNN->m_Layers[layerIdx].m_Size) /
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride + 1;

	m_YOLODeepNN->m_Layers[layerIdx].m_OutW = (m_YOLODeepNN->m_Layers[layerIdx].m_W + 2 * m_YOLODeepNN->m_Layers[layerIdx].m_Pad - m_YOLODeepNN->m_Layers[layerIdx].m_Size) /
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride + 1;

	m_YOLODeepNN->m_Layers[layerIdx].m_OutC = m_YOLODeepNN->m_Layers[layerIdx].m_N;

	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC;

	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_W * m_YOLODeepNN->m_Layers[layerIdx].m_H * m_YOLODeepNN->m_Layers[layerIdx].m_C;

	m_YOLODeepNN->m_Layers[layerIdx].m_Output = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_Batch *  m_YOLODeepNN->m_Layers[layerIdx].m_Outputs, sizeof(float));

	if (m_YOLODeepNN->m_Layers[layerIdx].m_BatchNormalize) {

		m_YOLODeepNN->m_Layers[layerIdx].m_Scales = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));

		for (int i = 0; i < m_YOLODeepNN->m_Layers[layerIdx].m_N; ++i)
			m_YOLODeepNN->m_Layers[layerIdx].m_Scales[i] = 1;

		m_YOLODeepNN->m_Layers[layerIdx].m_RollingMean = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));
		m_YOLODeepNN->m_Layers[layerIdx].m_RollingVariance = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));
	}


	if (m_YOLODeepNN->m_GpuIndex >= 0) {

		m_YOLODeepNN->m_Layers[layerIdx].m_Weights_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Weights, weightsLength);
		m_YOLODeepNN->m_Layers[layerIdx].m_Biases_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Biases, m_YOLODeepNN->m_Layers[layerIdx].m_N);
		
		for(int i = 0; i < 2; i++) 
			m_YOLODeepNN->m_Layers[layerIdx].m_OutputSwapGPUBuffers[i] = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Output,
				m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_OutH *
				m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_N);
	}

	m_YOLODeepNN->m_Layers[layerIdx].m_Workspace_Size = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW *
		m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_C * sizeof(float);
	m_YOLODeepNN->m_Layers[layerIdx].m_Activation = activation;

	printf("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", 
		m_YOLODeepNN->m_Layers[layerIdx].m_N, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Size, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Size, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride, 
		m_YOLODeepNN->m_Layers[layerIdx].m_W, 
		m_YOLODeepNN->m_Layers[layerIdx].m_H, 
		m_YOLODeepNN->m_Layers[layerIdx].m_C, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutW, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutH, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutC);

	return true;
}


bool YOLONeuralNet::PrepareRegionTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int coords = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "coords", 4);
	int classes = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "classes", 20);
	int num = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "num", 1);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION;
	m_YOLODeepNN->m_Layers[layerIdx].m_N = num;
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_Classes = classes;
	m_YOLODeepNN->m_Layers[layerIdx].m_Coords = coords;
	m_YOLODeepNN->m_Layers[layerIdx].m_Biases = (float*)calloc(num * 2, sizeof(float));
	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = layerFeedParams->m_H * layerFeedParams->m_W * num * (classes + coords + 1);
	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_Outputs;
	
	for (int i = 0; i < num * 2; ++i)
		m_YOLODeepNN->m_Layers[layerIdx].m_Biases[i] = .5;


#ifndef PINNED_MEM_OUTPUT
	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Output, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_Outputs);
#else

	m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer = m_OCLManager->InitializePinnedFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_Outputs);
	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer->m_OCLBuffer;
	m_YOLODeepNN->m_Layers[layerIdx].m_PinnedOutput = (float*)m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer->m_PinnedMemory;
#endif

	printf("detection\n");
	srand(0);


	m_YOLODeepNN->m_Layers[layerIdx].m_ClassFix = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "classfix", 0);

	char outStr[512];
	strcpy(outStr, m_IniReader->GetValue((char*)m_LayerNames[sectionIdx].c_str(), "anchors", ""));

	char *a = outStr;

	if (a) {

		int len = (int)strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {

			if (a[i] == ',') 
				++n;
		}

		for (i = 0; i < n; ++i) {

			float bias = (float)atof(a);
			m_YOLODeepNN->m_Layers[layerIdx].m_Biases[i] = bias;
			a = strchr(a, ',') + 1;
		}
	}

	return true;
}

bool YOLONeuralNet::PrepareMaxpoolTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int stride = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "stride", 1);
	int size = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", stride);
	int padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "padding", (size - 1) / 2);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL;
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_C = layerFeedParams->m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Pad = padding;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutW = (m_YOLODeepNN->m_Layers[layerIdx].m_W + 2 * padding) / stride;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutH = (m_YOLODeepNN->m_Layers[layerIdx].m_H + 2 * padding) / stride;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutC = m_YOLODeepNN->m_Layers[layerIdx].m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC;
	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_H * m_YOLODeepNN->m_Layers[layerIdx].m_W * m_YOLODeepNN->m_Layers[layerIdx].m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Size = size;
	m_YOLODeepNN->m_Layers[layerIdx].m_Stride = stride;
	int outSize = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC * m_YOLODeepNN->m_Layers[layerIdx].m_Batch;

	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_OCLManager->InitializeFloatArray(NULL, outSize);


	printf("max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, 
						m_YOLODeepNN->m_Layers[layerIdx].m_W, 
						m_YOLODeepNN->m_Layers[layerIdx].m_H, 
						m_YOLODeepNN->m_Layers[layerIdx].m_C, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutW, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutH, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutC);
	return true;
}

bool YOLONeuralNet::ParseNNLayers() {

	EnumYOLODeepNNLayerType layerType;
	int layerCount = 0;
	size_t workspaceSize = 0;

	StructLayerFeedParams layerFeedParams;
	layerFeedParams.m_H = m_YOLODeepNN->m_H;
	layerFeedParams.m_W = m_YOLODeepNN->m_W;
	layerFeedParams.m_C = m_YOLODeepNN->m_C;
	layerFeedParams.m_Inputs = m_YOLODeepNN->m_Inputs;
	layerFeedParams.m_Batch = m_YOLODeepNN->m_BatchSize;

	for (int i = 1; i <= m_YOLODeepNN->m_TotalLayers; i++) {

		layerFeedParams.m_Index = layerCount;
		printf("%5d ", layerCount);
		layerType = MapNNLayerTypeStr(m_LayerNames[i]);

		switch (layerType) {
			
			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:
				
				PrepareConvolutionalTypeLayer(i, layerCount, &layerFeedParams);
				break;

			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

				PrepareRegionTypeLayer(i, layerCount, &layerFeedParams);
				break;

			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

				PrepareMaxpoolTypeLayer(i, layerCount, &layerFeedParams);
				break;

			default:
				break;
		}

		
		m_YOLODeepNN->m_Layers[layerCount].m_DontLoad = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[i].c_str(), "dontload", 0);
		m_YOLODeepNN->m_Layers[layerCount].m_DontLoadScales = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[i].c_str(), "dontloadscales", 0);
		
		if (m_YOLODeepNN->m_Layers[layerCount].m_Workspace_Size > workspaceSize)
			workspaceSize = m_YOLODeepNN->m_Layers[layerCount].m_Workspace_Size;

		layerFeedParams.m_H = m_YOLODeepNN->m_Layers[layerCount].m_OutH;
		layerFeedParams.m_W = m_YOLODeepNN->m_Layers[layerCount].m_OutW;
		layerFeedParams.m_C = m_YOLODeepNN->m_Layers[layerCount].m_OutC;
		layerFeedParams.m_Inputs = m_YOLODeepNN->m_Layers[layerCount].m_Outputs;
		++layerCount;
	}

	if (workspaceSize)
		m_YOLODeepNN->m_Workspace = m_OCLManager->InitializeFloatArray(NULL, (workspaceSize - 1) / sizeof(float) + 1);

	return true;
}


//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO

void transpose_matrix(float *a, int rows, int cols) {

	float *transpose = (float*)calloc(rows*cols, sizeof(float));
	int x, y;
	for (x = 0; x < rows; ++x)
		for (y = 0; y < cols; ++y)
			transpose[y*rows + x] = a[x*cols + y];

	memcpy(a, transpose, rows*cols * sizeof(float));
	free(transpose);
}

bool YOLONeuralNet::ParseNNWeights() {

	FILE *fp = fopen(m_WeightsFile, "rb");
	if (!fp) {

		printf("ERROR - Failed to find NN weights file %s\n", m_WeightsFile);
		return false;
	}

	int majorRev;
	int minorRev;
	int revNum;
	int filterIdx;
	int totalExamples;
	StructYOLODeepNNLayer yoloDeepNNLayer;
	fread(&majorRev, sizeof(int), 1, fp);
	fread(&minorRev, sizeof(int), 1, fp);
	fread(&revNum, sizeof(int), 1, fp);
	fread(&totalExamples, sizeof(int), 1, fp);

	int isTransposeEnabled = (majorRev > 1000) || (minorRev > 1000);

	for (int i = 0; i < m_YOLODeepNN->m_TotalLayers ; ++i) {

		yoloDeepNNLayer = m_YOLODeepNN->m_Layers[i];

		if (yoloDeepNNLayer.m_DontLoad)
			continue;

		if (yoloDeepNNLayer.m_LayerType == EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL) {
			
			int numWeights = yoloDeepNNLayer.m_N * yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size;
			fread(yoloDeepNNLayer.m_Biases, sizeof(float), yoloDeepNNLayer.m_N, fp);

			if (yoloDeepNNLayer.m_BatchNormalize 
				&& (!yoloDeepNNLayer.m_DontLoadScales)) {

				fread(yoloDeepNNLayer.m_Scales, sizeof(float), yoloDeepNNLayer.m_N, fp);
				fread(yoloDeepNNLayer.m_RollingMean, sizeof(float), yoloDeepNNLayer.m_N, fp);
				fread(yoloDeepNNLayer.m_RollingVariance, sizeof(float), yoloDeepNNLayer.m_N, fp);
			}

			fread(yoloDeepNNLayer.m_Weights, sizeof(float), numWeights, fp);

			if (yoloDeepNNLayer.m_BatchNormalize
				&& (!yoloDeepNNLayer.m_DontLoadScales)) {

				//Nice trick to fold batch normalization layer into convolution layer
				//This saves good amount of processing power
				//https://github.com/hollance/Forge/blob/master/Examples/YOLO/yolo2metal.py
				//http://machinethink.net/blog/object-detection-with-yolo/

				for (int j = 0; j < yoloDeepNNLayer.m_N; j++) {

					yoloDeepNNLayer.m_Biases[j] = yoloDeepNNLayer.m_Biases[j] - (yoloDeepNNLayer.m_RollingMean[j] * yoloDeepNNLayer.m_Scales[j]
						/ sqrt(yoloDeepNNLayer.m_RollingVariance[j] + 0.0001f));
				}

				for (int j = 0; j < numWeights; j++) {

					filterIdx = j / (yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size);
					yoloDeepNNLayer.m_Weights[j] = yoloDeepNNLayer.m_Weights[j] * yoloDeepNNLayer.m_Scales[filterIdx]
						/ sqrt(yoloDeepNNLayer.m_RollingVariance[filterIdx] + 0.0001f);
				}
			}

			if (yoloDeepNNLayer.m_Flipped)
				transpose_matrix(yoloDeepNNLayer.m_Weights, yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size, yoloDeepNNLayer.m_N);


			if (m_YOLODeepNN->m_GpuIndex >= 0) {
			
				m_OCLManager->WriteFloatArray(yoloDeepNNLayer.m_Weights_Gpu, yoloDeepNNLayer.m_Weights, numWeights);
				m_OCLManager->WriteFloatArray(yoloDeepNNLayer.m_Biases_Gpu, yoloDeepNNLayer.m_Biases, yoloDeepNNLayer.m_N);
			}
		}
	}

	fclose(fp);
	return true;
}

bool YOLONeuralNet::Initialize() {

	std::ifstream classLabelsFile(m_ClassLabelsFile);

	if (!classLabelsFile.good()) {

		//LOG file doesnot exist
		return false;
	}

	std::copy(std::istream_iterator<std::string>(classLabelsFile),
		std::istream_iterator<std::string>(),
		std::back_inserter(m_ClassLabels));

	m_IniReader = new CSimpleIniA(false, false, false);
	m_IniReader->LoadFile(m_NetworkConfigFile);

	m_OCLManager = new OCLManager();
	if (m_OCLManager->Initialize() != OCL_STATUS_READY) {
	
		//Log error
		return false;
	}

	strcpy(m_OCLDeviceName, m_OCLManager->GetDeviceName());


	m_YOLODeepNN = new StructYOLODeepNN;
	memset(m_YOLODeepNN, 0, sizeof(m_YOLODeepNN));
	ParseNetworkConfiguration();
	ParseNNLayers();
	ParseNNWeights();

	m_YOLODeepNN->m_BatchSize = 1;
	for(int i = 0; i < m_YOLODeepNN->m_TotalLayers; i++)
		m_YOLODeepNN->m_Layers[i].m_Batch = 1;

	srand(2222222);


	return true;
}

void YOLONeuralNet::Finalize() {

	if (m_YOLODeepNN->m_Workspace != NULL) {

		delete m_YOLODeepNN->m_Workspace;
		m_YOLODeepNN->m_Workspace = NULL;
	}

	for (int i = 0; i < m_YOLODeepNN->m_TotalLayers; i++) {

		switch(m_YOLODeepNN->m_Layers[i].m_LayerType){

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:

			free(m_YOLODeepNN->m_Layers[i].m_Biases);
			m_YOLODeepNN->m_Layers[i].m_Biases = NULL;

			free(m_YOLODeepNN->m_Layers[i].m_RollingMean);
			m_YOLODeepNN->m_Layers[i].m_RollingMean = NULL;

			free(m_YOLODeepNN->m_Layers[i].m_RollingVariance);
			m_YOLODeepNN->m_Layers[i].m_RollingVariance = NULL;

			free(m_YOLODeepNN->m_Layers[i].m_Weights);
			m_YOLODeepNN->m_Layers[i].m_Weights = NULL;

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Weights_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Weights_Gpu = NULL;

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Biases_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Biases_Gpu = NULL;

			for (int j = 0; j < 2; j++) {

				m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[j]);
				m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[j] = NULL;
			}


			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Output_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Output_Gpu = NULL;

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

			free(m_YOLODeepNN->m_Layers[i].m_Biases);
			m_YOLODeepNN->m_Layers[i].m_Biases = NULL;

 			m_OCLManager->FinalizePinnedFloatArray(m_YOLODeepNN->m_Layers[i].m_PinnedBuffer);
			m_YOLODeepNN->m_Layers[i].m_PinnedBuffer = NULL;

			break;
		}
	}

	delete m_YOLODeepNN;
	delete m_IniReader;

	m_OCLManager->Finalize();
	delete m_OCLManager;
	m_ClassLabels.clear();
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float overlap(float x1, float w1, float x2, float w2) {

	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_intersection(StructDetectionBBox a, StructDetectionBBox b) {

	float w = overlap(a.m_X, a.m_W, b.m_X, b.m_W);
	float h = overlap(a.m_Y, a.m_H, b.m_Y, b.m_H);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_union(StructDetectionBBox a, StructDetectionBBox b) {

	float i = box_intersection(a, b);
	float u = a.m_W * a.m_H + b.m_W * b.m_W - i;
	return u;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_iou(StructDetectionBBox a, StructDetectionBBox b) {

	return box_intersection(a, b) / box_union(a, b);
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
int nms_comparator(const void *pa, const void *pb) {

	StructSortableBBox a = *(StructSortableBBox *)pa;
	StructSortableBBox b = *(StructSortableBBox *)pb;
	float diff = a.m_ProbScores[a.m_Index][b.m_ClassIdx] - b.m_ProbScores[b.m_Index][b.m_ClassIdx];
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void YOLONeuralNet::ApplyNMS(StructDetectionBBox *boxes, float **probs, int total, int classes, float thresh) {

	StructSortableBBox *s = (StructSortableBBox*)calloc(total, sizeof(StructSortableBBox));

	for (int i = 0; i < total; ++i) {

		s[i].m_Index = i;
		s[i].m_ClassIdx = 0;
		s[i].m_ProbScores = probs;
	}

	for (int k = 0; k < classes; ++k) {
		for (int i = 0; i < total; ++i) {

			s[i].m_ClassIdx = k;
		}

		qsort(s, total, sizeof(StructSortableBBox), nms_comparator);

		for (int i = 0; i < total; ++i) {

			if (probs[s[i].m_Index][k] == 0) 
				continue;

			StructDetectionBBox a = boxes[s[i].m_Index];

			for (int j = i + 1; j < total; ++j) {

				StructDetectionBBox b = boxes[s[j].m_Index];
				if (box_iou(a, b) > thresh)
					probs[s[j].m_Index][k] = 0;
			}
		}
	}
	free(s);
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
StructDetectionBBox YOLONeuralNet::GetRegionBBox(float *x, float *biases, int n, int index, int i, int j, int w, int h) {

	StructDetectionBBox bBox;
	bBox.m_X = (i + LogisticActivate(x[index + 0])) / w;
	bBox.m_Y = (j + LogisticActivate(x[index + 1])) / h;
	bBox.m_W = exp(x[index + 2]) * biases[2 * n];
	bBox.m_H = exp(x[index + 3]) * biases[2 * n + 1];
	
	if (DOABS) {

		bBox.m_W = exp(x[index + 2]) * biases[2 * n] / w;
		bBox.m_H = exp(x[index + 3]) * biases[2 * n + 1] / h;
	}
	return bBox;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void YOLONeuralNet::GetDetectionBBoxes(StructYOLODeepNNLayer *nnLayer, int w, int h, float thresh, float **probs, StructDetectionBBox *bBoxes, int onlyObjectness, int *map) {

	int j;
	//float *predictions = nnLayer->m_Output;
	float *predictions = NULL;
	
#ifndef PINNED_MEM_OUTPUT
	predictions = nnLayer->m_Output;
#else
	predictions = nnLayer->m_PinnedOutput;
#endif

	for (int i = 0; i < nnLayer->m_W * nnLayer->m_H; ++i) {

		int row = i / nnLayer->m_W;
		int col = i % nnLayer->m_W;
		
		for (int n = 0; n < nnLayer->m_N; ++n) {

			int index = i * nnLayer->m_N + n;
			int p_index = index * (nnLayer->m_Classes + 5) + 4;
			float scale = predictions[p_index];
			if (nnLayer->m_ClassFix == -1 && scale < .5) 
				scale = 0;

			int box_index = index * (nnLayer->m_Classes + 5);
			bBoxes[index] = GetRegionBBox(predictions, nnLayer->m_Biases, n, box_index, col, row, nnLayer->m_W, nnLayer->m_H);
			bBoxes[index].m_X *= w;
			bBoxes[index].m_Y *= h;
			bBoxes[index].m_W *= w;
			bBoxes[index].m_H *= h;

			int class_index = index * (nnLayer->m_Classes + 5) + 5;
			for (j = 0; j < nnLayer->m_Classes; ++j) {

				float prob = scale*predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (onlyObjectness) 
				probs[index][0] = scale;
		}
	}
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void add_pixel(StructImage *m, int x, int y, int c, float val) {

	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x] += val;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void set_pixel(StructImage *m, int x, int y, int c, float val) {

	if (x < 0 || y < 0 || c < 0 || x >= m->m_W || y >= m->m_H || c >= m->m_C) return;
	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x] = val;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float get_pixel(StructImage *m, int x, int y, int c) {

	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	return m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x];
}

void YOLONeuralNet::Resizeimage(int w, int h) {

	if (m_ResizedImage == NULL) {

		m_ResizedImage = new StructImage;
		m_ResizedImage->m_H = h;
		m_ResizedImage->m_W = w;
		m_ResizedImage->m_C = m_InImage->m_C;
		m_ResizedImage->m_DataArray = (float*)calloc(m_ResizedImage->m_H * m_ResizedImage->m_W * m_ResizedImage->m_C, sizeof(float));

		m_TempImage = new StructImage;
		m_TempImage->m_H = m_InImage->m_H;
		m_TempImage->m_W = w;
		m_TempImage->m_C = m_InImage->m_C;
		m_TempImage->m_DataArray = (float*)calloc(m_TempImage->m_H * m_TempImage->m_W * m_TempImage->m_C, sizeof(float));
	}

	//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
	int r, c, k;
	float w_scale = (float)(m_InImage->m_W - 1) / (w - 1);
	float h_scale = (float)(m_InImage->m_H - 1) / (h - 1);
	for (k = 0; k < m_InImage->m_C; ++k) {
		for (r = 0; r < m_InImage->m_H; ++r) {
			for (c = 0; c < w; ++c) {
				float val = 0;
				if (c == w - 1 || m_InImage->m_W == 1) {
					val = get_pixel(m_InImage, m_InImage->m_W - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(m_InImage, ix, r, k) + dx * get_pixel(m_InImage, ix + 1, r, k);
				}
				set_pixel(m_TempImage, c, r, k, val);
			}
		}
	}
	for (k = 0; k < m_InImage->m_C; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				float val = (1 - dy) * get_pixel(m_TempImage, c, iy, k);
				set_pixel(m_ResizedImage, c, r, k, val);
			}
			if (r == h - 1 || m_InImage->m_H == 1) continue;
			for (c = 0; c < w; ++c) {
				float val = dy * get_pixel(m_TempImage, c, iy + 1, k);
				add_pixel(m_ResizedImage, c, r, k, val);
			}
		}
	}
}



bool YOLONeuralNet::LoadInputImage(char const* fileName) {

	int flag = -1;
	int channels = 3;
	if (channels == 0) flag = -1;
	else if (channels == 1) flag = 0;
	else if (channels == 3) flag = 1;
	else
		fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);

	/*if (m_CurrentIplImage != NULL) {

		cvReleaseImage(&m_CurrentIplImage);
		m_CurrentIplImage = NULL;
	}*/

	//if (m_CurrentImage != NULL) {
	if(m_CurrentImage.data != NULL)
		m_CurrentImage.release();

	//}

	//if ((m_CurrentIplImage = cvLoadImage(fileName, flag)) == 0) {
	m_CurrentImage = cv::imread(std::string(fileName));
	if(m_CurrentImage.data == NULL) {

		fprintf(stderr, "Cannot load image \"%s\"\n", fileName);
		char buff[256];
		sprintf(buff, "echo %s >> bad.list", fileName);
		system(buff);
		return false;
	}

	//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
	//unsigned char *data = (unsigned char *)m_CurrentIplImage->imageData;
	unsigned char *data = (unsigned char *)m_CurrentImage.data;
	//int h = m_CurrentIplImage->height;
	int h = m_CurrentImage.rows;
	//int w = m_CurrentIplImage->width;
	int w = m_CurrentImage.cols;
	//int c = m_CurrentIplImage->nChannels;
	int c = m_CurrentImage.channels();
	//int step = m_CurrentIplImage->widthStep;
	int step = m_CurrentImage.step;

	if (m_InImage == NULL) {

		m_InImage = new StructImage;
		m_InImage->m_DataArray = (float*)calloc(h*w*c, sizeof(float));
	}

	int count = 0;
	m_InImage->m_W = w;
	m_InImage->m_H = h;
	m_InImage->m_C = c;


	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				m_InImage->m_DataArray[count++] = (float)(data[i*step + j*c + k] / 255.);
			}
		}
	}

	for (int i = 0; i < m_InImage->m_W * m_InImage->m_H; ++i) {

		float swap = m_InImage->m_DataArray[i];
		m_InImage->m_DataArray[i] = m_InImage->m_DataArray[i + m_InImage->m_W * m_InImage->m_H * 2];
		m_InImage->m_DataArray[i + m_InImage->m_W * m_InImage->m_H * 2] = swap;
	}

	return true;
}


//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float get_color(int c, int x, int max) {

	float ratio = ((float)x / max) * 5;
	int i = (int)floor(ratio);
	int j = (int)ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * BBOX_COLORS[i][c] + ratio*BBOX_COLORS[j][c];
	//printf("%f\n", r);
	return r;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
int max_index(float *a, int n) {

	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {

			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void DrawDetections(StructImage *im, int num, float thresh, StructDetectionBBox *boxes, float **probs, 
					std::vector<std::string> &names, int classes, cv::Mat &renderMat) { // * renderImage) {

	cv::Rect overlayRect;

	for (int i = 0; i < num; ++i) {
		
		int classidx = max_index(probs[i], classes);
		float prob = probs[i][classidx];

		if (prob > thresh) {

			int width = (int)(im->m_H * .009);

			//if(m_)
				//printf("%s: %.0f%%\n", names[classidx].c_str(), prob * 100);

			int offset = classidx * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			StructDetectionBBox b = boxes[i];

			int left = (int)((b.m_X - b.m_W / 2.) * im->m_W);
			int right = (int)((b.m_X + b.m_W / 2.) * im->m_W);
			int top = (int)((b.m_Y - b.m_H / 2.) * im->m_H);
			int bot = (int)((b.m_Y + b.m_H / 2.) * im->m_H);
			 
			if (left < 0) left = 0;
			if (right > im->m_W - 1) right = im->m_W - 1;
			if (top < 0) top = 0;
			if (bot > im->m_H - 1) bot = im->m_H - 1;

			overlayRect.x = left;
			overlayRect.y = top;
			overlayRect.width = right - left;
			overlayRect.height = bot - top;

			//cv::rectangle((cv::Mat)renderImage, overlayRect, cv::Scalar(blue * 255, green * 255, red * 255), 2);
			cv::rectangle(renderMat, overlayRect, cv::Scalar(blue * 255, green * 255, red * 255), 2);
			//draw_box_width(im, left, top, right, bot, width, red, green, blue);
		}
	}
}


void YOLONeuralNet::PutCairoTimeOverlay(
	std::string const& timeText,
	cv::Point2d timeCenterPoint,
	std::string const& fontFace,
	double fontSize,
	cv::Scalar textColor,
	bool fontItalic,
	bool fontBold) {

	if (m_CairoSurface == NULL) {

		m_CairoSurface = cairo_image_surface_create(
			CAIRO_FORMAT_ARGB32,
			m_OverlayMat.cols,
			m_OverlayMat.rows);

		m_Cairo = cairo_create(m_CairoSurface);

		m_CairoTarget = cv::Mat(
			cairo_image_surface_get_height(m_CairoSurface),
			cairo_image_surface_get_width(m_CairoSurface),
			CV_8UC4,
			cairo_image_surface_get_data(m_CairoSurface),
			cairo_image_surface_get_stride(m_CairoSurface));
	}

	cv::cvtColor(m_OverlayMat, m_CairoTarget, cv::COLOR_BGR2BGRA);

	cairo_select_font_face(
		m_Cairo,
		fontFace.c_str(),
		fontItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL,
		fontBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);

	cairo_set_font_size(m_Cairo, fontSize);
	cairo_set_source_rgb(m_Cairo, textColor[2], textColor[1], textColor[0]);

	cairo_text_extents_t extents;
	cairo_text_extents(m_Cairo, timeText.c_str(), &extents);

	cairo_move_to(
		m_Cairo,
		timeCenterPoint.x - extents.width / 2 - extents.x_bearing,
		timeCenterPoint.y - extents.height / 2 - extents.y_bearing);

	cairo_show_text(m_Cairo, timeText.c_str());
	cv::cvtColor(m_CairoTarget, m_OverlayMat, cv::COLOR_BGRA2BGR);
}

void YOLONeuralNet::ComputeYOLONNOutput(char* inputFile) {

	cv::Rect overlayRect;
	char fileName[256];
	char overlayText[256];
	char overlayDeviceProp[256];
	char outFolder[256];
	char outImage[256];
	//IplImage tmpPtr;
	float threshold = 0.2f;
	float nms = 0.45f;
	//m_CurrentIplImage = NULL;
	

	sprintf(outFolder, "%s\\output", ExePath().c_str());
	CreateDirectory(outFolder, NULL);

	sprintf(overlayDeviceProp, "Device : %s", m_OCLDeviceName);
	StructYOLODeepNNLayer *finalLayer = &m_YOLODeepNN->m_Layers[m_YOLODeepNN->m_TotalLayers - 1];

	StructDetectionBBox *detBBoxes = (StructDetectionBBox*)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(StructDetectionBBox));
	float **detProbScores = (float**)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(float *));

	for (int j = 0; j < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; ++j)
		detProbScores[j] = (float*)calloc(finalLayer->m_Classes, sizeof(float));

	int inputSize = m_YOLODeepNN->m_Layers[0].m_Inputs * m_YOLODeepNN->m_BatchSize;


	sprintf(fileName, inputFile);
	LoadInputImage(fileName);
	Resizeimage(m_YOLODeepNN->m_W, m_YOLODeepNN->m_H);
	m_OverlayMat = cv::Mat(cv::Size(m_InImage->m_W, 50), CV_8UC3);
	m_OverlayMat.setTo((cv::Scalar)0);
	m_OverlayFinalMat = cv::Mat(cv::Size(m_OverlayMat.cols, m_OverlayMat.rows), CV_8UC3);
	overlayRect.x = 0;
	overlayRect.y = 0;
	overlayRect.width = m_OverlayMat.cols;
	overlayRect.height = m_OverlayMat.rows;
	cv::Mat displayImageMat = cv::Mat(cv::Size(m_InImage->m_W, m_InImage->m_H), CV_8UC3);
	//displayImageMat = cv::cvarrToMat(m_CurrentIplImage, true);
	displayImageMat == m_CurrentImage.clone();

	StructYOLODeepNNState yoloNNCurrentState;
	memset(&yoloNNCurrentState, 0, sizeof(StructYOLODeepNNState));

	int BURN_ITERATIONS = 50;
	int iterIdx = 0;
	int NUM_ITERATIONS = (ENABLE_DISPLAY == 1) ? 5 : 1000;
	auto timings = std::vector<double>(NUM_ITERATIONS);

	yoloNNCurrentState.m_InputRefGpu = m_OCLManager->InitializeFloatArray(m_ResizedImage->m_DataArray, inputSize);

	for (auto &timing : timings) {

		const auto start_time = std::chrono::steady_clock::now();
		yoloNNCurrentState.m_LayerIndex = 0;
		yoloNNCurrentState.m_DeepNN = m_YOLODeepNN;
		yoloNNCurrentState.m_InputGpu = yoloNNCurrentState.m_InputRefGpu;
		yoloNNCurrentState.m_Workspace = m_YOLODeepNN->m_Workspace;
		yoloNNCurrentState.m_ConvSwapBufIdx = (yoloNNCurrentState.m_ConvSwapBufIdx == 0) ? 1 : 0;

		for (int i = 0; i < m_YOLODeepNN->m_TotalLayers; ++i) {

			yoloNNCurrentState.m_LayerIndex = i;

			PropagateLayerInputsForward(&m_YOLODeepNN->m_Layers[i], &yoloNNCurrentState);
			if (m_YOLODeepNN->m_Layers[i].m_LayerType == EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL)
				yoloNNCurrentState.m_InputGpu = m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[yoloNNCurrentState.m_ConvSwapBufIdx];
			else
				yoloNNCurrentState.m_InputGpu = m_YOLODeepNN->m_Layers[i].m_Output_Gpu;
			yoloNNCurrentState.m_InputSize = m_YOLODeepNN->m_Layers[i].m_Batch * m_YOLODeepNN->m_Layers[i].m_OutH *
				m_YOLODeepNN->m_Layers[i].m_OutW * m_YOLODeepNN->m_Layers[i].m_N;
		}

		const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
		printf("Predicted in %2.2f ms. Expected Proc speed is  : %2.2f FPS \n", timing, 1000 / timing);
		
		GetDetectionBBoxes(finalLayer, 1, 1, threshold, detProbScores, detBBoxes, 0, 0);
		ApplyNMS(detBBoxes, detProbScores, finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, finalLayer->m_Classes, nms);
		DrawDetections(m_InImage, finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, threshold, 
			detBBoxes, detProbScores, m_ClassLabels, finalLayer->m_Classes, m_CurrentImage);
		//cv::imwrite("test.jpg", m_CurrentImage);
		m_OverlayMat.setTo((cv::Scalar)0);
		sprintf(overlayText, "Inference Duration : %2.2f ms Speed : %2.2f fps", timing, 1000 / timing);
		PutCairoTimeOverlay(overlayText, cv::Point2d(180, 20), "arial", 15, cv::Scalar(0, 255, 255), false, true);
		PutCairoTimeOverlay(overlayDeviceProp, cv::Point2d(180, 40), "arial", 15, cv::Scalar(0, 255, 255), false, true);
		//displayImageMat = cv::cvarrToMat(m_CurrentIplImage, true);
		displayImageMat = m_CurrentImage.clone();
		cv::addWeighted(m_OverlayMat, 1, displayImageMat(overlayRect), 0.5, 0.0, m_OverlayFinalMat);
		m_OverlayFinalMat += 0.4 * m_OverlayFinalMat;
		m_OverlayFinalMat.copyTo(displayImageMat(overlayRect));

		if (m_SaveOutput) {
		
			sprintf(outImage, "%s\\frame_%06d.jpg", outFolder, iterIdx);
			//tmpPtr = displayImageMat;
			//cvSaveImage(outImage, &tmpPtr);
			cv::imwrite(outImage, displayImageMat);
		}

		if (m_EnableDisplay) {
			
			cv::imshow("Detections", displayImageMat);
			cvWaitKey(0);
		}

		cvWaitKey(1);
		iterIdx++;
	}

	if (NUM_ITERATIONS > BURN_ITERATIONS) {
	
		float avgSpeed = (float)std::accumulate(timings.begin() + BURN_ITERATIONS, timings.end(), 0.0) / (timings.size() - BURN_ITERATIONS);
		printf("\n****** YOLO DNN Avg Proc Speed {Time, FPS} : {%f, %f} ******\n\n", avgSpeed, 1000 / avgSpeed);
	}

	m_OCLManager->FinalizeFloatArray(yoloNNCurrentState.m_InputRefGpu);
	yoloNNCurrentState.m_InputRefGpu = NULL;

	free(m_InImage->m_DataArray);
	delete m_InImage;
	m_InImage = NULL;
	free(m_ResizedImage->m_DataArray);
	delete m_ResizedImage;
	m_ResizedImage = NULL;
	free(detBBoxes);

	for (int i = 0; i < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; i++)
		free(detProbScores[i]);

	//if (m_CurrentIplImage != NULL) {

		//cvReleaseImage(&m_CurrentIplImage);
		//m_CurrentIplImage = NULL;
	//}

	if (m_CurrentImage.data != NULL)
		m_CurrentImage.release();

	m_OverlayMat.release();
	m_OverlayFinalMat.release();
	cvWaitKey(0);
	cvDestroyAllWindows();
}

void PrintOCLBuffer(OCLBuffer *inBuffer, OCLManager *oclManager, char* fileName, int numItems) {

	
	float *verfArray = (float*)calloc(numItems, sizeof(float));
	oclManager->ReadFloatArray(verfArray, inBuffer, numItems);
	
	std::ofstream myfile(fileName);
	if (myfile.is_open()) {

		for (int count = 0; count < numItems; count++)
			myfile << verfArray[count] << "\n";
		myfile.close();
	}
	free(verfArray);
}

float YOLONeuralNet::PropagateLayerInputsForward(StructYOLODeepNNLayer *inLayer, StructYOLODeepNNState *netState) {

	int m = 0;
	int k = 0;
	int n = 0;
	int size = 0;
	int index = 0;
	int mapIndex = 0;
	int offset = 0;
	int swapIdx = 0;
	int inputSize = 0, arrayLen = 0;
	float *inCpu = NULL; 
	unsigned int outsize = 0;
	
	float timeAccumulator = 0.0f;
	//char debugFileName[256];
	//const auto start_time = std::chrono::steady_clock::now();

	switch (inLayer->m_LayerType) {

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:

			m = inLayer->m_N;
			k = inLayer->m_Size * inLayer->m_Size * inLayer->m_C;
			n = inLayer->m_OutW * inLayer->m_OutH;

			timeAccumulator += m_OCLManager->ConvertImageToColumnArray(netState->m_InputGpu, inLayer->m_C, inLayer->m_H,
				inLayer->m_W, inLayer->m_Size, inLayer->m_Stride, inLayer->m_Pad, netState->m_Workspace);
			
			timeAccumulator += m_OCLManager->ComputeGEMM(false, false, m, n, k, 1.0f, inLayer->m_Weights_Gpu, 0, k,
								netState->m_Workspace, 0, n, 1.0f, inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], 0, n);

			timeAccumulator += m_OCLManager->AddBias(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], inLayer->m_Biases_Gpu, inLayer->m_Batch,
				inLayer->m_N, inLayer->m_OutH * inLayer->m_OutW);

			swapIdx = (netState->m_ConvSwapBufIdx == 0) ? 1 : 0;

			timeAccumulator += m_OCLManager->ActivateInputs(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx],
				inLayer->m_OutputSwapGPUBuffers[swapIdx], inLayer->m_Outputs * inLayer->m_Batch, inLayer->m_Activation);

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

			timeAccumulator += m_OCLManager->FlattenArray(netState->m_InputGpu, inLayer->m_H * inLayer->m_W,
				inLayer->m_N * (inLayer->m_Coords + inLayer->m_Classes + 1), inLayer->m_Batch, 1, inLayer->m_Output_Gpu);

			timeAccumulator += m_OCLManager->SoftMax(inLayer->m_Output_Gpu, inLayer->m_Classes, inLayer->m_Classes + 5,
							inLayer->m_W * inLayer->m_H * inLayer->m_N * inLayer->m_Batch, 1, inLayer->m_Output_Gpu, 5);

#ifndef PINNED_MEM_OUTPUT
			m_OCLManager->ReadFloatArray(inLayer->m_Output, inLayer->m_Output_Gpu, inLayer->m_Batch * inLayer->m_Outputs);
#else
			m_OCLManager->ReadFloatArray(inLayer->m_PinnedOutput, inLayer->m_Output_Gpu, inLayer->m_Batch * inLayer->m_Outputs);
#endif
			
			size = inLayer->m_Coords + inLayer->m_Classes + 1;

			arrayLen = inLayer->m_H * inLayer->m_W * inLayer->m_N;
			
			#pragma omp parallel num_threads(inLayer->m_N)	
			for (int i = 0; i < arrayLen; ++i) {

				index = size * i;
#ifndef PINNED_MEM_OUTPUT
				inLayer->m_Output[index + 4] = LogisticActivate(inLayer->m_Output[index + 4]);
#else
				inLayer->m_PinnedOutput[index + 4] = LogisticActivate(inLayer->m_PinnedOutput[index + 4]);
#endif
			}

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

			n = inLayer->m_OutH * inLayer->m_OutW * inLayer->m_C * inLayer->m_Batch;
			timeAccumulator += m_OCLManager->MaxPool(n, inLayer->m_H, inLayer->m_W, inLayer->m_C, inLayer->m_Stride,
				inLayer->m_Size, inLayer->m_Pad, netState->m_InputGpu, inLayer->m_Output_Gpu);// , inLayer->m_Indexes_Gpu);

			break;

		default:
			break;
	}

	//const auto elapsed_time = std::chrono::steady_clock::now() - start_time;

	//auto timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
	//printf("*******Layer{%d} exec time is %f : Diff : %f\n\n\n", netState->m_LayerIndex, timing, timing - timeAccumulator);

	return timeAccumulator;
}


/*

int fourcc = CV_FOURCC('x', '2', '6', '4');
//int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));
//int fourcc = -1;
cv::VideoWriter outputVideo;
outputVideo.write(displayImageMat);

sprintf(fileName, "%s\\Output.avi", ExePath().c_str());
bool outVideoOpened = outputVideo.open(fileName, -1, 25, cv::Size(m_InImage->m_W, m_InImage->m_H), true);
outputVideo.release();

*/