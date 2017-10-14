
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


#include <stdio.h>
#include "YoloOCLDNN.h"


#ifdef _DEBUG
//#include <vld.h>
#endif


YOLONeuralNet	*m_YOLODeepNNObj;


inline bool FileExists(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}


int main(int argc, char* argv[]) {

	printf("YoloOCLInference Started..\n");

	char	labelsFile[MAX_PATH];
	char	configFile[MAX_PATH];
	char	weightsFile[MAX_PATH];
	string	currentDir = ExePath();

	char	inputImage[MAX_PATH];
	int		enableDisplay = 0;
	int		saveOutput = 0;

	for (int i = 1; i < argc; i++) {

		if (strcmp(argv[i], "-input") == 0) {

			if (++i >= argc) {

				printf("ERROR - Invalid param for %s\n", argv[i - 1]);
				return -1;
			}
			strcpy(inputImage, argv[i]);
		}
		else if (strcmp(argv[i], "-display") == 0) {

			if (++i >= argc || sscanf(argv[i], "%d", &enableDisplay) != 1) {

				printf("ERROR - Invalid param for %s\n", argv[i - 1]);
				return -1;
			}
		}
		else if (strcmp(argv[i], "-save") == 0) {

			if (++i >= argc || sscanf(argv[i], "%d", &saveOutput) != 1) {

				printf("ERROR - Invalid param for %s\n", argv[i - 1]);
				return -1;
			}
		}
	}

	if (!FileExists(inputImage)) {

		printf("ERROR - Input file is not valid. Terminating...\n");
		return -1;
	}

	sprintf(labelsFile, "%s\\coco.names", currentDir.c_str());
	sprintf(configFile, "%s\\tiny-yolo.cfg", currentDir.c_str());
	sprintf(weightsFile, "%s\\tiny-yolo.weights", currentDir.c_str());
	
	m_YOLODeepNNObj = new YOLONeuralNet(labelsFile, configFile, weightsFile, 
		(enableDisplay == 1)?true:false, (saveOutput == 1)?true:false);
	m_YOLODeepNNObj->Initialize();
	m_YOLODeepNNObj->ComputeYOLONNOutput(inputImage);

	m_YOLODeepNNObj->Finalize();
	delete m_YOLODeepNNObj;

	return 0;
}