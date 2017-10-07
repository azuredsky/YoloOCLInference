
//https://www.codeproject.com/Articles/10809/A-Small-Class-to-Read-INI-File
//Check the required licensing in the link above

#pragma once

#ifndef INI_UTILS_H
#define INI_UTILS_H



#include <iostream>
#include <vector>
#include <string>
#include <Windows.h>

class CIniReader
{
public:
	CIniReader(char* szFileName);
	std::vector<std::string> ReadSections();
	int ReadInteger(char* szSection, char* szKey, int iDefaultValue);
	float ReadFloat(char* szSection, char* szKey, float fltDefaultValue);
	bool ReadBoolean(char* szSection, char* szKey, bool bolDefaultValue);
	void ReadString(char* szSection, char* szKey, const char* szDefaultValue, char* outValue);
private:
	char m_szFileName[255];
	std::vector<std::string> m_Sections;
};

class CIniWriter
{
public:
	CIniWriter(char* szFileName);
	void WriteInteger(char* szSection, char* szKey, int iValue);
	void WriteFloat(char* szSection, char* szKey, float fltValue);
	void WriteBoolean(char* szSection, char* szKey, bool bolValue);
	void WriteString(char* szSection, char* szKey, char* szValue);
private:
	char m_szFileName[255];
};

#endif
