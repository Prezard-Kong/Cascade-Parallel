#include "data_io.h"

#include <io.h>

#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace cascade{

using std::cout;
using std::cin;
using std::endl;
using std::ofstream;
using std::ifstream;

const int HardExamplesPerFile = 1e6;
const bool RandomOutput = 1;

static bool CheckLoad(const string& path){
	if (_access(path.c_str(),4)==-1){
		cout<<"file does not have load permission, ";
		cout<<"or file does not exist."<<endl;
		cout<<"please check and procedure will exit."<<endl;
		return false;
	}
	return true;
}
static bool CheckWrite(const string& path){
	if (_access(path.c_str(), 0) != -1){
		cout << "file " << path << " exists" << endl;
		cout << "press ENTER if you want to overwrite it." << endl;
		cout << "press any other key, procedure will exit" << endl;
		if (cin.get() != '\n'){
			return false;
		}
		if (_access(path.c_str(), 2) == -1){
			cout << "file does not have write permission, procedure will exit." << endl;
			return false;
		}
	}
	return true;
}
static bool CheckTxt(const string& path){
	int idx = path.find_last_of('.');
	string tmp(&path[idx+1]);
	if (tmp != "txt"){
		cout<<"path is not txt file, please check and procedure will exit."<<endl;
		return false;
	}
	return true;
}
static void ToTxt(const MatrixXf& mat, const string& path){
	ofstream fout(path);
	fout.setf(std::ios::fixed);
	fout.precision(6);
	for (int i = 0;i<mat.rows();++i){
		for (int j = 0;j<mat.cols();++j){
			fout<<mat(i,j);
			if (j == mat.cols()-1){
				fout<<endl;
			}else{
				fout<<"\t";
			}
		}
	}
	fout.close();
}
int CountColAndBackToBegin(ifstream& fin){
	float tmp;
	int ans = 0;
	while(1){
		fin>>tmp;
		ans++;
		if (fin.get()=='\n'){
			break;
		}
	}
	fin.clear();
	fin.seekg(std::ios::beg);
	return ans;
}
int CountRowAndBackToBegin(ifstream& fin){
	string tmp;
	int num = 0;
	while(!fin.eof()){
		std::getline(fin,tmp);
		num++;
		if (fin.eof()||fin.get()=='\n'){
			break;
		}
	}
	fin.clear();
	fin.seekg(std::ios::beg);
	return num;
}
static void FromTxt(const string& path, MatrixXf& mat){
	ifstream fin(path);
	int cols = CountColAndBackToBegin(fin);
	int rows = CountRowAndBackToBegin(fin);
	vector<float> tmp(cols);
	int col = 0;
	int row = 0;
	mat.resize(cols,rows);
	for (int j = 0;j<rows;++j){
		for (int i = 0;i<cols;++i){
			fin>>tmp[i];
		}
		memcpy(mat.col(col++).data(),tmp.data(),cols*sizeof(float));
	}
	mat.transposeInPlace();
	fin.close();
}
bool DataIO::MatToTxt(const MatrixXf& mat, const string& save_path){
	if (!CheckTxt(save_path)){
		return false;
	}
	if (!CheckWrite(save_path)){
		return false;
	}
	ToTxt(mat,save_path);
	return true;	
}
bool DataIO::TxtToMat(const string& load_path, MatrixXf& mat){
	if (!CheckTxt(load_path)){
		return false;
	}
	if (!CheckLoad(load_path)){
		return false;
	}
	FromTxt(load_path,mat);
	return true;
}
bool DataIO::VecToTxt(const vector<int>& vec, const string& save_path){
	if (!CheckTxt(save_path)){
		return false;
	}
	if (!CheckWrite(save_path)){
		return false;
	}
	ofstream fout(save_path);
	for each(int v in vec){
		fout<<v<<"\t";
	}
	fout.close();
	return true;
}
bool DataIO::TxtToVec(const string& load_path, vector<int>& vec){
	if (!CheckTxt(load_path)){
		return false;
	}
	if (!CheckLoad(load_path)){
		return false;
	}
	ifstream fin(load_path);
	vec.clear();
	int tmp;
	while(!fin.eof()){
		fin>>tmp;
		if (fin.eof()){
			break;
		}
		vec.push_back(tmp);
	}
	fin.close();
	return true;
}

bool DataIO::LoadHardExampleVec( const string& load_path, int row, vector<float>& vec ){
	if (!CheckTxt(load_path)||!CheckLoad(load_path)){
		return false;
	}
	vec.clear();
	ifstream fin(load_path);
	string aaa;
	for (int i = 0;i<row;++i){
		getline(fin,aaa);
		fin.clear();
	}
	float tmp;
	while(!fin.eof()){
		fin>>tmp;
		vec.push_back(tmp);
		if(fin.get()=='\n'||fin.eof()){
			break;
		}

	}
	fin.close();
	return true;
}

int DataIO::CountRows( const string& load_path ){
	ifstream fin(load_path);
	int n = CountRowAndBackToBegin(fin);
	fin.close();
	return n;
}

bool DataIO::DivideHardExamples(const string& origin_path, const string& save_path){
	if (!CheckTxt(origin_path) || !CheckLoad(origin_path)){
		return false;
	}

	bool flag = true;
	bool check = true;
	int n = 0;
	ifstream fin(origin_path);
	int cols = CountColAndBackToBegin(fin);
	while (!fin.eof()){
		MatrixXf features(cols, HardExamplesPerFile);
		int i = 0;
		for (; i < HardExamplesPerFile; ++i){
			vector<float> tmp(cols);
			for (int j = 0; j < cols; ++j){
				fin >> tmp[j];
			}
			memcpy(features.col(i).data(), tmp.data(), cols*sizeof(float));
			if (fin.eof()){
				break;
			}
		}
		if (i < HardExamplesPerFile){
			MatrixXf features_tmp(cols, i);
			memcpy(features_tmp.data(), features.data(), cols*i*sizeof(float));
			features.resize(cols, i);
			memcpy(features.data(), features_tmp.data(), cols*i*sizeof(float));
		}

		std::stringstream ss;
		ss << save_path << "\\" << n++ << ".txt";
		string path;
		ss >> path;
		if (flag)
		{
			if (!CheckWrite(path)){
				return false;
			}
			if (check){
				cout << "do you want to ignore the warning ";
				cout << "and overwrite duplicate files automatically?";
				cout << " y/N:";
				char a;
				cin >> a;
				if (a == 'y' || a == 'Y'){
					flag = false;
				}
				else{
					check = false;
				}
			}
		}

		features.transposeInPlace();
		ToTxt(features, path);
		if (fin.eof()){
			break;
		}
	}
	return true;
}

bool DataIO::SaveHardExamplesToTxt(const MatrixXf& features, const string& path, 
	int start_num /*= 0*/){

	int rows = features.rows();
	int cols = features.cols();
	vector<int> index(features.rows());
	for (int i = 0; i < index.size(); ++i){
		index[i] = i;
	}
	//ÊÇ·ñÂÒÐòÊä³ö//
	if (RandomOutput){
		std::random_shuffle(index.begin(), index.end());
	}

	bool flag = true;
	bool check = true;
	for (int i = 0; i < rows / HardExamplesPerFile; ++i){
		std::stringstream ss;
		ss << path << "\\" << start_num++ << ".txt";
		string name;
		ss >> name;
		if (flag)
		{
			if (!CheckWrite(name)){
				return false;
			}
			if (check){
				cout << "do you want to ignore the warning ";
				cout << "and overwrite duplicate files automatically?";
				cout << " y/N:";
				char a;
				cin >> a;
				cin.get();
				if (a == 'y' || a == 'Y'){
					flag = false;
				}
				else{
					check = false;
				}
			}
		}
		ofstream fout(name);
		fout.setf(std::ios::fixed);
		fout.precision(6);
		for (int k = i*HardExamplesPerFile; k<(i+1)*HardExamplesPerFile; ++k){
			for (int j = 0; j < cols; ++j){
				fout << features(index[k], j);
				if (j == cols - 1){
					fout << endl;
				}
				else{
					fout << "\t";
				}
			}
		}
		fout.close();
	}

	if (rows%HardExamplesPerFile){
		std::stringstream ss;
		ss << path << "\\" << start_num++ << ".txt";
		string name;
		ss >> name;
		if (flag)
		{
			if (!CheckWrite(name)){
				return false;
			}
			if (check){
				cout << "do you want to ignore the warning ";
				cout << "and overwrite duplicate files automatically?";
				cout << " y/N:";
				char a;
				cin >> a;
				cin.get();
				if (a == 'y' || a == 'Y'){
					flag = false;
				}
				else{
					check = false;
				}
			}
		}
		ofstream fout(name);
		fout.setf(std::ios::fixed);
		fout.precision(6);
		for (int i = rows / HardExamplesPerFile*HardExamplesPerFile; i < rows; ++i){
			for (int j = 0; j < cols; ++j){
				fout << features(index[i], j);
				if (j == cols - 1){
					fout << endl;
				}
				else{
					fout << "\t";
				}
			}
		}
		fout.close();
	}
	return true;
}

vector<string> DataIO::HardExampleFiles(const string& path){

	if (_access(path.c_str(), 0) == -1){
		return vector<string>(0);
	}

	int n = 0;
	vector<string> ans;
	while (1){
		std::stringstream ss;
		ss << path << "\\" << n++ << ".txt";
		string name;
		ss >> name;
		if (_access(name.c_str(), 0) == -1){
			break;
		}
		ans.push_back(name);
	}
	n = 0;
	while (1){
		std::stringstream ss;
		ss << path << "\\" << n++ << ".byte";
		string name;
		ss >> name;
		if (_access(name.c_str(), 0) == -1){
			break;
		}
		ans.push_back(name);
	}
	return ans;
}

bool DataIO::MatToBin(const MatrixXf& mat, const string& save_path){
	if (!CheckWrite(save_path)){
		return false;
	}
	remove(save_path.c_str());
	FILE* fp = fopen(save_path.c_str(), "wb");
	int rows = mat.rows();
	int cols = mat.cols();
	fwrite(&rows, sizeof(int), 1, fp);
	fwrite(&cols, sizeof(int), 1, fp);
	fwrite(mat.data(), sizeof(float), rows*cols, fp);
	fclose(fp);
	return true;
}

bool DataIO::BinToMat(const string& load_path, MatrixXf& mat){
	if (!CheckLoad(load_path)){
		return false;
	}
	FILE* fp = fopen(load_path.c_str(), "rb");
	int rows, cols;
	fread(&rows, sizeof(int), 1, fp);
	fread(&cols, sizeof(int), 1, fp);
	mat.resize(rows, cols);
	fread(mat.data(), sizeof(float), rows*cols, fp);
	fclose(fp);
	return false;
}

bool DataIO::VecToBin(const vector<int>& vec, const string& save_path){
	if (!CheckWrite(save_path)){
		return false;
	}
	remove(save_path.c_str());
	FILE* fp = fopen(save_path.c_str(), "wb");
	int size = vec.size();
	fwrite(&size, sizeof(int), 1, fp);
	fwrite(vec.data(), sizeof(int), size, fp);
	fclose(fp);
	return true;
}

bool DataIO::BinToVec(const string& load_path, vector<int>& vec){
	if (!CheckLoad(load_path)){
		return false;
	}
	FILE* fp = fopen(load_path.c_str(), "rb");
	int size;
	fread(&size, sizeof(int), 1, fp);
	vec = vector<int>(size);
	fread(vec.data(), sizeof(int), size, fp);
	fclose(fp);
	return true;
}

bool DataIO::SaveHardExamplesToBin(const MatrixXf& features, const string& path,
	int start_num /*= 0*/){

	int rows = features.rows();
	int cols = features.cols();
	vector<int> index(features.rows());
	for (int i = 0; i < index.size(); ++i){
		index[i] = i;
	}
	//ÊÇ·ñÂÒÐòÊä³ö//
	if (RandomOutput){
		std::random_shuffle(index.begin(), index.end());
	}

	bool flag = true;
	bool check = true;
	for (int i = 0; i < rows / HardExamplesPerFile; ++i){
		std::stringstream ss;
		ss << path << "\\" << start_num++ << ".byte";
		string name;
		ss >> name;
		if (flag)
		{
			if (!CheckWrite(name)){
				return false;
			}
			if (check){
				cout << "do you want to ignore the warning ";
				cout << "and overwrite duplicate files automatically?";
				cout << " y/N:";
				char a;
				cin >> a;
				cin.get();
				if (a == 'y' || a == 'Y'){
					flag = false;
				}
				else{
					check = false;
				}
			}
		}
		remove(name.c_str());
		FILE* fp = fopen(name.c_str(), "wb");
		fwrite(&HardExamplesPerFile, sizeof(int), 1, fp);
		fwrite(&cols, sizeof(int), 1, fp);
		for (int j = 0; j < cols; ++j){
			vector<float> feat_tmp(HardExamplesPerFile);
			int n = 0;
			for (int k = i*HardExamplesPerFile; k < (i + 1)*HardExamplesPerFile; ++k){
				feat_tmp[n++] = features(index[k], j);
			}
			fwrite(feat_tmp.data(), sizeof(float), HardExamplesPerFile, fp);
		}
		fclose(fp);
	}

	if (rows%HardExamplesPerFile){
		std::stringstream ss;
		ss << path << "\\" << start_num++ << ".byte";
		string name;
		ss >> name;
		if (flag)
		{
			if (!CheckWrite(name)){
				return false;
			}
			if (check){
				cout << "do you want to ignore the warning ";
				cout << "and overwrite duplicate files automatically?";
				cout << " y/N:";
				char a;
				cin >> a;
				cin.get();
				if (a == 'y' || a == 'Y'){
					flag = false;
				}
				else{
					check = false;
				}
			}
		}
		remove(name.c_str());
		FILE* fp = fopen(name.c_str(), "wb");
		int row_to_save = rows%HardExamplesPerFile;
		fwrite(&row_to_save, sizeof(int), 1, fp);
		fwrite(&cols, sizeof(int), 1, fp);
		for (int j = 0; j < cols; ++j){
			vector<float> feat_tmp(row_to_save);
			int n = 0;
			for (int i = rows / HardExamplesPerFile*HardExamplesPerFile; i < rows; ++i){
				feat_tmp[n++] = features(index[i], j);
			}
			fwrite(feat_tmp.data(), sizeof(float), row_to_save, fp);
		}
		fclose(fp);
	}
	return true;
}

} //namespace cascade