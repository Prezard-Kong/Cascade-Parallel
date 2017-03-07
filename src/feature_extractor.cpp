#include "feature_extractor.h"

#include <Windows.h>
#include <io.h>
#include <assert.h>

#include <memory>
#include <fstream>

#include "mpi.h"

namespace cascade{

const int MaxFeatNum = 10000;

static vector<string> LoadSample( const string& path ){
	long hFile = 0;
	_finddata_t fileinfo;
	vector<string> ans;
	string tmp(path);
	tmp.append("\\*");
	if ((hFile=_findfirst(tmp.c_str(),&fileinfo))!=-1){
		do{
			if (fileinfo.attrib != _A_SUBDIR){
				tmp = path;
				tmp.append("\\");
				tmp.append(fileinfo.name);
				if (IsImage(tmp)){
					ans.push_back(tmp);
				}
			}
		}while(_findnext(hFile,&fileinfo)==0);
		_findclose(hFile);
	}
	return ans;
}
typedef struct {
	int		num;
	unsigned char* feats;
}FeatureResult;
typedef int (*ProcFunc)(void *, void *, FeatureResult *);

static vector<ProcFunc> LoadFuncs(const Config& pathes){
	vector<ProcFunc> funcs(pathes.func_num());
	for (int i = 0;i<pathes.func_num();++i){
		HINSTANCE hdll = LoadLibraryExA(pathes.func_path(i).c_str(),
			NULL,LOAD_WITH_ALTERED_SEARCH_PATH);
		funcs[i] = (ProcFunc)GetProcAddress(hdll, "process");
		assert(funcs[i]!=NULL);
	}
	return funcs;
}

void FeatureExtractor::NormalExtractorToMat(const Config& pathes,
	MatrixXf& features,	vector<int>& labels){

	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

	vector<string> samples;
	int pos_num=0;
	int neg_num=0;
	for (int i = 0;i<pathes.pos_num();++i){
		vector<string> tmp = LoadSample(pathes.pos_path(i));
		samples.insert(samples.end(),tmp.begin(),tmp.end());
		pos_num += tmp.size();
	}
	for (int i = 0;i<pathes.neg_num();++i){
		vector<string> tmp = LoadSample(pathes.neg_path(i));
		samples.insert(samples.end(),tmp.begin(),tmp.end());
		neg_num += tmp.size();
	}


	//每一列代表一个样本//
	int cols = pos_num+neg_num;
	int rows = pathes.feat_num();
	if (processor_rank == 0){
		features.resize(rows,cols);
	}else{
		features.resize(0,0);
	}
	labels = vector<int>(cols, -1);
	for (int i = 0;i<pos_num;++i){
		labels[i] = 1;
	}
	vector<int> counts(processor_num);
	vector<int> displs(processor_num);
	for (int i = 0;i<processor_num;++i){
		counts[i] = DivideNums(i, processor_num, cols)*rows;
		displs[i] = DivideStart(i,processor_num, cols)*rows;
	}
	MatrixXf local_features(rows,counts[processor_rank]/rows);

	vector<ProcFunc> funcs = LoadFuncs(pathes);

	//为与特征提取函数相适应，特征提取的结果以裸指针形式返回
	//单次特征提取最多返回10000个特征
	//如需修改个数，修改本文件内的MaxFeatNum
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];
	
	int st = displs[processor_rank]/rows;
	int ed = processor_rank<processor_num-1?displs[processor_rank+1]/rows:cols;
	int col = 0;
	for (int i = st;i<ed;++i){
		int row = 0;
		cv::Mat img = cv::imread(samples[i],0);
		for (int j = 0;j<funcs.size();++j){
			funcs[j](NULL,&img,&result);
			float* feats = (float*)result.feats;
			for (int k = 0;k<result.num;++k){
				local_features(row++,col) = feats[k];
			}
		}
		col++;
	}
	delete[] result.feats;

	MPI_Gatherv(local_features.data(),counts[processor_rank],MPI_FLOAT,
		features.data(),counts.data(),displs.data(),MPI_FLOAT,0,MPI_COMM_WORLD);
	//转换为每行一个样本//
	features.transposeInPlace();
}

//采用样本均分的形式，由于每张图像大小不一，可能造成运算负担的不平均//
void FeatureExtractor::HardSampleExtractorToMat( const Config& pathes,MatrixXf& features){
	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

	int block_width = pathes.block_width();
	int block_height = pathes.block_height();
	int step_x = pathes.step_x();
	int step_y = pathes.step_y();


	vector<string> samples;
	for (int i = 0;i<pathes.bak_num();++i){
		vector<string> tmp = LoadSample(pathes.bak_path(i));
		samples.insert(samples.end(),tmp.begin(),tmp.end());
	}
	int sample_num = samples.size();
	vector<int> counts(processor_num);
	vector<int> displs(processor_num);
	for (int i = 0;i<processor_num;++i){
		counts[i] = DivideNums(i, processor_num, sample_num);
		displs[i] = DivideStart(i,processor_num, sample_num);
	}

	//模拟动态矩阵
	vector<float> local_features;

	double scale[] = {1.0,0.8,0.64,0.512,0.4096,0.32768,0.262144,
		0.2097152,0.16777216,0.134217728,0.10737418,
		0.0858993,0.06871947,0.054975581,0.04398046,
		0.0351843,0.02814749,0.022517998,0.01801439,
		0.0144115};
	int scale_num = sizeof(scale)/sizeof(double);

	vector<ProcFunc> funcs = LoadFuncs(pathes);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];

	int st = displs[processor_rank];
	int ed = processor_rank<processor_num-1?displs[processor_rank+1]:sample_num;
	for (int i=st;i<ed;++i){
		cv::Mat img = cv::imread(samples[i],0);
		for (int j = 0;j<scale_num;++j){
			cv::Mat img_resized;
			cv::Size size(img.cols*scale[j], img.rows*scale[j]);
			if (size.width < block_width || size.height < block_height){
				continue;
			}
			cv::resize(img,img_resized,size);
			int safe_area_width = img_resized.cols - block_width;
			int safe_area_height = img_resized.rows - block_height;
			int x = 0;
			int y = 0;
			while (y<safe_area_height){
				cv::Rect roi(x,y,block_width,block_height);
				cv::Mat img_block(img_resized,roi);
				for (int k = 0;k<funcs.size();++k){
					funcs[k](NULL,&img_block,&result);
					float* feats = (float*)result.feats;
					local_features.insert(local_features.end(),
						feats,feats+result.num);
				}
				x += step_x;
				if (x>safe_area_width){
					x = 0;
					y += step_y;
				}
			}
		}
	}
	delete[] result.feats;

	int output_num = local_features.size();  
	vector<int> output_num_vec(processor_num);
	MPI_Gather(&output_num,1,MPI_INT,output_num_vec.data(),1,
		MPI_INT,0,MPI_COMM_WORLD);
	int all_num = 0;
	if (processor_rank == 0){
		for each(int num in output_num_vec){
			all_num += num;
		}
	}

	vector<int> locs(processor_num, 0);
	for (int i = 1;i<processor_num;++i){
		locs[i] = locs[i-1]+output_num_vec[i-1];
	}
	MPI_Bcast(output_num_vec.data(),processor_num,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(locs.data(),processor_num,MPI_INT,0,MPI_COMM_WORLD);
	vector<float> all_features(all_num);
	MPI_Gatherv(local_features.data(),output_num,MPI_FLOAT,all_features.data(),
		output_num_vec.data(),locs.data(),MPI_FLOAT,0,MPI_COMM_WORLD);

	features.resize(pathes.feat_num(),all_num/pathes.feat_num());
	memcpy(features.data(),all_features.data(),all_num*sizeof(float));
	features.transposeInPlace();
}

void FeatureExtractor::ExtractorToMat( const Config& pathes, MatrixXf& sample_features, 
	vector<int>& labels,MatrixXf& hard_sample_features){

	NormalExtractorToMat(pathes,sample_features,labels);
	HardSampleExtractorToMat(pathes,hard_sample_features);
}

void FeatureExtractor::MonoExtractor(const cv::Mat& img_resized, const Config& config, 
	int step_x, int step_y, MatrixXf& features, MatrixXi& locs){

	vector<ProcFunc> funcs = LoadFuncs(config);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];

	vector<float> local_features;
	vector<int> xs;
	vector<int> ys;

	int safe_area_width = img_resized.cols - config.block_width();
	int safe_area_height = img_resized.rows - config.block_height();
	int x = 0;
	int y = 0;
	while (y < safe_area_height){
		xs.push_back(x);
		ys.push_back(y);
		cv::Rect roi(x, y, config.block_width(), config.block_height());
		cv::Mat img_block(img_resized, roi);
		for (int k = 0; k < funcs.size(); ++k){
			funcs[k](NULL, &img_block, &result);
			float* feats = (float*)result.feats;
			local_features.insert(local_features.end(),
				feats, feats + result.num);
		}
		x += step_x;
		if (x > safe_area_width){
			x = 0;
			y += step_y;
		}
	}
	delete[] result.feats;

	features.resize(config.feat_num(), local_features.size() / config.feat_num());
	memcpy(features.data(), local_features.data(), local_features.size()*sizeof(float));
	features.transposeInPlace();
	locs.resize(xs.size(), 2);
	memcpy(locs.data(), xs.data(), xs.size()*sizeof(int));
	memcpy(locs.col(1).data(), ys.data(), ys.size()*sizeof(int));

}

void FeatureExtractor::MonoExtractor(const cv::Mat& img_resized, const Config& config,
	int pt_x, int pt_y, MatrixXf& feature){

	vector<ProcFunc> funcs = LoadFuncs(config);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];

	feature.resize(1, config.feat_num());
	cv::Rect roi(pt_x, pt_y, config.block_width(), config.block_height());
	cv::Mat img_block(img_resized, roi);
	int k = 0;
	for (int i = 0; i < funcs.size(); ++i){
		funcs[i](NULL, &img_block, &result);
		memcpy(feature.col(k).data(), result.feats, result.num*sizeof(float));
		k += result.num;
	}
	delete[] result.feats;
}

static void SetHandle(unsigned char* handle, bool trigger, int pt_x, int pt_y,
	int block_width, int block_height){

	memcpy(handle, &trigger, sizeof(bool));
	memcpy(handle + sizeof(bool), &pt_x, sizeof(int));
	memcpy(handle + sizeof(bool) + sizeof(int), &pt_y, sizeof(int));
	memcpy(handle + sizeof(bool) + 2 * sizeof(int), &block_width, sizeof(int));
	memcpy(handle + sizeof(bool) + 3 * sizeof(int), &block_height, sizeof(int));
}

void FeatureExtractor::MonoExtractorWithTrigger(const cv::Mat& img, 
	const Config& config, int pt_x, int pt_y, int block_width, int block_height, 
	bool trigger, MatrixXf& feature){

	vector<ProcFunc> funcs = LoadFuncs(config);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];

	feature.resize(1, config.feat_num());
	unsigned char* handle = new unsigned char[sizeof(bool) + 4 * sizeof(int)];
	SetHandle(handle, trigger, pt_x, pt_y, block_width, block_height);
	int k = 0;
	for (int i = 0; i < funcs.size(); ++i){
		funcs[i](handle, &const_cast<cv::Mat&>(img) , &result);
		memcpy(feature.col(k).data(), result.feats, result.num*sizeof(float));
		k += result.num;
	}

	delete[] result.feats;
	delete[] handle;
}

void FeatureExtractor::MonoExtractorWithTrigger(const cv::Mat& img, 
	const Config& config, int min_width, int min_height, int max_width, 
	int max_height, int width_step, int height_step, double step_ratio, 
	MatrixXf& features, MatrixXi& rects){

	vector<ProcFunc> funcs = LoadFuncs(config);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];
	unsigned char* handle = new unsigned char[sizeof(bool) + 4 * sizeof(int)];

	vector<float> local_features;
	vector<int> xs; //水平坐标
	vector<int> ys; //竖直坐标
	vector<int> ws;	//矩形框宽度
	vector<int> hs; //矩形框高度

	bool trigger = true;
	for (int h = min_height; h <= max_height; h += height_step){
		for (int w = min_width; w <= max_width; w += width_step){
			int safe_area_width = img.cols - w;
			int safe_area_height = img.rows - h;
			int x = 0;
			int y = 0;
			int step_x = max(step_ratio*w, 1);
			int step_y = max(step_ratio*h, 1);
			while (y < safe_area_height){
				xs.push_back(x);
				ys.push_back(y);
				ws.push_back(w);
				hs.push_back(h);
				SetHandle(handle, trigger, x, y, w, h);
				for (int k = 0; k < funcs.size(); ++k){
					funcs[k](handle, &const_cast<cv::Mat&>(img), &result);
					float* feats = (float*)result.feats;
					local_features.insert(local_features.end(),
						feats, feats + result.num);
				}
				x += step_x;
				if (x > safe_area_width){
					x = 0;
					y += step_y;
				}
				trigger = false;
			}
		}
	}
	delete[] result.feats;
	delete[] handle;


	features.resize(config.feat_num(), local_features.size() / config.feat_num());
	memcpy(features.data(), local_features.data(), local_features.size()*sizeof(float));
	local_features.clear();
	features.transposeInPlace();
	rects.resize(features.rows(), 4);
	memcpy(rects.data(), xs.data(), xs.size()*sizeof(int));
	memcpy(rects.col(1).data(), ys.data(), ys.size()*sizeof(int));
	memcpy(rects.col(2).data(), ws.data(), ws.size()*sizeof(int));
	memcpy(rects.col(3).data(), hs.data(), hs.size()*sizeof(int));

}

void FeatureExtractor::MonoExtractorWithTrigger(const cv::Mat& img, const Config& config, 
	int min_width, int min_height, int max_width, int max_height, int width_step, 
	int height_step, int x_step, int y_step, MatrixXf& features, MatrixXi& rects){

	vector<ProcFunc> funcs = LoadFuncs(config);
	FeatureResult result;
	result.feats = new unsigned char[MaxFeatNum*sizeof(float)];
	unsigned char* handle = new unsigned char[sizeof(bool) + 4 * sizeof(int)];

	vector<float> local_features;
	vector<int> xs; //水平坐标
	vector<int> ys; //竖直坐标
	vector<int> ws;	//矩形框宽度
	vector<int> hs; //矩形框高度

	bool trigger = true;
	for (int h = min_height; h <= max_height; h += height_step){
		for (int w = min_width; w <= max_width; w += width_step){
			int safe_area_width = img.cols - w;
			int safe_area_height = img.rows - h;
			int x = 0;
			int y = 0;
			while (y < safe_area_height){
				xs.push_back(x);
				ys.push_back(y);
				ws.push_back(w);
				hs.push_back(h);
				SetHandle(handle, trigger, x, y, w, h);
				for (int k = 0; k < funcs.size(); ++k){
					funcs[k](handle, &const_cast<cv::Mat&>(img), &result);
					float* feats = (float*)result.feats;
					local_features.insert(local_features.end(),
						feats, feats + result.num);
				}
				x += x_step;
				if (x > safe_area_width){
					x = 0;
					y += y_step;
				}
				trigger = false;
			}
		}
	}
	delete[] result.feats;
	delete[] handle;


	features.resize(config.feat_num(), local_features.size() / config.feat_num());
	memcpy(features.data(), local_features.data(), local_features.size()*sizeof(float));
	local_features.clear();
	features.transposeInPlace();
	rects.resize(features.rows(), 4);
	memcpy(rects.data(), xs.data(), xs.size()*sizeof(int));
	memcpy(rects.col(1).data(), ys.data(), ys.size()*sizeof(int));
	memcpy(rects.col(2).data(), ws.data(), ws.size()*sizeof(int));
	memcpy(rects.col(3).data(), hs.data(), hs.size()*sizeof(int));
}

} //namspace cascade