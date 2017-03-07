#include "detect.h"

#include "feature_extractor.h"

#include "data_io.h"

namespace cascade{

void CascadeDetector::DetectNoGrouping(const cv::Mat& img, const Config& config,
	vector<cv::Rect>& rects){

// 	double scale[] = { 1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144,
// 		0.2097152, 0.16777216, 0.134217728, 0.10737418,
// 		0.0858993, 0.06871947, 0.054975581, 0.04398046,
// 		0.0351843, 0.02814749, 0.022517998, 0.01801439,
// 		0.0144115 };
	double scale[] = { 1, 0.8, 0.64, 0.5, 0.3 };
//	double scale[] = { 1 };
	int scale_num = sizeof(scale) / sizeof(double);

	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

	vector<int> displs(processor_num, 0);
	vector<int> counts(processor_num, 0);
	for (int i = 0; i < processor_num; ++i){
		displs[i] = DivideStart(i, processor_num, scale_num);
		counts[i] = DivideNums(i, processor_num, scale_num);
	}

	vector<cv::Rect> rects_tmp;
	for (int i = displs[processor_rank]; 
		i < displs[processor_rank] + counts[processor_rank]; ++i){

		cv::Mat img_resized;
		cv::resize(img, img_resized, {}, scale[i], scale[i]);
		MatrixXf features;
		MatrixXi locs;
		FeatureExtractor::MonoExtractor(img_resized, config, 8, 8, features, locs);
// 		FeatureExtractor::MonoExtractorWithTrigger(img_resized, config, 
// 			config.block_width(), config.block_height(), config.block_width(),
// 			config.block_height(), 1, 1, 8, 8, features, locs);
		vector<int> answer;
		classifier_->Predict(features, answer);
		for (int j = 0; j < answer.size(); ++j){
			if (answer[j] == 1){
				cv::Rect rect(locs(j, 0) / scale[i], locs(j, 1) / scale[i],
					config.block_width() / scale[i], config.block_height() / scale[i]);
				rects_tmp.push_back(rect);
			}
		}

		//DataIO::MatToTxt(features, "d:\\2.txt");
	}

	int total_num;
	int num = rects_tmp.size();
	MPI_Reduce(&num, &total_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	rects = vector<cv::Rect>(total_num);
	vector<int> recive_counts(processor_num);
	MPI_Gather(&num, 1, MPI_INT, recive_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	vector<int> recive_displs(processor_num, 0);
	for (int i = 0; i < processor_num; ++i){
		recive_counts[i] *= sizeof(cv::Rect);
	}
	for (int i = 1; i < processor_num; ++i){
		recive_displs[i] = recive_displs[i - 1] + recive_counts[i - 1];
	}

	MPI_Gatherv(rects_tmp.data(), rects_tmp.size()*sizeof(cv::Rect), MPI_CHAR, rects.data(),
		recive_counts.data(), recive_displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

}

void CascadeDetector::DetectNoGrouping(const cv::Mat& img, const Config& config, 
	int min_block_width, int min_block_height, int max_block_width, int max_block_heigth,
	int width_step, int height_step, double move_step_ratio, vector<cv::Rect>& rects){


	int size = std::min((max_block_width - min_block_width) / width_step, 
		(max_block_heigth - min_block_height) / height_step);
	size += 1;
	vector<int> hs(size, 0);
	vector<int> ws(size, 0);
	hs[0] = min_block_height;
	ws[0] = min_block_width;
	for (int i = 1; i < size; ++i){
		hs[i] = hs[i - 1] + height_step;
		ws[i] = ws[i - 1] + width_step;
	}

	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
	vector<int> displs(processor_num, 0);
	vector<int> counts(processor_num, 0);
	for (int i = 0; i < processor_num; ++i){
		displs[i] = DivideStart(i, processor_num, size);
		counts[i] = DivideNums(i, processor_num, size);
	}

	vector<cv::Rect> rects_tmp;
	if (counts[processor_rank]>0){
		MatrixXf features;
		MatrixXi rects_mat;
		FeatureExtractor::MonoExtractorWithTrigger(img, config,
			ws[displs[processor_rank]],	hs[displs[processor_rank]], 
			ws[displs[processor_rank]]+counts[processor_rank]-1, 
			hs[displs[processor_rank]]+counts[processor_rank]-1, 
			width_step, height_step, move_step_ratio, features, rects_mat);
		vector<int> preds;
		classifier_->Predict(features, preds);
		for (int i = 0; i < preds.size(); ++i){
			if (preds[i] == 1){
				cv::Rect rect(rects_mat(i, 0), rects_mat(i, 1), 
					rects_mat(i, 2), rects_mat(i, 3));
				rects_tmp.push_back(rect);
			}
		}
	}
	int total_num;
	int num = rects_tmp.size();
	MPI_Reduce(&num, &total_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	rects = vector<cv::Rect>(total_num);
	vector<int> recive_counts(processor_num);
	MPI_Gather(&num, 1, MPI_INT, recive_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	vector<int> recive_displs(processor_num, 0);
	for (int i = 0; i < processor_num; ++i){
		recive_counts[i] *= sizeof(cv::Rect);
	}
	for (int i = 1; i < processor_num; ++i){
		recive_displs[i] = recive_displs[i - 1] + recive_counts[i - 1];
	}

	MPI_Gatherv(rects_tmp.data(), rects_tmp.size()*sizeof(cv::Rect), MPI_CHAR, rects.data(),
		recive_counts.data(), recive_displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

}

CascadeDetector::CascadeDetector(){
	classifier_ = &Cascade::Get();
}

CascadeDetector::CascadeDetector(const string& model_path){
	classifier_ = &Cascade::Get();
	classifier_->LoadModel(model_path);
}

void CascadeDetector::LoadModel(const string& model_path){
	classifier_->LoadModel(model_path);
}

static void MatRoiSet(const cv::Mat& mat, const cv::Rect& roi){
	int loc = roi.y*mat.cols + roi.x;
	for (int i = 0; i < roi.height*roi.width; ++i){
		if (mat.data[loc] < 255){
			mat.data[loc]++;
		}
		++loc;
		if (i>0&&(i%roi.width == 0)){
			loc += mat.cols-roi.width;
		}
	}
}

void CascadeDetector::DetectHeatMap(const cv::Mat& img, 
	const Config& config, cv::Mat& heatmap){

	heatmap = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar::all(0));
	vector<cv::Rect> rects;
	DetectNoGrouping(img, config, rects);
	for each(auto r in rects){
		MatRoiSet(heatmap, r);
	}
}

void CascadeDetector::DetectHeatMap(const cv::Mat& img, const Config& config, 
	int min_block_width, int min_block_height, int max_block_width, 
	int max_block_heigth, int width_step, int height_step, double move_step_ratio, 
	cv::Mat& heatmap){

	heatmap = cv::Mat(img.rows, img.cols, CV_8U, cv::Scalar::all(0));
	vector<cv::Rect> rects;
	DetectNoGrouping(img, config, min_block_width, min_block_height, max_block_width,
		max_block_heigth, width_step, height_step, move_step_ratio, rects);
	for each(auto r in rects){
		MatRoiSet(heatmap, r);
	}
}

void CascadeDetector::Detect(const cv::Mat img, const Config& config, 
	const int thres, vector<cv::Rect>& rects){

	rects.clear();
	cv::Mat heatmap;
	DetectHeatMap(img, config, heatmap);
	cv::Mat tmp;
	threshold(heatmap, tmp, thres, 255, CV_THRESH_BINARY);
	vector<vector<cv::Point>> contours;
	cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for each(auto c in contours){
		cv::Rect rect = cv::boundingRect(c);
		rects.push_back(rect);
	}
}

void CascadeDetector::Detect(const cv::Mat& img, const Config& config, 
	int min_block_width, int min_block_height, int max_block_width, 
	int max_block_heigth, int width_step, int height_step, double move_step_ratio,
	int thres, vector<cv::Rect>& rects){

	rects.clear();
	cv::Mat heatmap;
	DetectHeatMap(img, config, min_block_width, min_block_height, max_block_width,
		max_block_heigth, width_step, height_step, move_step_ratio, heatmap);
	cv::Mat tmp;
	threshold(heatmap, tmp, thres, 255, CV_THRESH_BINARY);
	vector<vector<cv::Point>> contours;
	cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for each(auto c in contours){
		cv::Rect rect = cv::boundingRect(c);
		rects.push_back(rect);
	}
}

}