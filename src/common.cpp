#include "common.h"

#include <io.h>
#include <stdio.h>
#include <Windows.h>

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <map>

#include "mpi.h"

#include "data_io.h"

#define DEBUG 0
namespace cascade{

using std::cout;
using std::cin;
using std::endl;

const float HardFileUseTimes = 1;

shared_ptr<Cascade> Cascade::instance_;

Cascade& Cascade::Get(){
	if (!instance_.get()){
		instance_.reset(new Cascade);
	}
	return *(instance_.get());
}

void Cascade::LoadBreak(const string& path,int layers){
	model_.clear();
	tinyxml2::XMLDocument doc;
	doc.LoadFile(path.c_str());
	tinyxml2::XMLElement* root = doc.RootElement();
	tinyxml2::XMLElement* sub = root->FirstChildElement();
	tinyxml2::XMLUtil util;
	int num;
	util.ToInt(sub->GetText(),&num);
	sub = sub->NextSiblingElement();
	string type = sub->GetText();
	for (int i = 0;i<layers;++i){
		sub = sub->NextSiblingElement();
		model_.push_back(shared_ptr<Boost>(new Boost(type)));
		model_[i]->LoadModel(sub);
	}
}

void Cascade::SaveHead( const string& path, const string& type ){
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* root = doc.NewElement("cascade");
	tinyxml2::XMLElement* sub = doc.NewElement("num");
	sub->SetText(0);
	root->LinkEndChild(sub);
	sub = doc.NewElement("type");
	sub->SetText(type.c_str());
	root->LinkEndChild(sub);
	sub = doc.NewElement("boosts");
	root->LinkEndChild(sub);
	doc.LinkEndChild(root);
	doc.SaveFile(path.c_str());
}

void Cascade::SaveBreak( const string& path ){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(path.c_str());
	tinyxml2::XMLElement* root = doc.RootElement();
	tinyxml2::XMLElement* sub = root->FirstChildElement();
	int num;
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&num);
	num++;
	sub->SetText(num);
	sub = sub->NextSiblingElement()->NextSiblingElement();
	tinyxml2::XMLElement* sub1 = model_[model_.size()-1]->SaveModel(doc);
	sub->LinkEndChild(sub1);
	doc.SaveFile(path.c_str());
}

class Param{
public:
	MatrixXf* mean_;
	MatrixXf* std_;
	std::map<string, float>* config_;
	Param(MatrixXf& mean, MatrixXf& std, std::map<string, float>& config)
		:mean_(&mean), std_(&std), config_(&config){};
	Param() :mean_(NULL), std_(NULL), config_(NULL){};

	static void SaveParam(const string& path, const Param& param);

	static void LoadParam(const string& path, Param& param);
};


int PredictHardExample(const Cascade& model,MatrixXf &feat){
	vector<int> pred(1, 1);
	model.PredictForTrain(feat,pred);
	return pred[0];
}

static void LoadExamples(const string& path, MatrixXf& mat){
	int idx = path.find_last_of('.');
	string tmp(&path[idx + 1]);
	if (tmp == "txt"){
		DataIO::TxtToMat(path, mat);
	}else{
		DataIO::BinToMat(path, mat);
	}
}

static void LoadExamples(const string& path, vector<int>& vec){
	int idx = path.find_last_of('.');
	string tmp(&path[idx + 1]);
	if (tmp == "txt"){
		DataIO::TxtToVec(path, vec);
	}
	else{
		DataIO::BinToVec(path, vec);
	}
}

static void CalcMeanAndStd(const MatrixXf& features, MatrixXf& mean, MatrixXf& std){
	mean = features.colwise().mean();
	std = (features.rowwise() - Eigen::RowVectorXf(mean)).array()
		.pow(2.0).colwise().mean().cwiseSqrt();	
}

//rowwise:一行代表一个样本//
static void Normalize(MatrixXf& features, MatrixXf& mean, MatrixXf& std, 
	bool rowwise=true){

	if (rowwise)
		features = (features.rowwise() - Eigen::RowVectorXf(mean)).array().rowwise() 
			/ Eigen::RowVectorXf(std).array();
	else
		features = (features.colwise() - Eigen::RowVectorXf(mean).transpose())
			.array().colwise() / Eigen::RowVectorXf(std).transpose().array();
}

static bool ReplaceHardExample(const Cascade& model, MatrixXf& features,
	const vector<int>& labels, vector<int>& pred_labels, const string& hard_path,
	int& total_nums, const Param& param){

	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
	vector<int> correct_negtive_loc;
	for (int i = 0; i < pred_labels.size(); ++i){
		if (labels[i] == -1 && pred_labels[i] == -1){
			correct_negtive_loc.push_back(i);
		}
	}

	bool need_normalize = param.config_->find("need_normalize")->second;

	//if (processor_rank==0) cout<<"num: "<<correct_negtive_loc.size()<<endl;
#if 1
	MPI_Barrier(MPI_COMM_WORLD);
	if (processor_rank == 0){
		cout << "examples num: "<<correct_negtive_loc.size() << endl;
	}
	int nn = 0;
#endif

	int cols = features.cols();
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	MatrixXf total_features;
	if (processor_rank == 0){
		//DataIO::TxtToMat(hard_path, total_features);
		LoadExamples(hard_path, total_features);
	}
	int total_features_rows = total_features.rows();
	MPI_Bcast(&total_features_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	vector<int> feature_counts(processor_num);
	vector<int> feature_displs(processor_num);
	for (int i = 0; i < processor_num; ++i){
		feature_counts[i] = DivideNums(i, processor_num, total_features_rows)*cols;
		feature_displs[i] = DivideStart(i,processor_num,total_features_rows)*cols;
	}

	total_features.transposeInPlace();
	MatrixXf local_features(cols, feature_counts[processor_rank] / cols);
	MPI_Scatterv(total_features.data(), feature_counts.data(), feature_displs.data(), MPI_FLOAT,
		local_features.data(), feature_counts[processor_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
	total_features.resize(0, 0);


	vector<int> counts(processor_num);
	vector<int> displs(processor_num);
	for (int i = 0; i < processor_num; ++i){
		counts[i] = DivideNums(i, processor_num, correct_negtive_loc.size())*cols;
		displs[i] = DivideStart(i, processor_num, correct_negtive_loc.size())*cols;
	}

	vector<int> replaced(correct_negtive_loc.size(), 0);
	MatrixXf totals(cols, correct_negtive_loc.size());
	MatrixXf locals(cols, counts[processor_rank] / cols);
	int nums = 0;
	std::random_device rd;
	//std::mt19937_64 mt(rd());
	//std::uniform_int_distribution<int> dist(0, local_features.cols()-1);
	
#if DEBUG	
	cout << processor_rank << " total nums: " << counts[processor_rank] / cols << endl;
	std::stringstream ss;
	ss << "D:\\data\\tmp\\debug_log\\" << processor_rank << ".txt";
	string name;
	ss >> name;
	std::ofstream fout(name);

	MPI_Barrier(MPI_COMM_WORLD);
	cout << processor_rank << " begin" << endl;
#endif

	int total_rows = local_features.cols()*HardFileUseTimes;
	for (int i = 0; i<counts[processor_rank] / cols; ++i){
		int ans = -1;
		MatrixXf tmp(1, cols);
		int random_row;
#if DEBUG
		if (i == counts[processor_rank] / cols - 1){
			cout << processor_rank << "." << endl;
		}
#endif
		while (true){
#if DEBUG
			if (counts[processor_rank] / cols % 2 && i == counts[processor_rank] - 1){
				cout << processor_rank << "*" << endl;
			}
#endif
			random_row = rd() % local_features.cols();
			//random_row = mt()%local_features.cols();
			memcpy(tmp.data(), local_features.col(random_row).data(), cols*sizeof(float));
			if (need_normalize){
				Normalize(tmp, *(param.mean_), *(param.std_));
			}
			nums++;
			ans = PredictHardExample(model, tmp);
			if (ans == 1 || nums>total_rows){
				break;
			}
		}
#if DEBUG
		if (i == counts[processor_rank] / cols - 1){
			cout << processor_rank << ".." << endl;
		}
#endif
		if (ans == 1){
			memcpy(locals.col(i).data(), tmp.data(), cols*sizeof(float));
			replaced[displs[processor_rank] / cols + i] = 1;
		}
#if DEBUG
		nn++;
		fout << i << "\t" << random_row << endl;
// 		if (nn % 100 == 0){
// 			cout << processor_rank << ": 100 examples has been replaced" << endl;
// 		}
#endif
	}

#if DEBUG 
	fout.close();
	cout << processor_rank << ": " << nn << "examples replaced" << endl;
	cout.flush();
	MPI_Barrier(MPI_COMM_WORLD);
	// 	if (processor_rank == 0){
	// 		cout << "all replacement done" << endl;
	// 	}
	cout << processor_rank << " all done\t" << nums << " consumed" << endl;
#endif

	int flag = nums > total_rows;
	int total_flag;
	MPI_Allreduce(&flag, &total_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	int total_nums_tmp;
	MPI_Allreduce(&nums, &total_nums_tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	total_nums += total_nums_tmp;
	vector<int> replaced_total(replaced.size());
	MPI_Reduce(replaced.data(), replaced_total.data(), replaced.size(), 
		MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Gatherv(locals.data(), counts[processor_rank], MPI_FLOAT, totals.data(),
		counts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
	features.transposeInPlace();
	if (processor_rank == 0){
		for (int i = 0; i < correct_negtive_loc.size(); ++i){
			if (replaced_total[i]){
				memcpy(features.col(correct_negtive_loc[i]).data(),
					totals.col(i).data(), cols*sizeof(float));
				pred_labels[correct_negtive_loc[i]] = 1;
			}
		}
	}
	MPI_Bcast(pred_labels.data(), pred_labels.size(), MPI_INT, 0, MPI_COMM_WORLD);
	features.transposeInPlace();
	return total_nums < total_features_rows*HardFileUseTimes;
}


static void HardWarning(string& path){
	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

	char tmp[1000];
	if (processor_rank == 0){
		cout<<"The examples in \""<<path<<"\" has been all used."<<endl;
		cout << "please input new hard examples path or input exit:" << endl;
		cin.get();
		//cin.getline(tmp, 1000);
		//cin.get();
		cin.get(tmp, 1000, '\n');
	}
	cin.clear();
	//MPI_Bcast(const_cast<char*>(path.c_str()),path.size()+1,MPI_CHAR,0,MPI_COMM_WORLD);
	MPI_Bcast(tmp, 1000, MPI_CHAR, 0, MPI_COMM_WORLD);
	path = string(tmp);
}

static void CountNegLive(const vector<int>& labels, const vector<int>& pred_labels,
	vector<int>& neg_live){

	for (int i = 0; i < labels.size(); ++i){
		if (labels[i] == -1){
			if (pred_labels[i] == 1){
				neg_live[i]++;
			}
			else{
				neg_live[i] = 0;
			}
		}
	}
}

void Cascade::Train( const string& features_path, const string& labels_path, 
	const string& hard_path,const Config& config, const string& model_path){
	
	//MPI_Init(NULL,NULL);
	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
	MatrixXf features;

	//cout << processor_rank << ": done" << endl;
	int cancel = 0;
	if (processor_rank==0){
		if (_access(model_path.c_str(),0)!=-1){
			cout << model_path << " exists";
			cout << " do you want to overwrite it? y/N:";
			char a;
			//cin >> a;
			cin.get(a);
			if (a == 'y' || a == 'Y'){
				remove(model_path.c_str());
			}
			else{
				cout << "training cancel" << endl;
				cancel = 1;
			}
		}
		//DataIO::TxtToMat(features_path, features);
		LoadExamples(features_path, features);
	}
	MPI_Bcast(&cancel, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (cancel){
		return;
	}
	
	vector<int> labels;
	//DataIO::TxtToVec(labels_path,labels);
	LoadExamples(labels_path,labels);
	
#if DEBUG
	int positive_num = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	if (processor_rank == 0){
		for each (int a in labels){
			if (a == 1) positive_num++;
		}
		cout << "total: " << labels.size() << endl;
		cout << "pos: " << positive_num << endl;
		cout << "neg: " << labels.size() - positive_num << endl;
	}
#endif

	if (config.start_layer()>1){
		LoadBreak(model_path, config.start_layer());
	}

	int sample_nums = features.rows();
	MPI_Bcast(&sample_nums, 1, MPI_INT, 0, MPI_COMM_WORLD);

	vector<string> hard = DataIO::HardExampleFiles(hard_path);
	int hard_file_id = 0;
	if (processor_rank == 0){
		SaveHead(model_path, config.type());
	}
	int total_nums = 0;

	//统计负样本存活层数//
	vector<int> neg_live(labels.size(), 0);

	std::map<string, float> weaker_config;
	config.GetWeakerConfig(weaker_config);

	//归一化//
	string param_path(model_path);
	param_path.insert(param_path.find_last_of('.'), "_params");
	MatrixXf mean, std;
	Param param(mean, std, weaker_config);
	bool need_normalize = weaker_config.find("need_normalize")->second;
	if (need_normalize){
		mean.resize(1, config.feat_num());
		std.resize(1, config.feat_num());
		if (processor_rank == 0){
			if (config.start_layer() == 1){
				CalcMeanAndStd(features, mean, std);
				Normalize(features, mean, std);
			}
			else{
				Param::LoadParam(param_path, param);
			}
		}
		MPI_Bcast(mean.data(), config.feat_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(std.data(), config.feat_num(), MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
	if (processor_rank == 0){
		if (config.start_layer() == 1){
			if (_access(param_path.c_str(), 0) != -1){
				remove(param_path.c_str());
			}
			Param::SaveParam(param_path, param);
		}
	}

	for (int i = config.start_layer()-1;i<config.max_layer_num();++i){
		if (processor_rank == 0){
			cout << "training the " << i+1 << "th layer..." << endl;
		}
		model_.push_back(shared_ptr<Boost>(new Boost(config.type())));
		model_[i]->Train(features, labels, config.false_pos_rate(i),
			config.recall(i), config.max_weaker_num(), weaker_config);

		if (processor_rank == 0){
			SaveBreak(model_path);
#if 1
			cout << "recall: " << model_[i]->recall() << "\t false alarm: " << model_[i]->false_pos_rate() << endl;
#endif
		}

		if (i == config.max_layer_num()-1){
			break;
		}
		vector<int> pred_labels(sample_nums, 1);
		if (processor_rank == 0){
			PredictForTrain(features, pred_labels);
			CountNegLive(labels, pred_labels, neg_live);
			for (int i = 0; i<neg_live.size(); ++i){
				if (neg_live[i]>config.max_neg_live()){
					pred_labels[i] = -1;
				}
			}
		}
		MPI_Bcast(pred_labels.data(), sample_nums, MPI_INT, 0, MPI_COMM_WORLD);

		if (processor_rank == 0){
			cout << "replacing hard examples..." << endl;
		}
		bool flag = false; //是否替换完成//
		string hard_path_unconst(hard_path);
		while(!flag){
			flag = ReplaceHardExample(*this,features,labels,pred_labels,
				hard[hard_file_id], total_nums, param);
			if (!flag){
				if (processor_rank == 0){
					cout << hard_file_id+1<<"th file exhausted" << endl;
				}
				++hard_file_id;
				total_nums = 0;
			}
			while (hard_file_id>=hard.size()){
				MPI_Barrier(MPI_COMM_WORLD);
				HardWarning(hard_path_unconst);
				if (hard_path_unconst == "exit"){
					//MPI_Finalize();
					return;
				}
				else{
					hard = DataIO::HardExampleFiles(hard_path_unconst);
					if (processor_rank == 0){
						cout << "file num: " << hard.size() << endl;
						if (hard.size() > 0){
							cout << "continue replacing..." << endl;
						}
					}
					hard_file_id = 0;
				}
			}
		}
		if (processor_rank == 0){
			cout << "done" << endl;
		}
	}

	//MPI_Finalize();
}

void Cascade::Predict(const MatrixXf& features, vector<int>& anwser) const{
	static MatrixXf mean, std;
	static std::map<string, float> config;
	static shared_ptr<Param> param;
	if (!param.get()){
		param.reset(new Param(mean, std, config));
		Param::LoadParam(param_path_, *(param.get()));
	}
	
	MatrixXf feats(features);
	if (param->config_->find("need_normalize")->second){
		Normalize(feats, *(param->mean_), *(param->std_));
	}
	PredictForTrain(feats, anwser);
}

void Cascade::PredictForTrain( const MatrixXf& features, vector<int>& anwser ) const{

	int rows = features.rows();
	anwser = vector<int>(rows, 1);
	for (int i = 0;i<model_.size();++i){
		vector<int> tmp;
		model_[i]->Predict(features,tmp);
		for (int j = 0;j<rows;++j){
			if (tmp[j]==-1){
				anwser[j]=-1;
			}
		}
	}
}

void Cascade::SaveModel( const string& path ){
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* root = doc.NewElement("cascade");
	tinyxml2::XMLElement* sub = doc.NewElement("num");
	sub->SetText((int)model_.size());
	root->LinkEndChild(sub);
	sub = doc.NewElement("type");
	sub->SetText(model_[0]->type().c_str());
	root->LinkEndChild(sub);
	sub = doc.NewElement("boosts");
	for (int i = 0;i<model_.size();++i){
		tinyxml2::XMLElement* sub1 = model_[i]->SaveModel(doc);
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	doc.LinkEndChild(root);
	doc.SaveFile(path.c_str());
}

void Cascade::LoadModel( const string& path ){
	model_.clear();
	tinyxml2::XMLDocument doc;
	doc.LoadFile(path.c_str());
	tinyxml2::XMLElement* sub = doc.RootElement()->FirstChildElement();
	int num;
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&num);
	sub = sub->NextSiblingElement();
	string type(sub->GetText());
	sub = sub->NextSiblingElement()->FirstChildElement();
	for (int i = 0;i<num;++i){
		model_.push_back(shared_ptr<Boost>(new Boost(type)));
		model_[i]->LoadModel(sub);
		if (i<num-1){
			sub = sub->NextSiblingElement();
		}
	}

	param_path_ = string(path);
	param_path_.insert(param_path_.find_last_of('.'), "_params");
}



//用于将均值和方差向量转换为string格式保存//
static inline void ToString(const MatrixXf& vec, string& str){
	std::stringstream ss;
	for (int i = 0; i < vec.cols(); ++i){
		ss << vec(0, i) << " ";
	}
	str = ss.str();
}

static inline void ToMat(const string& str, MatrixXf& vec){
	std::stringstream ss;
	ss << str;
	vector<float> tmp;
	while (!ss.eof()){
		float val;
		ss >> val;
		if (ss.eof())
			break;
		tmp.push_back(val);
	}
	vec.resize(1, tmp.size());
	memcpy(vec.data(), tmp.data(), tmp.size()*sizeof(float));
}

void Param::SaveParam(const string& path, const Param& param){
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLElement* root = doc.NewElement("params");
	tinyxml2::XMLElement* sub = doc.NewElement("need_normalize");
	bool flag = param.config_->find("need_normalize")->second;
	sub->SetText(flag);
	if (flag){
		tinyxml2::XMLElement* sub1 = doc.NewElement("mean");
		string tmp;
		ToString(*(param.mean_), tmp);
		sub1->SetText(tmp.c_str());
		sub->LinkEndChild(sub1);
		sub1 = doc.NewElement("std");
		ToString(*(param.std_), tmp);
		sub1->SetText(tmp.c_str());
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	doc.LinkEndChild(root);
	doc.SaveFile(path.c_str());	
}

void Param::LoadParam(const string& path, Param& param){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(path.c_str());
	tinyxml2::XMLElement* sub = doc.RootElement()->FirstChildElement();
	bool flag;
	tinyxml2::XMLUtil::ToBool(sub->GetText(), &flag);
	param.config_->clear();
	param.config_->insert(std::make_pair("need_normalize", flag));
	if (flag){
		string tmp;
		tinyxml2::XMLElement* sub1 = sub->FirstChildElement();
		tmp = string(sub1->GetText());
		ToMat(tmp, *(param.mean_));
		sub1 = sub1->NextSiblingElement();
		tmp = string(sub1->GetText());
		ToMat(tmp, *(param.std_));
	}
}

}//namespace cascade