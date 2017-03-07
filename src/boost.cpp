#include "boost.h"

#include <stdio.h>

#include <iostream>
#include <map>

#include "mpi.h"

#include "util.h"

namespace cascade{

using std::make_pair;
using std::multimap;

static void WeightsNormalize(vector<float>& weights){
	float sum = 0;
	for (int i = 0;i<weights.size();++i){
		sum += weights[i];
	}
	for (int i = 0;i<weights.size();++i){
		weights[i] /= sum;
	}
}

static void WeightsUpdate(vector<float>& weights, const vector<int>& labels, 
	const vector<int>& y, float alpha){
	for (int i = 0;i<weights.size();++i){
		if (labels[i] != y[i]){
			weights[i] *= exp(alpha);
		}
	}
}
static void FeatureSort(const MatrixXf& features, MatrixXi& index){
	index = MatrixXi(features.rows(),features.cols());
	multimap<float, int> map;
	for (int j = 0;j<features.cols();++j){
		for (int i = 0;i<features.rows();++i){
			map.insert(make_pair(features(i,j),i));
		}
		int i = 0;
		for (auto iter = map.begin();iter!=map.end();++iter,++i){
			index(i,j) = iter->second;
		}
		map.clear();
	}
}
//ans += val*vec
static void VectorProductAdd(float val, const vector<int>& vec, vector<float> &ans){
	for (int i = 0;i<vec.size();++i){
		ans[i] += val*vec[i];
	}
}

static void CalcPredictVal(vector<shared_ptr<Weaker>>& boost, vector<int>& locs, 
	int p_rank, const MatrixXf& feats, vector<float>& ans){

	vector<int> pred_labels(feats.rows(),0);
	for (int j=0;j<boost.size();++j){
		if (locs[j] == p_rank){
			boost[j]->Predict(feats,pred_labels);
			VectorProductAdd(boost[j]->alpha(),pred_labels,ans);
		}
	}
}

static float FindBoostThreshold(const vector<int>& labels, 
	const vector<float>& pred_vals, int pos_num, float min_recall){

	int pos_false_num_thres = pos_num*(1 - min_recall);
	multimap<float, int> map;
	for (int i = 0; i < pred_vals.size(); ++i){
		map.insert(make_pair(pred_vals[i], i));
	}

	int pos_false_num = 0;
	float threshold = FLT_MAX;
	float threshold_old = threshold;
	for (auto iter = map.begin(); iter != map.end(); ++iter){
		pos_false_num += (labels[iter->second] == 1);
		if (!IsZero(threshold - iter->first)){
			threshold_old = threshold;
			threshold = iter->first;
		}
		if (pos_false_num > pos_false_num_thres&&!IsZero(FLT_MAX-threshold_old)){
			threshold = (threshold_old + threshold) / 2;
			break;
		}
	}
	return threshold;
}

static void PredictByVal(Boost& boost, const vector<float>& val, vector<int>& ans){
	for (int i = 0;i<val.size();++i){
		ans[i] = val[i]<boost.threshold()?-1:1;
	}
}

static float CalcFalsePosRate(const vector<int>& labels, const vector<int>& pred_labels){
	float ans = 0;
	int negtive_num = 0;
	for (int i = 0;i<labels.size();++i){
		if (labels[i] == -1){
			++negtive_num;
			if (pred_labels[i] == 1){
				++ans;
			}
		}
	}
	return ans/negtive_num;
}

static float CalcRecall(const vector<int>& labels, const vector<int>& pred_labels){
	float ans = 0;
	int positive_num = 0;
	for (int i = 0;i<labels.size();++i){
		if (labels[i] == 1){
			++positive_num;
			if (pred_labels[i] == 1){
				++ans;
			}
		}
	}
	return ans/positive_num;
}

//根进程保有全部训练数据
//所有进程保有模型//
void Boost::Train( const MatrixXf& features, vector<int>& labels, 
		float max_false_pos_rate, float min_recall, int max_weaker_num,
		const map<string, float>& parameters){
	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
	int rows = features.rows();
	int cols = features.cols();
	MPI_Bcast(&rows,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&cols,1,MPI_INT,0,MPI_COMM_WORLD);
	vector<float> weights(rows, 1.0/rows);
	vector<int> counts(processor_num);
	vector<int> displs(processor_num);
	for (int i = 0;i<processor_num;++i){
		counts[i] = DivideNums(i, processor_num, cols)*rows;
		displs[i] = DivideStart(i,processor_num, cols)*rows;
	}

	int pos_num = 0;
	for each(int val in labels){
		pos_num += (val==1);
	}

	MatrixXf local_features(rows,counts[processor_rank]/rows);
	MPI_Scatterv(features.data(),counts.data(),displs.data(),MPI_FLOAT,
		local_features.data(),counts[processor_rank],MPI_FLOAT,0,MPI_COMM_WORLD);
	MatrixXi local_index;
	if (bool(parameters.find("need_sort")->second)){
		local_index.resize(rows, local_features.cols());
		FeatureSort(local_features, local_index);
	}
	struct PassValue{
		float err;
		int loc;
	};
	float false_pos_rate = 0;
	float recall = 0;
	vector<int> best_loc_vec; //用于保存每个best weaker的进程号//
	for (int i = 0;i<max_weaker_num;++i){
		shared_ptr<Weaker> weaker = WeakerFactory::SetUp(type_);
		float error = weaker->Train(local_features, labels, weights, local_index, parameters);

		PassValue pass_value;
		pass_value.err = error;
		pass_value.loc = processor_rank;

		PassValue best_value;
		MPI_Reduce(&pass_value,&best_value,1,MPI_FLOAT_INT,MPI_MINLOC,0,MPI_COMM_WORLD);
		MPI_Bcast(&best_value.loc, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(weaker->pass_value(),1,weaker->PassType(),best_value.loc,MPI_COMM_WORLD);
		
		if (best_value.err > 0.5){
			if (processor_rank == 0){
				std::cout << "best weaker error is " << best_value.err << std::endl;
				std::cout << "training break, because the error is larger than 0.5";
				std::cout << std::endl;
			}
			break;
		}

		weaker->FromPassValue();
		boost_.push_back(weaker);
		best_loc_vec.push_back(best_value.loc);

		vector<float> pred_val(rows,0);
		CalcPredictVal(boost_,best_loc_vec,processor_rank,local_features,pred_val);
		vector<float> pred_val_sum(rows,0);
		MPI_Reduce(pred_val.data(),pred_val_sum.data(),rows,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
		float threshold=0;
		if (processor_rank == 0){
			threshold = FindBoostThreshold(labels,pred_val_sum,pos_num,min_recall);
		}

		MPI_Bcast(&threshold,1,MPI_FLOAT, 0, MPI_COMM_WORLD);
		set_threshold(threshold);

		vector<int> pred_label(rows, 0);
		if (processor_rank == 0){
			PredictByVal(*this,pred_val_sum,pred_label);
			WeightsUpdate(weights,labels,pred_label,boost_[boost_.size()-1]->alpha());
			WeightsNormalize(weights);
			false_pos_rate = CalcFalsePosRate(labels,pred_label);
			recall = CalcRecall(labels, pred_label);
		}

		MPI_Bcast(weights.data(),rows,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(&false_pos_rate,1,MPI_FLOAT,0,MPI_COMM_WORLD);
		MPI_Bcast(&recall, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (false_pos_rate<max_false_pos_rate&&recall>min_recall){
			break;
		}
	}
	
	recall_ = recall;
	false_pos_rate_ = false_pos_rate;
	//weaker中的feat_idx是每个进程中部分数据的feat_idx
	//下面转化为全局feat_idx
	for (int i = 0;i<boost_.size();++i){
		boost_[i]->set_feat_index(boost_[i]->feat_index()+displs[best_loc_vec[i]]/rows);
	}
}
/*
void Boost::Predict( const MatrixXf& features, vector<int>& answers ){
	int processor_num;
	int processor_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
	int rows = features.rows();
	int cols = features.cols();
	MPI_Bcast(&rows,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&cols,1,MPI_INT,0,MPI_COMM_WORLD);
	vector<int> counts(processor_num);
	vector<int> displs(processor_num);
	for (int i = 0;i<processor_num;++i){
		counts[i] = DivideNums(i, processor_num, cols)*rows;
		displs[i] = DivideStart(i,processor_num, cols)*rows;
	}
	MatrixXf local_features(rows,counts[processor_rank]/rows);
	MPI_Scatterv(features.data(),counts.data(),displs.data(),MPI_FLOAT,
		local_features.data(),counts[processor_rank],MPI_FLOAT,0,MPI_COMM_WORLD);

	vector<int> locs(boost_.size());
	for (int i = 0;i<boost_.size();++i){
		locs[i] = Owner(i,processor_num,boost_.size());
	}
	//暂时将feat_idx变为local_feature相关变量
	for (int i = 0;i<boost_.size();++i){
		boost_[i]->set_feat_index(boost_[i]->feat_index()-displs[locs[i]]);
	}

	vector<float> pred_val(rows,0);
	CalcPredictVal(boost_,locs,processor_rank,local_features,pred_val);
	vector<float> pred_val_sum(rows,0);
	MPI_Reduce(pred_val.data(),pred_val_sum.data(),rows,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	answers.clear();
	answers = vector<int>(rows);
	if (processor_rank == 0){
		PredictByVal(*this,pred_val_sum,answers);
	}
	MPI_Bcast(answers.data(),answers.size(),MPI_INT,0,MPI_COMM_WORLD);

	//恢复feat_idx
	for (int i = 0;i<boost_.size();++i){
		boost_[i]->set_feat_index(boost_[i]->feat_index()-displs[locs[i]]);
	}
}
*/
//预测采用单进程，结果广播至所有进程//
// void Boost::Predict(const MatrixXf& features, vector<int>& answers){
// 	int processor_num;
// 	int processor_rank;
// 	MPI_Comm_size(MPI_COMM_WORLD, &processor_num);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
// 	int rows = features.rows();
// 	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
// 	vector<int> loc(boost_.size(), 0);
// 	vector<float> pred_val(features.rows());
// 
// 	CalcPredictVal(boost_, loc, processor_rank, features, pred_val);
// 
// 	answers = vector<int>(rows);
// 	if (processor_rank == 0){
// 		PredictByVal(*this, pred_val, answers);
// 	}
// 	MPI_Bcast(answers.data(), answers.size(), MPI_INT, 0, MPI_COMM_WORLD);
// }

static void CalcPredictVal(vector<shared_ptr<Weaker>>& boost,
	const MatrixXf& feats, vector<float>& ans){

	vector<int> pred_labels(feats.rows(), 0);
	for (int j = 0; j < boost.size(); ++j){
		boost[j]->Predict(feats, pred_labels);
		VectorProductAdd(boost[j]->alpha(), pred_labels, ans);
	}
}

void Boost::Predict(const MatrixXf& features, vector<int>& answers){
	vector<float> pred_val(features.rows());
	CalcPredictVal(boost_, features, pred_val);
	answers = vector<int>(pred_val.size());
	PredictByVal(*this, pred_val, answers);
}

tinyxml2::XMLElement* Boost::SaveModel( tinyxml2::XMLDocument& doc ){
	tinyxml2::XMLElement* sub = doc.NewElement("boost");
	tinyxml2::XMLElement* sub1 = doc.NewElement("type");
	sub1->SetText(type_.c_str());
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("threshold");
	sub1->SetText(threshold_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("recall");
	sub1->SetText(recall_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("false_pos_rate");
	sub1->SetText(false_pos_rate_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("weaker_num");
	sub1->SetText((int)boost_.size());
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("weakers");
	for (int i = 0;i<boost_.size();++i){
		tinyxml2::XMLElement* sub2 = boost_[i]->SaveModel(doc);
		sub1->LinkEndChild(sub2);
	}
	sub->LinkEndChild(sub1);
	return sub;
}

void Boost::LoadModel( tinyxml2::XMLElement* header ){
	tinyxml2::XMLElement* sub = header->FirstChildElement();
	type_ = sub->GetText();
	sub = sub->NextSiblingElement();
	tinyxml2::XMLUtil util;
	util.ToFloat(sub->GetText(),&threshold_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(),&recall_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(),&false_pos_rate_);
	sub = sub->NextSiblingElement();
	int weak_num;
	util.ToInt(sub->GetText(),&weak_num);
	sub = sub->NextSiblingElement()->FirstChildElement();
	for (int i = 0;i<weak_num;++i){
		shared_ptr<Weaker> weaker = WeakerFactory::SetUp(type_);
		weaker->LoadModel(sub);
		boost_.push_back(weaker);
		sub = sub->NextSiblingElement();
	}
}

} //namespace cascade