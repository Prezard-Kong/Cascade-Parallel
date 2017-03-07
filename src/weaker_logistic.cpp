#include "weaker_logistic.h"

#include <random>
#include <map>

#include "util.h"
namespace cascade{

const int PassValueSize = sizeof(int) + sizeof(float) * 3; //pass_value_内存大小//

static void Permutation(int size, vector<int>& permutation){
	permutation = vector<int>(size);
	for (int i = 0; i < size; ++i){
		permutation[i] = i;
	}
	std::random_shuffle(permutation.begin(), permutation.end());
}

static void SelectBatch(const MatrixXf& features, const int col,
	const vector<float>& weights, const vector<int>& labels, int &st, int batch_size,
	const vector<int>& permutation, MatrixXf& feat, MatrixXf& ws, MatrixXf& labs){

	int k = 0;
	for (int i = st; i < st + batch_size; ++i){
		int loc = i%features.rows();
		feat(k, 0) = features(permutation[loc], col);
		feat(k, 1) = 1;
		ws(k, 0) = weights[permutation[loc]];
		labs(k, 0) = labels[permutation[loc]] == 1 ? 1 : 0;
		++k;
	}
	st = (st + batch_size) % features.rows();
}

static void Select(const MatrixXf& features, const int col, MatrixXf& feat){
	feat.resize(features.rows(), 2);
	memcpy(feat.data(), features.col(col).data(), features.rows()*sizeof(float));
	memcpy(feat.col(1).data(), vector<int>(features.rows(), 1).data(), 
		features.rows()*sizeof(float));
}

static void Logistic(const MatrixXf& feat, const MatrixXf& param , MatrixXf& logis){
	logis = (1 + (feat*param).array().exp()).cwiseInverse();
}

static void NormalizeWeights(MatrixXf& weights){
	weights /= weights.sum();
}

static void PredictByVal(const MatrixXf& logis, vector<int>& preds){
	preds = vector<int>(logis.rows());
	for (int i = 0; i < logis.rows(); ++i){
		preds[i] = logis(i, 0) > 0.5 ? 1 : -1;
	}
}

static float CalcError(const vector<float>& weights, const vector<int>& labels,
	const vector<int>& preds){

	float ans = 0;
	for (int i = 0; i < weights.size(); ++i){
		ans += weights[i] * (labels[i] == preds[i] ? 0 : 1);
	}
	return ans;
}

static float CalcError(const MatrixXf& features, int feat_idx, const MatrixXf& param,
	const vector<float>& weights, const vector<int>& labels){

	MatrixXf feats;
	Select(features, feat_idx, feats);
	MatrixXf logis;
	Logistic(feats, param, logis);
	vector<int> preds;
	PredictByVal(logis, preds);
	return CalcError(weights, labels, preds);
}

static float Train(const MatrixXf& features, const int col, const vector<float>& weights, 
	const vector<int>& labels,	const float eta, const float lambda, const int batch_size, 
	const int max_iters, MatrixXf& params){

	params.resize(2, 1);
	for (int i = 0; i < params.rows(); ++i){
		params(i, 0) = 0;
	}
	int st = 0;
	vector<int> permutation;
	Permutation(features.rows(), permutation);
	MatrixXf feat(batch_size, 2);
	MatrixXf ws(batch_size, 1);
	MatrixXf labs(batch_size, 1);
	//每max_iters/10步检验一次误差,结果取检验误差最小时的参数//
	int test_interval = std::max(max_iters / 10, 1);
	MatrixXf params_tmp(params);
	float err_min = FLT_MAX;
	for (int i = 0; i < max_iters; ++i){
		SelectBatch(features, col, weights, labels, st, batch_size, 
			permutation, feat, ws, labs);
		MatrixXf logis;
		Logistic(feat, params, logis);
		NormalizeWeights(ws);
		feat.transposeInPlace();
		params -= eta*(feat*ws.cwiseProduct(logis - labs) / batch_size + lambda*params);
		feat.transposeInPlace();
		if (i%test_interval == 0){
			float err = CalcError(features, col, params, weights, labels);
			if (err < err_min){
				err_min = err;
				memcpy(params_tmp.data(), params.data(), params.rows()*params.cols()*sizeof(float));
			}
		}
	}
	float err = CalcError(features, col, params, weights, labels);
	if (err < err_min){
		err_min = err;
		memcpy(params_tmp.data(), params.data(), params.rows()*params.cols()*sizeof(float));
	}
	memcpy(params.data(), params_tmp.data(), params.rows()*params.cols()*sizeof(float));
	return err_min;
}

float WeakerLogistic::Train(const MatrixXf& features, const vector<int>& labels, 
	const vector<float>& weights, const MatrixXi& index,
	const map<string, float>& parameters /* = map<string, float>() */){

	auto GetVal = [&](string key){return parameters.find(key)->second; };
	float eta = GetVal("learning_rate");
	float lambda = GetVal("weight_decay");
	int batch_size = GetVal("batch_size");
	int max_iter = GetVal("max_iter");

	float err_min = FLT_MAX;
	for (int i = 0; i < features.cols(); ++i){
		MatrixXf param;
		float err = cascade::Train(features, i, weights, labels, 
			eta, lambda, batch_size, max_iter, param);
		if (err < err_min){
			err_min = err;
			theta_ = param(0, 0);
			bias_ = param(1, 0);
			feat_index_ = i;
		}
	}
	calc_alpha(err_min);
	ToPassValue();
	return err_min;
}

void WeakerLogistic::Predict(const MatrixXf& features, vector<int>& answer){
	MatrixXf feats;
	Select(features, feat_index_, feats);
	MatrixXf param(2, 1);
	param(0, 0) = theta_;
	param(1, 0) = bias_;
	MatrixXf logis;
	Logistic(feats, param, logis);
	PredictByVal(logis, answer);
}

tinyxml2::XMLElement* WeakerLogistic::SaveModel(tinyxml2::XMLDocument& doc){
	tinyxml2::XMLElement *sub = doc.NewElement("_");
	tinyxml2::XMLElement *sub1 = doc.NewElement("weight");
	sub1->SetText(alpha_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("featidx");
	sub1->SetText(feat_index_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("theta");
	sub1->SetText(theta_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("bias");
	sub1->SetText(bias_);
	sub->LinkEndChild(sub1);
	return sub;
}

void WeakerLogistic::LoadModel(tinyxml2::XMLElement* header){
	tinyxml2::XMLElement *sub = header->FirstChildElement();
	tinyxml2::XMLUtil util;
	util.ToFloat(sub->GetText(), &alpha_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(), &feat_index_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(), &theta_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(), &bias_);
}

MPI_Datatype WeakerLogistic::PassType(){
	//alpha, feat_index, theta, bias, threshold
	MPI_Datatype old_type[4] = { MPI_FLOAT, MPI_INT, MPI_FLOAT, MPI_FLOAT };
	int block_count[4] = { 1, 1, 1, 1 };
	MPI_Aint offset[4] = { 0 };
	offset[1] = sizeof(float);
	offset[2] = offset[1] + sizeof(int);
	offset[3] = offset[2] + sizeof(float);
	MPI_Datatype new_type;
	MPI_Type_struct(4, block_count, offset, old_type, &new_type);
	MPI_Type_commit(&new_type);
	return new_type;
}

void WeakerLogistic::FromPassValue(){
	int offset[5] = { 0 };
	offset[1] = sizeof(float);
	offset[2] = offset[1] + sizeof(int);
	offset[3] = offset[2] + sizeof(float);
	memcpy(&alpha_, pass_value_.get() + offset[0], sizeof(float));
	memcpy(&feat_index_, pass_value_.get() + offset[1], sizeof(int));
	memcpy(&theta_, pass_value_.get() + offset[2], sizeof(float));
	memcpy(&bias_, pass_value_.get() + offset[3], sizeof(float));
}

void WeakerLogistic::ToPassValue(){
	int offset[4] = { 0 };
	offset[1] = sizeof(float);
	offset[2] = offset[1] + sizeof(int);
	offset[3] = offset[2] + sizeof(float);
	memcpy(pass_value_.get() + offset[0], &alpha_, sizeof(float));
	memcpy(pass_value_.get() + offset[1], &feat_index_, sizeof(int));
	memcpy(pass_value_.get() + offset[2], &theta_, sizeof(float));
	memcpy(pass_value_.get() + offset[3], &bias_, sizeof(float));
}

char* WeakerLogistic::pass_value() const{
	return pass_value_.get();
}

shared_ptr<char> WeakerLogistic::pass_value_ = shared_ptr<char>(new char[PassValueSize]);

}