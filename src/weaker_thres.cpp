#include "weaker_thres.h"

#include <iostream>

#include "util.h"

namespace cascade{

const int PassValueSize = sizeof(int) * 2 + sizeof(float) * 2; //pass_value_内存大小//

WeakerThres FindThreshold(const MatrixXf& features, const vector<int>& labels, const vector<float>& weights,
	const MatrixXi& index, int col, float &error){

	WeakerThres ans;
	int rows = features.rows();
	float left_err=0, right_err=0; //left为不等号向左的误差，即小于阈值为正//
	float left_err_min=FLT_MAX,right_err_min=FLT_MAX;
	int left_index = 0;
	int right_index = rows-1;
	int idx = 0;
	//error init
	for (int i = 0; i < rows; ++i){
		idx = index(i, col);
		left_err += weights[idx] * (labels[idx] == 1 ? 1 : 0);
		right_err += weights[idx] * (labels[idx] == -1 ? 1 : 0);
	}
	for (int i = 0;i<rows;++i){
		idx = index(i, col);
		left_err += weights[idx]*(labels[idx]==1?-1:1);
		right_err += weights[idx]*(labels[idx]==-1?-1:1);
		if (i<rows-1 && 
			IsZero(features(index(i, col), col)-features(index(i+1,col), col))){
			continue;
		}
		if (left_err<left_err_min){
			left_err_min = left_err;
			left_index = idx;
		}
		if (right_err<right_err_min){
			right_err_min = right_err;
			right_index = idx;
		}
	}
	if (left_err_min<right_err_min){
		error = left_err_min;
		ans.bias_ = 1;
		ans.feat_index_ = col;
		ans.threshold_ = features(left_index, col);
	}else{
		error = right_err_min;
		ans.bias_ = -1;
		ans.feat_index_ = col;
		ans.threshold_ = features(right_index, col);
	}
	return ans;		
}
float WeakerThres::Train( const MatrixXf& features, const vector<int>& labels, 
	const vector<float>& weights, const MatrixXi& index, 
	const map<string, float>& parameters /* = map<string, float>() */){

	int rows = features.rows();
	int cols = features.cols();
	float err_min = FLT_MAX;
	float err = 0;
	for (int i = 0;i<features.cols();++i){
		WeakerThres weaker = FindThreshold(features,labels,weights,index,i,err);
		if (err < err_min){
			err_min = err;
			*this = weaker;
		}
	}
	calc_alpha(err_min);
	ToPassValue();
	return err_min;
}

void WeakerThres::Predict( const MatrixXf& features, vector<int>& answer ){
	answer.clear();
	answer = vector<int>(features.rows());
	for (int i = 0;i<features.rows();++i){
		answer[i] = features(i, feat_index_)*bias_<threshold_*bias_?1:-1;
	}
}

tinyxml2::XMLElement* WeakerThres::SaveModel( tinyxml2::XMLDocument& doc ){
	tinyxml2::XMLElement *sub = doc.NewElement("_");
	//sub->SetAttribute("type","threshold");
	tinyxml2::XMLElement *sub1 = doc.NewElement("weight");
	sub1->SetText(alpha_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("featidx");
	sub1->SetText(feat_index_);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("threshold");
	sub1->SetText(threshold_);
	sub	->LinkEndChild(sub1);
	sub1 = doc.NewElement("bias");
	sub1->SetText(bias_);
	sub->LinkEndChild(sub1);
	return sub;
}

void WeakerThres::LoadModel( tinyxml2::XMLElement* header ){
	tinyxml2::XMLElement *sub = header->FirstChildElement();
	tinyxml2::XMLUtil util;
	util.ToFloat(sub->GetText(), &alpha_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(), &feat_index_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(), &threshold_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(), &bias_);
}


MPI_Datatype WeakerThres::PassType(){
	//alpha,bias,threshold,feat_index
	MPI_Datatype old_type[4] = {MPI_FLOAT, MPI_INT, MPI_FLOAT, MPI_INT};
	int block_count[4] = {1, 1, 1, 1};
	MPI_Aint offset[4] = {0};
	offset[1] = sizeof(float);
	offset[2] = offset[1]+sizeof(int);
	offset[3] = offset[2]+sizeof(float);
	MPI_Datatype new_type;
	MPI_Type_struct(4,block_count,offset,old_type,&new_type);
	MPI_Type_commit(&new_type);
	return new_type;
}

void WeakerThres::FromPassValue(){
	int offset[4] = {0};
	offset[1] = sizeof(float);
	offset[2] = offset[1]+sizeof(int);
	offset[3] = offset[2]+sizeof(float);
	memcpy(&alpha_,pass_value_.get()+offset[0],sizeof(float));
	memcpy(&bias_,pass_value_.get()+offset[1],sizeof(int));
	memcpy(&threshold_,pass_value_.get()+offset[2],sizeof(float));
	memcpy(&feat_index_,pass_value_.get()+offset[3],sizeof(int));
}

void WeakerThres::ToPassValue(){
	int offset[4] = {0};
	offset[1] = sizeof(float);
	offset[2] = offset[1]+sizeof(int);
	offset[3] = offset[2]+sizeof(float);
	memcpy(pass_value_.get()+offset[0],&alpha_,sizeof(float));
	memcpy(pass_value_.get()+offset[1],&bias_,sizeof(int));
	memcpy(pass_value_.get()+offset[2],&threshold_,sizeof(float));
	memcpy(pass_value_.get()+offset[3],&feat_index_,sizeof(int));
}

char* WeakerThres::pass_value() const{
	return pass_value_.get();
}

shared_ptr<char> WeakerThres::pass_value_ = shared_ptr<char>(new char[PassValueSize]);

}