#include "config.h"

#include <io.h>

#include "tinyxml2.h"

namespace cascade{

using std::make_pair;

static string LoadFunc( const string& xml_path, int number,int& feat_num ){
	if (&xml_path == NULL)
		return string(0);
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	int num;
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&num);
	if (number>num)
		return string(0);
	sub = sub->NextSiblingElement();
	for (int i = 0;i<number;++i)
		sub = sub->NextSiblingElement();
	sub = sub->FirstChildElement()->NextSiblingElement();
	string ans(sub->GetText());
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&feat_num);

	return ans;
}

vector<string> LoadAllFunc( const string& xml_path, int& func_num,int& feat_num ){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&func_num);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&feat_num);
	int tmp;
	vector<string> ans(func_num);
	for (int i = 0;i<func_num;++i){
		ans[i] = LoadFunc(xml_path,i+1,tmp);
	}
	return ans;
}

bool Config::SavePath( const string& xml_path, const vector<string>& pos_path, const vector<string>& neg_path, const vector<string>& bak_path ){
	if (&xml_path == NULL){
		return false;
	}
	tinyxml2::XMLDocument doc;

	tinyxml2::XMLDeclaration *decl = doc.NewDeclaration("xml version=\"1.0\" encoding=\"gb2312\"" );
	tinyxml2::XMLComment *comment = doc.NewComment("训练样本路径配置");
	doc.LinkEndChild(decl);
	doc.LinkEndChild(comment);

	tinyxml2::XMLElement *root = doc.NewElement("paths");
	tinyxml2::XMLElement *sub = doc.NewElement("posList");
	sub->SetAttribute("posNum",(int)pos_path.size());
	tinyxml2::XMLElement *sub1;
	for (int i = 0;i<pos_path.size();++i){
		sub1 = doc.NewElement("path");
		sub1->SetText(pos_path[i].c_str());
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	sub = doc.NewElement("negList");
	sub->SetAttribute("negNum",(int)neg_path.size());
	for (int i = 0;i<neg_path.size();++i){
		sub1 = doc.NewElement("path");
		sub1->SetText(neg_path[i].c_str());
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	sub = doc.NewElement("bakList");
	sub->SetAttribute("bakNum",(int)bak_path.size());
	for (int i = 0;i<bak_path.size();++i){
		sub1 = doc.NewElement("path");
		sub1->SetText(bak_path[i].c_str());
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	doc.LinkEndChild(root);
	int errid = doc.SaveFile(xml_path.c_str());
	if (errid){
		return errid;
	}
	return true;
}

bool Config::SaveFeatFunc( const string& xml_path,const string& name, int feat_num ){
	if (&xml_path == NULL)
		return false;
	tinyxml2::XMLDocument doc;
	if (_access(xml_path.c_str(),0)==-1)
	{
		tinyxml2::XMLDeclaration *decl = doc.NewDeclaration("xml version=\"1.0\" encoding=\"gb2312\"");
		doc.LinkEndChild(decl);
		tinyxml2::XMLComment *cmmt = doc.NewComment("特征提取函数配置");
		doc.LinkEndChild(cmmt);
		tinyxml2::XMLElement *root = doc.NewElement("functions");
		tinyxml2::XMLElement *sub = doc.NewElement("funcNum");
		sub->SetText(0);
		root->LinkEndChild(sub);
		sub = doc.NewElement("featNum");
		sub->SetText(0);
		root->LinkEndChild(sub);
		doc.LinkEndChild(root);
		doc.SaveFile(xml_path.c_str());
	}
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	int num;
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&num);
	++num;
	sub->SetText(num);
	sub = sub->NextSiblingElement();
	int feats;
	util.ToInt(sub->GetText(),&feats);
	feats+=feat_num;
	sub->SetText(feats);

	sub = doc.NewElement("func");
	tinyxml2::XMLElement *sub1 = doc.NewElement("No.");
	sub1->SetText(num);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("path");
	sub1->SetText(name.c_str());
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("featNum");
	sub1->SetText(feat_num);
	sub->LinkEndChild(sub1);
	sub1 = doc.NewElement("config");
	tinyxml2::XMLComment *cmmt = doc.NewComment("请在此写入特征提取函数参数");
	sub1->LinkEndChild(cmmt);
	sub->LinkEndChild(sub1);
	doc.RootElement()->LinkEndChild(sub);
	doc.SaveFile(xml_path.c_str());

	return true;
}

bool Config::SaveConfig( const string& xml_path,const int block_width, 
	const int block_height,const int step_x,const int step_y,
	const int max_neg_live, const int start_layer,const int max_layer_num,
	const int max_weaker_num, const string& type, const vector<float>& recall,
	const vector<float>& false_pos_rate ){

	if (&xml_path == NULL)
		return false;
	tinyxml2::XMLDocument doc;
	tinyxml2::XMLDeclaration *decl = doc.NewDeclaration("xml version=\"1.0\" encoding=\"gb2312\"");
	tinyxml2::XMLComment *cmmt = doc.NewComment("Cascade Configure");
	doc.LinkEndChild(decl);
	doc.LinkEndChild(cmmt);
	tinyxml2::XMLElement *root = doc.NewElement("config");
	tinyxml2::XMLElement *sub = doc.NewElement("maxlayers");
	sub->SetText(max_layer_num);
	root->LinkEndChild(sub);
	sub = doc.NewElement("neg_live");
	sub->SetText(max_neg_live);
	root->LinkEndChild(sub);
	sub = doc.NewElement("max_weaker");
	sub->SetText(max_weaker_num);
	root->LinkEndChild(sub);
	sub = doc.NewElement("img_width");
	sub->SetText(block_width);
	root->LinkEndChild(sub);
	sub = doc.NewElement("img_height");
	sub->SetText(block_height);
	root->LinkEndChild(sub);
	sub = doc.NewElement("stepX");
	sub->SetText(step_x);
	root->LinkEndChild(sub);
	sub = doc.NewElement("stepY");
	sub->SetText(step_y);
	root->LinkEndChild(sub);
	sub = doc.NewElement("startlayer");
	sub->SetText(start_layer);
	root->LinkEndChild(sub);
	sub = doc.NewElement("type");
	sub->SetText(type.c_str());
	root->LinkEndChild(sub);
	sub = doc.NewElement("layer_configs");
	for (int i = 0;i<max_layer_num;++i){
		tinyxml2::XMLElement *sub1 = doc.NewElement("_");
		sub1->SetAttribute("layer",i+1);
		tinyxml2::XMLElement *sub2 = doc.NewElement("hit_rate");
		sub2->SetText(recall[i]);
		sub1->LinkEndChild(sub2);
		sub2 = doc.NewElement("false_rate");
		sub2->SetText(false_pos_rate[i]);
		sub1->LinkEndChild(sub2);
		sub->LinkEndChild(sub1);
	}
	root->LinkEndChild(sub);
	doc.LinkEndChild(root);
	doc.SaveFile(xml_path.c_str());	
	return true;
}

void Config::LoadPath( const string& xml_path ){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *root = doc.RootElement();
	tinyxml2::XMLElement *sub = root->FirstChildElement();
	int pos_num = atoi(sub->FirstAttribute()->Value());
	tinyxml2::XMLElement *sub1 = sub->FirstChildElement();
	pos_path_ = vector<string>(pos_num);
	for (int i = 0;i<pos_num;++i){
		pos_path_[i] = sub1->GetText();
		sub1 = sub1->NextSiblingElement();
	}
	sub = sub->NextSiblingElement();
	int neg_num = atoi(sub->FirstAttribute()->Value());
	neg_path_ = vector<string>(neg_num);
	sub1 = sub->FirstChildElement();
	for (int i = 0;i<neg_num;++i){
		neg_path_[i] = sub1->GetText();
		sub1 = sub1->NextSiblingElement();
	}
	sub = sub->NextSiblingElement();
	int bak_num = atoi(sub->FirstAttribute()->Value());
	bak_path_ = vector<string>(bak_num);
	sub1 = sub->FirstChildElement();
	for (int i = 0;i<bak_num;++i){
		bak_path_[i] = sub1->GetText();
		sub1 = sub1->NextSiblingElement();
	}
}

void Config::LoadAllFunc( const string& xml_path, int& func_num,int& feat_num ){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&func_num);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&feat_num);
	int tmp;
	func_path_ = vector<string>(func_num);
	for (int i = 0;i<func_num;++i){
		func_path_[i] = LoadFunc(xml_path,i+1,tmp);
	}
}

void Config::LoadAllFunc( const string& xml_path ){
	int func_num = 0;
	LoadAllFunc(xml_path,func_num,feat_num_);
}

void Config::LoadConfig( const string& xml_path ){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	tinyxml2::XMLUtil util;
	util.ToInt(sub->GetText(),&max_layer_num_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&max_neg_live_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&max_weaker_num_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&block_width_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&block_height_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&step_x_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&step_y_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(),&start_layer_);
	sub = sub->NextSiblingElement();
	type_=string(sub->GetText());
	sub = sub->NextSiblingElement()->FirstChildElement();

	recall_ = vector<float>(max_layer_num_);
	false_pos_rate_ = vector<float>(max_layer_num_);

	float tmp;
	tinyxml2::XMLElement *sub1;
	for (int i = 0;i<max_layer_num_;++i){
		sub1 = sub->FirstChildElement();
		util.ToFloat(sub1->GetText(),&tmp);
		recall_[i] = tmp;
		sub1 = sub1->NextSiblingElement();
		util.ToFloat(sub1->GetText(),&tmp);
		false_pos_rate_[i] = tmp;
		sub = sub->NextSiblingElement();
	}
}

void Config::Load( const string& sample_path, const string& func_path, 
	const string& config_path ){
	
	LoadPath(sample_path);
	LoadAllFunc(func_path);
	LoadConfig(config_path);
}

Config::Config( const string& sample_path,const string& func_path, 
	const string& config_path){
	Load(sample_path,func_path,config_path);
}

void Config::LoadWeakerConfig(const string& xml_path){
	if (type_ == "threshold"){
		weaker_config.reset(new ThresConfig());
	}
	else if (type_ == "logistic"){
		weaker_config.reset(new LogisticConfig());
	}
	else{
		weaker_config.reset();
	}
	weaker_config->Load(xml_path);
}

void Config::GetWeakerConfig(map<string, float>& parameters)const{
	weaker_config->Get(parameters);
}

void Config::ThresConfig::Load(const string& xml_path){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLUtil util;
	tinyxml2::XMLElement *sub = doc.RootElement()->FirstChildElement();
	sub = sub->FirstChildElement();
	util.ToBool(sub->GetText(), &need_sort_);
	sub = sub->NextSiblingElement();
	util.ToBool(sub->GetText(), &need_normalize_);
}

void Config::ThresConfig::Get(map<string, float>& parameters){
	parameters.clear();
	parameters.insert(make_pair("need_sort", need_sort_));
	parameters.insert(make_pair("need_normalize", need_normalize_));
}

void Config::LogisticConfig::Load(const string& xml_path){
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xml_path.c_str());
	tinyxml2::XMLUtil util;
	tinyxml2::XMLElement *sub = doc.RootElement()->
		FirstChildElement()->NextSiblingElement();
	sub = sub->FirstChildElement();
	util.ToBool(sub->GetText(), &need_sort_);
	sub = sub->NextSiblingElement();
	util.ToBool(sub->GetText(), &need_normalize_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(), &learning_rate_);
	sub = sub->NextSiblingElement();
	util.ToFloat(sub->GetText(), &weight_decay_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(), &batch_size_);
	sub = sub->NextSiblingElement();
	util.ToInt(sub->GetText(), &max_iter_);
}

void Config::LogisticConfig::Get(map<string, float>& parameters){
	parameters.clear();
	parameters.insert(make_pair("need_sort", need_sort_));
	parameters.insert(make_pair("need_normalize", need_normalize_));
	parameters.insert(make_pair("learning_rate", learning_rate_));
	parameters.insert(make_pair("weight_decay", weight_decay_));
	parameters.insert(make_pair("batch_size", batch_size_));
	parameters.insert(make_pair("max_iter", max_iter_));
}

} //namespace cascade



