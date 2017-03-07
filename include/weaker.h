#ifndef CASCADE_WEAKER_H_
#define CASCADE_WEAKER_H_

#include <vector>
#include <memory>
#include <map>
#include <string>

#include "tinyxml2.h"
#include "Eigen/Dense"
#include "mpi.h"

namespace cascade{

using std::vector;
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using std::shared_ptr;
using std::map;
using std::string;

/// 弱分类器接口，抽象基类
class Weaker{
public:

	/** 训练函数
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [in] const vector<int> & labels 标签向量
	 * @param [in] const vector<float> & weights 权重向量
	 * @param [in] const MatrixXi & index 特征矩阵排序索引
	 * @param [in] const map<string, float> & parameters 训练参数
	 * @return float 训练误差
	 * @note  
	 */
	virtual float Train(const MatrixXf& features, const vector<int>& labels, 
		const vector<float>& weights, const MatrixXi& index,
		const map<string,float>& parameters = map<string, float>())=0;

	/** 预测函数，采用单线程的方式
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [out] vector<int> & answer 预测结果向量
	 * @return void 
	 * @note  
	 */
	virtual void Predict(const MatrixXf& features, vector<int>& answer)=0;

	/** 模型保存
	 *  
	 * @param [in] tinyxml2::XMLDocument & doc xml文档
	 * @return tinyxml2::XMLElement* 数据保存的xml结点，用于与其他结点连接
	 * @note  
	 */
	virtual tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc)=0;

	/** 模型读取
	 *  
	 * @param [in] tinyxml2::XMLElement * header 数据保存所在的结点
	 * @return void 
	 * @note  
	 */
	virtual void LoadModel(tinyxml2::XMLElement *header)=0;

	/** 注册自定义的MPI_Datatype */
	virtual MPI_Datatype PassType()=0;

	/** 将pass_value_中的值写入对象的成员变量中 */
	virtual void FromPassValue()=0;

	/** 将对象的成员变量中的值写入pass_value_ */
	virtual void ToPassValue()=0; 

	/** 返回pass_value_指针数据 */
	virtual char* pass_value() const = 0;

	/** 析构函数 */
	virtual ~Weaker(){};

	/** 返回弱分类器的权重 */
	inline float alpha() const {return alpha_;};

	/** 返回弱分类器所用的特征序号 */
	inline int feat_index() const {return feat_index_;};

	/** 设置弱分类器所用的特征序号
	 *  
	 * @param [in] int index 特征序号
	 * @return void 
	 * @note  由于采用多线程，每个弱分类器中的特征序号是局部的，
	 *		  用该函数将序号修改为全局的
	 */
	inline void set_feat_index(int index){feat_index_ = index;};

	/** 返回pass_value_指针数据 */
// 	inline char* pass_value() const {
// 		return pass_value_.get();
// 	};

protected:

	/** 计算弱分类器的权重
	 *  
	 * @param [in] float error 训练误差
	 * @return void 
	 * @note  
	 */
	inline void calc_alpha(float error){
		float tmp = log((1-error)/(error + 1e-6));
		alpha_ = tmp<3?tmp:3; //用于截断，防止某一个分类器权重过大//
	};

protected:

	float alpha_; /**< 弱分类器权重 */
	int feat_index_; /**< 弱分类器所用的特征序号 */
	
};

} // namespace cascade
#endif // CASCADE_WEAKER_H_