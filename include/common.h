#ifndef CASCADE_COMMON_H_
#define CASCADE_COMMON_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "tinyxml2.h"
#include "Eigen/Dense"

#include "boost.h"
#include "util.h"
#include "config.h"

namespace cascade{

using std::vector;
using std::string;
using std::shared_ptr;
using Eigen::MatrixXf;

/// Cascade类
class Cascade{
public:

	/** 获取Cascade实例 */
	static Cascade& Get();

	/** 训练函数
	 *  
	 * @param [in] const string & features_path 特征矩阵文件路径（包含正负样本）
	 * @param [in] const string & labels_path 标签向量文件路径
	 * @param [in] const string & hard_path 负例样本文件夹路径
	 * @param [in] const Config & config 配置参数
	 * @param [in] const string & model_path 模型保存路径
	 * @return void 
	 * @note 最终会保存模型xml和相应参数xml，使用模型时需要二者对应
	 */
	void Train(const string& features_path, const string& labels_path,
		const string& hard_path, const Config& config, 
		const string& model_path);

	/** 预测函数，采用单线程方式
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [out] vector<int> & anwser 预测结果
	 * @return void 
	 * @note  
	 */
	void Predict(const MatrixXf& features, vector<int>& anwser) const;

	/** 模型保存
	 *  
	 * @param [in] const string & path 模型保存路径
	 * @return void 
	 * @note  
	 */
	void SaveModel(const string& path);

	/** 模型读取
	 *  
	 * @param [in] const string & path 模型读取路径
	 * @return void 
	 * @note  
	 */
	void LoadModel(const string& path);

private:
	vector<shared_ptr<Boost>> model_; /**< 实际模型数据 */
	static shared_ptr<Cascade> instance_; /**< Cascade类实例 */

private:

	/** 私有化构造函数，通过Get获取实例，禁止显式实例化 */
	Cascade(){};

	/** 读取模型训练断点
	 *  
	 * @param [in] const string & path 模型文件路径
	 * @param [in] int layers 起始层数
	 * @return void 
	 * @note  
	 */
	void LoadBreak(const string& path,int layers);

	/** 保存模型头信息
	 *  
	 * @param [in] const string & path 模型文件路径
	 * @param [in] const string & type 弱分类器类型
	 * @return void 
	 * @note  
	 */
	void SaveHead(const string& path, const string& type);

	/** 保存模型训练断点
	 *  
	 * @param [in] const string & path 模型文件路径
	 * @return void 
	 * @note  
	 */
	void SaveBreak(const string& path);

private:
	/** 预测函数，采用单线程方式
	*
	* @param [in] const MatrixXf & features 特征矩阵
	* @param [out] vector<int> & anwser 预测结果
	* @return void
	* @note
	*/
	void PredictForTrain(const MatrixXf& features, vector<int>& anwser) const;

	/* 友元函数，用于替换困难样本时进行分类 */
	friend int PredictHardExample(const Cascade& model, MatrixXf &feat);

	/* 与模型对应的参数的保存路径，根据模型路径自动生成 */
	string param_path_;

UNCOPABLE(Cascade);
};

} // namespace cascade

#endif // CASCADE_COMMON_H_