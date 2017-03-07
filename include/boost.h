#ifndef CASCADE_BOOST_H_
#define CASCADE_BOOST_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "Eigen/Dense"
#include "tinyxml2.h"

#include "weaker_factory.h"
#include "weaker.h"

namespace cascade{

using std::vector;
using std::shared_ptr;
using Eigen::MatrixXf;
using std::string;
using std::map;

/// 强分类器Boost类
class Boost{
public:

	/** 构造函数
	 *  
	 * @param [in] const string & type 包含的弱分类器类型
	 * @return  
	 * @note  
	 */
	explicit Boost(const string& type):type_(type){};

	/** 训练函数
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [in] vector<int> & labels 标签向量
	 * @param [in] float max_false_pos_rate 最大虚警率
	 * @param [in] float min_recall 最小召回率
	 * @param [in] int max_weaker_num 最大弱分类器个数
	 * @param [in] const map<string,float> & parameters 弱分类器训练参数
	 * @return void 
	 * @note  
	 */
	void Train(const MatrixXf& features, vector<int>& labels, 
		float max_false_pos_rate, float min_recall, int max_weaker_num,
		const map<string,float>& parameters);

	/** 预测函数，采用单线程的方式
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [out] vector<int> & answers 预测结果
	 * @return void 
	 * @note  
	 */
	void Predict(const MatrixXf& features, vector<int>& answers);

	/** 模型保存
	 *  
	 * @param [in] tinyxml2::XMLDocument & doc xml文档
	 * @return tinyxml2::XMLElement* 模型保存的xml结点，用于和其他结点连接
	 * @note  
	 */
	tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc);

	/** 模型读取
	 *  
	 * @param [in] tinyxml2::XMLElement * header 数据保存所在的结点
	 * @return void 
	 * @note  
	 */
	void LoadModel(tinyxml2::XMLElement* header);

	/** 返回弱分类器个数 */
	inline int weaker_num() const {return boost_.size();};

	/** 返回强分类器阈值 */
	inline float threshold() const {return threshold_;};

	/** 返回训练时的召回率结果 */
	inline float recall() const {return recall_;};

	/** 返回训练时的虚警率结果 */
	inline float false_pos_rate() const {return false_pos_rate_;};

	/** 返回弱分类器的类型 */
	inline string type() const {return type_;};

	/** 设置强分类器阈值，用于模型的手动调整 */
	inline void set_threshold(float val){threshold_ = val;};
private:
	vector<shared_ptr<Weaker>> boost_; /**< 实际模型数据 */
	float threshold_; /**< 强分类器阈值 */
	float recall_; /**< 训练时的召回率 */
	float false_pos_rate_; /**< 训练时的虚警率 */
	string type_; /**< 弱分类器类型 */

UNCOPABLE(Boost);
};

} // namespace cascade

#endif // CASCADE_BOOST_H_