#ifndef CASCADE_WEAKER_LOGISTIC_H_
#define CASCADE_WEAKER_LOGISTIC_H_

#include "weaker.h"

namespace cascade{

///逻辑回归弱分类器
class WeakerLogistic : public Weaker{
public:

	/** 训练函数
	*
	* @param [in] const MatrixXf & features 特征矩阵
	* @param [in] const vector<int> & labels 标签向量
	* @param [in] const vector<float> & weights 权重向量
	* @param [in] const MatrixXi & index 特征矩阵排序索引
	* @param [in] const map<string, float> & parameters 训练参数
	* @return float 训练误差
	* @note 本分类器需要训练参数包括：(学习率：learning_rate),
	* (L2正则化系数：weight_decay), (随机梯度下降batch size：batch_size),
	* (最大迭代步数：max_iter)
	*/
	virtual float Train(const MatrixXf& features, const vector<int>& labels,
		const vector<float>& weights, const MatrixXi& index,
		const map<string, float>& parameters = map<string, float>());

	/** 预测函数，采用单线程的方式
	*
	* @param [in] const MatrixXf & features 特征矩阵
	* @param [out] vector<int> & answer 预测结果向量
	* @return void
	* @note
	*/
	virtual void Predict(const MatrixXf& features, vector<int>& answer);

	/** 模型保存
	*
	* @param [in] tinyxml2::XMLDocument & doc xml文档
	* @return tinyxml2::XMLElement* 数据保存的xml结点，用于与其他结点连接
	* @note
	*/
	virtual	tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc);

	/** 模型读取
	*
	* @param [in] tinyxml2::XMLElement * header 数据保存所在的结点
	* @return void
	* @note
	*/
	virtual void LoadModel(tinyxml2::XMLElement* header);

	/** 注册自定义的MPI_Datatype */
	virtual MPI_Datatype PassType();

	/** 将pass_value_中的值写入对象的成员变量中 */
	virtual void FromPassValue();

	/** 将对象的成员变量中的值写入pass_value_ */
	virtual void ToPassValue();

	virtual char* pass_value() const;

	/** 返回逻辑回归系数 */
	inline float theta() const { return theta_; };

	/** 返回逻辑回归偏置 */
	inline float bias() const { return bias_; };

private:
	float theta_; /**< 回归系数 */
	float bias_; /**< 回归偏置 */
	static shared_ptr<char> pass_value_; /**< MPI进程间参数传递数据指针 */
};

}
#endif // !CASCADE_WEAKER_LOGISTIC_H_
