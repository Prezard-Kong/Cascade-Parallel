#ifndef CASCADE_WEAKER_THRES_H_
#define CASCADE_WEAKER_THRES_H_

#include "weaker.h"

namespace cascade{

/// 阈值弱分类器
class WeakerThres:public Weaker{
public:

	/** 训练函数
	*
	* @param [in] const MatrixXf & features 特征矩阵
	* @param [in] const vector<int> & labels 标签向量
	* @param [in] const vector<float> & weights 权重向量
	* @param [in] const MatrixXi & index 特征矩阵排序索引
	* @param [in] const map<string, float> & parameters 训练参数
	* @return float 训练误差
	* @note 本分类器不需要任何具体的训练参数
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

	/** 返回弱分类器不等号方向，小于阈值为正则返回1，否则返回-1 */
	inline int bias() const {return bias_;};

	/** 返回弱分类器阈值 */
	inline float threshold() const {return threshold_;};
private:

	/** 计算阈值
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [in] const vector<int> & labels 标签向量
	 * @param [in] const vector<float> & weights 权重矩阵
	 * @param [in] const MatrixXi & index 特征矩阵排序索引
	 * @param [in] int col 计算的特征序号
	 * @param [out] float & error 训练误差
	 * @return WeakerThres 该特征的最佳阈值分类器
	 * @note  
	 */
	friend WeakerThres FindThreshold(const MatrixXf& features, 
		const vector<int>& labels, const vector<float>& weights,
		const MatrixXi& index, int col, float& error);
private:
	int bias_; /**< 不等号方向{-1,1} */
	float threshold_; /**< 分类器阈值 */
	static shared_ptr<char> pass_value_; /**< MPI进程间参数传递数据指针 */
};

} //namespace cascade

#endif //CASCADE_WEAKER_THRES_H_