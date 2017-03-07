#ifndef CASCADE_WEAKER_THRES_H_
#define CASCADE_WEAKER_THRES_H_

#include "weaker.h"

namespace cascade{

/// ��ֵ��������
class WeakerThres:public Weaker{
public:

	/** ѵ������
	*
	* @param [in] const MatrixXf & features ��������
	* @param [in] const vector<int> & labels ��ǩ����
	* @param [in] const vector<float> & weights Ȩ������
	* @param [in] const MatrixXi & index ����������������
	* @param [in] const map<string, float> & parameters ѵ������
	* @return float ѵ�����
	* @note ������������Ҫ�κξ����ѵ������
	*/
	virtual float Train(const MatrixXf& features, const vector<int>& labels, 
		const vector<float>& weights, const MatrixXi& index,
		const map<string, float>& parameters = map<string, float>());

	/** Ԥ�⺯�������õ��̵߳ķ�ʽ
	*
	* @param [in] const MatrixXf & features ��������
	* @param [out] vector<int> & answer Ԥ��������
	* @return void
	* @note
	*/
	virtual void Predict(const MatrixXf& features, vector<int>& answer);

	/** ģ�ͱ���
	*
	* @param [in] tinyxml2::XMLDocument & doc xml�ĵ�
	* @return tinyxml2::XMLElement* ���ݱ����xml��㣬�����������������
	* @note
	*/
	virtual	tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc);

	/** ģ�Ͷ�ȡ
	*
	* @param [in] tinyxml2::XMLElement * header ���ݱ������ڵĽ��
	* @return void
	* @note
	*/
	virtual void LoadModel(tinyxml2::XMLElement* header);

	/** ע���Զ����MPI_Datatype */
	virtual MPI_Datatype PassType();

	/** ��pass_value_�е�ֵд�����ĳ�Ա������ */
	virtual void FromPassValue();

	/** ������ĳ�Ա�����е�ֵд��pass_value_ */
	virtual void ToPassValue();

	virtual char* pass_value() const;

	/** ���������������Ⱥŷ���С����ֵΪ���򷵻�1�����򷵻�-1 */
	inline int bias() const {return bias_;};

	/** ��������������ֵ */
	inline float threshold() const {return threshold_;};
private:

	/** ������ֵ
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [in] const vector<int> & labels ��ǩ����
	 * @param [in] const vector<float> & weights Ȩ�ؾ���
	 * @param [in] const MatrixXi & index ����������������
	 * @param [in] int col ������������
	 * @param [out] float & error ѵ�����
	 * @return WeakerThres �������������ֵ������
	 * @note  
	 */
	friend WeakerThres FindThreshold(const MatrixXf& features, 
		const vector<int>& labels, const vector<float>& weights,
		const MatrixXi& index, int col, float& error);
private:
	int bias_; /**< ���Ⱥŷ���{-1,1} */
	float threshold_; /**< ��������ֵ */
	static shared_ptr<char> pass_value_; /**< MPI���̼������������ָ�� */
};

} //namespace cascade

#endif //CASCADE_WEAKER_THRES_H_