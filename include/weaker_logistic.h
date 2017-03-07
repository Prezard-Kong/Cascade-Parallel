#ifndef CASCADE_WEAKER_LOGISTIC_H_
#define CASCADE_WEAKER_LOGISTIC_H_

#include "weaker.h"

namespace cascade{

///�߼��ع���������
class WeakerLogistic : public Weaker{
public:

	/** ѵ������
	*
	* @param [in] const MatrixXf & features ��������
	* @param [in] const vector<int> & labels ��ǩ����
	* @param [in] const vector<float> & weights Ȩ������
	* @param [in] const MatrixXi & index ����������������
	* @param [in] const map<string, float> & parameters ѵ������
	* @return float ѵ�����
	* @note ����������Ҫѵ������������(ѧϰ�ʣ�learning_rate),
	* (L2����ϵ����weight_decay), (����ݶ��½�batch size��batch_size),
	* (������������max_iter)
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

	/** �����߼��ع�ϵ�� */
	inline float theta() const { return theta_; };

	/** �����߼��ع�ƫ�� */
	inline float bias() const { return bias_; };

private:
	float theta_; /**< �ع�ϵ�� */
	float bias_; /**< �ع�ƫ�� */
	static shared_ptr<char> pass_value_; /**< MPI���̼������������ָ�� */
};

}
#endif // !CASCADE_WEAKER_LOGISTIC_H_
