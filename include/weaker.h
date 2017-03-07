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

/// ���������ӿڣ��������
class Weaker{
public:

	/** ѵ������
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [in] const vector<int> & labels ��ǩ����
	 * @param [in] const vector<float> & weights Ȩ������
	 * @param [in] const MatrixXi & index ����������������
	 * @param [in] const map<string, float> & parameters ѵ������
	 * @return float ѵ�����
	 * @note  
	 */
	virtual float Train(const MatrixXf& features, const vector<int>& labels, 
		const vector<float>& weights, const MatrixXi& index,
		const map<string,float>& parameters = map<string, float>())=0;

	/** Ԥ�⺯�������õ��̵߳ķ�ʽ
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [out] vector<int> & answer Ԥ��������
	 * @return void 
	 * @note  
	 */
	virtual void Predict(const MatrixXf& features, vector<int>& answer)=0;

	/** ģ�ͱ���
	 *  
	 * @param [in] tinyxml2::XMLDocument & doc xml�ĵ�
	 * @return tinyxml2::XMLElement* ���ݱ����xml��㣬�����������������
	 * @note  
	 */
	virtual tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc)=0;

	/** ģ�Ͷ�ȡ
	 *  
	 * @param [in] tinyxml2::XMLElement * header ���ݱ������ڵĽ��
	 * @return void 
	 * @note  
	 */
	virtual void LoadModel(tinyxml2::XMLElement *header)=0;

	/** ע���Զ����MPI_Datatype */
	virtual MPI_Datatype PassType()=0;

	/** ��pass_value_�е�ֵд�����ĳ�Ա������ */
	virtual void FromPassValue()=0;

	/** ������ĳ�Ա�����е�ֵд��pass_value_ */
	virtual void ToPassValue()=0; 

	/** ����pass_value_ָ������ */
	virtual char* pass_value() const = 0;

	/** �������� */
	virtual ~Weaker(){};

	/** ��������������Ȩ�� */
	inline float alpha() const {return alpha_;};

	/** ���������������õ�������� */
	inline int feat_index() const {return feat_index_;};

	/** ���������������õ��������
	 *  
	 * @param [in] int index �������
	 * @return void 
	 * @note  ���ڲ��ö��̣߳�ÿ�����������е���������Ǿֲ��ģ�
	 *		  �øú���������޸�Ϊȫ�ֵ�
	 */
	inline void set_feat_index(int index){feat_index_ = index;};

	/** ����pass_value_ָ������ */
// 	inline char* pass_value() const {
// 		return pass_value_.get();
// 	};

protected:

	/** ��������������Ȩ��
	 *  
	 * @param [in] float error ѵ�����
	 * @return void 
	 * @note  
	 */
	inline void calc_alpha(float error){
		float tmp = log((1-error)/(error + 1e-6));
		alpha_ = tmp<3?tmp:3; //���ڽضϣ���ֹĳһ��������Ȩ�ع���//
	};

protected:

	float alpha_; /**< ��������Ȩ�� */
	int feat_index_; /**< �����������õ�������� */
	
};

} // namespace cascade
#endif // CASCADE_WEAKER_H_