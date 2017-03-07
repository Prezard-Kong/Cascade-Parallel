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

/// ǿ������Boost��
class Boost{
public:

	/** ���캯��
	 *  
	 * @param [in] const string & type ������������������
	 * @return  
	 * @note  
	 */
	explicit Boost(const string& type):type_(type){};

	/** ѵ������
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [in] vector<int> & labels ��ǩ����
	 * @param [in] float max_false_pos_rate ����龯��
	 * @param [in] float min_recall ��С�ٻ���
	 * @param [in] int max_weaker_num ���������������
	 * @param [in] const map<string,float> & parameters ��������ѵ������
	 * @return void 
	 * @note  
	 */
	void Train(const MatrixXf& features, vector<int>& labels, 
		float max_false_pos_rate, float min_recall, int max_weaker_num,
		const map<string,float>& parameters);

	/** Ԥ�⺯�������õ��̵߳ķ�ʽ
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [out] vector<int> & answers Ԥ����
	 * @return void 
	 * @note  
	 */
	void Predict(const MatrixXf& features, vector<int>& answers);

	/** ģ�ͱ���
	 *  
	 * @param [in] tinyxml2::XMLDocument & doc xml�ĵ�
	 * @return tinyxml2::XMLElement* ģ�ͱ����xml��㣬���ں������������
	 * @note  
	 */
	tinyxml2::XMLElement* SaveModel(tinyxml2::XMLDocument& doc);

	/** ģ�Ͷ�ȡ
	 *  
	 * @param [in] tinyxml2::XMLElement * header ���ݱ������ڵĽ��
	 * @return void 
	 * @note  
	 */
	void LoadModel(tinyxml2::XMLElement* header);

	/** ���������������� */
	inline int weaker_num() const {return boost_.size();};

	/** ����ǿ��������ֵ */
	inline float threshold() const {return threshold_;};

	/** ����ѵ��ʱ���ٻ��ʽ�� */
	inline float recall() const {return recall_;};

	/** ����ѵ��ʱ���龯�ʽ�� */
	inline float false_pos_rate() const {return false_pos_rate_;};

	/** ������������������ */
	inline string type() const {return type_;};

	/** ����ǿ��������ֵ������ģ�͵��ֶ����� */
	inline void set_threshold(float val){threshold_ = val;};
private:
	vector<shared_ptr<Weaker>> boost_; /**< ʵ��ģ������ */
	float threshold_; /**< ǿ��������ֵ */
	float recall_; /**< ѵ��ʱ���ٻ��� */
	float false_pos_rate_; /**< ѵ��ʱ���龯�� */
	string type_; /**< ������������ */

UNCOPABLE(Boost);
};

} // namespace cascade

#endif // CASCADE_BOOST_H_