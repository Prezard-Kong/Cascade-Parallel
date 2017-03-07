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

/// Cascade��
class Cascade{
public:

	/** ��ȡCascadeʵ�� */
	static Cascade& Get();

	/** ѵ������
	 *  
	 * @param [in] const string & features_path ���������ļ�·������������������
	 * @param [in] const string & labels_path ��ǩ�����ļ�·��
	 * @param [in] const string & hard_path ���������ļ���·��
	 * @param [in] const Config & config ���ò���
	 * @param [in] const string & model_path ģ�ͱ���·��
	 * @return void 
	 * @note ���ջᱣ��ģ��xml����Ӧ����xml��ʹ��ģ��ʱ��Ҫ���߶�Ӧ
	 */
	void Train(const string& features_path, const string& labels_path,
		const string& hard_path, const Config& config, 
		const string& model_path);

	/** Ԥ�⺯�������õ��̷߳�ʽ
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [out] vector<int> & anwser Ԥ����
	 * @return void 
	 * @note  
	 */
	void Predict(const MatrixXf& features, vector<int>& anwser) const;

	/** ģ�ͱ���
	 *  
	 * @param [in] const string & path ģ�ͱ���·��
	 * @return void 
	 * @note  
	 */
	void SaveModel(const string& path);

	/** ģ�Ͷ�ȡ
	 *  
	 * @param [in] const string & path ģ�Ͷ�ȡ·��
	 * @return void 
	 * @note  
	 */
	void LoadModel(const string& path);

private:
	vector<shared_ptr<Boost>> model_; /**< ʵ��ģ������ */
	static shared_ptr<Cascade> instance_; /**< Cascade��ʵ�� */

private:

	/** ˽�л����캯����ͨ��Get��ȡʵ������ֹ��ʽʵ���� */
	Cascade(){};

	/** ��ȡģ��ѵ���ϵ�
	 *  
	 * @param [in] const string & path ģ���ļ�·��
	 * @param [in] int layers ��ʼ����
	 * @return void 
	 * @note  
	 */
	void LoadBreak(const string& path,int layers);

	/** ����ģ��ͷ��Ϣ
	 *  
	 * @param [in] const string & path ģ���ļ�·��
	 * @param [in] const string & type ������������
	 * @return void 
	 * @note  
	 */
	void SaveHead(const string& path, const string& type);

	/** ����ģ��ѵ���ϵ�
	 *  
	 * @param [in] const string & path ģ���ļ�·��
	 * @return void 
	 * @note  
	 */
	void SaveBreak(const string& path);

private:
	/** Ԥ�⺯�������õ��̷߳�ʽ
	*
	* @param [in] const MatrixXf & features ��������
	* @param [out] vector<int> & anwser Ԥ����
	* @return void
	* @note
	*/
	void PredictForTrain(const MatrixXf& features, vector<int>& anwser) const;

	/* ��Ԫ�����������滻��������ʱ���з��� */
	friend int PredictHardExample(const Cascade& model, MatrixXf &feat);

	/* ��ģ�Ͷ�Ӧ�Ĳ����ı���·��������ģ��·���Զ����� */
	string param_path_;

UNCOPABLE(Cascade);
};

} // namespace cascade

#endif // CASCADE_COMMON_H_