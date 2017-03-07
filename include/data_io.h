#ifndef CASCADE_DATA_IO_H_
#define CASCADE_DATA_IO_H_

#include <string>
#include <vector>

#include "Eigen/Dense"

namespace cascade{

using std::vector;
using std::string;
using Eigen::MatrixXf;

/// ���������ͱ�ǩ��������� 
class DataIO{
public:

	/** �����󱣴�ΪTXT�ļ�
	 *  
	 * @param [in] const MatrixXf & mat �������
	 * @param [in] const string & save_path ����·��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  �粻��Ҫ�鿴���ݣ����鱣��Ϊ�������ļ��Խ�ʡ�ռ�
	 */
	static bool MatToTxt(const MatrixXf& mat, const string& save_path);

	/** ��TXT�ļ���ȡ���ݾ���
	 *  
	 * @param [in] const string & load_path ��ȡ·��
	 * @param [out] MatrixXf & mat ��ȡ�����ݾ���
	 * @return bool �Ƿ��ȡ�ɹ�
	 * @note  
	 */
	static bool TxtToMat(const string& load_path, MatrixXf& mat);

	/** ����������ΪTXT
	 *  
	 * @param [in] const vector<int> & vec ��������
	 * @param [in] const string & save_path ����·��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  �粻��Ҫ�鿴���ݣ����鱣��Ϊ�������ļ��Խ�ʡ�ռ�
	 */
	static bool VecToTxt(const vector<int>& vec, const string& save_path);

	/** ��TXT�ļ���ȡ��������
	 *  
	 * @param [in] const string & load_path ��ȡ·��
	 * @param [out] vector<int> & vec ��ȡ����������
	 * @return bool �Ƿ��ȡ�ɹ�
	 * @note  
	 */
	static bool TxtToVec(const string& load_path, vector<int>& vec);

	/** �����󱣴�Ϊ�������ļ�
	 *  
	 * @param [in] const MatrixXf & mat �������
	 * @param [in] const string & save_path ����·��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  
	 */
	static bool MatToBin(const MatrixXf& mat, const string& save_path);

	/** �Ӷ������ļ���ȡ���ݾ���
	 *  
	 * @param [in] const string & load_path ��ȡ·��
	 * @param [out] MatrixXf & mat ��ȡ�����ݾ���
	 * @return bool �Ƿ��ȡ�ɹ�
	 * @note  
	 */
	static bool BinToMat(const string& load_path, MatrixXf& mat);

	/** ����������Ϊ�������ļ�
	 *  
	 * @param [in] const vector<int> & vec ��������
	 * @param [in] const string & save_path ����·��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  
	 */
	static bool VecToBin(const vector<int>& vec, const string& save_path);

	/** �Ӷ������ļ���ȡ��������
	 *  
	 * @param [in] const string & load_path ��ȡ·��
	 * @param [out] vector<int> & vec ��ȡ����������
	 * @return bool �Ƿ��ȡ�ɹ� 
	 * @note  
	 */
	static bool BinToVec(const string& load_path, vector<int>& vec);

	/** ��ȡTXT���������ļ��е�һ�У��ѷ�����
	 *  
	 * @param [in] const string & load_path ��ȡ·��
	 * @param [in] int row ��ȡ�������
	 * @param [out] vector<float> & vec ��ȡ���
	 * @return bool �Ƿ��ȡ�ɹ�
	 * @note  
	 */
	static bool LoadHardExampleVec(const string& load_path, int row,vector<float>& vec);

	/** ��һ����ĸ��������ļ��ָ�Ϊ����С�ļ�
	 *  
	 * @param [in] const string & origin_path ԭʼ�ļ�·��
	 * @param [in] const string & save_path �����ļ���·��
	 * @return bool �Ƿ�ɹ�
	 * @note  �ָ�ΪС�ļ�����Լӿ������л����ٶ�
	 */
	static bool DivideHardExamples(const string& origin_path, const string& save_path);

	/** ������������������һ����С����Ϊ����TXT�ļ�
	 *  
	 * @param [in] const MatrixXf & features ��������
	 * @param [in] const string & path �����ļ���·��
	 * @param [in] int start_num ���濪ʼ����ţ������Ѿ�����õ��ļ������ǣ�
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  
	 */
	static bool SaveHardExamplesToTxt(const MatrixXf& features, const string& path, 
		int start_num = 0);

	/** ������������������һ����С����Ϊ���ɶ������ļ�
	*
	* @param [in] const MatrixXf & features ��������
	* @param [in] const string & path �����ļ���·��
	* @param [in] int start_num ���濪ʼ����ţ������Ѿ�����õ��ļ������ǣ�
	* @return bool �Ƿ񱣴�ɹ�
	* @note
	*/
	static bool SaveHardExamplesToBin(const MatrixXf& features, const string& path,
		int start_num = 0);

	/** ͳ��һ��TXT���������ļ��е�����
	 *  
	 * @param [in] const string & load_path ·��
	 * @return int ����
	 * @note  
	 */
	static int CountRows(const string& load_path);

	/** ͳ���ļ����°������ٸ����������ļ�
	 *  
	 * @param [in] const string & path �ļ���·��
	 * @return vector<string> ���и��������ļ�·��
	 * @note  
	 */
	static vector<string> HardExampleFiles(const string& path);

private:
	/* ˽�л����캯������ֹʵ���� */
	DataIO(){};
};

} // namespace cascade

#endif // CASCADE_DATA_IO_H_