#ifndef CASCADE_DATA_IO_H_
#define CASCADE_DATA_IO_H_

#include <string>
#include <vector>

#include "Eigen/Dense"

namespace cascade{

using std::vector;
using std::string;
using Eigen::MatrixXf;

/// 用于特征和标签的输入输出 
class DataIO{
public:

	/** 将矩阵保存为TXT文件
	 *  
	 * @param [in] const MatrixXf & mat 输入矩阵
	 * @param [in] const string & save_path 保存路径
	 * @return bool 是否保存成功
	 * @note  如不需要查看数据，建议保存为二进制文件以节省空间
	 */
	static bool MatToTxt(const MatrixXf& mat, const string& save_path);

	/** 从TXT文件读取数据矩阵
	 *  
	 * @param [in] const string & load_path 读取路径
	 * @param [out] MatrixXf & mat 读取的数据矩阵
	 * @return bool 是否读取成功
	 * @note  
	 */
	static bool TxtToMat(const string& load_path, MatrixXf& mat);

	/** 将向量保存为TXT
	 *  
	 * @param [in] const vector<int> & vec 输入向量
	 * @param [in] const string & save_path 保存路径
	 * @return bool 是否保存成功
	 * @note  如不需要查看数据，建议保存为二进制文件以节省空间
	 */
	static bool VecToTxt(const vector<int>& vec, const string& save_path);

	/** 从TXT文件读取数据向量
	 *  
	 * @param [in] const string & load_path 读取路径
	 * @param [out] vector<int> & vec 读取的数据向量
	 * @return bool 是否读取成功
	 * @note  
	 */
	static bool TxtToVec(const string& load_path, vector<int>& vec);

	/** 将矩阵保存为二进制文件
	 *  
	 * @param [in] const MatrixXf & mat 输入矩阵
	 * @param [in] const string & save_path 保存路径
	 * @return bool 是否保存成功
	 * @note  
	 */
	static bool MatToBin(const MatrixXf& mat, const string& save_path);

	/** 从二进制文件读取数据矩阵
	 *  
	 * @param [in] const string & load_path 读取路径
	 * @param [out] MatrixXf & mat 读取的数据矩阵
	 * @return bool 是否读取成功
	 * @note  
	 */
	static bool BinToMat(const string& load_path, MatrixXf& mat);

	/** 将向量保存为二进制文件
	 *  
	 * @param [in] const vector<int> & vec 输入向量
	 * @param [in] const string & save_path 保存路径
	 * @return bool 是否保存成功
	 * @note  
	 */
	static bool VecToBin(const vector<int>& vec, const string& save_path);

	/** 从二进制文件读取数据向量
	 *  
	 * @param [in] const string & load_path 读取路径
	 * @param [out] vector<int> & vec 读取的数据向量
	 * @return bool 是否读取成功 
	 * @note  
	 */
	static bool BinToVec(const string& load_path, vector<int>& vec);

	/** 读取TXT特征矩阵文件中的一行（已废弃）
	 *  
	 * @param [in] const string & load_path 读取路径
	 * @param [in] int row 读取的行序号
	 * @param [out] vector<float> & vec 读取结果
	 * @return bool 是否读取成功
	 * @note  
	 */
	static bool LoadHardExampleVec(const string& load_path, int row,vector<float>& vec);

	/** 将一个大的负例样本文件分割为若干小文件
	 *  
	 * @param [in] const string & origin_path 原始文件路径
	 * @param [in] const string & save_path 保存文件夹路径
	 * @return bool 是否成功
	 * @note  分割为小文件后可以加快数据切换的速度
	 */
	static bool DivideHardExamples(const string& origin_path, const string& save_path);

	/** 将负例样本特征矩阵按一定大小保存为若干TXT文件
	 *  
	 * @param [in] const MatrixXf & features 特征矩阵
	 * @param [in] const string & path 保存文件夹路径
	 * @param [in] int start_num 保存开始的序号（避免已经保存好的文件被覆盖）
	 * @return bool 是否保存成功
	 * @note  
	 */
	static bool SaveHardExamplesToTxt(const MatrixXf& features, const string& path, 
		int start_num = 0);

	/** 将负例样本特征矩阵按一定大小保存为若干二进制文件
	*
	* @param [in] const MatrixXf & features 特征矩阵
	* @param [in] const string & path 保存文件夹路径
	* @param [in] int start_num 保存开始的序号（避免已经保存好的文件被覆盖）
	* @return bool 是否保存成功
	* @note
	*/
	static bool SaveHardExamplesToBin(const MatrixXf& features, const string& path,
		int start_num = 0);

	/** 统计一个TXT特征矩阵文件中的行数
	 *  
	 * @param [in] const string & load_path 路径
	 * @return int 行数
	 * @note  
	 */
	static int CountRows(const string& load_path);

	/** 统计文件夹下包含多少个负例样本文件
	 *  
	 * @param [in] const string & path 文件夹路径
	 * @return vector<string> 所有负例样本文件路径
	 * @note  
	 */
	static vector<string> HardExampleFiles(const string& path);

private:
	/* 私有化构造函数，禁止实例化 */
	DataIO(){};
};

} // namespace cascade

#endif // CASCADE_DATA_IO_H_