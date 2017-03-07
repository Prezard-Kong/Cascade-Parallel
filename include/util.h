#ifndef CASCADE_UTIL_H_
#define CASCADE_UTIL_H_

#define UNCOPABLE(classname) \
private:\
	classname(const classname&);\
	classname& operator=(const classname&);

#include <algorithm>
#include <string>
namespace cascade{

using std::string;

/** �������ݷָ�ʱ������id���������������е���ʼλ��
 *  
 * @param [in] int id ���̱��
 * @param [in] int p ��������
 * @param [in] int total_nums �������� 
 * @return int ��ʼλ��
 * @note  
 */
inline int DivideStart(int id, int p, int total_nums){
	return (long long int(id))*total_nums/p;
}
/** �������ݷָ�ʱ������id����������
 *  
 * @param [in] int id ���̱��
 * @param [in] int p ��������
 * @param [in] int total_nums �������� 
 * @return int ��������
 * @note  
 */
inline int DivideNums(int id, int p, int total_nums){
	return (long long int(id+1))*total_nums/p - long long int(id)*total_nums/p;
}

/** �����������е�i��λ��Ӧ���ڵڼ�������
 *  
 * @param [in] int i �������������е�λ��
 * @param [in] int p ��������
 * @param [in] int total_nums �������� 
 * @return int Ӧ���Ľ��̱��
 * @note  
 */
inline int Owner(int i, int p, int total_nums){
	return (long long int(p)*(i+1)-1)/total_nums;
}

/** �ж�ֵ�Ƿ����0
 *  
 * @param [in] float val �жϵ�ֵ
 * @return bool �Ƿ�Ϊ0
 * @note  
 */
inline bool IsZero(float val){
	return std::abs(val)<1e-6;
}

/** �ж��Ƿ�Ϊһ��ͼ��
 *  
 * @param [in] const string & path �ļ�·��
 * @return bool �Ƿ�Ϊͼ��
 * @note ֧��bmp jpg jpeg png tif tiff��ʽ
 */
bool IsImage( const string& path );

} // namespace cascade
#endif // CASCADE_UTIL_H_