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

/** 计算数据分割时，进程id的数据在总数据中的起始位置
 *  
 * @param [in] int id 进程编号
 * @param [in] int p 进程总数
 * @param [in] int total_nums 数据总数 
 * @return int 起始位置
 * @note  
 */
inline int DivideStart(int id, int p, int total_nums){
	return (long long int(id))*total_nums/p;
}
/** 计算数据分割时，进程id的数据数量
 *  
 * @param [in] int id 进程编号
 * @param [in] int p 进程总数
 * @param [in] int total_nums 数据总数 
 * @return int 数据数量
 * @note  
 */
inline int DivideNums(int id, int p, int total_nums){
	return (long long int(id+1))*total_nums/p - long long int(id)*total_nums/p;
}

/** 计算总数据中第i个位置应属于第几个进程
 *  
 * @param [in] int i 数据在总数据中的位置
 * @param [in] int p 进程总数
 * @param [in] int total_nums 数据总数 
 * @return int 应属的进程编号
 * @note  
 */
inline int Owner(int i, int p, int total_nums){
	return (long long int(p)*(i+1)-1)/total_nums;
}

/** 判断值是否等于0
 *  
 * @param [in] float val 判断的值
 * @return bool 是否为0
 * @note  
 */
inline bool IsZero(float val){
	return std::abs(val)<1e-6;
}

/** 判断是否为一幅图像
 *  
 * @param [in] const string & path 文件路径
 * @return bool 是否为图像
 * @note 支持bmp jpg jpeg png tif tiff格式
 */
bool IsImage( const string& path );

} // namespace cascade
#endif // CASCADE_UTIL_H_