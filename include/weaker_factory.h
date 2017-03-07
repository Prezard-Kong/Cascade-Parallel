#ifndef CASCADE_WEAKER_FACTORY_H_
#define CASCADE_WEAKER_FACTORY_H_

#include <string>
#include <memory>

#include "util.h"
#include "weaker.h"
#include "weaker_thres.h"
#include "weaker_logistic.h"

namespace cascade{

using std::shared_ptr;
using std::string;

/// 弱分类器工厂类 
class WeakerFactory{
public:

	/** 利用工厂类生成不同类型的弱分类器
	 *  
	 * @param [in] const string & type 弱分类器类型
	 * @return shared_ptr<Weaker> 弱分类器指针
	 * @note  
	 */
	static shared_ptr<Weaker> SetUp(const string& type);

private:

	/* 私有化构造函数，通过SetUp函数实例化，禁止显式实例化 */
	WeakerFactory(){};

UNCOPABLE(WeakerFactory);
};

} // namespace cascade

#endif // CASCADE_WEAKER_FACTORY_H_