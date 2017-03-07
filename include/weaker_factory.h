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

/// �������������� 
class WeakerFactory{
public:

	/** ���ù��������ɲ�ͬ���͵���������
	 *  
	 * @param [in] const string & type ������������
	 * @return shared_ptr<Weaker> ��������ָ��
	 * @note  
	 */
	static shared_ptr<Weaker> SetUp(const string& type);

private:

	/* ˽�л����캯����ͨ��SetUp����ʵ��������ֹ��ʽʵ���� */
	WeakerFactory(){};

UNCOPABLE(WeakerFactory);
};

} // namespace cascade

#endif // CASCADE_WEAKER_FACTORY_H_