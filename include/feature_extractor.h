#ifndef CASCADE_FEATURE_EXTRACTOR_H_
#define CASCADE_FEATURE_EXTRACTOR_H_

#include <string>

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"

#include "util.h"
#include "config.h"

namespace cascade{

using std::string;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

/// �����������ȡ��ֻ�и����̱���ȫ������ 
class FeatureExtractor{
public:

	/** ��ȡͼ���С�Ѿ���һ����������������
	 *  
	 * @param [in] const Config & pathes ��������·��
	 * @param [out] MatrixXf & features ����������
	 * @param [out] vector<int> & labels ��ǩ�������
	 * @return void 
	 * @note  Config����ֻ��Ҫ���������Լ���������dll��·��
	 */
	static void NormalExtractorToMat(const Config& pathes,MatrixXf& features,
		vector<int>& labels);

	/** ��ȡ����ͼ��ĸ�����������
	 *  
	 * @param [in] const Config & pathes ��������
	 * @param [out] MatrixXf & features ����������
	 * @return void 
	 * @note  Config������Ҫ��ȡ��Ĵ�С������������ͼ���Լ���������dll��·��
	 */
	static void HardSampleExtractorToMat(const Config& pathes,MatrixXf& features);

	/** ��ȡ���������Լ�����ͼ�������
	 *  
	 * @param [in] const Config & pathes ��������
	 * @param [out] MatrixXf & sample_features ������������������
	 * @param [out] vector<int> & labels ��ǩ�������
	 * @param [out] MatrixXf & hard_sample_features ������������������
	 * @return void 
	 * @note  
	 */
	static void ExtractorToMat(const Config& pathes, MatrixXf& sample_features,
		vector<int>& labels,MatrixXf& hard_sample_features);

	/** ��ȡ����ͼ���е�����������for detect)
	 *  
	 * @param [in] const cv::Mat & img_resized �߶ȱ仯���ͼ�� 
	 * @param [in] const Config & config ��������
	 * @param [in] int step_x ���ˮƽ����
	 * @param [in] int step_y �����ֱ����
	 * @param [out] MatrixXf & features ��������
	 * @param [out] MatrixXi & locs λ�þ���
	 * @return void 
	 * @note  
	 */
	static void MonoExtractor(const cv::Mat& img_resized, const Config& config, 
		int step_x, int step_y, MatrixXf& features, MatrixXi& locs);

	/** ��ȡ����ͼ�����ض��������(for detect)
	 *  
	 * @param [in] const cv::Mat & img_resized 
	 * @param [in] const Config & config 
	 * @param [in] int pt_x 
	 * @param [in] int pt_y 
	 * @param [in] int step_x 
	 * @param [in] int step_y 
	 * @param [out] MatrixXf & feature 
	 * @return void 
	 * @note  
	 */
	static void MonoExtractor(const cv::Mat& img_resized, const Config& config,
		int pt_x, int pt_y, MatrixXf& feature);

	/** ��ȡ����ͼ�����ض��������(for detect)
	 *  
	 * @param [in] const cv::Mat & img ԭʼͼ��
	 * @param [in] const Config & config ��������
	 * @param [in] int pt_x ���ο����ϽǺ�����
	 * @param [in] int pt_y ���ο����Ͻ�������
	 * @param [in] int block_width ���ο��
	 * @param [in] int block_height ���θ߶�
	 * @param [in] bool trigger ����
	 * @param [out] MatrixXf & feature ������ȡ���
	 * @return void 
	 * @note  trigger Ϊ���أ���ʾ�Ƿ����ͼ�񣬷�ֹ������ȡ�����ظ�����
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config, 
		int pt_x, int pt_y, int block_width, int block_height,
		bool trigger, MatrixXf& feature);

	/** ��ȡ����ͼ���в�ͬ���С������������for detect)
	 *  
	 * @param [in] const cv::Mat & img ԭʼͼ��
	 * @param [in] const Config & config ��������
	 * @param [in] int min_width ���ο���С���
	 * @param [in] int min_height ���ο���С�߶�
	 * @param [in] int max_width ���ο������
	 * @param [in] int max_height ���ο����߶�
	 * @param [in] int width_step ���ο��С�仯ˮƽ����
	 * @param [in] int height_step ���ο��С�仯��ֱ����
	 * @param [in] double step_ratio ���ο��ƶ���������ߴ�ı�����
	 * @param [out] MatrixXf & features ������ȡ���
	 * @param [out] MatrixXi & rects ���ο���Ϣ
	 * @return void 
	 * @note  
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config,
		int min_width, int min_height, int max_width, int max_height, int width_step,
		int height_step, double step_ratio, MatrixXf& features, MatrixXi& rects);

	/** ��ȡ����ͼ���в�ͬ���С������������for detect)
	 *  
	 * @param [in] const cv::Mat & img ԭʼͼ��
	 * @param [in] const Config & config ��������
	 * @param [in] int min_width ���ο���С���
	 * @param [in] int min_height ���ο���С�߶�
	 * @param [in] int max_width ���ο������
	 * @param [in] int max_height ���ο����߶�
	 * @param [in] int width_step ���ο��С�仯ˮƽ����
	 * @param [in] int height_step ���ο��С�仯��ֱ����
	 * @param [in] int x_step ���ο�ˮƽ�ƶ�����
	 * @param [in] int y_step ���ο���ֱ�ƶ�����
	 * @param [out] MatrixXf & features ������ȡ���
	 * @param [out] MatrixXi & rects ���ο���Ϣ
	 * @return void 
	 * @note  
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config,
		int min_width, int min_height, int max_width, int max_height, int width_step,
		int height_step, int x_step, int y_step, MatrixXf& features, MatrixXi& rects);
private:
	/* ˽�л����캯������ֹʵ���� */
	FeatureExtractor(){};
};

} // namespace cascade
#endif // CASCADE_FEATURE_EXTRACTOR_H_