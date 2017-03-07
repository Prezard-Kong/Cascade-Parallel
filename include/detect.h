#ifndef _CASCADE_DETECT_H_
#define _CASCADE_DETECT_H_

#include <opencv2/opencv.hpp>

#include "common.h"
#include "util.h"

namespace cascade{

///Cascade Ŀ��̽����
class CascadeDetector {
public:
	/** Ĭ�Ϲ��캯��, ִ������ǰ��Ҫ��ȡģ�� */
	CascadeDetector();

	/** ���캯��
	 *  
	 * @param [in] const string & model_path ģ��·��
	 * @return  
	 * @note  
	 */
	CascadeDetector(const string& model_path);

	/** ��ȡģ��
	 *  
	 * @param [in] const string & model_path ģ��·��
	 * @return void 
	 * @note  
	 */
	void LoadModel(const string& model_path);

	/** Ŀ��̽�⣬���ϲ���
	 *  
	 * @param [in] const cv::Mat & img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [out] vector<cv::Rect> & rects ̽�������ο�
	 * @return void 
	 * @note �ڲ��ڶ���߶�ͼ���Ͻ�����̽�� 
	 */
	void DetectNoGrouping(const cv::Mat& img, 
		const Config& config, vector<cv::Rect>& rects);

	/** Ŀ��̽�⣬�γ�����ͼ
	 *  
	 * @param [in] const cv::Mat & img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [out] cv::Mat & heatmap �������ͼ
	 * @return void 
	 * @note  ����ͼ��ÿһ������Ǹõ�Ŀ�ĸ���
	 */
	void DetectHeatMap(const cv::Mat& img, const Config& config, cv::Mat& heatmap);

	/** Ŀ��̽�⣬���к�ѡ��ϲ�
	 *  
	 * @param [in] const cv::Mat img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [in] const int thres �ϲ���ֵ
	 * @param [out] vector<cv::Rect> & rects ̽�������ο�
	 * @return void 
	 * @note  ���һ���㱻���ںϲ���ֵ����ѡ�򸲸ǣ�������������ͨ���б������γɽ����
	 */
	void Detect(const cv::Mat img, const Config& config, const int thres, 
		vector<cv::Rect>& rects);

	/** Ŀ��̽�⣬���ϲ���
	 *  
	 * @param [in] const cv::Mat & img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [in] int min_block_width ��С���ο���
	 * @param [in] int min_block_height ��С���ο�߶�
	 * @param [in] int max_block_width �����ο���
	 * @param [in] int max_block_heigth �����ο�߶�
	 * @param [in] int width_step ���ο�ˮƽ�ߴ�仯����
	 * @param [in] int height_step ���ο���ֱ�ߴ�仯����
	 * @param [in] double move_step_ratio ���ο��ƶ�����ռ���ο�ߴ�ı���
	 * @param [out] vector<cv::Rect> & rects ̽�������ο�
	 * @return void 
	 * @note  
	 */
	void DetectNoGrouping(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step, 
		int height_step, double move_step_ratio, vector<cv::Rect>& rects);

	/** Ŀ��̽�⣬�γ�����ͼ
	 *  
	 * @param [in] const cv::Mat & img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [in] int min_block_width ��С���ο���
	 * @param [in] int min_block_height ��С���ο�߶�
	 * @param [in] int max_block_width �����ο���
	 * @param [in] int max_block_heigth �����ο�߶�
	 * @param [in] int width_step ���ο�ˮƽ�ߴ�仯����
	 * @param [in] int height_step ���ο���ֱ�ߴ�仯����
	 * @param [in] double move_step_ratio ���ο��ƶ�����ռ���ο�ߴ�ı���
	 * @param [out] cv::Mat & heatmap �������ͼ
	 * @return void 
	 * @note  ����ͼ��ÿһ������Ǹõ�Ŀ�ĸ���
	 */
	void DetectHeatMap(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step,
		int height_step, double move_step_ratio, cv::Mat& heatmap);

	/** Ŀ��̽�⣬���к�ѡ��ϲ�
	 *  
	 * @param [in] const cv::Mat & img �����ͼ��
	 * @param [in] const Config & config ģ�Ͳ���
	 * @param [in] int min_block_width ��С���ο���
	 * @param [in] int min_block_height ��С���ο�߶�
	 * @param [in] int max_block_width �����ο���
	 * @param [in] int max_block_heigth �����ο�߶�
	 * @param [in] int width_step ���ο�ˮƽ�ߴ�仯����
	 * @param [in] int height_step ���ο���ֱ�ߴ�仯����
	 * @param [in] double move_step_ratio ���ο��ƶ�����ռ���ο�ߴ�ı���
	 * @param [in] int thres �ϲ���ֵ
	 * @param [out] vector<cv::Rect> & rects ̽�������ο�
	 * @return void 
	 * @note  ���һ���㱻���ںϲ���ֵ����ѡ�򸲸ǣ�������������ͨ���б������γɽ����
	 */
	void Detect(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step,
		int height_step, double move_step_ratio, int thres, vector<cv::Rect>& rects);
private:
	Cascade* classifier_; /**< ģ��ָ�� */

UNCOPABLE(CascadeDetector);
};

} // namespace cascade

#endif // _CASCADE_DETECT_H_

