#ifndef _CASCADE_DETECT_H_
#define _CASCADE_DETECT_H_

#include <opencv2/opencv.hpp>

#include "common.h"
#include "util.h"

namespace cascade{

///Cascade 目标探测器
class CascadeDetector {
public:
	/** 默认构造函数, 执行任务前需要读取模型 */
	CascadeDetector();

	/** 构造函数
	 *  
	 * @param [in] const string & model_path 模型路径
	 * @return  
	 * @note  
	 */
	CascadeDetector(const string& model_path);

	/** 读取模型
	 *  
	 * @param [in] const string & model_path 模型路径
	 * @return void 
	 * @note  
	 */
	void LoadModel(const string& model_path);

	/** 目标探测，不合并框
	 *  
	 * @param [in] const cv::Mat & img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [out] vector<cv::Rect> & rects 探测结果矩形框
	 * @return void 
	 * @note 内部在多个尺度图像上进行了探测 
	 */
	void DetectNoGrouping(const cv::Mat& img, 
		const Config& config, vector<cv::Rect>& rects);

	/** 目标探测，形成热力图
	 *  
	 * @param [in] const cv::Mat & img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [out] cv::Mat & heatmap 结果热力图
	 * @return void 
	 * @note  热力图中每一点代表覆盖该点的框的个数
	 */
	void DetectHeatMap(const cv::Mat& img, const Config& config, cv::Mat& heatmap);

	/** 目标探测，进行候选框合并
	 *  
	 * @param [in] const cv::Mat img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [in] const int thres 合并阈值
	 * @param [out] vector<cv::Rect> & rects 探测结果矩形框
	 * @return void 
	 * @note  如果一个点被多于合并阈值个候选框覆盖，则保留，最终连通所有保留点形成结果框
	 */
	void Detect(const cv::Mat img, const Config& config, const int thres, 
		vector<cv::Rect>& rects);

	/** 目标探测，不合并框
	 *  
	 * @param [in] const cv::Mat & img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [in] int min_block_width 最小矩形框宽度
	 * @param [in] int min_block_height 最小矩形框高度
	 * @param [in] int max_block_width 最大矩形框宽度
	 * @param [in] int max_block_heigth 最大矩形框高度
	 * @param [in] int width_step 矩形框水平尺寸变化步长
	 * @param [in] int height_step 矩形框竖直尺寸变化步长
	 * @param [in] double move_step_ratio 矩形框移动步长占矩形框尺寸的比例
	 * @param [out] vector<cv::Rect> & rects 探测结果矩形框
	 * @return void 
	 * @note  
	 */
	void DetectNoGrouping(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step, 
		int height_step, double move_step_ratio, vector<cv::Rect>& rects);

	/** 目标探测，形成热力图
	 *  
	 * @param [in] const cv::Mat & img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [in] int min_block_width 最小矩形框宽度
	 * @param [in] int min_block_height 最小矩形框高度
	 * @param [in] int max_block_width 最大矩形框宽度
	 * @param [in] int max_block_heigth 最大矩形框高度
	 * @param [in] int width_step 矩形框水平尺寸变化步长
	 * @param [in] int height_step 矩形框竖直尺寸变化步长
	 * @param [in] double move_step_ratio 矩形框移动步长占矩形框尺寸的比例
	 * @param [out] cv::Mat & heatmap 结果热力图
	 * @return void 
	 * @note  热力图中每一点代表覆盖该点的框的个数
	 */
	void DetectHeatMap(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step,
		int height_step, double move_step_ratio, cv::Mat& heatmap);

	/** 目标探测，进行候选框合并
	 *  
	 * @param [in] const cv::Mat & img 待检测图像
	 * @param [in] const Config & config 模型参数
	 * @param [in] int min_block_width 最小矩形框宽度
	 * @param [in] int min_block_height 最小矩形框高度
	 * @param [in] int max_block_width 最大矩形框宽度
	 * @param [in] int max_block_heigth 最大矩形框高度
	 * @param [in] int width_step 矩形框水平尺寸变化步长
	 * @param [in] int height_step 矩形框竖直尺寸变化步长
	 * @param [in] double move_step_ratio 矩形框移动步长占矩形框尺寸的比例
	 * @param [in] int thres 合并阈值
	 * @param [out] vector<cv::Rect> & rects 探测结果矩形框
	 * @return void 
	 * @note  如果一个点被多于合并阈值个候选框覆盖，则保留，最终连通所有保留点形成结果框
	 */
	void Detect(const cv::Mat& img, const Config& config, int min_block_width,
		int min_block_height, int max_block_width, int max_block_heigth, int width_step,
		int height_step, double move_step_ratio, int thres, vector<cv::Rect>& rects);
private:
	Cascade* classifier_; /**< 模型指针 */

UNCOPABLE(CascadeDetector);
};

} // namespace cascade

#endif // _CASCADE_DETECT_H_

