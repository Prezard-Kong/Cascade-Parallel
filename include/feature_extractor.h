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

/// 多进程特征提取，只有根进程保有全部数据 
class FeatureExtractor{
public:

	/** 提取图像大小已经归一化的正负样本特征
	 *  
	 * @param [in] const Config & pathes 样本所在路径
	 * @param [out] MatrixXf & features 特征矩阵结果
	 * @param [out] vector<int> & labels 标签向量结果
	 * @return void 
	 * @note  Config类中只需要正负样本以及特征函数dll的路径
	 */
	static void NormalExtractorToMat(const Config& pathes,MatrixXf& features,
		vector<int>& labels);

	/** 提取背景图像的负例样本特征
	 *  
	 * @param [in] const Config & pathes 属性配置
	 * @param [out] MatrixXf & features 特征矩阵结果
	 * @return void 
	 * @note  Config类中需要提取框的大小、步长、背景图像以及特征函数dll的路径
	 */
	static void HardSampleExtractorToMat(const Config& pathes,MatrixXf& features);

	/** 提取正负样本以及背景图像的特征
	 *  
	 * @param [in] const Config & pathes 属性配置
	 * @param [out] MatrixXf & sample_features 正负样本特征矩阵结果
	 * @param [out] vector<int> & labels 标签向量结果
	 * @param [out] MatrixXf & hard_sample_features 负例样本特征矩阵结果
	 * @return void 
	 * @note  
	 */
	static void ExtractorToMat(const Config& pathes, MatrixXf& sample_features,
		vector<int>& labels,MatrixXf& hard_sample_features);

	/** 提取单张图像中的所有特征（for detect)
	 *  
	 * @param [in] const cv::Mat & img_resized 尺度变化后的图像 
	 * @param [in] const Config & config 特征参数
	 * @param [in] int step_x 检测水平步长
	 * @param [in] int step_y 检测竖直步长
	 * @param [out] MatrixXf & features 特征矩阵
	 * @param [out] MatrixXi & locs 位置矩阵
	 * @return void 
	 * @note  
	 */
	static void MonoExtractor(const cv::Mat& img_resized, const Config& config, 
		int step_x, int step_y, MatrixXf& features, MatrixXi& locs);

	/** 提取单张图像中特定点的特征(for detect)
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

	/** 提取单张图像中特定点的特征(for detect)
	 *  
	 * @param [in] const cv::Mat & img 原始图像
	 * @param [in] const Config & config 参数设置
	 * @param [in] int pt_x 矩形框左上角横坐标
	 * @param [in] int pt_y 矩形框左上角纵坐标
	 * @param [in] int block_width 矩形宽度
	 * @param [in] int block_height 矩形高度
	 * @param [in] bool trigger 开关
	 * @param [out] MatrixXf & feature 特征提取结果
	 * @return void 
	 * @note  trigger 为开关，表示是否更换图像，防止特征提取函数重复计算
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config, 
		int pt_x, int pt_y, int block_width, int block_height,
		bool trigger, MatrixXf& feature);

	/** 提取单张图像中不同框大小的所有特征（for detect)
	 *  
	 * @param [in] const cv::Mat & img 原始图像
	 * @param [in] const Config & config 参数设置
	 * @param [in] int min_width 矩形框最小宽度
	 * @param [in] int min_height 矩形框最小高度
	 * @param [in] int max_width 矩形框最大宽度
	 * @param [in] int max_height 矩形框最大高度
	 * @param [in] int width_step 矩形框大小变化水平步长
	 * @param [in] int height_step 矩形框大小变化竖直步长
	 * @param [in] double step_ratio 矩形框移动步长（与尺寸的比例）
	 * @param [out] MatrixXf & features 特征提取结果
	 * @param [out] MatrixXi & rects 矩形框信息
	 * @return void 
	 * @note  
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config,
		int min_width, int min_height, int max_width, int max_height, int width_step,
		int height_step, double step_ratio, MatrixXf& features, MatrixXi& rects);

	/** 提取单张图像中不同框大小的所有特征（for detect)
	 *  
	 * @param [in] const cv::Mat & img 原始图像
	 * @param [in] const Config & config 参数设置
	 * @param [in] int min_width 矩形框最小宽度
	 * @param [in] int min_height 矩形框最小高度
	 * @param [in] int max_width 矩形框最大宽度
	 * @param [in] int max_height 矩形框最大高度
	 * @param [in] int width_step 矩形框大小变化水平步长
	 * @param [in] int height_step 矩形框大小变化竖直步长
	 * @param [in] int x_step 矩形框水平移动步长
	 * @param [in] int y_step 矩形框竖直移动步长
	 * @param [out] MatrixXf & features 特征提取结果
	 * @param [out] MatrixXi & rects 矩形框信息
	 * @return void 
	 * @note  
	 */
	static void MonoExtractorWithTrigger(const cv::Mat& img, const Config& config,
		int min_width, int min_height, int max_width, int max_height, int width_step,
		int height_step, int x_step, int y_step, MatrixXf& features, MatrixXi& rects);
private:
	/* 私有化构造函数，禁止实例化 */
	FeatureExtractor(){};
};

} // namespace cascade
#endif // CASCADE_FEATURE_EXTRACTOR_H_