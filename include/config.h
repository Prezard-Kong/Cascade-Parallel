#ifndef CASCADE_CONFIG_H_
#define CASCADE_CONFIG_H_

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace cascade{

using std::string;
using std::vector;
using std::shared_ptr;
using std::map;

/// 用于属性配置和配置文件的存取
class Config{
public:

	/* 默认构造函数 */
	Config(){};

	/** 构造函数，通过传参的方式配置各数据
	 *  
	 * @param [in] const vector<string> & pos_path 所有正样本文件夹路径
	 * @param [in] const vector<string> & neg_path 所有负样本文件夹路径
	 * @param [in] const vector<string> & bak_path 所有背景样本文件夹路径
	 * @param [in] const vector<string> & func_path 所有特征提取函数dll路径
	 * @param [in] int feat_num 特征总数
	 * @param [in] int block_width 特征提取框宽度
	 * @param [in] int block_height 特征提取框高度
	 * @param [in] int step_x 特征提取框在背景图像中移动的水平步长
	 * @param [in] int step_y 特征提取框在背景图像中移动的竖直步长
	 * @param [in] int max_neg_live 负例样本存活的最大层数
	 * @param [in] int start_layer 训练起始层数
	 * @param [in] int max_layer_num 模型最大层数
	 * @param [in] int max_weaker_num 每层弱分类器最大个数
	 * @param [in] const string & type 弱分类器类型
	 * @param [in] const vector<float> & recall 每层强分类器最小召回率
	 * @param [in] const vector<float> & false_pos_rate 每层强分类器最大虚警率
	 * @return  
	 * @note  
	 */
	Config(const vector<string>& pos_path, const vector<string>& neg_path,
		const vector<string>& bak_path, const vector<string>& func_path,
		int feat_num, int block_width, int block_height, int step_x, int step_y,
		int max_neg_live, int start_layer, int max_layer_num, int max_weaker_num,
		const string& type,const vector<float>& recall, 
		const vector<float>& false_pos_rate): 
		pos_path_(pos_path),neg_path_(neg_path),
		bak_path_(bak_path),func_path_(func_path),feat_num_(feat_num), 
		block_width_(block_width),block_height_(block_height),step_x_(step_x),
		step_y_(step_y),max_neg_live_(max_neg_live),start_layer_(start_layer),
		max_layer_num_(max_layer_num),max_weaker_num_(max_weaker_num),
		type_(type),recall_(recall),false_pos_rate_(false_pos_rate){};

	/** 构造函数，通过读取配置文件的方式配置各数据
	 *  
	 * @param [in] const string & sample_path 样本配置文件路径
	 * @param [in] const string & func_path 特征提取函数配置文件路径
	 * @param [in] const string & config_path 参数配置文件路径
	 * @return  
	 * @note  
	 */
	Config(const string& sample_path,const string& func_path, const string& config_path);

	/** 保存样本配置文件
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @param [in] const vector<string> & pos_path 所有正样本文件夹路径 
	 * @param [in] const vector<string> & neg_path 所有负样本文件夹路径
	 * @param [in] const vector<string> & bak_path 所有背景样本文件夹路径
	 * @return bool 是否保存成功
	 * @note  
	 */
	static bool SavePath(const string& xml_path, const vector<string>& pos_path,
		const vector<string>& neg_path, const vector<string>& bak_path);

	/** 保存单个特征提取函数配置文件
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @param [in] const string & name 特征提取函数dll路径
	 * @param [in] int feat_num 特征函数提取的特征数目
	 * @return bool 是否保存成功 
	 * @note  
	 */
	static bool SaveFeatFunc(const string& xml_path,const string& name, int feat_num);

	/** 保存参数配置文件
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @param [in] const int block_width 特征提取框宽度
	 * @param [in] const int block_height 特征提取框高度
	 * @param [in] const int step_x 特征提取框在背景图像中移动的水平步长
	 * @param [in] const int step_y 特征提取框在背景图像中移动的竖直步长
	 * @param [in] const int max_neg_live 负例样本最大存活层数
	 * @param [in] const int start_layer 训练起始层数
	 * @param [in] const int max_layer_num 模型最大层数
	 * @param [in] const int max_weaker_num 每层最大弱分类器个数
	 * @param [in] const string & type 弱分类器类型
	 * @param [in] const vector<float> & recall 每层强分类器最小召回率
	 * @param [in] const vector<float> & false_pos_rate 每层强分类器最大虚警率
	 * @return bool 是否保存成功
	 * @note  
	 */
	static bool SaveConfig(const string& xml_path,const int block_width, 
		const int block_height,const int step_x,const int step_y,const int max_neg_live,
		const int start_layer,const int max_layer_num,const int max_weaker_num,
		const string& type, const vector<float>& recall,
		const vector<float>& false_pos_rate);

	/** 从路径配置文件中读取数据
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @return void 
	 * @note  
	 */
	void LoadPath(const string& xml_path);

	/** 从特征提取函数配置文件中读取数据
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @return void 
	 * @note  
	 */
	void LoadAllFunc(const string& xml_path);

	/** 从参数配置文件中读取数据
	 *  
	 * @param [in] const string & xml_path 配置文件路径
	 * @return void 
	 * @note  
	 */
	void LoadConfig(const string& xml_path);

	/** 从配置文件中读取各数据
	 *  
	 * @param [in] const string & sample_path 样本配置文件路径
	 * @param [in] const string & func_path 特征提取函数配置文件路径
	 * @param [in] const string & config_path 参数配置文件路径
	 * @return void 
	 * @note  
	 */
	void Load(const string& sample_path, const string& func_path, const string& config_path);

	/** 读取弱分类器训练参数
	 *  
	 * @param [in] const string & xml_path 参数路径
	 * @return void 
	 * @note  
	 */
	void LoadWeakerConfig(const string& xml_path);
	
	/** 获取弱分类器训练参数
	 *  
	 * @param [out] map<string, float> & parameters 弱分类器训练参数
	 * @return void 
	 * @note  
	 */
	void GetWeakerConfig(map<string, float>& parameters) const;

	/** 返回所有正样本文件夹路径 */
	inline vector<string> pos_path() const {return pos_path_;};

	/** 返回所有负样本文件夹路径 */
	inline vector<string> neg_path() const {return neg_path_;};

	/** 返回所有背景样本文件夹路径 */
	inline vector<string> bak_path() const {return bak_path_;};

	/** 返回所有特征提取函数dll文件路径 */
	inline vector<string> func_path() const {return func_path_;};

	/** 返回指定的正样本文件夹路径 */
	inline string pos_path(const int i) const {return pos_path_[i];};

	/** 返回指定的负样本文件夹路径 */
	inline string neg_path(const int i) const {return neg_path_[i];};

	/** 返回指定的背景样本文件夹路径 */
	inline string bak_path(const int i) const {return bak_path_[i];};

	/** 返回指定的特征提取函数dll文件路径 */
	inline string func_path(const int i) const {return func_path_[i];};

	/** 返回正样本文件夹个数 */
	inline int pos_num() const {return pos_path_.size();};

	/** 返回负样本文件夹个数 */
	inline int neg_num() const {return neg_path_.size();};

	/** 返回背景样本文件夹个数 */
	inline int bak_num() const {return bak_path_.size();};

	/** 返回特征提取函数个数 */
	inline int func_num() const {return func_path_.size();};

	/** 返回提取的特征总数 */
	inline int feat_num() const {return feat_num_;};

	/** 返回特征提取框宽度 */
	inline int block_width() const {return block_width_;};

	/** 返回特征提取框刚度 */
	inline int block_height() const {return block_height_;};

	/** 返回特征提取框水平步长 */
	inline int step_x() const {return step_x_;};

	/** 返回特征提取框竖直步长 */
	inline int step_y() const {return step_y_;};

	/** 返回负例样本最大存活层数 */
	inline int max_neg_live() const {return max_neg_live_;};

	/** 返回训练起始层数 */
	inline int start_layer() const {return start_layer_;};

	/** 返回模型最大层数 */
	inline int max_layer_num() const {return max_layer_num_;};

	/** 返回每层最大弱分类器个数 */
	inline int max_weaker_num() const {return max_weaker_num_;};

	/** 返回弱分类器类型 */
	inline string type() const {return type_;};

	/** 返回每层最小召回率 */
	inline vector<float> recall() const {return recall_;};

	/** 返回每层最大虚警率 */
	inline vector<float> false_pos_rate() const {return false_pos_rate_;};

	/** 返回指定层的最小召回率 */
	inline float recall(const int i)const {return recall_[i];};

	/** 返回指定层的最大虚警率 */
	inline float false_pos_rate(const int i)const{return false_pos_rate_[i];};
private:

	/** 加载所有特征提取函数
	 *  
	 * @param [in] const string & xml_path 
	 * @param [in] int & func_num 
	 * @param [in] int & feat_num 
	 * @return void 
	 * @note  
	 */
	void LoadAllFunc(const string& xml_path, int& func_num,int& feat_num);
private:
	//以下参数用于特征提取//
	vector<string> pos_path_; /**< 所有正样本文件夹路径 */
	vector<string> neg_path_; /**< 所有负样本文件夹路径 */
	vector<string> bak_path_; /**< 所有背景样本文件夹路径 */
	vector<string> func_path_; /**< 所有特征提取函数dll文件路径 */
	int feat_num_; /**< 提取的特征总数 */
	int block_width_; /**< 特征提取框宽度 */
	int block_height_; /**< 特征提取框高度 */
	int step_x_; /**< 特征提取框水平步长 */
	int step_y_; /**< 特征提取框竖直步长 */
	//以下为cascade训练参数//
	int max_neg_live_; /**< 负例样本最大存活层数 */
	int start_layer_; /**< 训练起始层数 */
	int max_layer_num_; /**< 模型最大层数 */
	int max_weaker_num_; /**< 每层弱分类器最大个数 */
	string type_; /**< 弱分类器类型 */
	vector<float> recall_; /**< 每层强分类器最小召回率 */
	vector<float> false_pos_rate_; /**< 每层强分类器最大虚警率 */

private:
	//弱分类器训练参数基类
	class WeakerConfig{
	public:
		/** 读取弱分类器参数的函数接口 */
		virtual void Load(const string& xml_path) = 0; 

		/**< 获取弱分类器参数的函数接口 */
		virtual void Get(map<string, float>& parameters) = 0; 

		/** 返回弱分类器训练时是否需要排序 */
		inline bool need_sort() const { return need_sort_; };

		/** 返回弱分类器训练时是否需要标准化 */
		inline bool need_normalize() const { return need_normalize_; };
	protected:
		bool need_sort_; /**< 是否需要排序 */
		bool need_normalize_; /**< 是否需要标准化 */
	};

	//单阈值弱分类器训练参数
	class ThresConfig:public WeakerConfig{
	public:
		/** 读取弱分类器参数 */
		virtual void Load(const string& xml_path);
		/** 获取弱分类器参数 */
		virtual void Get(map<string, float>& parameters);
	};

	//单变量逻辑回归弱分类器训练参数
	class LogisticConfig:public WeakerConfig{
	public:
		/** 读取弱分类器参数 */
		virtual void Load(const string& xml_path);
		/** 获取弱分类器参数 */
		virtual void Get(map<string, float>& parameters);
	private:
		float learning_rate_; /**< 学习率 */
		float weight_decay_; /**< L2正则化系数 */
		int batch_size_; /**< batch size */
		int max_iter_; /**< 最大训练步数 */
	};
private:
	shared_ptr<WeakerConfig> weaker_config; /**< 实际弱分类器训练参数 */
};

} // namespace cascade

#endif // CASCADE_CONFIG_H_