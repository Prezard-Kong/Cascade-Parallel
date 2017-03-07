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

/// �����������ú������ļ��Ĵ�ȡ
class Config{
public:

	/* Ĭ�Ϲ��캯�� */
	Config(){};

	/** ���캯����ͨ�����εķ�ʽ���ø�����
	 *  
	 * @param [in] const vector<string> & pos_path �����������ļ���·��
	 * @param [in] const vector<string> & neg_path ���и������ļ���·��
	 * @param [in] const vector<string> & bak_path ���б��������ļ���·��
	 * @param [in] const vector<string> & func_path ����������ȡ����dll·��
	 * @param [in] int feat_num ��������
	 * @param [in] int block_width ������ȡ����
	 * @param [in] int block_height ������ȡ��߶�
	 * @param [in] int step_x ������ȡ���ڱ���ͼ�����ƶ���ˮƽ����
	 * @param [in] int step_y ������ȡ���ڱ���ͼ�����ƶ�����ֱ����
	 * @param [in] int max_neg_live ������������������
	 * @param [in] int start_layer ѵ����ʼ����
	 * @param [in] int max_layer_num ģ��������
	 * @param [in] int max_weaker_num ÿ����������������
	 * @param [in] const string & type ������������
	 * @param [in] const vector<float> & recall ÿ��ǿ��������С�ٻ���
	 * @param [in] const vector<float> & false_pos_rate ÿ��ǿ����������龯��
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

	/** ���캯����ͨ����ȡ�����ļ��ķ�ʽ���ø�����
	 *  
	 * @param [in] const string & sample_path ���������ļ�·��
	 * @param [in] const string & func_path ������ȡ���������ļ�·��
	 * @param [in] const string & config_path ���������ļ�·��
	 * @return  
	 * @note  
	 */
	Config(const string& sample_path,const string& func_path, const string& config_path);

	/** �������������ļ�
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @param [in] const vector<string> & pos_path �����������ļ���·�� 
	 * @param [in] const vector<string> & neg_path ���и������ļ���·��
	 * @param [in] const vector<string> & bak_path ���б��������ļ���·��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  
	 */
	static bool SavePath(const string& xml_path, const vector<string>& pos_path,
		const vector<string>& neg_path, const vector<string>& bak_path);

	/** ���浥��������ȡ���������ļ�
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @param [in] const string & name ������ȡ����dll·��
	 * @param [in] int feat_num ����������ȡ��������Ŀ
	 * @return bool �Ƿ񱣴�ɹ� 
	 * @note  
	 */
	static bool SaveFeatFunc(const string& xml_path,const string& name, int feat_num);

	/** ������������ļ�
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @param [in] const int block_width ������ȡ����
	 * @param [in] const int block_height ������ȡ��߶�
	 * @param [in] const int step_x ������ȡ���ڱ���ͼ�����ƶ���ˮƽ����
	 * @param [in] const int step_y ������ȡ���ڱ���ͼ�����ƶ�����ֱ����
	 * @param [in] const int max_neg_live ����������������
	 * @param [in] const int start_layer ѵ����ʼ����
	 * @param [in] const int max_layer_num ģ��������
	 * @param [in] const int max_weaker_num ÿ�����������������
	 * @param [in] const string & type ������������
	 * @param [in] const vector<float> & recall ÿ��ǿ��������С�ٻ���
	 * @param [in] const vector<float> & false_pos_rate ÿ��ǿ����������龯��
	 * @return bool �Ƿ񱣴�ɹ�
	 * @note  
	 */
	static bool SaveConfig(const string& xml_path,const int block_width, 
		const int block_height,const int step_x,const int step_y,const int max_neg_live,
		const int start_layer,const int max_layer_num,const int max_weaker_num,
		const string& type, const vector<float>& recall,
		const vector<float>& false_pos_rate);

	/** ��·�������ļ��ж�ȡ����
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @return void 
	 * @note  
	 */
	void LoadPath(const string& xml_path);

	/** ��������ȡ���������ļ��ж�ȡ����
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @return void 
	 * @note  
	 */
	void LoadAllFunc(const string& xml_path);

	/** �Ӳ��������ļ��ж�ȡ����
	 *  
	 * @param [in] const string & xml_path �����ļ�·��
	 * @return void 
	 * @note  
	 */
	void LoadConfig(const string& xml_path);

	/** �������ļ��ж�ȡ������
	 *  
	 * @param [in] const string & sample_path ���������ļ�·��
	 * @param [in] const string & func_path ������ȡ���������ļ�·��
	 * @param [in] const string & config_path ���������ļ�·��
	 * @return void 
	 * @note  
	 */
	void Load(const string& sample_path, const string& func_path, const string& config_path);

	/** ��ȡ��������ѵ������
	 *  
	 * @param [in] const string & xml_path ����·��
	 * @return void 
	 * @note  
	 */
	void LoadWeakerConfig(const string& xml_path);
	
	/** ��ȡ��������ѵ������
	 *  
	 * @param [out] map<string, float> & parameters ��������ѵ������
	 * @return void 
	 * @note  
	 */
	void GetWeakerConfig(map<string, float>& parameters) const;

	/** ���������������ļ���·�� */
	inline vector<string> pos_path() const {return pos_path_;};

	/** �������и������ļ���·�� */
	inline vector<string> neg_path() const {return neg_path_;};

	/** �������б��������ļ���·�� */
	inline vector<string> bak_path() const {return bak_path_;};

	/** ��������������ȡ����dll�ļ�·�� */
	inline vector<string> func_path() const {return func_path_;};

	/** ����ָ�����������ļ���·�� */
	inline string pos_path(const int i) const {return pos_path_[i];};

	/** ����ָ���ĸ������ļ���·�� */
	inline string neg_path(const int i) const {return neg_path_[i];};

	/** ����ָ���ı��������ļ���·�� */
	inline string bak_path(const int i) const {return bak_path_[i];};

	/** ����ָ����������ȡ����dll�ļ�·�� */
	inline string func_path(const int i) const {return func_path_[i];};

	/** �����������ļ��и��� */
	inline int pos_num() const {return pos_path_.size();};

	/** ���ظ������ļ��и��� */
	inline int neg_num() const {return neg_path_.size();};

	/** ���ر��������ļ��и��� */
	inline int bak_num() const {return bak_path_.size();};

	/** ����������ȡ�������� */
	inline int func_num() const {return func_path_.size();};

	/** ������ȡ���������� */
	inline int feat_num() const {return feat_num_;};

	/** ����������ȡ���� */
	inline int block_width() const {return block_width_;};

	/** ����������ȡ��ն� */
	inline int block_height() const {return block_height_;};

	/** ����������ȡ��ˮƽ���� */
	inline int step_x() const {return step_x_;};

	/** ����������ȡ����ֱ���� */
	inline int step_y() const {return step_y_;};

	/** ���ظ��������������� */
	inline int max_neg_live() const {return max_neg_live_;};

	/** ����ѵ����ʼ���� */
	inline int start_layer() const {return start_layer_;};

	/** ����ģ�������� */
	inline int max_layer_num() const {return max_layer_num_;};

	/** ����ÿ����������������� */
	inline int max_weaker_num() const {return max_weaker_num_;};

	/** ���������������� */
	inline string type() const {return type_;};

	/** ����ÿ����С�ٻ��� */
	inline vector<float> recall() const {return recall_;};

	/** ����ÿ������龯�� */
	inline vector<float> false_pos_rate() const {return false_pos_rate_;};

	/** ����ָ�������С�ٻ��� */
	inline float recall(const int i)const {return recall_[i];};

	/** ����ָ���������龯�� */
	inline float false_pos_rate(const int i)const{return false_pos_rate_[i];};
private:

	/** ��������������ȡ����
	 *  
	 * @param [in] const string & xml_path 
	 * @param [in] int & func_num 
	 * @param [in] int & feat_num 
	 * @return void 
	 * @note  
	 */
	void LoadAllFunc(const string& xml_path, int& func_num,int& feat_num);
private:
	//���²�������������ȡ//
	vector<string> pos_path_; /**< �����������ļ���·�� */
	vector<string> neg_path_; /**< ���и������ļ���·�� */
	vector<string> bak_path_; /**< ���б��������ļ���·�� */
	vector<string> func_path_; /**< ����������ȡ����dll�ļ�·�� */
	int feat_num_; /**< ��ȡ���������� */
	int block_width_; /**< ������ȡ���� */
	int block_height_; /**< ������ȡ��߶� */
	int step_x_; /**< ������ȡ��ˮƽ���� */
	int step_y_; /**< ������ȡ����ֱ���� */
	//����Ϊcascadeѵ������//
	int max_neg_live_; /**< ���������������� */
	int start_layer_; /**< ѵ����ʼ���� */
	int max_layer_num_; /**< ģ�������� */
	int max_weaker_num_; /**< ÿ���������������� */
	string type_; /**< ������������ */
	vector<float> recall_; /**< ÿ��ǿ��������С�ٻ��� */
	vector<float> false_pos_rate_; /**< ÿ��ǿ����������龯�� */

private:
	//��������ѵ����������
	class WeakerConfig{
	public:
		/** ��ȡ�������������ĺ����ӿ� */
		virtual void Load(const string& xml_path) = 0; 

		/**< ��ȡ�������������ĺ����ӿ� */
		virtual void Get(map<string, float>& parameters) = 0; 

		/** ������������ѵ��ʱ�Ƿ���Ҫ���� */
		inline bool need_sort() const { return need_sort_; };

		/** ������������ѵ��ʱ�Ƿ���Ҫ��׼�� */
		inline bool need_normalize() const { return need_normalize_; };
	protected:
		bool need_sort_; /**< �Ƿ���Ҫ���� */
		bool need_normalize_; /**< �Ƿ���Ҫ��׼�� */
	};

	//����ֵ��������ѵ������
	class ThresConfig:public WeakerConfig{
	public:
		/** ��ȡ������������ */
		virtual void Load(const string& xml_path);
		/** ��ȡ������������ */
		virtual void Get(map<string, float>& parameters);
	};

	//�������߼��ع���������ѵ������
	class LogisticConfig:public WeakerConfig{
	public:
		/** ��ȡ������������ */
		virtual void Load(const string& xml_path);
		/** ��ȡ������������ */
		virtual void Get(map<string, float>& parameters);
	private:
		float learning_rate_; /**< ѧϰ�� */
		float weight_decay_; /**< L2����ϵ�� */
		int batch_size_; /**< batch size */
		int max_iter_; /**< ���ѵ������ */
	};
private:
	shared_ptr<WeakerConfig> weaker_config; /**< ʵ����������ѵ������ */
};

} // namespace cascade

#endif // CASCADE_CONFIG_H_