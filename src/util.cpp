#include "util.h"

#include <vector>

namespace cascade{
using std::vector;
bool IsImage( const string& path ){
	int idx = path.find_last_of('.');
	string tmp(&path[idx+1]);
	vector<string> format;
	format.push_back("jpg");
	format.push_back("bmp");
	format.push_back("png");
	format.push_back("tif");
	format.push_back("tiff");
	format.push_back("jpeg");
	for (int i = 0;i<format.size();i++){
		if (format[i]==tmp){
			return true;
		}
	}
	return false;
}
}
