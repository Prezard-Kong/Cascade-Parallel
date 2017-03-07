#include "weaker_factory.h"

#include <iostream>

namespace cascade{

shared_ptr<Weaker> WeakerFactory::SetUp( const string& type ){
	if (type == "threshold"){
		return shared_ptr<Weaker>(new WeakerThres);
	}
	else if (type == "logistic"){
		return shared_ptr<Weaker>(new WeakerLogistic);
	}
	else{
		std::cerr<<type<<" is not defined."<<std::endl;
		std::cerr<<"please reset your params"<<std::endl;
		std::cerr<<"procedure is going to exit"<<std::endl;
		system("pause");
		exit(0);
	}
}

}