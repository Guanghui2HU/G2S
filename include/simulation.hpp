#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "samplingModule.hpp"
#include <thread>


void simulation(FILE *logFile,g2s::DataImage &di, std::vector<g2s::DataImage> &TIs, g2s::DataImage &kernel, SamplingModule &samplingModule,
 std::vector<std::vector<int> > &pathPosition, unsigned* solvingPath, unsigned numberOfPointToSimulate, float* seedAray, unsigned* importDataIndex,
  unsigned numberNeighbor, unsigned nbThreads=1, float fastSimulation=0.f){

	unsigned* posterioryPath=(unsigned*)malloc( sizeof(unsigned) * di.dataSize()/di._nbVariable);
	memset(posterioryPath,255,sizeof(unsigned) * di.dataSize()/di._nbVariable);
	for (int i = 0; i < di.dataSize()/di._nbVariable; ++i)
	{
		bool isPureNan=true;
		for (int j = 0; j < di._nbVariable; ++j)
		{
			isPureNan&=std::isnan(di._data[i*di._nbVariable+j]);
		}
		if(!isPureNan)
			posterioryPath[i]=0;
	}
	for (int i = 0; i < numberOfPointToSimulate; ++i)
	{
		posterioryPath[solvingPath[i]]=i+1;
	}

	
	std::mt19937 *randGenArray;
	if(fastSimulation!=0.f){
		std::mt19937 randgen(seedAray[0]);
		randGenArray= new std::mt19937[nbThreads];
		for (int i = 0; i < nbThreads; ++i)
		{
			randGenArray[i].seed(randgen());
		}
	}
	
	unsigned numberOfVariable=di._nbVariable;
	#pragma omp parallel for num_threads(nbThreads) schedule(dynamic,1) default(none) firstprivate(numberOfPointToSimulate, fastSimulation, \
		posterioryPath, solvingPath, seedAray, numberNeighbor, importDataIndex, logFile) shared( pathPosition, di, samplingModule, TIs)
	for (int indexPath = 0; indexPath < numberOfPointToSimulate; ++indexPath){
		
		/*if(indexPath<TIs[0].dataSize()/TIs[0]._nbVariable-300){
			unsigned currentCell=solvingPath[indexPath];
			memcpy(di._data+currentCell*di._nbVariable,TIs[0]._data+currentCell*TIs[0]._nbVariable,TIs[0]._nbVariable*sizeof(float));
			continue;
		}*/

		unsigned moduleID=0;
		#if _OPENMP
			moduleID=omp_get_thread_num();
		#endif
		unsigned currentCell=solvingPath[indexPath];
		float localSeed=seedAray[indexPath];

		bool isDone=true;
		for (int j = 0; j < di._nbVariable; ++j)
		{
			isDone&=!std::isnan(di._data[currentCell*di._nbVariable+j]);
		}
		if(isDone) continue;

		std::vector<std::vector<int> > neighborArrayVector;
		std::vector<std::vector<float> > neighborValueArrayVector;
		
		unsigned positionSearch=0;
		while((neighborArrayVector.size()<numberNeighbor)&&(positionSearch<pathPosition.size())){
			unsigned dataIndex;
			if(di.indexWithDelta(dataIndex, currentCell, pathPosition[positionSearch]))
			{
				if(posterioryPath[dataIndex]<indexPath+1){
					std::vector<float> data(di._nbVariable);
					unsigned numberOfNaN=0;
					float val;
					while(true) {
						numberOfNaN=0;
						for (int i = 0; i < di._nbVariable; ++i)
						{
							#pragma omp atomic read
							val=di._data[dataIndex*di._nbVariable+i];
							numberOfNaN+=std::isnan(val);
							data[i]=val;
						}
						if(numberOfNaN==0)break;
						std::this_thread::sleep_for(std::chrono::microseconds(250));
					}
					neighborValueArrayVector.push_back(data);
					neighborArrayVector.push_back(pathPosition[positionSearch]);
				}
			}
			positionSearch++;
		}

		SamplingModule::matchLocation importIndex;

		if(neighborArrayVector.size()>0){
			importIndex=samplingModule.sample(neighborArrayVector,neighborValueArrayVector,localSeed,moduleID);
		}else{
			// sample from the marginal
			unsigned cumulated=0;
			for (int i = 0; i < TIs.size(); ++i)
			{
				cumulated+=TIs[i].dataSize();
			}
			
			unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)))*TIs[0]._nbVariable;

			cumulated=0;
			for (int i = 0; i < TIs.size(); ++i)
			{
				if(position<cumulated+TIs[i].dataSize()){
					importIndex.TI=i;
					importIndex.index=TIs[i]._data[position-cumulated];
					break;
				}else{
					cumulated+=TIs[i].dataSize();
				}
			}

			bool hasNaN=false;

			for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
			{
				if(std::isnan(TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j])){
					hasNaN=true;
				}
			}
		
			if(hasNaN){ // nan safe, much slower
				unsigned cumulated=0;
				for (int i = 0; i < TIs.size(); ++i)
				{
					for (int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
					{
						bool locHasNan=false;
						for (int j = 0; j < TIs[i]._nbVariable; ++j)
						{
							locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
						}
						cumulated+=!locHasNan;
					}
				}
				unsigned position=int(floor(localSeed*(cumulated/TIs[0]._nbVariable)))*TIs[0]._nbVariable;

				cumulated=0;

				for (int i = 0; i < TIs.size(); ++i)
				{
					for (int k = 0; k < TIs[i].dataSize()/TIs[i]._nbVariable; ++k)
					{
						bool locHasNan=false;
						for (int j = 0; j < TIs[i]._nbVariable; ++j)
						{
							locHasNan|=std::isnan(TIs[i]._data[k*TIs[i]._nbVariable+j]);
						}
						cumulated+=!locHasNan;
						if(position>=cumulated){
							importIndex.TI=i;
							importIndex.index=k;
							break;
						}
					}
					if(position>=cumulated)break;
				}
			}
		}
		// import data
		for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
		{
			if(std::isnan(di._data[currentCell*di._nbVariable+j])){
				#pragma omp atomic write
				di._data[currentCell*di._nbVariable+j]=TIs[importIndex.TI]._data[importIndex.index*TIs[importIndex.TI]._nbVariable+j];
			}
		}
		#pragma omp atomic write
		importDataIndex[currentCell]=importIndex.index*TIs.size()+importIndex.TI;

		if((fastSimulation!=0.f) && (neighborArrayVector.size()>=numberNeighbor)){
			std::uniform_real_distribution<float> uniformDis(0.f,1.f);
			unsigned indexCenter=0;
			for (int i =  kernel._dims.size()-1; i>=0 ; i--)
			{
				indexCenter=indexCenter*kernel._dims[i]+kernel._dims[i]/2;
			}
			unsigned AoN=std::min(numberNeighbor,unsigned(positionSearch*fastSimulation));
			for (int k = 0; k < AoN; ++k)
			{
				unsigned indexInKernel;
				unsigned dataIndex;
				kernel.indexWithDelta(indexInKernel, indexCenter, pathPosition[k]);
				bool isSetable=uniformDis(randGenArray[moduleID])<numberNeighbor/AoN*fastSimulation;//*(1.0-k/(positionSearch*fastSimulation);
				if (isSetable && di.indexWithDelta(dataIndex, currentCell, pathPosition[k])){
					unsigned dataIndexImport;
					if(!TIs[importIndex.TI].indexWithDelta(dataIndexImport, importIndex.index, pathPosition[k])) continue;
					for (int j = 0; j < TIs[importIndex.TI]._nbVariable; ++j)
					{
						if(std::isnan(di._data[dataIndex*di._nbVariable+j]))
						{
							#pragma omp atomic write
							di._data[dataIndex*di._nbVariable+j]=TIs[importIndex.TI]._data[dataIndexImport*TIs[importIndex.TI]._nbVariable+j];
							#pragma omp atomic write
							importDataIndex[dataIndex]=dataIndexImport*TIs.size()+importIndex.TI;
						}
					}
				}
			}
		}
		if(indexPath%(numberOfPointToSimulate/100)==0)fprintf(logFile, "progress : %.2f%%\n",float(indexPath)/numberOfPointToSimulate*100);
	}

	free(posterioryPath);
	if(fastSimulation!=0.f) delete[] randGenArray;
}

#endif // SIMULATION_HPP