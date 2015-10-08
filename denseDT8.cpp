#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/flann/flann.hpp"
#define BUILD_VOCAB 0

using namespace cv;
using namespace std;

char classes[11][20]={"close","pour","open","spread","scoop","take","fold", "shake", "put","stir","x"};
//char classes[11][20]={"close","pour","open","spread","scoop","take","grate", "shake", "put","stir","crack","spray", "unwrap","cut","x"};

struct feature{
	int size;
	vector<int> index;
	vector<int> val;
};



Mat kMeansCluster(Mat &data,int clusterSize, int type = CV_8UC1)
{ 
	/**
	 *  implements k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
	 *  
	 *  Parameters: # data- Each sample in rows. Type should be CV_32FC1
	 *              # clusterSize- Number of clusters.
	 **/

	TermCriteria termCriteria(CV_TERMCRIT_ITER,100,0.01);
	int nAttempts=3;
	int flags=KMEANS_PP_CENTERS;

	Mat clusterCenters, temp;
	kmeans(data, clusterSize, temp, termCriteria, nAttempts, flags, clusterCenters);
	clusterCenters.convertTo(clusterCenters,type);
	return clusterCenters;
}

Mat hiKMeansCluster(Mat &data,int clusterSize, int type = CV_8UC1)
{   
	/**
	 *  implements Hierarchical k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
	 *
	 *  Parameters: # data- Each sample in rows. Type should be CV_32FC1.
	 *              # clusterSize- Number of clusters.
	 **/
	Mat clusterCenters=Mat(clusterSize,data.cols,CV_32F);
	cvflann::KMeansIndexParams kParams = cvflann::KMeansIndexParams(2, 1*clusterSize, cvflann::FLANN_CENTERS_KMEANSPP,0.2);
	int numClusters =cv::flann::hierarchicalClustering<cvflann::L2<float> >(data, clusterCenters, kParams);
	clusterCenters = clusterCenters.rowRange(cv::Range(0,numClusters));
	clusterCenters.convertTo(clusterCenters, type);

	return clusterCenters;
}

void writeCSV(string filename, Mat m) 
{
	cv::Formatter const * c_formatter(cv::Formatter::get("CSV"));
	c_formatter->write(myfile, m);
	myfile.close();
}

void writeToYMLFile(Mat &dataToWrite, char *fileName)
{
	/**
	 *  writes the data to the .yml file.
	 *
	 *  Parameters: # dataToWrite - matrix to write in .yml file.
	 *              # fileName - name of yml file.
	 **/

	stringstream ss;
	ss<<fileName<<".yml";
	string s = ss.str();
	FileStorage fileStorage(s, FileStorage::WRITE);
	s= string(fileName);

	fileStorage << s << dataToWrite;

	fileStorage.release();
}


void writeToBinaryFile(Mat &dataToWrite , char *fileName)
{
	/**
	 *  writes the data to the .bin file .
	 *  
	 *  Parameters: # dataToWrite - matrix to write in binbary file. Type CV_8UC1.
	 *              # fileName - name of binary file.
	 **/

	fstream binaryFile(fileName,ios::binary|ios::out);
	if(!binaryFile.is_open())
	{
		printf("\nerror in opening: %s", fileName);
		return;
	}

	binaryFile.write((char *)dataToWrite.data, dataToWrite.rows*dataToWrite.cols) ;

	binaryFile.close();
}


int get_label(char *a)
{
	for(int i=0;i<11;i++)
	{
		if(!strcmp(a,classes[i]))
			return i;
	}

	return 11;
}

Mat get_feature_video(char *command)
{
	FILE *temp_file = popen(command,"r");

	Mat feature_video = Mat(0,436,CV_32FC1);
	while(!feof(temp_file))
	{
		float feature_array[436];
		for(int i=0;i<436 && !feof(temp_file);i++)
		{
			fscanf(temp_file, "%f", &feature_array[i]);
		}

		Mat feature_row = Mat(1,436,CV_32FC1, &feature_array);

		feature_video.push_back(feature_row);
	}
	fclose(temp_file);

	return feature_video;
}

feature get_feature(Mat a)
{
	feature f;
	f.size =0;
	for(int i=0;i<a.cols;i++)
	{
		if(a.at<int>(0,i)>0)
		{
			f.size++;
			f.index.push_back(i);
			f.val.push_back(a.at<int>(0,i));
		}
	}

	return f;
}


void get_video_BOFhist_one_by_one(vector<Mat> &vocabulary, char *path, vector<feature> &feature_histogram, int skip, char *prefix, ofstream csvfile)
{

	FILE *fp = fopen(path, "r");
	int status=0;

	//cv::flann::KMeansIndexParams indexParams0,indexParams1,indexParams2,indexParams3,indexParams4;
	cv::flann::KDTreeIndexParams indexParams0,indexParams1,indexParams2,indexParams3,indexParams4;
	cv::flann::Index kdtree0(vocabulary[0], indexParams0);
	cv::flann::Index kdtree1(vocabulary[1], indexParams1);
	cv::flann::Index kdtree2(vocabulary[2], indexParams2);
	cv::flann::Index kdtree3(vocabulary[3], indexParams3);
	cv::flann::Index kdtree4(vocabulary[4], indexParams4);



	vector<int> index2(1);
	vector<float> dist(1);


	while(!feof(fp))
	{
		char feature_file_name[200], command[200], label[50];

		for(int i=0;i<=skip;i++)
			fscanf(fp,"%s%s", feature_file_name, label);

		//        sprintf(command, "cat %s > %s", feature_file_name, prefix);
		//        system(command);
		strcat(feature_file_name,".avi");
		sprintf(command, "/home/suriya/dense_trajectory_release_v1.2/release/DenseTrack %s", feature_file_name);

		Mat feature_video;
		feature_video.push_back(get_feature_video(command));
		/* 
		////////////////// bidt////////////////////        
		sprintf(command, "tac %s > %s", feature_file_name, prefix);
		system(command);
		sprintf(command, "/home/suriya/dense_trajectory_release_v1.2/release/DenseTrack %s", prefix);

		feature_video.push_back(get_feature_video(command));
		 */

		if(feature_video.rows<2)
			continue;

		feature_video = feature_video(Rect(10,0,feature_video.cols-10, feature_video.rows)).clone();

		Mat BOFhist;
		vector<Mat> B;
		for(int i=0;i<5;i++)
		{
			Mat b = Mat::zeros(1,vocabulary[i].rows,CV_32SC1 );
			B.push_back(b);
		}

		for(int j=0;j<feature_video.rows;j++)
		{
			Mat r = feature_video.row(j).clone();

			Mat t = r(Rect(0,0,30,1)).clone();
			kdtree0.knnSearch(t, index2, dist, 1, cv::flann::SearchParams(32));
			B[0].at<int>(0,index2[0]) = B[0].at<int>(0,index2[0])+1;

			t = r(Rect(30,0,96,1)).clone();
			kdtree1.knnSearch(t, index2, dist, 1, cv::flann::SearchParams(32));
			B[1].at<int>(0,index2[0]) = B[1].at<int>(0,index2[0])+1;

			t = r(Rect(30+96,0,108,1)).clone();
			kdtree2.knnSearch(t, index2, dist, 1, cv::flann::SearchParams(32));
			B[2].at<int>(0,index2[0]) = B[2].at<int>(0,index2[0])+1;

			t = r(Rect(30+96+108,0,96,1)).clone();
			kdtree3.knnSearch(t, index2, dist, 1, cv::flann::SearchParams(32));
			B[3].at<int>(0,index2[0]) = B[3].at<int>(0,index2[0])+1;

			t = r(Rect(30+96+108+96,0,96,1)).clone();
			kdtree4.knnSearch(t, index2, dist, 1, cv::flann::SearchParams(32));
			B[4].at<int>(0,index2[0]) = B[4].at<int>(0,index2[0])+1;
		}

		//B[0] = B[0]/norm(B[0]);	
		BOFhist = B[0].clone();
		for(int i=1;i<5;i++)
		{
			//B[i] = B[i]/norm(B[i]);	
			hconcat(BOFhist, B[i], BOFhist);
		}

		Mat l = Mat(1,1,CV_32S);
		l.at<int>(0,0) = get_label(label);

		hconcat(l, BOFhist, BOFhist);

		//print_here
		csvfile << format(BOFhist,"csv") << endl;

		feature f = get_feature(BOFhist);

		feature_histogram.push_back(f);


		printf("\n%d %dx%d : %s\t[label = %s (%d)] %d",status++, feature_histogram.size(), BOFhist.cols, feature_file_name, label, get_label(label) , f.size );
	}

	fclose(fp);
}


void writeToBinaryFile(vector<feature> feature_histogram, int feature_size , char *fileName)
{
	fstream binaryFile(fileName, ios::out | ios::binary);
	if(!binaryFile.is_open())
	{
		printf("\nerror in opening: %s", fileName);
		return;
	}
	char name[100];
	sprintf(name, "%s_sizes.txt", fileName);
	FILE *fp = fopen(name,"w");
	fprintf(fp,"%d %d\n", feature_histogram.size(), feature_size);
	for(int i=0;i<feature_histogram.size();i++)
	{
		fprintf(fp,"%d\n", feature_histogram[i].size);
		for(int j=0;j<feature_histogram[i].size;j++)
		{

			int index = feature_histogram[i].index[j];
			int val = feature_histogram[i].val[j];
			binaryFile.write((char *)&index,sizeof(index)) ;
			binaryFile.write((char *)&val,sizeof(val)) ;
		}
	}

	fclose(fp);

	binaryFile.close();
}

int main(int argc, char **argv)
{
	char *path_to_train_feature_files = argv[1];
	char *path_to_test_feature_files = argv[2];
	char *path_to_vocabulary = argv[3];
	char *vocab_name = argv[4];
	string csv_file_train(argv[5]);
	string csv_file_test(argv[6]);
	char *prefix = argv[7];
	char *vocab_sizes_file = argv[8];

	printf("\n*********** dense DT start ***********");
	printf("\n");


	vector<Mat> vocabulary;

	int feature_size=0;
	vector<int> vocab_sizes;	

	char path[100], name[100];
	Mat vocab;
	sprintf(path,"%s/%s_trajectory.yml", path_to_vocabulary,vocab_name);
	sprintf(name,"%s_trajectory", vocab_name);

	FileStorage fileStorage(path, FileStorage::READ);
	fileStorage[name]>> vocab;
	printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
	vocabulary.push_back(vocab);
	feature_size += vocab.rows;
	vocab_sizes.push_back(vocab.rows);

	sprintf(path,"%s/%s_HOG.yml", path_to_vocabulary, vocab_name);
	sprintf(name,"%s_HOG", vocab_name);


	fileStorage = FileStorage(path, FileStorage::READ);
	fileStorage[name]>> vocab;
	printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
	vocabulary.push_back(vocab);
	feature_size += vocab.rows;
	vocab_sizes.push_back(vocab.rows);

	sprintf(path,"%s/%s_HOF.yml", path_to_vocabulary,vocab_name);
	sprintf(name,"%s_HOF", vocab_name);
	fileStorage = FileStorage(path, FileStorage::READ);
	fileStorage[name]>> vocab;
	printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
	vocabulary.push_back(vocab);
	vocab_sizes.push_back(vocab.rows);
	feature_size += vocab.rows;

	sprintf(path,"%s/%s_MBHx.yml", path_to_vocabulary,vocab_name);
	sprintf(name,"%s_MBHx", vocab_name);
	fileStorage = FileStorage(path, FileStorage::READ);
	fileStorage[name]>> vocab;
	printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
	vocabulary.push_back(vocab);
	feature_size += vocab.rows;
	vocab_sizes.push_back(vocab.rows);

	sprintf(path,"%s/%s_MBHy.yml", path_to_vocabulary, vocab_name);
	sprintf(name,"%s_MBHy", vocab_name);
	fileStorage = FileStorage(path, FileStorage::READ);
	fileStorage[name]>> vocab;
	printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
	vocabulary.push_back(vocab);
	feature_size += vocab.rows;
	vocab_sizes.push_back(vocab.rows);

	fileStorage.release();

	FILE *vs = fopen(vocab_sizes_file,"w");
	for(int i =0;i<vocab_sizes.size();i++)
		fprintf(vs, "%d\n", vocab_sizes[i]);
	fclose(vs);

	feature_size+=1; //label
	printf("\nFeature Size : %d", feature_size);
	printf("\n*********** get train feature histogram ***********");
	printf("\n");

	vector<feature> feature_histogram;
	ofstream csvfile;
	csvfile.open(csv_file_train.c_str());
	get_video_BOFhist_one_by_one(vocabulary, argv[1], feature_histogram, atoi(argv[9]), prefix, csvfile);
	csvfile.close();

	//writeToBinaryFile(feature_histogram, feature_size,argv[5]);
	feature_histogram.clear();

	printf("\n*********** get test feature histogram ***********");   
	printf("\n"); 
	vector<feature> test_feature_histogram;
	csvfile.open(csv_file_test.c_str());
	get_video_BOFhist_one_by_one(vocabulary, argv[2], test_feature_histogram,1, prefix, csvfile);
	test_feature_histogram.clear();
	csvfile.close();
	//writeToBinaryFile(test_feature_histogram, feature_size,argv[6]);

	printf("\n*********** finish ***********");
	printf("\n");

	return 0;
}
