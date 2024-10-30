#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceRecognizer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

void extract_feature(const seeta::FaceDetector &FD,
					 const seeta::FaceLandmarker &FL,
					 const seeta::FaceRecognizer &FR,
					 std::string image_path,
					 float* feature)
{
	cv::Mat img = cv::imread(image_path);
	SeetaImageData simg;
    simg.height = img.rows;
    simg.width = img.cols;
    simg.channels = img.channels();
    simg.data = img.data;

    auto faces = FD.detect(simg);

	if(faces.size <= 0)
	{
		std::cout << "no face detected in" << image_path << std::endl;
		return;
	}
	
	SeetaPointF points[5];
	FL.mark(simg, faces.data[0].pos, points);
	
	FR.Extract(simg, points, feature);
}

int main()
{
	seeta::ModelSetting fd_setting;
    fd_setting.set_device(SEETA_DEVICE_CPU);
    fd_setting.append("face_detector.csta");
    seeta::FaceDetector FD(fd_setting);
	
	seeta::ModelSetting fl_setting;
    fl_setting.set_device(SEETA_DEVICE_CPU);
    fl_setting.append("face_landmarker_pts5.csta");
    seeta::FaceLandmarker FL(fl_setting);
	
	seeta::ModelSetting fr_setting;
    fr_setting.set_id(0);
    fr_setting.append("face_recognizer.csta");
    fr_setting.set_device(SEETA_DEVICE_CPU);
    seeta::FaceRecognizer FR(fr_setting);
	
	std::shared_ptr<float> feature1(new float[FR.GetExtractFeatureSize()]);
	std::shared_ptr<float> feature2(new float[FR.GetExtractFeatureSize()]);
	
	extract_feature(FD, FL, FR, "1.png", feature1.get());
	extract_feature(FD, FL, FR, "2.png", feature2.get());
	
	float sim = FR.CalculateSimilarity(feature1.get(), feature2.get());
	
	std::cout<<"face's similarity in 1.png and 2.png is" << sim << std::endl;
	return 0;
}