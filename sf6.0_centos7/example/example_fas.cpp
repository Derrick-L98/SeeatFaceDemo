//#include "seeta/EyeBlinkDetector.h"
#include "seeta/FaceDetector.h"
#include "seeta/Common/Struct.h"

#include <seeta/FaceLandmarker.h>
#include <seeta/FaceAntiSpoofing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

#include <fstream>
using namespace cv;

auto red = CV_RGB(255, 0, 0);
auto green = CV_RGB(0, 255, 0);
auto blue = CV_RGB(0, 0, 255);

int main_video_fas()
{
	bool isFirst = true;

	int device_id = 0;
	std::string ModelPath = "./model/";


	seeta::ModelSetting anti_setting;
	anti_setting.set_device(seeta::ModelSetting::CPU);
	anti_setting.append("model/fas_first.csta");
	anti_setting.append("./model/fas_second.csta");


	seeta::FaceAntiSpoofing processor(anti_setting);
	processor.SetThreshold(0.3, 0.90);    // 设置默认阈值，另外一组阈值为(0.7, 0.55)


	seeta::ModelSetting FD_model;
	FD_model.append(ModelPath + "face_detector.csta");
	FD_model.set_device(seeta::ModelSetting::CPU);
	FD_model.set_id(device_id);

	seeta::ModelSetting PD_model;
	PD_model.append(ModelPath + "face_landmarker_pts5.csta");
	PD_model.set_device(seeta::ModelSetting::CPU);
	PD_model.set_id(device_id);


	seeta::FaceDetector fd(FD_model); //人脸检测的初始化

	seeta::FaceLandmarker FL(PD_model); //关键点检测模型初始化


	Mat frame;
	Mat origin;
	VideoCapture capture(0); //打开视频文件

	if (!capture.isOpened())       //检测是否正常打开:成功打开时，isOpened返回ture
		std::cout << "fail to open!" << std::endl;
	while (true)
	{
		if (!capture.read(frame))
		{
			std::cout << "can not read any frame" << std::endl;
			break;
		}
		flip(frame, frame, 1); //左右旋转摄像头，使电脑中图像和人的方向一致
		if (frame.channels() == 4)     //如果为4通道则转为3通道的rgb图像
		{
			cv::cvtColor(frame, frame, CV_RGBA2BGR);
		}

		origin = frame.clone();
		//seeta::cv::ImageData image = frame;

		SeetaImageData image;
		image.height = frame.rows;
		image.width = frame.cols;
		image.channels = frame.channels();
		image.data = frame.data;
		// 从外部进行人脸检测和特征点定位
		auto faces = fd.detect(image);
		std::cout << "faces.size:" << faces.size << std::endl;
		if (faces.size == 1)
		{
			auto &face = faces.data[0].pos;

			cv::Scalar color;
			color = CV_RGB(0, 255, 0);

			std::vector<SeetaPointF> points(FL.number());
			FL.mark(image, face, points.data());

			//单帧的活体检测
			//基于视频的活体检测


			auto status = processor.PredictVideo(image, face, points.data());

			std::string stateOfTheFace;
			switch (status)
			{
			case seeta::FaceAntiSpoofing::SPOOF:
				stateOfTheFace = "spoof";
				color = CV_RGB(255, 0, 0);
				break;
			case seeta::FaceAntiSpoofing::REAL:
				stateOfTheFace = "real";
				std::cout << "real" << std::endl;

				color = CV_RGB(0, 255, 0);
				break;
			case seeta::FaceAntiSpoofing::FUZZY:
				// stateOfTheFace = "fuzzy";
				break;
			case seeta::FaceAntiSpoofing::DETECTING:
				// stateOfTheFace = "detecting";
				break;
			}
			float clarity;
			float reality;

			processor.GetPreFrameScore(&clarity, &reality);

			std::cout << "Clarity = " << clarity << ", Reality = " << reality << std::endl;

			cv::putText(frame, stateOfTheFace, cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

			rectangle(frame, cv::Rect(face.x, face.y, face.width, face.height), color, 2, 8, 0);

		}
		else
		{
			for (int i = 0; i < faces.size; i++)
			{
				auto face = faces.data[i].pos;
				rectangle(frame, cv::Rect(face.x, face.y, face.width, face.height), cv::Scalar(255, 0, 0), 2, 8, 0); //画人脸检测框
			}
			
		}

		cv::imshow("SeetaFaceAntiSpoofing", frame);   //显示视频
		if (cv::waitKey(1) == 27)    //退出条件：1，按exc键;2，达到显示“通过或未通过“的帧数;
			break;
	}

	return 0;
}

int main()
{
	return main_video_fas();
}