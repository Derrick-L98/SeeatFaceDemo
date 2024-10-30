#include "seeta/EyeStateDetector.h"
#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/Common/Struct.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

#include <fstream>


std::string getstatus(seeta::EyeStateDetector::EYE_STATE state)
{
    if(state == seeta::EyeStateDetector::EYE_CLOSE)
        return "CLOSE";
    else if (state == seeta::EyeStateDetector::EYE_OPEN)
        return "OPEN";
    else if (state == seeta::EyeStateDetector::EYE_RANDOM)
        return "RANDOM";
    else
        return "";
}

auto red = CV_RGB(255, 0, 0);
auto green = CV_RGB(0, 255, 0);
auto blue = CV_RGB(0, 0, 255);

class triger
{
public:
        bool signal(bool level)
        {
                bool now_level = !m_pre_level && level;
                m_pre_level = level;
                return now_level;
        }
private:
        bool m_pre_level = false;
};

int main_video()
{
    std::cout << "== Load model ==" << std::endl;

    seeta::ModelSetting setting;
    setting.set_device( SEETA_DEVICE_GPU );
    setting.set_id( 0 );
    setting.append( "./model/eye_state.csta" );

    seeta::EyeStateDetector EBD( setting);


    seeta::ModelSetting setting2;
    setting2.set_device( SEETA_DEVICE_GPU );
    setting2.set_id( 0 );
    setting2.append( "./model/face_detector.csta" );
    seeta::FaceDetector FD( setting2 );
    
    seeta::ModelSetting setting3;
    setting3.set_device( SEETA_DEVICE_GPU );
    setting3.set_id( 0 );
    
    setting3.append( "./model/face_landmarker_pts5.csta" );
    seeta::FaceLandmarker FL(setting3);

    std::cout << "== Open camera ==" << std::endl;

    cv::VideoCapture capture( 0 );
    cv::Mat frame, canvas;
    std::stringstream oss;

    triger triger_left, triger_right;
    int count_blink_times = 0;

    while( capture.isOpened() )
    {
        capture >> frame;
        if( frame.empty() ) continue;
        canvas = frame.clone();


        SeetaImageData simage;
        simage.width = frame.cols;
        simage.height = frame.rows;
        simage.channels = frame.channels();
        simage.data = frame.data;

        auto faces = FD.detect( simage );

        for( int i=0; i< faces.size; i++ )
        {

            SeetaPointF points[5];

            FL.mark( simage, faces.data[i].pos, points );

            
            SeetaPointF pts[5];
            for(int m=0; m<5; m++) {
                pts[m].x = points[m].x;
                pts[m].y = points[m].y;
            }
           
            seeta::EyeStateDetector::EYE_STATE leftstate,rightstate;
            EBD.Detect( simage, pts, leftstate, rightstate );
			
            oss.str( "left,right " );
            oss << "(";
            oss << getstatus(leftstate);
            oss << ", ";
            oss << getstatus(rightstate);
            oss << ")";

            cv::rectangle( canvas, cv::Rect( faces.data[i].pos.x, faces.data[i].pos.y, faces.data[i].pos.width, faces.data[i].pos.height ), cv::Scalar( 128, 0, 0 ), 3 );
            cv::putText( canvas, oss.str(), cv::Point( faces.data[i].pos.x, faces.data[i].pos.y - 10 ), 0, 0.5, cv::Scalar( 0, 128, 0 ), 2 );
        }



        cv::imshow( "Faces", canvas );
        auto key = cv::waitKey( 30 );
        if( key >= 0 ) break;
    }

    return EXIT_SUCCESS;


    return 0;
}

int main()
{
    return main_video();
}
