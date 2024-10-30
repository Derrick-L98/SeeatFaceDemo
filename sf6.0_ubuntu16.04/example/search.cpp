#pragma warning(disable: 4819)

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>

#include "platform.h"
#include <queue>

#if ORZ_PLATFORM_OS_WINDOWS
#include <io.h>
#else
#include <dirent.h>
#include <cstring>
#endif
namespace orz {
	const std::string FileSeparator() {
#if ORZ_PLATFORM_OS_WINDOWS
		return "\\";
#else
		return "/";
#endif
	}

	static std::vector<std::string> FindFilesCore(const std::string &path, std::vector<std::string> *dirs = nullptr) {
		std::vector<std::string> result;
		if (dirs) dirs->clear();
#if ORZ_PLATFORM_OS_WINDOWS
		_finddata_t file;
		std::string pattern = path + FileSeparator() + "*";
		auto handle = _findfirst(pattern.c_str(), &file);

		if (handle == -1L) return result;
		do {
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0) continue;
			if (file.attrib & _A_SUBDIR) {
				if (dirs) dirs->push_back(file.name);
			}
			else {
				result.push_back(file.name);
			}
		} while (_findnext(handle, &file) == 0);

		_findclose(handle);
#else
		struct dirent *file;

		auto handle = opendir(path.c_str());

		if (handle == nullptr) return result;

		while ((file = readdir(handle)) != nullptr)
		{
			if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) continue;
			if (file->d_type & DT_DIR)
			{
				if (dirs) dirs->push_back(file->d_name);
			}
			else if (file->d_type & DT_REG)
			{
				result.push_back(file->d_name);
			}
			// DT_LNK // for linkfiles
		}

		closedir(handle);
#endif
		return std::move(result);
	}

	std::vector<std::string> FindFiles(const std::string &path) {
		return FindFilesCore(path);
	}

	std::vector<std::string> FindFiles(const std::string &path, std::vector<std::string> &dirs) {
		return FindFilesCore(path, &dirs);
	}

	std::vector<std::string> FindFilesRecursively(const std::string &path, int depth) {
		std::vector<std::string> result;
		std::queue<std::pair<std::string, int> > work;
		std::vector<std::string> dirs;
		std::vector<std::string> files = FindFiles(path, dirs);
		result.insert(result.end(), files.begin(), files.end());
		for (auto &dir : dirs) work.push({ dir, 1 });
		while (!work.empty()) {
			auto local_pair = work.front();
			work.pop();
			auto local_path = local_pair.first;
			auto local_depth = local_pair.second;
			if (depth > 0 && local_depth >= depth) continue;
			files = FindFiles(path + FileSeparator() + local_path, dirs);
			for (auto &file : files) result.push_back(local_path + FileSeparator() + file);
			for (auto &dir : dirs) work.push({ local_path + FileSeparator() + dir, local_depth + 1 });
		}
		return result;
	}
}
bool extract_feature(const seeta::FaceDetector &FD,
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

	if (faces.size <= 0)
	{
		std::cout << "no face detected in" << image_path << std::endl;
		return false;
	}

	SeetaPointF points[5];
	FL.mark(simg, faces.data[0].pos, points);

	FR.Extract(simg, points, feature);

	return true;
}


void extract_feature(const seeta::FaceLandmarker &FL,
	const seeta::FaceRecognizer &FR,
	const seeta::ImageData image_data,
	const SeetaRect &pos,
	float* feature)
{
	SeetaPointF points[5];
	FL.mark(image_data, pos, points);

	FR.Extract(image_data, points, feature);
}

int main()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;
    seeta::ModelSetting FD_model("model/face_detector.csta", device, id);
    seeta::ModelSetting FL_model("model/face_landmarker_pts5.csta", device, id);
    seeta::ModelSetting FR_model("model/face_recognizer.csta", device, id);

	seeta::FaceDetector FD(FD_model);
	FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 80);

	seeta::FaceLandmarker FL(FL_model);
	seeta::FaceRecognizer FR(FR_model);
    // recognization threshold
    float threshold = 0.60;

	std::vector<std::pair<std::string, std::shared_ptr<float> > > vec_name_2_feat;
    // Register 4 faces
	std::string gallery_path = "./gallery";
	std::vector<std::string> GalleryImageFilename = orz::FindFilesRecursively(gallery_path, -1);
    for (size_t i = 0; i < GalleryImageFilename.size(); ++i)
    {
        std::string &filename = GalleryImageFilename[i];

		std::shared_ptr<float> feats(new float[FR.GetExtractFeatureSize()]);
		auto result = extract_feature(FD, FL, FR, gallery_path + "/" + filename, feats.get());
		if (!result) continue;//no face

		std::string pure_name = filename.substr(0, filename.find_last_of('.'));
		vec_name_2_feat.emplace_back(std::pair<std::string, std::shared_ptr<float>>(pure_name, feats));
    }
	std::cout << "gallery size:" << vec_name_2_feat.size() << std::endl;
  //  std::map<int64_t, std::string> GalleryIndexMap;
  //  for (size_t i = 0; i < GalleryImageFilename.size(); ++i)
  //  {
		//size_t name_length = GalleryImageFilename[i].find_last_of('.');
		//std::string pure_name = GalleryImageFilename[i].substr(0, name_length);
  //      GalleryIndexMap.insert(std::make_pair(i, pure_name));
  //  }

    // Open default USB camera
    cv::VideoCapture capture;
    capture.open(0);

    cv::Mat frame;

    while (capture.isOpened())
    {
        capture >> frame;
        if (frame.empty()) continue;

        seeta::cv::ImageData image = frame;

        // Detect all faces
		auto face_infos = FD.detect(image);

		for (int i = 0; i < face_infos.size; ++i)
		{
			int64_t target_index = -1;
			float max_sim = 0;

			auto &face = face_infos.data[i];
			std::unique_ptr<float[]> feature(new float[FR.GetExtractFeatureSize()]);
			extract_feature(FL, FR, image, face.pos, feature.get());

			for (size_t index = 0; index < vec_name_2_feat.size(); ++index)
			{
				auto & pair_name_feat = vec_name_2_feat[index];
				float current_sim = FR.CalculateSimilarity(feature.get(), pair_name_feat.second.get());
				if (current_sim > max_sim)
				{
					max_sim = current_sim;
					target_index = index;
				}
			}

			cv::rectangle(frame, cv::Rect(face.pos.x, face.pos.y, face.pos.width, face.pos.height), CV_RGB(255, 0, 0), 3);

			// similarity greater than threshold, means recognized
			if (max_sim > threshold)
			{
				cv::putText(frame, vec_name_2_feat[target_index].first, cv::Point(face.pos.x, face.pos.y - 5), CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 0, 255));
			}
		}
        cv::imshow("Frame", frame);

        auto key = cv::waitKey(20);
        if (key == 27)
        {
            break;
        }
    }

	return 0;
}
