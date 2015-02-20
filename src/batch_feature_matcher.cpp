// The MIT License (MIT)

// Copyright (c) 2015 Miquel Massot Campos

//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#include <batch_feature_matching/batch_feature_matcher.h>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <boost/filesystem.hpp>
#include <string>
#include <iostream>
#include <fstream>

#define CROSS_CHECK_FILTER 1
#define DISTANCE_FILTER 0

// using namespace boost::filesystem;
// using namespace std;

BatchFeatureMatcher::BatchFeatureMatcher(std::string path, std::string format)
    : path_(path), format_(format) {
  cv::initModule_nonfree();
  detector_  = cv::FeatureDetector::create("SIFT");
  extractor_ = cv::DescriptorExtractor::create("SIFT");
  matcher_   = cv::DescriptorMatcher::create("FlannBased");

  if(detector_.empty()) {
    std::cout << "ERROR: Cannot create detector of given type!" << std::endl;
  }
  if (extractor_.empty()) {
    std::cout << "ERROR: Cannot create extractor of given type!" << std::endl;
  }
  if (matcher_.empty()) {
    std::cout << "ERROR: Cannot create matcher of given type!" << std::endl;
  }

  matcher_filter_type_ = CROSS_CHECK_FILTER;

  matching_threshold_ = 0.9;  // DISTANCE_FILTER Only

  std::cout << "Extracting features..." << std::endl;
  extractAllFeatures();
  std::cout << "Matching features..." << std::endl;
  matchAll2All();
}

void BatchFeatureMatcher::extractAllFeatures() {
  boost::filesystem::path p(path_);
  std::vector<boost::filesystem::directory_entry> file_entries;
  try {
    if (boost::filesystem::exists(p)) {
      if (boost::filesystem::is_regular_file(p)) {
        std::cout << p << " size is " << file_size(p) << '\n';
      } else if (boost::filesystem::is_directory(p)) {
        std::copy(boost::filesystem::directory_iterator(p),
                  boost::filesystem::directory_iterator(),
                  std::back_inserter(file_entries));
      } else {
        std::cout << p <<
          " exists, but is neither a regular file nor a directory\n";
      }
    } else {
      std::cout << p << " does not exist\n";
    }
  } catch (const boost::filesystem::filesystem_error& ex) {
    std::cout << ex.what() << '\n';
  }

  #pragma omp parallel for
  for (size_t i = 0; i < file_entries.size(); i++) {
    std::string full_image_path(file_entries[i].path().string());
    std::string name(file_entries[i].path().stem().string());
    std::string extension(file_entries[i].path().extension().string());
    if (extension == "."+format_) {
      std::cout << "Reading image " << full_image_path << "..." << std::endl;
      cv::Mat img = cv::imread(full_image_path, CV_LOAD_IMAGE_GRAYSCALE);
      extractFeatures(img, name);
    }
  }
}

void BatchFeatureMatcher::matchAll2All() {
  boost::filesystem::path p(boost::filesystem::current_path());
  std::vector<boost::filesystem::directory_entry> file_entries;
  try {
    if (boost::filesystem::exists(p)) {
      if (boost::filesystem::is_regular_file(p)) {
        std::cout << p << " size is " << file_size(p) << '\n';
      } else if (boost::filesystem::is_directory(p)) {
        std::copy(boost::filesystem::directory_iterator(p),
                  boost::filesystem::directory_iterator(),
                  std::back_inserter(file_entries));
      } else {
        std::cout << p <<
          " exists, but is neither a regular file nor a directory\n";
      }
    } else {
      std::cout << p << " does not exist\n";
    }
  } catch (const boost::filesystem::filesystem_error& ex) {
    std::cout << ex.what() << '\n';
  }

  // Get the number of yaml files
  std::vector<std::string> full_names;
  std::vector<std::string> names;
  for (size_t i = 0; i < file_entries.size(); i++) {
    std::string full_name(file_entries[i].path().string());
    std::string name(file_entries[i].path().stem().string());
    std::string extension(file_entries[i].path().extension().string());
    if (extension == ".yaml") {
      full_names.push_back(full_name);
      names.push_back(name);
    }
  }

  int nelem = names.size();
  int sd_nelem = nelem*(nelem-1)/2;
  std::cout << "Number of elements: " << nelem << " and " << sd_nelem << "\n";
  std::vector<std::vector<int> > matches_vec(nelem,
                                     std::vector<int>(sd_nelem));
  std::vector<std::vector<int> > inliers_vec(matches_vec);
  int_map m;

  #pragma omp parallel for
  for (size_t i = 0; i < names.size(); i++) {
    for (size_t j = i + 1; j < names.size(); j++) {
      std::vector<cv::KeyPoint> kp1, kp2;
      cv::Mat d1, d2;
      getKpAndDesc(full_names[i], kp1, d1);
      getKpAndDesc(full_names[j], kp2, d2);
      std::vector<cv::DMatch> filt_m12;
      int matches, inliers;
      match(kp1, d1, kp2, d2, filt_m12, matches, inliers);
      std::cout << "Detected " << matches << " matches between "
        << names[i] << " and " << names[j] << " with " << inliers
        << " inliers" << std::endl;
      m.insert(int_map::value_type(std::make_pair(i, j),
                                   std::make_pair(matches, inliers)));
    }
  }

  std::ofstream results_file("results.txt");
  typedef int_map::const_iterator const_int_map_it;
  for(const_int_map_it it = m.begin(); it != m.end(); it++) {
    int i = it->first.first;
    int j = it->first.second;
    int matches = it->second.first;
    int inliers = it->second.second;
    results_file << names[i]
          << " " << names[j]
          << " " << matches
          << " " << inliers << std::endl;
  }
  results_file.close();
}

void BatchFeatureMatcher::getKpAndDesc(std::string filename,
                  std::vector<cv::KeyPoint>& kp,
                  cv::Mat& d) {
  cv::FileStorage f(filename, cv::FileStorage::READ);
  cv::FileNode fn = f["keypoints"];
  cv::read(fn, kp);
  f["descriptors"] >> d;
  int n;
  f["n"] >> n;
  if (n != kp.size()) {
    std::cout << "ERROR File " << filename << " was NOT read correctly!" << std::endl;
  }
}

void BatchFeatureMatcher::extractFeatures(const cv::Mat& image, std::string name) {

  // Check if file already exists
  boost::filesystem::path my_file(name + ".yaml");
  if (boost::filesystem::exists(my_file)) {
    std::cout << "File " << name << ".yaml already exists. Skipping..." << std::endl;
    // return
  } else {
    if (image.empty()) {
      std::cerr << "Image " << name << " is empty" << std::endl;
      return;
    }

    std::vector<cv::KeyPoint> keypoints;
    std::cout << "Detecting keypoints (" << name << ")" << std::endl;
    detector_->detect(image, keypoints);
    std::cout << "Detected " << keypoints.size() << " keypoints!" << std::endl;
    cv::Mat descriptors;
    std::cout << "Extracting descriptors (" << name << ")" << std::endl;
    extractor_->compute(image, keypoints, descriptors);

    std::cout << "Writting keypoints and descriptors to file (" << name << "_kp.yaml)" << std::endl;
    cv::FileStorage fs(name + ".yaml", cv::FileStorage::WRITE);
    fs << "n" << (int)keypoints.size();
    fs << "keypoints" << keypoints;
    fs << "descriptors" << descriptors;
    fs.release();
  }
}

void BatchFeatureMatcher::match(const std::vector<cv::KeyPoint>& kp1,
                                const cv::Mat& d1,
                                const std::vector<cv::KeyPoint>& kp2,
                                const cv::Mat& d2,
                                std::vector<cv::DMatch>& filt_m12,
                                int& matches, int& inliers) {
  // Clear the output vector first
  filt_m12.clear();
  // Check if there are any descriptors to match
  if (d1.empty() || d2.empty()) {
    // return
  } else {
    switch (matcher_filter_type_) {
      case CROSS_CHECK_FILTER :
        crossCheckMatching(matcher_, d1, d2, filt_m12, 1);
        break;
      case DISTANCE_FILTER:
        thresholdMatching(matcher_, d1, d2, filt_m12, matching_threshold_);
        break;
      default :
        simpleMatching(matcher_, d1, d2, filt_m12);
        break;
    }
  }

  matches = (int)filt_m12.size();

  if (matches > 0) {
    // Get the matched keypoints
    std::vector<cv::Point2f> matched_kp1, matched_kp2;
    for (int i = 0; i < matches; i++) {
      matched_kp1.push_back(kp1[filt_m12[i].trainIdx].pt);
      matched_kp2.push_back(kp2[filt_m12[i].queryIdx].pt);
    }

    // Check the epipolar geometry
    cv::Mat status;
    cv::Mat F = cv::findFundamentalMat(matched_kp1,
                                       matched_kp2,
                                       CV_FM_RANSAC,
                                       3,  // Epipolar threshold
                                       0.999,
                                       status);

    // Is the fundamental matrix valid?
    // cv::Scalar f_sum_parts = cv::sum(F);
    // float f_sum = (float)f_sum_parts[0]
    //            + (float)f_sum_parts[1]
    //            + (float)f_sum_parts[2];
    // if (f_sum < 1e-3)
    //   return false;

    // Check inliers size
    inliers = (int)cv::sum(status)[0];
  } else {
    inliers = 0;
  }
}

/** @function simpleMatching */
void BatchFeatureMatcher::simpleMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                    const cv::Mat& d1, const cv::Mat& d2,
                    std::vector<cv::DMatch>& m12 ) {
  std::vector<cv::DMatch> matches;
  dm->match(d1, d2, m12);
}

/** @function crossCheckMatching */
void BatchFeatureMatcher::crossCheckMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                        const cv::Mat& d1, const cv::Mat& d2,
                        std::vector<cv::DMatch>& filt_m12, int knn=1) {
  filt_m12.clear();
  std::vector<std::vector<cv::DMatch> > m12, matches21;
  dm->knnMatch(d1, d2, m12, knn);
  dm->knnMatch(d2, d1, matches21, knn);
  for (size_t m = 0; m < m12.size(); m++) {
    bool findCrossCheck = false;
    for (size_t fk = 0; fk < m12[m].size(); fk++) {
      cv::DMatch forward = m12[m][fk];
      for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++) {
        cv::DMatch backward = matches21[forward.trainIdx][bk];
        if (backward.trainIdx == forward.queryIdx) {
          filt_m12.push_back(forward);
          findCrossCheck = true;
          break;
        }
      }
      if (findCrossCheck) break;
    }
  }
}

void BatchFeatureMatcher::thresholdMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                       const cv::Mat& d1, const cv::Mat& d2,
                       std::vector<cv::DMatch>& filt_m12,
                       double matching_threshold) {
  filt_m12.clear();
  std::vector<std::vector<cv::DMatch> > m12;
  int knn = 2;
  dm->knnMatch(d1, d2, m12, knn);
  for (size_t m = 0; m < m12.size(); m++) {
    if (m12[m].size() == 1) {
      filt_m12.push_back(m12[m][0]);
    } else if (m12[m].size() == 2) {  // normal case
      if (m12[m][0].distance / m12[m][1].distance
          < matching_threshold) {
        filt_m12.push_back(m12[m][0]);
      }
    }
  }
}
