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

#define CROSS_CHECK_FILTER 1
#define DISTANCE_FILTER 0

BatchFeatureMatcher::BatchFeatureMatcher(std::string path, std::string format)
    : path_(path), format_(format) {
  detector_ = cv::FeatureDetector::create("SIFT");
  extractor_ = cv::DescriptorExtractor::create("SIFT");
  matcher_ = cv::DescriptorMatcher::create("FlannBased");

  matcher_filter_type_ = CROSS_CHECK_FILTER;

  matching_threshold_ = 0.9;  // DISTANCE_FILTER Only

  extractAllFeatures();
  matchAll2All();
}

void BatchFeatureMatcher::extractAllFeatures() {
  boost::filesystem::path p(path_);
  #pragma omp parallel for
  for (boost::filesystem::path::iterator it = p.begin(); it != p.end(); ++it) {
    if (it->extension() == format_) {
      cv::Mat img = cv::imread(it->string(), CV_LOAD_IMAGE_COLOR);
      std::string name(it->stem().string());
      std::cout << "Extracting features of image " << name << std::endl;
      extractFeatures(img, name);
    }
  }
}

void BatchFeatureMatcher::matchAll2All() {
  boost::filesystem::path p1(path_);
  boost::filesystem::path p2(path_);
  #pragma omp parallel for
  for (boost::filesystem::path::iterator it1 = p1.begin();
      it1 != p1.end(); ++it1) {
    if (it1->extension() == "yaml") {
      for (boost::filesystem::path::iterator it2 = p2.begin();
           it2 != p2.end(); ++it2) {
        if (it2->extension() == "yaml") {
          std::string img1(it1->string());
          std::string img2(it2->string());
          std::vector<cv::KeyPoint> kp1, kp2;
          cv::Mat d1, d2;
          getKpAndDesc(img1, kp1, d1);
          getKpAndDesc(img2, kp2, d2);
          std::vector<cv::DMatch> filt_m12;
          match(d1, d2, filt_m12);
        }
      }
    }
  }
}

void BatchFeatureMatcher::getKpAndDesc(std::string filename,
                  std::vector<cv::KeyPoint>& kp,
                  cv::Mat& d) {
  cv::FileStorage f(filename, cv::FileStorage::READ);
  cv::FileNode kptFileNode = f["keypoints"];
  cv::read( kptFileNode, kp );
  f["descriptors"] >> d;
}

void BatchFeatureMatcher::extractFeatures(const cv::Mat& image, std::string name) {
  if (image.empty()) {
    std::cerr << "Image " << name << " is empty" << std::endl;
    return;
  }

  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(image, keypoints);
  cv::Mat descriptors;
  extractor_->compute(image, keypoints, descriptors);

  cv::FileStorage fs(name + "_kp.yaml", cv::FileStorage::WRITE);
  fs << "keypoints" << keypoints;
  fs << "descriptors" << descriptors;
  fs.release();
}

void BatchFeatureMatcher::match(const cv::Mat& d1, const cv::Mat& d2,
                                std::vector<cv::DMatch>& filt_m12) {
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
