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

#ifndef BATCH_FEATURE_MATCHER_H
#define BATCH_FEATURE_MATCHER_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

class BatchFeatureMatcher {
 public:
  BatchFeatureMatcher(std::string path, std::string format);

 private:
  int matcher_filter_type_;
  double matching_threshold_;
  std::vector<cv::DMatch> filtered_matches_;
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  std::vector<cv::KeyPoint> keypoints_;
  std::vector<cv::Point2f> points_;
  cv::Mat descriptors_;
  std::vector<char> matches_mask_;
  std::string path_, format_;

  void extractAllFeatures();
  void matchAll2All();
  void getKpAndDesc(std::string filename,
                    std::vector<cv::KeyPoint>& kp,
                    cv::Mat& d);
  void extractFeatures(const cv::Mat& image, std::string name);
  void match(const cv::Mat& d1, const cv::Mat& d2,
             std::vector<cv::DMatch>& filt_m12);
  void simpleMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                      const cv::Mat& d1, const cv::Mat& d2,
                      std::vector<cv::DMatch>& m12 );
  void crossCheckMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                          const cv::Mat& d1, const cv::Mat& d2,
                          std::vector<cv::DMatch>& filt_m12, int knn);
  void thresholdMatching(cv::Ptr<cv::DescriptorMatcher>& dm,
                         const cv::Mat& d1, const cv::Mat& d2,
                         std::vector<cv::DMatch>& filt_m12,
                         double matching_threshold);
};
#endif  // BATCH_FEATURE_MATCHER_H
