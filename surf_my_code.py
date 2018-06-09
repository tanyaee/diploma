import cv2
import numpy as np
from matplotlib import pyplot as plt
bool ReadFeatureDescriptors(const std::string& path,
    std::vector<cv::KeyPoint>& keypoints,
    cv2::Mat& descriptors) {
    cv2::Mat img = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data) {
    return false;
  }

  cv2::SurfFeatureDetector detector;
  cv2::SurfDescriptorExtractor extractor;

  detector.detect(img, keypoints);
  extractor.compute(img, keypoints, descriptors);

  return true;
