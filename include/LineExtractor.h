#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <iostream>
#include <vector>
#include <list>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "auxiliar.h"

using namespace std;
using namespace cv;

namespace ORB_SLAM2{
class LINEextractor
{
public:
    LINEextractor(){}
    LINEextractor( int _numOctaves, float _scale, unsigned int _nLSDFeature, double _min_line_length);
    ~LINEextractor(){}

    void operator()( cv::InputArray image, cv::InputArray mask, std::vector<line_descriptor::KeyLine>& keylines, cv::OutputArray descriptors, std::vector<Eigen::Vector3d> &lineVec2d);

    int inline GetLevels(){
        return numOctaves;}

    float inline GetScaleFactor(){
        return scale;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

protected:
    double min_line_length;
    int numOctaves;
    unsigned int nLSDFeature;
    float scale;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

};
}

#endif