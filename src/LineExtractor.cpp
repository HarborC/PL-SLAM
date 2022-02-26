#include"LineExtractor.h"
#include <opencv2/line_descriptor/descriptor.hpp>

namespace ORB_SLAM2{
LINEextractor::LINEextractor( int _numOctaves, float _scale, unsigned int _nLSDFeature, double _min_line_length):numOctaves(_numOctaves), scale(_scale), nLSDFeature(_nLSDFeature), min_line_length(_min_line_length)
{
    mvScaleFactor.resize(numOctaves);
    mvLevelSigma2.resize(numOctaves);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<numOctaves; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scale;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(numOctaves);
    mvInvLevelSigma2.resize(numOctaves);
    for(int i=0; i<numOctaves; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }
}

void LINEextractor::operator()( cv::InputArray _image, cv::InputArray _mask, std::vector<KeyLine>& _keylines, cv::OutputArray _descriptors, std::vector<Eigen::Vector3d>&  _lineVec2d)
{ 

    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    Mat mask = _mask.getMat();
    //assert(mask.type() == CV_8UC1 && !mask.empty());

    // detect line feature
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    lsd->detect(image, _keylines, scale, numOctaves, mask);

    // filter lines
    sort(_keylines.begin(), _keylines.end(), sort_lines_by_response());
    int total, index;
    if(_keylines.size()>nLSDFeature)
    {
        total = nLSDFeature;
        index = nLSDFeature;

    }else{
        total = _keylines.size();
        index = _keylines.size();
    }
    
    if(_keylines[total-1].lineLength < min_line_length){
        for(int i = 0; i<total-1;i++){
            if(_keylines[i].lineLength>=min_line_length && _keylines[i+1].lineLength<min_line_length){
                index = i;
                break;
            }
        }
    }
    else
        index--;

    _keylines.resize(index + 1);
    for(unsigned int i=0; i<index + 1; i++){
         _keylines[i].class_id = i;
    }

    Mat descriptors;
    if( _keylines.size() == 0 ){
        _descriptors.release();
        return;
    }else{
        descriptors = _descriptors.getMat();
    }
    
    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    lbd->compute(image, _keylines, descriptors);     //计算特征线段的描述子

    // 计算特征线段所在直线的系数
    _lineVec2d.clear();
    for(vector<KeyLine>::iterator it=_keylines.begin(); it!=_keylines.end(); ++it)
    {
        Eigen::Vector3d sp_l; sp_l << it->startPointX, it->startPointY, 1.0;
        Eigen::Vector3d ep_l; ep_l << it->endPointX, it->endPointY, 1.0;
        Eigen::Vector3d lineV;     //直线方程
        lineV << sp_l.cross(ep_l);
        lineV = lineV / sqrt(lineV(0)*lineV(0)+lineV(1)*lineV(1));
        _lineVec2d.push_back(lineV);
    }

    descriptors.copyTo(_descriptors);
}

}
