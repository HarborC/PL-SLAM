//
// Created by lan on 17-12-26.
//

#ifndef ORB_SLAM2_LSDMATCHER_H
#define ORB_SLAM2_LSDMATCHER_H

//#include <line_descriptor/descriptor_custom.hpp>
//#include <line_descriptor_custom.hpp>

#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

#include <thread>
#include <mutex>

using namespace line_descriptor;

namespace ORB_SLAM2
{
    class LSDmatcher
    {
    public:
        LSDmatcher(float nnratio=0.7, bool checkOri=true);

        // 通过投影线段，对上一帧的特征线进行跟踪
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame);

        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

        // 通过投影，对Local MapLine进行跟踪
        int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);

        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<int> &LineMatches);

        // For Iniitialize
        int SearchDouble(Frame &InitialFrame, Frame &CurrentFrame, vector<int> &LineMatches);

        // For Track KeyFrame 
        int SearchDouble(KeyFrame *KF, Frame &CurrentFrame);

        static int DescriptorDistance(const Mat &a, const Mat &b);

        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t>> &vMatchedPairs);
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble = false);
        int SearchForTriangulationNew(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble = false);

        // Project MapLines into KeyFrame and search for duplicated MapLines
        int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines, float th = 3.0);

    public:

        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;

        cv::Mat pic; 

    protected:
        float RadiusByViewingCos(const float &viewCos);

        // For Initialize 
        void FrameBFMatch(cv::Mat ldesc1, cv::Mat ldesc2, vector<int>& LineMatches, float TH);
        void FrameBFMatchNew(cv::Mat ldesc1, cv::Mat ldesc2, vector<int>& LineMatches, vector<KeyLine> kls1, vector<KeyLine> kls2, vector<Eigen::Vector3d> kls2func, cv::Mat F, float TH);

        float mutualOverlap(const std::vector<cv::Mat>& collinear_points);
        cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

        void lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const;

        void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

        float mfNNratio;
        bool mbCheckOrientation;
    };
}


#endif //ORB_SLAM2_LSDMATCHER_H
