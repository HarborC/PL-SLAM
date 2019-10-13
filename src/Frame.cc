/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>
#include "LocalMapping.h"
#include "lineIterator.h"
#include <unordered_set>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mnScaleLevelsLine(frame.mnScaleLevelsLine),
     mfScaleFactorLine(frame.mfScaleFactorLine), mfLogScaleFactorLine(frame.mfLogScaleFactorLine),
     mvScaleFactorsLine(frame.mvScaleFactorsLine), mvInvScaleFactorsLine(frame.mvInvScaleFactorsLine),
     mvLevelSigma2Line(frame.mvLevelSigma2Line), mvInvLevelSigma2Line(frame.mvInvLevelSigma2Line),
     mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn), mvpMapLines(frame.mvpMapLines),  //线特征相关的类成员变量
     mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions), ImageGray(frame.ImageGray.clone())
{
    // Points
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    // Lines
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGridForLine[i][j]=frame.mGridForLine[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

/// 双目初始化
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

/// RGBD初始化建立frame
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    imGray.copyTo(ImageGray);

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


/// 单目初始化建立frame
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* orbextractor,LINEextractor* lsdextractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const cv::Mat &mask)
    :mpORBvocabulary(voc),mpORBextractorLeft(orbextractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mpLSDextractorLeft(lsdextractor), 
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    imGray.copyTo(ImageGray);

    // Scale Level Info for point
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Scale Level Info for line
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();

    cv::Mat mUndistX, mUndistY, mImGray_remap;
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(imGray.cols, imGray.rows), CV_32F, mUndistX, mUndistY);
    cv::remap(imGray, mImGray_remap, mUndistX, mUndistY, cv::INTER_LINEAR);

    thread threadPoint(&Frame::ExtractORB, this, 0, imGray);
    thread threadLine(&Frame::ExtractLSD, this, mImGray_remap, mask);
    threadPoint.join();
    threadLine.join();

    NL = mvKeylinesUn.size(); //特征线的数量
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
        
    //mvKeysUn = mvKeys;
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));   // 初始化N个MapPoint，且都为NULL
    mvbOutlier = vector<bool>(N,false);     // 初始化每个特征点都不是外点

    mvpMapLines = vector<MapLine*>(NL,static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    //AssignFeaturesToGrid();
    //AssignFeaturesToGridForLine();

    thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
    thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
    threadAssignPoint.join();
    threadAssignLine.join();

}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::AssignFeaturesToGridForLine()
{
    int nReserve = 0.5f*NL/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGridForLine[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    //#pragma omp parallel for
    for(int i=0;i<NL;i++)
    {
        const KeyLine &kl = mvKeylinesUn[i];

        list<pair<int, int>> line_coords;

        LineIterator* it = new LineIterator(kl.startPointX * mfGridElementWidthInv, kl.startPointY * mfGridElementHeightInv, kl.endPointX * mfGridElementWidthInv, kl.endPointY * mfGridElementHeightInv);

        std::pair<int, int> p;
        while (it->getNext(p))
            if (p.first >= 0 && p.first < FRAME_GRID_COLS && p.second >= 0 && p.second < FRAME_GRID_ROWS)
                mGridForLine[p.first][p.second].push_back(i);

        delete [] it;
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

// line feature extractor, 自己添加的
void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &mask)
{
    (*mpLSDextractorLeft)(im,mask,mvKeylinesUn, mLdesc, mvKeyLineFunctions);
}

// 根据两个匹配的特征线计算特征线的3D坐标, frame1是当前帧，frame2是前一帧
void Frame::ComputeLine3D(Frame &frame1, Frame &frame2)
{
    //-------------------------计算两帧的匹配线段-----------------------------
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    Mat ldesc1, ldesc2;
    vector<vector<DMatch>> lmatches;
    vector<DMatch> good_matches;
    ldesc1 = frame1.mLdesc;
    ldesc2 = frame2.mLdesc;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);
    // sort matches by the distance between the best and second best matches
    double nn_dist_th, nn12_dist_th;
    frame1.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
//    nn12_dist_th = nn12_dist_th * 0.1;
        nn12_dist_th = nn12_dist_th * 0.5;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    vector<KeyLine> keylines1 = frame1.mvKeylinesUn;     //暂存mvKeylinesUn的集合
    frame1.mvKeylinesUn.clear();    //清空当前帧的mvKeylinesUn
    vector<KeyLine> keylines2;
    for(int i=0; i<lmatches.size(); i++)
    {
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12 > nn12_dist_th)
        {
            //认为这个匹配比较好，应该更新该帧的的匹配线
            good_matches.push_back(lmatches[i][0]);
            frame1.mvKeylinesUn.push_back(keylines1[lmatches[i][0].queryIdx]);  //更新当前帧的匹配线
            keylines2.push_back(frame2.mvKeylinesUn[lmatches[i][0].trainIdx]);  //暂存前一帧的匹配线，用于计算3D端点
        }
    }
    //-------------------计算当前帧mvKeylinesUn对应的3D端点---------------------
    ///-step 1：frame1的R,t，世界坐标系，相机内参
    //-step 1.1:先获取frame1的R,t
    Mat Rcw1 = frame1.mRcw;     //world to camera
    Mat Rwc1 = frame1.mRwc;
    Mat tcw1 = frame1.mtcw;
    Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //-step 1.2：得到当前帧在世界坐标系中的坐标
    Mat Ow1 = frame1.mOw;

    //-step 1.3:获取frame1的相机内参
    const float &fx1 = frame1.fx;
    const float &fy1 = frame1.fy;
    const float &cx1 = frame1.cx;
    const float &cy1 = frame1.cy;
    const float &invfx1 = frame1.invfx;
    const float &invfy1 = frame1.invfy;

    ///-step 2: frame2的R,t，世界坐标系，相机内参
    //-step 2.1：获取R，t
    Mat Rcw2 = frame2.mRcw;
    Mat Rwc2 = frame2.mRwc;
    Mat tcw2 = frame2.mtcw;
    Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    //-step 2.2：获取frame2在世界坐标系中的坐标
    Mat Ow2 = frame2.mOw;

    //-step 2.3：获取frame2的相机内参
    const float &fx2 = frame2.fx;
    const float &fy2 = frame2.fy;
    const float &cx2 = frame2.cx;
    const float &cy2 = frame2.cy;
    const float &invfx2 = frame2.invfx;
    const float &invfy2 = frame2.invfy;

    ///-step 3:对每对匹配通过三角化生成3D端点
    Mat vBaseline = Ow2 - Ow1;  //frame1和frame2的位移
    const float baseline = norm(vBaseline);

    //-step 3.1:根据两帧的姿态计算两帧之间的基本矩阵, Essential Matrix: t12叉乘R2
    const Mat &K1 = frame1.mK;
    const Mat &K2 = frame2.mK;
    Mat R12 = Rcw1*Rwc2;
    Mat t12 = -Rcw1*Rwc2*tcw2 + tcw1;
    Mat t12x = SkewSymmetricMatrix(t12);
    Mat essential_matrix = K1.t().inv()*t12x*R12*K2.inv();

    //-step3.2：三角化
       const int nlmatches = good_matches.size();
        for (int i = 0; i < nlmatches; ++i)
        {
            KeyLine &kl1 = frame1.mvKeylinesUn[i];
            KeyLine &kl2 = keylines2[i];

            //------起始点,start points-----
            // 得到特征线段的起始点在归一化平面上的坐标
            Mat sn1 = (Mat_<float>(3,1) << (kl1.startPointX-cx1)*invfx1, (kl1.startPointY-cy1)*invfy1, 1.0);
            Mat sn2 = (Mat_<float>(3,1) << (kl2.startPointX-cx2)*invfx2, (kl2.startPointY-cy2)*invfy2, 1.0);

            // 把对应的起始点坐标转换到世界坐标系下
            Mat sray1 = Rwc1 * sn1;
            Mat sray2 = Rwc2 * sn2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallax_sn = sray1.dot(sray2)/(norm(sray1) * norm(sray2));
            Mat s3D;
            if(cosParallax_sn > 0 && cosParallax_sn < 0.998)
            {
                // linear triangulation method
                Mat A(4,4,CV_32F);
                A.row(0) = sn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = sn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = sn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = sn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                Mat w1, u1, vt1;
                SVD::compute(A, w1, u1, vt1, SVD::MODIFY_A|SVD::FULL_UV);

                s3D = vt1.row(3).t();

                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);
            }
            Mat s3Dt = s3D.t();

            //-------结束点,end points-------
            // 得到特征线段的起始点在归一化平面上的坐标
            Mat en1 = (Mat_<float>(3,1) << (kl1.endPointX-cx1)*invfx1, (kl1.endPointY-cy1)*invfy1, 1.0);
            Mat en2 = (Mat_<float>(3,1) << (kl2.endPointX-cx2)*invfx2, (kl2.endPointY-cy2)*invfy2, 1.0);

            // 把对应的起始点坐标转换到世界坐标系下
            Mat eray1 = Rwc1 * en1;
            Mat eray2 = Rwc2 * en2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallax_en = eray1.dot(eray2)/(norm(eray1) * norm(eray2));
            Mat e3D;
            if(cosParallax_en > 0 && cosParallax_en < 0.998)
            {
                // linear triangulation method
                Mat B(4,4,CV_32F);
                B.row(0) = en1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                B.row(1) = en1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                B.row(2) = en2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                B.row(3) = en2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                Mat w2, u2, vt2;
                SVD::compute(B, w2, u2, vt2, SVD::MODIFY_A|SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);
            }
            Mat e3Dt = e3D.t();

            //-step 3.3：检测生成的3D点是否在相机前方，两个帧都需要检测
            float sz1 = Rcw1.row(2).dot(s3Dt) + tcw1.at<float>(2);
            if(sz1<=0)
                continue;

            float sz2 = Rcw2.row(2).dot(s3Dt) + tcw2.at<float>(2);
            if(sz2<=0)
                continue;

            float ez1 = Rcw1.row(2).dot(e3Dt) + tcw1.at<float>(2);
            if(ez1<=0)
                continue;

            float ez2 = Rcw2.row(2).dot(e3Dt) + tcw2.at<float>(2);
            if(ez2<=0)
                continue;

            //生成特征点时还有检测重投影误差和检测尺度连续性两个步骤，但是考虑到线特征的特殊性，先不写这两步
            //MapLine(int idx_, Vector6d line3D_, Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);
//            MapLine* pML = new MapLine();

        }

}

// line descriptor MAD, 自己添加的
void Frame::lineDescriptorMAD( vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
{
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;
//    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief 判断MapLine的两个端点是否在视野内
 *
 * @param pML               MapLine
 * @param viewingCosLimit   视角和平均视角的方向阈值
 * @return                  true if the MapLine is in view
 */
bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit)
{
    pML->mbTrackInView = false;

    Vector6d P = pML->GetWorldPos();

    cv::Mat SP = (Mat_<float>(3,1) << P(0), P(1), P(2));
    cv::Mat EP = (Mat_<float>(3,1) << P(3), P(4), P(5));

    // 两个端点在相机坐标系下的坐标
    const cv::Mat SPc = mRcw*SP + mtcw;
    const float &SPcX = SPc.at<float>(0);
    const float &SPcY = SPc.at<float>(1);
    const float &SPcZ = SPc.at<float>(2);

    const cv::Mat EPc = mRcw*EP + mtcw;
    const float &EPcX = EPc.at<float>(0);
    const float &EPcY = EPc.at<float>(1);
    const float &EPcZ = EPc.at<float>(2);

    // 检测两个端点的Z值是否为正
    if(SPcZ<0.0f || EPcZ<0.0f)
        return false;

    // V-D 1) 将端点投影到当前帧上，并判断是否在图像内
    const float invz1 = 1.0f/SPcZ;
    const float u1 = fx * SPcX * invz1 + cx;
    const float v1 = fy * SPcY * invz1 + cy;

    if(u1<mnMinX || u1>mnMaxX)
        return false;
    if(v1<mnMinY || v1>mnMaxY)
        return false;

    const float invz2 = 1.0f/EPcZ;
    const float u2 = fx*EPcX*invz2 + cx;
    const float v2 = fy*EPcY*invz2 + cy;

    if(u2<mnMinX || u2>mnMaxX)
        return false;
    if(v2<mnMinY || v2>mnMaxY)
        return false;

    // V-D 3)计算MapLine到相机中心的距离，并判断是否在尺度变化的距离内
    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
    // 世界坐标系下，相机到线段中点的向量，向量方向由相机指向中点
    const cv::Mat OM = 0.5*(SP+EP) - mOw;
    const float dist = cv::norm(OM);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // V-D 2)计算当前视角和平均视角夹角的余弦值，若小于cos(60°),即夹角大于60°则返回
    Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (Mat_<float>(3,1) << Pn(0), Pn(1), Pn(2));
    const float viewCos = OM.dot(pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征在一层）
    const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

    // Data used by the tracking
    // 标记该特征将来要被投影
    pML->mbTrackInView = true;
    pML->mTrackProjX1 = u1;
    pML->mTrackProjY1 = v1;
    pML->mTrackProjX2 = u2;
    pML->mTrackProjY2 = v2;
    pML->mnTrackScaleLevel = nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief 找到在以x,y为中心，半径为r的圆内且在[minLevel, maxLevel]的特征点
 * @param x         图像坐标u
 * @param y         图像坐标v
 * @param r         边长
 * @param minLevel  最小尺度
 * @param maxLevel  最大尺度
 * @return          满足条件的特征点的序号
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

vector<size_t> Frame::GetFeaturesInAreaForLine(const float &x1, const float &y1, const float &x2, const float &y2, const float  &r, const int minLevel, const int maxLevel,const float TH) const
{
    vector<size_t> vIndices;
    vIndices.reserve(NL);
    unordered_set<size_t> vIndices_set;

    float x[3] = {x1, (x1+x2)/2.0, x2};
    float y[3] = {y1, (y1+y2)/2.0, y2}; 

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(int i = 0; i<3;i++){
        const int nMinCellX = max(0,(int)floor((x[i]-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            continue;

        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x[i]-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            continue;

        const int nMinCellY = max(0,(int)floor((y[i]-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            continue;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y[i]-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            continue;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGridForLine[ix][iy];
                if(vCell.empty())
                    continue;

                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    if(vIndices_set.find(vCell[j]) != vIndices_set.end())
                        continue;

                    const KeyLine &klUn = mvKeylinesUn[vCell[j]];

                    float delta2x = klUn.startPointX - klUn.endPointX;
                    float delta2y = klUn.startPointY - klUn.endPointY;
                    float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
                    delta2x /= norm_delta2;
                    delta2y /= norm_delta2;
                    float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

                    if(CosSita < TH)
                        continue;

                    Eigen::Vector3d Lfunc = mvKeyLineFunctions[vCell[j]]; 
                    const float dist = Lfunc(0)*x[i] + Lfunc(1)*y[i] + Lfunc(2);

                    if(fabs(dist)<r)
                    {
                        if(vIndices_set.find(vCell[j]) == vIndices_set.end())
                        {
                            vIndices.push_back(vCell[j]);
                            vIndices_set.insert(vCell[j]);
                        }
                    }
                }
            }
        }
    }
    
    return vIndices;
}

vector<size_t> Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                                     const int minLevel, const int maxLevel, const float TH) const
{
    vector<size_t> vIndices;

    vector<KeyLine> vkl = this->mvKeylinesUn;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>0);

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(size_t i=0; i<vkl.size(); i++)
    {
        KeyLine keyline = vkl[i];

        // 1.对比中点距离
        float distance = (0.5*(x1+x2)-keyline.pt.x)*(0.5*(x1+x2)-keyline.pt.x)+(0.5*(y1+y2)-keyline.pt.y)*(0.5*(y1+y2)-keyline.pt.y);
        if(distance > r*r)
            continue;

        float delta2x = vkl[i].startPointX - vkl[i].endPointX;
        float delta2y = vkl[i].startPointY - vkl[i].endPointY;
        float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
        delta2x /= norm_delta2;
        delta2y /= norm_delta2;
        float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

        if(CosSita < TH)
            continue;

        // 3.比较金字塔层数
        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)
                continue;
            if(maxLevel>=0 && keyline.octave>maxLevel)
                continue;
        }

        vIndices.push_back(i);
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);    //N是关键点的个数
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N); //没有畸变的特征点
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
