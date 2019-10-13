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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 自己添加的，获取img的width和height
    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    if((mask = imread("./masks/mask.png", cv::IMREAD_GRAYSCALE)).empty())
        mask = cv::Mat();

    /*cout << "img_width = " << img_width << endl;
    cout << "img_height = " << img_height << endl;

    // 自己添加的，在这里先得到映射值
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(img_width, img_height), CV_32F, mUndistX, mUndistY);

    cout << "mUndistX size = " << mUndistX.size << endl;
    cout << "mUndistY size = " << mUndistY.size << endl;*/

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    int nFeaturesLine = fSettings["LINEextractor.nFeatures"];
    float fScaleFactorLine = fSettings["LINEextractor.scaleFactor"];
    int nLevelsLine = fSettings["LINEextractor.nLevels"];
    int min_length = fSettings["LINEextractor.min_line_length"];

    mpLSDextractorLeft = new LINEextractor(nLevelsLine, fScaleFactorLine, nFeaturesLine, min_length);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 在这里纠正畸变图片,自己添加的
    //cv::remap(mImGray, mImGray, mUndistX, mUndistY, cv::INTER_LINEAR);
    /*cv::Mat mask;
    mImGray.copyTo(mask);
    for(int i = 0; i<mask.rows;i++){
        for(int j = 0;j<mask.cols;j++){
            if(i>=14&&i<=464&&j>=14&&j<=624){
                mask.at<uchar>(i,j) = 255;
            }else{
                mask.at<uchar>(i,j) = 0;
            }
        }
    }

    imwrite("./mask.png", mask);*/

    // Kitti数据集
//    mImGray = mImGray(Rect(15, 15, 1211, 346));  //这里是为了剪切掉修正后图片的弧形边缘，针对不同数据集，image的尺寸不一样，所以这里应该改成动态的
    // TUM数据集
    //mImGray = mImGray(Rect(15, 15, 610, 450));*/

    static int count=0;
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpLSDextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mask);
    }
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpLSDextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mask);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    // Track包含两部分：估计运动、跟踪局部地图

    // mState为tracking的状态机
    // SYSTEM_NOT_READY=-1, NO_IMAGES_YET=0, NOT_INITIALIZED=1, OK=2, LOST=3
    // 如果图像复位过，或者第一次运行，则为NO_IMAGES_YET状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.

        // bOK是临时变量，用于表示每个函数是否执行成功
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            // 正常初始化成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧被替换的MapPoints 和 MapLines
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
                    
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();

                    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
                    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
                    cout << "Track time: " << time_used.count() << endl;
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                // mbVO是mbOnlyTracking为true时才有的一个变量
                // mbVO为false，表示此帧匹配了很多的MapPoints,跟踪很正常
                // mbVO为true，表明此帧匹配了很少的MapPoints，少于10个，要跪了
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMapWithLines();
            }

        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMapWithLines();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                // 更新恒速运动模型TrackWithMotionModel中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                    if(pML->Observations()<1)
                    {
                        mCurrentFrame.mvbLineOutlier[i] = false;
                        mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // 只用于双目或rgbd
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }


            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 剔除那些在BA中检测为outlier的3D map点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }

            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                if(mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    int num = 100;
    // 如果单目初始器还没有没创建，则创建单目初始器
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>num)
        {
            // step 1：得到用于初始化的第一帧，初始化需要两帧
            mInitialFrame = Frame(mCurrentFrame);
            // 记录最近的一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPreMatched最大的情况就是当前帧所有的特征点都被匹配上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            // 由当前帧构造初始化器， sigma:1.0    iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            mvIniLastLineMatches = vector<int>(mCurrentFrame.mvKeys.size(), -1);

            mbIniFirst = false;

            return;
        }
    }
    else
    {
        // Try to initialize
        // step2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
        // 如果当前帧特征点太少，重新构造初始器
        // 因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        if((int)mCurrentFrame.mvKeys.size()<=num)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            return;
        }

        // Find correspondences
        // step3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
        // mvbPrevMatched为前一帧的特征点，存储了mInitialFrame中哪些点将进行接下来的匹配,类型  std::vector<cv::Point2f> mvbPrevMatched;
        // mvIniMatches存储mInitialFrame, mCurrentFrame之间匹配的特征点，类型为std::vector<int> mvIniMatches; ????
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        LSDmatcher lmatcher;   //建立线特征之间的匹配
        //int lineMatches = lmatcher.SerachForInitialize(mInitialFrame, mCurrentFrame, mvIniLineMatches);
        int lineMatches = lmatcher.SearchDouble(mLastFrame, mCurrentFrame, mvIniLineMatches);

        // Check if there are enough correspondences
        // step4：如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        if(!mbIniFirst)
        {
            mvIniLastLineMatches = mvIniLineMatches;
            mbIniFirst = true;
        }else{
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                if(j >= 0 ){
                    mvIniLastLineMatches[i] = mvIniLineMatches[j];
                }
            }

            lmatcher.SearchDouble(mInitialFrame,mCurrentFrame, mvIniLineMatches);
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                int k = mvIniLineMatches[i];
                if(j != k){
                    mvIniLastLineMatches[i] = -1;
                }
            }
        }

        mvIniLineMatches = mvIniLastLineMatches;

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        vector<bool> mvbLineTriangulated;   //匹配的线特征是否能够三角化

        cv::Mat cubemapMatch_rgb = DrawLines(&mInitialFrame, &mCurrentFrame);
        cubemapMatch_rgb.copyTo(mpInitializer->cubemapMatch_rgb);

        cv::Mat cubemapMatch_rgb_Temp;
        cubemapMatch_rgb.copyTo(cubemapMatch_rgb_Temp);
        for(int i = 0;i<mvIniLineMatches.size();i++){
            cv::Point Point_1, Point_2;
            int pa = mvIniLineMatches[i];

            if(pa<0)
                continue;

            KeyLine line_1 = mInitialFrame.mvKeylinesUn[i];
            KeyLine line_2 = mCurrentFrame.mvKeylinesUn[pa];
            Point_1.x = (line_1.startPointX + line_1.endPointX)/2.0;
            Point_1.y = (line_1.startPointY + line_1.endPointY)/2.0;
            Point_2.x = (line_2.startPointX + line_2.endPointX)/2.0 + mCurrentFrame.ImageGray.cols;
            Point_2.y = (line_2.startPointY + line_2.endPointY)/2.0;
            cv::line(cubemapMatch_rgb_Temp, Point_1, Point_2, Scalar(0,0,255), 1, CV_AA);
        }
            
        cv::imwrite("./matchResultTempIni.jpg", cubemapMatch_rgb_Temp);

        // step5：通过H或者F进行单目初始化，得到两帧之间相对运动，初始化MapPoints
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvIniLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            for(size_t i=0, iend=mvIniLineMatches.size(); i<iend;i++)
            {
                if(mvIniLineMatches[i]>=0 && !mvbLineTriangulated[i])
                {
                    mvIniLineMatches[i]=-1;
                    lineMatches--;
                }
            }

            // Set Frame Poses
            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // step6：将三角化得到的3D点包装成MapPoints
            /// 如果要修改，应该是从这个函数开始
            CreateInitialMapMonoWithLine();
        }

        mLastFrame = Frame(mCurrentFrame);
    }
}

/**
 * @brief 为单目摄像头三角化生成MapPoints，只有特征点
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // step 1: 将初始关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    // step 2：将当前关键帧的描述子转为BoW
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // step3：将关键帧插入到地图，凡是关键帧，都要插入地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // step4：将特征点的3D点包装成MapPoints
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        // step4.1：用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        // step4.2：为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // step4.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame观测到，以及对应的第几个特征点
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();

        // c.更新该MapPoint的平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        // step4.4：在地图中添加该MapPoint
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    // step5:更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每一个边有一个权重，边的权重是该关键帧与当前关键帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // step6：BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    // step7：将MapPoints的中值深度归一化到1，并归一化两帧之间的变换
    // 评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;  //至此，初始化成功
}

/**
 * @brief 为单目摄像头三角化生成带有线特征的Map，包括MapPoints和MapLine
 */
void Tracking::CreateInitialMapMonoWithLine()
{
    // step1:创建关键帧，即用于初始化的前两帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // step2：将两个关键帧的描述子转为BoW，这里的BoW只有ORB的词袋
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // step3：将关键帧插入到地图，凡是关键帧，都要插入地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // step4：将特征点的3D点包装成MapPoints
    for(size_t i=0; i<mvIniMatches.size(); i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        // Create MapPoint
        cv::Mat worldPos(mvIniP3D[i]);

        // step4.1：用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

        // step4.2：为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // step4.3：表示该KeyFrame的哪个特征点对应到哪个3D点
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame观测到，以及对应的第几个特征点
        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add to Map
        // step4.4：在地图中添加该MapPoint
        mpMap->AddMapPoint(pMP);
    }

    // step5：将特征线包装成MapLines
    for(size_t i=0; i<mvIniLineMatches.size(); i++)
    {
        if(mvIniLineMatches[i] < 0)
            continue;

        // Create MapLine
        Vector6d worldPos;
        worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

        //step5.1：用线段的两个端点构造MapLine
        MapLine* pML = new MapLine(worldPos, pKFcur, mpMap);

        //step5.2：为该MapLine添加属性：
        // a.观测到该MapLine的关键帧
        // b.该MapLine的描述子
        // c.该MapLine的平均观测方向和深度范围？

        //step5.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
        pKFini->AddMapLine(pML,i);
        pKFcur->AddMapLine(pML,mvIniLineMatches[i]);

        //a.表示该MapLine可以被哪个KeyFrame观测到，以及对应的第几个特征线
        pML->AddObservation(pKFini, i);
        pML->AddObservation(pKFcur, mvIniLineMatches[i]);

        //b.MapPoint中是选取区分度最高的描述子，pl-slam直接采用前一帧的描述子,这里先按照ORB-SLAM的过程来
        pML->ComputeDistinctiveDescriptors();

        //c.更新该MapLine的平均观测方向以及观测距离的范围
        pML->UpdateAverageDir();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapLines[mvIniLineMatches[i]] = pML;
        mCurrentFrame.mvbLineOutlier[mvIniLineMatches[i]] = false;

        // step5.4: Add to Map
        mpMap->AddMapLine(pML);
    }

    // step6：更新关键帧间的连接关系
    // 1.最初是在3D点和关键帧之间建立边，每一个边有一个权重，边的权重是该关键帧与当前关键帧公共3D点的个数
    // 2.加入线特征后，这个关系应该和特征线也有一定的关系，或者就先不加关系，只是单纯的添加线特征

    // step7：全局BA优化，这里需要再进一步修改优化函数，参照OptimizePose函数
    cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and "<< mpMap->MapLinesInMap() << " lines." << endl;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); //true代表使用有线特征的BA

    // step8：将MapPoints的中值深度归一化到1，并归一化两帧之间的变换
    // Q：MapPoints的中值深度归一化为1，MapLine是否也归一化？
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    cout << "medianDepth = " << medianDepth << endl;
    //cout << "pKFcur->TrackedMapPoints(1) = " << pKFcur->TrackedMapPoints(1) << endl;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<80)
    {
        cout << "Wrong initialization, reseting ... " << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale Points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // Scale Line Segments
    vector<MapLine*> vpAllMapLines = pKFini->GetMapLineMatches();
    for(size_t iML=0; iML < vpAllMapLines.size(); iML++)
    {
        if(vpAllMapLines[iML])
        {
            MapLine* pML = vpAllMapLines[iML];
            pML->SetWorldPos(pML->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mvpLocalMapLines = mpMap->GetAllMapLines();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    // 点特征
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
    // 线特征
    for(int i=0; i<mLastFrame.NL; i++)
    {
        MapLine* pML = mLastFrame.mvpMapLines[i];

        if(pML)
        {
            MapLine* pReL = pML->GetReplaced();
            if(pReL)
            {
                mLastFrame.mvpMapLines[i] = pReL;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    cv::Mat pic = DrawLines(mpReferenceKF, &mCurrentFrame);

    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    LSDmatcher lmatcher(0.8, true);

    lmatcher.pic = pic;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    int lmatches = 0;
    lmatches = lmatcher.SearchDouble(mpReferenceKF, mCurrentFrame);

    //cerr<<lmatches<<"__________________"<<endl;

    if(nmatches<15  && lmatches<5)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // 通过优化3D-2D的重投影误差来获得位姿
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    if(nmatches > 20){
         Optimizer::PoseOptimizationWithPoints(&mCurrentFrame);

         for(int i =0; i<mCurrentFrame.NL; i++)
            if(mCurrentFrame.mvpMapLines[i])
                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);

    }else{
        Optimizer::PoseOptimization(&mCurrentFrame);
    }

    // Discard outliers
    // 剔除优化后的outlier匹配点（MapPoints）
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    int nLinematchesMap = 0;
    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                lmatches--;
                //cerr<< i<<" ";
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }

        mCurrentFrame.mvbLineOutlier[i] = false;
    }
    //cerr<<endl;
    //cerr<<lmatches<<"__________________"<<endl;

    return nmatchesMap>=10;
}


/**
 * @brief 双目或rgbd相机根据深度值为上一帧产生新的MapPoints
 *
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些）
 * 可以通过深度值产生一些新的MapPoints
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // step1: 更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    // 如果上一帧为关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}


/**
 * @brief 根据匀速模型对上一帧的MapPoints进行跟踪
 *
 * 1.非单目情况，需要对上一帧产生一些新的MapPoints(临时)
 * 2.将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3.根据匹配对估计当前帧的姿态
 * 4.根据姿态剔除误匹配
 * @return 如果匹配数大于10，则返回true
 */
bool Tracking::TrackWithMotionModel()
{
    cv::Mat pic = DrawLines(&mLastFrame, &mCurrentFrame);

    // --step1: 建立ORB特征点的匹配
    ORBmatcher matcher(0.9,true);
    LSDmatcher lmatcher;

    lmatcher.pic = pic;

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // --step2: 更新上一帧的位姿
    UpdateLastFrame();

    // --step3:根据Const Velocity Model，估计当前帧的位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    // --step4：根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);
    int lmatches = 0;
    lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, th);
    //lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame);

    //cerr<<lmatches<<"__________________"<<endl;

    // If few matches, uses a wider window search
    // 如果跟踪的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20 && lmatches<5)
        return false;

    // Optimize frame pose with all matches
    // --step5: 优化位姿
    //Optimizer::PoseOptimization(&mCurrentFrame);
    /*if(nmatches > 30){
         Optimizer::PoseOptimizationWithPoints(&mCurrentFrame);
         std::vector<bool> bOutlier = std::vector<bool>(mCurrentFrame.NL, true);
         mCurrentFrame.mvbLineOutlier = bOutlier;
    }else{*/
    cerr<<"TrackWithMotionModel: ";
    Optimizer::PoseOptimization(&mCurrentFrame);
    //}

    // Discard outliers
    // --step6：优化位姿后剔除outlier的mvpMapPoints 和 mvpMapLines
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    int nLinematchesMap = 0;
    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
      //          cerr<< i<<" ";
                lmatches--;
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }

        mCurrentFrame.mvbLineOutlier[i] = false;

    }

    //cerr<<endl;
    //cerr<<lmatches<<"__________________"<<endl;

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

/**
 * @brief 对Local Map的MapPoints进行跟踪
 * 1.更新局部地图，包括局部关键帧和关键点
 * 2.对局部MapPoints进行投影匹配
 * 3.根据匹配对估计当前帧的姿态
 * 4.根据姿态剔除误匹配
 * @return true if success
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/**
 * @brief 类比TrackLocalMap函数，对局部地图的地图点和地图线进行跟踪
 * @return
 */
bool Tracking::TrackLocalMapWithLines()
{
    // step1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints，局部地图线mvpLocalMapLines
    UpdateLocalMap();

    thread threadPoints(&Tracking::SearchLocalPoints, this);
    thread threadLines(&Tracking::SearchLocalLines, this);
    threadPoints.join();
    threadLines.join();

    // step4：更新局部所有MapPoints和MapLines后对位姿再次优化
    cerr<<"TrackMap: ";
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;
    mnLineMatchesInliers = 0;

    // Update MapPoints Statistics
    // step5：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // 更新MapLines Statistics
    // step6：更新当前帧的MapLines被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(!mCurrentFrame.mvbLineOutlier[i])
            {
                mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                        mnLineMatchesInliers++;
                }
                else
                    mnLineMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // step7：决定是否跟踪成功
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/**
 * @brief 判断当前帧是否为关键帧
 * @return 是否需要关键帧
 */
bool Tracking::NeedNewKeyFrame()
{
    // step1:如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // step2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的id
    // mMaxFrames等于图像输入的帧率
    // 如果关键帧比较少，则考虑插入关键帧，或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // step3：得到参考关键帧跟踪到的MapPoints数量
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视成都最高的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // step4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // step6：决策是否需要插入关键帧
    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

/**
 * @brief 对local MapPoints进行跟踪
 *
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::SearchLocalLines()
{
    // step1：遍历在当前帧的mvpMapLines，标记这些MapLines不参与之后的搜索，因为当前的mvpMapLines一定在当前帧的视野中
    for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;
        if(pML)
        {
            if(pML->isBad())
            {
                *vit = static_cast<MapLine*>(NULL);
            } else{
                // 更新能观测到该线段的帧数加1
                pML->IncreaseVisible();
                // 标记该点被当前帧观测到
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该线段将来不被投影，因为已经匹配过
                pML->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // step2：将所有局部MapLines投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapLine*>::iterator vit=mvpLocalMapLines.begin(), vend=mvpLocalMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;

        // 已经被当前帧观测到MapLine，不再判断是否能被当前帧观测到
        if(pML->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pML->isBad())
            continue;

        // step2.1：判断LocalMapLine是否在视野内
        if(mCurrentFrame.isInFrustum(pML, 0.5))
        {
            // 观察到该点的帧数加1，该MapLine在某些帧的视野范围内
            pML->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        LSDmatcher matcher;
        int th = 1;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        int nmatches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);

        // 利用线约束对线进行剔除
        /*if(nmatches)
        {
            for(int i = 0; i<mCurrentFrame.mvpMapLines.size(); i++)
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    Eigen::Vector3d tWorldVector = pML->GetWorldVector();
                    cv::Mat tWorldVector_ = (cv::Mat_<float>(3, 1) << tWorldVector(0), tWorldVector(1), tWorldVector(2));
                    KeyLine tkl = mCurrentFrame.mvKeylinesUn[i];
                    cv::Mat tklS = (cv::Mat_<float>(3, 1) << tkl.startPointX, tkl.startPointY, 1.0);
                    cv::Mat tklE = (cv::Mat_<float>(3, 1) << tkl.endPointX, tkl.endPointY, 1.0);
                    cv::Mat K = mCurrentFrame.mK;
                    cv::Mat tklS_ = K.inv() * tklS; cv::Mat tklE_ = K.inv() * tklE;

                    cv::Mat NormalVector_ = tklS_.cross(tklE_);
                    double norm_ = cv::norm(NormalVector_);
                    NormalVector_ /= norm_;

                    cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
                    cv::Mat tCameraVector_ = Rcw * tWorldVector_;
                    double CosSita = abs(NormalVector_.dot(tCameraVector_));

                    if(CosSita>0.09)
                    {
                        mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                    }
                    //cerr<<CosSita<<endl;
                }
            }
        }*/
    }
}

/**
 * @brief 更新LocalMap
 *
 * 局部地图包括：
 * 1.当前关键帧，与当前关键帧的临近关键帧和参考关键帧
 * 2.由这些关键帧观测到的MapPoints和MapLines
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalLines();
}

/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 *
 * 局部关键帧mvpLocalKeyFrames的MapPoints更新mvpLocalMapPoints
 */
void Tracking::UpdateLocalPoints()
{
    // step1：清空局部MapPoints
    mvpLocalMapPoints.clear();

    // step2：遍历局部关键帧mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // step3：将局部关键帧的MapPoints添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // mnTrackReferenceForFrame防止重复添加局部MapPoint
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalLines()
{
    // step1：清空局部MapLines
    mvpLocalMapLines.clear();

    // step2：遍历局部关键帧mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();

        //step3：将局部关键帧的MapLines添加到mvpLocalMapLines
        for(vector<MapLine*>::const_iterator itML=vpMLs.begin(), itEndML=vpMLs.end(); itML!=itEndML; itML++)
        {
            MapLine* pML = *itML;
            if(!pML)
                continue;
            if(pML->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pML->isBad())
            {
                mvpLocalMapLines.push_back(pML);
                pML->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

cv::Mat Tracking::DrawLines(Frame* F1, Frame* F2)
{
        cv::Mat mImRGBPrevTemp, mImRGBTemp;
        int nRows = F2->ImageGray.rows; 
        int nCols = F2->ImageGray.cols;
        cv::cvtColor(F1->ImageGray, mImRGBPrevTemp, cv::COLOR_GRAY2BGR);
        cv::cvtColor(F2->ImageGray, mImRGBTemp, cv::COLOR_GRAY2BGR);
        cv::Mat cubemapMatch_rgb(nRows, nCols * 2, mImRGBTemp.type());

        drawKeylines(mImRGBPrevTemp, F1->mvKeylinesUn, mImRGBPrevTemp, Scalar(200, 0, 0));
        drawKeylines(mImRGBTemp, F2->mvKeylinesUn, mImRGBTemp, Scalar(0, 200, 0));

        for(int i = 0; i < F1->mvKeylinesUn.size(); i++){
            cv::Point origin;
            KeyLine line_temp = F1->mvKeylinesUn[i];
            origin.x = (line_temp.startPointX + line_temp.endPointX)/2.0;
            origin.y = (line_temp.startPointY + line_temp.endPointY)/2.0;
            cv::putText(mImRGBPrevTemp, std::to_string(i), origin, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(200, 0, 0), 1, 8, 0);
        }
        (mImRGBPrevTemp.rowRange(0, mImRGBPrevTemp.rows).colRange(0, mImRGBPrevTemp.cols)).copyTo(cubemapMatch_rgb.colRange(0, nCols));
            
        for(int i = 0; i < F2->mvKeylinesUn.size(); i++){
            cv::Point origin;
            KeyLine line_temp = F2->mvKeylinesUn[i];
            origin.x = (line_temp.startPointX + line_temp.endPointX)/2.0;
            origin.y = (line_temp.startPointY + line_temp.endPointY)/2.0;
            cv::putText(mImRGBTemp, std::to_string(i), origin, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 200, 0), 1, 8, 0);
        }
        (mImRGBTemp.rowRange(0, mImRGBTemp.rows).colRange(0, mImRGBTemp.cols)).copyTo(cubemapMatch_rgb.colRange(nCols, nCols * 2));

        return cubemapMatch_rgb;
}

cv::Mat Tracking::DrawLines(KeyFrame* KF1, Frame* F2)
{
        cv::Mat mImRGBPrevTemp, mImRGBTemp;
        int nRows = F2->ImageGray.rows; 
        int nCols = F2->ImageGray.cols;
        cv::cvtColor(KF1->ImageGray, mImRGBPrevTemp, cv::COLOR_GRAY2BGR);
        cv::cvtColor(F2->ImageGray, mImRGBTemp, cv::COLOR_GRAY2BGR);
        cv::Mat cubemapMatch_rgb(nRows, nCols * 2, mImRGBTemp.type());

        drawKeylines(mImRGBPrevTemp, KF1->mvKeyLines, mImRGBPrevTemp, Scalar(200, 0, 0));
        drawKeylines(mImRGBTemp, F2->mvKeylinesUn, mImRGBTemp, Scalar(0, 200, 0));

        for(int i = 0; i < KF1->mvKeyLines.size(); i++){
            cv::Point origin;
            KeyLine line_temp = KF1->mvKeyLines[i];
            origin.x = (line_temp.startPointX + line_temp.endPointX)/2.0;
            origin.y = (line_temp.startPointY + line_temp.endPointY)/2.0;
            cv::putText(mImRGBPrevTemp, std::to_string(i), origin, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(200, 0, 0), 1, 8, 0);
        }
        (mImRGBPrevTemp.rowRange(0, mImRGBPrevTemp.rows).colRange(0, mImRGBPrevTemp.cols)).copyTo(cubemapMatch_rgb.colRange(0, nCols));
            
        for(int i = 0; i < F2->mvKeylinesUn.size(); i++){
            cv::Point origin;
            KeyLine line_temp = F2->mvKeylinesUn[i];
            origin.x = (line_temp.startPointX + line_temp.endPointX)/2.0;
            origin.y = (line_temp.startPointY + line_temp.endPointY)/2.0;
            cv::putText(mImRGBTemp, std::to_string(i), origin, cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 200, 0), 1, 8, 0);
        }
        (mImRGBTemp.rowRange(0, mImRGBTemp.rows).colRange(0, mImRGBTemp.cols)).copyTo(cubemapMatch_rgb.colRange(nCols, nCols * 2));

        return cubemapMatch_rgb;
}

} //namespace ORB_SLAM
