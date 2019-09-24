#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <iostream>
#include "math.h"

#include "Converter.h"
#include  "lineEdge.h"

void build_data(std::vector<Eigen::Vector3d>& funcline, std::vector<cv::Point3f>& pts3d, std::vector<cv::Point2f>& pts2d,
                std::vector<cv::Vec6f>& lns3d, std::vector<cv::Vec4f>& lns2d,
                cv::Mat K, Eigen::Matrix3d R, Eigen::Vector3d t)
{
    float noise_pixel = 1;
    auto get_rand = [noise_pixel](){return noise_pixel*((rand()%1001-500)/500.0);};

    float fx= K.at<float>(0,0);
    float fy= K.at<float>(1,1);
    float cx = K.at<float>(0,2);
    float cy = K.at<float>(1,2);

    pts3d = {cv::Point3f(0,0,0), cv::Point3f(0,0,3),cv::Point3f(1,0.5,1),cv::Point3f(0.5,-0.5,4),cv::Point3f(-0.5,-1,0),cv::Point3f(-1,0.5,6), cv::Point3f(1,1,3), cv::Point3f(-1,2,5), cv::Point3f(5,2,-1)};

    pts2d.clear();
    for(const auto & pt: pts3d)
    {
        Eigen::Vector3d pt_new = R*Eigen::Vector3d(pt.x,pt.y,pt.z) + t;
        pts2d.push_back(cv::Point2f(fx*pt_new(0)/pt_new(2)+cx, fy*pt_new(1)/pt_new(2)+cy));
    }

    lns3d = {cv::Vec6f(3,2,0,2,-1,1),cv::Vec6f(2,1,0,-3,3,2),cv::Vec6f(0,0,0,-2,-2,0), cv::Vec6f(-1,2,0,3,-1,-1), cv::Vec6f(1,2,0,4,-1,0), cv::Vec6f(3,1,2,6,-2,1)};
    lns2d.clear();
    for(const auto& ln: lns3d)
    {
        Eigen::Vector3d pt1, pt2;
        pt1<<ln[0],ln[1],ln[2];
        pt2<<ln[3],ln[4],ln[5];
        pt1 = R*pt1+t;
        pt2 = R*pt2+t;
        lns2d.push_back(cv::Vec4f(fx*pt1(0)/pt1(2)+cx, fy*pt1(1)/pt1(2)+cy, fx*pt2(0)/pt2(2)+cx, fy*pt2(1)/pt2(2)+cy));
        //cerr<<pt1(2)<<" , "<<pt2(2)<<endl;
    }

   for(auto& pt:pts2d)
    {
        pt.x+=get_rand();
        pt.y+=get_rand();
    }
    for(auto& ln:lns2d)
    {
        float dx = ln[2]-ln[0];
        float dy = ln[3]-ln[1];
        float n = hypotf(dx,dy);
        dx/=n;
        dy/=n;
        ln[0]+=dx*0.1;
        ln[1]+=dy*0.1;
        ln[2]-=dx-0.2;
        ln[3]-=dy-0.2;

        ln[0]+=get_rand();
        ln[1]+=get_rand();
        ln[2]+=get_rand();
        ln[3]+=get_rand();
    }

    funcline.clear();

    for(auto& ln:lns2d){
        Eigen::Vector3d t1(ln[0], ln[1], 1);
        Eigen::Vector3d t2(ln[2], ln[3], 1);
        Eigen::Vector3d t = t1.cross(t2);
        t = t/sqrt(t[0]*t[0] + t[1] * t[1]);
        funcline.push_back(t);
    }
}

int pnpl(std::vector<Eigen::Vector3d> funcline, std::vector<cv::Vec6f> lns3d, std::vector<cv::Point2f> pts2d, std::vector<cv::Point3f> pts3d, cv::Mat K, cv::Mat &R, cv::Mat &t, bool isUsePoint, bool isUseLine, cv::Mat mTcw)
{
    double invSigma = 1;

    // 1.构造求解器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Eigen::Vector3d trans(0.1,0.1,0.1);
    Eigen::Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q,trans);
    vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(mTcw));
    //vSE3->setEstimate(pose);
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pts2d.size();

    //vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vector<bool> vbEdgeMonoOuter;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);
    vbEdgeMonoOuter.resize(N, false);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);
    const float deltaLend = sqrt(3.84);

    if(isUsePoint){
        for(int i=0; i<N; i++)
        {
            Eigen::Matrix<double,2,1> obs;
            obs << pts2d[i].x, pts2d[i].y;

            /*g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());*/

            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Eigen::Vector3d(pts3d[i].x, pts3d[i].y, pts3d[i].z));
            vPoint->setId(i+1);
            vPoint->setFixed(true);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPoint));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity());
            
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            //rk->setDelta(deltaMono);

            e->fx = K.at<float>(0, 0);
            e->fy = K.at<float>(1, 1);
            e->cx = K.at<float>(0, 2);
            e->cy = K.at<float>(1, 2);

            optimizer.addEdge(e);

            vpEdgesMono.push_back(e);
            vnIndexEdgeMono.push_back(i);
        }
    }

    ///添加相机位姿和特征线段之间的误差边
    // Set MapLine vertices
    const int NL = funcline.size();
    vector<bool> vbEdgeLineEdgeOuter;
    vbEdgeLineEdgeOuter.resize(NL, false);

    // 起始点
    //vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineSp;
    vector<EdgeLineProjectXYZ*> vpEdgesLineSp;
    vector<size_t> vnIndexLineEdgeSp;
    vpEdgesLineSp.reserve(NL);
    vnIndexLineEdgeSp.reserve(NL);

    // 终止点
    //vector<EdgeLineProjectXYZOnlyPose*> vpEdgesLineEp;
    vector<EdgeLineProjectXYZ*> vpEdgesLineEp;
    vector<size_t> vnIndexLineEdgeEp;
    vpEdgesLineEp.reserve(NL);
    vnIndexLineEdgeEp.reserve(NL);

    if(isUseLine){
        for(int i=0; i<NL; i++)
        {
                    Eigen::Vector3d line_obs;
                    line_obs = funcline[i];

                    Eigen::Vector3d pt1;
                    pt1<<lns3d[i][0],lns3d[i][1],lns3d[i][2];

                    Eigen::Vector3d pt2;
                    pt2<<lns3d[i][3],lns3d[i][4],lns3d[i][5];

                    // 特征线段的起始点
                    /*EdgeLineProjectXYZOnlyPose* els = new EdgeLineProjectXYZOnlyPose();
                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity());*/

                    g2o::VertexSBAPointXYZ* vPointS = new g2o::VertexSBAPointXYZ();
                    vPointS->setEstimate(pt1);
                    vPointS->setId(N+ 2 * i+1);
                    vPointS->setFixed(true);
                    vPointS->setMarginalized(true);
                    optimizer.addVertex(vPointS);

                    EdgeLineProjectXYZ* els = new EdgeLineProjectXYZ();
                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPointS));
                    els->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber* rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    //rk_line_s->setDelta(deltaMono);

                    els->fx = K.at<float>(0, 0);
                    els->fy = K.at<float>(1, 1);
                    els->cx = K.at<float>(0, 2);
                    els->cy = K.at<float>(1, 2);

                    //els->Xw = pt1;

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    // 特征点的终止点
                    /*EdgeLineProjectXYZOnlyPose* ele = new EdgeLineProjectXYZOnlyPose();
                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity());*/

                    g2o::VertexSBAPointXYZ* vPointE = new g2o::VertexSBAPointXYZ();
                    vPointE->setEstimate(pt2);
                    vPointE->setId(N+ 2 * i+2);
                    vPointE->setFixed(true);
                    vPointE->setMarginalized(true);
                    optimizer.addVertex(vPointE);

                    EdgeLineProjectXYZ* ele = new EdgeLineProjectXYZ();
                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vPointE));
                    ele->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber* rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    //rk_line_e->setDelta(deltaMono);

                    ele->fx = K.at<float>(0, 0);
                    ele->fy = K.at<float>(1, 1);
                    ele->cx = K.at<float>(0, 2);
                    ele->cy = K.at<float>(1, 2);

                    //ele->Xw = pt2;

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
        }
    }

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const float chi2LEnd[4]={3.84,3.84,3.84, 3.84};
    const int its[4]={10,10,10,10};    

    int nBad=0;     //点特征
    int nLineBad=0; //线特征
    for(size_t it=0; it<1; it++)
    {

        Eigen::Vector3d trans(0.1,0.1,0.1);
        Eigen::Quaterniond q;
        q.setIdentity();
        g2o::SE3Quat pose(q,trans);

        vSE3->setEstimate(ORB_SLAM2::Converter::toSE3Quat(mTcw));
        //vSE3->setEstimate(pose);
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        if(isUsePoint){
            nBad=0;
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
            {
                //g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
                g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if(vbEdgeMonoOuter[i])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if(chi2>chi2Mono[it])
                {                
                    vbEdgeMonoOuter[i]=true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    vbEdgeMonoOuter[i]=false;
                    e->setLevel(0);
                }

                if(it==2)
                    e->setRobustKernel(0);
            }
        }

        if(isUseLine){
            nLineBad=0;
            for(size_t i=0, iend=vpEdgesLineSp.size(); i<iend; i++)
            {
                //EdgeLineProjectXYZOnlyPose* e1 = vpEdgesLineSp[i];  //线段起始点
                //EdgeLineProjectXYZOnlyPose* e2 = vpEdgesLineEp[i];  //线段终止点
                EdgeLineProjectXYZ* e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeLineProjectXYZ* e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];

                if(vbEdgeLineEdgeOuter[i])
                {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chi2();
                const float chi2_e = e2->chi2();

                if(chi2_s > chi2LEnd[it] || chi2_e > chi2LEnd[it])
                {
                    vbEdgeLineEdgeOuter[i]=true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                } else
                {
                    vbEdgeLineEdgeOuter[i]=false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if(it==2)
                {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }
        }

    
        if(optimizer.edges().size()<3)
                break;
    }    

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    Eigen::MatrixXd T = Eigen::Isometry3d(vSE3_recov->estimate()).matrix();
    R = (cv::Mat_<float>(3,3)<< T(0,0),T(0,1),T(0,2),
                                T(1,0),T(1,1),T(1,2),
                                T(2,0),T(2,1),T(2,2));
    t = (cv::Mat_<float>(3,1)<<T(0,3),T(1,3),T(2,3));

    if(isUsePoint)
    {
        return nBad;
    }else if(isUseLine)
    {
        return nLineBad;
    }
    return nBad;
}

int main (){

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    std::vector<cv::Vec6f> lns3d;
    std::vector<cv::Vec4f> lns2d;
    std::vector<Eigen::Vector3d> funcline;

    // Camera.fx: 535.4, Camera.fy: 539.2, Camera.cx: 320.1, Camera.cy: 247.6
    cv::Mat K;
    K = (cv::Mat_<float>(3,3)<<535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1);

    Eigen::Matrix3d R = (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())*
                        Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitY())*
                        Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX())
                        ).matrix();
    Eigen::Vector3d t(1,-2,10);

    std::cout<<"["<<R<<"]"<<std::endl;
    std::cout<<"["<<t<<"]"<<std::endl;
    
    build_data(funcline, pts3d,pts2d,lns3d,lns2d,K, R, t);

    cv::Mat R_result, t_result;
    cv::Mat mTcw(4, 4, CV_32F);
    /*mTcw.at<float>(0,0) = R(0); mTcw.at<float>(0,1) = R(1); mTcw.at<float>(0,2) = R(2);
    mTcw.at<float>(1,0) = R(3); mTcw.at<float>(1,1) = R(4); mTcw.at<float>(1,2) = R(5);
    mTcw.at<float>(2,0) = R(6); mTcw.at<float>(2,1) = R(7); mTcw.at<float>(2,2) = R(8);
    mTcw.at<float>(0,3) = t(0); mTcw.at<float>(1,3) = t(1); mTcw.at<float>(2,3) = t(2);
    mTcw.at<float>(3,0) = 0; mTcw.at<float>(3,1) = 0; mTcw.at<float>(3,2) = 0; mTcw.at<float>(3,3) = 1;*/

    mTcw.at<float>(0,0) = R(0); mTcw.at<float>(1,0) = R(1); mTcw.at<float>(2,0) = R(2);
    mTcw.at<float>(0,1) = R(3); mTcw.at<float>(1,1) = R(4); mTcw.at<float>(2,1) = R(5);
    mTcw.at<float>(0,2) = R(6); mTcw.at<float>(1,2) = R(7); mTcw.at<float>(2,2) = R(8);
    mTcw.at<float>(0,3) = t(0) + 10; mTcw.at<float>(1,3) = t(1) + 10; mTcw.at<float>(2,3) = t(2)-4;
    mTcw.at<float>(3,0) = 0; mTcw.at<float>(3,1) = 0; mTcw.at<float>(3,2) = 0; mTcw.at<float>(3,3) = 1;

    std::cout<<mTcw<<std::endl;
    std::cout << "nBad = " << pnpl(funcline, lns3d, pts2d, pts3d, K, R_result, t_result, false, true, mTcw)<<std::endl;

    std::cout<<std::endl;
    std::cout<<"______________________________"<<std::endl;
    std::cout<<std::endl;

    std::cout<<R_result<<std::endl;
    std::cout<<t_result<<std::endl;

    return 0;
}