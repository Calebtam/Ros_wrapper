#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <regex>

using namespace std;
using namespace cv;

struct point {
    Eigen::Vector2f p;
    Eigen::Vector2f v;
};


class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4); 
    std::vector<cv::KeyPoint> vKeys;                                                             
    cv::Point2i UL, UR, BL, BR;                                                                  
    std::list<ExtractorNode>::iterator lit;                                                     
    bool bNoMore;                                                                                
};
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfx = ceil(static_cast<float>(UR.x - UL.x) / 2);
    const int halfy = ceil(static_cast<float>(BR.y - UL.y) / 2);

    // 矩形切四块
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfx, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfy);
    n1.BR = cv::Point2i(UL.x + halfx, UL.y + halfy);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfy);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    for (size_t i = 0; i < vKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}
void GridFastDetector(cv::Mat& image_input, std::vector<cv::KeyPoint>& kp, int num)
{
    const float W = 64;
    int features_needs = num;

    const int max_iter = 4; 
    const float EDGE = 6;
    const float detector_width = image_input.cols - 2 * EDGE;
    const float detector_height = image_input.rows - 2 * EDGE;

    const int nCols = detector_width/W;   
    const int nRows = detector_height/W;  
    const int wCell = ceil(detector_width/nCols);  
    const int hCell = ceil(detector_height/nRows); 
    
    std::vector<cv::KeyPoint> all_keypoints;
    std::vector<cv::Point2f> kkkps;
    all_keypoints.clear();
    all_keypoints.resize(features_needs * 20);  

    for(int i = 0; i <= nRows; i++)
    {
        const float iniY = i * hCell + EDGE;
        float maxY = iniY + hCell + 1;
        if(iniY < 0 || iniY > detector_height + EDGE - 2)
            continue;
        if(maxY >= detector_height + EDGE)
        {
            maxY = detector_height + EDGE;
        }

        for(int j = 0; j <= nCols; j++)
        {
            const float iniX = j * wCell + EDGE;
            float maxX = iniX + wCell + 1;
            if(iniX < 0 || iniX > detector_width + EDGE - 2)
                continue;
            if(maxX >= detector_width + EDGE)
            {
                maxX = detector_width + EDGE;    
            }
        
            std::vector<cv::KeyPoint> vKeysCell;
            cv::FAST(image_input.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, 20, true); // 20

            if(vKeysCell.empty()) {
                cv::FAST(image_input.rowRange(iniY,maxY).colRange(iniX,maxX), vKeysCell, 10, true); // 10
            }

            if(!vKeysCell.empty())
            {
                for(std::vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                {
                    (*vit).pt.x+=(j*wCell+ EDGE);
                    (*vit).pt.y+=(i*hCell+ EDGE);
            
                    // these initial features detected from fast corner isnt locate in mask 
                    // if(mask.at<uchar>((*vit).pt) > 0)
                    {
                        all_keypoints.push_back((*vit));
                        // kkkps.push_back((*vit).pt);
                        // LOG(INFO) << "  (*vit)[i].pt  all_keypoints  " <<  (*vit).pt.x << "  " << (*vit).pt.y << "  " << all_keypoints.size();

                    }
                    // else
                    // {
                    //     kkkps.push_back((*vit).pt);
                    //     // LOG(INFO) << "  (*vit)[i].pt  kkkps  " <<  (*vit).pt.x << "  " << (*vit).pt.y << "  " << all_keypoints.size();
                    // }
                }
            }   
        }
    }

    // quadtree construct
    const int init_node_num = round(static_cast<float>(detector_width / detector_height));
    const float interval_x = static_cast<float>(detector_width / init_node_num) ;

    std::vector<ExtractorNode *> init_nodes;
    init_nodes.resize(init_node_num);
    std::list<ExtractorNode> list_allNodes;
    list_allNodes.clear();
    init_nodes.clear();

    // root tree 
    for (int i = 0; i < init_node_num; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(interval_x * static_cast<float>(i) + EDGE, 0 + EDGE);
        ni.UR = cv::Point2i(interval_x * static_cast<float>(i + 1) + EDGE, 0 + EDGE);
        ni.BL = cv::Point2i(ni.UL.x, detector_height + EDGE);
        ni.BR = cv::Point2i(ni.UR.x, detector_height + EDGE);
        ni.vKeys.reserve( features_needs * 20 / init_node_num);

        list_allNodes.push_back(ni);
        init_nodes[i] = &list_allNodes.back();
    }
    for (size_t i = 0; i < all_keypoints.size(); i++)
    {
        const cv::KeyPoint &kp = all_keypoints.at(i);
        if(kp.pt.x <= 0 || kp.pt.y <= 0 || kp.pt.x >= image_input.cols || kp.pt.y >= image_input.rows) continue;
        init_nodes[std::floor((kp.pt.x - EDGE) / interval_x)]->vKeys.push_back(kp); 
    }
    std::list<ExtractorNode>::iterator list_iter = list_allNodes.begin();
    while (list_iter != list_allNodes.end())
    {
        if (list_iter->vKeys.size() == 1)
        {
            list_iter->bNoMore = true;
            list_iter++;
        }
        else if (list_iter->vKeys.empty())
            list_iter = list_allNodes.erase(list_iter);
        else
            list_iter++;
    }

    bool is_finish = false;
    int iteration = 0;
    std::vector<std::pair<int, ExtractorNode *>> keys_size_and_node;
    keys_size_and_node.reserve(list_allNodes.size() * 4);
  
    while (!is_finish)
    {
        iteration++;
        int pre_size = list_allNodes.size();

        list_iter = list_allNodes.begin();
        int to_expand_num = 0;
        keys_size_and_node.clear();
        
        while (list_iter != list_allNodes.end())
        {
            if (list_iter->bNoMore)  
            {
                list_iter++;
                continue;
            }
            else
            {
                ExtractorNode n1, n2, n3, n4;
                list_iter->DivideNode(n1, n2, n3, n4);
                if (n1.vKeys.size() > 0)
                {
                    list_allNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        to_expand_num++;
                        keys_size_and_node.push_back(std::make_pair(n1.vKeys.size(), &list_allNodes.front()));
                        list_allNodes.front().lit = list_allNodes.begin();
                    }
                }
                
                if (n2.vKeys.size() > 0)
                {
                    list_allNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        to_expand_num++;
                        keys_size_and_node.push_back(std::make_pair(n2.vKeys.size(), &list_allNodes.front()));
                        list_allNodes.front().lit = list_allNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    list_allNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        to_expand_num++;
                        keys_size_and_node.push_back(std::make_pair(n3.vKeys.size(), &list_allNodes.front()));
                        list_allNodes.front().lit = list_allNodes.begin();
                    }
                }

                if (n4.vKeys.size() > 0)
                {
                    list_allNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        to_expand_num++;
                        keys_size_and_node.push_back(std::make_pair(n4.vKeys.size(), &list_allNodes.front()));
                        list_allNodes.front().lit = list_allNodes.begin();
                    }
                }
                list_iter = list_allNodes.erase(list_iter);
                continue;
            }
        }

        if ((int)list_allNodes.size() >= features_needs || (int)list_allNodes.size() == pre_size || iteration > max_iter)
        {
            is_finish = true;
        }
        else if (((int)list_allNodes.size() + to_expand_num * 3) > features_needs) 
        {

            while (!is_finish)
            {
                pre_size = list_allNodes.size();

                std::vector<std::pair<int, ExtractorNode *>> prev_size_and_node = keys_size_and_node;
                keys_size_and_node.clear();

                sort(prev_size_and_node.begin(), prev_size_and_node.end());
                for (int j = prev_size_and_node.size() - 1; j >= 0; j--)
                {

                    ExtractorNode n1, n2, n3, n4;
                    prev_size_and_node[j].second->DivideNode(n1, n2, n3, n4);

                    if (n1.vKeys.size() > 0)
                    {
                        list_allNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            keys_size_and_node.push_back(std::make_pair(n1.vKeys.size(), &list_allNodes.front()));
                            list_allNodes.front().lit = list_allNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        list_allNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            keys_size_and_node.push_back(std::make_pair(n2.vKeys.size(), &list_allNodes.front()));
                            list_allNodes.front().lit = list_allNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        list_allNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            keys_size_and_node.push_back(std::make_pair(n3.vKeys.size(), &list_allNodes.front()));
                            list_allNodes.front().lit = list_allNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        list_allNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            keys_size_and_node.push_back(std::make_pair(n4.vKeys.size(), &list_allNodes.front()));
                            list_allNodes.front().lit = list_allNodes.begin();
                        }
                    }

                    list_allNodes.erase(prev_size_and_node[j].second->lit);
                    if ((int)list_allNodes.size() >= features_needs)
                        break;
                }
                if ((int)list_allNodes.size() >= features_needs || (int)list_allNodes.size() == pre_size)
                    is_finish = true;
            }
        }

    }

    std::vector<cv::KeyPoint> result_keys;
    result_keys.reserve(features_needs);
    for (std::list<ExtractorNode>::iterator Liter = list_allNodes.begin(); Liter != list_allNodes.end(); Liter++)
    {
        std::vector<cv::KeyPoint> &node_keys = Liter->vKeys;
        cv::KeyPoint *keypoint = &node_keys[0];
        float max_response = keypoint->response;

        for (size_t k = 1; k < node_keys.size(); k++)
        {
            if (node_keys[k].response > max_response)
            {
                keypoint = &node_keys[k];
                max_response = node_keys[k].response;
            }
        }
        // result_keys.push_back(*keypoint);
        kp.push_back(*keypoint);
    }
    // kp = result_keys;
}

void test(std::vector<cv::Mat>& img) {
    float show = 1.5;
    // Extract image pyramid
    int pyr_levels = 3;
    cv::Size win_size = cv::Size(15, 15);
    std::vector<std::vector<cv::Mat>> imgpyr;
    // cv::Mat logImage(img.at(0).size(), CV_32FC1);
    // cv::log(img.at(0) + 1, logImage);
    // cv::imshow(" by opencv", logImage);
    // cv::imshow(" by opencv", logTransform2(img.at(0)));
    for(auto img1 : img)
    {
        std::vector<cv::Mat> imgpyr_curr(pyr_levels);
        cv::buildOpticalFlowPyramid(img1, imgpyr_curr, win_size, pyr_levels);
        imgpyr.push_back(imgpyr_curr);
    }
    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    // Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    // detector->detect(img1, kp1);
    // std::cout << "1、 " << std::endl;

    GridFastDetector(img.at(0), kp1, 50);
    // use opencv's flow for validation

    // std::cout << "kp1.size() " << kp1.size() << std::endl;

    vector<Point2f> pt1, pt2, pt3;
    // for(int i = 0; i < Lpt1.size(); i++)
    // {
    //     pt1.push_back(cv::Point2f(Lpt1.at(i).p.x(), Lpt1.at(i).p.y()));     // curr left

    //     if(Lpt1.at(i).p.x()-Lpt1.at(i).v.x() < img.at(0).rows && Lpt1.at(i).p.y()-Lpt1.at(i).v.y() < img.at(0).cols)
    //         pt2.push_back(cv::Point2f(Lpt1.at(i).p.x()-Lpt1.at(i).v.x(), Lpt1.at(i).p.y()-Lpt1.at(i).v.x()));   // next left
    //     else
    //         pt2.push_back(cv::Point2f(Lpt1.at(i).p.x(), Lpt1.at(i).p.y()));

    //     pt3.push_back(cv::Point2f(Lpt1.at(i).p.x(), Lpt1.at(i).p.y()));     // curr left
    // }
    // for(int i = 0; i < Lpt2.size(); i++)
    // {
    //     pt1.push_back(cv::Point2f(Lpt2.at(i).p.x(), Lpt2.at(i).p.y()));     // curr left
    //     pt2.push_back(cv::Point2f(Lpt1.at(i).p.x(), Lpt1.at(i).p.y()));
    //     if(Lpt2.at(i).p.x()-Lpt2.at(i).v.x() < img.at(0).rows && Lpt2.at(i).p.y()-Lpt2.at(i).v.y() < img.at(0).cols)
    //         pt3.push_back(cv::Point2f(Lpt2.at(i).p.x()-Lpt2.at(i).v.x(), Lpt2.at(i).p.y()-Lpt2.at(i).v.y()));   // curr right
    //     else
    //         pt3.push_back(cv::Point2f(Lpt2.at(i).p.x(), Lpt2.at(i).p.y()));
    // }
    // std::cout << "2、 " << std::endl;

    for (auto &kp: kp1) 
    {
        pt1.push_back(kp.pt);
        pt2.push_back(kp.pt);
        pt3.push_back(kp.pt);
    }
    vector<uchar> status, statu;
    vector<float> error, erro;
    // std::cout << "3、 " << std::endl;

    if(imgpyr.size() >= 2)
    {
        // cv::calcOpticalFlowPyrLK(imgpyr.at(0), imgpyr.at(1), pt1, pt2, status, error, win_size, pyr_levels,
        //     cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        //     cv::OPTFLOW_USE_INITIAL_FLOW);

        cv::calcOpticalFlowPyrLK(imgpyr.at(0), imgpyr.at(1), pt1, pt2, status, error, win_size, pyr_levels,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-3);

        Mat img2_CV;
        cv::cvtColor(img.at(1), img2_CV, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < pt2.size(); i++) {
            if (status[i]) {
                cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0),2);
            }
        }
        cv::resize(img2_CV,img2_CV,cv::Size(img2_CV.cols/show,img2_CV.rows/show));
        cv::imshow("stereo match", img2_CV);
    }
    if(imgpyr.size() >= 3)
    {
        cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);//tune: change 30 to 15
        cv::calcOpticalFlowPyrLK(imgpyr.at(0), imgpyr.at(2), pt1, pt3, statu, erro, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);//time cost! 50%

        Mat img3_CV;
        cv::cvtColor(img.at(2), img3_CV, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < pt3.size(); i++) {
            if (statu[i]) {
                cv::circle(img3_CV, pt3[i], 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img3_CV, pt1[i], pt3[i], cv::Scalar(0, 250, 0),2);
            }
        }
        cv::resize(img3_CV,img3_CV,cv::Size(img3_CV.cols/show,img3_CV.rows/show));
        cv::imshow("left frame track  ", img3_CV);
    }
    // std::cout << "4、 " << std::endl;    

    Mat img1_CV;
    cv::cvtColor(img.at(1), img1_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt1.size(); i++) {
        if (statu[i] || status[i]) {
            cv::circle(img1_CV, pt1[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img1_CV, pt3[i], pt1[i], cv::Scalar(150, 150, 0),2);
            cv::line(img1_CV, pt2[i], pt1[i], cv::Scalar(0, 250, 250),2);
        }
    }
    cv::resize(img1_CV,img1_CV,cv::Size(img1_CV.cols/show,img1_CV.rows/show));
    cv::imshow("left stereo ", img1_CV);
}
