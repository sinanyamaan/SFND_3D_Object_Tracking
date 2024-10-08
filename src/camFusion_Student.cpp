
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

double getPercentile(const vector<double> &data, const double percentile)
{
    vector sortedData(data);
    sort(sortedData.begin(), sortedData.end());

    const auto n = sortedData.size();
    const auto index = (n - 1) * percentile + 1;

    if(index == floor(index))
    {
        return sortedData[index - 1];
    }
    const auto lower = floor(index);
    const auto upper = ceil(index);
    return sortedData[lower - 1] + (sortedData[upper - 1] - sortedData[lower - 1]) * (index - lower);
}

vector<double> removeOutlierPoints(const vector<LidarPoint> &data)
{

    vector<double> lidar_data_x;
    for(const auto& p: data)
        lidar_data_x.push_back(p.x);

    const auto q1 = getPercentile(lidar_data_x, 0.25);
    const auto q3 = getPercentile(lidar_data_x, 0.75);
    const auto iqr = q3 - q1;
    const auto lower_bound = q1 - 1.5 * iqr;
    const auto upper_bound = q3 + 1.5 * iqr;

    for(auto it = lidar_data_x.begin(); it != lidar_data_x.end();)
    {
        if(*it < lower_bound || *it > upper_bound)
        {
            it = lidar_data_x.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return lidar_data_x;
}

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto & lidarPoint : lidarPoints)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<double> kptDistances;
    for (cv::DMatch match : kptMatches) {
        const auto pt_curr = kptsCurr[match.trainIdx].pt;
        const auto pt_prev = kptsPrev[match.queryIdx].pt;
        if (boundingBox.roi.contains(pt_curr)) {
            boundingBox.kptMatches.push_back(match);
            kptDistances.push_back(cv::norm(pt_curr - pt_prev));
        }
    }

    const auto q1 = getPercentile(kptDistances, 0.25);
    const auto q3 = getPercentile(kptDistances, 0.75);
    const auto iqr = q3 - q1;
    const auto lower_bound = q1 - 1.5 * iqr;
    const auto upper_bound = q3 + 1.5 * iqr;

    for(auto it = boundingBox.kptMatches.begin(); it != boundingBox.kptMatches.end();)
    {
        const auto pt_curr = kptsCurr[it->trainIdx].pt;
        const auto pt_prev = kptsPrev[it->queryIdx].pt;

        const auto distance = cv::norm(pt_curr - pt_prev);

        if(distance < lower_bound || distance > upper_bound)
        {
            it = boundingBox.kptMatches.erase(it);
        }
        else
        {
            ++it;
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            const double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            const double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            constexpr double minDist = 100.0;

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    if (distRatios.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    const auto medianDistRatio = distRatios[distRatios.size() / 2];

    TTC = -1.0 / frameRate / (1 - medianDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    auto lidarPointsPrevFiltered = removeOutlierPoints(lidarPointsPrev);
    auto lidarPointsCurrFiltered = removeOutlierPoints(lidarPointsCurr);

    sort(lidarPointsPrevFiltered.begin(), lidarPointsPrevFiltered.end());
    sort(lidarPointsCurrFiltered.begin(), lidarPointsCurrFiltered.end());

    const auto prevSize = lidarPointsPrevFiltered.size();
    const auto currSize = lidarPointsCurrFiltered.size();

    TTC = lidarPointsCurrFiltered[currSize/2] * (1.0 / frameRate) / (lidarPointsPrevFiltered[prevSize/2] - lidarPointsCurrFiltered[currSize/2]);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<pair<int, int>, vector<double>> bbMatchCount;
    vector<double> distances;

    for(const auto& match : matches)
    {
        cv::Point prev_pt = prevFrame.keypoints[match.queryIdx].pt;
        cv::Point curr_pt = currFrame.keypoints[match.trainIdx].pt;

        for(auto& bb_prev : prevFrame.boundingBoxes)
        {
            if(bb_prev.roi.contains(prev_pt))
            {
                for(auto& bb_curr : currFrame.boundingBoxes)
                {
                    if(bb_curr.roi.contains(curr_pt))
                    {
                        const auto dist = abs(cv::norm(prev_pt - curr_pt));
                        bbMatchCount[make_pair(bb_prev.boxID, bb_curr.boxID)].push_back(dist);
                        distances.push_back(dist);
                    }
                }
            }
        }
    }

    // Remove outliers based on IQR
    const auto q1 = getPercentile(distances, 0.25);
    const auto q3 = getPercentile(distances, 0.75);
    const auto iqr = q3 - q1;
    const auto lower_bound = q1 - 1.5 * iqr;
    const auto upper_bound = q3 + 1.5 * iqr;

    for(auto outer_it = bbMatchCount.begin(); outer_it != bbMatchCount.end();)
    {
        for(auto inner_it  = outer_it->second.begin(); inner_it != outer_it->second.end();)
        {
            if(*inner_it < lower_bound || *inner_it > upper_bound)
            {
                inner_it = outer_it->second.erase(inner_it);
            }
            else
            {
                ++inner_it;
            }
        }

        if(outer_it->second.empty())
        {
            outer_it = bbMatchCount.erase(outer_it);
        }
        else
        {
            ++outer_it;
        }
    }

    vector<pair<pair<int, int>, vector<double>> > bbMatchCountVec(bbMatchCount.begin(), bbMatchCount.end());
    sort(bbMatchCountVec.begin(), bbMatchCountVec.end(), [](const auto &a, const auto &b)
    {
        return a.second.size() > b.second.size();
    });


    vector<int> prevFrameBoxIDs(0);
    vector<int> currFrameBoxIDs(0);

    for(const auto& [bb_pair, kp_distances] : bbMatchCountVec)
    {
        const auto& [prev_bb, curr_bb] = bb_pair;

        if(find(prevFrameBoxIDs.begin(), prevFrameBoxIDs.end(), prev_bb) == prevFrameBoxIDs.end() &&
              find(currFrameBoxIDs.begin(), currFrameBoxIDs.end(), curr_bb) == currFrameBoxIDs.end())
          {
                bbBestMatches[prev_bb] = curr_bb;
                prevFrameBoxIDs.push_back(prev_bb);
                currFrameBoxIDs.push_back(curr_bb);
          }
    }
}
