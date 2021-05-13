#include "../PhaseCongruency/phase.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp> 
#include <opencv2/calib3d.hpp>
#include <iostream>


using namespace cv;
using namespace std;

#define MAT_TYPE CV_64FC1
#define MAT_TYPE_CNV CV_64F






bool compare_keypoints (KeyPoint i,KeyPoint j) { return (i.response > j.response); }


Mat MIM(Mat im,  std::vector<std::vector<cv::Mat> > eo, int s, int o)
{
    // eo = std::vector<std::vector<Mat> >(nscale,vector<Mat>(norient)); //s * o * height * width
    int yim = im.rows;
    int xim = im.cols;
    vector<Mat> CS(o) ;
    for(int j = 0; j < o; j++){
        Mat CS_tmp = Mat::zeros(yim, xim, MAT_TYPE);
        for(int i = 0; i < s; i++){
            Mat complex[2];
            Mat abs_tmp;
            split(eo[i][j], complex);
            magnitude(complex[0], complex[1], abs_tmp);
            add(CS_tmp, abs_tmp, CS_tmp);     
        }
        CS[j] = CS_tmp;
    }

   
    
    Mat MIM = Mat::zeros(yim, xim, MAT_TYPE);
    for(int y = 0; y < yim; y++){
        for(int x = 0; x < xim; x++){
            int curr_max_o = 0;
            double curr_max = 0;
            for(int curr = 0; curr < o; curr++){
                if (CS[curr].at<double>(y, x) >  curr_max){
                    curr_max = CS[curr].at<double>(y, x);
                    curr_max_o = curr;
                }
            }
            MIM.at<double>(y, x) = (double)curr_max_o;
        }
    }
    return MIM;
}

vector<KeyPoint> ignore_kps(Mat image, vector<KeyPoint> kps, int patch_size){
    int yim = image.rows;
    int xim = image.cols;
    int kps_num = kps.size();

    vector<KeyPoint> remained_kps;
    for (int k = 0; k < kps_num; k++){
        int x = kps[k].pt.x;
        int y = kps[k].pt.y;
        int x1 = x - round(patch_size / 2);
        int y1 = y - round(patch_size / 2);
        int x2 = x + round(patch_size / 2);
        int y2 = y + round(patch_size / 2);
        if (x1 < 0 || y1 < 0 || x2 >= xim || y2 >= yim){
            continue;
        }
        remained_kps.push_back(kps[k]);
    }
    return remained_kps;
}


Mat compute_des(Mat MIM, vector<KeyPoint> kps, int o, int patch_size){
    int yim = MIM.rows;
    int xim = MIM.cols;
    int kps_num = kps.size();
    int ns = 6;
    Mat des(kps_num, 36 * o, MAT_TYPE);
    for (int k = 0; k < kps_num; k++){
        //cout <<k<<" "<< kps[k].pt << endl;
        int x = kps[k].pt.x;
        int y = kps[k].pt.y;
        int x1 = x - round(patch_size / 2);
        int y1 = y - round(patch_size / 2);
        int x2 = x + round(patch_size / 2);
        int y2 = y + round(patch_size / 2);
        Mat patch(MIM, Rect(x1, y1, patch_size, patch_size));
        Mat RIFT_des= Mat::zeros(1, 36 * o, MAT_TYPE);
        for(int j = 0; j < ns; j++){
            for(int i = 0; i < ns; i++){
                int clip_width = round(patch_size/ns);
                int clip_height = round(patch_size/ns);
                while (round(j*patch_size/ns+1)+ clip_width >= patch.cols){
                    clip_width--;
                }
                while (round(i*patch_size/ns+1)+ clip_height >= patch.rows){
                    clip_height--;
                }
                for (int clip_j = round(j*patch_size/ns+1); clip_j < round(j*patch_size/ns+1) + clip_width; clip_j++){
                    for (int clip_i = round(i*patch_size/ns+1); clip_i < round(i*patch_size/ns+1) + clip_width; clip_i++){
                        int oo = patch.at<double>(clip_i, clip_j);
                        RIFT_des.at<double>(0, i * ns * o + j * o + oo) += 1;
                    }
                }
                
            }
        }
        if (norm(RIFT_des) != 0){
            divide(RIFT_des, norm(RIFT_des), RIFT_des);
        }
        Mat dsttemp = des.row(k);
	    RIFT_des.copyTo(dsttemp);



    }

    return des;
}



static void help()
{
    cout << "\nThis program seek for letter bounding box on image\n"
        << "Call:\n"
        << "/.edge input_image_name [output_image_name]"
        << endl;
}

int main(int argc, char** argv)
{
    try
    {   
        int s = 4;
        int o = 6;


        const String inFileKey = "@inputFiles";
        const String keys =
            "{help h usage ?    |      | print this message }"
            "{" + inFileKey + " |<none>| input image        }";
        CommandLineParser parser(argc, argv, keys);
        if (parser.has("help") || !parser.has(inFileKey))
        {
            help();
            return 0;
        }
        const string inputFolderName = parser.get<String>(inFileKey);


        Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
        Mat image1 = imread(inputFolderName + "/pair1.jpg", IMREAD_GRAYSCALE);
        PhaseCongruency pc1(image1.size(), s, o);
        Mat edges1, corners1;
        vector<vector<Mat>> eo1;
        pc1.feature(image1, edges1, corners1);
        eo1 = pc1.eo;
        vector<KeyPoint> keypoints1;
        
        detector->detect(edges1,keypoints1,Mat()); // The higher the response, the stronger
        
        sort(keypoints1.begin(), keypoints1.end(), compare_keypoints); 
        keypoints1.erase(keypoints1.begin() + 5000, keypoints1.end());  // can set # of keypoints
        vector<KeyPoint> keypoints1_left = ignore_kps(edges1, keypoints1, 96);
        imwrite(inputFolderName + "/PC1.jpg", edges1);
        drawKeypoints(edges1, keypoints1_left, edges1, 255);
        imwrite(inputFolderName + "/feature1.jpg", edges1);    
        Mat image2 = imread(inputFolderName + "/pair2.jpg", IMREAD_GRAYSCALE);
        PhaseCongruency pc2(image2.size(), s, o);
        Mat edges2, corners2;
        vector<vector<Mat>> eo2;
        pc2.feature(image2, edges2, corners1);
        eo2 = pc2.eo;
        vector<KeyPoint> keypoints2;
        detector->detect(edges2,keypoints2,Mat()); // The higher the response, the stronger
        sort(keypoints2.begin(), keypoints2.end(), compare_keypoints); 
        keypoints2.erase(keypoints2.begin() + 5000, keypoints2.end());  // can set # of keypoints
        vector<KeyPoint> keypoints2_left = ignore_kps(edges2, keypoints2, 96);
        imwrite(inputFolderName + "/PC2.jpg", edges2);
        drawKeypoints(edges2, keypoints2_left, edges2, 255);
        imwrite(inputFolderName + "/feature2.jpg", edges2);

        Mat MIM1, MIM2, MIM1_show, MIM2_show;
        MIM1 = MIM(edges1, eo1, s, o);
        MIM2 = MIM(edges2, eo2, s, o);
        divide(MIM1, 6, MIM1_show);
        MIM1_show.convertTo(MIM1_show, CV_8U, 255);
        imwrite(inputFolderName + "/MIM1.jpg",MIM1_show);
        divide(MIM2, 6, MIM2_show);
        MIM2_show.convertTo(MIM2_show, CV_8U, 255);
        imwrite(inputFolderName + "/MIM2.jpg",MIM2_show);

        Mat descriptors1, descriptors2;
        descriptors1 = compute_des(MIM1, keypoints1_left, o, 96);
        descriptors2 = compute_des(MIM2, keypoints2_left, o, 96);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector<vector<DMatch> > knn_matches;
        descriptors1.convertTo(descriptors1, CV_32F); 
        descriptors2.convertTo(descriptors2, CV_32F);
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        
        const float ratio_thresh = 1;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //-- Draw matches
        Mat img_matches;
        Mat image1_show = imread(inputFolderName + "/pair1.jpg", IMREAD_COLOR);
        Mat image2_show = imread(inputFolderName + "/pair2.jpg", IMREAD_COLOR);
        drawMatches( image1_show, keypoints1_left, image2_show, keypoints2_left, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        
        imwrite(inputFolderName + "/match_no_ransac.jpg", img_matches);



        //根据matches将特征点对齐,将坐标转换为float类型
        vector<KeyPoint> R_keypoint01,R_keypoint02;
        for (size_t i=0;i<good_matches.size();i++)   
        {
            R_keypoint01.push_back(keypoints1_left[good_matches[i].queryIdx]);
            R_keypoint02.push_back(keypoints2_left[good_matches[i].trainIdx]);
        }
    
        //坐标转换
        vector<Point2f>p01,p02;
        for (size_t i=0;i<good_matches.size();i++)
        {
            p01.push_back(R_keypoint01[i].pt);
            p02.push_back(R_keypoint02[i].pt);
        }
    
        //利用基础矩阵剔除误匹配点
        vector<uchar> RansacStatus;
        Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);


    
        vector<KeyPoint> RR_keypoint01,RR_keypoint02;
        vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
        int index=0;
        for (size_t i=0;i<good_matches.size();i++)
        {
            if (RansacStatus[i]!=0)
            {
                RR_keypoint01.push_back(R_keypoint01[i]);
                RR_keypoint02.push_back(R_keypoint02[i]);
                good_matches[i].queryIdx=index;
                good_matches[i].trainIdx=index;
                RR_matches.push_back(good_matches[i]);
                index++;
            }
        }
        Mat img_RR_matches;
        drawMatches(image1_show,RR_keypoint01,image2_show,RR_keypoint02,RR_matches,img_RR_matches,Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imwrite(inputFolderName + "/match.jpg", img_RR_matches);




        

    }
    catch (Exception& e)
    {
        const char* err_msg = e.what();
        std::cout << "Exception caught: " << err_msg << std::endl;
    }
    waitKey();

    return 0;
}
