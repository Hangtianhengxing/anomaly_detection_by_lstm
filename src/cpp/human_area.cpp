#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using std::cout;
using std::endl;
using std::string;

extern void display_info(string, int, int, int, int, double);
extern cv::Mat read_mask(string, bool);
extern void write_csv(std::vector<float>&, string);
extern void save_frame_num(std::vector<int>&, string);


float calc_area_ratio(cv::Mat &img, cv::Mat &bin_mask) {
    if (bin_mask.channels() != 1) {
        cout << "Error: mask is NOT binary!" << endl;
        exit(1);
    }

    // convert to gray scale
    cv::Mat gray_img = img.clone();
    if (gray_img.channels() != 1) {
        cv::cvtColor(gray_img, gray_img, CV_RGB2GRAY);
    }

    // binarizes the input image with the threshold value 150 and extracts only mask region
    cv::threshold(gray_img, gray_img, 120, 255, cv::THRESH_BINARY);

    int positive_num = 0; // number of human point
    int total_num = 0; // number of no masked area point
    for (int y=0;y<gray_img.rows;y++) {
        for (int x=0;x<gray_img.cols;x++) {
            int p1 = bin_mask.at<uchar>(y, x);
            if (p1==1) { //no mask area
                total_num++;
                int p2 = gray_img.at<uchar>(y, x);
                if (p2 == 0) { // point of human area
                    positive_num++;
                }
            }
        }
    }

    // 0.0 <= ratio <= 1.0
    // The larger value is ,the larger area of human is. 
    float ratio = float(positive_num)/total_num;

    return ratio;
}


void human_area(string input_file_path, string human_mask_path, string output_human_path, string output_frame_num_path) {
    cv::VideoCapture capture(input_file_path);
    int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int total_frame = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fourcc = (int)capture.get(CV_CAP_PROP_FOURCC);
    double fps = (double)capture.get(CV_CAP_PROP_FPS);

    // end if video can not be read
    if (!capture.isOpened()) {
        cout << "Error: can not open file." << endl;
        cout << "PATH: " << input_file_path << endl;
        exit(1);
    }

    // display information of input file
    display_info(input_file_path, width, height, total_frame, fourcc, fps);

    // initializetion
    cv::Mat frame, gray_frame;
    cv::Mat bin_human_mask = read_mask(human_mask_path, true);
    float area_ratio;
    std::vector<float> human_vec;
    human_vec.reserve(total_frame+10);
    int frame_num = 0;
    std::vector<int> frame_vec;
    frame_vec.reserve(total_frame+10);

    //skip untill first frame (initial frame is company logo)
    for (int i=0;i<4*30;i++) {
        capture >> frame;
        frame_num++;
    }

    while(true) {
        capture >> frame;
        if (frame.empty()) break;
        
        frame_num++;
        frame_vec.push_back(frame_num);
        if (frame_num%1000==0) {
            cout << "Frame number: " << frame_num << "/" << total_frame << endl;
        }

        cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
        area_ratio = calc_area_ratio(gray_frame, bin_human_mask);
        human_vec.push_back(area_ratio);

        // remove company logo
        if (frame_num >= total_frame-3*30){
            break;
        } 
    }
    cv::destroyAllWindows();

    write_csv(human_vec, output_human_path);
    save_frame_num(frame_vec, output_frame_num_path);
    cout << "Done: save human area data." << endl;

}
