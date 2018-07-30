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

extern cv::Mat read_mask(std::string mask_path, bool binary);
extern void write_csv(std::vector<float> &vec_data, std::string output_csv_file_path);


float calc_area_ratio(cv::Mat &img, cv::Mat &bin_mask) {
    if (bin_mask.channels() != 1) {
        std::cout << "Error: mask is NOT binary!" << std::endl;
        exit(1);
    }

    // convert to gray scale
    cv::Mat gray_img = img.clone();
    if (gray_img.channels() != 1) {
        cv::cvtColor(gray_img, gray_img, CV_RGB2GRAY);
    }

    // binarizes the input image with the threshold value 150 and extracts only mask region
    cv::threshold(gray_img, gray_img, 150, 255, cv::THRESH_BINARY);

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


void human_area(std::string input_file_path, std::string human_mask_path, std::string output_file_path) {
    cv::VideoCapture capture(input_file_path);
    int width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int total_frame = (int)capture.get(CV_CAP_PROP_FRAME_COUNT);
    int fourcc = (int)capture.get(CV_CAP_PROP_FOURCC);
    double fps = (double)capture.get(CV_CAP_PROP_FPS);

    // end if video can not be read
    if (!capture.isOpened()) {
        std::cout << "Error: can not open file." << std::endl;
        std::cout << "PATH: " << input_file_path << std::endl;
        exit(1);
    }

    // display information of input file
    std::cout << "\n*******************************************" << std::endl;
    std::cout << "MOVIE PATH: " << input_file_path << std::endl;
    std::cout << "WIDTH: " << width << std::endl;
    std::cout << "HEIGHT: " << height << std::endl;
    std::cout << "TOTAL FRAME: " << total_frame << std::endl;
    std::cout << "FOURCC: " << fourcc << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "*******************************************\n" << std::endl;

    // initializetion
    cv::Mat frame, gray_frame;
    cv::Mat bin_human_mask = read_mask(human_mask_path, true);
    float area_ratio;
    std::vector<float> human_vec;
    human_vec.reserve(total_frame+10);
    int frame_num = 0;


    while(true) {
        capture >> frame;
        if (frame.empty()) break;
        
        frame_num++;
        if (frame_num%1000==0) {
            std::cout << "Frame number: " << frame_num << "/" << total_frame << std::endl;
        }

        cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
        area_ratio = calc_area_ratio(gray_frame, bin_human_mask);
        human_vec.push_back(area_ratio); 
    }
    cv::destroyAllWindows();

    write_csv(human_vec, output_file_path);
    std::cout << "Done: save human area data." << std::endl;

}
