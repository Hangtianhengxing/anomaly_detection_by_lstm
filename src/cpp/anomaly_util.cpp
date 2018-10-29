#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <ctype.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

using std::cout;
using std::endl;
using std::string;

void display_info(string input_file_path, int width, int height, int total_frame, int fourcc, double fps){
    // display information of input file
    cout << "\n*******************************************" << endl;
    cout << "MOVIE PATH: " << input_file_path << endl;
    cout << "WIDTH: " << width << endl;
    cout << "HEIGHT: " << height << endl;
    cout << "TOTAL FRAME: " << total_frame << endl;
    cout << "FOURCC: " << fourcc << endl;
    cout << "FPS: " << fps << endl;
    cout << "*******************************************\n" << endl;
}

cv::Mat read_mask(string mask_path, bool binarization=false) {
    cv::Mat mask = cv::imread(mask_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (mask.empty()) {
        cout << "Error: can not open file." << endl;
        cout << "PATH: " << mask_path << endl;
        exit(1);
    }

    if (binarization) {
        if (mask.channels() != 1){
            cv::cvtColor(mask, mask, CV_BGR2GRAY);
        }
        cv::threshold(mask, mask, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
    }

    return mask;
}

void read_csv(string input_csv_file_path, std::vector<std::vector <int> > &table, const char delimiter = ',') {
    /**
     * to make vector table of csv data
     * 
     * input:
     *      input_csv_file_path:
     *      table:
     *      delimeter:
    **/
    std::fstream filestream(input_csv_file_path);
    if (!filestream.is_open()) {
        cout << "ERROR: can not open file. please check file path." << endl;
        cout << "PATH: " << input_csv_file_path << endl;
        exit(1);
    }

    while (!filestream.eof()) {
        string buffer;
        filestream >> buffer;

        std::vector<int> record;
        std::istringstream streambuffer(buffer);
        string token;
        while (getline(streambuffer, token, delimiter)) {
            record.push_back(atoi(token.c_str()));
        }
        if (!record.empty())
            table.push_back(record);
    }
}


void write_csv(std::vector<float> &vec_data, string output_csv_file_path) {
    /**
     * write csv file by vector data
     * 
     * input:
     *   vec_data: 
     *   output_csv_file_path: absolute path
    **/

    std::ofstream ofs(output_csv_file_path);
    if (ofs) {
        for (unsigned int i = 0; i < vec_data.size(); ++i) {
            ofs << vec_data[i] << endl;
        }
    }
    else {
        cout << "ERROR: can not open file. please check file path." << endl;
        cout << "PATH: " << output_csv_file_path << endl;
        exit(1);
    }

    ofs.close();
    cout << "SAVE PATH: " << output_csv_file_path << endl;
}


void save_frame_num(std::vector<int> &vec_data, string output_csv_file_path) {
    /**
     * write csv file by vector data
     * 
     * input:
     *   vec_data: 
     *   output_csv_file_path: absolute path
    **/

    std::ofstream ofs(output_csv_file_path);
    if (ofs) {
        for (unsigned int i = 0; i < vec_data.size(); ++i) {
            ofs << vec_data[i] << endl;
        }
    }
    else {
        cout << "ERROR: can not open file. please check file path." << endl;
        cout << "PATH: " << output_csv_file_path << endl;
        exit(1);
    }

    ofs.close();
    cout << "SAVE PATH: " << output_csv_file_path << endl;
}