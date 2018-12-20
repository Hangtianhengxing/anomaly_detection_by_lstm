#include <iostream>
#include <chrono>
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
using std::cin;
using std::endl;
using std::string;

extern void human_area(string, string, string);

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    string input_file_path;
    string human_mask_path;
    string output_human_path;
    string output_frame_num_path;

    // receive processing file path from standard input
    // cout << "Input video file path: ";
    // cin >> input_file_path;
    // cout << "Human mask path: ";
    // cin >> human_mask_path;
    // cout << "Output file path: ";
    // cin >> output_file_path;

    string input_root_dirc;
    string output_root_dirc;
    //input_root_dirc = "/Users/sakka/cnn_anomaly_detection/video/20170422/";
    input_root_dirc = "/Users/sakka/cnn_anomaly_detection/video/20181030/";
    human_mask_path = "/Users/sakka/cnn_anomaly_detection/image/human_mask.png";
    output_root_dirc = "/Users/sakka/cnn_anomaly_detection/data/human_area/20181030/";


    input_file_path = input_root_dirc + "201810300900.mp4";
    output_human_path = output_root_dirc + "human_9.csv";
    human_area(input_file_path, human_mask_path, output_human_path);

    for (int i=10;i<17;i++){
        input_file_path = input_root_dirc + "20181030" + std::to_string(i) + "00.mp4";
        output_human_path = output_root_dirc + "human_" + std::to_string(i) + ".csv";
        human_area(input_file_path, human_mask_path, output_human_path);
    }

    // display calculation time
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast <std::chrono::seconds> (end - start).count();
    cout << "Calcuration time: " << elapsed << endl;
}
