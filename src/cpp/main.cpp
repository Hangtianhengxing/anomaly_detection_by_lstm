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
    string output_file_path;

    // receive processing file path from standard input
    cout << "Input movie file path: ";
    cin >> input_file_path;
    cout << "Human mask path: ";
    cin >> human_mask_path;
    cout << "Output file path: ";
    cin >> output_file_path;

    human_area(input_file_path, human_mask_path, output_file_path);
    
    // display calculation time
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast <std::chrono::seconds> (end - start).count();
    cout << "Calcuration time: " << elapsed << endl;

}