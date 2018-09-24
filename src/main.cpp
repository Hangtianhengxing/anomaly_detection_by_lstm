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

extern void human_area(std::string, std::string, std::string);

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    std::string input_file_path;
    std::string human_mask_path;
    std::string output_file_path;

    // receive processing file path from standard input
    std::cout << "Input movie file path: ";
    std::cin >> input_file_path;
    std::cout << "Human mask path: ";
    std::cin >> human_mask_path;
    std::cout << "Output file path: ";
    std::cin >> output_file_path;

    human_area(input_file_path, human_mask_path, output_file_path);
    
    // display calculation time
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast <std::chrono::seconds> (end - start).count();
    std::cout << "Calcuration time: " << elapsed << std::endl;

}