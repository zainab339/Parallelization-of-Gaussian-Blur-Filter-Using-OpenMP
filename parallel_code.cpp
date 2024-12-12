
// Copyright (C) 2017 Basile Fraboni
// Copyright (C) 2014 Ivan Kutskir


// Include necessary libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string> 
#include "/usr/local/opt/libomp/include/omp.h" 
#include <vector>

void std_to_box(int boxes[], float sigma, int n)
{
    float wi = std::sqrt((12 * sigma * sigma / n) + 1);
    int wl = std::floor(wi);
    if (wl % 2 == 0)
        wl--;
    int wu = wl + 2;

    float mi = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
    int m = std::round(mi);

    for (int i = 0; i < n; i++)
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;
}

void horizontal_blur(int *in, int *out, int w, int h, int r)
{
    float iarr = 1.f / (r + r + 1);
    // int chunk_size = 10;
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        int ti = i * w, li = ti, ri = ti + r, fv = in[ti], lv = in[ti + w - 1], val = (r + 1) * fv;
        for (int j = 0; j < r; j++)
            val += in[ti + j];
        for (int j = 0; j <= r; j++)
        {
            val += in[ri++] - fv;
            out[ti++] = std::round(val * iarr);
        }
        for (int j = r + 1; j < w - r; j++)
        {
            val += in[ri++] - in[li++];
            out[ti++] = std::round(val * iarr);
        }
        for (int j = w - r; j < w; j++)
        {
            val += lv - in[li++];
            out[ti++] = std::round(val * iarr);
        }
    }
}

void total_blur(int *in, int *out, int w, int h, int r)
{
    float iarr = 1.f / (r + r + 1);
    // int chunk_size = 10;
#pragma omp parallel for
    for (int i = 0; i < w; i++)
    {
        int ti = i, li = ti, ri = ti + r * w, fv = in[ti], lv = in[ti + w * (h - 1)], val = (r + 1) * fv;
        for (int j = 0; j < r; j++)
            val += in[ti + j * w];
        for (int j = 0; j <= r; j++)
        {
            val += in[ri] - fv;
            out[ti] = std::round(val * iarr);
            ri += w;
            ti += w;
        }
        for (int j = r + 1; j < h - r; j++)
        {
            val += in[ri] - in[li];
            out[ti] = std::round(val * iarr);
            li += w;
            ri += w;
            ti += w;
        }
        for (int j = h - r; j < h; j++)
        {
            val += lv - in[li];
            out[ti] = std::round(val * iarr);
            li += w;
            ti += w;
        }
    }
}

void box_blur(int *&in, int *&out, int w, int h, int r)
{
    std::swap(in, out);
    horizontal_blur(out, in, w, h, r);
    total_blur(in, out, w, h, r);
}

void fast_gaussian_blur(int *&in, int *&out, int w, int h, float sigma)
{
    int boxes[3];
    std_to_box(boxes, sigma, 3);
    box_blur(in, out, w, h, boxes[0]);
    box_blur(out, in, w, h, boxes[1]);
    box_blur(in, out, w, h, boxes[2]);
}

int main(int argc, char *argv[])
{
    omp_set_num_threads(1);
    // Measure the entire program execution
    auto total_start = std::chrono::system_clock::now();

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_list_file> [sigma] [output_directory]" << std::endl;
        return 1;
    }

    const std::string list_file = argv[1];
    const float sigma = argc > 2 ? std::atof(argv[2]) : 3.0f;
    const std::string output_directory = argc > 3 ? argv[3] : "output";

    std::ifstream file(list_file);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file: " << list_file << std::endl;
        return 1;
    }

    std::string image_file;
    int image_count = 0;
    float total_elapsed = 0.0f, total_io = 0.0f, total_preprocessing = 0.0f, total_save = 0.0f;

    while (std::getline(file, image_file))
    {
        if (image_file.empty())
            continue;

        // Measure the loading image time
        auto io_start = std::chrono::system_clock::now();
        int width, height, channels;
        unsigned char *image_data = stbi_load(image_file.c_str(), &width, &height, &channels, 0);
        auto io_end = std::chrono::system_clock::now();

        if (!image_data)
        {
            std::cerr << "Error loading image: " << image_file << std::endl;
            continue;
        }

        float io_time = std::chrono::duration_cast<std::chrono::milliseconds>(io_end - io_start).count();
        total_io += io_time;
        std::cout << "Image loading time: " << io_time << " ms" << std::endl;

        if (channels < 3)
        {
            std::cerr << "Image must be an RGB image (3 channels). Skipping " << image_file << std::endl;
            stbi_image_free(image_data);
            continue;
        }
        // Measure image preprocessing
        auto preprocess_start = std::chrono::system_clock::now();
        int size = width * height;
        int *newb = new int[size];
        int *newg = new int[size];
        int *newr = new int[size];
        int *oldb = new int[size];
        int *oldg = new int[size];
        int *oldr = new int[size];

        for (int i = 0; i < size; ++i)
        {
            oldb[i] = image_data[channels * i + 0];
            oldg[i] = image_data[channels * i + 1];
            oldr[i] = image_data[channels * i + 2];
        }
        auto preprocess_end = std::chrono::system_clock::now();

        float preprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - preprocess_start).count();
        total_preprocessing += preprocess_time;
        std::cout << "Image preprocessing time: " << preprocess_time << " ms" << std::endl;

        // Measure the image processing time
        auto process_start = std::chrono::system_clock::now();
        fast_gaussian_blur(oldb, newb, width, height, sigma);
        fast_gaussian_blur(oldg, newg, width, height, sigma);
        fast_gaussian_blur(oldr, newr, width, height, sigma);
        auto process_end = std::chrono::system_clock::now();

        float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start).count();
        total_elapsed += elapsed;
        std::cout << "Processing time: " << elapsed << " ms" << std::endl;

        for (int i = 0; i < size; ++i)
        {
            image_data[channels * i + 0] = static_cast<unsigned char>(std::min(255, std::max(0, newb[i])));
            image_data[channels * i + 1] = static_cast<unsigned char>(std::min(255, std::max(0, newg[i])));
            image_data[channels * i + 2] = static_cast<unsigned char>(std::min(255, std::max(0, newr[i])));
        }

#pragma omp parallel
        {
#pragma omp single nowait
            {
                auto save_start = std::chrono::system_clock::now();
                std::string filename = image_file.substr(image_file.find_last_of("/\\") + 1);
                std::string name_without_extension = filename.substr(0, filename.find_last_of('.'));
                std::string output_file = output_directory + "/" + name_without_extension + "_blur.png";
                stbi_write_png(output_file.c_str(), width, height, channels, image_data, width * channels);
                auto save_end = std::chrono::system_clock::now();
                float save_time = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start).count();
#pragma omp atiomic
                total_save += save_time;
#pragma omp critical
                std::cout << "Image saving time: " << save_time << " ms" << std::endl;
            }
        }

        stbi_image_free(image_data);
        delete[] newr;
        delete[] newb;
        delete[] newg;
        delete[] oldr;
        delete[] oldb;
        delete[] oldg;

        image_count++;
    }

    auto total_end = std::chrono::system_clock::now();
    float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\nSummary:\n";
    std::cout << "Processed " << image_count << " images.\n";
    std::cout << "Total program execution time: " << total_time << " ms.\n";
    // std::cout << "Total I/O time: " << total_io << " ms.\n";
    std::cout << "Total load time: " << total_io << " ms.\n";
    std::cout << "Total save time: " << total_save << " ms.\n";
    std::cout << "Total preprocessing time: " << total_preprocessing << " ms.\n";
    std::cout << "Total processing (Gaussian blur) time: " << total_elapsed << " ms.\n";

    return 0;
}
