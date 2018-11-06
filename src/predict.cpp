//
// Created by wk on 18-11-1.
//

#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <iomanip>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <c_predict_api.h>

//int main(){
//    printf("test my project \n ");
//    return 0;
//}

const mx_float DEFAULT_MEAN = 117.0;

static std::string trim(const std::string& input) {
    auto not_space = [](int ch) {
        return !std::isspace(ch);
    };
    auto output = input;
    output.erase(output.begin(), std::find_if(output.begin(), output.end(), not_space));
    output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(), output.end());
    return output;
}

// Read file to buffer
class BufferFile {
public :
    std::string file_path_;
    std::size_t length_ = 0;
    std::unique_ptr<char[]> buffer_;

    explicit BufferFile(const std::string& file_path)
            : file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            return;
        }

        ifs.seekg(0, std::ios::end);
        length_ = static_cast<std::size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

        buffer_.reset(new char[length_]);
        ifs.read(buffer_.get(), length_);
        ifs.close();
    }

    std::size_t GetLength() {
        return length_;
    }

    char* GetBuffer() {
        return buffer_.get();
    }
};

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        auto data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
                mean_data++;
            }
            if (channels > 1) {
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(const std::string& synset_file) {
    std::ifstream fi(synset_file.c_str());

    if (!fi.is_open()) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while (fi >> synset) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cout<< " data size : "<<data.size()<<std::endl;
        std::cerr << "Result data and synset size do not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
//        std::cout << "Accuracy[" << i << "] = " << std::setprecision(8) << data[i] << std::endl;

        if (data[i] > best_accuracy) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }
//    best_idx -= 1; //下标从0开始，需要切换到1
    std::cout << "Best Result: " << trim(synset[best_idx]) << " (id=" << best_idx << ", " <<
              "accuracy=" << std::setprecision(8) << best_accuracy << ")" << std::endl;
}

void predict(PredictorHandle pred_hnd, const std::vector<mx_float> &image_data,
             NDListHandle nd_hnd, const std::string &synset_file, int i) {
    auto image_size = image_data.size();
    printf("image_size : %ld  \n",image_size);
    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint* shape = nullptr;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    std::size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

    // Release NDList
    if (nd_hnd) {
        MXNDListFree(nd_hnd);
    }

    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
    auto synset = LoadSynset(synset_file);

    // Print Output Data
    PrintOutputResult(data, synset);
}

int main(int argc, char* argv[]) {
//    if (argc < 2) {
//        std::cout << "No test image here." << std::endl
//                  << "Usage: ./predict apple.jpg [num_threads]" << std::endl;
//        return EXIT_FAILURE;
//    }

//    std::string test_file(argv[1]);
//    std::string test_file = "../model/1920px-Honeycrisp.jpg" ;
    std::string test_file = "../model/cat.jpg" ;
    int num_threads = 1;
//    if (argc == 3)
//        num_threads = std::atoi(argv[2]);

    // Models path for your model, you have to modify it
//    std::string json_file = "../model/Inception-BN-symbol.json";
//    std::string param_file = "../model/Inception-BN-0126.params";
//    std::string synset_file = "../model/synset.txt";
    std::string json_file = "../vocModel/my_ssd_512_mobilenet1.0_voc-symbol.json";
    std::string param_file = "../vocModel/my_ssd_512_mobilenet1.0_voc-0000.params";
    std::string synset_file = "../vocModel/label.txt";
    std::string nd_file = "../model/mean_224.nd";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward 表示网络的输入个数
    //input_key ： 对应网络的输入节点的名字,如果有两个节点input1, input2 = mx.sym.Variable('data1'), mx.sym.Variable('data2')，
    // 需要改成input_key[2] = {"data1", "data2"};
    const char* input_key[1] = { "data" };
    const char** input_keys = input_key;

    // Image size and channels
//    int width = 224;
//    int height = 224;
//    int channels = 3;
    // ssd  input size
    int width = 512;
    int height = 512;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    const mx_uint input_shape_data[4] = { 1,
                                          static_cast<mx_uint>(channels),
                                          static_cast<mx_uint>(height),
                                          static_cast<mx_uint>(width) };

    if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
        return EXIT_FAILURE;
    }

    auto image_size = static_cast<std::size_t>(width * height * channels);
    printf("---333---\n");
    // Read Mean Data
    const mx_float* nd_data = nullptr;
    NDListHandle nd_hnd = nullptr;
    BufferFile nd_buf(nd_file);

    if (nd_buf.GetLength() > 0) {
        mx_uint nd_index = 0;
        mx_uint nd_len;
        const mx_uint* nd_shape = nullptr;
        const char* nd_key = nullptr;
        mx_uint nd_ndim = 0;

        MXNDListCreate(static_cast<const char*>(nd_buf.GetBuffer()),
                       static_cast<int>(nd_buf.GetLength()),
                       &nd_hnd, &nd_len);

        MXNDListGet(nd_hnd, nd_index, &nd_key, &nd_data, &nd_shape, &nd_ndim);
    }
    printf("---222---\n");
    // Read Image Data
    std::vector<mx_float> image_data(image_size);

    GetImageFile(test_file, image_data.data(), channels, cv::Size(width, height), nd_data);

    if (num_threads == 1) {
        // Create Predictor
        //这个是与MXNet交互的界面，通过这个handle我们可以设置要输入到网络的数据，可以获取相应的输出
        PredictorHandle pred_hnd;
        MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
                     static_cast<const char*>(param_data.GetBuffer()),
                     static_cast<int>(param_data.GetLength()),
                     dev_type,
                     dev_id,
                     num_input_nodes,
                     input_keys,
                     input_shape_indptr,
                     input_shape_data,
                     &pred_hnd);
        assert(pred_hnd);
        printf("---111---\n");
        predict(pred_hnd, image_data, nd_hnd, synset_file, 0);
    } else {
        // Create Predictor
        std::vector<PredictorHandle> pred_hnds(num_threads, nullptr);
        MXPredCreateMultiThread(static_cast<const char*>(json_data.GetBuffer()),
                                static_cast<const char*>(param_data.GetBuffer()),
                                static_cast<int>(param_data.GetLength()),
                                dev_type,
                                dev_id,
                                num_input_nodes,
                                input_keys,
                                input_shape_indptr,
                                input_shape_data,
                                pred_hnds.size(),
                                pred_hnds.data());
        for (auto hnd : pred_hnds)
            assert(hnd);

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; i++)
            threads.emplace_back(predict, pred_hnds[i], image_data, nd_hnd, synset_file, i);
        for (int i = 0; i < num_threads; i++)
            threads[i].join();
    }
    printf("run successfully\n");

    return EXIT_SUCCESS;
}