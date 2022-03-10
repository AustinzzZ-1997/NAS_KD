#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Interpreter.hpp"
#include "Tensor.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
using namespace MNN;
using namespace cv;

int main(int argc,char *argv[])
{
   // 测试图像和mnn模型文件路径
   

    std::string image_name = argv[1];
    const char* model_name = argv[2];
    // printf("the pic is %S\n",image_name);
    // 一些任务调度中的配置参数
    int forward = MNN_FORWARD_CPU;
    // int forward = MNN_FORWARD_OPENCL;
    int precision  = 2;
    int power      = 0;
    int memory     = 0;
    int threads    = 1;
    int INPUT_SIZE = 32;

    cv::Mat raw_image    = cv::imread(image_name.c_str());
    //imshow("image", raw_image);
    int raw_image_height = raw_image.rows;
    int raw_image_width  = raw_image.cols;
    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    // 1. 创建Interpreter, 通过磁盘文件创建: static Interpreter* createFromFile(const char* file);
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(model_name));
    MNN::ScheduleConfig config;
    // 2. 调度配置,
    // numThread决定并发数的多少，但具体线程数和并发效率，不完全取决于numThread
    // 推理时，主选后端由type指定，默认为CPU。在主选后端不支持模型中的算子时，启用由backupType指定的备选后端。
    config.numThread = threads;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    // 3. 后端配置
    // memory、power、precision分别为内存、功耗和精度偏好
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory;
    config.backendConfig = &backendConfig;
    // 4. 创建session
    auto session = net->createSession(config);
    net->releaseModel();

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    image = image / 255.0f;
    // 5. 输入数据
    // wrapping input tensor, convert nhwc to nchw
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "x";
    // 获取输入tensor
    // 拷贝数据, 通过这类拷贝数据的方式，用户只需要关注自己创建的tensor的数据布局，
    // copyFromHostTensor会负责处理数据布局上的转换（如需）和后端间的数据拷贝（如需）。
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // 6. 运行会话
    net->runSession(session);

    // 7. 获取输出
    std::string output_tensor_name0 = "y";
    // 获取输出tensor
    MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    // 拷贝数据
    tensor_scores->copyToHostTensor(&tensor_scores_host);

    // post processing steps
    auto scores_dataPtr  = tensor_scores_host.host<float>();

    // softmax
    float exp_sum = 0.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val = scores_dataPtr[i];
        exp_sum += val;
    }
    // get result idx
    int  idx = 0;
    float max_prob = -10.0f;
    for (int i = 0; i < 10; ++i)
    {
        float val  = scores_dataPtr[i];
        float prob = val / exp_sum;
        if (prob > max_prob)
        {
            max_prob = prob;
            idx      = i;
        }
    }
    printf("the result is %d\n", idx);

    return 0;
}

