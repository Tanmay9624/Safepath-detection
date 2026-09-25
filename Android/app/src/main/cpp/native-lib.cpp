#include "jni.h"
#include <string>
#include <vector>
#include "android/log.h"
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

// Setup Android logging so we can see print statements in Logcat
#define LOG_TAG "SafePathEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


#include "thread"
#include "queue"
#include "mutex"
#include "condition_variable"
#include "chrono"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

// ---------------------------------------------------------
// PIPELINE DATA STRUCTURES & QUEUES
// ---------------------------------------------------------

struct FrameData {
    int frame_id;
    cv::Mat image;
};

struct InferenceResult {
    int frame_id = -1;
    cv::Mat deeplab_mask;              // 256x384 (Walkable path binary mask)
    cv::Mat depth_map;                 // 518x518 (Depth map from Depth Anything V2)
    std::vector<cv::Rect2f> yolo_boxes; // Normalized [0.0, 1.0] coordinates
    std::vector<int> yolo_classes;
};

template <typename T>
class BoundedQueue {
private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size = 2;

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (queue.size() >= max_size) queue.pop();
        queue.push(item);
        cv.notify_one();
    }
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !queue.empty(); });
        item = queue.front();
        queue.pop();
        return true;
    }
};

// Global queues for our three worker threads
BoundedQueue<FrameData> dl_in, yolo_in, depth_in;
BoundedQueue<InferenceResult> dl_out, yolo_out, depth_out;

// Global pointers for the ONNX Environment and our three models
Ort::Env* ortEnv = nullptr;
Ort::Session* yoloSession = nullptr;
Ort::Session* deepLabSession = nullptr;
Ort::Session* midasSession = nullptr;
// Add this line for the decision engine!
std::thread* t_decision = nullptr;
// Global decision string and a lock to prevent thread collisions
std::string global_nav_text = "BOOTING SYSTEM";
std::mutex nav_mtx;

// SIMD-Vectorized Pre-processing helper
std::vector<float> prepare_tensor(const cv::Mat& image, int w, int h, const float mean[3], const float std_dev[3]) {
    cv::Mat resized, float_img;
    cv::resize(image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat>channels(3);
    cv::split(float_img, channels);

    bool has_norm = (mean[0] != 0.0f || mean[1] != 0.0f || mean[2] != 0.0f ||
                     std_dev[0] != 1.0f || std_dev[1] != 1.0f || std_dev[2] != 1.0f);
    if (has_norm) {
        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - mean[c]) / std_dev[c];
        }
    }

    std::vector<float> tensor_vals(3 * h * w);
    int plane_size = h * w;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(tensor_vals.data() + c * plane_size, channels[c].data, plane_size * sizeof(float));
    }
    return tensor_vals;
}

// 1. DeepLabV3+ Worker (256x384)
void worker_deeplab() {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t> input_shape = {1, 3, 256, 384};
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    FrameData data;

    while (dl_in.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 384, 256, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};
        // Run inference on the phone's CPU using the global session
        auto ort_out = deepLabSession->Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        const float* out_arr = ort_out.front().GetTensorMutableData<float>();
        cv::Mat mask(256, 384, CV_8UC1);

        int total_pixels = 256 * 384;
        const float* p0 = out_arr;
        const float* p1 = out_arr + total_pixels;
        const float* p2 = out_arr + 2 * total_pixels;
        const float* p3 = out_arr + 3 * total_pixels;
        uchar* mask_ptr = mask.data;

        for (int i = 0; i < total_pixels; ++i) {
            float v0 = p0[i], v1 = p1[i], v2 = p2[i], v3 = p3[i];
            mask_ptr[i] = (v1 > v0 && v1 > v2 && v1 > v3) ? 255 : 0;
        }

        cv::Mat safe_mask;
        cv::erode(mask, safe_mask, erode_kernel);

        InferenceResult res;
        res.frame_id = data.frame_id;
        res.deeplab_mask = safe_mask;
        dl_out.push(res);
    }
}

// 2. YOLOv8 Worker (640x480)
void worker_yolo() {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const float mean[] = {0.0f, 0.0f, 0.0f};
    const float std_dev[] = {1.0f, 1.0f, 1.0f};
    std::vector<int64_t>input_shape = {1, 3, 480, 640};
    FrameData data;

    while (yolo_in.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 640, 480, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"images"};
        const char* out_names[] = {"output0"};
        auto ort_out = yoloSession->Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        float* out_arr = ort_out.front().GetTensorMutableData<float>();
        int num_anchors = 6300;
        int num_classes = 80;

        std::vector<cv::Rect> raw_boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < num_anchors; ++i) {
            float max_conf = 0.0f;
            int best_class = -1;
            for (int c = 0; c < num_classes; ++c) {
                float conf = out_arr[(4 + c) * num_anchors + i];
                if (conf > max_conf) { max_conf = conf; best_class = c; }
            }
            if (max_conf > 0.40f) {
                float cx = out_arr[0 * num_anchors + i] / 640.0f;
                float cy = out_arr[1 * num_anchors + i] / 480.0f;
                float w  = out_arr[2 * num_anchors + i] / 640.0f;
                float h  = out_arr[3 * num_anchors + i] / 480.0f;

                raw_boxes.push_back(cv::Rect(int((cx - w/2.0f)*1000), int((cy - h/2.0f)*1000), int(w*1000), int(h*1000)));
                confidences.push_back(max_conf);
                class_ids.push_back(best_class);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(raw_boxes, confidences, 0.40f, 0.45f, indices);

        InferenceResult res;
        res.frame_id = data.frame_id;
        for (int idx : indices) {
            auto b = raw_boxes[idx];
            res.yolo_boxes.push_back(cv::Rect2f(b.x / 1000.0f, b.y / 1000.0f, b.width / 1000.0f, b.height / 1000.0f));
            res.yolo_classes.push_back(class_ids[idx]);
        }
        yolo_out.push(res);
    }
}

// 3. Depth Anything V2 Worker (518x518)
void worker_depth_anything() {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t>input_shape = {1, 3, 518, 518};
    FrameData data;

    while (depth_in.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 518, 518, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"depth"};
        auto ort_out = midasSession->Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        float* out_arr = ort_out.front().GetTensorMutableData<float>();
        cv::Mat raw_depth(518, 518, CV_32FC1, out_arr);

        InferenceResult res;
        res.frame_id = data.frame_id;
        cv::normalize(raw_depth, res.depth_map, 0.0, 255.0, cv::NORM_MINMAX);
        depth_out.push(res);
    }
}

// 4. Assistive Navigation Decision Engine
void worker_decision() {
    InferenceResult dl_res, yolo_res, depth_res;
    InferenceResult cached_depth_res;

    while (true) {
        // Block and wait for the AI models to finish the current frame
        dl_out.pop(dl_res);
        yolo_out.pop(yolo_res);

        // Optimization: Depth Cadence Caching (Match the every-2nd-frame logic)
        if (dl_res.frame_id % 2 == 0) {
            depth_out.pop(cached_depth_res);
        }
        depth_res = cached_depth_res;

        // 1. Sector-Based Assistive Navigation Analysis (Left, Center, Right)
        cv::Mat lower_dl = dl_res.deeplab_mask(cv::Rect(0, 128, 384, 128));
        int left_walkable   = cv::countNonZero(lower_dl(cv::Rect(0, 0, 128, 128)));
        int center_walkable = cv::countNonZero(lower_dl(cv::Rect(128, 0, 128, 128)));
        int right_walkable  = cv::countNonZero(lower_dl(cv::Rect(256, 0, 128, 128)));

        int left_hazards = 0, center_hazards = 0, right_hazards = 0;
        bool center_near = false, left_near = false, right_near = false;

        for (size_t i = 0; i < yolo_res.yolo_boxes.size(); ++i) {
            cv::Rect2f norm_box = yolo_res.yolo_boxes[i];

            int dl_x = std::max(0, (int)(norm_box.x * 384));
            int dl_y = std::max(0, (int)(norm_box.y * 256));
            int dl_x2 = std::min(384, (int)((norm_box.x + norm_box.width) * 384));
            int dl_y2 = std::min(256, (int)((norm_box.y + norm_box.height) * 256));
            int dl_w = std::max(0, dl_x2 - dl_x);
            int dl_h = std::max(0, dl_y2 - dl_y);

            if (dl_w <= 0 || dl_h <= 0) continue;

            cv::Mat path_roi = dl_res.deeplab_mask(cv::Rect(dl_x, dl_y, dl_w, dl_h));
            double overlap = cv::countNonZero(path_roi) / (double)(dl_w * dl_h);

            int md_x = std::max(0, (int)(norm_box.x * 518));
            int md_y = std::max(0, (int)(norm_box.y * 518));
            int md_x2 = std::min(518, (int)((norm_box.x + norm_box.width) * 518));
            int md_y2 = std::min(518, (int)((norm_box.y + norm_box.height) * 518));
            int md_w = std::max(0, md_x2 - md_x);
            int md_h = std::max(0, md_y2 - md_y);

            bool is_near = false;
            float est_d = 3.5f;

            if (md_w > 0 && md_h > 0 && !depth_res.depth_map.empty()) {
                cv::Mat depth_roi = depth_res.depth_map(cv::Rect(md_x, md_y, md_w, md_h));
                cv::Scalar avg_d = cv::mean(depth_roi);
                double md = avg_d[0];
                est_d = (md > 0.0) ? std::max(0.5f, std::round(((255.0f - (float)md) / 255.0f * 4.5f + 0.5f) * 10.0f) / 10.0f) : 3.5f;
                is_near = (est_d <= 1.8f) || (md > 175.0);
            }

            float cx = norm_box.x + norm_box.width / 2.0f;
            if (is_near) {
                if (cx < 0.35f) left_near = true;
                else if (cx > 0.65f) right_near = true;
                else center_near = true;
            }

            if (overlap > 0.15) {
                if (cx < 0.35f) left_hazards++;
                else if (cx <= 0.65f) center_hazards++;
                else right_hazards++;
            }
        }

        // 2. Assistive Navigation Steering Decision
        bool c_blocked = center_near || (center_hazards > 0);
        bool l_walk = (left_walkable > 200) && !left_near;
        bool r_walk = (right_walkable > 200) && !right_near;

        std::string nav_text;
        if (c_blocked) {
            if (l_walk && !r_walk) nav_text = "HAZARD IN CENTER -> VEER LEFT";
            else if (r_walk && !l_walk) nav_text = "HAZARD IN CENTER -> VEER RIGHT";
            else if (l_walk && r_walk) nav_text = "HAZARD IN CENTER -> VEER RIGHT";
            else nav_text = "CROWD BLOCKED -> STOP / CAUTION";
        } else if (left_near) {
            nav_text = "HAZARD ON LEFT -> BIAS RIGHT";
        } else if (right_near) {
            nav_text = "HAZARD ON RIGHT -> BIAS LEFT";
        } else if (center_walkable > 300) {
            nav_text = "PATH CLEAR - PROCEED FORWARD";
        } else {
            nav_text = "SCANNING FOR WALKABLE PATH";
        }

        // Save the real-time decision to our global variable
        {
            std::lock_guard<std::mutex> lock(nav_mtx);
            global_nav_text = nav_text;
        }
    }
}

// Global thread pointers
std::thread* t_dl = nullptr;
std::thread* t_yolo = nullptr;
std::thread* t_depth = nullptr;

// We need a global frame counter for the Android camera
int global_frame_count = 0;



// Helper function to read .onnx files from the Android assets folder into memory
std::vector<char> readModelBytes(AAssetManager* assetManager, const char* filename) {
    AAsset* asset = AAssetManager_open(assetManager, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        LOGE("Failed to open model: %s", filename);
        return {};
    }
    
    off_t length = AAsset_getLength(asset);
    std::vector<char> buffer(length);
    AAsset_read(asset, buffer.data(), length);
    AAsset_close(asset);
    return buffer;
}

// JNI Bridge: This function is called from Kotlin to initialize the C++ engine
// Updated JNI Bridge: Kotlin passes the exact physical file paths to C++
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_initEngine(JNIEnv* env, jobject /* this */,
                                                       jstring yoloPath,
                                                       jstring deeplabPath,
                                                       jstring depthPath) {

    LOGI("Initializing SafePath ONNX Engine via file paths...");

    if (!ortEnv) {
        ortEnv = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SafePathEnv");
    }

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(2);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Convert Android jstrings to standard C++ characters
    const char* yolo_c = env->GetStringUTFChars(yoloPath, nullptr);
    const char* deeplab_c = env->GetStringUTFChars(deeplabPath, nullptr);
    const char* depth_c = env->GetStringUTFChars(depthPath, nullptr);

    try {
        // Load models directly from the physical storage paths
        yoloSession = new Ort::Session(*ortEnv, yolo_c, sessionOptions);
        deepLabSession = new Ort::Session(*ortEnv, deeplab_c, sessionOptions);
        midasSession = new Ort::Session(*ortEnv, depth_c, sessionOptions);
    } catch (const Ort::Exception& e) {
        LOGE("ONNX Error: %s", e.what());
        return JNI_FALSE;
    }

    // Free the string memory
    env->ReleaseStringUTFChars(yoloPath, yolo_c);
    env->ReleaseStringUTFChars(deeplabPath, deeplab_c);
    env->ReleaseStringUTFChars(depthPath, depth_c);

    LOGI("Engine Initialized Successfully!");

    // Start the background worker threads!
    if (!t_dl) t_dl = new std::thread(worker_deeplab);
    if (!t_yolo) t_yolo = new std::thread(worker_yolo);
    if (!t_depth) t_depth = new std::thread(worker_depth_anything);
    // Add this line!
    if (!t_decision) t_decision = new std::thread(worker_decision);
    return JNI_TRUE;




}

// JNI Bridge: Now returns the latest steering decision back to Kotlin
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_processFrame(JNIEnv* env, jobject /* this */,
                                                         jint width, jint height,
                                                         jbyteArray yuvPixels) {

    jbyte* pixels = env->GetByteArrayElements(yuvPixels, nullptr);
    cv::Mat frame(height, width, CV_8UC4, (unsigned char*)pixels);

    cv::Mat bgrFrame;
    cv::cvtColor(frame, bgrFrame, cv::COLOR_RGBA2BGR);

    cv::Mat small_frame;
    cv::resize(bgrFrame, small_frame, cv::Size(640, 360));

    FrameData current_frame = {global_frame_count++, small_frame};
    dl_in.push(current_frame);
    yolo_in.push(current_frame);

    if (current_frame.frame_id % 2 == 0) {
        depth_in.push(current_frame);
    }

    env->ReleaseByteArrayElements(yuvPixels, pixels, JNI_ABORT);

    // Safely grab the most recent decision from the worker thread
    std::string latest_decision;
    {
        std::lock_guard<std::mutex> lock(nav_mtx);
        latest_decision = global_nav_text;
    }

    // Return the text back to the Kotlin camera loop
    return env->NewStringUTF(latest_decision.c_str());
}

