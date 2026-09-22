#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>

namespace fs = std::filesystem;

inline std::string resolve_file(const std::string& name) {
    if (fs::exists(name)) return name;
    if (fs::exists("../" + name)) return "../" + name;
    if (fs::exists("../../" + name)) return "../../" + name;
    return name;
}

#ifdef _WIN32
inline std::wstring resolve_model_path(const std::string& name) {
    std::string path = resolve_file(name);
    return std::wstring(path.begin(), path.end());
}
#define RESOLVE_MODEL(name) resolve_model_path(name).c_str()
#else
inline std::string resolve_model_path(const std::string& name) {
    return resolve_file(name);
}
#define RESOLVE_MODEL(name) resolve_model_path(name).c_str()
#endif

struct FrameData {
    int frame_id;
    cv::Mat image;
};

struct InferenceResult {
    int frame_id;
    cv::Mat deeplab_mask;              // 256x384 (Walkable path binary mask)
    cv::Mat depth_map;                 // 518x518 (Depth map from Depth Anything V2)
    std::vector<cv::Rect2f> yolo_boxes;// Normalized [0.0, 1.0] coordinates
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

// SIMD-Vectorized Pre-processing (Replaces slow nested pixel loops)
std::vector<float> prepare_tensor(const cv::Mat& image, int w, int h, const float mean[3], const float std_dev[3]) {
    cv::Mat resized, float_img;
    cv::resize(image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std_dev[c];
    }

    std::vector<float> tensor_vals(3 * h * w);
    int plane_size = h * w;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(tensor_vals.data() + c * plane_size, channels[c].data, plane_size * sizeof(float));
    }
    return tensor_vals;
}

// Helper to configure CUDA Session Options
void configure_session_options(Ort::SessionOptions& opts) {
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(2);
    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    } catch (const std::exception& e) {
        std::cerr << "[WARN] CUDA provider failed, falling back to CPU: " << e.what() << std::endl;
    }
}

// 1. DeepLabV3+ Worker (256x384)
void worker_deeplab(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts);
    Ort::Session session(env, RESOLVE_MODEL("deeplabv3_mobilenet_safepath.onnx"), opts);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t> input_shape = {1, 3, 256, 384};

    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    FrameData data;

    while (in_q.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 384, 256, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};
        auto ort_out = session.Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        float* out_arr = ort_out.front().GetTensorMutableData<float>();
        cv::Mat mask(256, 384, CV_8UC1);

        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 384; ++x) {
                int best_c = 0;
                float max_val = -1000.0f;
                for (int c = 0; c < 4; ++c) {
                    float val = out_arr[c * (256 * 384) + y * 384 + x];
                    if (val > max_val) { max_val = val; best_c = c; }
                }
                mask.at<uchar>(y, x) = (best_c == 1) ? 255 : 0; // Class 1 = Walkable
            }
        }

        // Apply physical safety margin (erosion)
        cv::Mat safe_mask;
        cv::erode(mask, safe_mask, erode_kernel);

        InferenceResult res;
        res.frame_id = data.frame_id;
        res.deeplab_mask = safe_mask;
        out_q.push(res);
    }
}

// 2. YOLOv8 Worker (640x480)
void worker_yolo(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts);
    Ort::Session session(env, RESOLVE_MODEL("yolov8n_hazards.onnx"), opts);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const float mean[] = {0.0f, 0.0f, 0.0f};
    const float std_dev[] = {1.0f, 1.0f, 1.0f};
    std::vector<int64_t> input_shape = {1, 3, 480, 640};

    FrameData data;
    while (in_q.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 640, 480, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"images"};
        const char* out_names[] = {"output0"};
        auto ort_out = session.Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

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
            if (max_conf > 0.45f) {
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
        cv::dnn::NMSBoxes(raw_boxes, confidences, 0.45f, 0.45f, indices);

        InferenceResult res;
        res.frame_id = data.frame_id;
        for (int idx : indices) {
            auto b = raw_boxes[idx];
            res.yolo_boxes.push_back(cv::Rect2f(b.x / 1000.0f, b.y / 1000.0f, b.width / 1000.0f, b.height / 1000.0f));
            res.yolo_classes.push_back(class_ids[idx]);
        }
        out_q.push(res);
    }
}

// 3. Depth Anything V2 Worker (518x518)
void worker_depth_anything(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts);
    Ort::Session session(env, RESOLVE_MODEL("depth_anything_v2_small.onnx"), opts);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t> input_shape = {1, 3, 518, 518};

    FrameData data;
    while (in_q.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 518, 518, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"depth"};
        auto ort_out = session.Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        float* out_arr = ort_out.front().GetTensorMutableData<float>();
        cv::Mat depth(518, 518, CV_32FC1, out_arr);

        InferenceResult res;
        res.frame_id = data.frame_id;
        depth.copyTo(res.depth_map);
        out_q.push(res);
    }
}

// Main Pipeline Loop (Visual Overlay & Console Diagnostics)
int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SafePathEdge");

    BoundedQueue<FrameData> dl_in, yolo_in, depth_in;
    BoundedQueue<InferenceResult> dl_out, yolo_out, depth_out;

    std::thread t_dl(worker_deeplab, std::ref(dl_in), std::ref(dl_out), std::ref(env));
    std::thread t_yolo(worker_yolo, std::ref(yolo_in), std::ref(yolo_out), std::ref(env));
    std::thread t_depth(worker_depth_anything, std::ref(depth_in), std::ref(depth_out), std::ref(env));

    std::string video_source = resolve_file("test_01_urban_crowd.mp4");
    cv::VideoCapture cap(video_source);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video source: " << video_source << "\n";
        return -1;
    }

    int frame_count = 0;
    cv::Mat raw_frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    double smooth_fps = 0.0;

    std::cout << "[INFO] Starting SafePath C++ Multi-Threaded GPU Pipeline...\n";

    while (cap.read(raw_frame)) {
        // Pre-downscale raw frame ONCE to 640x360 to eliminate high-res CPU overhead
        cv::Mat frame;
        cv::resize(raw_frame, frame, cv::Size(640, 360));

        FrameData current_frame = {frame_count++, frame};
        dl_in.push(current_frame);
        yolo_in.push(current_frame);
        depth_in.push(current_frame);

        InferenceResult dl_res, yolo_res, depth_res;
        dl_out.pop(dl_res);
        yolo_out.pop(yolo_res);
        depth_out.pop(depth_res);

        if (dl_res.frame_id == yolo_res.frame_id && dl_res.frame_id == depth_res.frame_id) {
            int orig_w = frame.cols;
            int orig_h = frame.rows;

            // 1. Overlay Walkable Path (Green)
            cv::Mat full_mask;
            cv::resize(dl_res.deeplab_mask, full_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

            cv::Mat green_overlay = frame.clone();
            green_overlay.setTo(cv::Scalar(0, 255, 0), full_mask > 0);
            cv::addWeighted(green_overlay, 0.35, frame, 0.65, 0.0, frame);

            bool hazard_on_path = false;

            // 2. Sector-Based Assistive Navigation Analysis (Left, Center, Right)
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
                int dl_w = std::min(384 - dl_x, (int)(norm_box.width * 384));
                int dl_h = std::min(256 - dl_y, (int)(norm_box.height * 256));

                if (dl_w <= 0 || dl_h <= 0) continue;

                cv::Mat path_roi = dl_res.deeplab_mask(cv::Rect(dl_x, dl_y, dl_w, dl_h));
                double overlap = cv::countNonZero(path_roi) / (double)(dl_w * dl_h);

                int orig_x = std::max(0, (int)(norm_box.x * orig_w));
                int orig_y = std::max(0, (int)(norm_box.y * orig_h));
                int orig_box_w = std::min(orig_w - orig_x, (int)(norm_box.width * orig_w));
                int orig_box_h = std::min(orig_h - orig_y, (int)(norm_box.height * orig_h));

                cv::Scalar box_color = cv::Scalar(0, 255, 255); // Yellow: general obstacle
                std::string label = "Obstacle";

                if (overlap > 0.15) {
                    hazard_on_path = true;
                    box_color = cv::Scalar(0, 0, 255); // Red: blocking walkable path
                    bool is_near = false;

                    // Check Depth Anything proximity (518x518)
                    int md_x = std::max(0, (int)(norm_box.x * 518));
                    int md_y = std::max(0, (int)(norm_box.y * 518));
                    int md_w = std::min(518 - md_x, (int)(norm_box.width * 518));
                    int md_h = std::min(518 - md_y, (int)(norm_box.height * 518));

                    cv::Mat depth_roi = depth_res.depth_map(cv::Rect(md_x, md_y, md_w, md_h));
                    double min_d, max_d;
                    cv::minMaxLoc(depth_roi, &min_d, &max_d);

                    if (max_d > 180.0) {
                        label = "HAZARD: NEAR (<2m)";
                        is_near = true;
                    } else {
                        label = "HAZARD: AHEAD";
                    }

                    // Sector assignment
                    float cx = norm_box.x + norm_box.width / 2.0f;
                    if (cx < 0.35f) {
                        left_hazards++;
                        if (is_near) left_near = true;
                    } else if (cx <= 0.65f) {
                        center_hazards++;
                        if (is_near) center_near = true;
                    } else {
                        right_hazards++;
                        if (is_near) right_near = true;
                    }
                }

                cv::rectangle(frame, cv::Rect(orig_x, orig_y, orig_box_w, orig_box_h), box_color, 2);
                cv::putText(frame, label, cv::Point(orig_x, std::max(20, orig_y - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2);
            }

            // 3. Assistive Navigation Steering Decision
            bool c_blocked = center_near || (center_hazards > 0);
            bool l_walk = (left_walkable > 200) && !left_near;
            bool r_walk = (right_walkable > 200) && !right_near;

            std::string nav_text;
            cv::Scalar nav_color;

            if (!c_blocked && center_walkable > 300) {
                nav_text = "NAV: PATH CLEAR - PROCEED FORWARD";
                nav_color = cv::Scalar(0, 255, 0); // Green
            } else if (c_blocked) {
                if (l_walk && !r_walk) {
                    nav_text = "NAV: HAZARD IN CENTER -> VEER LEFT";
                    nav_color = cv::Scalar(0, 255, 255); // Yellow
                } else if (r_walk && !l_walk) {
                    nav_text = "NAV: HAZARD IN CENTER -> VEER RIGHT";
                    nav_color = cv::Scalar(0, 255, 255); // Yellow
                } else if (l_walk && r_walk) {
                    nav_text = "NAV: HAZARD IN CENTER -> VEER RIGHT";
                    nav_color = cv::Scalar(0, 255, 255); // Yellow
                } else {
                    nav_text = "NAV: CROWD BLOCKED -> STOP / CAUTION";
                    nav_color = cv::Scalar(0, 0, 255); // Red
                }
            } else if (left_near) {
                nav_text = "NAV: HAZARD ON LEFT -> BIAS RIGHT";
                nav_color = cv::Scalar(0, 255, 255);
            } else if (right_near) {
                nav_text = "NAV: HAZARD ON RIGHT -> BIAS LEFT";
                nav_color = cv::Scalar(0, 255, 255);
            } else {
                nav_text = "NAV: SCANNING FOR WALKABLE PATH";
                nav_color = cv::Scalar(200, 200, 200);
            }

            // Calculate instantaneous FPS
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - last_time).count();
            last_time = now;
            double current_fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
            smooth_fps = (smooth_fps == 0.0) ? current_fps : (0.9 * smooth_fps + 0.1 * current_fps);

            // 4. High-Contrast Assistive HUD Overlay
            cv::Mat hud_bg = frame.clone();
            cv::rectangle(hud_bg, cv::Rect(10, 8, 620, 74), cv::Scalar(15, 15, 15), -1);
            cv::addWeighted(hud_bg, 0.65, frame, 0.35, 0.0, frame);

            cv::putText(frame, nav_text, cv::Point(20, 42), cv::FONT_HERSHEY_SIMPLEX, 0.72, nav_color, 2);

            std::string telemetry = "C++ GPU: " + std::to_string((int)smooth_fps) + " FPS | Walkable: " + std::to_string(center_walkable) + "px";
            cv::putText(frame, telemetry, cv::Point(20, 72), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220, 220, 220), 1);

            // Draw corridor boundary indicators at bottom
            cv::line(frame, cv::Point((int)(0.35 * orig_w), orig_h - 25), cv::Point((int)(0.35 * orig_w), orig_h), cv::Scalar(255, 255, 255), 1);
            cv::line(frame, cv::Point((int)(0.65 * orig_w), orig_h - 25), cv::Point((int)(0.65 * orig_w), orig_h), cv::Scalar(255, 255, 255), 1);
        }

        cv::imshow("SafePath Edge C++: Assistive Navigation", frame);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    exit(0);
}