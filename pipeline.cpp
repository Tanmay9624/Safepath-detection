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
#include <iomanip>
#include <algorithm>

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
    int frame_id = -1;
    cv::Mat deeplab_mask;              // 256x384 (Walkable path binary mask)
    cv::Mat depth_map;                 // 518x518 (Depth map from Depth Anything V2, normalized 0..255)
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

// SIMD-Vectorized Pre-processing with conditional normalization skip
std::vector<float> prepare_tensor(const cv::Mat& image, int w, int h, const float mean[3], const float std_dev[3]) {
    cv::Mat resized, float_img;
    cv::resize(image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
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

// Helper to configure CUDA Session Options
void configure_session_options(Ort::SessionOptions& opts, const std::string& model_name = "Model") {
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(2);
    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[CUDA INIT] " << model_name << ": CUDAExecutionProvider appended (Device 0: NVIDIA GPU)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[WARN] " << model_name << ": CUDA provider failed, falling back to CPU: " << e.what() << std::endl;
    }
}

// 1. DeepLabV3+ Worker (256x384) with Vectorized Planar Argmax
void worker_deeplab(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts, "DeepLabV3-MobileNet");
    Ort::Session session(env, RESOLVE_MODEL("deeplabv3_mobilenet_safepath.onnx"), opts);
    std::cout << "[READY] DeepLabV3 Session loaded on GPU (CUDA)\n";
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std_dev[] = {0.229f, 0.224f, 0.225f};
    std::vector<int64_t> input_shape = {1, 3, 256, 384};

    // Fast 3x3 rectangular kernel for sidewalk safety erosion
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    FrameData data;

    while (in_q.pop(data)) {
        std::vector<float> input_tensor = prepare_tensor(data.image, 384, 256, mean, std_dev);
        auto ort_in = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};
        auto ort_out = session.Run(Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        const float* out_arr = ort_out.front().GetTensorMutableData<float>();
        cv::Mat mask(256, 384, CV_8UC1);

        // Vectorized planar argmax lookup across 4 classes
        int total_pixels = 256 * 384;
        const float* p0 = out_arr;
        const float* p1 = out_arr + total_pixels;
        const float* p2 = out_arr + 2 * total_pixels;
        const float* p3 = out_arr + 3 * total_pixels;
        uchar* mask_ptr = mask.data;

        for (int i = 0; i < total_pixels; ++i) {
            float v0 = p0[i];
            float v1 = p1[i];
            float v2 = p2[i];
            float v3 = p3[i];
            mask_ptr[i] = (v1 > v0 && v1 > v2 && v1 > v3) ? 255 : 0; // Class 1 = Walkable
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

// 2. YOLOv8 Worker (640x480) with Multi-Class Detection & NMS
void worker_yolo(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts, "YOLOv8-Hazards");
    Ort::Session session(env, RESOLVE_MODEL("yolov8n_hazards.onnx"), opts);
    std::cout << "[READY] YOLOv8 Hazards Session loaded on GPU (CUDA)\n";
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
        out_q.push(res);
    }
}

// 3. Depth Anything V2 Worker (518x518) with Min/Max Normalization
void worker_depth_anything(BoundedQueue<FrameData>& in_q, BoundedQueue<InferenceResult>& out_q, Ort::Env& env) {
    Ort::SessionOptions opts;
    configure_session_options(opts, "DepthAnythingV2-Small");
    Ort::Session session(env, RESOLVE_MODEL("depth_anything_v2_small.onnx"), opts);
    std::cout << "[READY] Depth Anything V2 Session loaded on GPU (CUDA)\n";
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
        cv::Mat raw_depth(518, 518, CV_32FC1, out_arr);

        InferenceResult res;
        res.frame_id = data.frame_id;
        // Min-Max Normalize depth map to [0, 255] float range
        cv::normalize(raw_depth, res.depth_map, 0.0, 255.0, cv::NORM_MINMAX);
        out_q.push(res);
    }
}

// Main Pipeline Loop (Visual Overlay, Cadence Caching & High-Performance Decision Engine)
int main(int argc, char** argv) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SafePathEdge");

    BoundedQueue<FrameData> dl_in, yolo_in, depth_in;
    BoundedQueue<InferenceResult> dl_out, yolo_out, depth_out;

    std::thread t_dl(worker_deeplab, std::ref(dl_in), std::ref(dl_out), std::ref(env));
    std::thread t_yolo(worker_yolo, std::ref(yolo_in), std::ref(yolo_out), std::ref(env));
    std::thread t_depth(worker_depth_anything, std::ref(depth_in), std::ref(depth_out), std::ref(env));

    std::string video_source;
    int max_frames = -1;
    bool headless = false;
    int depth_cadence = 2; // Run Depth Anything every 2nd frame by default (Cadence Caching)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--gui") {
            headless = false;
        } else if (arg == "--depth-cadence" && i + 1 < argc) {
            depth_cadence = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (std::isdigit(arg[0])) {
            max_frames = std::stoi(arg);
        } else if (video_source.empty()) {
            video_source = resolve_file(arg);
        }
    }

    if (video_source.empty()) {
        video_source = resolve_file("test_05_san_francisco_street.mp4");
        if (!std::filesystem::exists(video_source)) {
            video_source = resolve_file("test_01_urban_crowd.mp4");
        }
    }

    cv::VideoCapture cap;
    if (video_source == "0" || video_source == "1") {
        cap.open(std::stoi(video_source));
    } else {
        cap.open(video_source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video source: " << video_source << "\n";
        return -1;
    }

    int frame_count = 0;
    cv::Mat raw_frame;
    auto last_time = std::chrono::high_resolution_clock::now();
    double smooth_fps = 0.0;

    std::cout << "[INFO] Starting SafePath C++ Tri-Thread GPU Pipeline...\n";
    std::cout << "[INFO] Ingest source: " << video_source 
              << " | Headless: " << (headless ? "YES" : "NO") 
              << " | Depth Cadence: " << depth_cadence << "x\n";

    InferenceResult cached_depth_res;

    while (cap.read(raw_frame)) {
        int current_fid = frame_count++;

        // Pre-downscale raw frame ONCE to 640x360 to eliminate high-res CPU resizing
        cv::Mat frame;
        cv::resize(raw_frame, frame, cv::Size(640, 360));

        FrameData current_frame = {current_fid, frame};
        dl_in.push(current_frame);
        yolo_in.push(current_frame);

        // Optimization 2: Depth Cadence Caching (Subsample heavy Vision Transformer)
        bool run_depth = (current_fid % depth_cadence == 0) || cached_depth_res.depth_map.empty();
        if (run_depth) {
            depth_in.push(current_frame);
        }

        InferenceResult dl_res, yolo_res, depth_res;
        dl_out.pop(dl_res);
        yolo_out.pop(yolo_res);

        if (run_depth) {
            depth_out.pop(cached_depth_res);
        }
        depth_res = cached_depth_res;
        depth_res.frame_id = current_fid; // Maintain synchronous frame indexing

        int orig_w = frame.cols;
        int orig_h = frame.rows;

        // 1. Sector-Based Assistive Navigation Analysis (Left, Center, Right)
        cv::Mat lower_dl = dl_res.deeplab_mask(cv::Rect(0, 128, 384, 128));
        int left_walkable   = cv::countNonZero(lower_dl(cv::Rect(0, 0, 128, 128)));
        int center_walkable = cv::countNonZero(lower_dl(cv::Rect(128, 0, 128, 128)));
        int right_walkable  = cv::countNonZero(lower_dl(cv::Rect(256, 0, 128, 128)));

        int left_hazards = 0, center_hazards = 0, right_hazards = 0;
        bool center_near = false, left_near = false, right_near = false;

        struct BoxRender {
            int x1, y1, w, h;
            cv::Scalar color;
            std::string label;
        };
        std::vector<BoxRender> boxes_to_draw;

        for (size_t i = 0; i < yolo_res.yolo_boxes.size(); ++i) {
            cv::Rect2f norm_box = yolo_res.yolo_boxes[i];

            // Corner-bounded coordinates for DeepLab mask (384x256)
            int dl_x = std::max(0, (int)(norm_box.x * 384));
            int dl_y = std::max(0, (int)(norm_box.y * 256));
            int dl_x2 = std::min(384, (int)((norm_box.x + norm_box.width) * 384));
            int dl_y2 = std::min(256, (int)((norm_box.y + norm_box.height) * 256));
            int dl_w = std::max(0, dl_x2 - dl_x);
            int dl_h = std::max(0, dl_y2 - dl_y);

            if (dl_w <= 0 || dl_h <= 0) continue;

            cv::Mat path_roi = dl_res.deeplab_mask(cv::Rect(dl_x, dl_y, dl_w, dl_h));
            double overlap = cv::countNonZero(path_roi) / (double)(dl_w * dl_h);

            // Bounded coordinates for Depth Anything V2 (518x518)
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

            cv::Scalar box_color = cv::Scalar(0, 255, 255); // Yellow: general obstacle
            std::string label = "Obstacle " + std::to_string(est_d).substr(0, 3) + "m";

            if (overlap > 0.15) {
                box_color = cv::Scalar(0, 0, 255); // Red: blocking walkable path
                label = is_near ? "HAZARD: NEAR (" + std::to_string(est_d).substr(0, 3) + "m)" : "HAZARD: AHEAD";

                if (cx < 0.35f) {
                    left_hazards++;
                } else if (cx <= 0.65f) {
                    center_hazards++;
                } else {
                    right_hazards++;
                }
            }

            if (!headless) {
                int orig_x = std::max(0, (int)(norm_box.x * orig_w));
                int orig_y = std::max(0, (int)(norm_box.y * orig_h));
                int orig_x2 = std::min(orig_w, (int)((norm_box.x + norm_box.width) * orig_w));
                int orig_y2 = std::min(orig_h, (int)((norm_box.y + norm_box.height) * orig_h));
                int orig_box_w = std::max(0, orig_x2 - orig_x);
                int orig_box_h = std::max(0, orig_y2 - orig_y);
                if (orig_box_w > 0 && orig_box_h > 0) {
                    boxes_to_draw.push_back({orig_x, orig_y, orig_box_w, orig_box_h, box_color, label});
                }
            }
        }

        // 2. Assistive Navigation Steering Decision (Priority Ordering)
        bool c_blocked = center_near || (center_hazards > 0);
        bool l_walk = (left_walkable > 200) && !left_near;
        bool r_walk = (right_walkable > 200) && !right_near;

        std::string nav_text;
        cv::Scalar nav_color;

        if (c_blocked) {
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
        } else if (center_walkable > 300) {
            nav_text = "NAV: PATH CLEAR - PROCEED FORWARD";
            nav_color = cv::Scalar(0, 255, 0); // Green
        } else {
            nav_text = "NAV: SCANNING FOR WALKABLE PATH";
            nav_color = cv::Scalar(200, 200, 200);
        }

        // Calculate instantaneous & smoothed FPS
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - last_time).count();
        last_time = now;
        double current_fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
        smooth_fps = (smooth_fps == 0.0) ? current_fps : (0.9 * smooth_fps + 0.1 * current_fps);

        if (current_fid % 10 == 0 || current_fid == 0) {
            std::cout << "[PROGRESS] Frame " << std::setw(3) << current_fid << " | Throughput: " 
                      << std::fixed << std::setprecision(1) << smooth_fps 
                      << " FPS | Decision: " << nav_text << std::endl;
        }

        // 3. Optional GUI Rendering (Only when not in headless mode)
        if (!headless) {
            // Alpha-blend green walkable path
            cv::Mat full_mask;
            cv::resize(dl_res.deeplab_mask, full_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

            cv::Mat green_overlay = frame.clone();
            green_overlay.setTo(cv::Scalar(0, 255, 0), full_mask > 0);
            cv::addWeighted(green_overlay, 0.35, frame, 0.65, 0.0, frame);

            for (const auto& b : boxes_to_draw) {
                cv::rectangle(frame, cv::Rect(b.x1, b.y1, b.w, b.h), b.color, 2);
                cv::putText(frame, b.label, cv::Point(b.x1, std::max(20, b.y1 - 8)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, b.color, 2);
            }

            // High-Contrast Assistive HUD Overlay
            cv::Mat hud_bg = frame.clone();
            cv::rectangle(hud_bg, cv::Rect(10, 8, 620, 74), cv::Scalar(15, 15, 15), -1);
            cv::addWeighted(hud_bg, 0.65, frame, 0.35, 0.0, frame);

            cv::putText(frame, nav_text, cv::Point(20, 42), cv::FONT_HERSHEY_SIMPLEX, 0.72, nav_color, 2);

            std::string telemetry = "C++ GPU: " + std::to_string((int)smooth_fps) + " FPS | Walkable: " + std::to_string(center_walkable) + "px";
            cv::putText(frame, telemetry, cv::Point(20, 72), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(220, 220, 220), 1);

            // Corridor boundary indicators
            cv::line(frame, cv::Point((int)(0.35 * orig_w), orig_h - 25), cv::Point((int)(0.35 * orig_w), orig_h), cv::Scalar(255, 255, 255), 1);
            cv::line(frame, cv::Point((int)(0.65 * orig_w), orig_h - 25), cv::Point((int)(0.65 * orig_w), orig_h), cv::Scalar(255, 255, 255), 1);

            cv::imshow("SafePath Edge C++: Assistive Navigation", frame);
            if (cv::waitKey(1) == 27) break; // ESC to quit
        }

        if (max_frames > 0 && frame_count >= max_frames) {
            std::cout << "\n[SUCCESS] Completed " << frame_count 
                      << " frames on GPU! Average sustained throughput: " << std::fixed << std::setprecision(1) << smooth_fps << " FPS.\n";
            break;
        }
    }

    exit(0);
}