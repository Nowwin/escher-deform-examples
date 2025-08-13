#include <SDL.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// ===================== Window / Layout Constants =====================
const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const int HORIZONTAL_SPACING = -49;
const int VERTICAL_SPACING = -1023;
const int SPACING_X = -900 - 238;
const int SPACING_Y = -3472;

int g_unit_width = 0;
int g_unit_height = 0;

// ===================== Global State =====================
bool is_dragging = false;
bool drag_updated = false;
cv::Point drag_start, drag_end;
cv::Point picked_point;
double sigma = 10.0;
SDL_Point drag_screen_start;
SDL_Point drag_screen_end;
cv::Point last_dragged_point;

// ===================== Time / Filesystem Helpers =====================
static std::string now_timestamp() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static bool ensure_dir(const fs::path& p) {
    std::error_code ec;
    if (fs::exists(p, ec)) return fs::is_directory(p, ec);
    return fs::create_directories(p, ec);
}

// Save PNG (lossless). If offx/offy != 0, pad with transparent pixels and
// place the image at (offx, offy) inside the new canvas.
static bool save_png_lossless(const cv::Mat& img, const std::string& path,
                              int offx = 0, int offy = 0)
{
    std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 3 };

    // Fast path: no offset
    if (offx == 0 && offy == 0) {
        return cv::imwrite(path, img, params);
    }

    // Ensure 4-channel so padding is transparent
    cv::Mat src;
    if (img.channels() == 4) {
        src = img;
    } else if (img.channels() == 3) {
        cv::cvtColor(img, src, cv::COLOR_BGR2BGRA);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, src, cv::COLOR_GRAY2BGRA);
    } else {
        src = img.clone();
    }

    // Compute padding for positive/negative offsets
    int left   = std::max(offx, 0);
    int top    = std::max(offy, 0);
    int right  = std::max(-offx, 0);
    int bottom = std::max(-offy, 0);

    cv::Mat canvas(src.rows + top + bottom, src.cols + left + right,
                   src.type(), cv::Scalar(0, 0, 0, 0)); // transparent BG

    src.copyTo(canvas(cv::Rect(left, top, src.cols, src.rows)));
    return cv::imwrite(path, canvas, params);
}


struct DeformationCache {
    cv::Point last_picked_point;
    cv::Point last_drag_vector;
    double last_sigma;
    std::vector<cv::Mat> last_cover;

    cv::Mat field_x;
    cv::Mat field_y;

    bool isValid(const cv::Point& p, const cv::Point& v, double s, const std::vector<cv::Mat>& covers) const {
        if (last_picked_point != p || last_drag_vector != v || last_sigma != s || covers.size() != last_cover.size()) return false;
        for (size_t i = 0; i < covers.size(); ++i) {
            if (cv::countNonZero(covers[i] != last_cover[i]) > 0) return false;
        }
        return true;
    }

    void store(const cv::Point& p, const cv::Point& v, double s, const std::vector<cv::Mat>& covers,
               const cv::Mat& fx, const cv::Mat& fy) {
        last_picked_point = p;
        last_drag_vector = v;
        last_sigma = s;
        last_cover = covers;
        field_x = fx.clone();
        field_y = fy.clone();
    }
};

DeformationCache g_cache;

// ===================== Vector Field / Deformation =====================
cv::Mat applyVectorField(const cv::Mat& image, const cv::Mat& field_x, const cv::Mat& field_y) {
    int height = image.rows;
    int width = image.cols;

    cv::Mat map_x(height, width, CV_32FC1);
    cv::Mat map_y(height, width, CV_32FC1);
    cv::Mat out_of_bounds = cv::Mat::zeros(height, width, CV_8UC1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = static_cast<float>(field_x.at<double>(y, x));
            float dy = static_cast<float>(field_y.at<double>(y, x));

            float src_x = x - dx;
            float src_y = y - dy;

            map_x.at<float>(y, x) = src_x;
            map_y.at<float>(y, x) = src_y;

            if (src_x < 0 || src_x >= width || src_y < 0 || src_y >= height) {
                out_of_bounds.at<uchar>(y, x) = 255;
            }
        }
    }

    cv::Mat warped;
    cv::remap(image, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));

    if (image.channels() == 4) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (out_of_bounds.at<uchar>(y, x)) {
                    cv::Vec4b& px = warped.at<cv::Vec4b>(y, x);
                    px = cv::Vec4b(0, 0, 0, 0);
                }
            }
        }
    }

    return warped;
}

void saveKernelToCSV(const cv::Mat& kernel, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < kernel.rows; ++i) {
        for (int j = 0; j < kernel.cols; ++j) {
            file << kernel.at<double>(i, j);
            if (j < kernel.cols - 1) file << ",";
        }
        file << "\n";
    }
}

cv::Mat tileInfluenceToMatchSize(const cv::Mat& small_kernel, int target_rows, int target_cols) {
    int kernel_rows = small_kernel.rows;
    int kernel_cols = small_kernel.cols;

    int repeat_y = static_cast<int>(std::ceil(static_cast<double>(target_rows) / kernel_rows));
    int repeat_x = static_cast<int>(std::ceil(static_cast<double>(target_cols) / kernel_cols));

    cv::Mat tiled;
    cv::repeat(small_kernel, repeat_y, repeat_x, tiled);

    return tiled(cv::Rect(0, 0, target_cols, target_rows));
}

cv::Mat getPointsNotUniform(int color_region_size_x, int color_region_size_y, int num_points_x, int num_points_y) {
    std::vector<double> x_values(num_points_x);
    std::vector<double> y_values(num_points_y);

    for (int i = 0; i < num_points_x; ++i)
        x_values[i] = static_cast<double>(i) * color_region_size_x / static_cast<double>(num_points_x - 1);
    for (int j = 0; j < num_points_y; ++j)
        y_values[j] = static_cast<double>(j) * color_region_size_y / static_cast<double>(num_points_y - 1);

    cv::Mat points(num_points_x * num_points_y, 2, CV_64F);

    for (int iy = 0; iy < num_points_y; ++iy) {
        for (int ix = 0; ix < num_points_x; ++ix) {
            int idx = iy * num_points_x + ix;
            points.at<double>(idx, 0) = x_values[ix];
            points.at<double>(idx, 1) = y_values[iy];
        }
    }

    return points;
}

cv::Mat closeFormSolutionUnitSquare(const cv::Mat& points, double sigma = 10.0) {
    CV_Assert(points.cols == 2 && points.type() == CV_64F);

    int N = points.rows;
    cv::Mat colors(N, 1, CV_64F);

    double denom = std::pow(std::pow(2.0, sigma) - 1.0, 2.0);

    for (int i = 0; i < N; ++i) {
        double x = points.at<double>(i, 0);
        double y = points.at<double>(i, 1);

        double tmp1 = sigma * (x + y);
        double tmp2 = sigma * (y - x);

        double numer = std::pow(2.0, tmp1)
                     + std::pow(2.0, 2.0 * sigma - tmp1)
                     + std::pow(2.0, sigma - tmp2)
                     + std::pow(2.0, sigma + tmp2);

        colors.at<double>(i, 0) = numer / denom;
    }

    return colors;
}

cv::Mat reshapeFlatTo2D(const cv::Mat& flat_kernel, int Lx, int Ly) {
    CV_Assert(flat_kernel.rows == Lx * Ly && flat_kernel.cols == 1);

    cv::Mat reshaped_kernel(Ly, Lx, CV_64F);
    for (int i = 0; i < flat_kernel.rows; ++i) {
        int iy = i / Lx;
        int ix = i % Lx;
        reshaped_kernel.at<double>(iy, ix) = flat_kernel.at<double>(i, 0);
    }

    return reshaped_kernel;
}

void computeVectorField(
    cv::Mat& field_x, cv::Mat& field_y,
    const std::vector<cv::Mat>& toric_cover_lst_local,
    const cv::Point& picked_point,
    const cv::Point& drag_vector,
    int Lx, int Ly, int width, int height,
    double sigma
) {
    cv::Mat points = getPointsNotUniform(Lx, Ly, Lx, Ly);
    field_x.setTo(0);
    field_y.setTo(0);

    for (size_t idx = 0; idx < toric_cover_lst_local.size(); ++idx) {
        const auto& matrix = toric_cover_lst_local[idx];
        cv::Mat pt = (cv::Mat_<double>(3, 1) << picked_point.x, picked_point.y, 1);
        cv::Mat transformed_pt = matrix * pt;
        int px = static_cast<int>(transformed_pt.at<double>(0));
        int py = static_cast<int>(transformed_pt.at<double>(1));

        cv::Mat points_local(points.rows, 2, CV_64F);
        for (int i = 0; i < points.rows; ++i) {
            double x = points.at<double>(i, 0);
            double y = points.at<double>(i, 1);
            double local_x = std::fmod(std::abs(x + (Lx - px)), Lx) / static_cast<double>(Lx);
            double local_y = std::fmod(std::abs(y + (Ly - py)), Ly) / static_cast<double>(Ly);
            points_local.at<double>(i, 0) = local_x;
            points_local.at<double>(i, 1) = local_y;
        }

        cv::Mat colors_vector = closeFormSolutionUnitSquare(points_local, sigma);
        cv::Mat rolled_kernel = reshapeFlatTo2D(colors_vector, Lx, Ly);
        cv::Mat influence = tileInfluenceToMatchSize(rolled_kernel, height, width);

        cv::Mat vec = (cv::Mat_<double>(3,1) << drag_vector.x, drag_vector.y, 0);
        cv::Mat transformed_vec = matrix * vec;
        double vx = transformed_vec.at<double>(0);
        double vy = transformed_vec.at<double>(1);

        field_x += influence * vx;
        field_y += influence * vy;
    }
}

cv::Mat deformUnitTile(
    const cv::Mat& unit,
    const std::vector<cv::Mat>& toric_cover_lst_local,
    const cv::Point& picked_point,
    const cv::Point& drag_vector,
    double sigma = 10.0
) {
    int height = unit.rows;
    int width = unit.cols;
    int Lx = g_unit_width + SPACING_X;
    int Ly = g_unit_height + SPACING_Y;

    if (!g_cache.isValid(picked_point, drag_vector, sigma, toric_cover_lst_local)) {
        cv::Mat fx = cv::Mat::zeros(height, width, CV_64F);
        cv::Mat fy = cv::Mat::zeros(height, width, CV_64F);

        computeVectorField(fx, fy, toric_cover_lst_local, picked_point, drag_vector, Lx, Ly, width, height, sigma);
        g_cache.store(picked_point, drag_vector, sigma, toric_cover_lst_local, fx, fy);
    }

    return applyVectorField(unit, g_cache.field_x, g_cache.field_y);
}

// ===================== Compositing =====================
cv::Mat blendPatternWithOverlay(const cv::Mat& base_pattern, const cv::Mat& overlay) {
    CV_Assert(base_pattern.size() == overlay.size());
    CV_Assert(base_pattern.type() == CV_8UC4 && overlay.type() == CV_8UC4);

    cv::Mat result;
    base_pattern.copyTo(result);

    std::vector<cv::Mat> overlay_channels;
    cv::split(overlay, overlay_channels);
    cv::Mat alpha_mask = overlay_channels[3];

    overlay.copyTo(result, alpha_mask);

    return result;
}

// ===================== Unit Construction / Tiling =====================
cv::Mat create_base_unit(
    const cv::Mat& tile1, const cv::Mat& tile2,
    std::vector<cv::Mat>& toric_cover_lst,
    std::vector<cv::Mat>& toric_cover_lst_local,
    bool show_unit = false
) {
    int padding = 200;
    int max_width = std::max(tile1.cols, tile2.cols);
    int base_height = tile1.rows + tile2.rows + std::abs(VERTICAL_SPACING);
    int total_width = max_width + std::abs(HORIZONTAL_SPACING) + 2 * padding;
    int total_height = base_height + 2 * padding;

    g_unit_width = total_width;
    g_unit_height = total_height;

    cv::Mat unit(total_height, total_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    int center_x = total_width / 2;
    int center_y = total_height / 2;

    // BLUE (tile1)
    int blue_translate_x = center_x - tile1.cols / 2;
    int blue_translate_y = center_y - tile1.rows;
    cv::Mat blue_affine = (cv::Mat_<double>(2, 3) << 1, 0, -blue_translate_x, 0, 1, -blue_translate_y);
    cv::Mat blue_transformed;
    cv::Mat inv_blue_affine;
    cv::invertAffineTransform(blue_affine, inv_blue_affine);
    cv::warpAffine(tile1, blue_transformed, inv_blue_affine, unit.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    std::vector<cv::Mat> blue_channels;
    cv::split(blue_transformed, blue_channels);
    cv::Mat blue_mask;
    cv::threshold(blue_channels[3], blue_mask, 150, 255, cv::THRESH_BINARY);
    blue_transformed.copyTo(unit, blue_mask);

    // RED (tile2) flipped + translated
    cv::Mat red_reflect = (cv::Mat_<double>(2, 3) << -1, 0, tile2.cols, 0, 1, 0);
    cv::Mat red_flipped;
    cv::warpAffine(tile2, red_flipped, red_reflect, tile2.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    int red_translate_x = center_x - tile2.cols / 2 + HORIZONTAL_SPACING;
    int red_translate_y = center_y + VERTICAL_SPACING;
    cv::Mat red_affine = (cv::Mat_<double>(2, 3) << 1, 0, -red_translate_x, 0, 1, -red_translate_y);
    cv::Mat red_transformed;
    cv::Mat inv_red_affine;
    cv::invertAffineTransform(red_affine, inv_red_affine);
    cv::warpAffine(red_flipped, red_transformed, inv_red_affine, unit.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    std::vector<cv::Mat> red_channels;
    cv::split(red_transformed, red_channels);
    cv::Mat red_mask;
    cv::threshold(red_channels[3], red_mask, 150, 255, cv::THRESH_BINARY);
    red_transformed.copyTo(unit, red_mask);

    // Save affine transforms
    toric_cover_lst.resize(2);
    toric_cover_lst_local.resize(2);

    // Global transforms
    toric_cover_lst[0] = cv::Mat::eye(3, 3, CV_64F);
    toric_cover_lst[0].at<double>(0, 2) = blue_translate_x;
    toric_cover_lst[0].at<double>(1, 2) = blue_translate_y;

    toric_cover_lst[1] = (cv::Mat_<double>(3, 3) << -1, 0, tile2.cols + red_translate_x, 0, 1, red_translate_y, 0, 0, 1);

    // Local transforms
    toric_cover_lst_local[0] = cv::Mat::eye(3, 3, CV_64F);
    toric_cover_lst_local[1] = (cv::Mat_<double>(3, 3) << -1, 0, tile1.cols + HORIZONTAL_SPACING, 0, 1, tile1.rows + VERTICAL_SPACING, 0, 0, 1);

    return unit;
}

cv::Mat generate_tiled_pattern(const cv::Mat& unit,
                               double scale = 2.0,
                               int repeats_x = 10,
                               int repeats_y = 11,
                               bool crop_and_resize = true)                // <— new flag (default true)
{
    int crop_width  = WINDOW_WIDTH;
    int crop_height = WINDOW_HEIGHT;

    int unit_width  = unit.cols;
    int unit_height = unit.rows;

    int stride_x = unit_width  + SPACING_X;
    int stride_y = unit_height + SPACING_Y;

    int offset_x = 2000;
    int offset_y = 4000;

    int total_width  = stride_x * repeats_x + offset_x;
    int total_height = stride_y * repeats_y + offset_y;

    cv::Mat pattern(total_height, total_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // Use unit's alpha as mask if present
    cv::Mat alpha;
    if (unit.channels() == 4) {
        std::vector<cv::Mat> ch;
        cv::split(unit, ch);
        alpha = ch[3];
    }

    for (int y = 0; y < repeats_y; ++y) {
        for (int x = 0; x < repeats_x; ++x) {
            int pos_x = x * stride_x;
            int pos_y = y * stride_y;

            if (pos_x < 0 || pos_y < 0 ||
                pos_x + unit_width > pattern.cols ||
                pos_y + unit_height > pattern.rows) {
                continue;
            }

            cv::Rect roi(pos_x, pos_y, unit_width, unit_height);
            if (!alpha.empty()) {
                unit.copyTo(pattern(roi), alpha);
            } else {
                unit.copyTo(pattern(roi));
            }
        }
    }

    // If the caller wants the raw, full pattern, return it now.
    if (!crop_and_resize) {
        return pattern;
    }

    // Otherwise: resize to a target width and center-crop to the window.
    cv::Mat resized_pattern;
    int target_width  = static_cast<int>(std::clamp(scale, 1.0, 10.0) * WINDOW_WIDTH);
    int target_height = static_cast<int>(target_width * (static_cast<float>(pattern.rows) / pattern.cols));
    cv::resize(pattern, resized_pattern, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

    int crop_x = std::max(0, (resized_pattern.cols - crop_width) / 2);
    int crop_y = std::max(0, (resized_pattern.rows - crop_height) / 2);
    crop_x = std::min(crop_x, resized_pattern.cols - crop_width);
    crop_y = std::min(crop_y, resized_pattern.rows - crop_height);

    cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
    return resized_pattern(crop_rect).clone();
}

// ===================== SDL Helpers =====================
SDL_Texture* matToTexture(const cv::Mat& mat, SDL_Renderer* renderer) {
    cv::Mat converted;
    if (mat.channels() == 3)
        cv::cvtColor(mat, converted, cv::COLOR_BGR2RGBA);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, converted, cv::COLOR_BGRA2RGBA);
    else
        cv::cvtColor(mat, converted, cv::COLOR_GRAY2RGBA);

    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
        (void*)converted.data, converted.cols, converted.rows, 32, converted.step,
        0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000
    );
    if (!surface) return nullptr;

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    return texture;
}

inline int computeDefaultRepeats(int window_dim, int unit_dim, int spacing) {
    int stride = unit_dim + spacing;
    if (stride <= 0) stride = unit_dim;
    return (window_dim + stride - 1) / stride;
}

// ===================== Main =====================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " blue_monster.png red_monster.png\n";
        return 1;
    }

    std::string blue_path = argv[1];
    std::string red_path = argv[2];

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "Monster Pattern (Tiling)",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI
    );
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_RenderSetLogicalSize(renderer, WINDOW_WIDTH, WINDOW_HEIGHT);
    int render_w, render_h;
    SDL_GetRendererOutputSize(renderer, &render_w, &render_h);
    float dpi_scale_x = static_cast<float>(render_w) / WINDOW_WIDTH;
    float dpi_scale_y = static_cast<float>(render_h) / WINDOW_HEIGHT;

    cv::Mat base_blue = cv::imread(blue_path, cv::IMREAD_UNCHANGED);
    cv::Mat base_red = cv::imread(red_path, cv::IMREAD_UNCHANGED);
    cv::Mat blue = base_blue.clone();
    cv::Mat red = base_red.clone();

    if (blue.empty() || red.empty()) {
        std::cerr << "Failed to load one or both images.\n";
        return 1;
    }

    double scale = 2.0;

    std::vector<cv::Mat> toric_cover_lst, toric_cover_lst_local;
    cv::Mat unit = create_base_unit(blue, red, toric_cover_lst, toric_cover_lst_local);
    cv::Mat pattern = generate_tiled_pattern(unit, scale);

    cv::Mat overlay_img;
    if (argc == 4) {
        overlay_img = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
        if (overlay_img.empty()) {
            std::cerr << "Warning: Could not load overlay image: " << argv[3] << "\n";
            overlay_img = cv::Mat::zeros(pattern.size(), CV_8UC4);
        } else {
            cv::resize(overlay_img, overlay_img, pattern.size());
        }
    } else {
        overlay_img = cv::Mat::zeros(pattern.size(), CV_8UC4);
    }

    cv::Mat blended = blendPatternWithOverlay(pattern, overlay_img);

    SDL_Texture* texture_current = matToTexture(blended, renderer);
    if (!texture_current) {
        std::cerr << "[Pattern] Texture conversion failed.\n";
        return 1;
    }

    SDL_Texture* red_texture_current = matToTexture(red, renderer);
    if (!red_texture_current) {
        std::cerr << "[Tile] Texture conversion failed.\n";
        return 1;
    }

    int mini_w = 300;
    int mini_h = base_red.rows * mini_w / base_red.cols;
    int mini_x = 20, mini_y = WINDOW_HEIGHT - mini_h - 20;

    SDL_Rect preview_rect = { mini_x, mini_y, mini_w, mini_h };

    bool running = true;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                int mx = event.button.x;
                int my = event.button.y;

                if (mx >= preview_rect.x && mx < preview_rect.x + preview_rect.w &&
                    my >= preview_rect.y && my < preview_rect.y + preview_rect.h) {
                    drag_screen_start = { mx, my };
                    is_dragging = true;

                    float rel_x = float(mx - preview_rect.x) / float(preview_rect.w);
                    float rel_y = float(my - preview_rect.y) / float(preview_rect.h);

                    rel_x = std::clamp(rel_x, 0.0f, 1.0f);
                    rel_y = std::clamp(rel_y, 0.0f, 1.0f);

                    picked_point.x = int(rel_x * (base_blue.cols - 1));
                    picked_point.y = int(rel_y * (base_blue.rows - 1));

                    is_dragging = true;
                    last_dragged_point = picked_point;
                }
            }
            else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                is_dragging = false;

                int mx = event.button.x;
                int my = event.button.y;

                cv::Point drag_vector(
                    mx - drag_screen_start.x,
                    my - drag_screen_start.y
                );

                last_dragged_point = picked_point + drag_vector;

                red = deformUnitTile(base_red, toric_cover_lst_local, picked_point, drag_vector, sigma);
                blue = deformUnitTile(base_blue, toric_cover_lst_local, picked_point, drag_vector, sigma);

                base_red = red.clone();
                base_blue = blue.clone();

                unit = create_base_unit(blue, red, toric_cover_lst, toric_cover_lst_local);
                pattern = generate_tiled_pattern(unit, scale);

                cv::Mat resized_overlay;
                if (!overlay_img.empty() && overlay_img.size() != pattern.size()) {
                    cv::resize(overlay_img, resized_overlay, pattern.size());
                } else {
                    resized_overlay = overlay_img;
                }
                cv::Mat blended2 = blendPatternWithOverlay(pattern, resized_overlay);

                SDL_DestroyTexture(texture_current);
                texture_current = matToTexture(blended2, renderer);

                SDL_DestroyTexture(red_texture_current);
                red_texture_current = matToTexture(red, renderer);
            }
            else if (event.type == SDL_MOUSEMOTION && is_dragging) {
                int mx = event.button.x;
                int my = event.button.y;

                cv::Point drag_vector(
                    mx - drag_screen_start.x,
                    my - drag_screen_start.y
                );

                last_dragged_point = picked_point + drag_vector;

                cv::Mat red_preview = deformUnitTile(base_red, toric_cover_lst_local, picked_point, drag_vector, sigma);
                SDL_DestroyTexture(red_texture_current);
                red_texture_current = matToTexture(red_preview, renderer);
            }

            if (event.type == SDL_MOUSEWHEEL) {
                if (event.wheel.y > 0) scale *= 1.1;
                else if (event.wheel.y < 0) scale /= 1.1;

                scale = std::clamp(scale, 1.0, 10.0);

                pattern = generate_tiled_pattern(unit, scale);

                cv::Mat resized_overlay;
                if (!overlay_img.empty() && overlay_img.size() != pattern.size()) {
                    cv::resize(overlay_img, resized_overlay, pattern.size());
                } else {
                    resized_overlay = overlay_img;
                }
                cv::Mat blended2 = blendPatternWithOverlay(pattern, resized_overlay);

                SDL_DestroyTexture(texture_current);
                texture_current = matToTexture(blended2, renderer);
            }

            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_s) {
                    fs::path outdir = "exports";
                    if (!ensure_dir(outdir)) {
                        std::cerr << "[Save] Could not ensure exports directory.\n";
                    } else {
                        std::string ts = now_timestamp();

                        fs::path blue_path = outdir / ("Anteater1_deformed_" + ts + ".png");
                        fs::path red_path  = outdir / ("Anteater2_deformed_"  + ts + ".png");
                        fs::path pattern_path = outdir / ("pattern_anteater_deformed_" + ts + ".png");

                        cv::Mat red_flipped;
                        cv::flip(red, red_flipped, 1);

                        const int minus_w = 100, minus_h = 100;
                        int w = std::max(1, pattern.cols - minus_w);
                        int h = std::max(1, pattern.rows - minus_h);
                        cv::Mat pattern_cropped = pattern(cv::Rect(0, 0, w, h)).clone();

                        bool ok1 = save_png_lossless(blue, blue_path.string());
                        bool ok2 = save_png_lossless(red_flipped, red_path.string());
                        bool ok3 = save_png_lossless(pattern_cropped, pattern_path.string());

                        if (ok1 && ok2 && ok3) {
                            std::cout << "[Save] Completed!:\n";
                        } else {
                            std::cerr << "[Save] Failed to save one or both or pattern.\n";
                        }
                    }
                }
                else if (event.key.keysym.sym == SDLK_EQUALS || event.key.keysym.sym == SDLK_PLUS) {
                    scale *= 1.1;
                    scale = std::clamp(scale, 1.0, 10.0);
                    pattern = generate_tiled_pattern(unit, scale);

                    cv::Mat resized_overlay;
                    if (!overlay_img.empty() && overlay_img.size() != pattern.size()) {
                        cv::resize(overlay_img, resized_overlay, pattern.size());
                    } else {
                        resized_overlay = overlay_img;
                    }
                    cv::Mat blended2 = blendPatternWithOverlay(pattern, resized_overlay);

                    SDL_DestroyTexture(texture_current);
                    texture_current = matToTexture(blended2, renderer);
                }
                else if (event.key.keysym.sym == SDLK_MINUS || event.key.keysym.sym == SDLK_UNDERSCORE) {
                    scale /= 1.1;
                    scale = std::clamp(scale, 1.0, 10.0);
                    pattern = generate_tiled_pattern(unit, scale);

                    cv::Mat resized_overlay;
                    if (!overlay_img.empty() && overlay_img.size() != pattern.size()) {
                        cv::resize(overlay_img, resized_overlay, pattern.size());
                    } else {
                        resized_overlay = overlay_img;
                    }
                    cv::Mat blended2 = blendPatternWithOverlay(pattern, resized_overlay);

                    SDL_DestroyTexture(texture_current);
                    texture_current = matToTexture(blended2, renderer);
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 25, 30, 40, 255);
        SDL_RenderClear(renderer);

        SDL_Rect dst = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, texture_current, nullptr, &dst);

        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 190);
        SDL_RenderFillRect(renderer, &preview_rect);

        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
        SDL_RenderCopy(renderer, red_texture_current, nullptr, &preview_rect);

        float rel_x = std::clamp(last_dragged_point.x / float(base_red.cols - 1), 0.0f, 1.0f);
        float rel_y = std::clamp(last_dragged_point.y / float(base_red.rows - 1), 0.0f, 1.0f);

        int dot_x = preview_rect.x + int(rel_x * preview_rect.w);
        int dot_y = preview_rect.y + int(rel_y * preview_rect.h);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        int r_outer = 8;
        for (int dy = -r_outer; dy <= r_outer; ++dy) {
            for (int dx = -r_outer; dx <= r_outer; ++dx) {
                if (dx*dx + dy*dy <= r_outer*r_outer) {
                    SDL_RenderDrawPoint(renderer, dot_x + dx, dot_y + dy);
                }
            }
        }

        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
        int r_inner = r_outer - 2;
        for (int dy = -r_inner; dy <= r_inner; ++dy) {
            for (int dx = -r_inner; dx <= r_inner; ++dx) {
                if (dx*dx + dy*dy <= r_inner*r_inner) {
                    SDL_RenderDrawPoint(renderer, dot_x + dx, dot_y + dy);
                }
            }
        }
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture_current);
    SDL_DestroyTexture(red_texture_current);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
