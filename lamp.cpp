#include <SDL.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <cmath>
#include <omp.h>  // OpenMP
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// ==============================
// Utilities
// ==============================

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

static bool save_png_lossless(const cv::Mat& img, const std::string& path) {
    std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
    return cv::imwrite(path, img, params);
}

// ==============================
// Window / spacing constants
// ==============================

const int WINDOW_WIDTH  = 1200;
const int WINDOW_HEIGHT = 800;
const int HORIZONTAL_SPACING = -48;
const int VERTICAL_SPACING   = -228;

int g_unit_width  = 0;
int g_unit_height = 0;

// ==============================
// Global interaction state
// ==============================

bool is_dragging = false;
bool drag_updated = false;
cv::Point drag_start, drag_end;
cv::Point picked_point;
double sigma = 10.0;
SDL_Point drag_screen_start;
SDL_Point drag_screen_end;
cv::Point last_dragged_point;

int bravais_lattice_x = 911;
int bravais_lattice_y = 911;
int q2_spacing_x = 453;
int q2_spacing_y = -967;
int q3_spacing_x = 483;
int q3_spacing_y = -429;
int q4_spacing_x = 29;
int q4_spacing_y = -463;

// ==============================
// Deformation cache
// ==============================

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

// ==============================
// Math / geometry helpers
// ==============================

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

cv::Mat getBasePoints(int Lx, int Ly) {
    static cv::Mat cached_points;
    static int cached_Lx = -1, cached_Ly = -1;
    if (cached_Lx == Lx && cached_Ly == Ly && !cached_points.empty()) return cached_points;
    cached_points = getPointsNotUniform(Lx, Ly, Lx, Ly);
    cached_Lx = Lx; cached_Ly = Ly;
    return cached_points;
}

inline double positive_mod(double v, double m) {
    double r = std::fmod(v, m);
    return r < 0 ? r + m : r;
}

// Mirror configuration copied from Python defaults
const cv::Point2d mirror_location(516.0, 516.0);
const cv::Point2d mirror_direction_raw(1.0, 1.0);
const cv::Point2d mirror_direction =
    mirror_direction_raw / std::sqrt(mirror_direction_raw.x * mirror_direction_raw.x +
                                     mirror_direction_raw.y * mirror_direction_raw.y);

cv::Point2d mirror_direction_vector(const cv::Point2d& direction, bool flip_y = false) {
    cv::Point2d mirror_dir_norm(1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0));
    double dot = direction.x * mirror_dir_norm.x + direction.y * mirror_dir_norm.y;
    cv::Point2d projection(dot * mirror_dir_norm.x, dot * mirror_dir_norm.y);
    cv::Point2d perpendicular = direction - projection;
    cv::Point2d mirrored = direction - 2.0 * perpendicular;
    if (flip_y) mirrored.y = -mirrored.y;
    return cv::Point2d((int)mirrored.x, (int)mirrored.y);
}

cv::Point2d mirror_point(const cv::Point2d& point, bool flip_y = false) {
    cv::Point2d vector = point - mirror_location;
    double dot = vector.x * mirror_direction.x + vector.y * mirror_direction.y;
    cv::Point2d projection = dot * mirror_direction;
    cv::Point2d perpendicular(vector.x - projection.x, vector.y - projection.y);
    cv::Point2d mirrored_vector(vector.x - 2 * perpendicular.x, vector.y - 2 * perpendicular.y);
    cv::Point2d mirrored_point = mirror_location + mirrored_vector;
    if (flip_y) mirrored_point.y = -mirrored_point.y;
    return mirrored_point;
}

// ==============================
// CSV/debug helpers
// ==============================

void saveVectorField(const cv::Mat& fx, const cv::Mat& fy, const std::string& prefix) {
    std::ofstream file_x(prefix + "_x.csv");
    std::ofstream file_y(prefix + "_y.csv");
    for (int i = 0; i < fx.rows; ++i) {
        for (int j = 0; j < fx.cols; ++j) {
            file_x << fx.at<double>(i, j);
            file_y << fy.at<double>(i, j);
            if (j < fx.cols - 1) { file_x << ","; file_y << ","; }
        }
        file_x << "\n"; file_y << "\n";
    }
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

// ==============================
// Image transforms
// ==============================

cv::Mat rotateImage(const cv::Mat& src, double angle) {
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    rot_mat.at<double>(0,2) += bbox.width / 2.0 - src.cols / 2.0;
    rot_mat.at<double>(1,2) += bbox.height / 2.0 - src.rows / 2.0;
    cv::Mat dst;
    cv::warpAffine(src, dst, rot_mat, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0,0));
    return dst;
}

cv::Mat applyVectorField(const cv::Mat& image, const cv::Mat& field_x, const cv::Mat& field_y) {
    int height = image.rows, width = image.cols;
    cv::Mat map_x(height, width, CV_32FC1), map_y(height, width, CV_32FC1);
    cv::Mat out_of_bounds = cv::Mat::zeros(height, width, CV_8UC1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = static_cast<float>(field_x.at<double>(y, x));
            float dy = static_cast<float>(field_y.at<double>(y, x));
            float src_x = x - dx, src_y = y - dy;
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

// ==============================
// Kernel construction
// ==============================

cv::Mat tileInfluenceToMatchSize(const cv::Mat& small_kernel, int target_rows, int target_cols) {
    int kernel_rows = small_kernel.rows;
    int kernel_cols = small_kernel.cols;
    int repeat_y = static_cast<int>(std::ceil(static_cast<double>(target_rows) / kernel_rows));
    int repeat_x = static_cast<int>(std::ceil(static_cast<double>(target_cols) / kernel_cols));
    cv::Mat tiled;
    cv::repeat(small_kernel, repeat_y, repeat_x, tiled);
    return tiled(cv::Rect(0, 0, target_cols, target_rows));
}

cv::Mat closeFormSolutionUnitSquare(const cv::Mat& points, double s = 10.0) {
    CV_Assert(points.cols == 2 && points.type() == CV_64F);
    int N = points.rows;
    cv::Mat colors(N, 1, CV_64F);
    double denom = std::pow(std::pow(2.0, s) - 1.0, 2.0);

    for (int i = 0; i < N; ++i) {
        double x = points.at<double>(i, 0);
        double y = points.at<double>(i, 1);
        double tmp1 = s * (x + y);
        double tmp2 = s * (y - x);
        double numer = std::pow(2.0, tmp1)
                     + std::pow(2.0, 2.0 * s - tmp1)
                     + std::pow(2.0, s - tmp2)
                     + std::pow(2.0, s + tmp2);
        colors.at<double>(i, 0) = numer / denom;
    }
    return colors;
}

cv::Mat reshapeFlatTo2D(const cv::Mat& flat_kernel, int Lx, int Ly) {
    CV_Assert(flat_kernel.rows == Lx * Ly && flat_kernel.cols == 1);
    cv::Mat reshaped_kernel(Ly, Lx, CV_64F);
    for (int i = 0; i < flat_kernel.rows; ++i) {
        int iy = i / Lx, ix = i % Lx;
        reshaped_kernel.at<double>(iy, ix) = flat_kernel.at<double>(i, 0);
    }
    return reshaped_kernel;
}

// ==============================
// Deformation (parallel / serial)
// ==============================

cv::Mat deformUnitTile_parallel(const cv::Mat& unit,
                                const std::vector<cv::Mat>& toric_cover_lst_local,
                                const cv::Point& picked_point,
                                const cv::Point& drag_vector,
                                double s = 10.0)
{
    int height = unit.rows, width = unit.cols;
    int Lx = bravais_lattice_x, Ly = bravais_lattice_y;

    cv::Mat vec_handle = (cv::Mat_<double>(3,1) << drag_vector.x * 1.0, drag_vector.y * 1.0, 0.0);
    cv::Mat point_handle = (cv::Mat_<double>(3,1) << static_cast<double>(picked_point.x),
                                                      static_cast<double>(picked_point.y), 1.0);
    cv::Mat base_points = getBasePoints(Lx, Ly);

    cv::Mat vector_field_x = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat vector_field_y = cv::Mat::zeros(height, width, CV_64F);

    #pragma omp parallel for
    for (int i = 0; i < (int)toric_cover_lst_local.size(); ++i) {
        cv::Mat local_fx = cv::Mat::zeros(height, width, CV_64F);
        cv::Mat local_fy = cv::Mat::zeros(height, width, CV_64F);

        const cv::Mat& matrix = toric_cover_lst_local[i];
        cv::Mat transformed_point = matrix * point_handle;
        double px = transformed_point.at<double>(0);
        double py = transformed_point.at<double>(1);

        cv::Mat transformed_vector = matrix * vec_handle;
        cv::Point2d orig_vec(transformed_vector.at<double>(0),
                             transformed_vector.at<double>(1));
        cv::Point2d mirrored_vec = mirror_direction_vector(orig_vec);

        cv::Mat points_orig = base_points.clone();
        for (int row = 0; row < points_orig.rows; ++row) {
            double raw_x = points_orig.at<double>(row, 0) + (Lx - px);
            double raw_y = points_orig.at<double>(row, 1) + (Ly - py);
            points_orig.at<double>(row, 0) = positive_mod(raw_x, (double)Lx) / (double)Lx;
            points_orig.at<double>(row, 1) = positive_mod(raw_y, (double)Ly) / (double)Ly;
        }

        cv::Mat colors_vector = closeFormSolutionUnitSquare(points_orig, s);
        cv::Mat reshaped = reshapeFlatTo2D(colors_vector, Lx, Ly);
        cv::Mat magnitude_orig = tileInfluenceToMatchSize(reshaped, height, width);

        local_fx += magnitude_orig * orig_vec.x;
        local_fy += magnitude_orig * orig_vec.y;

        cv::Point2d mirrored_point = mirror_point(cv::Point2d(px, py));
        cv::Mat points_mirror = base_points.clone();
        for (int row = 0; row < points_mirror.rows; ++row) {
            double raw_x = points_mirror.at<double>(row, 0) + (Lx - mirrored_point.x);
            double raw_y = points_mirror.at<double>(row, 1) + (Ly - mirrored_point.y);
            points_mirror.at<double>(row, 0) = positive_mod(raw_x, (double)Lx) / (double)Lx;
            points_mirror.at<double>(row, 1) = positive_mod(raw_y, (double)Ly) / (double)Ly;
        }

        cv::Mat colors_vector_mirror = closeFormSolutionUnitSquare(points_mirror, s);
        cv::Mat reshaped_mirror = reshapeFlatTo2D(colors_vector_mirror, Lx, Ly);
        cv::Mat tiled_mirror;
        int repeat_y = static_cast<int>(std::ceil(static_cast<double>(height) / reshaped_mirror.rows));
        int repeat_x = static_cast<int>(std::ceil(static_cast<double>(width) / reshaped_mirror.cols));
        cv::repeat(reshaped_mirror, repeat_y, repeat_x, tiled_mirror);
        cv::Mat magnitude_mirror = tiled_mirror(cv::Rect(0, 0, width, height));

        local_fx += magnitude_mirror * mirrored_vec.x;
        local_fy += magnitude_mirror * mirrored_vec.y;

        #pragma omp critical
        {
            vector_field_x += local_fx;
            vector_field_y += local_fy;
        }
    }

    cv::Mat warped = applyVectorField(unit, vector_field_x, vector_field_y);
    return warped;
}

cv::Mat deformUnitTile_serial(const cv::Mat& unit,
                              const std::vector<cv::Mat>& toric_cover_lst_local,
                              const cv::Point& picked_point,
                              const cv::Point& drag_vector,
                              double s = 10.0)
{
    int height = unit.rows, width = unit.cols;
    int Lx = bravais_lattice_x, Ly = bravais_lattice_y;

    cv::Point2d picked_pt_d(static_cast<double>(picked_point.x), static_cast<double>(picked_point.y));
    cv::Point2d drag_vec_d(static_cast<double>(drag_vector.x), static_cast<double>(drag_vector.y));
    cv::Mat vec_handle = (cv::Mat_<double>(3,1) << drag_vec_d.x * 3.0, drag_vec_d.y * 3.0, 0.0);
    cv::Mat point_handle = (cv::Mat_<double>(3,1) << picked_pt_d.x, picked_pt_d.y, 1.0);

    struct FV { cv::Point2d point; cv::Point2d vector; };
    std::vector<FV> fundamental_vecs(8);

    for (int i = 0; i < (int)toric_cover_lst_local.size(); ++i) {
        const cv::Mat& matrix = toric_cover_lst_local[i];
        cv::Mat transformed_point = matrix * point_handle;
        double px = transformed_point.at<double>(0);
        double py = transformed_point.at<double>(1);
        fundamental_vecs[i].point = cv::Point2d(px, py);
        fundamental_vecs[i + 4].point = mirror_point(cv::Point2d(px, py));
    }

    cv::Mat base_points = getPointsNotUniform(Lx, Ly, Lx, Ly);
    std::vector<cv::Mat> tiled_magnitudes_lst(8);
    cv::Mat vector_field_x = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat vector_field_y = cv::Mat::zeros(height, width, CV_64F);

    for (int i = 0; i < (int)fundamental_vecs.size(); ++i) {
        double x0 = fundamental_vecs[i].point.x;
        double y0 = fundamental_vecs[i].point.y;

        cv::Mat points = base_points.clone();
        for (int row = 0; row < points.rows; ++row) {
            double raw_x = points.at<double>(row, 0) + (Lx - x0);
            double raw_y = points.at<double>(row, 1) + (Ly - y0);
            double wrapped_x = std::fmod(std::abs(raw_x), static_cast<double>(Lx)) / static_cast<double>(Lx);
            double wrapped_y = std::fmod(std::abs(raw_y), static_cast<double>(Ly)) / static_cast<double>(Ly);
            points.at<double>(row, 0) = wrapped_x;
            points.at<double>(row, 1) = wrapped_y;
        }

        cv::Mat colors_vector = closeFormSolutionUnitSquare(points, s);
        cv::Mat reshaped = reshapeFlatTo2D(colors_vector, Lx, Ly);

        int repeat_y = static_cast<int>(std::ceil(static_cast<double>(height) / reshaped.rows));
        int repeat_x = static_cast<int>(std::ceil(static_cast<double>(width) / reshaped.cols));
        cv::Mat tiled;
        cv::repeat(reshaped, repeat_y, repeat_x, tiled);
        cv::Mat tiled_magnitudes = tiled(cv::Rect(0, 0, width, height));
        tiled_magnitudes_lst[i] = tiled_magnitudes;
    }

    for (int i = 0; i < (int)toric_cover_lst_local.size(); ++i) {
        const cv::Mat& matrix = toric_cover_lst_local[i];
        cv::Mat transformed_vector = matrix * vec_handle;
        double vx = transformed_vector.at<double>(0);
        double vy = transformed_vector.at<double>(1);
        fundamental_vecs[i].vector = cv::Point2d(vx, vy);
        cv::Point2d mirrored_vec = mirror_direction_vector(cv::Point2d(vx, vy));
        fundamental_vecs[i + 4].vector = mirrored_vec;

        vector_field_x += tiled_magnitudes_lst[i]     * fundamental_vecs[i].vector.x;
        vector_field_y += tiled_magnitudes_lst[i]     * fundamental_vecs[i].vector.y;
        vector_field_x += tiled_magnitudes_lst[i + 4] * fundamental_vecs[i + 4].vector.x;
        vector_field_y += tiled_magnitudes_lst[i + 4] * fundamental_vecs[i + 4].vector.y;
    }

    cv::Mat warped = applyVectorField(unit, vector_field_x, vector_field_y);
    return warped;
}

// ==============================
// Composition / base unit / tiling
// ==============================

cv::Mat blendPatternWithOverlay(const cv::Mat& base_pattern, const cv::Mat& overlay) {
    CV_Assert(base_pattern.size() == overlay.size());
    CV_Assert(base_pattern.type() == CV_8UC4 && overlay.type() == CV_8UC4);
    cv::Mat result; base_pattern.copyTo(result);
    std::vector<cv::Mat> overlay_channels; cv::split(overlay, overlay_channels);
    cv::Mat alpha_mask = overlay_channels[3];
    overlay.copyTo(result, alpha_mask);
    return result;
}

cv::Mat create_base_unit_fast(const cv::Mat& tile1,
                              const cv::Mat& tile2,
                              const cv::Mat& tile3,
                              const cv::Mat& tile4,
                              std::vector<cv::Mat>& toric_cover_lst)
{
    int max_dim = std::max(tile1.cols, tile1.rows);
    int width = max_dim * 4, height = width;

    g_unit_width = width; g_unit_height = height;

    cv::Mat unit(height, width, CV_8UC4, cv::Scalar(0,0,0,0));
    int center_x = width / 2, center_y = height / 2;

    auto placeTile = [&](const cv::Mat& tile, int x, int y) {
        cv::Rect roi(x, y, tile.cols, tile.rows);
        if (roi.x<0 || roi.y<0 || roi.x+roi.width>unit.cols || roi.y+roi.height>unit.rows) {
            std::cerr << "[Warning] ROI out of bounds\n"; 
            return;
        }
        std::vector<cv::Mat> ch; cv::split(tile, ch);
        tile.copyTo(unit(roi), ch[3]);
    };

    toric_cover_lst.clear(); toric_cover_lst.reserve(4);

    // TILE 1
    int t1_x = center_x - tile1.cols;
    int t1_y = center_y - tile1.rows;
    placeTile(tile1, t1_x, t1_y);
    toric_cover_lst.push_back(cv::Mat::eye(3,3,CV_64F));

    // TILE 2 (90°)
    cv::Mat tile2_r = rotateImage(tile2, 90);
    int t2_x = center_x - tile2.rows + q2_spacing_x;
    int t2_y = center_y + q2_spacing_y;
    placeTile(tile2_r, t2_x, t2_y);

    cv::Mat T2_origin = cv::Mat::eye(3,3,CV_64F);
    T2_origin.at<double>(0,2) = -tile2.cols/2.0;
    T2_origin.at<double>(1,2) = -tile2.rows/2.0;
    cv::Mat R90 = (cv::Mat_<double>(3,3) << 0,1,0, -1,0,0, 0,0,1);
    cv::Mat T2_back = cv::Mat::eye(3,3,CV_64F);
    T2_back.at<double>(0,2) = t2_x + tile2_r.cols/2.0;
    T2_back.at<double>(1,2) = t2_y + tile2_r.rows/2.0;
    cv::Mat Tg2 = T2_back * R90 * T2_origin;
    Tg2.at<double>(0,2) -= t1_x; Tg2.at<double>(1,2) -= t1_y;
    toric_cover_lst.push_back(Tg2);

    // TILE 3 (180°)
    cv::Mat tile3_r = rotateImage(tile3, 180);
    int t3_x = center_x - tile1.cols + q3_spacing_x;
    int t3_y = center_y - tile1.rows + q3_spacing_y;
    placeTile(tile3_r, t3_x, t3_y);

    cv::Mat T3_origin = cv::Mat::eye(3,3,CV_64F);
    T3_origin.at<double>(0,2) = -tile3.cols/2.0;
    T3_origin.at<double>(1,2) = -tile3.rows/2.0;
    cv::Mat R180 = (cv::Mat_<double>(3,3) << -1,0,0, 0,-1,0, 0,0,1);
    cv::Mat T3_back = cv::Mat::eye(3,3,CV_64F);
    T3_back.at<double>(0,2) = t3_x + tile3_r.cols/2.0;
    T3_back.at<double>(1,2) = t3_y + tile3_r.rows/2.0;
    cv::Mat Tg3 = T3_back * R180 * T3_origin;
    Tg3.at<double>(0,2) -= t1_x; Tg3.at<double>(1,2) -= t1_y;
    toric_cover_lst.push_back(Tg3);

    // TILE 4 (270°)
    cv::Mat tile4_r = rotateImage(tile4, 270);
    int t4_x = center_x - tile1.cols + q4_spacing_x;
    int t4_y = center_y - tile1.rows + q4_spacing_y;
    placeTile(tile4_r, t4_x, t4_y);

    cv::Mat T4_origin = cv::Mat::eye(3,3,CV_64F);
    T4_origin.at<double>(0,2) = -tile4.cols/2.0;
    T4_origin.at<double>(1,2) = -tile4.rows/2.0;
    cv::Mat R270 = (cv::Mat_<double>(3,3) << 0,-1,0, 1,0,0, 0,0,1);
    cv::Mat T4_back = cv::Mat::eye(3,3,CV_64F);
    T4_back.at<double>(0,2) = t4_x + tile4_r.cols/2.0;
    T4_back.at<double>(1,2) = t4_y + tile4_r.rows/2.0;
    cv::Mat Tg4 = T4_back * R270 * T4_origin;
    Tg4.at<double>(0,2) -= t1_x; Tg4.at<double>(1,2) -= t1_y;
    toric_cover_lst.push_back(Tg4);

    return unit;
}

cv::Mat create_base_unit(const cv::Mat& tile1, const cv::Mat& tile2,
                         const cv::Mat& tile3, const cv::Mat& tile4,
                         std::vector<cv::Mat>& toric_cover_lst)
{
    int max_dim = std::max(tile1.cols, tile1.rows);
    int width = max_dim * 4, height = width;

    g_unit_width = width; g_unit_height = height;
    cv::Mat unit(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    int center_x = width / 2, center_y = height / 2;

    toric_cover_lst.clear();

    auto placeTile = [&](const cv::Mat& tile, int x, int y) {
        cv::Rect roi(x, y, tile.cols, tile.rows);
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > unit.cols || roi.y + roi.height > unit.rows) {
            std::cerr << "[Warning] ROI out of bounds while placing tile.\n";
            return;
        }
        std::vector<cv::Mat> ch; cv::split(tile, ch);
        cv::Mat mask = ch[3];
        tile.copyTo(unit(roi), mask);
    };

    int t1_x = center_x - tile1.cols;
    int t1_y = center_y - tile1.rows;
    placeTile(tile1, t1_x, t1_y);

    cv::Mat T1 = cv::Mat::eye(3,3,CV_64F);
    T1.at<double>(0,2) = t1_x;
    T1.at<double>(1,2) = t1_y;
    toric_cover_lst.push_back(T1);

    cv::Mat tile2_r = rotateImage(tile2, 90);
    int t2_x = center_x - tile2.rows + q2_spacing_x;
    int t2_y = center_y + q2_spacing_y;
    placeTile(tile2_r, t2_x, t2_y);

    cv::Mat T2_origin = cv::Mat::eye(3,3,CV_64F);
    T2_origin.at<double>(0,2) = -tile2.cols / 2.0;
    T2_origin.at<double>(1,2) = -tile2.rows / 2.0;
    cv::Mat R90 = (cv::Mat_<double>(3,3) << 0,1,0, -1,0,0, 0,0,1);
    cv::Mat T2_back = cv::Mat::eye(3,3,CV_64F);
    T2_back.at<double>(0,2) = t2_x + tile2_r.cols / 2.0;
    T2_back.at<double>(1,2) = t2_y + tile2_r.rows / 2.0;
    toric_cover_lst.push_back(T2_back * R90 * T2_origin);

    cv::Mat tile3_r = rotateImage(tile3, 180);
    int t3_x = center_x - tile1.cols + q3_spacing_x;
    int t3_y = center_y - tile1.rows + q3_spacing_y;
    placeTile(tile3_r, t3_x, t3_y);

    cv::Mat T3_origin = cv::Mat::eye(3,3,CV_64F);
    T3_origin.at<double>(0,2) = -tile3.cols / 2.0;
    T3_origin.at<double>(1,2) = -tile3.rows / 2.0;
    cv::Mat R180 = (cv::Mat_<double>(3,3) << -1,0,0, 0,-1,0, 0,0,1);
    cv::Mat T3_back = cv::Mat::eye(3,3,CV_64F);
    T3_back.at<double>(0,2) = t3_x + tile3_r.cols / 2.0;
    T3_back.at<double>(1,2) = t3_y + tile3_r.rows / 2.0;
    toric_cover_lst.push_back(T3_back * R180 * T3_origin);

    cv::Mat tile4_r = rotateImage(tile4, 270);
    int t4_x = center_x - tile1.cols + q4_spacing_x;
    int t4_y = center_y - tile1.rows + q4_spacing_y;
    placeTile(tile4_r, t4_x, t4_y);

    cv::Mat T4_origin = cv::Mat::eye(3,3,CV_64F);
    T4_origin.at<double>(0,2) = -tile4.cols / 2.0;
    T4_origin.at<double>(1,2) = -tile4.rows / 2.0;
    cv::Mat R270 = (cv::Mat_<double>(3,3) << 0,-1,0, 1,0,0, 0,0,1);
    cv::Mat T4_back = cv::Mat::eye(3,3,CV_64F);
    T4_back.at<double>(0,2) = t4_x + tile4_r.cols / 2.0;
    T4_back.at<double>(1,2) = t4_y + tile4_r.rows / 2.0;
    toric_cover_lst.push_back(T4_back * R270 * T4_origin);

    return unit;
}

cv::Mat cropToContent(const cv::Mat& img) {
    cv::Mat alpha;
    if (img.channels() == 4) {
        std::vector<cv::Mat> channels; cv::split(img, channels);
        alpha = channels[3];
    } else {
        cv::cvtColor(img, alpha, cv::COLOR_BGR2GRAY);
    }
    cv::Mat mask = alpha > 0;
    cv::Rect bbox = cv::boundingRect(mask);
    return img(bbox).clone();
}

cv::Mat generate_tiled_pattern(const cv::Mat& unit,
                               int repeats_x = 15, int repeats_y = 15,
                               int brx = 911, int bry = 911, float scale = 1.5f)
{
    int crop_width = WINDOW_WIDTH, crop_height = WINDOW_HEIGHT;
    int offset_x = 1000, offset_y = 1000;

    int total_width  = brx * repeats_x + offset_x;
    int total_height = bry * repeats_y + offset_y;

    cv::Mat pattern(total_height, total_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    std::vector<cv::Mat> ch; cv::split(unit, ch);
    cv::Mat alpha = ch[3];

    for (int y = 0; y < repeats_y; y++) {
        for (int x = 0; x < repeats_x; x++) {
            int px = x * brx, py = y * bry;
            cv::Rect roi(px, py,
                         std::min(unit.cols, pattern.cols - px),
                         std::min(unit.rows, pattern.rows - py));
            if (roi.width > 0 && roi.height > 0) {
                cv::Mat dstROI = pattern(roi);
                cv::Mat srcROI = unit(cv::Rect(0, 0, roi.width, roi.height));
                cv::Mat alphaROI = alpha(cv::Rect(0, 0, roi.width, roi.height));
                srcROI.copyTo(dstROI, alphaROI);
            }
        }
    }

    cv::Mat resized_pattern;
    int target_width  = std::min(std::max(1.0f, scale), 10.0f) * WINDOW_WIDTH;
    int target_height = static_cast<int>(target_width * (float(pattern.rows) / pattern.cols));
    cv::resize(pattern, resized_pattern, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

    int crop_x = std::max(0, (resized_pattern.cols - crop_width) / 2);
    int crop_y = std::max(0, (resized_pattern.rows - crop_height) / 2);
    crop_x = std::min(crop_x, resized_pattern.cols - crop_width);
    crop_y = std::min(crop_y, resized_pattern.rows - crop_height);

    cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
    cv::Mat visible = resized_pattern(crop_rect);
    return visible;
}

// ==============================
// SDL helpers
// ==============================

SDL_Texture* matToTexture(const cv::Mat& mat, SDL_Renderer* renderer) {
    cv::Mat converted;
    if (mat.channels() == 3)      cv::cvtColor(mat, converted, cv::COLOR_BGR2RGBA);
    else if (mat.channels() == 4) cv::cvtColor(mat, converted, cv::COLOR_BGRA2RGBA);
    else                          cv::cvtColor(mat, converted, cv::COLOR_GRAY2RGBA);

    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
        (void*)converted.data, converted.cols, converted.rows, 32, converted.step,
        0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000
    );
    if (!surface) return nullptr;

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    return texture;
}

cv::Mat loadAndResizeImage(const std::string& path, int max_size) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return img;
    }
    int original_width = img.cols, original_height = img.rows;
    float scale = 1.0f;
    if (std::max(original_width, original_height) > max_size) {
        scale = static_cast<float>(max_size) / std::max(original_width, original_height);
    }
    int new_width  = static_cast<int>(original_width  * scale);
    int new_height = static_cast<int>(original_height * scale);
    cv::Mat resized; cv::resize(img, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
    return resized;
}

cv::Mat loadImageRGBA(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) std::cerr << "Failed to load: " << path << "\n";
    if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
    return img;
}

inline SDL_Point to_renderer_space(int logical_x, int logical_y, float dpi_scale_x, float dpi_scale_y) {
    return { static_cast<int>(logical_x * dpi_scale_x),
             static_cast<int>(logical_y * dpi_scale_y) };
}

inline std::ostream& operator<<(std::ostream& os, const SDL_Point& p) { return os << "SDL_Point(" << p.x << "," << p.y << ")"; }
inline std::ostream& operator<<(std::ostream& os, const cv::Point& p) { return os << "cv::Point(" << p.x << "," << p.y << ")"; }
inline std::ostream& operator<<(std::ostream& os, const cv::Point2f& p) { return os << "cv::Point2f(" << p.x << "," << p.y << ")"; }

// ==============================
// Main
// ==============================

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " fish1.png fish2.png fish3.png fish4.png frame.png [overlays...]\n";
        return 1;
    }

    float scale = 2.0f;

    std::string fish1_path = argv[1];
    std::string fish2_path = argv[2];
    std::string fish3_path = argv[3];
    std::string fish4_path = argv[4];
    std::string frame_path = argv[5];

    SDL_Init(SDL_INIT_VIDEO);
    int win_w = WINDOW_WIDTH, win_h = WINDOW_HEIGHT;

    SDL_Window* window = SDL_CreateWindow("Fish Pattern (Tiling)",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          win_w, win_h,
                                          SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_RenderSetLogicalSize(renderer, win_w, win_h);
    int render_w, render_h;
    SDL_GetRendererOutputSize(renderer, &render_w, &render_h);
    float dpi_scale_x = static_cast<float>(render_w) / win_w;
    float dpi_scale_y = static_cast<float>(render_h) / win_h;

    cv::Mat base_fish1 = loadImageRGBA(fish1_path);
    cv::Mat base_fish2 = loadImageRGBA(fish2_path);
    cv::Mat base_fish3 = loadImageRGBA(fish3_path);
    cv::Mat base_fish4 = loadImageRGBA(fish4_path);

    if (base_fish1.empty() || base_fish2.empty() || base_fish3.empty() || base_fish4.empty()) {
        std::cerr << "[Error] Failed to load one or more tile images.\n";
        return 1;
    }

    cv::Mat fish1 = base_fish1.clone();
    cv::Mat fish2 = base_fish2.clone();
    cv::Mat fish3 = base_fish3.clone();
    cv::Mat fish4 = base_fish4.clone();

    std::vector<cv::Mat> toric_cover_lst;
    cv::Mat unit = create_base_unit_fast(fish1, fish2, fish3, fish4, toric_cover_lst);
    cv::Mat pattern = generate_tiled_pattern(unit, 12, 8, bravais_lattice_x, bravais_lattice_y, scale);

    cv::Mat overlay_img;
    if (argc == 6) {
        overlay_img = cv::imread(argv[5], cv::IMREAD_UNCHANGED);
        if (overlay_img.empty()) {
            std::cerr << "Warning: Could not load overlay image: " << argv[5] << "\n";
            overlay_img = cv::Mat::zeros(pattern.size(), CV_8UC4);
        } else {
            cv::resize(overlay_img, overlay_img, pattern.size());
        }
    } else {
        overlay_img = cv::Mat::zeros(pattern.size(), CV_8UC4);
    }

    cv::Mat blended = blendPatternWithOverlay(pattern, overlay_img);
    SDL_Texture* texture_current   = matToTexture(blended, renderer);
    SDL_Texture* mini_tile_texture = matToTexture(fish1, renderer);

    int mini_w = 300;
    int mini_h = fish1.rows * mini_w / fish1.cols;
    int mini_x = 20, mini_y = win_h - mini_h - 20;
    SDL_Rect preview_rect = { mini_x, mini_y, mini_w, mini_h };

    bool running = true, is_dragging = false;
    SDL_Event event;
    SDL_Point drag_screen_start{}, drag_screen_end{};
    cv::Point picked_point(0, 0), last_dragged_point(0, 0);
    bool processed_simulation = false;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;

            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                int mx = event.button.x, my = event.button.y;

                if (mx >= preview_rect.x && mx < preview_rect.x + preview_rect.w &&
                    my >= preview_rect.y && my < preview_rect.y + preview_rect.h)
                {
                    drag_screen_start = { mx, my };
                    is_dragging = true;

                    float rel_x = float(mx - preview_rect.x) / float(preview_rect.w);
                    float rel_y = float(my - preview_rect.y) / float(preview_rect.h);
                    rel_x = std::clamp(rel_x, 0.0f, 1.0f);
                    rel_y = std::clamp(rel_y, 0.0f, 1.0f);

                    picked_point.x = int(rel_x * (fish1.cols - 1));
                    picked_point.y = int(rel_y * (fish1.rows - 1));
                }
            }
            else if (event.type == SDL_MOUSEMOTION && is_dragging) {
                int mx = event.button.x, my = event.button.y;
                cv::Point drag_vector(mx - drag_screen_start.x, my - drag_screen_start.y);
                last_dragged_point = picked_point + drag_vector;

                cv::Mat preview = deformUnitTile_parallel(base_fish1, toric_cover_lst, picked_point, drag_vector, sigma);
                SDL_DestroyTexture(mini_tile_texture);
                mini_tile_texture = matToTexture(preview, renderer);
            }
            else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                if (!is_dragging) continue;
                is_dragging = false;

                int mx = event.button.x, my = event.button.y;
                cv::Point drag_vector(mx - drag_screen_start.x, my - drag_screen_start.y);
                last_dragged_point = picked_point + drag_vector;

                fish1 = deformUnitTile_parallel(base_fish1, toric_cover_lst, picked_point, drag_vector, sigma);
                fish2 = deformUnitTile_parallel(base_fish2, toric_cover_lst, picked_point, drag_vector, sigma);
                fish3 = deformUnitTile_parallel(base_fish3, toric_cover_lst, picked_point, drag_vector, sigma);
                fish4 = deformUnitTile_parallel(base_fish4, toric_cover_lst, picked_point, drag_vector, sigma);

                base_fish1 = fish1.clone();
                base_fish2 = fish2.clone();
                base_fish3 = fish3.clone();
                base_fish4 = fish4.clone();

                unit = create_base_unit_fast(fish1, fish2, fish3, fish4, toric_cover_lst);
                pattern = generate_tiled_pattern(unit, 12, 8, bravais_lattice_x, bravais_lattice_y, scale);

                cv::Mat resized_overlay;
                if (!overlay_img.empty() && overlay_img.size() != pattern.size()) {
                    cv::resize(overlay_img, resized_overlay, pattern.size());
                } else {
                    resized_overlay = overlay_img;
                }
                cv::Mat blended2 = blendPatternWithOverlay(pattern, resized_overlay);

                SDL_DestroyTexture(texture_current);
                texture_current = matToTexture(blended2, renderer);

                SDL_DestroyTexture(mini_tile_texture);
                mini_tile_texture = matToTexture(fish1, renderer);
            }
            
            if (event.type == SDL_MOUSEWHEEL) {
                if (event.wheel.y > 0) scale *= 1.1f;
                else if (event.wheel.y < 0) scale /= 1.1f;
                scale = std::clamp(scale, 1.0f, 10.0f);

                pattern = generate_tiled_pattern(unit, 12, 8, bravais_lattice_x, bravais_lattice_y, scale);

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
                        fs::path fish1_p = outdir / ("fish1_deformed_" + ts + ".png");
                        fs::path fish2_p = outdir / ("fish2_deformed_" + ts + ".png");
                        fs::path fish3_p = outdir / ("fish3_deformed_" + ts + ".png");
                        fs::path fish4_p = outdir / ("fish4_deformed_" + ts + ".png");
                        fs::path unit_p  = outdir / ("unit_deformed_"  + ts + ".png");

                        bool ok1 = save_png_lossless(fish1, fish1_p.string());
                        bool ok2 = save_png_lossless(fish2, fish2_p.string());
                        bool ok3 = save_png_lossless(fish3, fish3_p.string());
                        bool ok4 = save_png_lossless(fish4, fish4_p.string());
                        bool ok5 = save_png_lossless(unit,  unit_p.string());

                        if (ok1 && ok2 && ok3 && ok4 && ok5) {
                            std::cout << "[Save] Done!\n";
                        } else {
                            std::cerr << "[Save] Failed to save one or more outputs.\n";
                        }
                    }
                }
            }
        }

        // === Rendering ===
        SDL_SetRenderDrawColor(renderer, 25, 30, 40, 255);
        SDL_RenderClear(renderer);

        SDL_Rect dst = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, texture_current, nullptr, &dst);

        // Minimap background
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 190);
        SDL_RenderFillRect(renderer, &preview_rect);

        // Minimap tile
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);
        SDL_RenderCopy(renderer, mini_tile_texture, nullptr, &preview_rect);

        // Yellow dot
        float rel_x = std::clamp(last_dragged_point.x / float(fish1.cols - 1), 0.0f, 1.0f);
        float rel_y = std::clamp(last_dragged_point.y / float(fish1.rows - 1), 0.0f, 1.0f);
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
    SDL_DestroyTexture(mini_tile_texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
