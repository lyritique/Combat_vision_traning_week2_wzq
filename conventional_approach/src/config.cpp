
# include "config.hpp"

# include <opencv2/core.hpp>
# include <filesystem>
# include <stdexcept>

# ifndef PROJECT_ROOT
#   define PROJECT_ROOT "."
# endif

namespace
{
double dbl(const cv::FileNode& n, const char* key, double def)
{
    if (n.empty() || n[key].empty()) return def;
    double v = def;
    n[key] >> v;
    return v;
}

float flt(const cv::FileNode& n, const char* key, float def)
{
    return static_cast<float>(dbl(n, key, def));
}

int integer(const cv::FileNode& n, const char* key, int def)
{
    if (n.empty() || n[key].empty()) return def;
    int v = def;
    n[key] >> v;
    return v;
}

bool flag(const cv::FileNode& n, const char* key, bool def)
{
    return integer(n, key, def ? 1 : 0) != 0;
}

std::string str(const cv::FileNode& n, const char* key, const std::string& def)
{
    if (n.empty() || n[key].empty()) return def;
    std::string v;
    n[key] >> v;
    return v;
}

std::vector<double> vec(const cv::FileNode& n, const char* key)
{
    std::vector<double> out;
    if (n.empty() || n[key].empty()) return out;
    for (auto it = n[key].begin(); it != n[key].end(); ++it)
        out.push_back(static_cast<double>(*it));
    return out;
}

void readCam(const cv::FileNode& n, CamCfg& c)
{
    if (n.empty()) return;
    c.width  = integer(n, "width", c.width);
    c.height = integer(n, "height", c.height);
    c.fx = dbl(n, "fx", c.fx);
    c.fy = dbl(n, "fy", c.fy);
    c.cx = dbl(n, "cx", c.cx);
    c.cy = dbl(n, "cy", c.cy);
    const std::vector<double> d = vec(n, "dist");
    if (!d.empty()) c.dist = d;
}

void readRun(const cv::FileNode& n, RunCfg& r)
{
    if (n.empty()) return;
    r.input     = resolvePath(str(n, "input", r.input));
    r.out_dir   = resolvePath(str(n, "out_dir", r.out_dir));
    r.out_video = resolvePath(str(n, "out_video", r.out_video));
    r.out_csv   = resolvePath(str(n, "out_csv", r.out_csv));
    r.det_csv   = resolvePath(str(n, "det_csv", r.det_csv));
    r.color     = str(n, "color", r.color);
    r.codec     = str(n, "codec", r.codec);
    r.thresh    = integer(n, "thresh", r.thresh);
}
} // namespace

std::string resolvePath(const std::string& p)
{
    if (p.empty()) return p;
    const std::string full = (p[0] == '/') ? p : std::string(PROJECT_ROOT) + "/" + p;
    try
    {
        // 规范化掉 ".." 之类的片段，日志里看到的路径更直观
        return std::filesystem::weakly_canonical(full).string();
    }
    catch (const std::exception&)
    {
        return full;
    }
}

std::string findConfigPath(int argc, char** argv)
{
    for (int k = 1; k + 1 < argc; ++k)
        if (std::string(argv[k]) == "--config") return argv[k + 1];
    return resolvePath("config/default.yaml");
}

std::vector<std::string> positionalArgs(int argc, char** argv)
{
    std::vector<std::string> pos;
    for (int k = 1; k < argc; ++k)
    {
        const std::string a = argv[k];
        if (a == "--config") { ++k; continue; }
        pos.push_back(a);
    }
    return pos;
}

Config Config::load(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("打不开配置文件: " + path);

    Config c;
    c.path = path;

    // 注意：FileStorage 的嵌套节点必须逐级访问，不能用 "camera.stills" 这种点号写法
    readCam(fs["camera"]["stills"], c.stills);
    readCam(fs["camera"]["video"],  c.video);

    const cv::FileNode a = fs["armor"];
    c.armor.small_w    = flt(a, "small_width_mm", c.armor.small_w);
    c.armor.large_w    = flt(a, "large_width_mm", c.armor.large_w);
    c.armor.bar_len    = flt(a, "bar_length_mm", c.armor.bar_len);
    c.armor.max_reproj = flt(a, "max_reproj_px", c.armor.max_reproj);
    c.armor.use_large  = flag(a, "use_large", c.armor.use_large);

    const cv::FileNode l = fs["light"];
    c.light.morph      = integer(l, "morph_kernel", c.light.morph);
    c.light.min_pts    = integer(l, "min_contour_pts", c.light.min_pts);
    c.light.min_w      = flt(l, "min_bar_width_px", c.light.min_w);
    c.light.min_aspect = flt(l, "min_aspect", c.light.min_aspect);
    c.light.max_aspect = flt(l, "max_aspect", c.light.max_aspect);
    c.light.max_tilt   = flt(l, "max_tilt_deg", c.light.max_tilt);

    const cv::FileNode p = fs["pair"];
    c.pair.min_len  = flt(p, "min_length_ratio", c.pair.min_len);
    c.pair.max_ang  = flt(p, "max_angle_diff_deg", c.pair.max_ang);
    c.pair.max_perp = flt(p, "max_perp_diff_deg", c.pair.max_perp);
    c.pair.min_gap  = flt(p, "min_gap_ratio", c.pair.min_gap);
    c.pair.max_gap  = flt(p, "max_gap_ratio", c.pair.max_gap);
    c.pair.nominal  = flt(p, "nominal_ratio", c.pair.nominal);

    const cv::FileNode d = fs["digit"];
    c.digit.model        = resolvePath(str(d, "model", c.digit.model));
    c.digit.labels       = resolvePath(str(d, "labels", c.digit.labels));
    c.digit.conf_min     = flt(d, "conf_min", c.digit.conf_min);
    c.digit.filter_image = flag(d, "filter_in_image", c.digit.filter_image);
    c.digit.filter_video = flag(d, "filter_in_video", c.digit.filter_video);
    c.digit.w_small      = integer(d, "warp_small_w", c.digit.w_small);
    c.digit.w_large      = integer(d, "warp_large_w", c.digit.w_large);
    c.digit.h            = integer(d, "warp_h", c.digit.h);
    c.digit.roi_w        = integer(d, "roi_w", c.digit.roi_w);
    c.digit.top_y        = integer(d, "top_y", c.digit.top_y);
    c.digit.bot_y        = integer(d, "bottom_y", c.digit.bot_y);

    const cv::FileNode y = fs["yolo"];
    c.yolo.model     = resolvePath(str(y, "model", c.yolo.model));
    c.yolo.in_w      = integer(y, "input_w", c.yolo.in_w);
    c.yolo.in_h      = integer(y, "input_h", c.yolo.in_h);
    c.yolo.pad       = integer(y, "pad_value", c.yolo.pad);
    c.yolo.num_cls   = integer(y, "num_classes", c.yolo.num_cls);
    c.yolo.num_color = integer(y, "num_colors", c.yolo.num_color);
    c.yolo.conf      = flt(y, "conf_thresh", c.yolo.conf);
    c.yolo.nms       = flt(y, "nms_iou", c.yolo.nms);
    c.yolo.max_dist  = flt(y, "max_dist_m", c.yolo.max_dist);

    const cv::FileNode r = fs["run"];
    readRun(r["conventional_image"], c.conv_image);
    readRun(r["conventional_video"], c.conv_video);
    readRun(r["yolo_image"],         c.yolo_image);
    readRun(r["yolo_video"],         c.yolo_video);

    const cv::FileNode e = fs["eval"];
    c.eval.max_px = flt(e, "max_px", c.eval.max_px);

    return c;
}

CameraParams Config::camera(bool use_video) const
{
    const CamCfg& s = use_video ? video : stills;
    CameraParams p;
    p.fx = s.fx;
    p.fy = s.fy;
    p.cx = s.cx;
    p.cy = s.cy;
    p.dist = s.dist;
    return p;
}
