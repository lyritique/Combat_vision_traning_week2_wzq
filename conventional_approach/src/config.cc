#include "config.hpp"
#include <opencv2/core.hpp>
#include <stdexcept>

static float  f(const cv::FileNode& n, const char* k, float d)
{ if (n[k].empty()) return d; double v = d; n[k] >> v; return (float)v; }
static int    i(const cv::FileNode& n, const char* k, int d)
{ if (n[k].empty()) return d; int v = d; n[k] >> v; return v; }
static std::string s(const cv::FileNode& n, const char* k, const std::string& d)
{ if (n[k].empty()) return d; std::string v; n[k] >> v; return v; }

Config Config::load(const std::string& p)
{
    cv::FileStorage fs(p, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::runtime_error("打不开配置: " + p);
    Config c; c.path = p;

    auto cam = [&](const char* key, CamCfg& o) {
        cv::FileNode n = fs[key];
        if (n.empty()) return;
        o.width = i(n,"width",o.width); o.height = i(n,"height",o.height);
        o.fx = f(n,"fx",(float)o.fx); o.fy = f(n,"fy",(float)o.fy);
        o.cx = f(n,"cx",(float)o.cx); o.cy = f(n,"cy",(float)o.cy);
        o.dist.clear();
        for (auto it = n["dist"].begin(); it != n["dist"].end(); ++it) o.dist.push_back((double)*it);
    };
    cam("camera.stills", c.stills);
    cam("camera.video",  c.video);

    cv::FileNode a = fs["armor"];
    c.armor.small_w = f(a,"small_width_mm",135.f);
    /* ... 其余 key 同理，一行一个 ... */

    return c;
}

CameraParams Config::camera(bool use_video) const
{
    const CamCfg& s = use_video ? video : stills;
    CameraParams p; p.fx=s.fx; p.fy=s.fy; p.cx=s.cx; p.cy=s.cy; p.dist=s.dist;
    return p;
}

void checkResolution(const CamCfg& c, int w, int h, const char* tag)
{
    if (c.width && (c.width != w || c.height != h))
        throw std::runtime_error(std::string("内参与素材分辨率不符[") + tag + "]: 配置 " +
            std::to_string(c.width) + "x" + std::to_string(c.height) + "，实际 " +
            std::to_string(w) + "x" + std::to_string(h));
}


