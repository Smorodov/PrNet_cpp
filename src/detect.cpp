#include <iostream>
#include "opencv2/opencv.hpp"
#include "net.h"
#include <iostream>
#include <fstream>
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"

using namespace std;
// -----------------------------
// .txt data files loader
// -----------------------------
cv::Mat loadtxt(std::string fname)
{
    std::ifstream myfile(fname);

    float a;
    std::vector<float> data;
    while (myfile >> a)
    {
        data.push_back(a);
    }
    cv::Mat res = cv::Mat(data.size(), 1, CV_32FC1);
    for (int i = 0; i < data.size(); ++i)
    {
        res.at<float>(i) = data[i];
    }
    return res;
}
// -----------------------------
// NcNN tensor to cv::Mat converter
// To get displayable image.
// -----------------------------
void ncnn2Mat(ncnn::Mat& ncnn_img, cv::Mat& dst)
{
    dst = cv::Mat(ncnn_img.h, ncnn_img.w, CV_32FC3);
    for (int i = 0; i < ncnn_img.h; i++)
    {
        for (int j = 0; j < ncnn_img.w; j++)
        {
            int c = 0;
            float c1 = ((float*)ncnn_img.data)[j + i * ncnn_img.w + c * ncnn_img.h * ncnn_img.w];
            ++c;
            float c2 = ((float*)ncnn_img.data)[j + i * ncnn_img.w + c * ncnn_img.h * ncnn_img.w];
            ++c;
            float c3 = ((float*)ncnn_img.data)[j + i * ncnn_img.w + c * ncnn_img.h * ncnn_img.w];
            dst.at<cv::Vec3f>(i, j) = cv::Vec3f(c1, c2, c3);
        }
    }
}

// -----------------------------
// Facial 68 landmarks plotter
// -----------------------------
void plot_kpt(cv::Mat& image, std::vector<cv::Point2f> kpt)
{
    std::set<int> end_list = { 16, 21, 26, 41, 47, 30, 35, 67 };
    //Draw 68 key points
    // Args :
    // image: the input image
    // kpt : (68, 3).

    for (int i = 0; i < kpt.size() - 1; ++i)
    {
        cv::Point2d st = kpt[i];
        cv::circle(image, st, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        if (end_list.find(i) != end_list.end())
        {
            continue;
        }
        cv::Point2d ed = kpt[i + 1];
        cv::line(image, st, ed, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

// -----------------------------
// 3D model plotter class
// -----------------------------
class MeshPlot
{
public:
    MeshPlot() { ; }
    ~MeshPlot() { ; }
    // -----------------------------
    // triangle face description 
    // -----------------------------
    class triangle
    {
    public:
        triangle() { ; }
        ~triangle() { ; }
        cv::Vec3i vert_ind;
        cv::Vec3f normal;
        int tri_ind;
        float depth;
    };
private:
    // ----------------------------------------------
    // Comparsion function for triangle depth sorting
    // ----------------------------------------------
    static bool cmp(const triangle& a, const triangle& b)
    {
        return a.depth < b.depth;
    }
    // -----------------------------
    // triangles for visibility map drawing
    // -----------------------------
    std::vector<triangle> depth_tri;
    // --------------------------------------------------------------
    // create labels map for visibility mask drawing
    // --------------------------------------------------------------
    void DrawLabelsMask(cv::Mat& imgLabel, std::vector<cv::Vec3f>& points, std::vector<triangle>& tris, float scale = 1)
    {
        imgLabel = cv::Scalar::all(0);
        for (int i = 0; i < tris.size(); i++)
        {
            cv::Point t[3];
            int ind1 = tris[i].vert_ind[0];
            int ind2 = tris[i].vert_ind[1];
            int ind3 = tris[i].vert_ind[2];
            t[0].x = cvRound(points[ind1][0] * scale);
            t[0].y = cvRound(points[ind1][1] * scale);
            t[1].x = cvRound(points[ind2][0] * scale);
            t[1].y = cvRound(points[ind2][1] * scale);
            t[2].x = cvRound(points[ind3][0] * scale);
            t[2].y = cvRound(points[ind3][1] * scale);
            cv::fillConvexPoly(imgLabel, t, 3, cv::Scalar_<int>((tris[i].tri_ind + 1)));
        }
    }
    // -----------------------------
    // Visibility mask
    // -----------------------------
    void DrawVisibilityMask(cv::Mat& mask, std::set<int>& visible, std::vector<triangle>& tris, std::vector<cv::Vec3i>& triangles, cv::Mat& uv_kpt_ind)
    {
        for (int i = 0; i < triangles.size(); i++)
        {
            if (visible.find(i + 1) != visible.end())
            {
                int p1 = triangles[i][0];
                int y1 = cvRound(uv_kpt_ind.at<float>(p1)) % 256;
                int x1 = cvRound(uv_kpt_ind.at<float>(p1)) / 256;
                int p2 = triangles[i][1];
                int y2 = cvRound(uv_kpt_ind.at<float>(p2)) % 256;
                int x2 = cvRound(uv_kpt_ind.at<float>(p2)) / 256;
                int p3 = triangles[i][2];
                int y3 = cvRound(uv_kpt_ind.at<float>(p1)) % 256;
                int x3 = cvRound(uv_kpt_ind.at<float>(p1)) / 256;
                mask.at<uchar>(x1, y1) = 255;
                mask.at<uchar>(x2, y2) = 255;
                mask.at<uchar>(x3, y3) = 255;
            }
        }
    }
    // -----------------------------
    // Phong rendering 
    // -----------------------------
    void DrawMesh(cv::Mat& imgLabel, std::vector<cv::Vec3f>& points, std::vector<triangle>& tris, float scale = 1)
    {
        for (int i = 0; i < tris.size(); i++)
        {
            cv::Point t[3];
            int ind1 = tris[i].vert_ind[0];
            int ind2 = tris[i].vert_ind[1];
            int ind3 = tris[i].vert_ind[2];
            t[0].x = cvRound(points[ind1][0] * scale);
            t[0].y = cvRound(points[ind1][1] * scale);
            t[1].x = cvRound(points[ind2][0] * scale);
            t[1].y = cvRound(points[ind2][1] * scale);
            t[2].x = cvRound(points[ind3][0] * scale);
            t[2].y = cvRound(points[ind3][1] * scale);

            cv::Vec3f diffuse = { 0.4,0.0,0.0 };
            cv::Vec3f specular = { 0.4,0.4,0.4 };
            cv::Vec3f ambient = { 0.2,0.2,0.2 };

            float LightPower = 2;
            cv::Vec3f  LightDir(-1.0, 1.0, -1.0);
            LightDir = LightDir / cv::norm(LightDir);

            float c = (2 * (tris[i].normal.dot(LightDir) * tris[i].normal) - LightDir).dot(cv::Vec3f(0, 0, -1));
            //float c2 = (tris[i].normal.dot(LightDir));
            c = pow(c, 4);
            c *= LightPower;
            cv::Vec3f result_color = (specular)*c + diffuse + ambient;

            if (result_color[0] < 0)
            {
                result_color[0] = 0;
            }
            if (result_color[0] > 1)
            {
                result_color[0] = 1;
            }

            if (result_color[1] < 0)
            {
                result_color[1] = 0;
            }
            if (result_color[1] > 1)
            {
                result_color[1] = 1;
            }

            if (result_color[2] < 0)
            {
                result_color[2] = 0;
            }
            if (result_color[2] > 1)
            {
                result_color[2] = 1;
            }
            result_color *= 255;
            cv::Scalar color(result_color[0], result_color[1], result_color[2]);
            cv::fillConvexPoly(imgLabel, t, 3, color);

        }
    }
    public:
    // -----------------------------
    // Main plottint method
    // -----------------------------
    void Plot(cv::Mat& img, cv::Mat& mask, cv::Mat& uv_kpt_ind, std::vector<cv::Point3f>& verts, std::vector<cv::Vec3i>& triangles, float scale = 1, bool shading = true, bool wire = true)
    {
        std::vector<cv::Vec3f> vertices3d;
        // Point3f to Vec3f
        for (int i = 0; i < verts.size(); ++i)
        {
            vertices3d.push_back(cv::Vec3f(verts[i].x, verts[i].y, verts[i].z));
        }
        // fill triangle class fields.
        // compute normals and depth
        for (int i = 0; i < triangles.size(); i++)
        {
            // get triangle points
            cv::Vec3f a = vertices3d[triangles[i][0]];
            cv::Vec3f b = vertices3d[triangles[i][1]];
            cv::Vec3f c = vertices3d[triangles[i][2]];
            // compute normals
            cv::Vec3f v1 = b - a;
            cv::Vec3f v2 = b - c;
            cv::Vec3f N(v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0]);
            N = N / cv::norm(N);
            // fill triangle class fields
            triangle t;
            // vertices points indices
            t.vert_ind = cv::Vec3i(triangles[i][0], triangles[i][1], triangles[i][2]);
            // triangle index in trianngles vector
            t.tri_ind = i;
            // compute mead depth 
            t.depth = (a[2] + b[2] + c[2]) / 3.0;
            // assign normal
            t.normal = N;
            // check if face visible by normal direction
            if (N[2] <= 0)
            {
                // if visible, append to list
                depth_tri.push_back(t);
            }
        }
        // sort faces by depth for plotting in from far to near order
        std::sort(depth_tri.begin(), depth_tri.end(), cmp);
        // Draw shaded 
        if (shading)
        {
            DrawMesh(img, vertices3d, depth_tri, scale);
        }
        // Create visibility mask
        cv::Mat depth_map = cv::Mat::zeros(img.size(), CV_32SC1);
        DrawLabelsMask(depth_map, vertices3d, depth_tri, scale);
        std::set<int> visible;
        for (int i = 0; i < depth_map.rows; ++i)
        {
            for (int j = 0; j < depth_map.cols; ++j)
            {
                int ind = depth_map.at<int>(i, j);
                if (ind > 0)
                {
                    visible.insert(ind - 1);
                }
            }
        }

        DrawVisibilityMask(mask, visible, depth_tri, triangles, uv_kpt_ind);
        // Draw wireframe 
        if (wire)
        {

            for (int i = 0; i < triangles.size(); i++)
            {
                if (visible.find(i + 1) != visible.end())
                {
                    cv::Point2f kpt1 = cv::Point2f(vertices3d[triangles[i][0]][0], vertices3d[triangles[i][0]][1]);
                    cv::Point2f kpt2 = cv::Point2f(vertices3d[triangles[i][1]][0], vertices3d[triangles[i][1]][1]);
                    cv::Point2f kpt3 = cv::Point2f(vertices3d[triangles[i][2]][0], vertices3d[triangles[i][2]][1]);
                    cv::line(img, kpt1 * scale, kpt2 * scale, cv::Scalar(0, 255, 255), 1);
                    cv::line(img, kpt2 * scale, kpt3 * scale, cv::Scalar(0, 255, 255), 1);
                    cv::line(img, kpt3 * scale, kpt1 * scale, cv::Scalar(0, 255, 255), 1);
                }
            }
        }
    }
};
// -----------------------------
// Face detector class
// -----------------------------
class FaceDetector
{
public:    
    std::string param_path;
    std::string bin_path;
    ncnn::Net retina_net;
    float retina_pixel_mean[3];
    float retina_pixel_std[3];

    FaceDetector()
    {
        param_path = "./models/retina.param";
        bin_path = "./models/retina.bin";
        retina_net.load_param(param_path.data());
        retina_net.load_model(bin_path.data());
        retina_pixel_mean[0] = 0;
        retina_pixel_mean[1] = 0;
        retina_pixel_mean[2] = 0;
        retina_pixel_std[0] = 1;
        retina_pixel_std[1] = 1;
        retina_pixel_std[2] = 1;
    }
    ~FaceDetector() { ; }

    void detect(cv::Mat &img, std::vector<cv::Rect>& faces)
    {
        int w = img.cols;
        int h = img.rows;
        int bw = 0;
        int bh = 0;

        if (w > h)
        {
            bh = w - h;
        }
        if (w < h)
        {
            bw = h - w;
        }
        cv::copyMakeBorder(img, img, 0, bh, 0, bw, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        ncnn::Mat retina_input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 300, 300);
        retina_input.substract_mean_normalize(retina_pixel_mean, retina_pixel_std);
        ncnn::Extractor retina_extractor = retina_net.create_extractor();
        retina_extractor.input("data", retina_input);


        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i)
        {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }

        std::vector<Anchor> proposals;
        proposals.clear();

        for (int i = 0; i < _feat_stride_fpn.size(); ++i)
        {
            ncnn::Mat cls;
            ncnn::Mat reg;
            ncnn::Mat pts;

            // get blob output
            char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
            char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
            char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
            retina_extractor.extract(clsname, cls);
            retina_extractor.extract(regname, reg);
            retina_extractor.extract(ptsname, pts);
            ac[i].FilterAnchor(cls, reg, pts, proposals);
        }

        // nms
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);

        float scale = max(w, h) / 300.0;
        img = img(cv::Rect(0, 0, w, h)).clone();

        for (int i = 0; i < result.size(); i++)
        {
            result[i].finalbox.x *= scale;
            result[i].finalbox.y *= scale;
            result[i].finalbox.width *= scale;
            result[i].finalbox.height *= scale;

            for (int j = 0; j < result[i].pts.size(); ++j)
            {
                result[i].pts[j].x *= scale;
                result[i].pts[j].y *= scale;
            }
        }

        for (int i = 0; i < result.size(); i++)
        {
            std::vector<cv::Point> LM;
            cv::Point left_center(result[i].pts[0]);
            cv::Point right_center(result[i].pts[1]);
        }
        for (int i = 0; i < result.size(); i++)
        {
            cv::Point p1 = cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y);
            cv::Point p2 = cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height);
            cv::Point pc = (p1 + p2) / 2;
            int w = p2.x - p1.x;
            int h = p2.y - p1.y;
            int sz = max(w, h);
            float margin = h / 255.0 * 20.0;

            p1.x = pc.x - sz / 2 - margin;
            p1.y = pc.y - sz / 2;
            p2.x = pc.x + sz / 2 + margin;
            p2.y = pc.y + sz / 2 + 2 * margin;

            if (p1.x < 0)
            {
                p1.x = 0;
            }
            if (p1.x > img.cols)
            {
                p1.x = img.cols - 1;
            }


            if (p1.y < 0)
            {
                p1.y = 0;
            }
            if (p1.y > img.rows)
            {
                p1.y = img.rows - 1;
            }


            if (p2.x < 0)
            {
                p2.x = 0;
            }
            if (p2.x > img.cols)
            {
                p2.x = img.cols - 1;
            }


            if (p2.y < 0)
            {
                p2.y = 0;
            }
            if (p2.y > img.rows)
            {
                p2.y = img.rows - 1;
            }

            faces.push_back(cv::Rect(p1, p2));
        }
    }

};

// -----------------------------
// MAIN
// -----------------------------
int main()
{
    FaceDetector FD;
    // 68 landmarks uv coordinates
    std::string  uv_kpt_ind_path = "./uv_data/uv_kpt_ind.txt";
    // 3D mesh points uv coordinates
    std::string  face_ind_path = "./uv_data/face_ind.txt";
    // 3D mesh triangles vertices indices
    std::string  triangles_path = "./uv_data/triangles.txt";

    cv::Mat uv_kpt_ind = loadtxt(uv_kpt_ind_path);  // 2 x 68 get kpt
    uv_kpt_ind = uv_kpt_ind.reshape(1, 2);
    cv::Mat face_ind = loadtxt(face_ind_path);  // get valid vertices in the pos map
    cv::Mat triangles = loadtxt(triangles_path); // ntri x 3
    triangles = triangles.reshape(1, triangles.rows / 3);

    float pixel_mean[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
    float pixel_std[3] = { (1.0 / 0.229) / 255, (1.0 / 0.224) / 255, (1.0 / 0.225) / 255 };

    std::string param_path = "./models/frNet_model.param";
    std::string bin_path = "./models/frNet_model.bin";
    ncnn::Net _net;
    _net.load_param(param_path.data());
    _net.load_model(bin_path.data());

    cv::Mat img = cv::imread("./images/face-11.jpg");

    if (!img.data)
    {
        cout << "load error" << endl;
        return 0;
    }
    // ------------------------------------------------   
    // Detect faces
    // ------------------------------------------------   
    std::vector<cv::Rect> faces;
    FD.detect(img, faces);
    // ------------------------------------------------   
    // Fit BFM
    // ------------------------------------------------   
    if (!faces.empty())
    {
        cv::Mat face_img = img(faces[0]).clone();
        cv::resize(face_img, face_img, cv::Size(256, 256));

        double start = (double)cv::getTickCount();

        ncnn::Mat input = ncnn::Mat::from_pixels(face_img.data, ncnn::Mat::PIXEL_BGR, face_img.cols, face_img.rows);
        ncnn::Mat output;
        input.substract_mean_normalize(pixel_mean, pixel_std);
        ncnn::Extractor _extractor = _net.create_extractor();
        _extractor.input("input_img", input);
        _extractor.extract("pos_img", output);

        cout << "Detection Time: " << (cv::getTickCount() - start) / cv::getTickFrequency() << "s" << std::endl;

        cv::Mat pos;
        ncnn2Mat(output, pos);
        pos = pos * 255;

        std::vector<cv::Mat> ch;
        cv::split(pos, ch);
        //texture = cv2.remap(image, mapr, None, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0));
        cv::Mat texture;
        cv::remap(face_img, texture, ch[0], ch[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        std::vector<cv::Point2f> kpts;

        for (int i = 0; i < uv_kpt_ind.cols; ++i)
        {
            cv::Vec3f kpt = pos.at <cv::Vec3f>(uv_kpt_ind.at<float>(1, i) - 1, uv_kpt_ind.at<float>(0, i));
            kpts.push_back(cv::Point2f(kpt[0], kpt[1]));
            //cv::circle(img, cv::Point2i(kpt[0], kpt[1]),1,cv::Scalar(255,255,255), -1);           
        }

        std::vector<cv::Point3f> vertices3d;
        std::vector<cv::Vec3i> trianglesVec;

        for (int i = 0; i < face_ind.rows; i++)
        {
            cv::Vec3f kpt = pos.at <cv::Vec3f>(face_ind.at<float>(i));
            vertices3d.push_back(cv::Point3f(kpt[0], kpt[1], kpt[2]));
        }


        for (int i = 0; i < triangles.rows; i++)
        {
            trianglesVec.push_back(cv::Vec3i(triangles.at<float>(i, 0), triangles.at<float>(i, 1), triangles.at<float>(i, 2)));
        }
        MeshPlot mp;
        float scale = 4;
        cv::Mat tri = cv::Mat::zeros(img.rows * scale, img.cols * scale, CV_8UC3);
        cv::resize(face_img, tri, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::Mat mask = cv::Mat::zeros(256, 256, CV_8UC1);

        mp.Plot(tri, mask, face_ind, vertices3d, trianglesVec, scale, true, true);
        cv::resize(tri, tri, cv::Size(), 1.0 / scale, 1.0 / scale,cv::INTER_AREA);

        texture.setTo(cv::Scalar::all(0), ~mask);

        cv::imshow("texture", texture);

        cv::imshow("mask", mask);
        cv::imshow("tri", tri);
        cv::imwrite("FaceMesh.png", tri);
        plot_kpt(face_img, kpts);
        cv::imshow("img", face_img);
        cv::waitKey();
    }
    else
    {
        std::cout << "faces not detected" << std::endl;
    }
    return 0;
}

