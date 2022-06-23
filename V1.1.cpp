#include "iostream"
#include <math.h>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

bool RUN_TYPE = 0;//运行模式——0:正常运行  1:调试

template<typename T1>//计算cv::Point A, B两点间的距离
constexpr auto DIS(T1 A, T1  B) { return sqrt(pow((A.x - B.x), 2) + pow((A.y - B.y), 2)); }
template<typename S1>//计算cv::Point A, B两点间的距离
constexpr auto DIS2(S1 A, S1  B) { return (pow((A.x - B.x), 2) + pow((A.y - B.y), 2)); }
template<typename K1>//计算cv::Point A, B两点连线的斜率
constexpr auto Slope(K1 A, K1 B) { return (float)(A.y - B.y) / (float)(A.x - B.x); }
template<typename C1>//计算cv::Point p1, p2, p3, p4四点的中心点
constexpr auto CenPt(C1 p1, C1 p2, C1 p3, C1 p4) {
    float x = ((float)(p3.y - Slope(p3, p4) * p3.x) - (float)(p1.y - Slope(p1, p2) * p1.x)) / (Slope(p1, p2) - Slope(p3, p4));
    float y = Slope(p1, p2) * x + (float)(p1.y - Slope(p1, p2) * p1.x);
    return cv::Point((int)x, (int)y);
}

std::vector<std::vector<cv::Point>> processs(cv::VideoCapture cap, cv::Mat& src_out, std::vector<double>& ANGLE);
std::vector<cv::Point> matching(std::vector<std::vector<cv::Point>> EDGES, std::vector<std::vector<int>> INDEX);
std::vector<cv::Point> refitting(cv::Mat src, std::vector<std::vector<cv::Point>> EDGES, std::vector<cv::Point> CENTRE);

/*四边形角块筛选函数*/double SQU_JUI(std::vector<cv::Point> Pt);
/*六边形角块筛选函数*/double POL_JUI(std::vector<cv::Point> Pt);
/*平面筛选函数*/double PLA_JUI(std::vector<cv::Point> Pt);
/*角点判断函数*/std::vector<std::vector<cv::Point>> EDGE_JUI(cv::Mat src, std::vector<std::vector<cv::Point>> EDGES_BE, std::vector<std::vector<cv::Point>>& EDGES_UN);
/*排列组合函数*/std::vector<std::vector<int>> ReChoose(std::vector<std::vector<cv::Point>> EDGES);
/*角点重排函数*/std::vector<cv::Point> Reline(std::vector<cv::Point> pt);
/*模型计算*/float PIX2MM(cv::Mat src, std::vector<cv::Point> Refit, cv::Point centre);

int main() {
    cv::VideoCapture cap("004.mp4");
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cv::Mat src_cp;
    std::vector<cv::Point> CENTRE, ReCENTRE;
    std::vector<std::vector<int>> INDEX;
    std::vector<std::vector<cv::Point>> EDGES, EDGES_UN;

    while (true) {
        std::vector<double> angles;
        std::vector<std::vector<cv::Point>> EDGES_BE = EDGES_UN;
        EDGES_UN = processs(cap, src_cp, angles);
        EDGES = EDGE_JUI(src_cp, EDGES_BE, EDGES_UN);

        for (int i = 0; i < EDGES.size(); i++)
            cv::drawContours(src_cp, EDGES, i, cv::Scalar(100, 255, 255), 3);

        INDEX = ReChoose(EDGES);
        if (INDEX.size() > 0) {
            CENTRE = matching(EDGES, INDEX);
        }
        else std::cout << "未拟合到足够的角块！" << std::endl;
        if (CENTRE.size() == 4) ReCENTRE = refitting(src_cp, EDGES, CENTRE);
        else std::cout << "未匹配到面！" << std::endl;

        float DISTANS = 0;
        if (ReCENTRE.size() == 4) {
            for (int i = 0; i < 4; i++)
                cv::line(src_cp, ReCENTRE[i], ReCENTRE[(i + 1) % 4], cv::Scalar(100, 250, 50), 4);
            cv::Point RECENTRE = CenPt(ReCENTRE[0], ReCENTRE[2], ReCENTRE[1], ReCENTRE[3]);
            DISTANS = PIX2MM(src_cp, ReCENTRE, RECENTRE);
        }

        if (DISTANS > -160 && DISTANS < 160) {
                ///在此处加入窗口通信，发送DISTANS

        }
        cv::imshow("001", src_cp);
        cv::waitKey(1);
    }
    return 0;
}

/*四边形角块筛选函数*/double SQU_JUI(std::vector<cv::Point> Pt) {
    float sl[4];
    double angle[4], angle_o[4], angle_min = 180;
    for (int i = 0; i < 4; i++) {
        angle[i] = acos((DIS2(Pt[i], Pt[(i + 1) % 4]) + DIS2(Pt[i], Pt[(i + 3) % 4]) - DIS2(Pt[(i + 1) % 4], Pt[(i + 3) % 4])) / (2 * DIS(Pt[i], Pt[(i + 1) % 4]) * DIS(Pt[i], Pt[(i + 3) % 4]))) * 57.3;
        angle_o[i] = acos((Pt[i].y - Pt[(i + 1) % 4].y) / DIS(Pt[i], Pt[(i + 1) % 4])) * 57.3;
        sl[i] = DIS(Pt[i], Pt[(i + 1) % 4]);
    }
    for (int i = 0; i < 4; i++)
        if (angle_o[i] < angle_min)
            angle_min = angle_o[i];
    for (int i = 0; i < 4; i++)
        if (abs(angle[i] + angle[(1 + i) % 4] - 180) > 15 || abs(angle[i] + angle[(3 + i) % 4] - 180) > 15 || angle[i] > 145 || angle[i] < 35)
            return 0;
    for (int i = 0; i < 4; i++)
        if (sl[i] > sl[(i + 1) % 4] * 4 || sl[i] < sl[(i + 1) % 4] / 4 || sl[i] < 10)
            return 0;
    return angle_min + 1;//返回结果+1，即成功判定后最小值为1，小于1则没有通过判定
}
/*六边形角块筛选函数*/double POL_JUI(std::vector<cv::Point> Pt) {
    float di_Dis[3];
    for (int i = 0; i < 3; i++) {
        di_Dis[i] = DIS(Pt[i], Pt[i + 3]);
        if (di_Dis[i] > 300)
            return 0;
    }
    double angle[6], angle_o[6], angle_min = 180;
    for (int i = 0; i < 6; i++) {
        angle[i] = acos((DIS2(Pt[i], Pt[(i + 1) % 6]) + DIS2(Pt[i], Pt[(i + 5) % 6]) - DIS2(Pt[(i + 1) % 6], Pt[(i + 5) % 6])) / (2 * DIS(Pt[i], Pt[(i + 1) % 6]) * DIS(Pt[i], Pt[(i + 5) % 6]))) * 57.3;
        angle_o[i] = acos((Pt[i].y - Pt[(i + 1) % 6].y) / DIS(Pt[i], Pt[(i + 1) % 6])) * 57.3;
    }
    for (int i = 0; i < 6; i++)
        if (angle_o[i] < angle_min)
            angle_min = angle_o[i];
    for (int i = 0; i < 6; i++)
        if (angle[i] > 150 || angle[i] < 30)
            return 0;
    return angle_min + 1;//返回结果+1，即成功判定后最小值为1，小于1则没有通过判定
}
/*平面筛选函数*/double PLA_JUI(std::vector<cv::Point> Pt) {
    float sl[4];
    int SL_Sim_ti = 0;
    float slope[2];

    double angle[4];
    for (int i = 0; i < 4; i++) {
        angle[i] = acos((DIS2(Pt[i], Pt[(i + 1) % 4]) + DIS2(Pt[i], Pt[(i + 3) % 4]) - DIS2(Pt[(i + 1) % 4], Pt[(i + 3) % 4])) / (2 * DIS(Pt[i], Pt[(i + 1) % 4]) * DIS(Pt[i], Pt[(i + 3) % 4]))) * 57.3;
        //cv::putText(src, std::to_string((int)angle[i]), Pt[i], 0, 1, cv::Scalar(255, 100, 255), 5);
    }
    for (int i = 0; i < 4; i++)
        if (angle[i] > 145)
            return 0;
    for (int i = 0; i < 2; i++)
        if (abs(angle[i] + angle[(1 + i) % 4] - 180) < 15 && abs(angle[(2 + i) % 4] + angle[(3 + i) % 4] - 180) < 15)
                return abs(angle[i] + angle[(1 + i) % 4] - 180) + abs(angle[(2 + i) % 4] + angle[(3 + i) % 4] - 180) + 1;//返回结果+1，即成功判定后最小值为1，小于1则没有通过判定

    return 0;
}
/*角点判断函数*/std::vector<std::vector<cv::Point>> EDGE_JUI(cv::Mat src, std::vector<std::vector<cv::Point>> EDGES_BE, std::vector<std::vector<cv::Point>>& EDGES_UN) {
    std::vector<std::vector<cv::Point>> OUT;
    std::vector<std::vector<int>> index;
    int times = 0;
    for (int a = 0; a < EDGES_BE.size(); a++)
        for (int b = 0; b < EDGES_UN.size(); b++)
            if (DIS(EDGES_BE[a][0], EDGES_UN[b][0]) < 75)
                OUT.push_back(EDGES_UN[b]);
    if (OUT.size() <= EDGES_BE.size()) {
        for (int a = 0; a < EDGES_BE.size(); a++) {
            for (int b = 0; b < OUT.size(); b++)
                if (DIS(EDGES_BE[a][0], OUT[b][0]) < 25) 
                    times = 1;
            if (times != 1) {
                OUT.push_back(EDGES_BE[a]);
                EDGES_UN.push_back(EDGES_BE[a]);
            }
            times = 0;
        }
    }
    return OUT;
}
/*排列组合函数*/std::vector<std::vector<int>> ReChoose(std::vector<std::vector<cv::Point>> EDGES) {
    std::vector<std::vector<int>> OUT;
    std::vector<int> SQU, POL;
    for (int i = 0; i < EDGES.size(); i++)
        if (EDGES[i].size() == 5)
            SQU.push_back(i);
        else if (EDGES[i].size() == 7)
            POL.push_back(i);
    if (SQU.size() > 1 && POL.size() > 1)
        for (int a = 0; a < SQU.size() - 1; a++)
            for (int b = a + 1; b < SQU.size(); b++)
                for (int c = 0; c < POL.size() - 1; c++)
                    for (int d = c + 1; d < POL.size(); d++)
                        OUT.push_back(std::vector<int>{SQU[a], SQU[b], POL[c], POL[d]});
    if (SQU.size() > 0 && POL.size() > 2)
        for (int a = 0; a < SQU.size(); a++)
            for (int b = 0; b < POL.size() - 2; b++)
                for (int c = b+ 1; c < POL.size() - 1; c++)
                    for (int d = c + 1; d < POL.size(); d++)
                        OUT.push_back(std::vector<int>{SQU[a], POL[b], POL[c], POL[d]});
    return OUT;
}
/*角点重排函数*/std::vector<cv::Point> Reline(std::vector<cv::Point> pt) {
    cv::Point centre = (pt[0] + pt[1] + pt[2] + pt[3]) / 4;
    float K[3] = { abs(Slope(centre, pt[1]) - Slope(centre, pt[0])), abs(Slope(centre, pt[2]) - Slope(centre, pt[0])), abs(Slope(centre, pt[3]) - Slope(centre, pt[0])) };
    float min = K[0];
    int MIN = 0;
    for (int i = 1; i < 3; i++)
        if (K[i] < min) {
            MIN = i;
            min = K[i];
        }
    std::vector<cv::Point> Ptout = { pt[0], pt[(MIN + 1) % 3 + 1], pt[MIN + 1], pt[(MIN + 2) % 3 + 1] };
    return Ptout;
}
/*模型计算*/float PIX2MM(cv::Mat src, std::vector<cv::Point> Refit, cv::Point centre) {
    float K = 0.0, dis = 0.0;
    for (int i = 0; i < 4; i++)
        if (abs((acos((Refit[i].y - Refit[(i + 1) % 4].y) / DIS(Refit[i], Refit[(i + 1) % 4])) * 57.3) - 90) < 30) {
            K += Slope(Refit[i], Refit[(i + 1) % 4]) / 2;
            dis += DIS(Refit[i], Refit[(i + 1) % 4]) / 2;
        }
    cv::Point MID = cv::Point(960, centre.y - K * (centre.x - 960));
    cv::line(src, centre, MID, cv::Scalar(0, 255, 0), 4);
    cv::circle(src, centre, 10, cv::Scalar(100, 255, 100), cv::FILLED);
    cv::circle(src, MID, 10, cv::Scalar(100, 255, 100), cv::FILLED);
    float dis_act = 150 * DIS(centre, MID) / dis;
    if (centre.x < 960)
        dis_act = dis_act * (-1);
    cv::putText(src, std::to_string((int)dis_act), MID, 0, 1, cv::Scalar(255, 255, 255), 3);
    return dis_act;
}

std::vector<std::vector<cv::Point>> processs(cv::VideoCapture cap, cv::Mat& src_out, std::vector<double>& ANGLE) {
    cv::Mat src, src_cut,src_cp, src_HSV, src_L, src_bi, src_cn, src_ps;
    cap.read(src);
    src_out = src.clone();
    src_cut = src(cv::Rect(0, 0, 1920, 620));
    src_cp = src_cut.clone();
    cvtColor(src_cut, src_HSV, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> src_3C;
    cv::split(src_HSV, src_3C);
    src_L = src_3C.at(2);
    cv::threshold(src_L, src_bi, 175, 255, cv::THRESH_BINARY);
    Canny(src_bi, src_cn, 25, 75);
    dilate(src_cn, src_ps, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(src_ps, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> conPoly(contours.size());
    std::vector<std::vector<cv::Point>> OUT;
    std::vector<double> angle;
    for (int i = 0; i < contours.size(); i++) {
        float peri = arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], conPoly[i], 0.024 * peri, true);
        int area = contourArea(contours[i]);
        int edge = conPoly[i].size();
        if (area > 800 && area < 15000 && hierarchy[i][2] != -1) 
            if (edge == 4) {
                double angle_o = SQU_JUI(conPoly[i]); 
                if (angle_o > 0.5) {
                    angle.push_back(angle_o);
                    std::vector<cv::Point> out;
                    out.push_back(CenPt(conPoly[i][0], conPoly[i][2], conPoly[i][1], conPoly[i][3]));
                    for (int k = 0; k < 4; k++)
                        out.push_back(conPoly[i][k]);
                    OUT.push_back(out);
                }
            }
            else if (edge == 6) {
                double angle_o = POL_JUI(conPoly[i]);
                if (angle_o > 0.5) {
                    angle.push_back(angle_o);
                    std::vector<cv::Point> out;
                    out.push_back(CenPt(conPoly[i][0], conPoly[i][3], conPoly[i][1], conPoly[i][4]));
                    for (int k = 0; k < 6; k++)
                        out.push_back(conPoly[i][k]);
                    OUT.push_back(out);
                }
            }
    }
    ANGLE = angle;
    return OUT;
}
std::vector<cv::Point> matching(std::vector<std::vector<cv::Point>> EDGES, std::vector<std::vector<int>> INDEX) {
    std::vector<cv::Point> Choose_o, Choose, out;
    double min = 360;
    int min_rank = 0;
    for (int i = 0; i < INDEX.size(); i++) {
        for (int k = 0; k < 4; k++)
            Choose_o.push_back(EDGES[INDEX[i][k]][0]);
        Choose = Reline(Choose_o);
        if (PLA_JUI(Choose) > 0.5 && PLA_JUI(Choose) < min) {
            min = PLA_JUI(Choose);
            out = Choose;
        }
    }
    return out;
}
std::vector<cv::Point> refitting(cv::Mat src, std::vector<std::vector<cv::Point>> EDGES, std::vector<cv::Point> CENTRE) {
    cv::Point centre = CenPt(CENTRE[0], CENTRE[2], CENTRE[1], CENTRE[3]);
    std::vector<cv::Point> OUT;
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < EDGES.size(); k++)
            if (CENTRE[i] == EDGES[k][0]) {
                std::vector<float> K;
                for (int z = 1; z < EDGES[k].size(); z++) 
                    K.push_back(acos((EDGES[k][z].y - centre.y) / DIS(EDGES[k][z], centre)) * 57.3);
                double min = 180;
                int p1 = 0, p2 = 0;
                for (int a = 0; a < EDGES[k].size() - 2; a++)
                    for (int b = a + 1; b < EDGES[k].size() - 1; b++)
                        if (abs(K[a] - K[b]) < min) {
                            min = abs(K[a] - K[b]);
                            p1 = a;
                            p2 = b;
                        }
                if (DIS(EDGES[k][p1 + 1], centre) > DIS(EDGES[k][p2 + 1], centre))
                    OUT.push_back(EDGES[k][p1 + 1]);
                else
                    OUT.push_back(EDGES[k][p2 + 1]);
            }
    return OUT;
}