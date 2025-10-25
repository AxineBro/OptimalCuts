// contour_debugger.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

using namespace std;
using namespace cv;

static Vec3b palette[] = {
    {0,0,255}, {0,128,255}, {0,255,255}, {0,255,128},
    {0,255,0}, {128,255,0}, {255,255,0}, {255,128,0},
    {255,0,0}, {255,0,128}, {200,100,255}, {180,180,180}
};

class ContourDebugger {
public:
    struct Result {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
    };

    // options: apply morphological opening/closing to remove noise
    static Result extractAndDebug(const Mat &binIn, bool applyMorph = true, int morphKernel = 3, const string &outDebugPath = "debug_contours.png") {
        Result res;
        if (binIn.empty() || binIn.type() != CV_8UC1) {
            cerr << "[ContourDebugger] Input image must be CV_8UC1 and non-empty\n";
            return res;
        }

        Mat bin = binIn.clone();

        // Ensure background is black and objects are white
        double whiteCount = countNonZero(bin);
        double total = bin.total();
        if (whiteCount == 0) {
            cerr << "[ContourDebugger] Warning: no white pixels found\n";
        } else if (whiteCount < total * 0.5) {
            // likely correct (objects are white)
        } else {
            // Too many whites: maybe inverted -> try invert and warn if needed
            // We won't auto-invert, but print hint
            cerr << "[ContourDebugger] Hint: more than 50% pixels are white. "
                 << "If background supposed to be black, consider inverting before call.\n";
        }

        if (applyMorph) {
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(morphKernel, morphKernel));
            // Open then close to remove small noise and fill small holes
            morphologyEx(bin, bin, MORPH_OPEN, kernel, Point(-1,-1), 1);
            morphologyEx(bin, bin, MORPH_CLOSE, kernel, Point(-1,-1), 1);
        }

        // Use RETR_TREE to have full parent-child hierarchy for nested holes
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        res.contours = contours;
        res.hierarchy = hierarchy;

        // Print summary
        size_t N = contours.size();
        cerr << "[ContourDebugger] Found contours: " << N << "\n";

        int externalCount = 0, holeCount = 0;
        for (size_t i = 0; i < N; ++i) {
            if (hierarchy[i][3] == -1) externalCount++;
            else holeCount++;
        }
        cerr << "[ContourDebugger] External contours: " << externalCount << ", holes: " << holeCount << "\n";

        // For each contour print brief info
        for (size_t i = 0; i < N; ++i) {
            const auto &c = contours[i];
            double area = contourArea(c);
            Rect bbox = boundingRect(c);
            Moments m = moments(c);
            Point2d centroid(m.m10 / (m.m00 + 1e-12), m.m01 / (m.m00 + 1e-12));
            int parent = hierarchy[i][3];
            int first_child = hierarchy[i][2];
            int next = hierarchy[i][0];
            int prev = hierarchy[i][1];

            cerr << "  [" << setw(3) << i << "] pts=" << setw(5) << c.size()
                 << " area=" << setw(10) << fixed << setprecision(2) << area
                 << " bbox=(" << bbox.x << "," << bbox.y << "," << bbox.width << "x" << bbox.height << ")"
                 << " center=(" << int(centroid.x) << "," << int(centroid.y) << ")"
                 << " parent=" << parent << " child=" << first_child << " next=" << next << " prev=" << prev
                 << "\n";

            // print first up to 6 points for quick inspection
            int toShow = (int)min<size_t>(6, c.size());
            cerr << "      pts0: ";
            for (int k = 0; k < toShow; ++k) {
                cerr << "(" << c[k].x << "," << c[k].y << ")";
                if (k+1 < toShow) cerr << ",";
            }
            if (c.size() > (size_t)toShow) cerr << " ...";
            cerr << "\n";
        }

        // Create debug visualization
        Mat debug;
        cvtColor(bin, debug, COLOR_GRAY2BGR);

        // Draw contours with labels, color by nesting level (approx)
        for (size_t i = 0; i < contours.size(); ++i) {
            // compute nesting level by walking parents
            int level = 0;
            int p = hierarchy[i][3];
            while (p != -1) {
                level++;
                p = hierarchy[p][3];
            }
            Vec3b col = palette[level % (sizeof(palette)/sizeof(palette[0]))];
            Scalar scol(col[0], col[1], col[2]);

            drawContours(debug, contours, (int)i, scol, 1);
            // draw index near centroid
            Moments mo = moments(contours[i]);
            Point2f cpt(0,0);
            if (mo.m00 != 0) cpt = Point2f((float)(mo.m10/mo.m00), (float)(mo.m01/mo.m00));
            else cpt = contours[i].front();
            putText(debug, to_string((int)i), cpt, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1, LINE_AA);
        }

        // Optionally draw parent-child arrows
        for (size_t i = 0; i < hierarchy.size(); ++i) {
            int p = hierarchy[i][3];
            if (p != -1 && !contours[i].empty() && !contours[p].empty()) {
                Moments mi = moments(contours[i]);
                Moments mp = moments(contours[p]);
                Point2f ci(mi.m10/(mi.m00+1e-12), mi.m01/(mi.m00+1e-12));
                Point2f cp(mp.m10/(mp.m00+1e-12), mp.m01/(mp.m00+1e-12));
                arrowedLine(debug, cp, ci, Scalar(200,200,200), 1, LINE_AA, 0, 0.1);
            }
        }

        // Save debug image
        bool ok = imwrite(outDebugPath, debug);
        if (!ok) {
            cerr << "[ContourDebugger] Failed to save debug image to " << outDebugPath << "\n";
        } else {
            cerr << "[ContourDebugger] Debug image saved to " << outDebugPath << "\n";
        }

        return res;
    }
};

// Simple CLI for testing
int main(int argc, char *argv[]) {

    string path = "Z:/20.png";
    string outDebug = "Z:/debug_contours.png";

    Mat img = imread(path, IMREAD_UNCHANGED);
    if (img.empty()) {
        cerr << "Failed to open " << path << "\n";
        return 2;
    }

    Mat gray;
    if (img.channels() > 1) cvtColor(img, gray, COLOR_BGR2GRAY);
    else gray = img;

    // Binarize with Otsu — better than fixed threshold in general
    Mat bin;
    double thr = threshold(gray, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cerr << "[Main] Otsu threshold = " << thr << "\n";

    // If background is white and objects black, invert
    // Heuristic: if number of white pixels > number of black pixels, assume inverted
    double white = countNonZero(bin);
    if (white > 0.5 * bin.total()) {
        // Objects might be black; invert to make objects white
        cerr << "[Main] Too many white pixels (" << white << ") -> inverting binary image to make objects white\n";
        bitwise_not(bin, bin);
    }

    // Run debugger
    auto res = ContourDebugger::extractAndDebug(bin, true, 3, outDebug);

    cerr << "[Main] Done. Found " << res.contours.size() << " contours. Debug image: " << outDebug << "\n";
    return 0;
}
