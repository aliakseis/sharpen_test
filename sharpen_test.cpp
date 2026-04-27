#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

using namespace cv;

//============================================================
// Helpers
//============================================================
static inline void normalizeToUnitSum(Mat& m)
{
    Scalar s = sum(m);
    if (std::abs((float)s[0]) > 1e-12f)
        m /= (float)s[0];
}

static inline void safeNormalizeMinMax(Mat& m)
{
    double mn, mx;
    minMaxLoc(m, &mn, &mx);
    const double d = mx - mn;
    if (d > 1e-12)
        m = (m - mn) / d;
    else
        m.setTo(0);
}

//============================================================
// Cached radial weighting for anomaly suppression
//============================================================
static Mat getRadialMidWeight(Size sz)
{
    static Size cachedSize;
    static Mat cached;

    if (cachedSize == sz && !cached.empty())
        return cached;

    cachedSize = sz;
    cached.create(sz, CV_32F);

    const int rows = sz.height;
    const int cols = sz.width;

    const float cx = cols * 0.5f;
    const float cy = rows * 0.5f;
    const float Rmax = std::min(cx, cy);

    for (int y = 0; y < rows; ++y)
    {
        float* row = cached.ptr<float>(y);
        const float dy = y - cy;

        for (int x = 0; x < cols; ++x)
        {
            const float dx = x - cx;
            const float r = std::sqrt(dx * dx + dy * dy) / Rmax;

            row[x] = std::exp(
                -0.5f * (r - 0.42f) * (r - 0.42f) / (0.18f * 0.18f));
        }
    }

    return cached;
}

//============================================================
// FFT shift
//============================================================
static void shiftPSF(const Mat& src, Mat& dst)
{
    dst.create(src.size(), src.type());

    const int cx = src.cols / 2;
    const int cy = src.rows / 2;

    src(Rect(cx, cy, src.cols - cx, src.rows - cy))
        .copyTo(dst(Rect(0, 0, src.cols - cx, src.rows - cy)));

    src(Rect(0, cy, cx, src.rows - cy))
        .copyTo(dst(Rect(src.cols - cx, 0, cx, src.rows - cy)));

    src(Rect(cx, 0, src.cols - cx, cy))
        .copyTo(dst(Rect(0, src.rows - cy, src.cols - cx, cy)));

    src(Rect(0, 0, cx, cy))
        .copyTo(dst(Rect(src.cols - cx, src.rows - cy, cx, cy)));
}

typedef std::array<uint32_t, 256> TopKMean;

//============================================================
// Single-pass industrial computeMaxDiffMatrix
//============================================================
Mat computeMaxDiffMatrix(const Mat& gray, int radius)
{
    CV_Assert(gray.type() == CV_8U);

    const int K = radius * 2 + 1;
    const int border = 2;
    const int nth = std::max(1, (gray.rows * gray.cols) / 100);

    std::vector<TopKMean> bins(K * K);
    //bins.reserve(K * K);
    //for (int i = 0; i < K * K; ++i)
    //    bins.emplace_back(nth);

    const int yStart = border + radius;
    const int yEnd = gray.rows - border - radius;
    const int xStart = border + radius;
    const int xEnd = gray.cols - border - radius;

    for (int y = yStart; y < yEnd; ++y)
    {
        std::vector<const uchar*> nbrRows(K);
        for (int j = -radius; j <= radius; ++j)
            nbrRows[j + radius] = gray.ptr<uchar>(y + j);

        const uchar* centerRow = gray.ptr<uchar>(y);

        for (int x = xStart; x < xEnd; ++x)
        {
            const uchar c = centerRow[x];

            for (int dy = 0; dy <= radius; ++dy)
            {
                const uchar* rowPos = nbrRows[dy + radius];
                const int dx0 = (dy == 0) ? 1 : -radius;

                for (int dx = dx0; dx <= radius; ++dx)
                {
                    const int d = std::abs(c - rowPos[x + dx]);

                    const int idx1 = (dy + radius) * K + (dx + radius);
                    const int idx2 = (-dy + radius) * K + (-dx + radius);

                    ++bins[idx1][d];
                    ++bins[idx2][d];
                }
            }
        }
    }

    Mat M(K, K, CV_32F);
    for (int y = 0; y < K; ++y)
    {
        float* row = M.ptr<float>(y);
        for (int x = 0; x < K; ++x)
        {
            const auto& bin = bins[y * K + x];
            float sum = 0.0f;
            int count = 0;
            for (int i = 255; i >= 0; --i)
            {
                if (count + bin[i] >= nth)
                {
                    count = nth;
                    sum += (nth - count) * i;
                    break;
                }
                else {
                    count += bin[i];
                    sum += bin[i] * i;
                }
            }
            row[x] = (count > 0) ? (sum / count) : 0.0f;
        }
    }

    double mn, mx;
    minMaxLoc(M, &mn, &mx);

    Mat P;
    if (mx - mn > 1e-12)
        P = (mx - M) / (mx - mn);
    else
        P = Mat::zeros(M.size(), CV_32F);

    normalizeToUnitSum(P);
    return P;
}

//============================================================
Mat clipPSFByHeap(const Mat& P, float fraction = 0.8f)
{
    std::vector<float> vals;
    vals.reserve(P.total());

    for (int y = 0; y < P.rows; ++y)
    {
        const float* row = P.ptr<float>(y);
        vals.insert(vals.end(), row, row + P.cols);
    }

    const int k = std::max(1, int(vals.size() * fraction));
    std::nth_element(vals.begin(), vals.begin() + k - 1, vals.end());
    const float threshold = vals[k - 1];

    Mat out = P.clone();

    for (int y = 0; y < out.rows; ++y)
    {
        float* row = out.ptr<float>(y);
        for (int x = 0; x < out.cols; ++x)
            row[x] = std::max(0.0f, row[x] - threshold);
    }

    return out;
}

//============================================================
Mat cropPSFToActiveRegionAndFixOdd(const Mat& psf)
{
    int top = 0, bottom = psf.rows - 1;
    int left = 0, right = psf.cols - 1;

    while (top <= bottom)
    {
        bool nz = false;
        const float* r1 = psf.ptr<float>(top);
        const float* r2 = psf.ptr<float>(bottom);

        for (int x = 0; x < psf.cols; ++x)
            if (r1[x] != 0.0f || r2[x] != 0.0f) { nz = true; break; }

        if (nz) break;
        ++top; --bottom;
    }

    while (left <= right)
    {
        bool nz = false;
        for (int y = top; y <= bottom; ++y)
        {
            const float* row = psf.ptr<float>(y);
            if (row[left] != 0.0f || row[right] != 0.0f) { nz = true; break; }
        }

        if (nz) break;
        ++left; --right;
    }

    if (top > bottom || left > right)
        return psf.clone();

    return psf(Rect(left, top, right - left + 1, bottom - top + 1)).clone();
}

//============================================================
Mat computeCorrelationFFT(const Mat& gray, int radius)
{
    Mat f32;
    gray.convertTo(f32, CV_32F);
    f32 -= mean(f32)[0];

    Mat gx, gy;
    Sobel(f32, gx, CV_32F, 1, 0, 3);
    Sobel(f32, gy, CV_32F, 0, 1, 3);

    magnitude(gx, gy, f32);
    threshold(f32, f32, 10.0f, 0.0f, THRESH_TOZERO);

    Mat F;
    {
        Mat planes[] = { f32, Mat::zeros(f32.size(), CV_32F) };
        merge(planes, 2, F);
        dft(F, F, DFT_COMPLEX_OUTPUT);
    }

    Mat Fc;
    mulSpectrums(F, F, Fc, 0, true);

    Mat C;
    idft(Fc, C, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat shifted;
    shiftPSF(C, shifted);

    const int cx = shifted.cols / 2;
    const int cy = shifted.rows / 2;

    Mat cropped = shifted(Rect(cx - radius, cy - radius, 2 * radius + 1, 2 * radius + 1)).clone();

    safeNormalizeMinMax(cropped);
    normalizeToUnitSum(cropped);

    return cropped;
}

//============================================================
Mat buildPSFFromM(const Mat& M)
{
    Mat P = clipPSFByHeap(M);
    P = cropPSFToActiveRegionAndFixOdd(P);
    normalizeToUnitSum(P);
    return P;
}

//============================================================
Mat buildInverseFilterFromPSF(const Mat& psfSmall, Size imgSize, float K = 0.01f)
{
    Mat psfPadded(imgSize, CV_32F, Scalar(0));

    const int x0 = (imgSize.width - psfSmall.cols) / 2;
    const int y0 = (imgSize.height - psfSmall.rows) / 2;

    psfSmall.copyTo(psfPadded(Rect(x0, y0, psfSmall.cols, psfSmall.rows)));

    Mat psfShifted;
    shiftPSF(psfPadded, psfShifted);

    Mat H;
    Mat planesH[] = { psfShifted, Mat::zeros(imgSize, CV_32F) };
    merge(planesH, 2, H);
    dft(H, H, DFT_COMPLEX_OUTPUT);

    Mat planes[2];
    split(H, planes);

    Mat& Re = planes[0];
    Mat& Im = planes[1];

    Mat mag2 = Re.mul(Re);
    mag2 += Im.mul(Im);

    const float adaptiveK = K * (float)(mean(mag2)[0] + 1e-8f);
    Mat denom = mag2 + adaptiveK;

    Re /= denom;
    Im /= denom;
    Im = -Im;

    Mat G;
    Mat outp[] = { Re, Im };
    merge(outp, 2, G);
    return G;
}

//============================================================
static void DeQuantization(Mat* planes)
{
    Mat& Re = planes[0];
    Mat& Im = planes[1];

    const float dcRe = Re.at<float>(0, 0);
    const float dcIm = Im.at<float>(0, 0);

    const float sigma2_axis =
        (1.0f / 12.0f) * static_cast<float>(Re.total()) * 0.5f;

    for (int y = 0; y < Re.rows; ++y)
    {
        float* re = Re.ptr<float>(y);
        float* im = Im.ptr<float>(y);

        for (int x = 0; x < Re.cols; ++x)
        {
            const float mag2 = re[x] * re[x] + im[x] * im[x] + 1e-12f;
            float alpha = 1.0f - sigma2_axis / mag2;
            if (alpha < 0.0f) alpha = 0.0f;

            re[x] *= alpha;
            im[x] *= alpha;
        }
    }

    Re.at<float>(0, 0) = dcRe;
    Im.at<float>(0, 0) = dcIm;
}

//============================================================
static void AnomalySuppression(Mat* cpl)
{
    Mat& Re = cpl[0];
    Mat& Im = cpl[1];

    Mat work;
    magnitude(Re, Im, work);
    work += 1.0f;
    log(work, work);

    Mat baseline;
    GaussianBlur(work, baseline, Size(0, 0), 18.0, 18.0, BORDER_REFLECT);

    work -= baseline;

    Scalar mu, sd;
    meanStdDev(work, mu, sd);

    const float meanv = (float)mu[0];
    const float stdv = std::max(0.001f, (float)sd[0]);

    const int rows = work.rows;
    const int cols = work.cols;

    for (int y = 0; y < rows; ++y)
    {
        float* wp = work.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
        {
            const float z = (wp[x] - meanv) / stdv;
            wp[x] = (z < 1.5f) ? 0.0f : std::min(1.0f, (z - 1.5f) / 2.5f);
        }
    }

    GaussianBlur(work, work, Size(0, 0), 3.5, 3.5, BORDER_REFLECT);

    const Mat radial = getRadialMidWeight(work.size());

    for (int y = 0; y < rows; ++y)
    {
        float* wp = work.ptr<float>(y);
        const float* rp = radial.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
            wp[x] = 1.0f - 0.30f * wp[x] * rp[x];
    }

    GaussianBlur(work, work, Size(0, 0), 2.0, 2.0, BORDER_REFLECT);

    for (int y = 0; y < rows; ++y)
    {
        const float* mp = work.ptr<float>(y);
        float* re = Re.ptr<float>(y);
        float* im = Im.ptr<float>(y);

        for (int x = 0; x < cols; ++x)
        {
            const float m = mp[x];
            re[x] *= m;
            im[x] *= m;
        }
    }
}

//============================================================
Mat applyFilterDFT(const Mat& F, const Mat& G)
{
    Mat Y;
    mulSpectrums(F, G, Y, 0);

    Mat out32;
    idft(Y, out32, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat out8;
    out32.convertTo(out8, CV_8U);
    return out8;
}

//============================================================
Mat deblurChannel(const Mat& gray)
{
    Mat Y;
    gray.convertTo(Y, CV_32F);

    const int radius = 15;

    Mat M = computeMaxDiffMatrix(gray, radius) + computeCorrelationFFT(Y, radius);
    Mat psf = buildPSFFromM(M);
    Mat G = buildInverseFilterFromPSF(psf, Y.size(), 0.1f);

    Mat F;
    {
        Mat planes0[] = { Y, Mat::zeros(Y.size(), CV_32F) };
        merge(planes0, 2, F);
        dft(F, F, DFT_COMPLEX_OUTPUT);
    }

    Mat planes[2];
    split(F, planes);

    DeQuantization(planes);
    AnomalySuppression(planes);

    merge(planes, 2, F);

    return applyFilterDFT(F, G);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <image_path> <output_path>\n";
        return -1;
    }

    Mat img = imread(argv[1]);
    if (img.empty()) return -1;

    // BGR → YCrCb
    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    std::vector<Mat> ch;
    split(ycrcb, ch);

    Mat Cr = ch[1];
    Mat Cb = ch[2];

    auto start = std::chrono::high_resolution_clock::now();
    ch[0] = deblurChannel(ch[0]);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << " seconds\n";

    merge(ch, ycrcb);

    Mat restoredBGR;
    cvtColor(ycrcb, restoredBGR, COLOR_YCrCb2BGR);

    imwrite(argv[2], restoredBGR);
    if (argc > 3)
        imwrite(argv[3], ch[0]);

    return 0;
}
