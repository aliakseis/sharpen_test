#include <opencv2/opencv.hpp>

#include <queue>
#include <vector>
#include <functional>

using namespace cv;

Mat computeMaxDiffMatrix(const Mat& gray, int radius)
{
    CV_Assert(gray.type() == CV_8U);
    const int K = 2 * radius + 1;

    const int border = 2; // отступ от краёв для безопасного доступа к пикселям

    const int num_vals = 30;

    int nth = (gray.rows + gray.cols) / 4 + num_vals; // примерное значение, можно настроить

    Mat M(K, K, CV_32F, Scalar(0));

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            // min-heap для хранения top-n максимумов
            std::priority_queue<
                float,
                std::vector<float>,
                std::greater<float>
            > heap;

            // --- отступ от краёв ---
            int yStart = std::max(border, -dy);
            int yEnd = std::min(gray.rows - border, gray.rows - dy);

            int xStart = std::max(border, -dx);
            int xEnd = std::min(gray.cols - border, gray.cols - dx);

            for (int y = yStart; y < yEnd; ++y)
            {
                const uchar* row1 = gray.ptr<uchar>(y);
                const uchar* row2 = gray.ptr<uchar>(y + dy);

                for (int x = xStart; x < xEnd; ++x)
                {
                    float d = float(row1[x]) - float(row2[x + dx]);
                    if (d > 0)
                    {
                        // добавляем в heap
                        if ((int)heap.size() < nth)
                        {
                            heap.push(d);
                        }
                        else if (d > heap.top())
                        {
                            heap.pop();
                            heap.push(d);
                        }
                    }
                }
            }

            if (heap.size() < nth)
                std::cerr << "Warning: only " << heap.size() << " values for offset (" << dx << "," << dy << ")\n";

            int count = 0;
            float nthMax = 0.0f;
            while (count < num_vals && !heap.empty())
            {
                nthMax += heap.top();
                heap.pop();
                count++;
            }
            M.at<float>(dy + radius, dx + radius) = (count == 0) ? 0.0f : (nthMax / count);
        }
    }

    return M;
}

Mat applyRadialHann(const Mat& P)
{
    int h = P.rows, w = P.cols;
    int cx = w / 2, cy = h / 2;
    float R = std::min(cx, cy);

    Mat W(P.size(), CV_32F);

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float dx = x - cx;
            float dy = y - cy;
            float r = std::sqrt(dx * dx + dy * dy);

            float t = std::min(1.0f, r / R);
            float wv = 0.5f - 0.5f * std::cos(2 * CV_PI * (1 - t));
            W.at<float>(y, x) = wv;
        }
    }

    Mat out;
    multiply(P, W, out);
    return out;
}

Mat clipPSFByHeap(const Mat& P, float fraction = 0.8f)
{
    CV_Assert(P.type() == CV_32F);

    const int total = P.rows * P.cols;
    const int k = std::max(1, int(total * fraction)); // сколько элементов считаем "краевыми"

    // --- min-heap для поиска k-го минимального ---
    std::priority_queue<
        float,
        std::vector<float>,
        std::less<float> // max-heap, но мы храним k минимальных
    > heap;

    for (int y = 0; y < P.rows; y++)
    {
        const float* row = P.ptr<float>(y);
        for (int x = 0; x < P.cols; x++)
        {
            float v = row[x];

            if ((int)heap.size() < k)
            {
                heap.push(v);
            }
            else if (v < heap.top())
            {
                heap.pop();
                heap.push(v);
            }
        }
    }

    // k-й минимальный элемент — это порог
    const float threshold = heap.top();

    // --- Сдвигаем и обнуляем ---
    Mat out = P.clone();
    for (int y = 0; y < out.rows; y++)
    {
        float* row = out.ptr<float>(y);
        for (int x = 0; x < out.cols; x++)
        {
            float v = row[x] - threshold;
            row[x] = (v > 0.0f ? v : 0.0f);
        }
    }

    return out;
}


Mat cropPSFToActiveRegionAndFixOdd(const Mat& psf)
{
    CV_Assert(psf.type() == CV_32F);

    int top = 0, bottom = psf.rows - 1;
    int left = 0, right = psf.cols - 1;

    // --- верх ---
    while (top <= bottom)
    {
        bool nonZero = false;
        const float* row = psf.ptr<float>(top);
        for (int x = 0; x < psf.cols; x++)
            if (row[x] != 0.0f) { nonZero = true; break; }
        if (nonZero) break;

        row = psf.ptr<float>(bottom);
        for (int x = 0; x < psf.cols; x++)
            if (row[x] != 0.0f) { nonZero = true; break; }
        if (nonZero) break;

        top++;
        bottom--;
    }

    // --- лево ---
    while (left <= right)
    {
        bool nonZero = false;
        for (int y = top; y <= bottom; y++)
            if (psf.at<float>(y, left) != 0.0f) { nonZero = true; break; }
        if (nonZero) break;

        for (int y = top; y <= bottom; y++)
            if (psf.at<float>(y, right) != 0.0f) { nonZero = true; break; }
        if (nonZero) break;

        left++;
        right--;
    }

    // Если всё нули — возвращаем копию
    if (top > bottom || left > right)
        return psf.clone();

    // --- Вырезаем активную область ---
    Mat cropped = psf(Rect(left, top, right - left + 1, bottom - top + 1)).clone();

    return cropped;
}


//---------------------------------------------
// 2. PSF из матрицы M: PSF ≈ max(M) - M, нормировка
//---------------------------------------------
Mat buildPSFFromM(const Mat& M)
{
    CV_Assert(M.type() == CV_32F);
    double minVal, maxVal;
    minMaxLoc(M, &minVal, &maxVal);

    Mat P = (M - minVal) / (maxVal - minVal); // 0..1
    P = 1.0f - P;                             // центр=1, края=0


    P = clipPSFByHeap(P);

    P = cropPSFToActiveRegionAndFixOdd(P);

    //P = applyRadialHann(P);                   // прижимаем края к 0

    //GaussianBlur(P, P, Size(3, 3), 1.0);       // сглаживание

    //pow(P, 2.0, P);                           // gamma correction

    Scalar s = sum(P);
    if (s[0] > 1e-6)
        P /= s[0];                            // нормировка суммы

    return P;
}

//---------------------------------------------
// 3. Сдвиг PSF (центр → (0,0))
//---------------------------------------------
void shiftPSF(const Mat& src, Mat& dst)
{
    dst.create(src.size(), src.type());
    int cx = src.cols / 2;
    int cy = src.rows / 2;

    Mat q0(src, Rect(0, 0, cx, cy));
    Mat q1(src, Rect(cx, 0, src.cols - cx, cy));
    Mat q2(src, Rect(0, cy, cx, src.rows - cy));
    Mat q3(src, Rect(cx, cy, src.cols - cx, src.rows - cy));

    Mat d0(dst, Rect(0, 0, dst.cols - cx, dst.rows - cy));
    Mat d1(dst, Rect(dst.cols - cx, 0, cx, dst.rows - cy));
    Mat d2(dst, Rect(0, dst.rows - cy, dst.cols - cx, cy));
    Mat d3(dst, Rect(dst.cols - cx, dst.rows - cy, cx, cy));

    q3.copyTo(d0);
    q2.copyTo(d1);
    q1.copyTo(d2);
    q0.copyTo(d3);
}

//---------------------------------------------
// 4. Обратный фильтр из PSF
//---------------------------------------------
Mat buildInverseFilterFromPSF(const Mat& psfSmall, Size imgSize, float eps = 1e-3f)
{
    Mat psfPadded(imgSize, CV_32F, Scalar(0));
    int x0 = (imgSize.width - psfSmall.cols) / 2;
    int y0 = (imgSize.height - psfSmall.rows) / 2;
    psfSmall.copyTo(psfPadded(Rect(x0, y0, psfSmall.cols, psfSmall.rows)));

    Mat psfShifted;
    shiftPSF(psfPadded, psfShifted);

    Mat planesH[] = { psfShifted.clone(), Mat::zeros(imgSize, CV_32F) };
    Mat H;
    merge(planesH, 2, H);
    dft(H, H, DFT_COMPLEX_OUTPUT);

    Mat planes[2];
    split(H, planes);
    Mat Re = planes[0];
    Mat Im = planes[1];

    Mat mag2;
    magnitude(Re, Im, mag2);
    mag2 = mag2.mul(mag2);

    Mat G_re = Re / (mag2 + eps);
    Mat G_im = -Im / (mag2 + eps);

    Mat G;
    Mat planesG[] = { G_re, G_im };
    merge(planesG, 2, G);

    return G;
}

//---------------------------------------------
// 5. Применение фильтра к одному каналу
//---------------------------------------------
Mat applyFilterDFT(const Mat& gray, const Mat& G)
{
    Mat f32;
    gray.convertTo(f32, CV_32F);

    Mat planesF[] = { f32, Mat::zeros(gray.size(), CV_32F) };
    Mat F;
    merge(planesF, 2, F);
    dft(F, F, DFT_COMPLEX_OUTPUT);

    Mat Y;
    mulSpectrums(F, G, Y, 0);

    Mat out32;
    idft(Y, out32, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat out8;
    out32.convertTo(out8, CV_8U);
    return out8;
}




//---------------------------------------------
// 6. Основной пример: работаем по Y, возвращаем цвет
//---------------------------------------------
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
    Mat Y = ch[0];
    Mat Cr = ch[1];
    Mat Cb = ch[2];

    const int radius = 15;

    // 1. M(dx,dy) по Y
    Mat M = computeMaxDiffMatrix(Y, radius);

    std::cout << "M:\n";
    for (int y = 0; y < M.rows; ++y)
    {
        for (int x = 0; x < M.cols; ++x)
        {
            std::cout << std::fixed << std::setprecision(0)
                << M.at<float>(y, x) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;


    // 2. PSF из M
    Mat psf = buildPSFFromM(M);

    std::cout << "PSF dim: " << psf.size() << "\n";

    // --- Вывод PSF в консоль ---
    //*
    std::cout << "PSF:\n";
    for (int y = 0; y < psf.rows; ++y)
    {
        for (int x = 0; x < psf.cols; ++x)
        {
            std::cout << std::fixed << std::setprecision(4)
                << psf.at<float>(y, x) << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    //*/

    // 3. Обратный фильтр
    Mat G = buildInverseFilterFromPSF(psf, Y.size(), 1e-3f);

    // 4. Деконволюция Y
    Mat Y_restored = applyFilterDFT(Y, G);

    // 5. Собираем обратно YCrCb → BGR
    ch[0] = Y_restored;
    merge(ch, ycrcb);

    Mat restoredBGR;
    cvtColor(ycrcb, restoredBGR, COLOR_YCrCb2BGR);

    imwrite(argv[2], restoredBGR);
    if (argc > 3)
        imwrite(argv[3], Y_restored);

    return 0;
}
