//
// Created by cooper maira on 10/9/25.
//

//#include "pathCam.h"
#include <cstdio>
#include <thread>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace cv;

enum BayerPattern { RGGB, BGGR, GBRG, GRBG };
// Return 0=R, 1=G, 2=R for given Bayer coordinate under BGGR
inline int bayerColor_BGGR(int y, int x)
{
    bool er = (y % 2 == 0);
    bool ec = (x % 2 == 0);
    if (er && ec)       return 2; // B
    if (er && !ec)      return 1; // G
    if (!er && ec)      return 1; // G
    return 0;                     // R
}
inline int bayerColor_BGGR_ind(int y, int x)
{
    bool er = (y & 1) == 0;
    bool ec = (x & 1) == 0;
    if (er && ec)  return 0; // B  (index 0 in kernels vector)
    if (er && !ec) return 1; // G  (index 1)
    if (!er && ec) return 1; // G  (index 1)
    return 2;                // R  (index 2)
}
cv::Vec3f computePerChannelScale_BGGR(const cv::Mat &imgA_in,
                                      const cv::Mat &imgB_in)
{
    CV_Assert(imgA_in.size() == imgB_in.size());
    CV_Assert(imgA_in.channels() == 1 && imgB_in.channels() == 1);

    // Convert to CV_32F
    cv::Mat A, B;
    if (imgA_in.type() == CV_32F) {
        A = imgA_in;
    } else {
        CV_Assert(imgA_in.type() == CV_8UC1);
        imgA_in.convertTo(A, CV_32F);
    }

    if (imgB_in.type() == CV_32F) {
        B = imgB_in;
    } else {
        CV_Assert(imgB_in.type() == CV_8UC1);
        imgB_in.convertTo(B, CV_32F);
    }

    int H = A.rows;
    int W = A.cols;

    // For each color channel: accumulate numerator and denominator
    // num[c] = sum A_i * B_i
    // den[c] = sum A_i^2
    double num[3] = {0.0, 0.0, 0.0};  // 0=R, 1=G, 2=B
    double den[3] = {0.0, 0.0, 0.0};

    for (int y = 0; y < H; ++y) {
        const float* aRow = A.ptr<float>(y);
        const float* bRow = B.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            int cIdx = bayerColor_BGGR(y, x);  // 0=R,1=G,2=B
            float a = aRow[x];
            float b = bRow[x];

            num[cIdx] += static_cast<double>(a) * static_cast<double>(b);
            den[cIdx] += static_cast<double>(a) * static_cast<double>(a);
        }
    }

    // Compute alpha per channel; fall back to 1.0 if den ~ 0
    float alphaR = 1.0f;
    float alphaG = 1.0f;
    float alphaB = 1.0f;

    const double eps = 1e-12;

    if (den[0] > eps) alphaR = static_cast<float>(num[0] / den[0]); // R
    if (den[1] > eps) alphaG = static_cast<float>(num[1] / den[1]); // G
    if (den[2] > eps) alphaB = static_cast<float>(num[2] / den[2]); // B

    // Return in BGGR order: B, G, R
    return {alphaB, alphaG, alphaR};
}
float computeGlobalAlpha(const cv::Mat &A_in, const cv::Mat &B_in)
{
    CV_Assert(A_in.size() == B_in.size());
    CV_Assert(A_in.channels() == 1 && B_in.channels() == 1);

    cv::Mat A, B;

    // Convert both to CV_32F
    if (A_in.type() == CV_32F) {
        A = A_in;
    } else {
        CV_Assert(A_in.type() == CV_8UC1);
        A_in.convertTo(A, CV_32F);
    }

    if (B_in.type() == CV_32F) {
        B = B_in;
    } else {
        CV_Assert(B_in.type() == CV_8UC1);
        B_in.convertTo(B, CV_32F);
    }

    int H = A.rows;
    int W = A.cols;

    double num = 0.0; // sum A_i * B_i
    double den = 0.0; // sum A_i^2

    for (int y = 0; y < H; ++y) {
        const float* aRow = A.ptr<float>(y);
        const float* bRow = B.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            float a = aRow[x];
            float b = bRow[x];
            num += static_cast<double>(a) * static_cast<double>(b);
            den += static_cast<double>(a) * static_cast<double>(a);
        }
    }

    const double eps = 1e-12;
    if (den <= eps) {
        // A is all zeros (or extremely small) – can't fit a scale reliably.
        // Return 1.0f as a neutral fallback.
        return 1.0f;
    }

    float alpha = static_cast<float>(num / den);
    return alpha;
}
void applyAlphaToBayer_BGGR(cv::Mat& img32f, const cv::Vec3f& alphaBGR)
{
    CV_Assert(img32f.type() == CV_32F && img32f.channels() == 1);

    int H = img32f.rows;
    int W = img32f.cols;

    float alphaB = alphaBGR[0];
    float alphaG = alphaBGR[1];
    float alphaR = alphaBGR[2];

    for (int y = 0; y < H; ++y) {
        float* row = img32f.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            int c = bayerColor_BGGR(y,x);   // 0=R,1=G,2=B

            if (c == 0)      row[x] *= alphaR;
            else if (c == 1) row[x] *= alphaG;
            else             row[x] *= alphaB;
        }
    }
}
cv::Mat convolveBayerWithKernels_BGGR(
    const cv::Mat &rawBayerIn,
    const std::vector<cv::Mat> &kernels  // 3 kernels, CV_32F, same size
)
{
    CV_Assert(kernels.size() == 3);
    CV_Assert(kernels[0].type() == CV_32F &&
              kernels[1].type() == CV_32F &&
              kernels[2].type() == CV_32F);
    CV_Assert(kernels[0].rows == kernels[0].cols);
    CV_Assert(kernels[1].rows == kernels[0].rows &&
              kernels[2].rows == kernels[0].rows &&
              kernels[1].cols == kernels[0].cols &&
              kernels[2].cols == kernels[0].cols);

    int ksize = kernels[0].rows;
    CV_Assert(ksize % 2 == 1);  // odd kernel size

    // Convert raw Bayer to CV_32F if needed
    cv::Mat raw32;
    if (rawBayerIn.type() == CV_32F) {
        raw32 = rawBayerIn;
    } else {
        CV_Assert(rawBayerIn.type() == CV_8UC1);
        rawBayerIn.convertTo(raw32, CV_32F);
    }

    CV_Assert(raw32.channels() == 1);

    int H = raw32.rows;
    int W = raw32.cols;

    int outH = H - ksize + 1;
    int outW = W - ksize + 1;
    CV_Assert(outH > 0 && outW > 0);

    cv::Mat out(outH, outW, CV_32F, cv::Scalar(0));

    int half = ksize / 2;

    for (int yOut = 0; yOut < outH; ++yOut)
    {
        for (int xOut = 0; xOut < outW; ++xOut)
        {
            // Center pixel in the original Bayer image
            int cy = yOut + half;
            int cx = xOut + half;

            int colorIdx = bayerColor_BGGR_ind(cy, cx); // 0=B,1=G,2=R
            const cv::Mat &k = kernels[colorIdx];

            float sum = 0.0f;

            for (int ky = 0; ky < ksize; ++ky)
            {
                const float* srcRow = raw32.ptr<float>(yOut + ky);
                const float* kRow   = k.ptr<float>(ky);

                for (int kx = 0; kx < ksize; ++kx)
                {
                    sum += kRow[kx] * srcRow[xOut + kx];
                }
            }

            out.at<float>(yOut, xOut) = sum;
        }
    }

    //out.convertTo(out,CV_8U);
    return out;  // CV_32F, BGGR mosaic, shrunk by ksize-1 in each dim
}
// Estimate a single kernel for one Bayer color (R/G/B) from a padded BGGR mosaic.
// sharpBayer: CV_32F, 1-channel, BGGR mosaic, size: H_b = 2*Hd + ksize - 1, W_b = 2*Wd + ksize - 1
// targetDense: CV_32F, 1-channel, dense output for that color, size Hd x Wd
// colorChar: 'R','G','B' (which Bayer color this kernel predicts)
// ksize: odd kernel size
// pattern: currently only BGGR supported
cv::Mat estimateKernelFromBayerSingleColor(
    const cv::Mat &sharpBayer,
    const cv::Mat &targetDense,
    int ksize,
    char colorChar,
    BayerPattern pattern = BGGR,
    double lambda = 1e-6)
{
    CV_Assert(sharpBayer.type() == CV_32F && sharpBayer.channels() == 1);
    CV_Assert(targetDense.type() == CV_32F && targetDense.channels() == 1);
    CV_Assert(ksize > 0 && (ksize % 2 == 1));
    CV_Assert(pattern == BGGR); // only BGGR implemented here

    int Hb = sharpBayer.rows;
    int Wb = sharpBayer.cols;

    int Hd = targetDense.rows;
    int Wd = targetDense.cols;

    int half = ksize / 2;

    // Enforce the padding relationship:
    // sharpBayer must be "ksize-1" larger than 2x dense in both dims
    CV_Assert(Hb == 2 * Hd + ksize - 1);
    CV_Assert(Wb == 2 * Wd + ksize - 1);

    int P = ksize * ksize; // number of kernel coefficients

    // Normal equations: (A^T A) k = A^T y
    cv::Mat AtA = cv::Mat::zeros(P, P, CV_64F);
    cv::Mat Aty = cv::Mat::zeros(P, 1, CV_64F);

    auto idx = [ksize](int u, int v) {
        return u * ksize + v;
    };

    // Map requested colorChar to 0/1/2 (R/G/B)
    int wantedColor;
    switch (colorChar)
    {
        case 'R': case 'r': wantedColor = 0; break;
        case 'G': case 'g': wantedColor = 1; break;
        case 'B': case 'b': wantedColor = 2; break;
        default:
        {
            std::cerr << "Unsupported color char, use R/G/B.\n";
            cv::Mat k = cv::Mat::zeros(ksize, ksize, CV_32F);
            k.at<float>(half, half) = 1.0f;
            return k;
        }
    }

    // Loop over every dense output pixel; each one defines a Bayer center
    // with a fully valid ksize×ksize neighborhood:
    //
    // For BGGR:
    //   B centers at (even,even)
    //   R centers at (odd,odd)
    //
    // With padding, center coordinates are:
    //   B: y = half + 2*yd, x = half + 2*xd
    //   R: y = half + 2*yd + 1, x = half + 2*xd + 1 (if you choose that)
    //
    // Here we'll implement B and R as examples.

    for (int yd = 0; yd < Hd; ++yd)
    {
        for (int xd = 0; xd < Wd; ++xd)
        {
            int y, x;

            if (wantedColor == 2) // B in BGGR at (even,even)
            {
                y = half + 2 * yd;
                x = half + 2 * xd;
            }
            else if (wantedColor == 0) // R in BGGR at (odd,odd)
            {
                y = half + 2 * yd + 1;
                x = half + 2 * xd + 1;
            }
            else
            {
                // For G, you'd need to define how your denseG is packed.
                // Right now, skip G:
                continue;
            }

            // Just sanity check color (can be turned into an assert if you like)
            int c = bayerColor_BGGR(y, x);
            if (c != wantedColor)
            {
                // If the padding alignment is right, this should never happen.
                continue;
            }

            float y_val = targetDense.at<float>(yd, xd);

            // A^T y
            for (int u = 0; u < ksize; ++u)
            {
                int yy = y + (u - half);
                const float* rowSrc = sharpBayer.ptr<float>(yy);
                for (int v = 0; v < ksize; ++v)
                {
                    int xx = x + (v - half);
                    double a_p = static_cast<double>(rowSrc[xx]);
                    int p = idx(u, v);
                    Aty.at<double>(p, 0) += a_p * y_val;
                }
            }

            // A^T A
            for (int u1 = 0; u1 < ksize; ++u1)
            {
                int y1 = y + (u1 - half);
                const float* row1 = sharpBayer.ptr<float>(y1);
                for (int v1 = 0; v1 < ksize; ++v1)
                {
                    int x1 = x + (v1 - half);
                    double a_p = static_cast<double>(row1[x1]);
                    int p = idx(u1, v1);

                    for (int u2 = 0; u2 < ksize; ++u2)
                    {
                        int y2 = y + (u2 - half);
                        const float* row2 = sharpBayer.ptr<float>(y2);
                        for (int v2 = 0; v2 < ksize; ++v2)
                        {
                            int x2 = x + (v2 - half);
                            double a_q = static_cast<double>(row2[x2]);
                            int q = idx(u2, v2);
                            AtA.at<double>(p, q) += a_p * a_q;
                        }
                    }
                }
            }
        }
    }

    // Regularize
    for (int p = 0; p < P; ++p)
        AtA.at<double>(p, p) += lambda;

    // Solve AtA * k = Aty
    cv::Mat k_vec;
    bool ok = cv::solve(AtA, Aty, k_vec, cv::DECOMP_CHOLESKY);
    if (!ok)
    {
        std::cerr << "Warning: solve failed, returning delta kernel.\n";
        cv::Mat k = cv::Mat::zeros(ksize, ksize, CV_32F);
        k.at<float>(half, half) = 1.0f;
        return k;
    }

    // Reshape into ksize×ksize
    cv::Mat kernel(ksize, ksize, CV_32F);
    for (int u = 0; u < ksize; ++u)
    {
        float* rowK = kernel.ptr<float>(u);
        for (int v = 0; v < ksize; ++v)
        {
            int p = idx(u, v);
            rowK[v] = static_cast<float>(k_vec.at<double>(p, 0));
        }
    }

    return kernel;
}

std::vector<cv::Mat> extractDenseBayerChannels(const cv::Mat &bayer,
                                               BayerPattern pattern)
{
    CV_Assert(bayer.channels() == 1);
    CV_Assert(bayer.rows % 2 == 0 && bayer.cols % 2 == 0);

    int H = bayer.rows;
    int W = bayer.cols;

    // Dense channel shapes
    cv::Mat denseR(H/2, W/2, bayer.type(), cv::Scalar(0));
    cv::Mat denseB(H/2, W/2, bayer.type(), cv::Scalar(0));
    cv::Mat denseG(H,   W/2, bayer.type(), cv::Scalar(0));  // two green rows

    // Functions to get color from CFA pattern
    auto colorAt = [&](int y, int x) {
        bool er = (y % 2 == 0);
        bool ec = (x % 2 == 0);

        switch(pattern)
        {
            case RGGB:
                if (er && ec)       return 0; // R
                if (er && !ec)      return 1; // G
                if (!er && ec)      return 1; // G
                return 2;                     // B

            case BGGR:
                if (er && ec)       return 2; // B
                if (er && !ec)      return 1; // G
                if (!er && ec)      return 1; // G
                return 0;                     // R

            case GRBG:
                if (er && ec)       return 1; // G
                if (er && !ec)      return 0; // R
                if (!er && ec)      return 2; // B
                return 1;                     // G

            case GBRG:
                if (er && ec)       return 1; // G
                if (er && !ec)      return 2; // B
                if (!er && ec)      return 0; // R
                return 1;                     // G
        }
        return -1;
    };

    // Populate dense channels
    for (int y = 0; y < H; ++y)
    {
        const uchar* srcRow = bayer.ptr<uchar>(y);
        for (int x = 0; x < W; ++x)
        {
            int c = colorAt(y, x);
            uchar v = srcRow[x];

            if (c == 0) // R
            {
                denseR.at<uchar>(y/2, x/2) = v;
            }
            else if (c == 1) // G
            {
                denseG.at<uchar>(y, x/2) = v;  // full rows, half columns
            }
            else // B
            {
                denseB.at<uchar>(y/2, x/2) = v;
            }
        }
    }

    return { denseR, denseG, denseB };
}
// Old, known-good forward operator for R/B
cv::Mat apply_Ak_RB(
    const cv::Mat &sharpBayer,
    const cv::Mat &k,
    int ksize,
    int half,
    int wantedColor,  // 0=R, 2=B
    int Hd, int Wd
) {
    cv::Mat u(Hd, Wd, CV_64F, cv::Scalar(0));

    for (int yd = 0; yd < Hd; ++yd) {
        for (int xd = 0; xd < Wd; ++xd) {
            int cy, cx;
            if (wantedColor == 2) {      // B
                cy = half + 2 * yd;
                cx = half + 2 * xd;
            } else {                     // R
                cy = half + 2 * yd + 1;
                cx = half + 2 * xd + 1;
            }

            double sum = 0.0;
            for (int u0 = 0; u0 < ksize; ++u0) {
                int yy = cy + (u0 - half);
                const float* srcRow = sharpBayer.ptr<float>(yy);
                const double* kRow  = k.ptr<double>(u0);
                for (int v0 = 0; v0 < ksize; ++v0) {
                    int xx = cx + (v0 - half);
                    sum += kRow[v0] * static_cast<double>(srcRow[xx]);
                }
            }

            u.at<double>(yd, xd) = sum;
        }
    }
    return u;
}

// Old, known-good adjoint for R/B
cv::Mat apply_ATr_RB(
    const cv::Mat &sharpBayer,
    const cv::Mat &r,
    int ksize,
    int half,
    int wantedColor,
    int Hd, int Wd
) {
    cv::Mat g(ksize, ksize, CV_64F, cv::Scalar(0));

    for (int yd = 0; yd < Hd; ++yd) {
        for (int xd = 0; xd < Wd; ++xd) {
            double rr = r.at<double>(yd, xd);
            if (rr == 0.0) continue;

            int cy, cx;
            if (wantedColor == 2) {      // B
                cy = half + 2 * yd;
                cx = half + 2 * xd;
            } else {                     // R
                cy = half + 2 * yd + 1;
                cx = half + 2 * xd + 1;
            }

            for (int u0 = 0; u0 < ksize; ++u0) {
                int yy = cy + (u0 - half);
                const float* srcRow = sharpBayer.ptr<float>(yy);
                double* gRow        = g.ptr<double>(u0);
                for (int v0 = 0; v0 < ksize; ++v0) {
                    int xx = cx + (v0 - half);
                    gRow[v0] += rr * static_cast<double>(srcRow[xx]);
                }
            }
        }
    }
    return g;
}
// New forward operator only for G
cv::Mat apply_Ak_G(
    const cv::Mat &sharpBayer,
    const cv::Mat &k,
    int ksize,
    int half,
    int Hd, int Wd
) {
    cv::Mat u(Hd, Wd, CV_64F, cv::Scalar(0));

    for (int yd = 0; yd < Hd; ++yd) {
        for (int xd = 0; xd < Wd; ++xd) {
            int cy = half + yd;
            int cx;
            if ((cy & 1) == 0) {
                // even row: G at x = 1,3,5,... => 2*xd+1
                cx = half + 2 * xd + 1;
            } else {
                // odd row: G at x = 0,2,4,... => 2*xd
                cx = half + 2 * xd;
            }

            double sum = 0.0;
            for (int u0 = 0; u0 < ksize; ++u0) {
                int yy = cy + (u0 - half);
                const float* srcRow = sharpBayer.ptr<float>(yy);
                const double* kRow  = k.ptr<double>(u0);
                for (int v0 = 0; v0 < ksize; ++v0) {
                    int xx = cx + (v0 - half);
                    sum += kRow[v0] * static_cast<double>(srcRow[xx]);
                }
            }

            u.at<double>(yd, xd) = sum;
        }
    }
    return u;
}

cv::Mat apply_ATr_G(
    const cv::Mat &sharpBayer,
    const cv::Mat &r,
    int ksize,
    int half,
    int Hd, int Wd
) {
    cv::Mat g(ksize, ksize, CV_64F, cv::Scalar(0));

    for (int yd = 0; yd < Hd; ++yd) {
        for (int xd = 0; xd < Wd; ++xd) {
            double rr = r.at<double>(yd, xd);
            if (rr == 0.0) continue;

            int cy = half + yd;
            int cx;
            if ((cy & 1) == 0) {
                cx = half + 2 * xd + 1;
            } else {
                cx = half + 2 * xd;
            }

            for (int u0 = 0; u0 < ksize; ++u0) {
                int yy = cy + (u0 - half);
                const float* srcRow = sharpBayer.ptr<float>(yy);
                double* gRow        = g.ptr<double>(u0);
                for (int v0 = 0; v0 < ksize; ++v0) {
                    int xx = cx + (v0 - half);
                    gRow[v0] += rr * static_cast<double>(srcRow[xx]);
                }
            }
        }
    }
    return g;
}
cv::Mat apply_Ak(
    const cv::Mat &sharpBayer,
    const cv::Mat &k,
    int ksize,
    int half,
    int wantedColor,
    int Hd, int Wd
) {
    if (wantedColor == 1) {
        return apply_Ak_G(sharpBayer, k, ksize, half, Hd, Wd);
    } else {
        return apply_Ak_RB(sharpBayer, k, ksize, half, wantedColor, Hd, Wd);
    }
}

cv::Mat apply_ATr(
    const cv::Mat &sharpBayer,
    const cv::Mat &r,
    int ksize,
    int half,
    int wantedColor,
    int Hd, int Wd
) {
    if (wantedColor == 1) {
        return apply_ATr_G(sharpBayer, r, ksize, half, Hd, Wd);
    } else {
        return apply_ATr_RB(sharpBayer, r, ksize, half, wantedColor, Hd, Wd);
    }
}
cv::Mat apply_M(
    const cv::Mat &sharpBayer,
    const cv::Mat &p,        // CV_64F, ksize×ksize
    int ksize,
    int half,
    int wantedColor,
    int Hd, int Wd,
    double lambda
) {
    cv::Mat Ap   = apply_Ak (sharpBayer, p, ksize, half, wantedColor, Hd, Wd);
    cv::Mat AtAp = apply_ATr(sharpBayer, Ap, ksize, half, wantedColor, Hd, Wd);
    AtAp += lambda * p;
    return AtAp;
}
cv::Mat estimateKernel_CGLS(
    const cv::Mat &sharpBayer,   // CV_32F, 1ch, padded: Hb x Wb
    const cv::Mat &targetDense,  // CV_32F, 1ch: Hd x Wd
    int ksize,
    char colorChar,
    BayerPattern pattern = BGGR,
    double lambda = 1e-6,
    int maxIters = 50,
    double tol = 1e-6
) {
    CV_Assert(sharpBayer.type() == CV_32F && sharpBayer.channels() == 1);
    CV_Assert(targetDense.type() == CV_32F && targetDense.channels() == 1);
    CV_Assert(ksize > 0 && (ksize % 2 == 1));
    CV_Assert(pattern == BGGR);

    int Hb = sharpBayer.rows;
    int Wb = sharpBayer.cols;
    int Hd = targetDense.rows;
    int Wd = targetDense.cols;
    int half = ksize / 2;

    int wantedColor;
    switch (colorChar)
    {
        case 'B': case 'b': wantedColor = 2; break;
        case 'R': case 'r': wantedColor = 0; break;
        case 'G': case 'g': wantedColor = 1; break;
        default:
        {
            std::cerr << "Unsupported colorChar; use 'R','G','B'.\n";
            cv::Mat k = cv::Mat::zeros(ksize, ksize, CV_32F);
            k.at<float>(half, half) = 1.0f;
            return k;
        }
    }

    // Geometry constraints differ by color
    if (wantedColor == 2 || wantedColor == 0) {
        // B / R: centers every 2 px in both directions
        CV_Assert(Hb == 2 * Hd + ksize - 1);
        CV_Assert(Wb == 2 * Wd + ksize - 1);
    } else {
        // G: dense in rows, half columns
        // Hd rows of centers, Wd greens per row
        CV_Assert(Hb == Hd + ksize - 1);
        CV_Assert(Wb == 2 * Wd + ksize - 1);
    }

    // Convert target to double
    cv::Mat y64;
    targetDense.convertTo(y64, CV_64F);

    // b = Aᵀ y
    cv::Mat b = apply_ATr(sharpBayer, y64, ksize, half, wantedColor, Hd, Wd);

    // Initialize k
    cv::Mat k = cv::Mat::zeros(ksize, ksize, CV_64F);

    // r = b - M k; since k=0, r = b
    cv::Mat r = b.clone();
    cv::Mat p = r.clone();

    double rr_old = (double)cv::sum(r.mul(r))[0];
    double rr0    = rr_old;

    for (int it = 0; it < maxIters; ++it)
    {
        // q = M p = Aᵀ(A p) + λ p
        cv::Mat q = apply_M(sharpBayer, p, ksize, half, wantedColor, Hd, Wd, lambda);

        double pq = (double)cv::sum(p.mul(q))[0];
        if (std::abs(pq) < 1e-20) {
            std::cerr << "CG: breakdown (p^T M p ~ 0)\n";
            break;
        }

        double alpha = rr_old / pq;

        // k_{n+1} = k_n + alpha p
        k += alpha * p;

        // r_{n+1} = r_n - alpha q
        r -= alpha * q;

        double rr_new = (double)cv::sum(r.mul(r))[0];
        double rel    = rr_new / rr0;

        std::cout << "Iter " << it
                  << "  ||r||^2 = " << rr_new
                  << "  rel = "   << rel << std::endl;

        if (rel < tol) {
            break;
        }

        double beta = rr_new / rr_old;

        // p_{n+1} = r_{n+1} + beta p_n
        p = r + beta * p;

        rr_old = rr_new;
    }

    // Optional: check data-space residual ||y - A k||^2
    {
        cv::Mat Ak = apply_Ak(sharpBayer, k, ksize, half, wantedColor, Hd, Wd);
        cv::Mat diff = y64 - Ak;
        double dataRes = (double)cv::sum(diff.mul(diff))[0];
        std::cout << "Final data residual ||y - A k||^2 = " << dataRes << std::endl;
    }

    // Optional: enforce sum(k) = 1 at the end
    double sumK = (double)cv::sum(k)[0];
    if (std::abs(sumK) > 1e-12) {
        k /= sumK;
    }

    cv::Mat kf;
    k.convertTo(kf, CV_32F);
    return kf;
}
double pearsonCorrelation(const cv::Mat &a, const cv::Mat &b) {
  CV_Assert(a.size() == b.size());
  CV_Assert(a.channels() == 1 && b.channels() == 1);

  cv::Mat af, bf;
  a.convertTo(af, CV_32F);
  b.convertTo(bf, CV_32F);

  const int rows = af.rows;
  const int cols = af.cols;
  const int N = rows * cols;

  double sumX = 0.0, sumY = 0.0;
  double sumX2 = 0.0, sumY2 = 0.0, sumXY = 0.0;

  for (int y = 0; y < rows; ++y) {
    const float* px = af.ptr<float>(y);
    const float* py = bf.ptr<float>(y);
    for (int x = 0; x < cols; ++x) {
      double vx = px[x];
      double vy = py[x];

      sumX  += vx;
      sumY  += vy;
      sumX2 += vx * vx;
      sumY2 += vy * vy;
      sumXY += vx * vy;
    }
  }

  double num = N * sumXY - sumX * sumY;
  double denX = N * sumX2 - sumX * sumX;
  double denY = N * sumY2 - sumY * sumY;
  double den = std::sqrt(denX * denY);

  if (den <= 1e-12) {
    return 0.0;  // degenerate case
  }
  return num / den;
}

void check_range(Mat& blurryPatch, Mat& sharpWhole,int patchSize, Point2i xRange, Point2i yRange,int id, std::vector<std::pair<double,Point2i>> *results) {
  Size imageSize(sharpWhole.cols,sharpWhole.rows);

  int xMin = xRange.x;
  int xMax = xRange.y;
  int yMin = yRange.x;
  int yMax = yRange.y;

  assert(xMin % 2 == 0 && yMin %2 == 0);
  assert(xMax + patchSize <= imageSize.width);
  assert(yMax + patchSize <= imageSize.height);
  assert(results->size() > id);


  double bestCorr = -2.0; // Pearson is in [-1, 1]
  int bestX = 0, bestY = 0;

  int totalChecks = (xMax - xMin)/2;
  int tenth  = totalChecks / 10;

  //search sharp image
  int count = 0;
  for (int x = xMin;x <= xMax; x += 2) {
    for (int y = yMin; y <= yMax; y += 2) {
      Rect sharpRoi(x,y,patchSize,patchSize);
      auto res = pearsonCorrelation(blurryPatch,sharpWhole(sharpRoi));

      if (res > bestCorr) {
        bestCorr = res;
        bestX = x;
        bestY = y;
      }

    }
    ++count;
    if (id == 0 && count % tenth == 0){
      std::cout<<"progress "<<count / tenth<<std::endl;
    }
  }
  results->at(id) = {bestCorr,{bestX,bestY}};
}

void find_best_matching_roi(Mat &rawBlur, Mat &rawSharp,int patchSize) {
  Size imageSize(rawBlur.cols,rawBlur.rows);
  Rect blurRect((rawBlur.cols - patchSize) / 2,(rawBlur.rows - patchSize) / 2,patchSize,patchSize);
  assert(blurRect.x % 2 == 0 && blurRect.y % 2 == 0);
  Mat blurCrop = rawBlur(blurRect).clone();

  std::string savePathBlur = "/home/pathcam/pcamdata/postProc/blur_test/train/misc/verificationPatches/blurPatch.png";
  std::string savePathSharp = "/home/pathcam/pcamdata/postProc/blur_test/train/misc/verificationPatches/sharpPatch.png";

  Mat blurPatchColor, sharpPatchColor;
  cvtColor(blurCrop,blurPatchColor,COLOR_BayerBG2BGR);

  imwrite(savePathBlur,blurPatchColor);


  int xMin = blurRect.x - 500;
  int xMax = blurRect.x + 500;
  int yMin = blurRect.y - 200;
  int yMax = blurRect.y + 200;
  assert(xMin % 2 == 0 && yMin %2 == 0);
  assert(xMax + patchSize <= imageSize.width);
  assert(yMax + patchSize <= imageSize.height);


  int nThreads = 20;

  auto results = new std::vector<std::pair<double,Point2i>>(20,{-2,{0,0}});

  int increment = (xMax - xMin) / nThreads;
  std::vector<std::thread> threads;
  for (int i = 0; i < nThreads; ++i) {
    int tMin = xMin + i * increment;
    tMin -= tMin % 2;
    int tMax = tMin + increment;
    tMax += tMax % 2;
    Point2i xRange(tMin,tMax);

    std::cout<<"range "<<i<<" "<<xRange.x<<" "<<xRange.y<<std::endl;
    Point2i yRange(yMin,yMax);
    threads.emplace_back(check_range,std::ref(blurCrop),std::ref(rawSharp),patchSize,xRange,yRange,i,results);
  }

  for (auto &t : threads) {
    t.join();
  }

  double bestCorr = -2.0; // Pearson is in [-1, 1]
  int bestX = 0, bestY = 0;

  for (auto &[score,loc] : *results) {
    if (score > bestCorr) {
      bestCorr = score;
      bestX = loc.x;
      bestY = loc.y;
    }
  }


  std::cout << "Best match in sharp frame at ("
          << bestX << ", " << bestY << ") with Pearson correlation = "
          << bestCorr << "\n";

  Rect bestSharpRoi(bestX,bestY,patchSize,patchSize);
  cvtColor(rawSharp(bestSharpRoi),sharpPatchColor,COLOR_BayerBG2BGR);
  imwrite(savePathSharp,sharpPatchColor);

  int k = 0;
}


Mat load_raw(const std::string &path) {
  Size image_size(6464,4852);

  char *bufHost = new char[image_size.width * image_size.height];
  std::ifstream stream;
  stream.open(path, std::ios::binary);
  stream.read(bufHost, image_size.width * image_size.height);
  stream.close();

  return {image_size,CV_8UC1,bufHost};
}



cv::Mat makeMotionKernel(int length, float angleDeg)
{
  // Ensure length is at least 1 and odd
  length = std::max(1, length);
  if (length % 2 == 0) length += 1;

  int ksize = length; // you can pad larger if you want, but this is enough
  cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);

  // Draw a horizontal line (center row) of "length" ones
  int center = ksize / 2;
  int half = length / 2;
  for (int x = center - half; x <= center + half; ++x)
    kernel.at<float>(center, x) = 1.0f;

  // Normalize to sum = 1
  kernel /= static_cast<float>(length);

  // Rotate the kernel by angleDeg around its center
  cv::Point2f rotCenter(ksize / 2.0f, ksize / 2.0f);
  cv::Mat rotMat = cv::getRotationMatrix2D(rotCenter, angleDeg, 1.0);

  cv::Mat rotated;
  cv::warpAffine(kernel, rotated, rotMat, kernel.size(),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Re-normalize (rotation/interpolation slightly changes sum)
  double sum = cv::sum(rotated)[0];
  if (sum != 0.0)
    rotated /= static_cast<float>(sum);

  return rotated;
}

bool loadRawToGpuGray(const std::string &path,
                             int width, int height,
                             cv::cuda::GpuMat &outGray,
                             int bayerCode = cv::COLOR_BayerBG2GRAY) {
  // Read file
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "[WARN] Cannot open file: " << path << "\n";
    return false;
  }
  std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);

  const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (size != static_cast<std::streamsize>(expected)) {
    std::cerr << "[WARN] Size mismatch for " << path
        << " (got " << size << ", expected " << expected << " bytes)\n";
    return false;
  }

  std::vector<unsigned char> buf(expected);
  if (!f.read(reinterpret_cast<char *>(buf.data()), size)) {
    std::cerr << "[WARN] Failed to read data: " << path << "\n";
    return false;
  }

  // Wrap CPU Mat as single-channel 8-bit
  cv::Mat raw(height, width, CV_8UC1, buf.data());

  // Upload to GPU
  cv::cuda::GpuMat dRaw, dGray;
  dRaw.upload(raw);

  // Debayer on GPU -> grayscale
  cv::cuda::cvtColor(dRaw, dGray, bayerCode);

  // Return
  outGray = dGray; // shallow copy of GpuMat header (data stays on GPU)
  return true;
}