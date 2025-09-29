//
// Created by cooper maira on 9/29/25.
//
#include <pathCam.h>

namespace pathCam {
  static double quadSubpixel1D(float vm1, float v0, float vp1) {
    // offset in [-1, +1], positive means peak is to the + direction
    double denom = (vm1 - 2.0*v0 + vp1);
    if (std::abs(denom) < 1e-12) return 0.0;
    return 0.5 * (vm1 - vp1) / denom;
  }


  static Rect centeredRect(int W, int H, int w, int h) {
    int x0 = (W - w) / 2;
    int y0 = (H - h) / 2;
    return cv::Rect(std::max(0, x0), std::max(0, y0),
                    std::min(w, W), std::min(h, H));
  }


  cuda::GpuMat CompositeVoronoi::preprocess_GPU(const cuda::GpuMat& bgr_or_gray, cudaStream_t stream) {
    CV_Assert(!bgr_or_gray.empty());
    cuda::GpuMat gray, f32, gx, gy, mag, magNorm, wx, wy, win, winApplied;

    if (bgr_or_gray.channels() == 3) {
      cuda::cvtColor(bgr_or_gray, gray, COLOR_BGR2GRAY, 0);
    } else {
      gray = bgr_or_gray;
    }
    gray.convertTo(f32, CV_32F);

    // Sobel (use separable filter from cudafilters)
    auto sobelX = cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    auto sobelY = cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
    sobelX->apply(f32, gx);
    sobelY->apply(f32, gy);

    cuda::magnitude(gx, gy, mag);

    // Normalize to [0,1] by dividing by max
    double minv, maxv;
    cuda::minMax(mag, &minv, &maxv);
    if (maxv > 0.0) cuda::multiply(mag, Scalar(1.0f / (float)maxv), magNorm, 1.0, -1);
    else mag.copyTo(magNorm);

    // Hann window: build 1D wins and apply outer-product
    ensure1DHann(magNorm.cols, magNorm.rows, wx, wy, stream);
    win.create(magNorm.size(), CV_32F);

    launch_apply_hann_2d(wx,wy,win,magNorm,stream);

    return win; // CV_32F 1ch
  }


  std::pair<Point2d, double> CompositeVoronoi::phase_correlate_GPU(const cuda::GpuMat& A32F, const cuda::GpuMat& B32F, cudaStream_t stream) {
        CV_Assert(A32F.size() == B32F.size() && A32F.type() == CV_32F && B32F.type() == CV_32F);
    Size sz = A32F.size();

    // 1) Forward FFTs (complex)
    cuda::GpuMat FA, FB;
    cuda::dft(A32F, FA, sz, 0);                   // CV_32FC2
    cuda::dft(B32F, FB, sz, 0);                   // CV_32FC2

    // 2) Cross-power spectrum on GPU
    cuda::GpuMat CPS( sz, CV_32FC2 );
    launch_CPS(FA,FB,CPS,1e-9,stream);

    // 3) Inverse FFT -> real correlation surface (still on GPU)
    cuda::GpuMat corr;
    cuda::dft(CPS, corr, sz, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // CV_32F

    // 4) Find global max on GPU (values on GPU, location returned as host Point)
    double peakVal = 0.0;
    Point peakLoc;
    cuda::minMaxLoc(corr, /*minVal*/nullptr, &peakVal, /*minLoc*/nullptr, &peakLoc, noArray());

    // 5) Turn peak location into wrapped shift (phase-corr convention)
    double dx = (peakLoc.x <= sz.width  / 2) ? peakLoc.x : (peakLoc.x - sz.width);
    double dy = (peakLoc.y <= sz.height / 2) ? peakLoc.y : (peakLoc.y - sz.height);

    // 6) Optional: subpixel refinement via 3x3 patch around peak (download only 3x3 ROI)
    //    Clamp ROI at borders
    int x0 = std::max(1, peakLoc.x) - 1;
    int y0 = std::max(1, peakLoc.y) - 1;
    int x1 = std::min(sz.width  - 2, peakLoc.x) + 1;
    int y1 = std::min(sz.height - 2, peakLoc.y) + 1;

    if (x1 - x0 + 1 == 3 && y1 - y0 + 1 == 3) {
        cuda::GpuMat roi = corr(Rect(x0, y0, 3, 3));
        Mat patch3x3;
        roi.download(patch3x3); // tiny (3x3x4B) transfer

        // rows: y-1, y, y+1 at the peak column; cols: x-1, x, x+1 at the peak row
        float cxm1 = patch3x3.at<float>(1, 0), cx0 = patch3x3.at<float>(1, 1), cxp1 = patch3x3.at<float>(1, 2);
        float cym1 = patch3x3.at<float>(0, 1), cy0 = patch3x3.at<float>(1, 1), cyp1 = patch3x3.at<float>(2, 1);

        double subx = quadSubpixel1D(cxm1, cx0, cxp1);
        double suby = quadSubpixel1D(cym1, cy0, cyp1);

        dx += subx;
        dy += suby;
    }

    // 7) A lightweight response: peak / sum(corr) (sum is a host scalar; small sync)
    Scalar s = cuda::sum(corr);
    double sumCorr = s[0] + 1e-12;
    double response = (sumCorr > 0.0) ? (peakVal / sumCorr) : 0.0;

    return { Point2d(dx, dy), response };
  }

  ScaleResult CompositeVoronoi::estimate_cale_discrete_GPU(const cuda::GpuMat &dLow_bgr, const cuda::GpuMat &dHigh_bgr, const std::vector<double> &scales, cudaStream_t stream) {

    constexpr int K = 128; // common evaluation size
    ScaleResult best;

    if (dLow_bgr.empty() || dHigh_bgr.empty()) return best;

    // 1) Center crop from LOW (assumes image is big enough)
    if (dLow_bgr.cols < K || dLow_bgr.rows < K) return best;
    Rect lowR = centeredRect(dLow_bgr.cols, dLow_bgr.rows, K, K);
    cuda::GpuMat lowK = dLow_bgr(lowR);

    // 2) For each scale hypothesis, extract (K*s)x(K*s) from HIGH, then downscale to 256x256
    for (double s : scales) {
        if (s <= 0.0) continue;
        int big = int(std::round(K * s));

        // Must be able to take a centered (big x big) from high
        if (dHigh_bgr.cols < big || dHigh_bgr.rows < big) continue;

        Rect highR = centeredRect(dHigh_bgr.cols, dHigh_bgr.rows, big, big);
        if (highR.width != big || highR.height != big) continue; // enforce exact size

        cuda::GpuMat highBig = dHigh_bgr(highR);

        // Downscale to KxK at low-mag pixel scale
        cuda::GpuMat highK;
        cuda::resize(highBig, highK, Size(K, K), 0, 0, INTER_AREA);

        // 3) Photometric preprocessing AFTER both are KxK (gray -> Sobel mag -> norm -> Hann)
        cuda::GpuMat lowProc  = preprocess_GPU(lowK,  stream);
        cuda::GpuMat highProc = preprocess_GPU(highK, stream);

        // 4) GPU phase correlation (FFT -> CPS -> iFFT), peak via cv::cuda::minMaxLoc
        auto [shift, response] = phase_correlate_GPU(lowProc, highProc, stream);

        double shiftMag = std::hypot(shift.x, shift.y);
        double diag = std::hypot(double(K), double(K));
        double shiftNormDiag = (diag > 0.0) ? (shiftMag / diag) : 0.0;

        if (!best.valid || response > best.response) {
            best.valid = true;
            best.scale = s;
            best.response = response;
            best.shift = shift;
            best.shiftNorm = shiftMag;
            best.shiftNormDiag = shiftNormDiag;
            best.cropSize = {K, K};
        }
    }

    return best;

  }



  ScaleResult CompositeVoronoi::estimate_scale_auto_GPU(const cuda::GpuMat& imgA, const cuda::GpuMat& imgB,
                                          const std::vector<double>& scales,
                                          cudaStream_t stream){
    ScaleResult ab = estimate_cale_discrete_GPU(imgA, imgB, scales, stream);
    ScaleResult ba = estimate_cale_discrete_GPU(imgB, imgA, scales, stream);
    if (!ab.valid) return ba;
    if (!ba.valid) return ab;
    return (ab.response >= ba.response) ? ab : ba;
  }


}