#pragma once
#include <vector>
#include <opencv2/core/types.hpp>

namespace cv
{
	class _InputArray;
	class _OutputArray;
    typedef const _InputArray& InputArray;
    typedef const _OutputArray& OutputArray;
}

struct PhaseCongruencyConst {
    double sigma;
    double mult = 1.6;
    double minwavelength = 3;
    double epsilon = 0.0002;
    double cutOff = 0.4;
    double g = 3.0;
    double k = 1.0;
    PhaseCongruencyConst();
    PhaseCongruencyConst(const PhaseCongruencyConst& _pcc);
    PhaseCongruencyConst& operator=(const PhaseCongruencyConst& _pcc);
};

class PhaseCongruency
{
public:
	PhaseCongruency(cv::Size _img_size, size_t _nscale, size_t _norient);
	~PhaseCongruency() {}
    void setConst(PhaseCongruencyConst _pcc);
    void calc(cv::InputArray _src, std::vector<cv::Mat> &_pc);
    void feature(std::vector<cv::Mat> &_pc, cv::OutputArray _edges, cv::OutputArray _corners);
    void feature(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners);

    std::vector<std::vector<cv::Mat> > eo; //s*o
    
private:
    cv::Size size;
    size_t norient;
    size_t nscale;

    PhaseCongruencyConst pcc;

    std::vector<cv::Mat> filter;

};