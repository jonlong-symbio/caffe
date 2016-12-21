#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class ParameterLayer : public Layer<Dtype> {
 public:
  explicit ParameterLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const ParameterParameter& param = this->layer_param_.parameter_param();
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(param.shape_size());
      for (int i = 0; i < param.shape_size(); ++i) {
        this->blobs_[i].reset(new Blob<Dtype>());
        this->blobs_[i]->Reshape(param.shape(i));
      }
    }
    for (int i = 0; i < param.shape_size(); ++i) {
      top[i]->Reshape(param.shape(i));
    }
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { }
  virtual inline const char* type() const { return "Parameter"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const ParameterParameter& param = this->layer_param_.parameter_param();
    for (int i = 0; i < param.shape_size(); ++i) {
      top[i]->ShareData(*(this->blobs_[i]));
      top[i]->ShareDiff(*(this->blobs_[i]));
    }
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  { }
};

}  // namespace caffe

#endif
