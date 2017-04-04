#include <vector>

#include "caffe/layers/col2im_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void Col2imLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(bottom_data + n * bottom_dim_, channels_,
          top[0]->shape(channel_axis_ + 1),
          top[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          top_data + n * top_dim_);
    } else {
      col2im_nd_gpu(bottom_data + n * bottom_dim_, num_spatial_axes_, top_dim_,
          top[0]->gpu_shape() + channel_axis_,
          bottom[0]->gpu_shape() + channel_axis_,
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), top_data + n * top_dim_);
    }
  }
}

template <typename Dtype>
void Col2imLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int num_kernels = channels_ * bottom[0]->count(channel_axis_ + 1);
  for (int n = 0; n < num_; ++n) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(top_diff + n * top_dim_, channels_,
          top[0]->shape(channel_axis_ + 1),
          top[0]->shape(channel_axis_ + 2),
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1],
          bottom_diff + n * bottom_dim_);
    } else {
      im2col_nd_gpu(top_diff + n * top_dim_, num_spatial_axes_,
          num_kernels, top[0]->gpu_shape() + channel_axis_,
          bottom[0]->gpu_shape() + channel_axis_,
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), bottom_diff + n * bottom_dim_);
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(Col2imLayer);

}  // namespace caffe
