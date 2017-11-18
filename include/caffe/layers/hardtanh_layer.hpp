#ifndef CAFFE_HARDTANH_LAYER_HPP_
#define CAFFE_HARDTANH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Hard TanH hyperbolic tangent non-linearity @f$
 *        y = max(-1, min(1, x))
 * 
 * See 6.3.3 in 'Deep learning' by I. Goodfellow et al. 
 * (http://deeplearningbook.org) 
 */
template <typename Dtype>
class HardTanHLayer : public NeuronLayer<Dtype> {
 public:
  explicit HardTanHLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "HardTanH"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_HARDTANH_LAYER_HPP_
