/**
*
* InnerProduct_layer.cpp deep compression implementation code
* author: Solomon He, Solomonhe@zhaoxin.com
*
**/
#include <vector>
#include <iostream>
using namespace std ;

#include "caffe/filler.hpp"
#include "caffe/layers/cmp_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/kmeans.hpp"
#define KERNEL_LEVEL_SPARSITY 0

namespace caffe {

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::ComputeBlobMask()
{
  int count = this->blobs()[0]->count();
  int num    = this->blobs_[0]->num();
  int channels = this->blobs_[0]->channels();
  int height = this->blobs_[0]->height();                                                 
  int width  = this->blobs_[0]->width();                                                   

  //calculate min max value of weight
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int *mask_data = this->masks_.mutable_cpu_data();

  vector<Dtype> sort_weight(count);

  for (int i = 0; i < count; ++i)
  {
     sort_weight[i] = fabs(weight[i]);
  }
#if  0 //KERNEL_LEVEL_SPARSITY
  int kernel_num = num * channels;
  vector<Dtype> kernel_level_sort_weight(kernel_num); //each vector element store each kernel maximum value

  //1. find each kernal max element then store to the kernel_level_sort_weight vector
  for(int kernelIdx = 0; kernelIdx < kernel_num; ++kernelIdx)
  {
     Dtype tmp_max_value = 0;
     for(int heightIdx = 0; heightIdx < height; heightIdx++)
         for(int widthIdx = 0; widthIdx < width; widthIdx++)
            if(sort_weight[kernelIdx * height * width + heightIdx * width + widthIdx] > tmp_max_value)
                tmp_max_value = sort_weight[kernelIdx * height * width + heightIdx * width + widthIdx];
     kernel_level_sort_weight[kernelIdx] = tmp_max_value; 
  }
  //2. sort kernel_level_sort_weight
  vector<Dtype> kernel_level_weight_original(kernel_level_sort_weight);
  sort(kernel_level_sort_weight.begin(), kernel_level_sort_weight.end());
  //3. base the sparse ratio setting to zero corresponding kernel
  float ratio = this->sparse_ratio_;
  int index = int(kernel_num * ratio);
  Dtype thr;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
//  int *mask_data = this->masks_.mutable_cpu_data();
  float rat = 0;
  if(index > 0){
     thr = kernel_level_sort_weight[index - 1];
     for(int kernelIdx = 0; kernelIdx < kernel_num; kernelIdx++)
     {
	if(kernel_level_weight_original[kernelIdx] > thr || kernel_level_weight_original[kernelIdx] < -thr)
	{
	    for(int heightIdx = 0; heightIdx < height; ++heightIdx)
	       for(int widthIdx = 0; widthIdx < width; ++widthIdx)
               {
		  mask_data[kernelIdx * height * width + heightIdx * width + widthIdx] = 1;		
	       }
	}
        else
        {
	    for(int heightIdx = 0; heightIdx < height; ++heightIdx)
	       for(int widthIdx = 0; widthIdx < width; ++widthIdx)
               {
		  mask_data[kernelIdx * height * width + heightIdx * width + widthIdx] = 0;		
		  muweight[kernelIdx * height * width + heightIdx * width + widthIdx] = 0;
                  rat += 1;		
	       }
	}
     }
  }
  else //index = 0
  {
     for(int i = 0; i < count; ++i)
     { 
          mask_data[i] = (weight[i] == 0 ? 0 :1); //keep unchanged
	  rat += (1-mask_data[i]) ;
     }
  }
#else

  sort(sort_weight.begin(), sort_weight.end());
  
  float ratio = this->sparse_ratio_;
  int index = int(count*ratio);
  Dtype thr ;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  float rat = 0;
  if(index >0) {
  thr = sort_weight[index - 1];
  LOG(INFO) << "InnerProduct THR: "<< thr << endl ;

  for (int i = 0; i < count; ++i)
  {
    mask_data[i] =  ((weight[i] > thr || weight[i] < -thr) ? 1 : 0) ;

   //data which fabs(weight[i]) < thr will be set to 0 ----Solomon
    muweight[i] *= mask_data[i];
   rat += (1-mask_data[i]) ;
  }
  }
  else {
      for (int i = 0; i < count; ++i)
      {    
         mask_data[i]  = (weight[i] == 0 ? 0 : 1);
         rat += (1-mask_data[i]) ;
      } 
  }
#endif
   LOG(INFO) << "sparsity: "<< rat/count <<endl; //rat means weight num who was set to 0 ----Solomon
    
  if(this->quantize_term_)
  {
    int nCentroid = this->class_num_;

    kmeans_cluster(this->indices_.mutable_cpu_data(), this->centroids_.mutable_cpu_data(), muweight, count, mask_data/*this->masks_*/, /*max_weight, min_weight,*/ nCentroid, 1000);
  }                                                  
}
template <typename Dtype>
void CmpInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
 
 //deep compression: prune + quantization data initialization ----Solomon 
  this->sparse_ratio_ = this->layer_param_.inner_product_param().sparse_ratio();
  this->class_num_ = this->layer_param_.inner_product_param().class_num();
  this->quantize_term_ = this->layer_param_.inner_product_param().quantize_term();
  int count = this->blobs_[0]->count() ; 
  vector<int> mask_shape(1,count);
  this->masks_.Reshape(mask_shape);
  caffe_set(count, 1, this->masks_.mutable_cpu_data());

  if(quantize_term_)
  {   
    this->indices_.Reshape(mask_shape);
    vector<int> cen_shape(1,class_num_);
    this->centroids_.Reshape(cen_shape);
  } 

}

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int *mask_data = this->masks_.cpu_data();
  int count = this->blobs_[0]->count();

  for (int i = 0; i < count; ++i)
    muweight[i] *= mask_data[i] ;

  if(this->quantize_term_)
  {
    const Dtype *cent_data = this->centroids_.cpu_data();
    const int *indice_data = this->indices_.cpu_data();

    for (int i = 0; i < count; ++i)
    {
       if (mask_data[i])
         muweight[i] = cent_data[indice_data[i]];
    }
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CmpInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  
  int count = this->blobs_[0]->count();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const int *mask_data = this->masks_.cpu_data();

  for (int j = 0; j < count; ++j)
    weight_diff[j] *= mask_data[j];
  
  if(this->quantize_term_)
  {
    const int *indice_data = this->indices_.cpu_data();
    vector<Dtype> tmpDiff(this->class_num_);
    vector<int> freq(this->class_num_);
    for (int j = 0; j < count; ++j)
    {
       if (mask_data[j])
       {
          tmpDiff[indice_data[j]] += weight_diff[j];
          freq[indice_data[j]]++;
       }
    }
    for (int j = 0; j < count; ++j)
    {
       if (mask_data[j])
          weight_diff[j] = tmpDiff[indice_data[j]] / freq[indice_data[j]];
    }
  }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CmpInnerProductLayer);
#endif

INSTANTIATE_CLASS(CmpInnerProductLayer);
REGISTER_LAYER_CLASS(CmpInnerProduct);

}  // namespace caffe
