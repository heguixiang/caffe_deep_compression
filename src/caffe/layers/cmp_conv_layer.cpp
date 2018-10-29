#include <vector>
#include <iostream>
#include "caffe/kmeans.hpp"
using namespace std;
#include "caffe/layers/cmp_conv_layer.hpp"

#define FEATURE_MAP_AND_WEIGHT 1 //Solomon 2018/10/29 for sparse matrix statistic in caffe framework 

#include <fstream>
#include <sstream>

namespace caffe {

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::ComputeBlobMask()
{
  int count = this->blobs_[0]->count();

  //calculate min max value of weight
  const Dtype* weight = this->blobs_[0]->cpu_data();
  vector<Dtype> sort_weight(count);
               
  for (int i = 0; i < count; ++i)
  {
     sort_weight[i] = fabs(weight[i]);
  }

  sort(sort_weight.begin(), sort_weight.end());
  float ratio = this->sparse_ratio_;

  int index = int(count*ratio) ; //int(count*(1- max_weight)) ;
  Dtype thr ;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  int *mask_data = this->masks_.mutable_cpu_data();
  float rat = 0;
  if(index > 0){

    //thr = ratio;
    thr= sort_weight[index-1];
    LOG(INFO) << "CONV THR: " <<thr << " " <<ratio <<endl;


    for (int i = 0; i < count; ++i)
    {
	mask_data[i] = ((weight[i] > thr || weight[i] < -thr) ? 1 : 0) ;
        //data which fabs(weight[i]) < thr will be set to 0 ----Solomon
        muweight[i] *= mask_data[i];
        rat += (1-mask_data[i]) ;
     }
   
  }
  else {
      for (int i = 0; i < count; ++i)
      {
          mask_data[i] = (weight[i]==0 ? 0 :1); //keep unchanged
	  rat += (1-mask_data[i]) ;
      }
  }
   LOG(INFO) << "sparsity: "<< rat/count <<endl; //rat means weight num who was set to 0 ----Solomon
  if(this->quantize_term_)
  {
    int nCentroid = this->class_num_;
    kmeans_cluster(this->indices_.mutable_cpu_data(), this->centroids_.mutable_cpu_data(),  muweight, count, mask_data/*this->masks_*/,/* max_weight, min_weight,*/ nCentroid, 1000);
  }
}
template <typename Dtype>
void CmpConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int *mask_data = this->masks_.cpu_data();
  int count = this->blobs_[0]->count();
  //vector<Dtype> sort_weight(count);
  for (int i = 0; i < count; ++i)
    muweight[i] *= mask_data[i] ;
  
  if(this->quantize_term_)
  {
    const Dtype *cent_data = this->centroids_.cpu_data();
    const int *indice_data = this->indices_.cpu_data();
    for (int i = 0; i < count; ++i)
    {
       //base on the index cluster and centroid to search its corresponding weight which from centroid   ----Solomon
       if (mask_data[i])
         muweight[i] = cent_data[indice_data[i]]; 
    }
  }

  const Dtype* weight = this->blobs_[0]->cpu_data();

#if FEATURE_MAP_AND_WEIGHT
//1. dump weight file
 static int conv_id_layer = 0;

 const int* kernel_shape_data = this->kernel_shape_.cpu_data(); 
 const int kernel_h = kernel_shape_data[0];
 const int kernel_w = kernel_shape_data[1];
 const int* pad_data = this->pad_.cpu_data();
 const int pad_h    = pad_data[0];
 const int pad_w    = pad_data[1];
 const int* stride_data = this->stride_.cpu_data();
 const int stride_h = stride_data[0];
 const int stride_w = stride_data[1];
 const int input_channels = bottom[0]->channels();
 const int output_channels = top[0]->channels();
 const int output_h = this->output_shape_[0];
 const int output_w = this->output_shape_[1]; 

 std::string feature_map_dir = "./FeatureMap_and_Weight/conv_";
 std::cout << "conv layer:" << conv_id_layer << ", output channels:" << output_channels
           << ", input_channels:" << input_channels << ", kernel_h:" << kernel_h
           << ", kernel_w:" << kernel_w << std::endl;
 std::string weight_filename = feature_map_dir + std::to_string(conv_id_layer) + "_weight.txt";
 std::ofstream fout;
 fout.open(weight_filename.c_str(), std::ios_base::out);
 if(!fout.is_open())
 {
     LOG(INFO) << "cannot open feature map dump file: " << weight_filename << "!"; 
 }
 fout << "output_channels:" << output_channels << " " << "input_channels:" << input_channels
      << " " << "kernel_h:" << kernel_h << " " << "kernel_w:" << kernel_w << std::endl;

 for(uint32_t output_channel_idx = 0; output_channel_idx < output_channels; output_channel_idx++)
 {
    for(uint32_t input_channel_idx = 0; input_channel_idx < input_channels; input_channel_idx++)
       for(uint32_t kernel_idx = 0 ; kernel_idx < kernel_h * kernel_w; kernel_idx++)
       {
	   fout << weight[(output_channel_idx * input_channels + input_channel_idx) * kernel_w * kernel_h + kernel_idx] << " ";           
       } 
    fout << std::endl;
 }
 fout.close();

#endif
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
#if FEATURE_MAP_AND_WEIGHT
      const int height = bottom[0]->height();
      const int width  = bottom[0]->width();

      std::string input_dir = "./FeatureMap_and_Weight/conv_";
      std::string input_filename = input_dir + std::to_string(conv_id_layer) + "_input_batch" + std::to_string(n) + ".txt";
      fout.open(input_filename.c_str(), std::ios_base::out);
      if(!fout.is_open())
      {
          LOG(INFO) << "cannot open feature map dump file: " << weight_filename << "!"; 
      }
      fout << "channels:" << input_channels << " " << "height:" << height << " " << "width:" << width << " "
           << "kernel_h:" << kernel_h << " " << "kernel_w:" << kernel_w << " " << "pad_h:" << pad_h << " " 
           << "pad_w:" << pad_w << " " << "stride_h:" << stride_h << " " << "stride_w:" << stride_w << " " 
           << "output_h:" << output_h << " " << "output_w:" << output_w << " " << std::endl;
      //n: denote batch id
      for(uint32_t input_channel_idx = 0; input_channel_idx < input_channels; input_channel_idx++)
      {
         for(uint32_t height_idx = 0; height_idx < height; height_idx++)
             for(uint32_t width_idx = 0; width_idx < width; width_idx++)
	     {
 		fout << bottom_data[(n * input_channels + input_channel_idx) * height * width + height_idx * width + width_idx];
                fout << " " ;
             }
         fout << std::endl;
      }
      fout.close();

      conv_id_layer++;
    
#endif
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void CmpConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 //LOG(INFO) << "conv backward" << endl;
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  int count = this->blobs_[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        const int *mask_data = this->masks_.cpu_data();
        for (int j = 0; j < count; ++j)
          weight_diff[j] *=  mask_data[j];

        if(this->quantize_term_)
        {
	  vector<Dtype> tmpDiff(this->class_num_);
          vector<int> freq(this->class_num_);
          const int *indice_data = this->indices_.cpu_data();

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
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CmpConvolutionLayer);
#endif

INSTANTIATE_CLASS(CmpConvolutionLayer);

}  // namespace caffe
