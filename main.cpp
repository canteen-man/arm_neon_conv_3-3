#include <iostream>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <arm_neon.h>

using namespace std;
using namespace cv;
int im2col_get_pixel(int *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

 void im2col_cpu(int* data_im,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, int* data_col)
    {
        int c,h,w;
        //计算卷积后的图像宽和高
        int height_col = (height + 2*pad - ksize) / stride + 1;
        int width_col = (width + 2*pad - ksize) / stride + 1;
        int channels_col = channels * ksize * ksize;//im2col后行数就是直接展开
        for (c = 0; c < channels_col; ++c) {//按照行逐点生成
            int w_offset = c % ksize;//第一行第几个权重对应的数
            int h_offset = (c / ksize) % ksize;//第几个卷积核中的第几个权重
            int c_im = c / ksize / ksize;//计算目前处理第几个通道的图像
           //根据当前卷积权重的偏移生成图像变换的矩阵
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad);
                }
            }

        }
    }


int* neon_gemm(int* data_col,int* w,int* out,int len)//3*3 conv
{
	
           int iter_total=len/9;  
           int* A=new int[9];
           int16_t* weight=(int16_t*)w;
   for(int iter=0;iter<iter_total;iter++){
          for(int mul=0;mul<9;mul++){
          A[mul]= data_col[iter*9+mul];    
         }
      int16_t* arr=(int16_t*)A;
      int out_9= (*(arr+8)) * (*(weight+8));
      int16x8_t sum_vec = vdupq_n_s16(0);
     for(int i= 0;i<8;i=i+8,arr=arr+8)
	{
          
	  int16x8_t data_vec = vld1q_s16(arr);
          int16x8_t w_vec = vld1q_s16(weight);
	  int16x8_t mul_v = vmulq_s16(data_vec, w_vec); 
          sum_vec = vaddq_s16(sum_vec, mul_v);
	}
       int sum = vgetq_lane_s16(sum_vec, 0)+vgetq_lane_s16(sum_vec, 1)+vgetq_lane_s16(sum_vec, 2)+vgetq_lane_s16(sum_vec, 3);
          
       int sum_rc=round((sum+out_9)/9);
       out[iter]=sum_rc;
    
  }
    delete A;
    return out;
}



int main()
{
   cv::Mat M=imread("/home/pi/conv_neon/cat_gray.jpg");
   int* im=new int[M.rows*M.cols];
   int row = M.rows;
       int col = M.cols;
      //动态创建二维数组，row行col列
       int **La = new int *[row];
       for (int i = 0; i < row; i ++){
           La[i] = new int[col];
       }
       // 循环二维数组和mat，并将mat对应值赋给二维数组对应值，
       for (int i = 0; i < row; i ++){
           for (int j = 0; j < col; j ++){
               La[i][j] = M.at<uchar>(i, j);
           }
       }
       for(int i=0;i<row;i++)
       {
       for(int j=0;j<col;j++)
       {
         im[i*col+j] = La[i][j];
       }
       }
     //  int length =row*col;
     //  for (int i=0;i<length;i++)
     //  cout<<im[i]<<endl;


    int height=M.rows;
    int width=M.cols;
    int channels=1;
    int ksize=3;
    int stride=1;
    int pad=1;
    int out_col = (width + 2*pad - ksize) / stride + 1;
    int out_row= (height + 2*pad - ksize) / stride + 1;
    int in_len=(channels *ksize *ksize*out_row ) * out_col ;
     cout<<in_len<<endl;
    int* data_col=new int[in_len];
    im2col_cpu( im,channels,  height,  width, ksize,   stride, pad, data_col);
    
    int weight[9]={1,2,3,4,5,6,7,8,9};//从此处读入卷积权重
    int* w=weight;
    int out_len=out_col*out_row;
    int* out=new int[out_len];
    out= neon_gemm(data_col,w,out,in_len);
    for(int i=0;i<out_len;i++)
	{
	cout<<out[i]<<endl;	
	}
       // 释放分配空间
       for (int i = 0; i < row; i ++){
           delete []La[i];
       }
       delete [] La;
       delete data_col;
       delete im;
       delete out;
   return 0;
}


