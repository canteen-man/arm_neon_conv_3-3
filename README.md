# arm_neon_conv_3-3
3*3 conv by arm neon
this is 3*3 conv in arm

the im2col function is  reference resources the darknet im2col.

Accelerate with the following instruction set：

 int16x8_t sum_vec = vdupq_n_s16（0）；
 //8 16-bit integers ser 0
  
 vgetq_lane_s16
  //Get the specified location data by SIMD
  
  int16x8_t w_vec = vld1q_s16(weight);
  //Read in eight data at the same time by SIMD
  
   int16x8_t mul_v = vmulq_s16
   //multiply in eight data at the same time by SIMD
   
   vaddq_s16
   //add eight data at the same time by SIMD
Compile command：
” g++ -o neon main.cpp -mfpu=neon `pkg-config --cflags --libs opencv`”
![Image text]（https://github.com/canteen-man/arm_neon_conv_3-3/blob/master/image.png）
