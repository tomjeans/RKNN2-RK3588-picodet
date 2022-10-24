// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <vector>
#include <iostream>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 0
 using namespace std;
 using namespace cv;
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

Mat resize_image_Picodet(Mat srcimg, int *newh, int *neww, int *top, int *left,bool keep_ratio,int inpHeight,int inpWidth)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = inpHeight;
	*neww = inpWidth;
	Mat dstimg;
	if (keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = inpHeight;
			*neww = int(inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)inpHeight * hw_scale;
			*neww = inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

struct BoxInfo
{
	float xMin;
	float yMin;
	float xMax;
	float yMax;
	float score;
	int label;
};


void softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void nms(vector<BoxInfo>& input_boxes)
{   float nms_threshold=0.3;
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).xMax - input_boxes.at(i).xMin + 1)
			* (input_boxes.at(i).yMax - input_boxes.at(i).yMin + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].xMin, input_boxes[j].xMin);
			float yy1 = (max)(input_boxes[i].yMin, input_boxes[j].yMin);
			float xx2 = (min)(input_boxes[i].xMax, input_boxes[j].xMax);
			float yy2 = (min)(input_boxes[i].yMax, input_boxes[j].yMax);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= nms_threshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}


void generate_proposal(vector<BoxInfo>& generate_boxes, const int stride_, float* out_score, float* out_box,int inpHeight,int inpWidth,int top,int left)
{   int reg_max=7;
    int num_class=80;
	float score_threshold=0.5;
	const int num_grid_y = (int)ceil((float)inpHeight / stride_);// inputheight 
	const int num_grid_x = (int)ceil((float)inpWidth / stride_);//inputwidth  
	////cout << "num_grid_x=" << num_grid_x << ",num_grid_y=" << num_grid_y << endl;
	const int reg_1max = reg_max + 1;//reg_max

	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const int idx = i * num_grid_x + j;//num_grid_x num_grid_y
			const float* scores = out_score + idx * num_class;//
			//std::cout<<"socores [0]:"<<scores[0]<<"scores [1]:"<<scores[1]<<std::endl;
			int max_ind = 0;
			float max_score = 0;
			for (int k = 0; k < num_class; k++)//num_class 
			{
				if (scores[k] > max_score)
				{
					max_score = scores[k];
					max_ind = k;
				}
			}
			if (max_score >= score_threshold)
			{
				const float* pbox = out_box + idx * reg_1max * 4;//
				float dis_pred[4];
				float* y = new float[reg_1max];
				for (int k = 0; k < 4; k++)
				{
					softmax_(pbox + k * reg_1max, y, reg_1max);
					float dis = 0.f;
					for (int l = 0; l < reg_1max; l++)
					{
						dis += l * y[l];
					}
					dis_pred[k] = dis * stride_;
				}
				delete[] y;
				float pb_cx = (j + 0.5f) * stride_ - 0.5;
				float pb_cy = (i + 0.5f) * stride_ - 0.5;
				float x0 = pb_cx - dis_pred[0];
				float y0 = pb_cy - dis_pred[1];
				float x1 = pb_cx + dis_pred[2];
				float y1 = pb_cy + dis_pred[3];
				std::cout<<" x0 :"<<x0<<" y0 :"<<y0<<" x1 :"<<x1<<" y1 :"<<y1<<" max_score :"<<max_score<<" max_ind :"<<max_ind<<std::endl;
				generate_boxes.push_back(BoxInfo{ x0, y0, x1, y1, max_score, max_ind });
			}
		}
	}
}

struct boxxs
{
	int xMin;
	int yMin;
	int xMax;
	int yMax;
	float confidence_;
	int class_id;
};

std::vector<boxxs> post_process_picodet(rknn_output outputs[],int newh,int neww,cv::Mat cv_image,int top,int left)
{   int inputheight=320;
	int inputwidth=320;
	int num_outs=4;
	std::vector<int> stride;
	for (int i = 0; i <num_outs; i++)
	{
		stride.push_back(int(8 * pow(2, i)));
	}
    std::vector <BoxInfo> generate_boxes;
	for (int i = 0; i <num_outs; i++)//
	{
		float* cls_score = (float*)outputs[i].buf;// 0 1 2 3
		float* bbox_pred = (float*)outputs[i + num_outs].buf;// 4 5 6 7
		generate_proposal(generate_boxes, stride[i], cls_score, bbox_pred,inputheight,inputwidth,top,left);
	}
	
	nms(generate_boxes);
	float ratioh = (float)cv_image.rows / newh;
	float ratiow = (float)cv_image.cols / neww;
	std::cout<<"ratioh :"<<ratioh<<" ratiow :"<<ratiow<<std::endl;
	std::cout<<"left :"<<left<<" top:"<<top<<std::endl;
	std::vector<boxxs> bbox_result;
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = (int)max((generate_boxes[i].xMin - left)*ratiow, 0.f);
		int ymin = (int)max((generate_boxes[i].yMin - top)*ratioh, 0.f);
		int xmax = (int)min((generate_boxes[i].xMax - left)*ratiow, (float)cv_image.cols);
		int ymax = (int)min((generate_boxes[i].yMax - top)*ratioh, (float)cv_image.rows);
		int class_id = generate_boxes[i].label;
		float confidence=generate_boxes[i].score;
		bbox_result.push_back(boxxs{xmin,ymin,xmax,ymax,confidence,class_id});
		std::cout<<"class_id :"<<class_id<<" confidence :"<<confidence<<" xmin :"<<xmin<<" ymin :"<<ymin<<" xmax :"<<xmax<<" ymax :"<<ymax<<std::endl;
		//rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		//string label = format("%.2f", generate_boxes[i].score);
		//label = this->class_names[generate_boxes[i].label] + ":" + label;
		//putText(srcimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
	return bbox_result;
}

// cv::Mat normalize_(Mat img)
// {   cv::Mat input_image_; 
	// img.convertTo(input_image_, CV_32FC3);
	// //cv::normalize(img,input_image_,1,0,CV_MINMAX);
	// //input_image_=img.clone();
	// return input_image_;
// }

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  int            status     = 0;
  char*          model_name = NULL;
  rknn_context   ctx;
  size_t         actual_size        = 0;
  int            img_width          = 0;
  int            img_height         = 0;
  int            img_channel        = 0;
  const float    nms_threshold      = NMS_THRESH;
  const float    box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;
  int            ret;

  // init rga context
  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect      src_rect;
  im_rect      dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  if (argc != 3) {
    printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    return -1;
  }

  printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  model_name       = (char*)argv[1];
  char* image_name = argv[2];

  printf("Read %s ...\n", image_name);
  cv::Mat orig_img = cv::imread(image_name, 1);//读取图片
  if (!orig_img.data) {
    printf("cv::imread %s fail!\n", image_name);
    return -1;
  }
  cv::Mat img1;
  cv::Mat resize_img;
  int newh=0;
  int neww=0;
  int top=0;
  int left=0;
  bool keep_ratio=true;
  int inpHeight = 320;
  int inpWidth = 320;
  resize_img=resize_image_Picodet(orig_img,&newh,&neww,&top,&left,keep_ratio,inpHeight,inpWidth);
  
  cv::cvtColor(resize_img, img1, cv::COLOR_BGR2RGB);
  cv::imwrite("input_im.jpg",img1);
  
  cv::Mat img=img1.clone();
  img_width  = img.cols;
  img_height = img.rows;
  printf("img width = %d, img height = %d\n", img_width, img_height);

  /* Create the neural network */
  printf("Loading mode...\n");
  int            model_data_size = 0;
  unsigned char* model_data      = load_model(model_name, &model_data_size);
  ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  //rknn_query 
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
  //

  rknn_tensor_attr input_attrs[io_num.n_input];//
  memset(input_attrs, 0, sizeof(input_attrs));
  //
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
	std::cout<<"the input_attrs data type :"<<input_attrs[i].type<<std::endl;//
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];//
  memset(output_attrs, 0, sizeof(output_attrs));
  //
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width   = 0;
  int height  = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height  = input_attrs[0].dims[2];
    width   = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    height  = input_attrs[0].dims[1];
    width   = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
	
  }
std::cout<<"height :"<<height<<" width :"<<width<<" channel :"<<channel<<std::endl;
  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = width * height * channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;//1不做任何转换 0做转换

  // You may not need resize when src resulotion equals to dst resulotion 
  void* resize_buf = nullptr;


  //这里是 rga硬件 resize部分
  if (img_width != width || img_height != height) {
    printf("resize with RGA!\n");
    resize_buf = malloc(height * width * channel);
    memset(resize_buf, 0x00, height * width * channel);

    src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_height, RK_FORMAT_RGB_888);//img.data 是cv::Mat的数据地址
    dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);//resize 硬件
    ret = imcheck(src, dst, src_rect, dst_rect);//
    if (IM_STATUS_NOERROR != ret) {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      return -1;
    }
    //IM_STATUS STATUS = imresize(src, dst);

    // for debug
    cv::Mat resize_img(cv::Size(width, height), CV_8UC3, resize_buf);
    cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;
  } else {
    inputs[0].buf = (void*)img.data;//
  }
  // rga硬件 resize部分
  //  
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs);
  //

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  //
  for (int i = 0; i < io_num.n_output; i++) {
    outputs[i].want_float = 1;//
  }
  //want_float 输出浮点型
  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  //
/*   (float*)outputs[0].buf 
  (float*)outputs[1].buf
  (float*)outputs[2].buf
  (float*)outputs[3].buf
  
  (float*)outputs[4].buf
  (float*)outputs[5].buf
  (float*)outputs[6].buf
  (float*)outputs[7].buf */
  //结果 0-7
  
  
  
  
  
  
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
  
  
  
  std::vector<boxxs> result=post_process_picodet(outputs,newh,neww,orig_img,top,left);
  
  for(int i=0;i<result.size();i++)
  {
	int xmin=result[i].xMin;
	int ymin=result[i].yMin;
	int xmax=result[i].xMax;
	int ymax=result[i].yMax;
	int class_id=result[i].class_id;
	int confidence=result[i].confidence_;
	cv::rectangle(orig_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0, 255), 3);
  }
  
  imwrite("./out.jpg", orig_img);
  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);


  // release
  ret = rknn_destroy(ctx);

  if (model_data) {
    free(model_data);
  }

  if (resize_buf) {
    free(resize_buf);
  }

  return 0;
}
