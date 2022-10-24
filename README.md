rknn2 rk3588   
1.使用不经过后处理的模型  
2.没有进行int量化  
3.模型由官方onnx 转换过来预处理已经包含在模型中 详细请参考https://github.com/rockchip-linux/rknn-toolkit2 rknn-toolkit2 转换  
4.测试安卓系统下 build-android_RK3588.sh 编译 linux同理
![微信截图_20221024111116](https://user-images.githubusercontent.com/37204571/197441443-f0c49d82-7028-420a-af6e-7a8eb191f9f7.png)
----------------------------------------------------------------------------------------------------------------------------
结果
![out_coco](https://user-images.githubusercontent.com/37204571/197457233-9a93ca33-6629-48bc-8fdf-6720930c144c.jpg)
