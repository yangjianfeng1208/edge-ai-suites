```bash
[2/4] 加载YOLO11n预训练模型...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt to 'yolo11n.pt': 100% ━━━━━━━━━━━━ 5.4MB 2.3MB/s 2.4s

[3/4] 开始训练...
  数据集: C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\data.yaml
  Epochs: 100
  图片尺寸: 640
  Batch: 16
  设备: cpu
  输出: C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex
Ultralytics 8.4.70  Python-3.12.12 torch-2.4.1+cpu CPU (Intel Core(TM) Ultra 9 285K)
engine\trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\data.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=yolo11n_hilex, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=10, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex, save_frames=False, save_json=False, save_period=10, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
Overriding model.yaml nc=80 with nc=6

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]      
  3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
  4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  6                  -1  1     87040  ultralytics.nn.modules.block.C3k2            [128, 128, 1, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1    249728  ultralytics.nn.modules.block.C2PSA           [256, 256, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    111296  ultralytics.nn.modules.block.C3k2            [384, 128, 1, False]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1     32096  ultralytics.nn.modules.block.C3k2            [256, 64, 1, False]           
 17                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1     86720  ultralytics.nn.modules.block.C3k2            [192, 128, 1, False]          
 20                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 23        [16, 19, 22]  1    431842  ultralytics.nn.modules.head.Detect           [6, 16, None, [64, 128, 256]] 
YOLO11n summary: 182 layers, 2,591,010 parameters, 2,590,994 gradients, 6.4 GFLOPs

Transferred 448/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access  (ping: 0.00.0 ms, read: 9.95.6 MB/s, size: 43.6 KB)
train: Scanning C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\labels.cache... 1378 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1378/1378  0.0s
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\05cefca3-UGC_570_jpg.rf.9f7c53551509c805e6f38def6c4792cd.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\0e5cbeb7-UGC_552_jpg.rf.c06ed4384a49faff5c60bdf49d172e90.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\0feaac1e-UGC_519_jpg.rf.f37ccb4fa705787516768a6f7d77b744.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\1f2f87fd-jee_2_817_jpg.rf.fb4ad2c8782213f1971ed31c1a0c9f51.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\27a4b82c-jee_2_763_jpg.rf.82423987b211a4d480666652500aad79.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\4dc8a520-UGC_529_jpg.rf.103515211e9b3335f98b4fd184e32d76.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\6b57c779-sat-practice-test-3_page-0038_jpg.rf.f5c6229927f246243e5086ba59d76ffd.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\6eac79cd-UGC_526_jpg.rf.0803b1dee631654ba0af9ae56e6ebc92.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\6f395c7e-sat-practice-test-1-digital_page-0006_jpg.rf.2df8037146ca71994592e4abb73f3a94.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\716975f4-sat-practice-test-3_page-0040_jpg.rf.a0e038c6c0ed4b00e3897e51c2e1c2be.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\7c4034ef-GATE_CS2-2021_286_jpg.rf.0d8cb63f1d792f58da1e0cdefef02a5b.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\925ad5b9-BANK_135_jpg.rf.0082cc26d21c5f7006db271f8035b086.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\9e2247f4-UGC_574_jpg.rf.98188b0fa953c6a3933e80b028d25531.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\b222e71c-sat-practice-test-1-digital_page-0048_jpg.rf.df7bb886a656025ce594d0e2ab6e2f69.jpg: 2 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\b896f533-UPSC_412_jpg.rf.37a5ec2f8e4502dd306043842fa935d0.jpg: 1 duplicate labels removed
train: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\train\images\fc1627c8-sat-practice-test-9_page-0017_jpg.rf.5e75502eaa4ec447b0ba0415c5831500.jpg: 2 duplicate labels removed
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access  (ping: 0.00.0 ms, read: 10.64.8 MB/s, size: 33.9 KB)
val: Scanning C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\valid\labels.cache... 388 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 388/388  0.0s
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\valid\images\048378a6-BANK_122_jpg.rf.7bd48b8cdadcc5842063f44e952de670.jpg: 3 duplicate labels removed
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\valid\images\4450c55b-UGC_539_jpg.rf.c5bd2dc79603d9cc377c75c6c5892583.jpg: 1 duplicate labels removed
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Plotting labels to C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\labels.jpg... 
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100         0G      1.656      3.067      1.752         29        640: 100% ━━━━━━━━━━━━ 87/87 1.3s/it 1:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 1.2s/it 15.1s
                   all        388       3598       0.82      0.135       0.29      0.151

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100         0G      1.379      1.915      1.553         25        640: 100% ━━━━━━━━━━━━ 87/87 1.3s/it 1:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 1.2s/it 15.5s
                   all        388       3598      0.715      0.417      0.409      0.225

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100         0G      1.292      1.622      1.482         35        640: 100% ━━━━━━━━━━━━ 87/87 1.2s/it 1:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 1.2s/it 15.7s
                   all        388       3598      0.748      0.424      0.431      0.251
... ...
77 epochs completed in 2.519 hours.
Optimizer stripped from C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\last.pt, 5.5MB
Optimizer stripped from C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best.pt, 5.5MB

Validating C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best.pt...
Ultralytics 8.4.70  Python-3.12.12 torch-2.4.1+cpu CPU (Intel Core(TM) Ultra 9 285K)
YOLO11n summary (fused): 101 layers, 2,583,322 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 1.4it/s 9.1s
                   all        388       3598      0.779      0.849       0.85      0.653
          Answer_Block        301        902      0.843      0.931      0.942      0.706
           Description         34         49      0.456      0.444      0.412      0.274
           Instruction         92        113      0.765      0.876      0.873      0.704
 Question_Answer_Block        357       1037      0.887      0.961      0.964      0.775
        Question_Block        352       1013       0.81      0.937      0.945      0.669
   Question_Paper_Area        387        484      0.912      0.946      0.964      0.792
Speed: 0.4ms preprocess, 18.9ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex

[4/4] 训练完成！
  最佳权重: C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best.pt

[验证] 在测试集上评估模型...
Ultralytics 8.4.70  Python-3.12.12 torch-2.4.1+cpu CPU (Intel Core(TM) Ultra 9 285K)
YOLO11n summary (fused): 101 layers, 2,583,322 parameters, 0 gradients, 6.3 GFLOPs
val: Fast image access  (ping: 0.00.0 ms, read: 747.9295.2 MB/s, size: 49.5 KB)
val: Scanning C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\test\labels.cache... 199 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 199/199  0.0s
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\test\images\03b0f6e7-UGC_544_jpg.rf.f82874b32e7f6690e7dabb94565ee274.jpg: 1 duplicate labels removed
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\test\images\0c0ea3c0-UGC_507_jpg.rf.12cda3f453ba993f1727f492fd2bd249.jpg: 1 duplicate labels removed
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\test\images\6641fdec-GMAT_468_jpg.rf.37ff0115f43aada7b0462d411704aeb5.jpg: 1 duplicate labels removed
val: C:\Users\user\jianfeng\EDU-AI\PR\HiLEx\HiLex_Yolo_Format\test\images\9013949e-sat-practice-test-9_page-0050_jpg.rf.9f708bf6300f19b8acb0685dbeb9d901.jpg: 1 duplicate labels removed
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 2.8it/s 4.7s
                   all        199       1918      0.862       0.76      0.829      0.632
          Answer_Block        143        460      0.874      0.876        0.9      0.665
           Description         22         56      0.741      0.179      0.395      0.252
           Instruction         47         59      0.826      0.723       0.83      0.654
 Question_Answer_Block        182        557      0.924      0.936      0.946      0.745
        Question_Block        180        536       0.87      0.885      0.919      0.649
   Question_Paper_Area        199        250      0.938      0.962      0.987      0.824
Speed: 0.3ms preprocess, 17.6ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\runs\detect\val

验证结果:
  mAP@50: 0.829
  mAP@50-95: 0.632

[导出] 转换为OpenVINO格式...
Ultralytics 8.4.70  Python-3.12.12 torch-2.4.1+cpu CPU (Intel Core(TM) Ultra 9 285K)
YOLO11n summary (fused): 101 layers, 2,583,322 parameters, 0 gradients, 6.3 GFLOPs

PyTorch: starting from 'C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 10, 8400) (5.2 MB)

OpenVINO: starting export with openvino 2026.3.0-22165-7b32f06bcaa...
OpenVINO: export success  1.5s, saved as 'C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best_openvino_model\' (5.4 MB)

Export complete (1.7s)
Results saved to C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best_openvino_model
Predict:         yolo predict task=detect model=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best_openvino_model\ imgsz=640 half
Validate:        yolo val task=detect model=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best_openvino_model\ imgsz=640 data=C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\data.yaml half 
Visualize:       https://netron.app
  OpenVINO模型: C:\Users\user\jianfeng\EDU-AI\PR\edge-ai-my-fork\education-ai-suite\smart-classroom\content_search\providers\assignment_grading\models\yolo_hilex\yolo11n_hilex\weights\best_openvino_model\

================================================================================
全部完成！
================================================================================

```