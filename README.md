# Document_Image_Dewarping

The code for "[Foreground and Text-lines Aware Document Image Rectification](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Foreground_and_Text-lines_Aware_Document_Image_Rectification_ICCV_2023_paper.pdf)", ICCV, 2023.

## Training Dataset
We use the Doc3D dataset for training. You can download the dataset on 
[DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) or [doc3D-dataset](https://github.com/fh2019ustc/doc3D-dataset).

## Evaluation Dataset
We evaluate on two datasets [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) and [DIR300](https://github.com/fh2019ustc/DocGeoNet).

## Inference
Please download the pre-trained model from 
[Google Drive](https://drive.google.com/drive/folders/1UWL7wWSCcyhHuWLSKQRI9g2_cp0M0aD-?usp=sharing) 
or [Baidu Cloud](https://pan.baidu.com/s/1JhEznQEjaVplPQww0CNbHA?pwd=p5yp). Then execute:
 
 `python predict.py --model_path /MODEL/PATH --img_path /BENCHMARK/DIR --save_path /SAVE/PATH`
 
## Evaluation

We follow the evaluation environment and code in [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) 
and [DocGeoNet](https://github.com/fh2019ustc/DocGeoNet).

For CER and ED metrics evaluation:

```text
Tesseract==5.0.1.20220118 (Windows)
pytesseract==0.3.8
```

The dewarped images can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PHyeZZF88-KzkeuV8YiKHJ_z9lC-3C9o) 
or [Baidu Cloud](https://pan.baidu.com/s/1Lq9tRbOM4nV-pQ9sbVfbww?pwd=y41i).
## Acknowledgement
Our methods and codes are inspired by many existing works, to which we would like to express special thanks to:

[DocUNet: Document Image Unwarping via A Stacked U-Net](https://www3.cs.stonybrook.edu/~cvl/content/papers/2018/Ma_CVPR18.pdf)

[DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D
Regression Networks](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/storage/paper.pdf)

[DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/pdf/2110.12942.pdf)

[Revisiting Document Image Dewarping by Grid Regularization](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.pdf)

[Geometric Representation Learning for Document Image Rectification](https://arxiv.org/pdf/2210.08161.pdf)


## Citation
If our methods and code are helpful to you, please refer to the following BibTeX format for citation:
```
@inproceedings{li2023foreground,
  title={Foreground and Text-lines Aware Document Image Rectification},
  author={Li, Heng and Wu, Xiangping and Chen, Qingcai and Xiang, Qianjin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={19574--19583},
  year={2023}
}
```


