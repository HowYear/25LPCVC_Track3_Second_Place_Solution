# LPCV 2025 Track 3 Solution

## 1. How to Run

1. Our project is based on [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Please refer to this project to complete the installation of related dependencies.
2. Download the Depth-Anything-V2-Small model weight trained on Indoor (Hypersim) from [metric_depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth). The weight can be found [here](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true).
3. Run `python submit.py` (modify the model_weight path in submit.py according to where you downloaded the weight)

## 2. Method Details

Our improvements are based on the competition's [Baseline code](https://github.com/lpcvai/25LPCVC_Track3_Sample_Solution/tree/main), focusing on both speed and accuracy:

1. **Accuracy Improvements:**
   - Replace the relative depth estimation model weight in the Baseline with Depth Anything V2's absolute depth estimation model weight trained on the Hypersim dataset.
   - Eliminate the normalization operation used for model postprocessing from the original Baseline.
   - Perform depth estimation on both the original input image and its horizontally flipped version, then average the results as the final output.

2. **Speed Improvements:**

   - Reduce input image resolution to 350×350.
   - Modify the original Attention implementation (Attentionv1): we split all the attention heads into separate subgraph instead of having a head axis and processing them together, which we call Attentionv2.
   - Replace input image normalization operation with BatchNorm.

     | Improvements                   | Inference Time |
     | ----------------------------- | -------------- |
     | 518+Attentionv2+BatchNorm     | 88.9 ms        |
     | 350+Attentionv1+BatchNorm     | 56.5 ms        |
     | 350+Attentionv2               | 29.4 ms        |
     | **350+Attentionv2+BatchNorm** | **28.6 ms**    |

3. **Ideas that did not work out:**
   - We used the DIODE dataset as our local validation set to evaluate model performance. We found that using a larger model (Depth-Anything-V2-Base) would achieve significantly better scores, but since Depth-Anything-V2-Base is open-sourced under the CC-BY-NC-4.0 license, it doesn't comply with competition rules, so we didn't use it.
   - We found that the DAV2 model trained on Hypersim performed significantly worse in outdoor scenes, so we tried to finetune using the outdoor portion of the DIODE dataset. However, the finetuned model showed degraded performance in both indoor and outdoor scenes (possibly due to issues with our finetuning approach).
   - We also tried better output post-processing methods for the DAV2 relative depth version model. We searched for optimal scale and shift factors for model output on the DIODE dataset. This method performed well on the local validation set but poorly on the actual competition dataset, perhaps due to significant differences between datasets.

## 3. LICENSE

This project is licensed under the Apache 2.0 License

## 4. Citation

```markdown
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and 	Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```
