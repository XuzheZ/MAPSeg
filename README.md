# MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D <ins>M</ins>asked <ins>A</ins>utoencoding and <ins>P</ins>seudo-Labeling

[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MAPSeg_Unified_Unsupervised_Domain_Adaptation_for_Heterogeneous_Medical_Image_Segmentation_CVPR_2024_paper.html) / [arXiv](https://arxiv.org/abs/2303.09373)

A **unified** UDA framework for **3D** medical image segmentation for several scenarios: 

![MAPseg can solver various problems in different settings](/figs/overview.png)

Built upon complementary masked autoencoding and pseudo-labeling: 

![Framework](/figs/framework.png)
## Usage: 

    conda create --name <mapseg> --file requirements.txt
    conda activate mapseg

For training: 
    
    python train.py --config=YOUR_PATH_TO_YAML

For inference: 

    python test.py # be sure to edit the test.py 

Parameters and data structure: 
There is a **detailed** explanation in /cfg/default.py

## Useful Suggestions:
More to be updated soon. 
1. The input image should have correct affine information (for orientation) in header. The data loader will automatically adjust it to the RAS space as defined in Nibabel (see [more](https://nipy.org/nibabel/coordinate_systems.html))
2. If orientation information is no longer available, please manually check all scans (the data matric) to ensure they are in the same orientation. This is extremely important in pseudo labeling (fine for MAE pretrain in different orientations). 
3. The data loader will erase all negative intensity to extract the boundary information appropriately. Please add an offset to CT to make it all positive if used. 
4. There are two versions of training script provided. In our some other experiments, it appears trainV2 is more stable than the version introduced in paper. We will share more information soon. 
5. Unfortunately, we need more time to integrate FL into this code space. My collaborator (Yuhao) is working on it. 
6. For MPL, because of the memory limitation in our GPU, we have not tested on batch size over 1, and the current data loading may not work well for larger batch size. We are working to improve it.
7. Directly using AMP does not really work for MAPSeg (degraded performance). There are some helpful discussions [here](https://github.com/facebookresearch/mae/issues/42). Adding gradient clipping might [help](https://github.com/facebookresearch/mae/issues/42#issuecomment-1327427371)
8. More to be updated. We have some exciting news regarding MAPSeg's extension applications (beyond cardiac and brain), stay tuned!

## Acknowledgements: 
Some components are borrowed from existing excellent repos, including patchify/unpatchify from [MAE](https://github.com/facebookresearch/mae), building block from [3D UNet](https://github.com/wolny/pytorch-3dunet), and [DeepLabV3](https://github.com/VainF/DeepLabV3Plus-Pytorch). We thank the authors for their open-source contribution!

# Cite:
If you found our work helpful, please cite our work:

    @InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Xuzhe and Wu, Yuhao and Angelini, Elsa and Li, Ang and Guo, Jia and Rasmussen, Jerod M. and O'Connor, Thomas G. and Wadhwa, Pathik D. and Jackowski, Andrea Parolin and Li, Hai and Posner, Jonathan and Laine, Andrew F. and Wang, Yun},
    title     = {MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {5851-5862}}