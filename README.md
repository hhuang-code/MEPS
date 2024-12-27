# Learning to learn point signature for 3D shape geometry

## Primary dependencies
- Python 3.9+
- PyTorch 2.0+
- Cuda 11.8
- Trimesh 4.1
- Matlab R2023a

Or install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate meps
```
```angular2html
# NOTE: The versions of the dependencies listed above are only for reference, 
and please check https://pytorch.org/ for pytorch installation command for your CUDA version.
```

## Shape correspondence

### Data preparation
The dataset for shape correspondence can be downloaded from here: [MPI-FAUST](https://drive.google.com/file/d/11g9ovtOSdfd9cuy_TRCwI8zqD6C-uTd0/view?usp=sharing).
After downloading, put it into the `corr` folder as the structure:

```angular2html
MEPS/
└── corr/
    └── MPI-FAUST/
       └── xiang/
           ├── tr_reg_000.npy
           ├── ...
           └── tr_reg_099.npy
```

### Training
To train the model on the MPI-FAUST dataset, run
```bash
cd corr
bash scripts/faust_train.sh 
```
Modify the other argument values as needed. 
By default, the log will be saved in the `log` folder, and the model will be saved in the `checkpoint` folder.

### Evaluation
To evaluate the trained model on the MPI-FAUST dataset, run
```bash
cd corr
# Evaluate without refinement
bash scripts/faust_prob.sh
# Refinement
matlab -nodisplay -nosplash -r "run('prob2corr.m'); exit;"
# Then, evaluate again with refinement
bash scripts/faust_prob.sh
# Plot geodesic error curve
chmod +777 example0
bash scripts/faust_curve.sh
```

### More visualizations
To generate shape matches, run
```bash
cd corr
bash scripts/faust_vis_matches.sh
```
To generate t-SNE visualization, run
```bash
cd corr
bash scripts/faust_vis_tsne.sh
```
To generate geodeisc errors, run
```bash
cd corr
bash scripts/faust_vis_errors.sh
```

## Shape segmentation

### Data preparation
The dataset for shape part segmentation can be downloaded from here: [ShapeNet](https://drive.google.com/file/d/1OptHSEvgDKELFHGswv6k-dfm-QJy8TaV/view?usp=sharing).
After downloading, put it into the `seg` folder as the structure:
```angular2html
MEPS/
└── seg/
    └── shapenet_part_seg_hdf5_data/
        ├── all_object_categories.txt
        ├── catid_partid_to_overallid.json
        ├── color_partid_catid_map.txt
        ├── overallid_to_catid_partid.json
        ├── part_color_mapping.json
        ├── ply_data_train0.h5
        ├── ...
        ├── ply_data_val0.h5
        ├── ply_data_test0.h5
        ├── ...
        ├── train_hdf5_file_list.txt
        ├── val_hdf5_file_list.txt
        └── test_hdf5_file_list.txt
```

### Training
To train the model on the ShapeNet dataset, run
```bash
cd seg
bash scripts/shapenet_train.sh 
```
Modify the other argument values as needed. 
By default, the log will be saved in the `log` folder and model will be saved in the`checkpoint` folder.

### Evaluation
To evaluate the model trained on ShapeNet, run
```bash
cd seg
bash scripts/shapenet_test.sh
```

### Visualization
To visualize the base-learner parameter weights, run
```bash
cd seg
bash scripts/shapenet_vis_weights.sh
```

## Acknowledgement
Our code is built upon the repositories [FeaStNet](https://github.com/nitika-verma/FeaStNet).
The matlab refinement library code is borrowed from [KernelMatching](https://github.com/zorah/KernelMatching/tree/master/tools/fastAuction_v2.5).
We would appreciate their authors.

## Citation
```angular2html
If you found this repository is helpful, please cite:

@article{huang2024learning,
  title={Learning to learn point signature for 3D shape geometry},
  author={Huang, Hao and Wang, Lingjing and Li, Xiang and Yuan, Shuaihang and Wen, Congcong and Hao, Yu and Fang, Yi},
  journal={Pattern Recognition Letters},
  year={2024},
  publisher={Elsevier}
}
```
