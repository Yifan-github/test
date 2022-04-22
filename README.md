### Validation on ScanNet with SuperGlue

1. Download images from https://drive.google.com/file/d/13Po1HNFL7_7psAB4szim2t8SPpp3v__h/view?usp=sharing
2. Unzip it and arrange files `PoseCorr/data/scannet_dataset`
3. Validate the dataset by `python run.py --action val_pair --database_name scannet_test`. 
You should see some image pairs in `data/validate`, which have some points and 
its corresponding epipolar lines drawn on the image. You can check that the epipolar geometry is correct.
4. download weights from https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights
5. arrange files like `data/model/superglue/superglue_indoor.pth`
6. evaluate on the scannet dataset 
```bash
python run.py --action eval \ 
              --database_name scannet_test \ 
              --dataset_type pair \
              --cfg configs/eval/superglue_indoor_pretrain.yaml
```
You should see some results in `data/vis_cache`


### Example: Train on ScanNet

1. Arrange files as `PoseCorr/data/scannet_seq_dataset/scene0000_00/color`.
2. Download image pair information from LoFTR  https://drive.google.com/file/d/1kKjEdnxmh501wQRrFhKMQmjpq3aVsvS1/view?usp=sharing
3. Arrange index files as `PoseCorr/data/scannet_seq_dataset/scannet_indices/scene_data/train`
4. Validate dataset by
```bash
python run.py --action val_seq \
              --database_name scannet_example/scene0000_00/480_640
              --pair_name loftr
```
You should see some epipolar lines and ground truth matching in `data/validate`.
5. Test the example training set `python run.py --action val_example_dataset`
6. Train on the sample set: `python run.py --action train --cfg configs/train/example.yaml`

