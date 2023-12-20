# pytorch-SparseLearnedRandomWalker
Modification of the LearnedRandomWalker module as described in:
* Paper: [End-To-End Learned Random Walker for Seeded Image Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.pdf)  
* [Supplementary Material](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Cerrone_End-To-End_Learned_Random_CVPR_2019_supplemental.pdf)  
* [CVPR2019 Poster](./data/cvpr19_LRW_poster.pdf)

## Data Procurement:
Download CREMI dataset A (training volume) [here](https://cremi.org/static/data/sample_A_20160501.hdf)

This hdf file should be in the data/ directory of the project.

## Environment Generation:
A yaml file is provided in the project to generate a working conda environment.
```
conda env create -f cs_684_slrw.yml
```
After activating the environment (named `srw`), please run,
```
pip install xxhash
```
Conda stores an outdated version of the package.

## Project uses code from:
```
@inproceedings{cerrone2019,
  title={End-to-end learned random walker for seeded image segmentation},
  author={Cerrone, Lorenzo and Zeilmann, Alexander and Hamprecht, Fred A},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12559--12568},
  year={2019}
}
```

