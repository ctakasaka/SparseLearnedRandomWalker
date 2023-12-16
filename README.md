# pytorch-SparseLearnedRandomWalker
Modification of the LearnedRandomWalker module as described in:
* Paper: [End-To-End Learned Random Walker for Seeded Image Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cerrone_End-To-End_Learned_Random_Walker_for_Seeded_Image_Segmentation_CVPR_2019_paper.pdf)  
* [Supplementary Material](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Cerrone_End-To-End_Learned_Random_CVPR_2019_supplemental.pdf)  
* [CVPR2019 Poster](./data/cvpr19_LRW_poster.pdf)

## Data Loading:
After downloading Cremi data into the data directory, enter the data directory and run the command:
```
python setup.py install
```

This will add the "cremi" dataset package to your Python path. To test the install,
```
python exampleDataloader.py
```

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

