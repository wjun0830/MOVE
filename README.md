# Pytorch Implementation for "Minority-Oriented Vicinity Expansion with Attentive Aggregation for Video Long-Tailed Recognition" (AAAI 2023 Oral) and Imbalanced-MiniKinetics200 dataset.

[Arxiv](https://arxiv.org/abs/2211.13471) | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25284/25056) | [Imbalanced-MiniKinetics200](https://drive.google.com/drive/folders/1ZaXo0a7eEStSaIfv1ql6Qex8AeVl2VOX?usp=sharing)

## 1. Requirements & Environments
To run the code, you need to install requirements below.
We suggest to work with torch version (1.2 ~ 1.7.1).
Other versions may work fine but we haven't tested them.

``
pip install -r requirements.txt
``


## 2. Training & Evaluation
Please refer to subdirectories for training each [VideoLT](VideoLT) and [Imbalanced-MiniKinetics200](Imbalanced-MiniKinetics200) dataset. 


##  Cite MOVE 
If you find this repository useful, please use the following entry for citation.
```
@inproceedings{moon2023minority,
  title={Minority-Oriented Vicinity Expansion with Attentive Aggregation for Video Long-Tailed Recognition},
  author={Moon, WonJun and Seong, Hyun Seok and Heo, Jae-Pil},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={1931--1939},
  year={2023}
}
```

## Contributors and Contact
If there are any questions, feel free to contact with the authors: [WonJun Moon](wjun0830@gmail.com) and [Hyun Seok Seong](gustjrdl95@gmail.com).

## Acknowledgement
This repository is built based on [VideoLT](https://github.com/17Skye17/VideoLT) repository.

