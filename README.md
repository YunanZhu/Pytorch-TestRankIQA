# Pytorch-TestRankIQA
RankIQA was proposed in a ICCV2017 paper by [Liu X](https://github.com/xialeiliu). You can get this paper from [arXiv](https://arxiv.org/abs/1707.08347v1) or [ICCV 2017 open access](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html).

This repo contains [RankIQA](https://github.com/xialeiliu/RankIQA) model files in Pytorch. And you can test RankIQA on TID2013 and LIVE dataset in Pytorch.

## Prerequisites
* Win10 (Not tested on Ubuntu yet)
* Python 3.6
* Numpy 1.19.1
* Pytorch 1.2
* Pandas 1.1.1 (Just use it to [save the predictive scores](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/main.py#L75), so it is not necessary)
* Opencv 4.5

The above versions are not mandatory, just because I ran the code in such an environment.

## Getting Started
```
python main.py --test_set "ur_path/TID2013/" --model_file "./pre-trained/Rank_tid2013.caffemodel.pt" --test_file "./data/ft_tid2013_test.txt" --res_file "./result.csv"
python main.py --test_set "ur_path/TID2013/" --model_file "./pre-trained/FT_tid2013.caffemodel.pt" --test_file "./data/ft_tid2013_test.txt" --res_file "./result.csv"

python main.py --test_set "ur_path/LIVE2/" --model_file "./pre-trained/Rank_live.caffemodel.pt" --test_file "./data/ft_live_test.txt" --res_file "./result.csv"
python main.py --test_set "ur_path/LIVE2/" --model_file "./pre-trained/FT_live.caffemodel.pt" --test_file "./data/ft_live_test.txt" --res_file "./result.csv"
```
Note: ```test_set``` is the dataset folder, ```model_file``` is the pre-trained model file, ```test_file``` is the txt file which contains MOS and image filenames (see [here](https://github.com/xialeiliu/RankIQA/tree/master/data)), ```res_file``` is the csv file to save the test results.

## About the pre-trained model files
I use [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) to transform the [Caffe](http://caffe.berkeleyvision.org/) model file to Pytorch format.
You can find the pre-trained Caffe model files of RankIQA in [here](https://github.com/xialeiliu/RankIQA/tree/master/pre-trained).

***You can also download the Pytorch model files transformed by myself from:***

- ***[Baidu disk](https://pan.baidu.com/s/1HjYFypg-RWE-W-TvNQ-02A), and the password is ```riqa```.***
- ***[Google drive](https://drive.google.com/drive/folders/1OQ0IQrWoricMhaIyfwqsJVlYpXHKPP1z).***

## Tips
***I cannot guarantee the correctness of the pre-trained Pytorch model files and the test results.***

I just tried to reproduce the results showed in the [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html),
and you can see the reproduced results on [TID2013](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20LIVE.xlsx) and [LIVE](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20TID2013.xlsx).

I didn't write the training code. If you are familiar with Pytorch, you can modify the code and test RankIQA on other datasets.
