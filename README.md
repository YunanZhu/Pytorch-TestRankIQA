# Pytorch-TestRankIQA
[RankIQA](https://github.com/xialeiliu/RankIQA) model files in Pytorch. Test [RankIQA](https://github.com/xialeiliu/RankIQA) on TID2013 or LIVE dataset in Pytorch.

## Prerequisites
* Win10 (Not tested on Ubuntu yet)
* Python 3.6
* Numpy 1.19.1
* Pytorch 1.2
* Pandas 1.1.1 (Just use it to [save the predictive scores](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/main.py#L75), so it is not necessary)
* Opencv 4.5

The above version is not mandatory, just because I ran the code in such an environment.

## Getting Started
`Python main.py --test_set "ur_path/TID2013/" --model_file "./pre-trained/FT_tid2013.caffemodel.pt" --test_file "./data/ft_tid2013_test.txt" --res_file "./result.csv"`

## About the pre-trained model files
I use [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch) to transform the [Caffe](http://caffe.berkeleyvision.org/) model file to Pytorch format.
You can find the pre-trained Caffe model files of RankIQA in [here](https://github.com/xialeiliu/RankIQA/tree/master/pre-trained).

## Tips
I cannot guarantee the correctness of the pre-trained pytorch model files.  
I just tried to reproduce the results showed in the [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Liu_RankIQA_Learning_From_ICCV_2017_paper.html).  
You can see the reproduced results of [TID2013](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20LIVE.xlsx) and [LIVE](https://github.com/YunanZhu/Pytorch-TestRankIQA/blob/main/results%20of%20RankIQA%20on%20TID2013.xlsx)
