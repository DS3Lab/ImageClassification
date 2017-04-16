# ImageClassification

work:
###tensorflow

-AlexNet:

usage: python classify.py

###### One can change the hyperparameters(learning rate, batch size, training fraction) in the code

-vgg

caffe:

alexnet, vgg_2, googlenet, resnet2



dis_alexn:

- distribution implementation of AlexNet using Tensorflow (Between-graph replication + Asynchronous training)
- input [in correspondance with the settings in the example code]:

```
$sgs-gpu-01: python train.py --job_name="ps" --task_index=0

$sgs-gpu-02: python train.py --job_name="worker" --task_index=0

$sgs-gpu-03: python train.py --job_name="worker" --task_index=1
```
