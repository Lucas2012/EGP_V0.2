# EvolvingGraphicalPlanner

1. Download the Precomputing ResNet Image Features, and extract them into `img_features/`:
```
mkdir -p img_features/
cd img_features/
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1 -O ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip
cd ..
```
(In case the URL above doesn't work, it is likely because the Room-to-Room dataset changed its feature URLs. You can find the latest download links [here](https://github.com/peteanderson80/Matterport3DSimulator#precomputing-resnet-image-features).)

After this step, `img_features/` should contain `ResNet-152-imagenet.tsv`. (Note that you only need to download the features extracted from ImageNet-pretrained ResNet to run the following experiments. Places-pretrained ResNet features or actual images are not required.)


2. Download the R2R dataset:
```
./tasks/R2R/data/download.sh
```

3. Run the code with following command:
```
./r2r.sh
```

The main module in EGP: (1) the planner that constructs graphs and prepare features to feed in Graph Nets; (2) the execution module that follows the sampled action to jump to one node on the graph; (3) "follow-gt" as the supervision strategy which selects node with highest metric on trajectory overlap.

The graph model in this repo is a bit different (outdated) from the main paper model, but should also work.

```
@article{deng2020evolving,
  title={Evolving graphical planner: Contextual global planning for vision-and-language navigation},
  author={Deng, Zhiwei and Narasimhan, Karthik and Russakovsky, Olga},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={20660--20672},
  year={2020}
}
```

Acknowledgement: This code repo is based on "https://github.com/chihyaoma/selfmonitoring-agent" and "https://github.com/ronghanghu/speaker_follower"
