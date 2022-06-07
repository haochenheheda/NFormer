# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# without re-ranking
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/home/haochen/workspace/project/NFORMER')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('test/nformer_model.pth')" TEST.TEST_NFORMER "('no')"

