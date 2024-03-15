class config:
    def __init__(self):
        # network
        self.mod = 'train'
        self.name = 'models.backbones.ST_GCN_18'

        self.in_channels = 3

        self.edge_importance_weighting = True
        self.layout = 'coco'
        self.strategy = 'spatial'
        self.seed = 1

        # dataset
        self.dataset_name = 'NTU'  # Dataset name：CMU or NTU

        # train
        self.train_dataset = '/home/yosubs/videosync/datasets/nturgbd/NTU-SYN/pose/train'
        self.num_track = 1
        self.num_keypoints = 17
        self.train_batchsize = 1

        # test
        self.test_dataset = '/home/yosubs/videosync/datasets/nturgbd/NTU-SYN/pose/test'
        self.test_batchsize = 1
        self.resume = True

        # model
        self.checkpoint_path = 'path to model'

        # others
        self.work_dir = '/home/yosubs/videosync/videosync/log'
        self.max_epochs = 400
        self.save_fre = 1
        self.workers = 3
        self.gpus = 1
