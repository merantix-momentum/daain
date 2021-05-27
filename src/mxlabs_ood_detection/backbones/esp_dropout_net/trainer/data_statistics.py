import numpy as np
import torch


class DataStatistics:
    def __init__(self, loader, normVal=1.10):
        """
        :param dataset: dataset
        :param classes: number of classes in the dataset
        :param cached_data_file: location where cached file has to be stored
        :param normVal: normalization value, as defined in ERFNet paper
        """
        self.num_classes = loader.dataset.num_classes + 1  # for the void class
        self.class_weights = np.ones(self.num_classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)

        self.ignore_index = loader.dataset.ignore_index

        self._process_loader(loader)

    def compute_class_weights(self, histogram):
        """
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the class_weights variable
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.num_classes):
            self.class_weights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def _process_loader(self, loader):
        """
        Function to read the data
        :param fileName: file that stores the image locations
        :param trainStg: if processing training or validation data
        :return: 0 if successful
        """
        global_hist = np.zeros(self.num_classes, dtype=np.float32)

        for batch_idx, (imgs, targets) in enumerate(loader):
            for img_batch_idx in range(imgs.shape[0]):
                hist = np.histogram(targets[img_batch_idx].cpu().numpy(), self.num_classes)
                global_hist += hist[0]

                # TODO improve this
                self.mean[0] += torch.mean(imgs[img_batch_idx, :, :, 0])
                self.mean[1] += torch.mean(imgs[img_batch_idx, :, :, 1])
                self.mean[2] += torch.mean(imgs[img_batch_idx, :, :, 2])

                self.std[0] += torch.std(imgs[img_batch_idx, :, :, 0])
                self.std[1] += torch.std(imgs[img_batch_idx, :, :, 1])
                self.std[2] += torch.std(imgs[img_batch_idx, :, :, 2])

        # divide the mean and std values by the sample space size
        self.mean /= len(loader.dataset)
        self.std /= len(loader.dataset)

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
