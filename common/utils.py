# define dataset class to feed the model
import numpy as np 
import os
import sys
import time

class Dataset():
    def __init__(self,data,label):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._label = label
        assert(data.shape[0] == label.shape[0])
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples
            self._label = self.label[idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            label_rest_part = self.label[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._label = self.label[idx0]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end],self._label[start:end]

class ProgressBar():
    def __init__(self,worksum,info="",auto_display=True):
        self.worksum = worksum
        self.info = info
        self.finishsum = 0
        self.auto_display = auto_display
    def startjob(self):
        self.begin_time = time.time()
    def complete(self,num):
        self.gaptime = time.time() - self.begin_time
        self.finishsum += num
        if self.auto_display == True:
            self.display_progress_bar()
    def display_progress_bar(self):
        percent = self.finishsum * 100 / self.worksum
        eta_time = self.gaptime * 100 / (percent + 0.001) - self.gaptime
        strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
        str_log = ("%s %.2f %% %s %s/%s \t used:%ds eta:%d s" % (self.info,percent,strprogress,self.finishsum,self.worksum,self.gaptime,eta_time))
        sys.stdout.write('\r' + str_log)

def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def split_dataset(dataset, split_ratio, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split<min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set