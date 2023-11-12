
from CNN_train import CNN_Train

from work1 import settings

from handwriting_train import handwriting_train

dataset_path = settings.training_set_folder



if __name__ == "__main__":
    # 使用函数
    # handwriting_train(1)  # 训练epoch，并指定数据集路径
    CNN_Train(lr=0.001, momentum=0.9, num_epochs=10)
