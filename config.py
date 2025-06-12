version = '3'
storage = '25'
epoch = 'best'

choose_name = ''            # train + train pro
# choose_name = '(train0)'     # train + train pro
# choose_name = '(train)'     # 2062 + 27
# choose_name = '(val)'     # 1017
# choose_name = '(test)'   #  1017
# choose_name = '(2034)'   # val + test
# choose_name = '(4096 + 27)'    # train + val + test (不带pro)
# choose_name = '(all)'    # train0 + val + test (带pro)

# choose_name = '(ttt)'            # ttt





##################################################
# Training Config
##################################################
GPU = '0'                   # GPU
workers = 0                 # number of Dataloader workers
epochs = 300                # number of epochs  160
batch_size = 1           # batch size  16
learning_rate = 1e-4        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (448, 448)     # size of training images
# image_size = (224, 224)     # size of training images
net = 'resnet50'  # feature extractor  # 'vgg19', 'vgg19_bn', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inception_mixed_6e', 'inception_mixed_7c'
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'mydataset'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models


# save_dir = './FGVC/CUB-200-2011/' + 'ckpt - ' + version + ' - storage/' + storage + '/'   # storage
save_dir = './FGVC/CUB-200-2011/ckpt - ' + version + '/'   # 没storage/
model_name = 'model_epoch_' + epoch + '.ckpt'
log_name = 'train.log'


# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = False
eval_ckpt = save_dir + model_name
eval_savepath = './FGVC/CUB-200-2011/visualize - ' + version + '/'