import os, argparse, sys
import numpy as np
from keras.models import load_model, Model
from dm_image import DMImageDataGenerator
from dm_keras_ext import (
    get_dl_model,  
    load_dat_ram,
    do_3stage_training,
    DMFlush
)
from dm_multi_gpu import make_parallel
import warnings
# import exceptions
# warnings.filterwarnings('ignore', category=exceptions.UserWarning)


def run(train_dir, val_dir, test_dir,
        img_size=[256, 256], img_scale=None, rescale_factor=None,
        featurewise_center=True, featurewise_mean=59.6,
        equalize_hist=True, augmentation=False,
        class_list=['background', 'malignant', 'benign'],
        batch_size=64, train_bs_multiplier=.5, nb_epoch=5,
        top_layer_epochs=10, all_layer_epochs=20,
        load_val_ram=False, load_train_ram=False,
        net='resnet50', use_pretrained=True,
        nb_init_filter=32, init_filter_size=5, init_conv_stride=2,
        pool_size=2, pool_stride=2,
        weight_decay=.0001, weight_decay2=.0001,
        alpha=.0001, l1_ratio=.0,
        inp_dropout=.0, hidden_dropout=.0, hidden_dropout2=.0,
        optim='sgd', init_lr=.01, lr_patience=10, es_patience=25,
        resume_from=None, auto_batch_balance=False,
        pos_cls_weight=1.0, neg_cls_weight=1.0,
        top_layer_nb=None, top_layer_multiplier=.1, all_layer_multiplier=.01,
        best_model='./modelState/patch_clf.h5',
        final_model="NOSAVE"):
    '''Train a deep learning model for patch classifications
    '''
    #给块分类训练一个深度学习模型
    # ======= Environmental variables ======== #
    random_seed = int(os.getenv('RANDOM_SEED', 12345))
    nb_worker = int(os.getenv('NUM_CPU_CORES', 4))
    gpu_count = int(os.getenv('NUM_GPU_DEVICES', 1))

    # ========= Image generator ============== #图片生成
    if featurewise_center:#数据集去中心化
        train_imgen = DMImageDataGenerator(featurewise_center=True)
        val_imgen = DMImageDataGenerator(featurewise_center=True)
        test_imgen = DMImageDataGenerator(featurewise_center=True)
        train_imgen.mean = featurewise_mean
        val_imgen.mean = featurewise_mean
        test_imgen.mean = featurewise_mean
    else:
        train_imgen = DMImageDataGenerator()
        val_imgen = DMImageDataGenerator()
        test_imgen = DMImageDataGenerator()

    # Add augmentation options.
    #图像增强
    if augmentation:
        train_imgen.horizontal_flip = True #进行随机水平翻转
        train_imgen.vertical_flip = True#进行随机垂直翻转
        train_imgen.rotation_range = 25.  # in degree.#整数，数据提升时图片随机转动的角度
        train_imgen.shear_range = .2  # in radians.浮点数，剪切强度（逆时针方向的剪切变换角度）
        train_imgen.zoom_range = [.8, 1.2]  # in proportion.
        '''
        浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        '''
        train_imgen.channel_shift_range = 20.  # in pixel intensity values.
        #.浮点数，随机通道偏移的幅度
        #通过对颜色通道的数值偏移，改变图片的整体的颜色

    # ================= Model creation ============== #模型创建
    '''
    一、weight decay（权值衰减）使用的目的是防止过拟合。
    在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，
    所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。
    hidden_dropout 防止过拟合
    init_conv_stride 卷积核步幅大小
    pool_size 池化层大小，pool_stride 池化层步幅（一般是最大值池化，和平均值）
    alpha 给图像添加透明度
    l1_ratio 交叉验证选择l1和l2惩罚之间的折中,类可以通过交叉验证来设置 alpha(α) 和 l1_ratio(ρ) **参数 :l1_ratio 参数来控制L1和L2的凸组合
    inp_dropout 输入权重随机抛弃
    '''
    model, preprocess_input, top_layer_nb = get_dl_model(
        net, nb_class=len(class_list), use_pretrained=use_pretrained,
        resume_from=resume_from, img_size=img_size, top_layer_nb=top_layer_nb,
        weight_decay=weight_decay, hidden_dropout=hidden_dropout,
        nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
        init_conv_stride=init_conv_stride, pool_size=pool_size,
        pool_stride=pool_stride, alpha=alpha, l1_ratio=l1_ratio,
        inp_dropout=inp_dropout)
    if featurewise_center:
        preprocess_input = None
    if gpu_count > 1:
        model, org_model = make_parallel(model, gpu_count)#并行计算
    else:
        org_model = model

    # ============ Train & validation set =============== #
    #训练和验证集
    train_bs = int(batch_size*train_bs_multiplier)#每批数据量的大小*乘数
    if net != 'yaroslav':#dm_keras_ext.py
        dup_3_channels = True
    else:
        dup_3_channels = False
    if load_train_ram:
        raw_imgen = DMImageDataGenerator()#t图片数据生成器
        #创建行训练集数据生成器
        print ("Create generator for raw train set")
        #以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
        '''
        equalize_hist 直方图均衡，
        shuffle 随机打乱数据
        '''
        raw_generator = raw_imgen.flow_from_directory(
            train_dir, target_size=img_size, target_scale=img_scale,
            rescale_factor=rescale_factor,
            equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
            classes=class_list, class_mode='categorical',
            batch_size=train_bs, shuffle=False)
        #加载行训练数据集到内存
        print ("Loading raw train set into RAM.",sys.stdout.flush())
        #行数据集
        raw_set = load_dat_ram(raw_generator, raw_generator.nb_sample)
        print ("Done."); sys.stdout.flush()
        #为训练集创建生成器
        print ("Create generator for train set")
        #接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
        train_generator = train_imgen.flow(
            raw_set[0], raw_set[1], batch_size=train_bs,
            auto_batch_balance=auto_batch_balance, preprocess=preprocess_input,
            shuffle=True, seed=random_seed)
    else:
        print ("Create generator for train set")
        #以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
        train_generator = train_imgen.flow_from_directory(
            train_dir, target_size=img_size, target_scale=img_scale,
            rescale_factor=rescale_factor,
            equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
            classes=class_list, class_mode='categorical',
            auto_batch_balance=auto_batch_balance, batch_size=train_bs,
            preprocess=preprocess_input, shuffle=True, seed=random_seed)
    #创建验证集生成器
    print ("Create generator for val set")
    # 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    validation_set = val_imgen.flow_from_directory(
        val_dir, target_size=img_size, target_scale=img_scale,
        rescale_factor=rescale_factor,
        equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
        classes=class_list, class_mode='categorical',
        batch_size=batch_size, preprocess=preprocess_input, shuffle=False)
    sys.stdout.flush()
    #是否加载验证集到内存中
    if load_val_ram:
        print ("Loading validation set into RAM.",
        sys.stdout.flush())
        validation_set = load_dat_ram(validation_set, validation_set.nb_sample)
        print ("Done."); sys.stdout.flush()

    # ==================== Model training ==================== #模型训练
    # Do 3-stage training.三个阶段训练
    train_batches = int(train_generator.nb_sample/train_bs) + 1
    #判断验证集是否三元组
    if isinstance(validation_set, tuple):
        val_samples = len(validation_set[0])
    else:
        val_samples = validation_set.nb_sample
    validation_steps = int(val_samples/batch_size)
    #### DEBUG ####
    # val_samples = 100
    #### DEBUG ####
    # import pdb; pdb.set_trace()
    #通过三阶段训练得到模型，损失率，准确率
    model, loss_hist, acc_hist = do_3stage_training(
        model, org_model, train_generator, validation_set, validation_steps,
        best_model, train_batches, top_layer_nb, net, nb_epoch=nb_epoch,
        top_layer_epochs=top_layer_epochs, all_layer_epochs=all_layer_epochs,
        use_pretrained=use_pretrained, optim=optim, init_lr=init_lr,
        top_layer_multiplier=top_layer_multiplier,
        all_layer_multiplier=all_layer_multiplier,
        es_patience=es_patience, lr_patience=lr_patience,
        auto_batch_balance=auto_batch_balance, nb_class=len(class_list),
        pos_cls_weight=pos_cls_weight, neg_cls_weight=neg_cls_weight,
        nb_worker=nb_worker, weight_decay2=weight_decay2,
        hidden_dropout2=hidden_dropout2)

    # Training report.
    #训练报告
    if len(loss_hist) > 0:
        min_loss_locs, = np.where(loss_hist == min(loss_hist))
        best_val_loss = loss_hist[min_loss_locs[0]]
        best_val_accuracy = acc_hist[min_loss_locs[0]]
        print ("\n==== Training summary ====")
        print ("Minimum val loss achieved at epoch:", min_loss_locs[0] + 1)
        print ("Best val loss:", best_val_loss)
        print ("Best val accuracy:", best_val_accuracy)
#保存模型
    if final_model != "NOSAVE":
        model.save(final_model)

    # ==== Predict on test set ==== #
    #基于测试集的预测
    print ("\n==== Predicting on test set ====")
    # 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    test_generator = test_imgen.flow_from_directory(
        test_dir, target_size=img_size, target_scale=img_scale,
        rescale_factor=rescale_factor,
        equalize_hist=equalize_hist, dup_3_channels=dup_3_channels,
        classes=class_list, class_mode='categorical', batch_size=batch_size,
        preprocess=preprocess_input, shuffle=False)
    print ("Test samples =", test_generator.nb_sample)
    #加载最好的模型
    print ("Load saved best model:", best_model + '.',
    sys.stdout.flush())
    #原始模型加载最好模型的权重
    org_model.load_weights(best_model)
    print ("Done.")
    #测试的步数
    test_steps = int(test_generator.nb_sample/batch_size)
    #### DEBUG ####
    # test_samples = 10
    #### DEBUG ####
    test_res = model.evaluate_generator(
        test_generator, test_steps, nb_worker=nb_worker,
        pickle_safe=True if nb_worker > 1 else False)
    print ("Evaluation result on test set:", test_res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DM patch clf training")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("val_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("--img-size", "-is", dest="img_size", nargs=2, type=int,
                        default=[256, 256])
    parser.add_argument("--img-scale", "-ic", dest="img_scale", type=float, default=None)
    parser.add_argument("--no-img-scale", "-nic", dest="img_scale", action="store_const", const=None)
    parser.add_argument("--rescale-factor", dest="rescale_factor", type=float, default=None)
    parser.add_argument("--no-rescale-factor", dest="rescale_factor", action="store_const", const=None)
    parser.add_argument("--featurewise-center", dest="featurewise_center", action="store_true")
    parser.add_argument("--no-featurewise-center", dest="featurewise_center", action="store_false")
    parser.set_defaults(featurewise_center=True)
    parser.add_argument("--featurewise-mean", dest="featurewise_mean", type=float, default=59.6)
    parser.add_argument("--equalize-hist", dest="equalize_hist", action="store_true")
    parser.add_argument("--no-equalize-hist", dest="equalize_hist", action="store_false")
    parser.set_defaults(equalize_hist=True)
    parser.add_argument("--batch-size", "-bs", dest="batch_size", type=int, default=64)
    parser.add_argument("--train-bs-multiplier", dest="train_bs_multiplier", type=float, default=.5)
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=False)
    parser.add_argument("--class-list", dest="class_list", nargs='+', type=str,
                        default=['background', 'malignant', 'benign'])
    parser.add_argument("--nb-epoch", "-ne", dest="nb_epoch", type=int, default=5)
    parser.add_argument("--top-layer-epochs", dest="top_layer_epochs", type=int, default=10)
    parser.add_argument("--all-layer-epochs", dest="all_layer_epochs", type=int, default=20)
    parser.add_argument("--load-val-ram", dest="load_val_ram", action="store_true")
    parser.add_argument("--no-load-val-ram", dest="load_val_ram", action="store_false")
    parser.set_defaults(load_val_ram=False)
    parser.add_argument("--load-train-ram", dest="load_train_ram", action="store_true")
    parser.add_argument("--no-load-train-ram", dest="load_train_ram", action="store_false")
    parser.set_defaults(load_train_ram=False)
    parser.add_argument("--net", dest="net", type=str, default="resnet50")
    parser.add_argument("--nb-init-filter", "-nif", dest="nb_init_filter", type=int, default=32)
    parser.add_argument("--init-filter-size", "-ifs", dest="init_filter_size", type=int, default=5)
    parser.add_argument("--init-conv-stride", "-ics", dest="init_conv_stride", type=int, default=2)
    parser.add_argument("--max-pooling-size", "-mps", dest="pool_size", type=int, default=2)
    parser.add_argument("--max-pooling-stride", "-mpr", dest="pool_stride", type=int, default=2)
    parser.add_argument("--weight-decay", "-wd", dest="weight_decay", type=float, default=.0001)
    parser.add_argument("--weight-decay2", "-wd2", dest="weight_decay2", type=float, default=.0001)
    parser.add_argument("--alpha", dest="alpha", type=float, default=.0001)
    parser.add_argument("--l1-ratio", dest="l1_ratio", type=float, default=.0)
    parser.add_argument("--inp-dropout", "-id", dest="inp_dropout", type=float, default=.0)
    parser.add_argument("--hidden-dropout", "-hd", dest="hidden_dropout", type=float, default=.0)
    parser.add_argument("--hidden-dropout2", "-hd2", dest="hidden_dropout2", type=float, default=.0)
    parser.add_argument("--optimizer", dest="optim", type=str, default="sgd")
    parser.add_argument("--init-learningrate", "-ilr", dest="init_lr", type=float, default=.01)
    parser.add_argument("--lr-patience", "-lrp", dest="lr_patience", type=int, default=10)
    parser.add_argument("--es-patience", "-esp", dest="es_patience", type=int, default=25)
    parser.add_argument("--resume-from", dest="resume_from", type=str, default=None)
    parser.add_argument("--no-resume-from", dest="resume_from", action="store_const", const=None)
    parser.add_argument("--auto-batch-balance", dest="auto_batch_balance", action="store_true")
    parser.add_argument("--no-auto-batch-balance", dest="auto_batch_balance", action="store_false")
    parser.set_defaults(auto_batch_balance=False)
    parser.add_argument("--pos-cls-weight", dest="pos_cls_weight", type=float, default=1.0)
    parser.add_argument("--neg-cls-weight", dest="neg_cls_weight", type=float, default=1.0)
    parser.add_argument("--use-pretrained", dest="use_pretrained", action="store_true")
    parser.add_argument("--no-use-pretrained", dest="use_pretrained", action="store_false")
    parser.set_defaults(use_pretrained=True)
    parser.add_argument("--top-layer-nb", dest="top_layer_nb", type=int, default=None)
    parser.add_argument("--no-top-layer-nb", dest="top_layer_nb", action="store_const", const=None)
    parser.add_argument("--top-layer-multiplier", dest="top_layer_multiplier", type=float, default=.1)
    parser.add_argument("--all-layer-multiplier", dest="all_layer_multiplier", type=float, default=.01)
    parser.add_argument("--best-model", "-bm", dest="best_model", type=str,
                        default="./modelState/patch_clf.h5")
    parser.add_argument("--final-model", "-fm", dest="final_model", type=str,
                        default="NOSAVE")

    args = parser.parse_args()
    #字典
    run_opts = dict(
        img_size=args.img_size, 
        img_scale=args.img_scale, 
        rescale_factor=args.rescale_factor,
        featurewise_center=args.featurewise_center,
        featurewise_mean=args.featurewise_mean,
        equalize_hist=args.equalize_hist,
        batch_size=args.batch_size, 
        train_bs_multiplier=args.train_bs_multiplier,
        augmentation=args.augmentation,
        class_list=args.class_list,
        nb_epoch=args.nb_epoch, 
        top_layer_epochs=args.top_layer_epochs,
        all_layer_epochs=args.all_layer_epochs,
        load_val_ram=args.load_val_ram,
        load_train_ram=args.load_train_ram,
        net=args.net,
        nb_init_filter=args.nb_init_filter, 
        init_filter_size=args.init_filter_size, 
        init_conv_stride=args.init_conv_stride, 
        pool_size=args.pool_size, 
        pool_stride=args.pool_stride, 
        weight_decay=args.weight_decay,
        weight_decay2=args.weight_decay2,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        inp_dropout=args.inp_dropout,
        hidden_dropout=args.hidden_dropout,
        hidden_dropout2=args.hidden_dropout2,
        optim=args.optim,
        init_lr=args.init_lr,
        lr_patience=args.lr_patience, 
        es_patience=args.es_patience,
        resume_from=args.resume_from,
        auto_batch_balance=args.auto_batch_balance,
        pos_cls_weight=args.pos_cls_weight,
        neg_cls_weight=args.neg_cls_weight,
        use_pretrained=args.use_pretrained,
        top_layer_nb=args.top_layer_nb,
        top_layer_multiplier=args.top_layer_multiplier,
        all_layer_multiplier=args.all_layer_multiplier,
        best_model=args.best_model,        
        final_model=args.final_model        
    )
    print ("\ntrain_dir=%s" % (args.train_dir))
    print ("val_dir=%s" % (args.val_dir))
    print ("test_dir=%s" % (args.test_dir))
    print ("\n>>> Model training options: <<<\n", run_opts, "\n")
    run(args.train_dir, args.val_dir, args.test_dir, **run_opts)









