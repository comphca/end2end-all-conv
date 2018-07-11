import numpy as np
import cv2
import pandas as pd
import os, sys, argparse
import pydicom
from dm_image import read_resize_img, crop_img, add_img_margins
from dm_preprocess import DMImagePreprocessor as imprep
from scipy.misc import toimage
from sklearn.model_selection import train_test_split


#### Define some functions to use ####
'''
    作用：对目录内的图片进行钙化或者肿块的命名
    输入参数：病人，左右，拍摄角度，目录，图像类型，异常
    输出（返回）：所在目录和重命名的png图片
'''
def const_filename(pat, side, view, directory, itype=None, abn=None):
    #获取列表，包括病人，左右，拍摄角度
    token_list = [pat, side, view]
    #图像类型不为空
    if itype is not None:
        #在token_list 0索引处中插入   钙化或者肿块的说明  -Training?
        token_list.insert(
            0, ('Calc' if itype == 'calc' else 'Mass') + '-Training')
        #列表后加异常说明
        token_list.append(str(abn))
    #命名病人图片
    fn = "_".join(token_list) + ".png"
    #返回目录和重命名的图片
    return os.path.join(directory, fn)
'''
    作用：将值截取至minv到maxv之间
    输入：需要判断的值，最小截取值，最大截取值
    输出（返回）：原值
'''
def crop_val(v, minv, maxv):
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v
'''
    作用：重叠区域的判断？
    输入：块中心（包含x，y坐标），块大小，感兴趣区域掩码，roi_patch_added加的值，截止值
    输出（返回）：后面说明
'''
def overlap_patch_roi(patch_center, patch_size, roi_mask,
                      add_val=1000, cutoff=.5):
    #取图像块的左下和右上坐标，定位图片
    x1,y1 = (patch_center[0] - patch_size/2,
             patch_center[1] - patch_size/2)
    x2,y2 = (patch_center[0] + patch_size/2,
             patch_center[1] + patch_size/2)
    #截取x1，y1到0至掩码列行数最大值之间，及裁剪图像至掩码大小
    # shape[0]读取行数，shape[1]读取列数
    x1 = crop_val(x1, 0, roi_mask.shape[1])
    y1 = crop_val(y1, 0, roi_mask.shape[0])
    x2 = crop_val(x2, 0, roi_mask.shape[1])
    y2 = crop_val(y2, 0, roi_mask.shape[0])
    #满足roi_mask>0的值的个数赋值给roi_area
    roi_area = (roi_mask>0).sum()
    #将掩码复制到roi_patch_added
    roi_patch_added = roi_mask.copy()
    #roi_patch_added中y1到y2行,x1到x2列都加此值
    roi_patch_added[y1:y2, x1:x2] += add_val
    #满足roi_patch_added>=add_val的值的个数赋值给patch_area
    patch_area = (roi_patch_added>=add_val).sum()
    #满足roi_patch_added>add_val的值的个数赋值给inter_area
    inter_area = (roi_patch_added>add_val).sum().astype('float32')
    #如果inter_area/roi_area或者  inter_area/patch_area大于截取值则为true
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)
'''
    作用：创建斑点检测器，来采集肿块，采样（最小区域面积）
    输入：感兴趣区域大小，斑点最小面积，
    输出（返回）：后面说明
'''
def create_blob_detector(roi_size=(128, 128), blob_min_area=3,
                         blob_min_int=.5, blob_max_int=.95, blob_th_step=10):
    #定义了一个名称为Params的结构体
    params = cv2.SimpleBlobDetector_Params()
    #斑点面积的限制变量
    params.filterByArea = True
    #斑点最小面积取输入值
    params.minArea = blob_min_area
    #斑点最大面积取感兴趣区域大小
    params.maxArea = roi_size[0]*roi_size[1]
    #斑点圆度的限制变量，默认是不限制
    params.filterByCircularity = False
    #斑点颜色的限制变量
    params.filterByColor = False
    #斑点凸度的限制变量
    params.filterByConvexity = False
    #斑点惯性率的限制变量
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    #二值化的起始阈值
    params.minThreshold = int(blob_min_int*255)
    #二值化的终止阈值
    params.maxThreshold = int(blob_max_int*255)
    #二值化的阈值步长
    params.thresholdStep = blob_th_step
    #返回params（版本不同返回函数不同）
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)

'''
    作用：
    输入：图像，感兴趣区域掩码，输出目录，图像ID，异常，阳性，块大小，
    输出（返回）：
'''
def sample_patches(img, roi_mask, out_dir, img_id, abn, pos, patch_size=256,
                   pos_cutoff=.75, neg_cutoff=.35,
                   nb_bkg=100, nb_abn=100, start_sample_nb=0, itype='calc',
                   bkg_dir='background',
                   calc_pos_dir='calc_mal', calc_neg_dir='calc_ben',
                   mass_pos_dir='mass_mal', mass_neg_dir='mass_ben',
                   verbose=False):
    #图像如果为阳性
    if pos:
        #是钙化点
        if itype == 'calc':
            #图像放入输出目录中钙化阳性目录
            roi_out = os.path.join(out_dir, calc_pos_dir)
        else:
            #否则放入输出目录中肿块阳性目录
            roi_out = os.path.join(out_dir, mass_pos_dir)
    #图像如果为阴性
    else:
        #是钙化点
        if itype == 'calc':
            #图像放入输出目录中钙化阴性目录
            roi_out = os.path.join(out_dir, calc_neg_dir)
        else:
            #否则放入输出目录中肿块阴性目录
            roi_out = os.path.join(out_dir, mass_neg_dir)
    #背景放到输出目录中背景目录下
    bkg_out = os.path.join(out_dir, bkg_dir)
    #命名
    basename = '_'.join([img_id, str(abn)])
    #图像增加边距
    img = add_img_margins(img, patch_size/2)
    #掩码增加边距
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.
    #获取感兴趣区域边界
    #掩码转为8位无符号整型
    roi_mask_8u = roi_mask.astype('uint8')
    #获取轮廓contours
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #获取轮廓面积
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    #返回轮廓面积的索引值
    idx = np.argmax(cont_areas)  # find the largest contour.
    #获取轮廓左上角点的坐标以及宽高
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    #显示详细信息为true
    if verbose:
        #将计算得到的矩以一个字典的形式返回至M
        M = cv2.moments(contours[idx])
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print ("ROI centroid=", (cx,cy)); sys.stdout.flush()
        except ZeroDivisionError:
            cx = rx + int(rw/2)
            cy = ry + int(rh/2)
            print ("ROI centroid=Unknown, use b-box center=", (cx,cy))
            sys.stdout.flush()
    #实现的随机数生成通常为伪随机数生成器，为了使得具备随机性的代码最终的结果可复现，需要设置相同的种子值
    rng = np.random.RandomState(12345)
    # Sample abnormality first.采样异常图片先进行
    sampled_abn = 0
    nb_try = 0
    while sampled_abn < nb_abn:
        if nb_abn > 1:#randint用于产生基质的均匀分布的随机整数
            x = rng.randint(rx, rx + rw)
            y = rng.randint(ry, ry + rh)
            nb_try += 1
            if nb_try >= 1000:
                # 试验的Nb达到最大值，重叠截止以0.05幅度降低
                print ("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
                sys.stdout.flush()
                pos_cutoff -= .05
                nb_try = 0
                if pos_cutoff <= .0:
                    # 重叠截止值达到了非阳性界限，检查感性区域掩码输入值
                    raise Exception("overlap cutoff becomes non-positive, "
                                    "check roi mask input.")

        else:
            x = cx
            y = cy
        # import pdb; pdb.set_trace()
        if nb_abn == 1 or overlap_patch_roi((x,y), patch_size, roi_mask,
                                            cutoff=pos_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            #max返回最大值，最小值
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            # patch = patch.reshape((patch.shape[0], patch.shape[1], 1))
            filename = basename + "_%04d" % (sampled_abn) + ".png"
            fullname = os.path.join(roi_out, filename)
            # import pdb; pdb.set_trace()
            patch_img.save(fullname)
            sampled_abn += 1#异常样本数量记录
            nb_try = 0
            if verbose:
                #将异常样本的块坐标显示
                print ("sampled an abn patch at (x,y) center=", (x,y))
                sys.stdout.flush()
    # Sample background.
    sampled_bkg = start_sample_nb
    while sampled_bkg < start_sample_nb + nb_bkg:
        x = rng.randint(patch_size/2, img.shape[1] - patch_size/2)
        y = rng.randint(patch_size/2, img.shape[0] - patch_size/2)
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            filename = basename + "_%04d" % (sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                #将样本背景中心块坐标显示
                print ("sampled a bkg patch at (x,y) center=", (x,y))
                sys.stdout.flush()
'''
    作用：难类样本采样
    输入：
    输出（返回）：保存样本背景并输出中心坐标
'''
def sample_hard_negatives(img, roi_mask, out_dir, img_id, abn,
                          patch_size=256, neg_cutoff=.35, nb_bkg=100,
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    #警告：hns（难类）的定义可能有问题。 已有研究表明ROI的背景对分类也很有用。
    '''WARNING: the definition of hns may be problematic.
    There has been study showing that the context of an ROI is also useful
    for classification.
    '''
    #添加背景目录并命名
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn)])
    #图像加0边距，掩码同样处理
    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    #获取ROI边框（上面有讲述）
    # Get ROI bounding box.
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print ("ROI centroid=", (cx,cy)); sys.stdout.flush()

    rng = np.random.RandomState(12345)
    # Sample hard negative samples.难类样本采样
    #初始化样本背景
    sampled_bkg = start_sample_nb
    #当样本背景小于初始样本数加背景数时进行循环
    while sampled_bkg < start_sample_nb + nb_bkg:
        #扩大感兴趣区域，取不小于patch_size/2,不超过img.shape - patch_size/2的x,y的随机值
        x1,x2 = (rx - patch_size/2, rx + rw + patch_size/2)
        y1,y2 = (ry - patch_size/2, ry + rh + patch_size/2)
        x1 = crop_val(x1, patch_size/2, img.shape[1] - patch_size/2)
        x2 = crop_val(x2, patch_size/2, img.shape[1] - patch_size/2)
        y1 = crop_val(y1, patch_size/2, img.shape[0] - patch_size/2)
        y2 = crop_val(y2, patch_size/2, img.shape[0] - patch_size/2)
        x = rng.randint(x1, x2)
        y = rng.randint(y1, y2)
        #如果patch与掩码不重叠则执行
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            #选取图像中patch
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            #patch由数组转为图像，并保存至背景
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            filename = basename + "_%04d" % (sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            sampled_bkg += 1
            if verbose:
                print ("sampled a hns patch at (x,y) center=", (x,y))
                sys.stdout.flush()

'''
    作用：阴性斑点样本采样
    输出：返回斑点数（保存斑点并输出中心坐标）
'''
def sample_blob_negatives(img, roi_mask, out_dir, img_id, abn, blob_detector,
                          patch_size=256, neg_cutoff=.35, nb_bkg=100,
                          start_sample_nb=0,
                          bkg_dir='background', verbose=False):
    #命名
    bkg_out = os.path.join(out_dir, bkg_dir)
    basename = '_'.join([img_id, str(abn)])
    #加0边距
    img = add_img_margins(img, patch_size/2)
    roi_mask = add_img_margins(roi_mask, patch_size/2)
    # Get ROI bounding box.获取ROI边界
    roi_mask_8u = roi_mask.astype('uint8')
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _,contours,_ = cv2.findContours(
            roi_mask_8u.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour.
    rx,ry,rw,rh = cv2.boundingRect(contours[idx])
    if verbose:
        M = cv2.moments(contours[idx])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print ("ROI centroid=", (cx,cy)); sys.stdout.flush()

    # Sample blob negative samples.阴性斑点样本采样
    #采取特征点？
    key_pts = blob_detector.detect((img/img.max()*255).astype('uint8'))
    rng = np.random.RandomState(12345)
    key_pts = rng.permutation(key_pts)
    sampled_bkg = 0
    #循环特征点
    for kp in key_pts:
        #样本背景数大于等于背景总数
        if sampled_bkg >= nb_bkg:
            break
        #获取特征点坐标
        x,y = int(kp.pt[0]), int(kp.pt[1])
        #如果patch与roi掩码不重叠则保存
        if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
            patch = img[y - patch_size/2:y + patch_size/2,
                        x - patch_size/2:x + patch_size/2]
            patch = patch.astype('int32')
            patch_img = toimage(patch, high=patch.max(), low=patch.min(),
                                mode='I')
            filename = basename + "_%04d" % (start_sample_nb + sampled_bkg) + ".png"
            fullname = os.path.join(bkg_out, filename)
            patch_img.save(fullname)
            if verbose:
                print ("sampled a blob patch at (x,y) center=", (x,y))
                sys.stdout.flush()
            sampled_bkg += 1
    return sampled_bkg

#### End of function definition ####


def run(roi_mask_path_file, roi_mask_dir, pat_train_list_file, full_img_dir,
        train_out_dir, val_out_dir,
        target_height=4096, target_width=None, patch_size=256,
        segment_breast=True,
        nb_bkg=30, nb_abn=30, nb_hns=15,
        pos_cutoff=.75, neg_cutoff=.35, val_size=.1,
        bkg_dir='background', calc_pos_dir='calc_mal', calc_neg_dir='calc_ben',
        mass_pos_dir='mass_mal', mass_neg_dir='mass_ben', verbose=True):

    # Print info for book-keeping.输出记录信息
    print ("Pathology file=", roi_mask_path_file)#病理文件
    print ("ROI mask dir=", roi_mask_dir)#ROI掩码路径
    print ("Patient train list=", pat_train_list_file)#病人训练表文件
    print ("Full image dir=", full_img_dir)#全图路径
    print ("Train out dir=", train_out_dir)#训练输出路径
    print ("Val out dir=", val_out_dir)#验证输出路径
    print ("===")
    sys.stdout.flush()

    # Read ROI mask table with pathology.读取病理中的ROI掩码表
    roi_mask_path_df = pd.read_csv(roi_mask_path_file, header=0)
    roi_mask_path_df = roi_mask_path_df.set_index(['patient_id', 'side', 'view'])
    roi_mask_path_df.sort_index(inplace=True)
    # Read train set patient IDs and subset the table.读取训练集病人ID和表的子集
    pat_train = pd.read_csv(pat_train_list_file, header=None)
    #将pet_train多维数组变为一维数组
    pat_train = pat_train.values.ravel()
    #？？？
    if len(pat_train) > 1:
        path_df = roi_mask_path_df.loc[pat_train.tolist()]
    else:
        locs = roi_mask_path_df.index.get_loc(pat_train[0])
        path_df = roi_mask_path_df.iloc[locs]
    # Determine the labels for patients.确定患者标签
    pat_labs = []
    #循环导出训练集病理中恶性的病人标签
    for pat in pat_train:
        #获取病理
        pathology = path_df.loc[pat]['pathology']
        #恶性
        malignant = 0
        for path in pathology:
            if path.startswith('MALIGNANT'):
                malignant = 1
                break
        pat_labs.append(malignant)
    # Split patient list into train and val lists.分割病人列表为训练和验证列表
    #写入病人列表
    def write_pat_list(fn, pat_list):
        with open(fn, 'w') as f:
            for pat in pat_list:
                f.write(str(pat) + "\n")
            f.close()
    if val_size > 0:
        # import pdb; pdb.set_trace()
        #分割训练集和验证集返回病人信息和标签
        pat_train, pat_val, labs_train, labs_val = train_test_split(
            pat_train, pat_labs, stratify=pat_labs, test_size=val_size,
            random_state=12345)
        #？？？
        if len(pat_val) > 1:
            val_df = roi_mask_path_df.loc[pat_val.tolist()]
        else:
            locs = roi_mask_path_df.index.get_loc(pat_val[0])
            val_df = roi_mask_path_df.iloc[locs]
        #验证集病人信息写入pat_lst.txt文件中
        write_pat_list(os.path.join(val_out_dir, 'pat_lst.txt'), pat_val.tolist())
    if len(pat_train) > 1:
        train_df = roi_mask_path_df.loc[pat_train.tolist()]
    else:
        locs = roi_mask_path_df.index.get_loc(pat_train[0])
        train_df = roi_mask_path_df.iloc[locs]
    #训练集病人信息写入文件
    write_pat_list(os.path.join(train_out_dir, 'pat_lst.txt'), pat_train.tolist())
    # Create a blob detector.创建斑点检测
    blob_detector = create_blob_detector(roi_size=(patch_size, patch_size))

    #### Define a functin to sample patches.定义块采集函数
    def do_sampling(pat_df, out_dir):
        #遍历病人信息
        for pat,side,view in pat_df.index.unique():
            #文件名命名
            full_fn = const_filename(pat, side, view, full_img_dir)
            # import pdb; pdb.set_trace()
            try:
                #如果未指定宽度则根据高度调整图片大小
                if target_width is None:
                    full_img = read_resize_img(
                        full_fn, target_height=target_height)
                else:
                    #根据宽高调整大小
                    full_img = read_resize_img(
                        full_fn, target_size=(target_height, target_width))
                #命名图片ID
                img_id = '_'.join([pat, side, view])
                print ("ID:%s, read image of size=%s" % (img_id, full_img.shape),)
                #图像中分割出乳腺为true
                if segment_breast:
                    #获取乳腺图像和坐标宽高
                    full_img, bbox = imprep.segment_breast(full_img)
                    #输出图像宽高通道数
                    print ("size after segmentation=%s" % (str(full_img.shape)))
                sys.stdout.flush()
                # Read mask image(s).读取掩码图
                #获取异常路径
                abn_path = roi_mask_path_df.loc[pat].loc[side].loc[view]
                #无论此路径是否为pd.Series类型，都要进行异常数，病理（良恶性），类型（肿块或钙化点）的数据获取
                if isinstance(abn_path, pd.Series):
                    abn_num = [abn_path['abn_num']]
                    pathology = [abn_path['pathology']]
                    itypes = [abn_path['type']]
                else:
                    abn_num = abn_path['abn_num']
                    pathology = abn_path['pathology']
                    itypes = abn_path['type']
                #
                bkg_sampled = False
                #遍历刚才获取的数据，转为掩码文件
                for abn, path, itype in zip(abn_num, pathology, itypes):
                    mask_fn = const_filename(pat, side, view, roi_mask_dir, itype, abn)
                    if target_width is None:
                        mask_img = read_resize_img(
                            mask_fn, target_height=target_height, gs_255=True)
                    else:
                        mask_img = read_resize_img(
                            mask_fn, target_size=(target_height, target_width),
                            gs_255=True)
                    if segment_breast:
                        mask_img = crop_img(mask_img, bbox)
                    # sample using mask and full image.样本使用掩码和全图
                    #获取难类数
                    nb_hns_ = nb_hns if not bkg_sampled else 0
                    if nb_hns_ > 0:
                        #返回阴性斑点采样得到的背景样本数
                        hns_sampled = sample_blob_negatives(
                            full_img, mask_img, out_dir, img_id,
                            abn, blob_detector, patch_size, neg_cutoff,
                            nb_hns_, 0, bkg_dir, verbose)
                    else:
                        hns_sampled = 0
                    pos = path.startswith('MALIGNANT')
                    #背景数为总背景数-阴性斑点采样数
                    nb_bkg_ = nb_bkg - hns_sampled if not bkg_sampled else 0
                    #对块进行采样
                    sample_patches(full_img, mask_img, out_dir, img_id, abn, pos,
                                   patch_size, pos_cutoff, neg_cutoff,
                                   nb_bkg_, nb_abn, hns_sampled, itype,
                                   bkg_dir, calc_pos_dir, calc_neg_dir,
                                   mass_pos_dir, mass_neg_dir, verbose)
                    bkg_sampled = True
            except AttributeError:
                print ("Read image error: %s" % (full_fn))
            except ValueError:
                print ("Error sampling from ROI mask image: %s" % (mask_fn))

    #####
    #训练集采样
    print ("Sampling for train set")
    sys.stdout.flush()
    do_sampling(train_df, train_out_dir)
    print ("Done.")
    #####
    #验证集采样
    if val_size > 0.:
        print ("Sampling for val set")
        sys.stdout.flush()
        do_sampling(val_df, val_out_dir)
        print ("Done.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sample patches for DDSM images")
    parser.add_argument("roi_mask_path_file", type=str)
    parser.add_argument("roi_mask_dir", type=str)
    parser.add_argument("pat_train_list_file", type=str)
    parser.add_argument("full_img_dir", type=str)
    parser.add_argument("train_out_dir", type=str)
    parser.add_argument("val_out_dir", type=str)
    parser.add_argument("--target-height", dest="target_height", type=int, default=4096)
    parser.add_argument("--target-width", dest="target_width", type=int, default=None)
    parser.add_argument("--no-target-width", dest="target_width", action="store_const", const=None)
    parser.add_argument("--segment-breast", dest="segment_breast", action="store_true")
    parser.add_argument("--no-segment-breast", dest="segment_breast", action="store_false")
    parser.set_defaults(segment_breast=True)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=256)
    parser.add_argument("--nb-bkg", dest="nb_bkg", type=int, default=30)
    parser.add_argument("--nb-abn", dest="nb_abn", type=int, default=30)
    parser.add_argument("--nb-hns", dest="nb_hns", type=int, default=15)
    parser.add_argument("--pos-cutoff", dest="pos_cutoff", type=float, default=.75)
    parser.add_argument("--neg-cutoff", dest="neg_cutoff", type=float, default=.35)
    parser.add_argument("--val-size", dest="val_size", type=float, default=.1)
    parser.add_argument("--bkg-dir", dest="bkg_dir", type=str, default="background")
    parser.add_argument("--calc-pos-dir", dest="calc_pos_dir", type=str, default="calc_mal")
    parser.add_argument("--calc-neg-dir", dest="calc_neg_dir", type=str, default="calc_ben")
    parser.add_argument("--mass-pos-dir", dest="mass_pos_dir", type=str, default="mass_mal")
    parser.add_argument("--mass-neg-dir", dest="mass_neg_dir", type=str, default="mass_ben")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    run_opts = dict(
        target_height=args.target_height,
        target_width=args.target_width,
        segment_breast=args.segment_breast,
        patch_size=args.patch_size,
        nb_bkg=args.nb_bkg,
        nb_abn=args.nb_abn,
        nb_hns=args.nb_hns,
        pos_cutoff=args.pos_cutoff,
        neg_cutoff=args.neg_cutoff,
        val_size=args.val_size,
        bkg_dir=args.bkg_dir,
        calc_pos_dir=args.calc_pos_dir,
        calc_neg_dir=args.calc_neg_dir,
        mass_pos_dir=args.mass_pos_dir,
        mass_neg_dir=args.mass_neg_dir,
        verbose=args.verbose
    )
    print ("\n>>> Model training options: <<<\n", run_opts, "\n")
    run(args.roi_mask_path_file, args.roi_mask_dir, args.pat_train_list_file,
        args.full_img_dir, args.train_out_dir, args.val_out_dir, **run_opts)

