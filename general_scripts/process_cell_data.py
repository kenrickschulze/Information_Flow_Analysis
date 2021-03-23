import os
from PIL import Image
import numpy as np
import glob
from tifffile import imsave, imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv


def process_imgs():
    data_path = "/home/kenrick/work/zuse/DataSets/cell_nuceli_2018/stage1_train"
    outdir = "/home/kenrick/work/zuse/DataSets/cell_nuceli_2018/processed"
    samples = os.listdir(data_path)

    for idx, sample in enumerate(samples):
        path_to_masks = os.path.join(data_path, sample, "masks")
        path_to_images = os.path.join(data_path, sample, "images")
        masks = glob.glob(path_to_masks + "/*png")
        image = glob.glob(path_to_images + "/*png")[0]
        im = np.array(Image.open(image).convert('L'))
        # normalize img
        im = (im / 255.0).astype(float)
        imsave(f"{outdir}/{idx}_image.tif", im)

        # merge masks into single file
        mask_tmp = np.zeros(im.shape)
        for mask in masks:
            m = np.array(Image.open(mask).convert('L'))
            mask_tmp[m > 0] = 1

        imsave(f"{outdir}/{idx}_mask.tif", mask_tmp)


def kmeans(Y, n_clusters=64):
    Kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Y)
    print(Kmeans.labels_)
    return Kmeans.labels_


def plot_clusters(masks, label, which_cluster=None, n_clusters=64):
    if which_cluster is None:
        cl = np.arange(n_clusters)
    else:
        cl = which_cluster
    print(masks.shape)
    for lidx in cl:
        w = np.where(label == lidx)[0]
        x = len(w)
        # print(lidx, len(np.where(label == lidx)[0]))
        fig, axs = plt.subplots(1, min(4, x), figsize=(8, 12))
        print("min_value", min(4, x))
        for i, ax in enumerate(range(len(axs))):
            axs[ax].imshow(masks[w[i]])
        fig.suptitle(lidx, fontsize=20)
        plt.show()

def create_summary_csv(labels, path_out):
    """
    saves list for later usage to infer class probabilities
    needed for KDE MI estimation
    :param labels:
    :param path_out:
    :return:
    """
    with open(os.path.join(path_out, 'all.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(labels))


def patch_dataset(path_in, path_out, patch_size=128, save=False, show_plot=False):
    """
    take from each sample a 128x128 patch and store it in new location together with map
    ### later add many patches
    :param in_path:
    :param out_path:
    :return:
    """
    # load data
    mask_paths = sorted(glob.glob(path_in + '/*_mask*'))
    img_paths = sorted(glob.glob(path_in + '/*_image*'))
    assert (len(mask_paths) == len(img_paths))
    cropped_img = []
    cropped_mask = []

    # create output dir if it does not excist
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        # open mask and img
        img = np.array(imread(img_path))
        mask = np.array(imread(mask_path))
        assert (len(mask.shape) == 2)

        # TODO refine that it takes only patches when given amount of pixels is not background

        # crop image at random position
        ori_x = np.random.randint(0, img.shape[0] - patch_size)
        ori_y = np.random.randint(0, img.shape[1] - patch_size)

        img_crop = img[ori_x:ori_x + patch_size, ori_y:ori_y + patch_size]
        mask_crop = mask[ori_x:ori_x + patch_size, ori_y:ori_y + patch_size]
        cropped_mask.append(mask_crop)
        cropped_img.append(img_crop)

    cropped_img = np.array(cropped_img)
    cropped_mask = np.array(cropped_mask)

    L = cropped_mask.reshape(cropped_mask.shape[0], -1)
    N_CLUSTERS = 6
    print('.. start clustering')
    labels = kmeans(L, n_clusters=N_CLUSTERS)
    print('.. start plotting')

    if show_plot:
        plot_clusters(cropped_mask, labels, n_clusters=N_CLUSTERS)

    # save images with label and idx in name
    if save:
        create_summary_csv(labels=labels, path_out=path_out)
        for idx, (img, mask, label) in enumerate(zip(cropped_img, cropped_mask, labels)):
            imsave(f"{path_out}/idx{idx}_mask_l{label}.tiff", mask)
            imsave(f"{path_out}/idx{idx}_image_l{label}.tiff", img)


def main():
    # kmeans("/home/kenrick/work/zuse/DataSets/cell_nuceli_2018/processed")
    patch_dataset(path_in="/home/kenrick/work/zuse/DataSets/cell_nuceli_2018/processed",
                  path_out="/home/kenrick/work/zuse/DataSets/cell_nuceli_2018/processed/patched")


if __name__ == "__main__":
    main()
