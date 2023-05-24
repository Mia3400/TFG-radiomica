import os
import re
from pathlib import Path
import matplotlib
import numpy as np
import pydicom
from matplotlib import pyplot as plt


def get_paths(root_path):
    """ Returns a generator of tuples (patient_idx, img_dirpath, mask_dirpath) """
    patient_dirs = os.listdir(root_path)
    patient_dirs = [dirname for dirname in patient_dirs if re.match(r'HCC_\d+', dirname)]
    patient_dirs.sort()
    for patient_dir in patient_dirs:
        # Infer series folder
        series_dir = None
        for subdir in os.listdir(os.path.join(root_path, patient_dir)):
            candidate_dirpath = os.path.join(root_path, patient_dir, subdir)
            if not os.path.isdir(candidate_dirpath):
                continue
            if any(dirname.startswith('300') for dirname in os.listdir(candidate_dirpath)):
                series_dir = subdir
                break
        if not series_dir:
            raise ValueError(f"Couldn't find series folder for patient {patient_dir}")
        # Infer acquisition folders
        subdirs = os.listdir(os.path.join(root_path, patient_dir, series_dir))
        subdirs = [dirname for dirname in subdirs if not dirname.startswith('.')]
        subdirs.sort(key=lambda dname: len(os.listdir(os.path.join(root_path, patient_dir, series_dir, dname))))
        img_dirpath = os.path.join(root_path, patient_dir, series_dir, subdirs[-1])  # "Largest" folder
        mask_dirpath = os.path.join(root_path, patient_dir, series_dir, subdirs[0])  # Only 1 file in acq folder
        # Return
        yield patient_dir, img_dirpath, mask_dirpath


def load_img(folder_dcm):
    """ Loads whole Dicom folder """
    img_dcmset = []
    for root, _, filenames in os.walk(folder_dcm):
        for filename in filenames:
            dcm_path = Path(root, filename)
            if dcm_path.suffix == ".dcm":
                try:
                    dicom = pydicom.dcmread(dcm_path, force=True)
                except IOError as e:
                    print(f"Can't import {dcm_path.stem}")
                else:
                    img_dcmset.append(dicom)
    img_dcmset.sort(key=lambda x: x['ImagePositionPatient'][2].real)
    try:
        # If there are multiple acquisitions, only keep the first one
        acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
        img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]
    except TypeError:
        # If no acquisition appears, do not filter out any slices
        pass
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)
    return img_dcmset, img_pixelarray

def load_mask(mask_dirpath, img_dcmset, img_pixelarray):
    """ Loads mask Dicom and reads header to organize PixelArray. """
    # Load mask
    mask_filepath = os.path.join(mask_dirpath, os.listdir(mask_dirpath)[0])
    mask_dcm = pydicom.dcmread(mask_filepath)
    mask_pixelarray_messy = mask_dcm.pixel_array        # Pydicom's unordered PixelArray
    mask_pixelarray = np.zeros_like(img_pixelarray)     # Rearranged PixelArray

    first_slice_depth = img_dcmset[0]['ImagePositionPatient'][2].real
    last_slice_depth = img_dcmset[-1]['ImagePositionPatient'][2].real
    slice_increment = (last_slice_depth - first_slice_depth) / (len(img_dcmset) - 1)
    for frame_idx, frame_info in enumerate(mask_dcm[0x52009230]):     # (5200 9230) -> Per-frame Functional Groups Sequence
        position = frame_info['PlanePositionSequence'][0]['ImagePositionPatient']
        slice_depth = position[2].real
        slice_idx = round((slice_depth-first_slice_depth)/slice_increment)

        segm_number = frame_info['SegmentIdentificationSequence'][0]['ReferencedSegmentNumber'].value
        if 0 <= slice_idx < mask_pixelarray.shape[0]:
            mask_pixelarray[slice_idx, :, :] += mask_pixelarray_messy[frame_idx, :, :] * segm_number  # Rearrange
    return mask_dcm, mask_pixelarray.astype('int')

def find_tag_recursively(dcm, tag):
    """ Finds tag in header dicoms recursively (iterating over sequences `SQ`)"""
    current_elements = [dcm]
    while current_elements:
        dcm_element = current_elements.pop()
        if tag in dcm_element:
            return dcm_element[tag]
        for child in dcm_element:
            if child.VR == "SQ":
                current_elements.extend([e for e in child])


def visualize(idx, img, mask, px_size, axial_slices=0, coronal_slices=20, show=True):
    """ Save visualizations of img and mask"""
    if axial_slices:
        range_slices = np.unique(np.linspace(0, img.shape[0]-1, axial_slices+2).astype('int'))
        range_slices = range_slices[1:-1]
    else:
        range_slices = []
    for slice_idx in range_slices:
        img_slice = img[slice_idx, :, :]
        norm = matplotlib.colors.Normalize(vmin=img_slice.min(), vmax=img_slice.max())
        img_cmapped = matplotlib.colormaps['bone'](norm(img_slice))[..., :3]
        mask_slice = mask[slice_idx, :, :]
        mask_nonzero = np.tile((mask_slice > 0)[..., np.newaxis], [1, 1, 3])
        mask_cmapped = matplotlib.colormaps['Set1'](mask_slice)[..., :3]
        mask_cmapped[~mask_nonzero] = 0

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(img_cmapped, aspect=px_size[1]/px_size[2])
        axs[0, 1].imshow(mask_cmapped, aspect=px_size[1]/px_size[2])
        axs[1, 0].imshow(img_cmapped*(1-0.5*mask_nonzero) + mask_cmapped*0.5*mask_nonzero, aspect=px_size[1]/px_size[2])
        fig.show()
        os.makedirs(f'results/{idx}/axial/', exist_ok=True)
        fig.savefig(f'results/{idx}/axial/axial_{idx}_{slice_idx}.png')
        if show and slice_idx == range_slices[range_slices.size//2]:
            os.makedirs(f'results/summary/', exist_ok=True)
            fig.savefig(f'results/summary/axial_{idx}_{slice_idx}.png')
            plt.show()
        plt.close()

    if coronal_slices:
        range_slices = np.unique(np.linspace(0, img.shape[1]-1, coronal_slices+2).astype('int'))
        range_slices = range_slices[1:-1]
    else:
        range_slices = []
    for slice_idx in range_slices:
        img_slice = img[:, slice_idx, :]
        norm = matplotlib.colors.Normalize(vmin=img_slice.min(), vmax=img_slice.max())
        img_cmapped = matplotlib.colormaps['bone'](norm(img_slice))[..., :3]
        mask_slice = mask[:, slice_idx, :]
        mask_nonzero = np.tile((mask_slice > 0)[..., np.newaxis], [1, 1, 3])
        mask_cmapped = matplotlib.colormaps['Set1'](mask_slice)[..., :3]
        mask_cmapped[~mask_nonzero] = 0

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(img_cmapped, aspect=px_size[0]/px_size[2])
        axs[0, 1].imshow(mask_cmapped, aspect=px_size[0]/px_size[2])
        axs[1, 0].imshow(img_cmapped*(1-0.5*mask_nonzero) + mask_cmapped*0.5*mask_nonzero, aspect=px_size[0]/px_size[2])
        os.makedirs(f'results/{idx}/coronal/', exist_ok=True)
        fig.savefig(f'results/{idx}/coronal/coronal_{idx}_{slice_idx}.png')
        if show and slice_idx == range_slices[range_slices.size//2]:
            os.makedirs(f'results/summary/', exist_ok=True)
            fig.savefig(f'results/summary/coronal_{idx}_{slice_idx}.png')
            plt.show()
        plt.close()

# DATASET = [{
#     'img_folder': "/Users/pedro/Desktop/NBIA/manifest-1643035385102/HCC-TACE-Seg/HCC_003/09-12-1997-NA-AP LIVER-64595/4.000000-Recon 2 LIVER 3 PHASE AP-18688",
#     'mask_filepath': "/Users/pedro/Desktop/NBIA/manifest-1643035385102/HCC-TACE-Seg/HCC_003/09-12-1997-NA-AP LIVER-64595/300.000000-Segmentation-45632/1-1.dcm",
# }, {
#     'img_folder': "/Users/pedro/Desktop/NBIA/manifest-1643035385102/HCC-TACE-Seg/HCC_004/10-06-1997-NA-LIVERPELVIS-14785/4.000000-Recon 2 LIVER 3 PHASE AP-69759",
#     'mask_filepath': "/Users/pedro/Desktop/NBIA/manifest-1643035385102/HCC-TACE-Seg/HCC_004/10-06-1997-NA-LIVERPELVIS-14785/300.000000-Segmentation-39561/1-1.dcm",
# }]

#ROOT = "C:/TFG-code/manifest-1680446438379/HCC-TACE-Seg"

# for patient_idx, img_dirpath, mask_filepath in get_paths(ROOT):
#     print(f'Processing {patient_idx}')
#     print(f'Img: {img_dirpath}')
#     print(f'Mask: {mask_filepath}')

#     img_dcmset, img = load_img(img_dirpath)
#     mask_dcm, mask = load_mask(mask_filepath, img_dcmset, img)
#     print(f'Img: {img.shape}; Mask: {mask.shape}')

#     pixel_spacing = find_tag_recursively(mask_dcm, 'PixelSpacing')
#     slice_spacing = find_tag_recursively(mask_dcm, 'SpacingBetweenSlices')
#     px_size = (slice_spacing.value.real, pixel_spacing.value[0].real, pixel_spacing.value[1].real)

#     visualize(patient_idx, img, mask, px_size=px_size, axial_slices=0, coronal_slices=1, show=True)

