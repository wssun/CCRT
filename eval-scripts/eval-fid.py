import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as TF
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from torchvision.models import inception_v3

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    model.eval()
    if batch_size > len(files):
        print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)
    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.dim() == 2:
            pred = pred.view(pred.size(0), -1)
        pred = pred.cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_activation_statistics(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_statistics_of_pair(file1, file2, model, batch_size, dims, device, num_workers=1):
    m1, s1 = calculate_activation_statistics([file1], model, batch_size, dims, device, num_workers)
    m2, s2 = calculate_activation_statistics([file2], model, batch_size, dims, device, num_workers)
    return m1, s1, m2, s2

def calculate_fid_given_pairs(path1, path2, batch_size, device, dims, num_workers=1):
    files1 = sorted(glob.glob(os.path.join(path1, "*")))
    files2 = sorted(glob.glob(os.path.join(path2, "*")))
    assert len(files1) == len(files2), "The two folders must contain the same number of files"
    
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    model.to(device)
    
    fid_values = []
    for file1, file2 in zip(files1, files2):
        m1, s1, m2, s2 = compute_statistics_of_pair(file1, file2, model, batch_size, dims, device, num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        fid_values.append(fid_value)
    
    average_fid = np.mean(fid_values)
    return average_fid


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument("--num-workers", type=int, help="Number of processes to use for data loading. Defaults to `min(8, num_cpus)`")
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument("--dims", type=int, default=2048, help="Dimensionality of Inception features to use. By default, uses pool3 features")
    parser.add_argument("path", type=str, nargs=2, help="Paths to the generated images or to .npz statistic files")
    args = parser.parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers
    average_fid = calculate_fid_given_pairs(args.path[0], args.path[1], args.batch_size, device, args.dims, num_workers)
    print("Average FID: ", average_fid)

if __name__ == "__main__":
    main()

