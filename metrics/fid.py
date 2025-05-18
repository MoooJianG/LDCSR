from pytorch_fid import fid_score


def calc_fid(paths, batch_size=16, device="cuda:1", dims=2048):
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=8)
