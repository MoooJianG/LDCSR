from pytorch_fid import fid_score

paths = [
    "logs/first_stage_kl_v7_aid/2024-05-26T12-37-11/results/AID_tiny/x4.0/reconstructions",
    "logs/first_stage_kl_v7_aid/2024-05-26T12-37-11/results/AID_tiny/x4.0/inputs",
]


def calc_fid(paths, batch_size=1, device="cuda:0", dims=2048):
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims)


if __name__ == "main":
    calc_fid(paths)
