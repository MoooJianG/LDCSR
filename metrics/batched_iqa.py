import torch
from tqdm import tqdm


def batched_iqa(iqa_measure, rslt_tensor, hr_tensor=None, batch_size=20, desc=""):
    # 确保两个输入张量的第一个维度（批次维度）相同
    has_hr_tensor = False if hr_tensor is None else True
    if has_hr_tensor:
        assert rslt_tensor.size(0) == hr_tensor.size(
            0
        ), "Batch sizes of the tensors must match."

    # 计算总样本数
    num_samples = rslt_tensor.size(0)

    # 存储结果
    measure_values = []

    # 将输入张量分成小批次
    for i in tqdm(range(0, num_samples, batch_size),desc=desc):
        # 选择当前批次的样本
        rslt_batch = rslt_tensor[i : i + batch_size]
        if has_hr_tensor:
            hr_batch = hr_tensor[i : i + batch_size]

        # 计算当前批次的 LPIPS 值
        if has_hr_tensor:
            measure_batch = iqa_measure(rslt_batch, hr_batch)
        else:
            measure_batch = iqa_measure(rslt_batch)

        # 将结果保存到列表中
        measure_values.append(measure_batch)

    # 将所有批次的结果拼接起来
    measure_values = torch.cat(measure_values, dim=0)

    return measure_values
