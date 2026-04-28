import time
from numbers import Number
from typing import Any, List

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis


def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shape = inputs[0].type().sizes()
    _, height, width, channels = input_shape
    token_count = height * width
    flops = token_count * channels * np.ceil(np.log2(token_count))
    return flops


def calc_flops(model, img_size=224, show_details=False, ratios=None):
    with torch.no_grad():
        example = torch.randn(1, 3, img_size, img_size)
        model.default_ratio = ratios
        flop_analysis = FlopCountAnalysis(model, example)
        flop_analysis.set_op_handle(
            **{
                'aten::fft_rfft2': rfft_flop_jit,
                'aten::fft_irfft2': rfft_flop_jit,
            }
        )
        total_flops = flop_analysis.total()
        if show_details:
            print(flop_analysis.by_module())
        print(f'#### GFLOPs: {total_flops / 1e9} for ratio {ratios}')
    return total_flops / 1e9


@torch.no_grad()
def throughput(images, model):
    model.eval()
    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]
    for _ in range(50):
        model(images)
    torch.cuda.synchronize()
    print('throughput averaged with 30 times')
    tic = time.time()
    for _ in range(30):
        model(images)
    torch.cuda.synchronize()
    toc = time.time()
    print(f'batch_size {batch_size} throughput {30 * batch_size / (toc - tic)}')
    memory_mb = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / memory_mb)
