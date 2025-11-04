import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, List
import gc

from qmllm.methods.gamq.gamq_process import gamq_process
from qmllm.quantization.qlinear import QuantLinear

def gamq_entry(
    model,
    prompt_inputs,
    prompt_kwargs,
    run_gamq_process=True,
    pseudo_quant=False,
    scale_path=None,
    q_group_size=128,
    w_bit=4,
    a_bit=8,
    wa_quant=True,
    reweight=True,
    distort=False,
    loss_mode="mse",
    entropy_threshold_ratio=0.1,
    hessian_lambda=1e-6,
    adaptive_alpha=True,
    alpha_range=(0.999, 0.99999),
    num_blocks=None,
):
    """
    GAMQ (Gradient-Augmented Modality Quantization) entry point.
    
    Args:
        model: VLM model to quantize
        prompt_inputs: Calibration inputs
        prompt_kwargs: Calibration keyword arguments
        run_gamq_process: Whether to run GAMQ optimization
        pseudo_quant: Whether to use pseudo quantization
        scale_path: Path to save/load scales
        q_group_size: Group size for weight quantization
        w_bit: Weight bit width
        a_bit: Activation bit width
        wa_quant: Whether to quantize both weights and activations
        reweight: Whether to use gradient-augmented reweighting
        distort: Whether to apply distortion
        loss_mode: Loss function mode ('mse' or 'mae')
        entropy_threshold_ratio: Ratio for entropy-based partitioning
        hessian_lambda: Regularization parameter for Hessian
        adaptive_alpha: Whether to use adaptive clipping alpha
        alpha_range: Range for adaptive alpha (min, max)
        num_blocks: Number of entropy-guided blocks (None for auto)
    """
    
    print(f"[GAMQ] Starting GAMQ quantization: W{w_bit}A{a_bit}")
    print(f"[GAMQ] Configuration: reweight={reweight}, entropy_threshold={entropy_threshold_ratio}")
    
    # Run GAMQ process if enabled
    if run_gamq_process:
        print("[GAMQ] Running GAMQ optimization process...")
        model = gamq_process(
            model=model,
            prompt_inputs=prompt_inputs,
            prompt_kwargs=prompt_kwargs,
            q_group_size=q_group_size,
            w_bit=w_bit,
            a_bit=a_bit,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
            loss_mode=loss_mode,
            entropy_threshold_ratio=entropy_threshold_ratio,
            hessian_lambda=hessian_lambda,
            adaptive_alpha=adaptive_alpha,
            alpha_range=alpha_range,
            num_blocks=num_blocks,
            scale_path=scale_path,
        )
    
    # Apply quantization if not pseudo quantization
    if not pseudo_quant:
        print("[GAMQ] Applying real quantization...")
        model = apply_gamq_quantization(model, w_bit, a_bit, q_group_size, wa_quant)
    
    print("[GAMQ] GAMQ quantization completed!")
    return model

def apply_gamq_quantization(model, w_bit, a_bit, q_group_size, wa_quant):
    """Apply actual quantization to the model."""
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace linear layers with quantized linear layers
            if hasattr(module, 'gamq_scale'):
                # Use GAMQ computed scales
                qlinear = QuantLinear(
                    w_bit=w_bit,
                    group_size=q_group_size,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )
                qlinear.pack(module, module.gamq_scale.to(module.weight.device))
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                setattr(parent, child_name, qlinear)
    
    return model