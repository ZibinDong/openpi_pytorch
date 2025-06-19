import copy
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = (
            nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        )
        self.merge_weights = merge_weights

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.merged = False
        if self.merge_weights:
            self.merge_lora_weights()

    def forward(self, x: torch.Tensor):
        original_output = self.original_linear(x)
        if self.merge_weights and self.merged:
            return original_output
        else:
            lora_output = self.lora_dropout(x) @ self.lora_A.T
            lora_output = lora_output @ self.lora_B.T * self.scaling
            return original_output + lora_output

    def merge_lora_weights(self):
        if self.merged:
            return
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        self.original_linear.weight.data += delta_W
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        self.merged = True


def apply_lora_to_model(
    model: nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    target_modules: list[str] = None,  # 要应用LoRA的模块名称，如 ["q_proj", "v_proj"]
    merge_weights_after_init: bool = False,  # 是否在初始化后立即合并权重 (通常训练时不合并，推理时可以合并)
) -> nn.Module:
    """
    遍历模型中的所有 nn.Linear 层，并将其替换为 LoRALinear 层。

    Args:
        model: 原始的 nn.Module 模型。
        r: LoRA 的秩。
        lora_alpha: LoRA 的缩放因子。
        lora_dropout: LoRA 适配器的 dropout 率。
        target_modules: 一个列表，包含要应用LoRA的子模块名称。如果为 None，则应用于所有 Linear 层。
                        例如：对于transformer，可能是 ["query", "value"]。
        merge_weights_after_init: 如果为 True，则在 LoRALinear 初始化后立即将 LoRA 权重合并到原始权重中。
                                  在训练时通常设置为 False，在推理部署时可以设置为 True。
                                  请注意：如果设置为 True，后续 LoRA 权重将无法单独保存。

    Returns:
        一个新的 nn.Module 实例，其中所有符合条件的 nn.Linear 层都被替换为 LoRALinear。
    """

    # 创建模型的深拷贝，以免修改原始模型
    # 或者直接在原地修改，这取决于你的需求。对于LoRA，通常原地修改更常见。
    # 这里我们选择原地修改，因为这是 peft 库的常见行为。
    # modified_model = copy.deepcopy(model)
    modified_model = model

    for name, module in modified_model.named_children():
        # 递归处理子模块
        if isinstance(module, nn.ModuleList):
            for i, sub_module in enumerate(module):
                # 构建完整的路径名，例如 'blocks.0'
                full_name = f"{name}.{i}"
                # 递归调用
                apply_lora_to_model(
                    sub_module,
                    r,
                    lora_alpha,
                    lora_dropout,
                    target_modules,
                    merge_weights_after_init,
                )
        elif isinstance(module, nn.ModuleDict):
            for key, sub_module in module.items():
                full_name = f"{name}.{key}"
                apply_lora_to_model(
                    sub_module,
                    r,
                    lora_alpha,
                    lora_dropout,
                    target_modules,
                    merge_weights_after_init,
                )
        elif isinstance(module, (nn.Sequential)):
            # Sequential 也是一种容器
            for i, sub_module in enumerate(module):
                full_name = f"{name}.{i}"
                apply_lora_to_model(
                    sub_module,
                    r,
                    lora_alpha,
                    lora_dropout,
                    target_modules,
                    merge_weights_after_init,
                )
        else:
            # 检查当前模块是否是 nn.Linear 且是否在目标模块列表中
            is_target_module = True
            if target_modules:
                # 检查当前模块的名称是否匹配 target_modules
                # 这里需要更精细的匹配逻辑，例如检查模块的全路径名
                # 或者，更简单地，假设 target_modules 包含的是直接子模块的名称
                # 对于深层嵌套的模块，你需要传递完整的路径信息
                # 比如 target_modules = ["model.decoder.layers.0.self_attn.q_proj"]
                # 这里的简单实现只检查了局部名称

                # 更健壮的匹配方式是检查当前模块的完整路径名
                # 这需要将父模块的名称传递下来，或者在外部构建完整的路径
                # For simplicity, we assume `target_modules` are direct attributes
                # e.g., if target_modules = ["q_proj", "v_proj"], it applies to any linear named q_proj/v_proj
                # A more robust solution might involve passing the full module path:
                # current_full_path = f"{parent_path}.{name}" if parent_path else name
                # then check if current_full_path ends with any target_module pattern.

                # For this generic wrapper, we'll assume target_modules refers to the *name* of the Linear layer
                # itself, not its full hierarchical path, which is simpler for demonstration.
                # If target_modules is specified, we check if the current module's name matches.
                # If target_modules is None, all Linear layers are targeted.
                if name not in target_modules:
                    is_target_module = False

            if isinstance(module, nn.Linear) and is_target_module:
                # 替换为 LoRALinear
                lora_linear = LoRALinear(
                    original_linear=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=merge_weights_after_init,
                )
                setattr(modified_model, name, lora_linear)
            # else:
            #     # 递归处理非Linear子模块
            #     apply_lora_to_model(
            #         module, r, lora_alpha, lora_dropout,
            #         target_modules, merge_weights_after_init
            #     )

    # 递归查找内部 Linear 层的逻辑需要更复杂，通常通过遍历 module._modules 字典来实现
    # 示例如下是一个更通用的递归函数
    def _apply_lora_recursively(parent_module, current_path=""):
        for child_name, child_module in parent_module.named_children():
            new_path = f"{current_path}.{child_name}" if current_path else child_name

            # 如果当前子模块是 nn.Linear
            if isinstance(child_module, nn.Linear):
                # 检查是否是目标模块
                apply = False
                if target_modules is None:
                    apply = True  # 如果没有指定目标模块，所有 Linear 都替换
                else:
                    # 检查 child_name 是否在 target_modules 中 (简单匹配)
                    if child_name in target_modules:
                        apply = True
                    # 或者，更复杂的匹配模式，例如 full_path.endswith(target_module_pattern)
                    # 例如，如果 target_modules = ["q_proj", "v_proj"]
                    # 并且 new_path 是 "model.decoder.layers.0.self_attn.q_proj"
                    # 你可能需要检查 new_path.split('.')[-1] == "q_proj"
                    # 或者使用正则表达式匹配

                if apply:
                    lora_linear = LoRALinear(
                        original_linear=child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        merge_weights=merge_weights_after_init,
                    )
                    setattr(parent_module, child_name, lora_linear)
            # 否则，如果不是 Linear 层，则递归处理其子模块
            else:
                _apply_lora_recursively(child_module, new_path)

    _apply_lora_recursively(modified_model)
    return modified_model


def mark_only_lora_as_trainable(model: nn.Module):
    """
    将模型中所有 LoRALinear 层的 LoRA 参数设置为 requires_grad=True，
    并将其余参数设置为 requires_grad=False。
    """
    for name, param in model.named_parameters():
        if "lora_" in name:  # 根据 LoRALinear 中的参数命名约定
            param.requires_grad = True
        else:
            param.requires_grad = False
