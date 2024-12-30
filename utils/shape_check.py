def print_shape_hook(module, input, output):
    print(f"Layer: {module.__class__.__name__}, Output shape: {output.shape}")
    
def register_hooks_recursive(module):
    hooks = []
    for child in module.children():
        # 为当前模块注册钩子
        hooks.append(child.register_forward_hook(print_shape_hook))
        # 递归注册子模块
        hooks.extend(register_hooks_recursive(child))
    return hooks
