import torch
import matplotlib.pyplot as plt


def apply_masked_highlight(img, pred_mask):
    """
    Apply low-light and high-light effects to an image based on the predicted mask.

    Args:
    - img (torch.Tensor): Original image tensor in shape (C, H, W), normalized to [0, 1].
    - pred_mask (torch.Tensor): Predicted mask tensor in shape (H, W) with values in [0, 1].

    Returns:
    - fused_img (torch.Tensor): Processed image where mask=0 regions are darkened and mask=1 regions are brightened.
    """
    # Ensure pred_mask has a channel dimension
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0)  # Add channel dimension

    # Low-light effect: Dim the areas where pred_mask=0
    decay = 0.5
    low_light_img = img * (1 - pred_mask) - decay * (1 - pred_mask)

    # High-light effect: Brighten the areas where pred_mask=1
    gain = 0.5  # Brightness gain factor
    high_light_img = img * pred_mask + gain * pred_mask

    # Combine low-light and high-light regions
    fused_img = low_light_img + high_light_img

    # Clamp the values to ensure they stay within [0, 1]
    fused_img = torch.clamp(fused_img, 0, 1)

    return fused_img


def visualize_fused_image(img, gd_mask, pred_mask, fused_img, save_path, file_name):
    """
    Visualize the original image, ground truth mask, predicted mask, and fused result.

    Args:
    - img (torch.Tensor): Original image tensor in CHW format.
    - gd_mask (torch.Tensor): Ground truth mask tensor in HW format.
    - pred_mask (torch.Tensor): Predicted mask tensor in HW format.
    - fused_img (torch.Tensor): Fused image tensor in CHW format.
    - save_path (str): Path to save the output visualization.
    - file_name (str): Name of the output file.
    """
    
    # Clamp data to valid ranges
    img = torch.clamp(img, 0, 1)  # Assuming input is in [0, 1] for floats
    gd_mask = torch.clamp(gd_mask, 0, 1)
    pred_mask = torch.clamp(pred_mask, 0, 1)
    fused_img = torch.clamp(fused_img, 0, 1)

    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot the original image
    axs[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original Image", pad=10)
    axs[0].axis("off")
    
    # Plot the ground truth mask
    axs[1].imshow(gd_mask.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Ground Truth Mask", pad=10)
    axs[1].axis("off")
    
    # Plot the predicted mask
    axs[2].imshow(pred_mask.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Predicted Mask", pad=10)
    axs[2].axis("off")
    
    # Plot the fused image
    axs[3].imshow(fused_img.permute(1, 2, 0).cpu().numpy())
    axs[3].set_title("Fused Image", pad=10)
    axs[3].axis("off")
    
    # Adjust layout and save the figure
    # Adjust layout and save the figure
    plt.tight_layout(pad=2)
    plt.subplots_adjust(wspace=0.3)  # Optional: Adjust spacing
    plt.savefig(f"{save_path}/{file_name}.jpg")
    plt.close()