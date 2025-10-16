import torch


def mask_tensor(input_tensor, mask_prob=0.15, padding_value=-1, add_cls=False):
    # Identify zero, nonzero, and padding elements
    zero_mask = (input_tensor == 0).float()
    nonzero_mask = (input_tensor != 0).float()
    padding_mask = (input_tensor == padding_value).float()

    # Generate random masks for zero and nonzero elements
    zero_random_mask = (torch.rand_like(input_tensor) < mask_prob).float()
    nonzero_random_mask = (torch.rand_like(input_tensor) < mask_prob).float()

    # Apply mask for zero values (masking zeros with 15% probability)
    zero_applied_mask = zero_mask * zero_random_mask

    # Apply mask for nonzero values (masking nonzero elements with 15% probability)
    nonzero_applied_mask = nonzero_mask * nonzero_random_mask

    # Get indices of nonzero values to sample from
    nonzero_indices = (input_tensor != 0) & (input_tensor != padding_value)

    # Sample nonzero values uniformly from the input tensor
    sampled_nonzero_values = input_tensor[nonzero_indices]
    if len(sampled_nonzero_values) > 0:
        sampled_nonzero_values = sampled_nonzero_values[
            torch.randint(0, len(sampled_nonzero_values), input_tensor.shape)
        ]

    # Apply mask strategy
    # Nonzero values should be masked by zero
    masked_tensor = input_tensor.clone()
    masked_tensor[nonzero_applied_mask.bool()] = 0

    # Zero values should be masked by sampled nonzero values
    masked_tensor[zero_applied_mask.bool()] = sampled_nonzero_values[zero_applied_mask.bool()]

    # Ensure padding values are not affected
    masked_tensor[padding_mask.bool()] = padding_value
    if add_cls:
        masked_tensor[:, 0] = input_tensor[:, 0]  # Retain the original first element

    apply_mask = zero_applied_mask + nonzero_applied_mask

    return masked_tensor, apply_mask, padding_mask
