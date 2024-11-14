import torch
from functools import wraps
from torchvision.transforms import ToTensor
from kornia.geometry.transform import resize
from kornia.filters import median_blur

pil_to_tensor = ToTensor()


def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


@no_grad
def fill_polygon(polygon, mask_dims, device):
    x_points = torch.arange(mask_dims[0], dtype=polygon.dtype, device=device).repeat(
        mask_dims[0]
    )
    y_points = torch.arange(
        mask_dims[1], dtype=polygon.dtype, device=device
    ).repeat_interleave(mask_dims[1])

    polygon_x1 = polygon[:, 0]
    polygon_y1 = polygon[:, 1]
    polygon_x2 = torch.roll(polygon_x1, shifts=-1, dims=0)
    polygon_y2 = torch.roll(polygon_y1, shifts=-1, dims=0)

    y_points = y_points[:, None]
    condition = (polygon_y1 > y_points) != (polygon_y2 > y_points)

    intersect_x = (y_points - polygon_y1) * (polygon_x2 - polygon_x1) / (
        polygon_y2 - polygon_y1
    ) + polygon_x1

    valid_intersections = torch.zeros(x_points.shape, dtype=torch.uint8, device=device)
    valid_intersections.add_(
        (condition & (intersect_x > x_points[:, None])).sum(dim=1) % 2 == 1
    )

    return valid_intersections.view(mask_dims)


def apply_median_blur(self, mask, median_blur_kernel=15):
    num_dims = mask.dim()

    if num_dims == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif num_dims == 3:
        mask = mask.unsqueeze(0)
    elif num_dims > 3:
        raise ValueError("Mask has too many dimensions to apply median_blur.")

    mask = median_blur(mask, median_blur_kernel)

    return mask
