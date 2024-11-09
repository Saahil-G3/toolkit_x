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

#@no_grad
#def set_default_gpu(device):
#    if torch.cuda.is_available():
#        torch.set_default_device(device)
#    else:
#        torch.set_default_device("cpu")

@no_grad
def fill_polygon(polygon, mask_dims, device):
    x_points = torch.arange(mask_dims[0], dtype=polygon.dtype, device=device).repeat(mask_dims[0])
    y_points = torch.arange(mask_dims[1], dtype=polygon.dtype, device=device).repeat_interleave(mask_dims[1])
    
    polygon_x1 = polygon[:, 0]
    polygon_y1 = polygon[:, 1]
    polygon_x2 = torch.roll(polygon_x1, shifts=-1, dims=0)
    polygon_y2 = torch.roll(polygon_y1, shifts=-1, dims=0)
    
    y_points = y_points[:, None]
    condition = (polygon_y1 > y_points) != (polygon_y2 > y_points)
    
    intersect_x = (y_points - polygon_y1) * (polygon_x2 - polygon_x1) / (polygon_y2 - polygon_y1) + polygon_x1
    
    valid_intersections = torch.zeros(x_points.shape, dtype=torch.uint8, device=device)
    valid_intersections.add_((condition & (intersect_x > x_points[:, None])).sum(dim=1) % 2 == 1)
    
    return valid_intersections.view(mask_dims)

@no_grad
def get_polygon_coordinates(polygon, device, scale_factor=1, origin=None):
    if origin is None:
        origin = torch.tensor([0, 0], dtype=torch.float32, device=device)
    exterior = (torch.tensor(polygon.exterior.coords, dtype=torch.float32, device=device) - origin) * scale_factor
    exterior = exterior.round().long()
    
    holes = [
        ((torch.tensor(interior.coords, dtype=torch.float32, device=device) - origin) * scale_factor).round().long()
        for interior in polygon.interiors
    ]
    return [exterior], holes



            
    



