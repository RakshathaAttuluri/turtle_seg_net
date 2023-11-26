import torch
from PIL import Image, ImageDraw
from torchvision import transforms


def gift_wrapping_convex_hull(image_tensor):
    # Get all index where pixel val is 1.
    white_pxl_coords = torch.stack(torch.where(image_tensor == 1), dim=1)

    # Get the leftmost pixel by sorting on x.
    sorted_coords = white_pxl_coords[white_pxl_coords[:, 1].argsort()]
    n = len(sorted_coords)
    if n < 3:
        return sorted_coords

    def is_next_pt_on_left(p, q, r):
        """ Compares slope of lin(pq) with lin(qr).
            If slope(pq) < slope(qr) -> {r} is to the left of {p, q}.
            If slope(pq) > slope(qr) -> {r} is to the right of {p, q}.
            If slope(pq) == slope(qr) -> {p, q, r} are collinear.
        """
        y1, x1 = sorted_coords[p]
        y2, x2 = sorted_coords[q]
        y3, x3 = sorted_coords[r]
        diff = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
        if diff == 0:
            return 0
        return 1 if diff < 0 else -1

    pt_on_hull = 0
    convex_hull = []
    while True:
        convex_hull.append(pt_on_hull)
        curr_pt = (pt_on_hull + 1) % n
        for next_pt in range(n):
            on_left = is_next_pt_on_left(pt_on_hull, curr_pt, next_pt)
            if on_left == 1:
                curr_pt = next_pt

        pt_on_hull = curr_pt
        if pt_on_hull == 0:
            break

    return sorted_coords[convex_hull]


def area(coords):
    ys, xs = coords[:, 0], coords[:, 1]
    x_seq = xs * torch.hstack((ys[1:], ys[0]))
    y_seq = torch.hstack((xs[1:], xs[0])) * ys
    return 0.5 * abs(torch.sum(x_seq) - torch.sum(y_seq))


def visualize_hull(convex_polygon):
    poly_xy = []
    for coord in convex_polygon.tolist():
        poly_xy.extend([coord[1], coord[0]])

    draw = ImageDraw.Draw(img)
    draw.polygon(poly_xy, outline='white')

    img.save('hull.png')


if __name__ == '__main__':
    img = Image.open('res.png').convert('L')
    polygon = gift_wrapping_convex_hull(transforms.ToTensor()(img).squeeze(dim=0))
    visualize_hull(polygon)

    polygon_area = area(polygon)
    print(f'Area of convex polygon is {polygon_area} px**2')
