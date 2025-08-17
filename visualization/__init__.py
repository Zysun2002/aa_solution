from .vis_to_pdf import display_as_gallery
from .vis_ddf import convert_ddf_to_image

def triCls_vis(data_path):
    print("visualize ...")

    vis_path = data_path / "vis"
    vis_path.mkdir(exist_ok=True)

    gallery_path = vis_path/"gallery"
    display_as_gallery((data_path/"val").resolve(), str(gallery_path))