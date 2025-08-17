import os
from pylatex import *

class MyDocument(Document):
    def __init__(self, output_path):
        super().__init__(output_path, inputenc=None)

        self.preamble.extend([
            Command('title', 'Image Gallery'),
            Command('author', 'Ziyu Sun'),
        ])

        self.packages.update([
            Package('graphicx'),
            Package('subcaption'), 
            Package('float')
        ])

        self.append(NoEscape(r"\maketitle"))
        self.append(NoEscape(r'\newpage'))

        self.count = 0

    def fill_document(self, image_path, doc):

        for sub_path in sorted(os.listdir(image_path)):

            sub_folder = os.path.join(image_path, sub_path)

            image_keys = ['anti-32', "aliased-64", 'path-based', 'core-based', "soft mask", "deblurring"]
            # image_keys = ['anti_32', "aliased_64"]

            image_paths = {
                key: os.path.join(sub_folder, filename)
                for key, filename in zip(image_keys, [
                    'padded_l.png',
                    'padded_h.png',
                    'mask_color_march.png',
                    'mask_core_based.png',
                    'confidence',
                    'res.png' 
                ])
            }

            with doc.create(Figure(position="H")) as images:
                for i, key in enumerate(image_keys):
                    with doc.create(
                        SubFigure(position="b", width=NoEscape(r"0.32\linewidth"))
                    ) as subfig:
                        subfig.add_image(image_paths[key], width=NoEscape(r"\linewidth"))
                        subfig.add_caption(key)
                        
                    if (i+1) % 3 == 0:
                        doc.append(NoEscape(r"\par\vspace{1em}"))

                name = sub_path
                images.add_caption(name[4:])
                
                self.count += 1
                if self.count % 3 == 0:
                    self.append(NoEscape(r'\clearpage'))
                    

def display_as_gallery(image_path, output_path):
    doc = MyDocument(output_path)
    doc.fill_document(image_path, doc)
    doc.generate_tex()
       

if __name__ == "__main__":
    output_path = "./temp/gallery"

    image_path = "/root/autodl-tmp/AA_deblurring/val"

    display_as_gallery(image_path, output_path)
