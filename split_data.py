from pathlib import Path
from shutil import copy

import numpy as np
from PIL import Image


def split_train_data():
    path = "/home/winfried/data/apdrawing_dataset/APDrawingDB/data/test"
    sp_a = "/home/winfried/data/apdrawing_dataset/pix2pix/val_a"
    sp_b = "/home/winfried/data/apdrawing_dataset/pix2pix/val_b"

    Path(sp_a).mkdir(exist_ok=True, parents=True)
    Path(sp_b).mkdir(exist_ok=True, parents=True)

    for i, p in enumerate(Path(path).glob("*.png")):
        img = np.array(Image.open(p))
        im1 = img[:, :512]
        im2 = img[:, 512:]

        Image.fromarray(im1).save(f"{sp_a}/{i}.png")
        Image.fromarray(im2).save(f"{sp_b}/{i}.png")

    print("done")


def split_result(name, it=0):
    it = "" if it == 0 else f"_iter{it}"

    path = f"/media/winfried/Daten/data_external/pix2pix_SEPT/results/{name}/test_latest{it}/images"
    ppp = Path(path) / f"../../result{it}"
    ppp.mkdir(exist_ok=True, parents=True)

    for i, p in enumerate(Path(path).glob("*.png")):
        if p.name[-10:] == "fake_B.png":
            copy(str(p), str(ppp))


if __name__ == '__main__':
    split_result("apdrawing_C")
    split_result("apdrawing_C2")
    split_result("apdrawing_A")
    split_result("apdrawing_A2")
    split_result("apdrawing_B")
    split_result("apdrawing_B2")
    split_result("apdrawing_D")
    split_result("apdrawing_D2")
    split_result("apdrawing_D3")
    split_result("apdrawing_E")
    split_result("apdrawing_E2")
    split_result("apdrawing_F2")
    split_result("apdrawing_F")

    print("done")
