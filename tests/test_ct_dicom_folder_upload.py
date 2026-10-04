import numpy as np
import SimpleITK as sitk

from main import load_ct_volume


def test_load_ct_volume_accepts_dicom_directory(tmp_path):
    study_dir = tmp_path / "ct_series"
    study_dir.mkdir()

    for idx in range(3):
        arr = np.full((12, 12), 100 + idx, dtype=np.uint16)
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing((1.0, 1.0))
        image.SetOrigin((0.0, 0.0))
        image.SetDirection((1.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(image, str(study_dir / f"slice_{idx:03d}.dcm"))

    image, array = load_ct_volume(str(study_dir))

    assert image.GetDimension() == 3
    assert array.ndim == 3
    assert array.shape[0] == 3
    assert array.shape[1:] == (12, 12)
