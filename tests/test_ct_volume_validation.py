import numpy as np
import pytest
import SimpleITK as sitk

from main import load_ct_volume


def test_rejects_2d_image_as_ct_volume(tmp_path):
    path = tmp_path / "slice.png"
    image = sitk.GetImageFromArray(np.zeros((24, 24), dtype=np.uint8))
    sitk.WriteImage(image, str(path))

    with pytest.raises(ValueError, match="3D, single-channel CT volume"):
        load_ct_volume(str(path))


def test_loads_scalar_3d_ct_volume(tmp_path):
    path = tmp_path / "volume.nii.gz"
    expected = np.zeros((3, 24, 24), dtype=np.int16)
    image = sitk.GetImageFromArray(expected)
    sitk.WriteImage(image, str(path))

    loaded_image, loaded_array = load_ct_volume(str(path))

    assert loaded_image.GetDimension() == 3
    assert loaded_array.shape == expected.shape
    assert np.array_equal(loaded_array, expected)
