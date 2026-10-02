import os
import numpy as np
import nibabel as nib

def create_sample_liver_ct(output_path="data/test.nii.gz"):
    """
    Creates a realistic 3D abdominal CT volume (Hounsfield Units)
    with liver parenchyma, spine/ribs (bone), spleen, fat, soft tissue, and a focal liver lesion.
    Voxel dimensions: (X=256, Y=256, Z=64) -> SimpleITK reads as (64, 256, 256)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Standard medical imaging shape: (x, y, z) = (256, 256, 64)
    nx, ny, nz = 256, 256, 64
    volume = np.full((nx, ny, nz), -1000.0, dtype=np.float32)  # Air background (-1000 HU)
    
    # 3D coordinate grids (X: Left-Right, Y: Anterior-Posterior, Z: Superior-Inferior)
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    
    # 1. Abdominal body contour (elliptical cylinder, soft tissue: ~40 HU, fat layer: ~ -70 HU)
    body_mask = ((x - 128.0)**2 / (105.0**2) + (y - 128.0)**2 / (85.0**2)) <= 1.0
    fat_mask = body_mask & (((x - 128.0)**2 / (96.0**2) + (y - 128.0)**2 / (76.0**2)) > 1.0)
    
    volume[body_mask] = 40.0 + np.random.normal(0, 4, size=volume[body_mask].shape)
    volume[fat_mask] = -75.0 + np.random.normal(0, 6, size=volume[fat_mask].shape)
    
    # 2. Spine & Ribs (Bone HU: 400 - 800)
    spine_mask = (((x - 128.0)**2 / (16.0**2) + (y - 185.0)**2 / (15.0**2)) <= 1.0) & body_mask
    volume[spine_mask] = 650.0 + np.random.normal(0, 30, size=volume[spine_mask].shape)
    
    # 3. Liver anatomy (Anatomical right = Image left: x: 50..130, y: 70..170, z: 12..52)
    # HU for normal liver parenchyma is typically 50 to 70 HU in portal venous CT
    liver_mask = (
        ((x - 88.0)**2 / (42.0**2) + (y - 122.0)**2 / (40.0**2) + (z - 32.0)**2 / (22.0**2)) <= 1.0
    ) & body_mask
    volume[liver_mask] = 65.0 + np.random.normal(0, 5, size=volume[liver_mask].shape)
    
    # 4. Spleen (Anatomical left = Image right: x: 170..205, y: 115..155)
    spleen_mask = (
        ((x - 172.0)**2 / (24.0**2) + (y - 130.0)**2 / (28.0**2) + (z - 32.0)**2 / (18.0**2)) <= 1.0
    ) & body_mask
    volume[spleen_mask] = 50.0 + np.random.normal(0, 5, size=volume[spleen_mask].shape)
    
    # 5. Hypoattenuating Focal Liver Lesion / Tumor (HU ~ 25-35 in center of right liver lobe)
    lesion_mask = (
        ((x - 82.0)**2 / (12.0**2) + (y - 118.0)**2 / (11.0**2) + (z - 32.0)**2 / (8.0**2)) <= 1.0
    ) & liver_mask
    volume[lesion_mask] = 28.0 + np.random.normal(0, 4, size=volume[lesion_mask].shape)
    
    # Save standard NIfTI image with 1.5mm x 1.5mm x 3.0mm voxel spacing
    affine = np.diag([1.5, 1.5, 3.0, 1.0])
    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nifti_img, output_path)
    print(f"[OK] Created sample CT scan at: {output_path} (NIfTI shape: {volume.shape}, Range: [{volume.min():.1f}, {volume.max():.1f}] HU)")

if __name__ == "__main__":
    create_sample_liver_ct()

