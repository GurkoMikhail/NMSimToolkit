import os
import numpy as np
import slicer

class VolumeHelpers:
    @staticmethod
    def extract_and_save_volume(volumeNode, save_dir, filename):
        """
        Extracts voxel data from vtkMRMLScalarVolumeNode, transposes from (Z,Y,X) to (X,Y,Z).
        Saves as .npy in the provided save_dir.
        Returns the path to the saved .npy file relative to the save_dir.
        """
        if volumeNode is None:
            return None

        # Slicer array is (Z, Y, X)
        array_zyx = slicer.util.arrayFromVolume(volumeNode)

        if array_zyx is None:
            return None

        # Transpose to (X, Y, Z)
        array_xyz = np.transpose(array_zyx, (2, 1, 0))

        # Save as .npy
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)
        np.save(filepath, array_xyz)

        # Return filename as the relative path
        return filename

    @staticmethod
    def extract_voxel_size(volumeNode):
        """
        Extracts spacing from a vtkMRMLScalarVolumeNode.
        Issues a warning if the spacing is not isotropic.
        Returns a formatted scalar string like "1.5 mm".
        """
        if volumeNode is None:
            return None

        spacing = volumeNode.GetSpacing()

        # Check for anisotropy
        tolerance = 1e-5
        if abs(spacing[0] - spacing[1]) > tolerance or abs(spacing[0] - spacing[2]) > tolerance:
            slicer.util.warningDisplay(f"Volume '{volumeNode.GetName()}' has anisotropic spacing {spacing}. "
                                       f"NMSimToolkit expects isotropic voxels. Using X dimension ({spacing[0]}) as voxel_size.")

        return f"{spacing[0]} mm"
