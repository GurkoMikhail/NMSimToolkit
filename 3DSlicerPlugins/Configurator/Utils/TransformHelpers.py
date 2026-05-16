import numpy as np

class TransformHelpers:
    @staticmethod
    def extract_matrix(transformNode):
        """
        Extracts vtkMatrix4x4 from a vtkMRMLTransformNode.
        Converts it into a standard 4x4 Python list of lists (row-major).
        """
        import vtk
        if transformNode is None:
            return None

        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)

        result = []
        for row in range(4):
            row_vals = []
            for col in range(4):
                row_vals.append(matrix.GetElement(row, col))
            result.append(row_vals)

        return {
            'type': 'matrix',
            'value': result
        }
