import os
import yaml
from Utils.VolumeHelpers import VolumeHelpers
import qt

class ConfigExporter:
    """
    Traverses UI state and Tree models to generate the final Python dict mirroring
    the Pydantic SimulationConfig. Handles yaml.dump logic and delegates .npy exports.
    """

    def __init__(self, global_settings_widget, scene_tree_widget):
        self.global_settings = global_settings_widget
        self.scene_tree = scene_tree_widget

    def generate_config_dict(self, export_dir):
        """
        Builds the entire configuration dictionary.
        """
        config = {}

        # 1. Global Settings
        config['pool_size'] = self.global_settings.get_pool_size()

        # 2. Simulation Manager
        config['simulation_manager'] = self.global_settings.get_simulation_manager_settings()

        # 3. Data Manager
        config['data_manager'] = self.global_settings.get_data_manager_settings()

        # 4. Protocol
        protocol = self.global_settings.get_protocol_settings()
        config['protocol'] = protocol

        # 5. Scene Editor (Traverse the tree)
        root_node = self.scene_tree.get_root_item()
        config['scene'] = self._traverse_node(root_node, export_dir)

        return config

    def _traverse_node(self, item, export_dir):
        """
        Recursively extracts data from a QTreeWidget item representing a Scene Node.
        """
        # Node properties are stored in UserRole of the item,
        # or we ask the InspectorWidget for the currently stored data.
        # For simplicity, let's assume each item has a custom data object or dict
        # attached to it that contains the node's current configuration.

        node_data = item.data(0, qt.Qt.UserRole)
        if not node_data:
            node_data = {'type': 'CompositeNode'} # Default fallback

        # Copy to avoid modifying the original
        config_dict = node_data.copy()

        # Extract name if any
        name = item.text(0)
        if name and name != "Main Scene": # Main Scene doesn't need name usually, but optional
            pass # Name isn't strictly in AnyNodeConfig unless it's an alias? Let's keep it clean
            # We don't add name unless required by Pydantic, but let's assume it's mostly structural

        # Remove internal Slicer UI state from config
        config_dict.pop('__transform_node_id', None)

        # Handle NumPy Extractions for specific types
        if config_dict.get('type') == 'WoodcockVoxelVolume':
            volume_node_id = config_dict.pop('__volume_node_id', None)
            if volume_node_id:
                import slicer
                volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
                if volume_node:
                    # Export the npy
                    filename = f"volume_{volume_node.GetID()}.npy"
                    npy_path = VolumeHelpers.extract_and_save_volume(volume_node, export_dir, filename)
                    # Set voxel size
                    config_dict['voxel_size'] = VolumeHelpers.extract_voxel_size(volume_node)
                    # Set distribution
                    config_dict['distribution'] = {
                        'format': 'numpy',
                        'path': npy_path
                    }

        elif config_dict.get('type') == 'Source':
            volume_node_id = config_dict.pop('__activity_node_id', None)
            if volume_node_id:
                import slicer
                volume_node = slicer.mrmlScene.GetNodeByID(volume_node_id)
                if volume_node:
                    filename = f"activity_{volume_node.GetID()}.npy"
                    npy_path = VolumeHelpers.extract_and_save_volume(volume_node, export_dir, filename)
                    config_dict['voxel_size'] = VolumeHelpers.extract_voxel_size(volume_node)
                    config_dict['distribution'] = {
                        'format': 'numpy',
                        'path': npy_path
                    }

        # Recursively process children
        children = []
        for i in range(item.childCount()):
            child_item = item.child(i)
            children.append(self._traverse_node(child_item, export_dir))

        if children:
            config_dict['children'] = children

        return config_dict

    def export(self, filepath):
        """
        Exports the config to the given YAML filepath.
        """
        export_dir = os.path.dirname(filepath)

        config_dict = self.generate_config_dict(export_dir)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
