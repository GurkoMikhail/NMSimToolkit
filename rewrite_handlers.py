with open("core/data/data_handlers_soa.py", "r") as f:
    code = f.read()

# Replace getattr with self.writer_callback checks
code = code.replace("if getattr(self, 'writer_callback', None):", "if self.writer_callback is not None:")

# Fix unused imports by checking if "import abc" is the only thing we need to clean up
# but actually the user is complaining about the imports themselves at the top
code = code.replace("import abc\n", "import abc\n")
# "Выдуманные импорты. Все объёмы лежат в volumes"
# Actually TransformableVolume and VolumeWithChilds are in core.geometry.volume according to previous project memory or they don't exist.
