"""Doublure de `comfy.utils.ProgressBar` : importée par le pipeline, jamais utilisée en texturage."""


class ProgressBar:
    def __init__(self, total=0, *args, **kwargs):
        self.total = total

    def update(self, *args, **kwargs):
        pass

    def update_absolute(self, *args, **kwargs):
        pass
