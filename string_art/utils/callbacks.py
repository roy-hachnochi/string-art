from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Optional, Tuple
from abc import ABC, abstractmethod
import os
import cv2
import numpy as np

if TYPE_CHECKING:
    from string_art.canvas.canvas import Canvas
    from string_art.globals import LINES_TYPE


class IterationCallback(ABC):
    def __init__(self, period: int):
        self.period = period

    @abstractmethod
    def __call__(self, iteration: int, canvas: Canvas, lines: Optional[LINES_TYPE] = None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass


class SaveCanvasCallback(IterationCallback):
    def __init__(self, period: int, path: str):
        super(SaveCanvasCallback, self).__init__(period)
        self.path = path
        os.makedirs(path, exist_ok=True)

    def __call__(self, iteration: int, canvas: Canvas, lines: Optional[LINES_TYPE] = None):
        if (iteration + 1) % self.period == 0:
            if lines is not None:
                canvas.render(lines)
            canvas.save(os.path.join(self.path, f'{iteration + 1}.jpg'))


class PlotCanvasCallback(IterationCallback):
    def __init__(self, period: int, show_nails: bool = True):
        super(PlotCanvasCallback, self).__init__(period)
        self.show_nails = show_nails

    def __call__(self, iteration: int, canvas: Canvas, lines: Optional[LINES_TYPE] = None):
        if (iteration + 1) % self.period == 0:
            if lines is not None:
                canvas.render(lines)
            canvas.plot(show_nails=self.show_nails)


class SaveCanvasMulticolorCallback(IterationCallback):
    def __init__(self, period: int, path: str):
        super(SaveCanvasMulticolorCallback, self).__init__(period)
        self.path = path
        self.cur_color = -1
        self.cur_path = None
        os.makedirs(path, exist_ok=True)

    def __call__(self, iteration: int, canvas: Canvas, lines: Optional[LINES_TYPE] = None):
        if iteration == 0:  # new color -> update counter and make new folder
            self.cur_color += 1
            self.cur_path = os.path.join(self.path, f'color_{self.cur_color}')
            os.makedirs(self.cur_path, exist_ok=True)
        if (iteration + 1) % self.period == 0:
            if lines is not None:
                canvas.render(lines)
            canvas.save(os.path.join(self.cur_path, f'{iteration + 1}.jpg'))


class SaveGifCallback(IterationCallback):
    def __init__(self, period: int, path: str, resolution: Tuple[int, int], fps: int = 20):
        super(SaveGifCallback, self).__init__(period)
        self.path = path
        res = (resolution[1], resolution[0])
        codec = "mp4v" if "ipykernel" in sys.modules else "avc1"  # avc1 is better but isn't supported in notebooks
        self.video_writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), fps, res)

    def __call__(self, iteration: int, canvas: Canvas, lines: Optional[LINES_TYPE] = None):
        if (iteration + 1) % self.period == 0:
            image = (canvas.get_image() * 255).astype(np.uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            else:
                image = np.transpose(image, (1, 2, 0))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.video_writer.write(image)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
