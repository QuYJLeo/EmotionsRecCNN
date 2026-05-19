#!/usr/bin/envpython
# -*-coding:UTF-8-*-
import time
import numpy as np

class Fps:
    """FPS (Frames Per Second) counter with exponential smoothing.
    
    This class calculates and displays the frames per second with a smoothing
    factor to provide a more stable reading.

    Args:
        name (str): Name for the FPS counter, used in output
    """
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lastTime = time.time()
        self.fps = None
        self.name = name if name else __name__

    def show(self):
        """Calculate and display the current FPS.
        
        Returns:
            float: Time difference since last call
        """
        now = time.time()
        dt = time.time() - self.lastTime
        if dt == 0:
            return dt
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0 / dt) * s
        print('%s %0.2f fps' % (self.name, self.fps))


if __name__ == "__main__":
    pass
