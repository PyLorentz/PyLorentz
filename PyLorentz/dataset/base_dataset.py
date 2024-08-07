import os
import warnings
from pathlib import Path
from typing import Optional, Union
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle

from PyLorentz.io.read import read_image, read_json
from PyLorentz.io.write import format_defocus, write_json
from PyLorentz.utils.filter import bandpass_filter

# Remapping keybindings for interactive matplotlib figures
mpl.rcParams.update(
    {
        "keymap.home": "",
        "keymap.back": "",
        "keymap.forward": "",
        "keymap.pan": "",
        "keymap.zoom": "",
        "keymap.save": "",
        "keymap.fullscreen": "",
        "keymap.grid": "",
        "keymap.grid_minor": "",
        "keymap.xscale": "",
        "keymap.yscale": "",
        "keymap.quit": "Q",
    }
)


class BaseDataset:
    """
    A base class for handling datasets, providing common attributes and methods.
    """

    def __init__(
        self,
        imshape: Optional[Union[tuple, np.ndarray]] = None,
        data_dir: Optional[os.PathLike] = None,
        scale: Optional[float] = None,
        verbose: Union[int, bool] = 1,
    ):
        """
        Initialize the BaseDataset object.

        Args:
            imshape (tuple | np.ndarray | None, optional): Shape of the image. Default is None.
            data_dir (os.PathLike | None, optional): Directory for data storage. Default is None.
            scale (float | None, optional): Scale factor for the dataset. Default is None.
            verbose (int | bool, optional): Verbosity level. Default is 1.
        """
        self._shape = imshape
        self.scale = scale
        self._transforms = {
            "rotation": 0,
            "top": 0,
            "bottom": imshape[0] if imshape else None,
            "left": 0,
            "right": imshape[1] if imshape else None,
        }
        self._filters = {
            "hotpix": False,
            "median": False,
            "q_highpass": None,
            "q_lowpass": None,
            "filter_type": None,
            "butterworth_order": None,
        }
        self._verbose = verbose
        self.data_dir = data_dir
        self._transforms_modified = False

    @staticmethod
    def _parse_mdata(metadata_file: Union[os.PathLike, dict]) -> dict:
        """
        Parse metadata from a file or dictionary.

        Args:
            metadata_file (os.PathLike | dict): Path to the metadata file or dictionary.

        Returns:
            dict: Parsed metadata.
        """
        if isinstance(metadata_file, dict):
            mdata = metadata_file
        else:
            mdata = read_json(metadata_file)

        keys = mdata.keys()
        if "defocus_values" not in keys:
            mdata["defocus_values"] = None
        if "defocus_unit" not in keys:
            print(
                f"`defocus_unit` not found in metadata file: {metadata_file}"
                + "\nSetting defocus_unit = `nm`"
            )
            mdata["defocus_unit"] = "nm"
        if "scale" not in keys:
            mdata["scale"] = None
        if "scale_unit" not in keys:
            print(
                f"`scale_unit` not found in metadata file: {metadata_file}"
                + "\nSetting scale_unit = `nm`"
            )
            mdata["scale_unit"] = "nm"
        if "beam_energy" not in keys:
            mdata["beam_energy"] = None

        defocus_unit = mdata["defocus_unit"].lower()
        if defocus_unit != "nm":
            unit_conversion = {"um": 1e3, "μm": 1e3, "mm": 1e6, "m": 1e9}
            conversion_factor = unit_conversion.get(defocus_unit)
            if conversion_factor:
                mdata["defocus_values"] = (
                    np.array(mdata["defocus_values"]) * conversion_factor
                )
            else:
                raise NotImplementedError(
                    f"Unknown defocus unit {mdata['defocus_unit']}"
                )
            mdata["defocus_unit"] = "nm"

        scale_unit = mdata["scale_unit"].lower()
        if scale_unit != "nm":
            unit_conversion = {"um": 1e3, "μm": 1e3, "mm": 1e6, "m": 1e9}
            conversion_factor = unit_conversion.get(scale_unit)
            if conversion_factor:
                mdata["scale"] = mdata["scale"] * conversion_factor
            else:
                raise NotImplementedError(f"Unknown scale unit {mdata['scale_unit']}")
            mdata["scale_unit"] = "nm"

        return mdata

    @property
    def shape(self):
        """Get the shape of the image."""
        return self._shape

    @property
    def data_dir(self):
        """Get the data directory."""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, p: Union[os.PathLike, None]):
        """Set the data directory."""
        if p is None:
            self._data_dir = p
        else:
            p = Path(p).absolute()
            if not p.exists():
                warnings.warn(
                    f"data_dir does not exist, but setting anyways. data_dir = {p}"
                )
            self._data_dir = p

    @property
    def scale(self):
        """Get the scale factor."""
        return self._scale

    @scale.setter
    def scale(self, val: Union[float, None]):
        """Set the scale factor."""
        if val is None:
            self._scale = None
        elif val > 0:
            self._scale = val
        else:
            raise ValueError(f"scale must be > 0, received {val}")

    def crop(self):
        """Placeholder for crop method."""
        pass

    def _select_ROI(self, image: np.ndarray, print_instructions=True, verbose=True):
        """
        Select a rectangular region of interest and rotation angle, and assign to self._transforms.

        Args:
            image (np.ndarray): Image to select ROI from.
            print_instructions (bool, optional): Whether to print instructions. Default is True.
            verbose (bool, optional): Verbosity level. Default is True.
        """
        vprint = print if verbose >= 1 else lambda *a, **k: None
        assert (
            image.shape == self._shape
        ), f"Incorrect image shape: expected {self._shape} received {image.shape}"

        if print_instructions and verbose:
            instructions = (
                "Interactive ROI selection:"
                "\n\tRight click | move closest corner to mouse position"
                "\n\t'j'/'k'     | rotate the image, shift + 'j'/'k' to increase step size"
                "\n\t'n'/'m'     | grow/shrink the ROI, shift + 'n'/'m' to increase step size"
                "\n\tarrow keys  | move the ROI, + shift to increase step size"
                "\n\t'c'         | center the ROI on the middle of the image"
                "\n\t's'         | make the ROI square"
                "\n\tshift+'r'   | reset the ROI to the starting conditions"
                "\n\tshift+'f'   | restore the full image with zero rotation"
                "\n\t'esc'       | save transforms and exit"
                "\nIf display is not responding, try clicking on the image and ensuring "
                "you ran %matplotlib widget"
            )
            print(instructions)

        fig, ax = plt.subplots()
        dy, dx = self._shape

        start_rotation = self._transforms["rotation"]
        points = np.array(
            [
                [self._transforms["top"], self._transforms["left"]],
                [self._transforms["bottom"], self._transforms["right"]],
            ]
        )
        start_points = points.copy()
        self._temp_rotation = start_rotation

        class Plotter:
            def __init__(self, points):
                self.plot_image(start_rotation)
                self.scat = None
                self.rect = Rectangle((0, 0), 1, 1, fc="none", ec="red")
                ax.add_patch(self.rect)
                if np.all(points >= 0):
                    self.plotrect(points)
                    self.plot(points)

            def plot(self, points):
                if self.scat is not None:
                    self.clear()
                ypoints = points[:, 0][points[:, 0] >= 0]
                xpoints = points[:, 1][points[:, 1] >= 0]
                xpoints = np.where(xpoints == dx, xpoints - 1, xpoints)
                ypoints = np.where(ypoints == dy, ypoints - 1, ypoints)
                self.scat = ax.scatter(xpoints, ypoints, c="r")

            def plot_image(self, rotation):
                im_min = image.min()
                imrot = ndi.rotate(image.copy(), rotation, reshape=False, order=1)
                imrot[imrot == 0] = im_min
                ax.matshow(imrot, cmap="gray")

            def print_update(self, rotation):
                vprint(
                    f"Rotation: {rotation:4} | Points: ({points[0,0]:4}, {points[0,1]:4}), "
                    + f"({points[1,0]:4}, {points[1,1]:4}) | Dimensions (h x w): "
                    + f"{points[1,0]-points[0,0]:4} x {points[1,1]-points[0,1]:4}",
                    end="\r",
                )

            def plotrect(self, points):
                (y0, x0), (y1, x1) = points
                self.rect.set_width(x1 - x0)
                self.rect.set_height(y1 - y0)
                self.rect.set_xy((x0, y0))
                ax.figure.canvas.draw()

            def clear(self):
                self.scat.remove()
                self.scat = None

        def on_click(event):
            if event.button is MouseButton.RIGHT:
                x, y = event.xdata, event.ydata
                if np.any(points[0] < 0):
                    points[0, 0] = y
                    points[0, 1] = x
                elif np.any(points[1] < 0):
                    points[1, 0] = y
                    points[1, 1] = x
                else:
                    dist0 = self._points_dist(points[0], [y, x])
                    dist1 = self._points_dist(points[1], [y, x])
                    if dist0 < dist1:
                        points[0, 0] = y
                        points[0, 1] = x
                    else:
                        points[1, 0] = y
                        points[1, 1] = x
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)

                p.print_update(self._temp_rotation)

        # TODO make a couple helper functions to shrink this
        def on_key_press(event):
            if event.key == "escape" or event.key == "q":
                if np.all(points >= 0):
                    plt.disconnect(binding_id)
                    plt.disconnect(binding_id2)
                    self._transforms["top"] = points[0, 0]
                    self._transforms["left"] = points[0, 1]
                    self._transforms["bottom"] = points[1, 0]
                    self._transforms["right"] = points[1, 1]
                    self._transforms["rotation"] = self._temp_rotation
                    print("\nSetting transforms.")
                    vprint(
                        f"Final image dimensions (h x w): {points[1,0]-points[0,0]} x "
                        + f"{points[1,1]-points[0,1]}"
                    )
                    vprint(
                        "Cropping can be returned to the full image by running "
                        + "self.reset_transforms()"
                    )
                    self._temp_rotation = None
                    self._transforms_modified = True
                else:
                    vprint(f"One or more points are not well defined.: {points}")
                    self._reset_transforms()
                plt.close(fig)
                return

            elif event.key == "k":
                self._temp_rotation += 1
                p.plot_image(self._temp_rotation)
                p.print_update(self._temp_rotation)

            elif event.key == "j":
                self._temp_rotation -= 1
                p.plot_image(self._temp_rotation)
                p.print_update(self._temp_rotation)

            elif event.key == "K":
                self._temp_rotation += 15
                p.plot_image(self._temp_rotation)
                p.print_update(self._temp_rotation)

            elif event.key == "J":
                self._temp_rotation -= 15
                p.plot_image(self._temp_rotation)
                p.print_update(self._temp_rotation)

            elif event.key == "n":
                points[0, 0] = max(0, points[0, 0] - 1)
                points[0, 1] = max(0, points[0, 1] - 1)
                points[1, 0] = min(self._shape[0], points[1, 0] + 1)
                points[1, 1] = min(self._shape[1], points[1, 1] + 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "N":
                points[0, 0] = max(0, points[0, 0] - 20)
                points[0, 1] = max(0, points[0, 1] - 20)
                points[1, 0] = min(self._shape[0], points[1, 0] + 20)
                points[1, 1] = min(self._shape[1], points[1, 1] + 20)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "m":
                points[0, 0] = min(points[0, 0] + 1, points[1, 0] - 1)
                points[0, 1] = min(points[0, 1] + 1, points[1, 1] - 1)
                points[1, 0] = max(points[1, 0] - 1, points[0, 0] + 1)
                points[1, 1] = max(points[1, 1] - 1, points[0, 1] + 1)
                p.plot(points)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "M":
                points[0, 0] = min(points[0, 0] + 20, points[1, 0] - 1)
                points[0, 1] = min(points[0, 1] + 20, points[1, 1] - 1)
                points[1, 0] = max(points[1, 0] - 20, points[0, 0] + 1)
                points[1, 1] = max(points[1, 1] - 20, points[0, 1] + 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "shift+up":
                points[0, 0] = max(0, points[0, 0] - 20)
                points[1, 0] = max(points[0, 0] + 1, points[1, 0] - 20)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "shift+down":
                points[1, 0] = min(self._shape[0], points[1, 0] + 20)
                points[0, 0] = min(points[1, 0] - 1, points[0, 0] + 20)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "shift+left":
                points[0, 1] = max(0, points[0, 1] - 20)
                points[1, 1] = max(points[0, 1] + 1, points[1, 1] - 20)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "shift+right":
                points[1, 1] = min(self._shape[1], points[1, 1] + 20)
                points[0, 1] = min(points[1, 1] - 1, points[0, 1] + 20)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "up":
                points[0, 0] = max(0, points[0, 0] - 1)
                points[1, 0] = max(points[0, 0] + 1, points[1, 0] - 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "down":
                points[1, 0] = min(self._shape[0], points[1, 0] + 1)
                points[0, 0] = min(points[1, 0] - 1, points[0, 0] + 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "left":
                points[0, 1] = max(0, points[0, 1] - 1)
                points[1, 1] = max(points[0, 1] + 1, points[1, 1] - 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "right":
                points[1, 1] = min(self._shape[1], points[1, 1] + 1)
                points[0, 1] = min(points[1, 1] - 1, points[0, 1] + 1)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "R":
                points[0, 0] = start_points[0, 0]
                points[0, 1] = start_points[0, 1]
                points[1, 0] = start_points[1, 0]
                points[1, 1] = start_points[1, 1]
                self._temp_rotation = start_rotation

                p.plot_image(self._temp_rotation)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "F":
                points[0, 0] = 0
                points[0, 1] = 0
                points[1, 0] = self._shape[0]
                points[1, 1] = self._shape[1]
                self._temp_rotation = 0

                p.plot_image(self._temp_rotation)
                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "s":
                dimy = points[1, 0] - points[0, 0]
                dimx = points[1, 1] - points[0, 1]
                if dimy > dimx:
                    points[1, 0] = points[0, 0] + dimx
                elif dimy < dimx:
                    points[1, 1] = points[0, 1] + dimy

                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            elif event.key == "c":
                "center"
                cy, cx = self._shape[0] // 2, self._shape[1] // 2
                dimy = points[1, 0] - points[0, 0]
                dimx = points[1, 1] - points[0, 1]
                points[0, 0] = cy - dimy // 2
                points[0, 1] = cx - dimx // 2
                points[1, 0] = cy + dimy // 2
                points[1, 1] = cx + dimx // 2

                p.plot(points)
                if np.all(points >= 0):
                    p.plotrect(points)
                p.print_update(self._temp_rotation)

            # else:
            #     print("key: ", event.key)

        def on_move(event):
            if np.any(points < 0):
                if event.xdata is not None and event.ydata is not None:
                    if 0 < event.xdata < dx and 0 < event.ydata < dy:
                        if np.all(points[0] > 0):
                            y0, x0 = points[0]
                        elif np.all(points[1] > 0):
                            y0, x0 = points[1]
                        else:
                            return
                        x1 = event.xdata
                        y1 = event.ydata
                        p.plotrect([[y0, x0], [y1, x1]])

        p = Plotter(points)
        binding_id = plt.connect("button_press_event", on_click)
        binding_id2 = plt.connect("motion_notify_event", on_move)
        plt.connect("key_press_event", on_key_press)
        plt.show()
        print("Current parameters:")

    def _reset_transforms(self):
        """Reset the ROI to the full image."""
        print("Resetting ROI to unrotated full image.")
        self._transforms["rotation"] = 0
        self._transforms["left"] = 0
        self._transforms["right"] = self._shape[1]
        self._transforms["top"] = 0
        self._transforms["bottom"] = self._shape[0]
        self._transforms_modified = True

    @property
    def transforms(self):
        """Get the transformation parameters."""
        return self._transforms

    @transforms.setter
    def transforms(self, d, verbose=True):
        """Set the transformation parameters."""
        if not hasattr(self, "_transforms"):
            self._transforms = {}
        if not isinstance(d, dict):
            raise TypeError(f"transforms should be dict, not {type(d)}")
        for key, val in d.items():
            key_lower = key.lower()
            if key_lower in ["rotation", "rot", "r"]:
                self._transforms["rotation"] = val
            elif key_lower in ["top", "t"]:
                self._transforms["top"] = val
            elif key_lower in ["bottom", "bot", "b"]:
                self._transforms["bottom"] = val
            elif key_lower in ["left", "l"]:
                self._transforms["left"] = val
            elif key_lower in ["right", "r"]:
                self._transforms["right"] = val
            else:
                s = (
                    f"Unknown key in transforms: {key}\n"
                    + "Allowed keys are 'rotation', 'top', 'bottom', 'left', 'right'"
                )
                warnings.warn(s)

        if self._verbose and verbose:
            rotation = self._transforms["rotation"]
            points = np.array(
                [
                    [self._transforms["top"], self._transforms["left"]],
                    [self._transforms["bottom"], self._transforms["right"]],
                ]
            )
            print(
                f"Rotation: {rotation:4} | Points: ({points[0,0]:4}, {points[0,1]:4}), "
                + f"({points[1,0]:4}, {points[1,1]:4}) | Dimensions (h x w): "
                + f"{points[1,0]-points[0,0]:4} x {points[1,1]-points[0,1]:4}",
            )
        self._transforms_modified = True

    @staticmethod
    def _points_dist(pos1, pos2):
        """
        Calculate the Euclidean distance between two 2D points.

        Args:
            pos1 (list): [y1, x1] point 1
            pos2 (list): [y2, x2] point 2

        Returns:
            float: Euclidean distance between the two points
        """
        assert len(pos1) == len(pos2) == 2
        squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
        return np.sqrt(squared)

    @staticmethod
    def _fmt_defocus(defval: Union[float, int], digits: int = 3, spacer: str = " "):
        """
        Format defocus value for display.

        Args:
            defval (Union[):,efocus] value.
            digits (int, optional): Number of digits. Default is 3.
            spacer (str, optional): Spacer string. Default is " ".

        Returns:
            str: Formatted defocus value.
        """
        return format_defocus(defval, digits, spacer=spacer)

    def vprint(self, *args, **kwargs):
        """Print messages if verbose is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    def _bandpass_filter(
        self,
        image: np.ndarray,
        q_lowpass: Optional[float] = None,
        q_highpass: Optional[float] = None,
        filter_type: str = "butterworth",
        butterworth_order: int = 2,
    ) -> np.ndarray:
        """
        Apply a bandpass filter to the image.

        Args:
            image (np.ndarray): Image to filter.
            q_lowpass (float | None, optional): Low-pass filter cutoff. Default is None.
            q_highpass (float | None, optional): High-pass filter cutoff. Default is None.
            filter_type (str, optional): Filter type, "butterworth" or "gaussian". Default is "butterworth".
            butterworth_order (int, optional): Order of the Butterworth filter. Default is 2.

        Returns:
            np.ndarray: Filtered image.
        """
        return bandpass_filter(
            image,
            sampling=1 / self.scale,
            q_lowpass=q_lowpass,
            q_highpass=q_highpass,
            filter_type=filter_type,
            butterworth_order=butterworth_order,
        )

    @property
    def fov(self):
        """Get the field of view."""
        return np.array(self.shape) * self.scale
