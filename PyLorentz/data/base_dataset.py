import numpy as np
from pathlib import Path
from PyLorentz.io.read import read_image, read_json
from PyLorentz.io.write import write_json
import os
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.ndimage as ndi

# BaseImage?
"""
BaseImage?
- single image, but have cropping extendable to stacks
"""


class BaseDataset(object):

    def __init__(self, imshape):
        # transforms will be rotation -> crop
        self._shape = imshape
        self._transformations = {
            "rotation": 0,
            "top": 0,
            "bottom": self._shape[0],
            "left": 0,
            "right": self._shape[1],
        }

        return

    @classmethod
    def _read_mdata(self, metadata_file) -> dict:
        mdata = read_json(metadata_file)
        if mdata["defocus_unit"] != "nm":
            raise NotImplementedError(f"Unknown defocus unit {mdata['defocus_unit']}")
        if mdata["scale_unit"] != "nm":
            raise NotImplementedError(f"Unknown scale unit {mdata['scale_unit']}")
        return mdata

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val: float):
        if val > 0:
            self._scale = val
        else:
            raise ValueError(f"scale must be > 0, received {val}")

    def crop(self):

        return

    # def __len__

    def _select_ROI(self, image: np.ndarray, print_instructions=True, verbose=True):
        """Select a rectangular region of interest and rotation angle
        assign to self._transformations
        don't do any actual cropping here

        Args:
            infocus (bool, optional): Whether to select a region from the infocus image.
                For some datasets the infocus image will have no contrast, and therefore set
                infocus=False to select a region from the most underfocused image.
                Defaults to True.
        """
        vprint = print if verbose >= 1 else lambda *a, **k: None
        assert (
            image.shape == self._shape
        ), f"Incorrect image shape: expected {self._shape} received {image.shape}"

        if print_instructions and verbose:
            s = (
                "Interactive ROI selection:"
                "\n\tRight click to move corner to mouse position"
                "\n\t'j'/'k' to rotate the image, shift + 'j'/'k' to increase step size"
                "\n\t'n'/'m' to grow/shrink the ROI, shift + 'n'/'m' to increase step size"
                "\n\tarrow keys to move the ROI, + shift to increase step size"
                "\n\t'c' to center the ROI on the middle of the image"
                "\n\t'r' to make the ROI square"
                "\n\t'f' to reset the ROI to the starting conditions, shift+'f' to reset to full image and zero rotation"
                "\n\t'esc' or 'q' when finished to exit"
                "\n(Click on image if not responding in notebook)"
            )
            print(s)

        # needs to take list as input so it can add them
        fig, ax = plt.subplots()

        # ax.matshow(image, cmap="gray")
        dy, dx = self._shape

        start_rotation = self._transformations["rotation"]
        points = np.array(
            [
                [self._transformations["top"], self._transformations["left"]],
                [self._transformations["bottom"], self._transformations["right"]],
            ]
        )
        start_points = points.copy()

        vprint("Starting parameters: ")
        vprint(
            f"Rotation: {start_rotation} | "
            f"Points: "
            + f"({points[0,0]}, {points[0,1]}), ({points[1,0]}, {points[1,1]})"
            + " | Dimensions (h x w): "
            + f"{points[1,0]-points[0,0]} x {points[1,1]-points[0,1]}",
            end="\r",
        )
        self._temp_rotation = start_rotation

        class plotter:
            def __init__(self, points):
                self.plot_image(start_rotation)
                self.scat = None
                self.rect = Rectangle((0, 0), 1, 1, fc="none", ec="red")
                ax.add_patch(self.rect)
                if np.all(points >= 0):
                    self.plotrect(points)
                    self.plot(points)

            def plot(self, points):
                # moving point left/up by 1 if > 0 to prevent plotting outside of window
                if self.scat is not None:
                    self.clear()
                ypoints = points[:, 0][points[:, 0] >= 0]
                xpoints = points[:, 1][points[:, 1] >= 0]
                xpoints = np.where(xpoints == dx, xpoints - 1, xpoints)
                ypoints = np.where(ypoints == dy, ypoints - 1, ypoints)
                self.scat = ax.scatter(xpoints, ypoints, c="r")

            def plot_image(self, rotation):
                imrot = ndi.rotate(image.copy(), rotation, reshape=False)
                ax.matshow(imrot, cmap="gray")

            def print_update(self, rotation):
                vprint(
                    f"Rotation: {rotation} | "
                    f"Points: "
                    + f"({points[0,0]}, {points[0,1]}), ({points[1,0]}, {points[1,1]})"
                    + " | Dimensions (h x w): "
                    + f"{points[1,0]-points[0,0]} x {points[1,1]-points[0,1]}",
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
            # make it move closer point not second one always
            if event.button is MouseButton.RIGHT:
                x, y = event.xdata, event.ydata
                if np.any(points[0] < 0):  # draw point0
                    points[0, 0] = y
                    points[0, 1] = x
                elif np.any(points[1] < 0):  # draw point1
                    points[1, 0] = y
                    points[1, 1] = x
                else:  # redraw closer point
                    dist0 = self._points_dist(points[0], [y, x])
                    dist1 = self._points_dist(points[1], [y, x])
                    if dist0 < dist1:  # change point0
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
                    self._transformations["top"] = points[0, 0]
                    self._transformations["left"] = points[0, 1]
                    self._transformations["bottom"] = points[1, 0]
                    self._transformations["right"] = points[1, 1]
                    self._transformations["rotation"] = self._temp_rotation
                    self._temp_rotation = None
                    vprint(f"dd._transformations: {self._transformations}")
                    vprint(
                        f"Final image dimensions (h x w): {points[1,0]-points[0,0]} x {points[1,1]-points[0,1]}"
                    )
                    vprint(
                        "Cropping can be returned to the full image by running dd.reset_transformations()"
                    )
                else:
                    vprint(f"One or more points are not well defined.: {points}")
                    self.reset_transformations()
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
                points[1, 0] = max(points[0, 0]+1, points[1, 0] - 20)
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
                points[1, 1] = max(points[0, 1]+1, points[1, 1] - 20)
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
                points[1, 0] = max(points[0, 0]+1, points[1, 0] - 1)
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
                points[1, 1] = max(points[0, 1]+1, points[1, 1] - 1)
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

            elif event.key == "f":
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

            elif event.key == "r":
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
            if np.any(points < 0):  # only drawing if not all points not placed
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

        p = plotter(points)
        # p.plot_image()
        binding_id = plt.connect("button_press_event", on_click)
        binding_id2 = plt.connect("motion_notify_event", on_move)
        plt.connect("key_press_event", on_key_press)
        plt.show()
        return

    def reset_transformations(self):
        """Reset the ptie.crop() region to the full image."""
        print("Resetting ROI to unrotated full image.")
        self._transformations["rotation"] = 0
        self._transformations["left"] = 0
        self._transformations["right"] = self._shape[1]
        self._transformations["top"] = 0
        self._transformations["bottom"] = self._shape[0]

    @staticmethod
    def _points_dist(pos1, pos2):
        """Distance between two 2D points

        Args:
            pos1 (list): [y1, x1] point 1
            pos2 (list): [y2, x2] point 2

        Returns:
            float: Euclidean distance between the two points
        """
        assert len(pos1) == len(pos2) == 2
        squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
        return np.sqrt(squared)
