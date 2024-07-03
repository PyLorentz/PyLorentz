import numpy as np
from PyLorentz.data.defocused_dataset import DefocusedDataset as DD
import os
from PyLorentz.tie.base_tie import BaseTIE
from pathlib import Path

class TIE(BaseTIE):

    def __init__(
        self,
        dd: DD,
        save_dir: os.PathLike|None=None,
        name: str = "",
        sym: bool = False,
        qc: float | None = None,
        verbose: int = 1,
    ):
        self.dd = dd
        if save_dir is None and dd.data_dir is not None:
            topdir = Path(dd.data_dir)
            if topdir.exists():
                save_dir = topdir / "TIE_outputs"

        BaseTIE.__init__(self, save_dir, name, sym, qc, verbose)
        self.qc = qc # for type checking
        self.sym = sym

        self._results["phase_E"] = None
        self._defval = None
        self._defval_index = None

        return


    def reconstruct(
        self,
        index: int | None = None,
        name: str = "",
        sym: bool = False,
        qc: float | None = None,
        save: bool=False,
        save_dir: os.PathLike|None =None,
        verbose: int|bool=1,
    ):
        assert isinstance(index, int)
        assert index < len(self.dd.defvals_index)
        self._defval_index = index
        self._defval = self.dd.defvals_index[index]
        self.name = name
        self.sym = sym
        self.qc = qc
        self._verbose = verbose

        if save:
            # checking here that dir exists before running
            self.save_dir = save_dir



        # do recon




        if save:
            # do save
            pass

        return

    @property
    def defval(self):
        if self._defval is None:
            print(f"defval is None or has not yet been specified with an index")
        return self._defval

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError(f"flip must be bool, not {type(val)}")
        if self.dd.flip:
            self._flip = val
        else:
            if val:
                raise ValueError(f"Cannot set flip=True because dataset has only only one TFS")
            else:
                self._flip = val

    @property
    def phase_E(self):
        if self.flip:
            return self._results["phase_E"]
        else:
            if self._results["phase_E"] is not None:
                self.vprint("Returning old phase_E as currently flip=False")
            else:
                raise ValueError(f"phase_E does not exist because flip=False")

    def visualize(self):
        """
        show phase + induction, if flip then show phase_e too
        options to save
        """

        return


class SITE(BaseTIE):

    def __init__(self):
        return