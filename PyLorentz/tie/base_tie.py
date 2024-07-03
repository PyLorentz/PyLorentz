import numpy as np
from PyLorentz.data.defocused_dataset import DefocusedDataset as DD
import os
from pathlib import Path


class BasePhaseReconstruction(object):

    def __init__(
        self,
        save_dir: os.PathLike|None=None,
        name: str = "",
        verbose: int|bool = 1,
    ):
        self._save_dir = Path(save_dir).absolute()
        self._name = str(name)
        self._verbose = verbose

        self._results = {
            "By": None,
            "Bx": None,
            "phase_B": None,
        }

        return



    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def By(self):
        return self._results["By"]

    @property
    def Bx(self):
        return self._results["Bx"]

    @property
    def B(self):
        return np.array([self._results["By"], self._results["Bx"]])


    @property
    def phase_B(self):
        return self._results["phase_B"]


    def vprint(self,*args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, p):
        p = Path(p)
        if not p.parents[0].exists():
            raise ValueError(f"save dir parent does not exist: {p.parents[0]}")
        else:
            self._save_dir = p

    def _save_keys(self, keys):
        # for k in keys:
        #     self._save_result(key)
        return

    def _save_key(self, key, color=False, fiji=True):
        return






class BaseTIE(BasePhaseReconstruction):

    def __init__(
        self,
        save_dir: os.PathLike|None=None,
        name: str = "",
        sym: bool = False,
        qc: float | None = None,
        verbose: int|bool = 1,
    ):
        BasePhaseReconstruction.__init__(self, save_dir, name, verbose)
        self._sym = sym
        self._qc = qc
        return


    @property
    def sym(self):
        return self._sym

    @sym.setter
    def sym(self, val):
        if isinstance(val, bool):
            self._sym = val
        else:
            raise ValueError(f"sym must be bool, not {type(val)}")

    @property
    def qc(self):
        return self._qc

    @qc.setter
    def qc(self, val):
        if val is None:
            self._qc = 0
        elif isinstance(val, (float, int)):
            if val < 0:
                raise ValueError(f"qc must be >= 0, not {val}")
            self._qc = float(val)
        else:
            raise ValueError(f"qc must be float, not {type(val)}")


    def show_B(self):
        """
        show induction
        """
        return

    def show_phase(self):
        """
        show_phase
        """
        return
