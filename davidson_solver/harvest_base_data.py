from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import read_info


@dataclass(slots=True)
class EOMUCCSDInputData:
    """
    Container for root-independent EOM-UCCSD and triples-correction data.

    Parameters
    ----------
    fock_file
        Path to the file containing orbital counts, orbital energies,
        and one-electron integrals.

    tei_file
        Path to the spin-orbital two-electron integral file.

    amplitudes_file
        Path to the file containing ground-state T1 and T2 amplitudes.

    spin_orbital_ordering
        Ordering used to construct spin-orbital energies from spatial-orbital
        energies:

        "interleaved":
            [eps_0_alpha, eps_0_beta, eps_1_alpha, eps_1_beta, ...]

        "blocked":
            [all alpha energies, all beta energies]

        Your even/odd spin-orbital indexing convention normally requires
        "interleaved".

    sort_orbital_energies
        Sort the spatial-orbital energies before constructing the spin-orbital
        energies. Usually this should remain False because orbital indices must
        stay consistent with the integral and amplitude files.
    """

    fock_file: str | Path
    tei_file: str | Path
    amplitudes_file: str | Path

    spin_orbital_ordering: Literal["interleaved", "blocked"] = "interleaved"
    sort_orbital_energies: bool = False

    # Attributes populated while reading the files.
    nocc: int = field(init=False)
    nvirt: int = field(init=False)

    mo_energies: np.ndarray = field(init=False, repr=False)
    spin_orbital_energies: np.ndarray = field(init=False, repr=False)

    oei: np.ndarray = field(init=False, repr=False)
    tei: np.ndarray = field(init=False, repr=False)

    t1amps: np.ndarray = field(init=False, repr=False)
    t2amps: np.ndarray = field(init=False, repr=False)

    o: slice = field(init=False, repr=False)
    v: slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the paths and load all root-independent input data."""

        self.fock_file = self._validate_file(self.fock_file, "Fock")
        self.tei_file = self._validate_file(self.tei_file, "TEI")
        self.amplitudes_file = self._validate_file(
            self.amplitudes_file,
            "amplitude",
        )

        self._read_fock_data()
        self._build_spin_orbital_energies()
        self._build_orbital_slices()
        self._read_two_electron_integrals()
        self._read_amplitudes()
        self._validate_loaded_data()

    @staticmethod
    def _validate_file(path: str | Path, description: str) -> Path:
        """Convert a path to Path and verify that the file exists."""

        path = Path(path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"{description} file does not exist or is not a file: {path}"
            )

        return path

    def _read_fock_data(self) -> None:
        """Read occupied/virtual counts, orbital energies, and OEIs."""

        nocc, nvirt, mo_energies, oei = read_info.read_fock(
            str(self.fock_file)
        )

        self.nocc = int(nocc)
        self.nvirt = int(nvirt)
        self.mo_energies = np.asarray(mo_energies, dtype=float)
        self.oei = np.asarray(oei)

        if self.mo_energies.ndim != 1:
            raise ValueError(
                "Expected mo_energies to be one-dimensional, but got "
                f"shape {self.mo_energies.shape}."
            )

    def _build_spin_orbital_energies(self) -> None:
        """Construct the spin-orbital energy array."""

        spatial_energies = self.mo_energies.copy()

        if self.sort_orbital_energies:
            spatial_energies = np.sort(spatial_energies)

        if self.spin_orbital_ordering == "interleaved":
            # [eps_0, eps_0, eps_1, eps_1, ...]
            self.spin_orbital_energies = np.repeat(spatial_energies, 2)

        elif self.spin_orbital_ordering == "blocked":
            # [all alpha energies, all beta energies]
            self.spin_orbital_energies = np.concatenate(
                (spatial_energies, spatial_energies)
            )

        else:
            raise ValueError(
                "spin_orbital_ordering must be either "
                "'interleaved' or 'blocked'."
            )

    def _build_orbital_slices(self) -> None:
        """Construct occupied and virtual spin-orbital slices."""

        self.o = slice(0, self.nocc)
        self.v = slice(self.nocc, self.nocc + self.nvirt)

    def _read_two_electron_integrals(self) -> None:
        """Read spin-orbital two-electron integrals."""

        self.tei = np.asarray(
            read_info.read_tei(
                str(self.tei_file),
                self.n_spin_orbitals,
            )
        )

    def _read_amplitudes(self) -> None:
        """Read ground-state T1 and T2 amplitudes."""

        t2amps, t1amps = read_info.read_tamps(
            str(self.amplitudes_file),
            self.nocc,
            self.nvirt,
        )

        self.t1amps = np.asarray(t1amps)
        self.t2amps = np.asarray(t2amps)

    def _validate_loaded_data(self) -> None:
        """Perform basic consistency checks on the loaded data."""

        expected_nso = self.nocc + self.nvirt

        if self.spin_orbital_energies.size != expected_nso:
            raise ValueError(
                "Inconsistent orbital dimensions: "
                f"nocc + nvirt = {expected_nso}, but the constructed "
                "spin-orbital energy array has length "
                f"{self.spin_orbital_energies.size}."
            )

        expected_t1_shape = (self.nocc, self.nvirt)
        expected_t2_shape = (
            self.nocc,
            self.nocc,
            self.nvirt,
            self.nvirt,
        )

        if self.t1amps.shape != expected_t1_shape:
            raise ValueError(
                f"T1 has shape {self.t1amps.shape}; "
                f"expected {expected_t1_shape}."
            )

        if self.t2amps.shape != expected_t2_shape:
            raise ValueError(
                f"T2 has shape {self.t2amps.shape}; "
                f"expected {expected_t2_shape}."
            )

    @property
    def n_spin_orbitals(self) -> int:
        """Total number of spin orbitals."""

        return self.nocc + self.nvirt

    @property
    def n_spatial_orbitals(self) -> int:
        """Number of spatial-orbital energies read from the Fock file."""

        return self.mo_energies.size

    def print_summary(self) -> None:
        """Print a compact summary without printing the full tensors."""

        print("EOM-UCCSD input data")
        print(f"  Fock file:       {self.fock_file}")
        print(f"  TEI file:        {self.tei_file}")
        print(f"  Amplitude file:  {self.amplitudes_file}")
        print(f"  Occupied:        {self.nocc}")
        print(f"  Virtual:         {self.nvirt}")
        print(f"  Spin orbitals:   {self.n_spin_orbitals}")
        print(f"  OEI shape:       {self.oei.shape}")
        print(f"  TEI shape:       {self.tei.shape}")
        print(f"  T1 shape:        {self.t1amps.shape}")
        print(f"  T2 shape:        {self.t2amps.shape}")


def build_d3_for_root(
    inputs: EOMUCCSDInputData,
    omega,
    level_shift=0.0,
    zero_tolerance=1.0e-12,
):
    """
    Build a root-dependent D3 inverse denominator from stored input data.
    """
    import read_info as ri
    return ri.build_inverse_d3_denominator(
        spin_orbital_energies=inputs.spin_orbital_energies,
        occupied=inputs.o,
        virtual=inputs.v,
        omega=omega,
        level_shift=level_shift,
        zero_tolerance=zero_tolerance,
    )

