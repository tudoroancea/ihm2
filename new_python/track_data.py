from strongpods import PODS
import numpy.typing as npt
import numpy as np

@PODS
class TrackGeometryData:
    X_center_line: npt.NDArray[np.float64]
    Y_center_line: npt.NDArray[np.float64]
    right_width: npt.NDArray[np.float64]
    left_width: npt.NDArray[np.float64]


@PODS
class TrackConeData:
    blue: npt.NDArray[np.float64]
    yellow: npt.NDArray[np.float64]
    big_orange: npt.NDArray[np.float64]
    small_orange: npt.NDArray[np.float64] | None


@PODS
class TrackData:
    center_line: TrackGeometryData
    cones: TrackConeData


def load_track_geometry_data(track_name: str) -> TrackGeometryData:
    pass


def load_track_cone_data(track_name: str) -> TrackConeData:
    pass


def load_track_data(track_name: str) -> TrackData:
    pass
