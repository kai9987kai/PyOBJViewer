from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as _np
except ImportError:
    _np = None

__version__ = "2.0.0"

DEFAULT_COLOR = "#6a8bd6"
LIGHT_DIRECTION = (-0.45, 0.65, -1.0)
NEAR_CLIP = 0.08
NORMAL_EPS = 1e-22
CREASE_ANGLE_DEG = 52.0
NUMPY_VERTEX_THRESHOLD = 1500
SHADOW_MIN_LIGHT_Y = 0.08

GAMMA = 2.2
_INV_GAMMA = 1.0 / GAMMA
_SRGB_TO_LINEAR = tuple((i / 255.0) ** GAMMA for i in range(256))


def _linear_to_srgb_byte(value: float) -> int:
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return 255
    return int(value**_INV_GAMMA * 255.0 + 0.5)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
    if length_sq <= 0.0:
        return (0.0, 0.0, 0.0)
    inv_length = 1.0 / math.sqrt(length_sq)
    return (vector[0] * inv_length, vector[1] * inv_length, vector[2] * inv_length)


def to_hex_color(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def from_hex_color(color: str) -> tuple[int, int, int]:
    cleaned = color.lstrip("#")
    if len(cleaned) != 6:
        return (106, 139, 214)
    return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))


def _lerp_color(
    start: tuple[int, int, int], end: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    blend = clamp(t, 0.0, 1.0)
    return (
        int(start[0] + (end[0] - start[0]) * blend),
        int(start[1] + (end[1] - start[1]) * blend),
        int(start[2] + (end[2] - start[2]) * blend),
    )


def depth_heat_color(depth_t: float) -> str:
    stops = (
        (0.00, (38, 68, 176)),
        (0.32, (54, 182, 235)),
        (0.58, (98, 216, 132)),
        (0.80, (246, 203, 78)),
        (1.00, (220, 67, 44)),
    )
    t = clamp(depth_t, 0.0, 1.0)
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if t <= t1:
            segment = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return to_hex_color(_lerp_color(c0, c1, segment))
    return to_hex_color(stops[-1][1])


def direction_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = math.cos(elevation) * math.cos(azimuth)
    y = math.sin(elevation)
    z = math.cos(elevation) * math.sin(azimuth)
    return normalize((x, y, z))


def _intersect_near_plane(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    near_z: float,
) -> tuple[float, float, float]:
    dz = point_b[2] - point_a[2]
    if abs(dz) <= 1e-12:
        return (point_a[0], point_a[1], near_z)
    t = (near_z - point_a[2]) / dz
    return (
        point_a[0] + (point_b[0] - point_a[0]) * t,
        point_a[1] + (point_b[1] - point_a[1]) * t,
        near_z,
    )


def clip_triangle_near_plane(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    near_z: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    polygon = [p0, p1, p2]
    clipped: list[tuple[float, float, float]] = []

    previous = polygon[-1]
    previous_inside = previous[2] >= near_z

    for current in polygon:
        current_inside = current[2] >= near_z
        if current_inside:
            if not previous_inside:
                clipped.append(_intersect_near_plane(previous, current, near_z))
            clipped.append(current)
        elif previous_inside:
            clipped.append(_intersect_near_plane(previous, current, near_z))

        previous = current
        previous_inside = current_inside

    if len(clipped) < 3:
        return []
    if len(clipped) == 3:
        return [(clipped[0], clipped[1], clipped[2])]

    return [
        (clipped[0], clipped[1], clipped[2]),
        (clipped[0], clipped[2], clipped[3]),
    ]


def _parse_color_channels(parts: list[str]) -> str | None:
    if len(parts) < 4:
        return None
    try:
        raw_r = float(parts[1])
        raw_g = float(parts[2])
        raw_b = float(parts[3])
    except ValueError:
        return None

    channels: list[int] = []
    for value in (raw_r, raw_g, raw_b):
        if value <= 1.0:
            channels.append(int(clamp(value, 0.0, 1.0) * 255))
        else:
            channels.append(int(clamp(value, 0.0, 255.0)))
    return to_hex_color((channels[0], channels[1], channels[2]))


def _load_mtl_colors(base_dir: Path, libraries: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for library_name in libraries:
        mtl_path = (base_dir / library_name).expanduser()
        if not mtl_path.exists() or not mtl_path.is_file():
            continue

        current_material: str | None = None
        try:
            with mtl_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("newmtl "):
                        current_material = line[7:].strip() or None
                        continue
                    if current_material is None:
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0] == "Kd":
                        parsed = _parse_color_channels(parts)
                        if parsed is not None:
                            colors[current_material] = parsed
                    elif parts[0] == "Ka" and current_material not in colors:
                        parsed = _parse_color_channels(parts)
                        if parsed is not None:
                            colors[current_material] = parsed
        except OSError:
            continue
    return colors


@dataclass(frozen=True)
class Triangle:
    i0: int
    i1: int
    i2: int
    color: str


def _corner_angle(
    apex: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    abx = b[0] - apex[0]
    aby = b[1] - apex[1]
    abz = b[2] - apex[2]
    acx = c[0] - apex[0]
    acy = c[1] - apex[1]
    acz = c[2] - apex[2]
    len_ab = math.sqrt(abx * abx + aby * aby + abz * abz)
    len_ac = math.sqrt(acx * acx + acy * acy + acz * acz)
    if len_ab <= 0.0 or len_ac <= 0.0:
        return 0.0
    cos_angle = (abx * acx + aby * acy + abz * acz) / (len_ab * len_ac)
    return math.acos(clamp(cos_angle, -1.0, 1.0))


class OBJModel:
    def __init__(
        self,
        vertices: list[tuple[float, float, float]],
        triangles: list[Triangle],
        corner_normals: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
            | None
        ]
        | None = None,
    ) -> None:
        self.vertices = vertices
        self.triangles = triangles
        self.normalized_vertices = self._normalize_vertices(vertices)
        self.normalized_bounds = self._compute_bounds(self.normalized_vertices)
        self.flat_normals = self._compute_flat_normals(self.normalized_vertices, triangles)
        self.smooth_normals = self._compute_smooth_normals(
            self.normalized_vertices, triangles, self.flat_normals, corner_normals
        )
        self._stats: dict[str, object] | None = None
        self._np_vertices = (
            _np.asarray(self.normalized_vertices, dtype=_np.float64)
            if _np is not None
            else None
        )

    @staticmethod
    def _normalize_vertices(
        vertices: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        zs = [vertex[2] for vertex in vertices]

        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        center_z = (min(zs) + max(zs)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if span == 0:
            span = 1.0
        scale = 2.0 / span

        return [
            (
                (vertex[0] - center_x) * scale,
                (vertex[1] - center_y) * scale,
                (vertex[2] - center_z) * scale,
            )
            for vertex in vertices
        ]

    @staticmethod
    def _compute_bounds(
        vertices: list[tuple[float, float, float]]
    ) -> tuple[float, float, float, float, float, float]:
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        zs = [vertex[2] for vertex in vertices]
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    @staticmethod
    def _compute_flat_normals(
        vertices: list[tuple[float, float, float]],
        triangles: list[Triangle],
    ) -> list[tuple[float, float, float]]:
        normals: list[tuple[float, float, float]] = []
        for tri in triangles:
            p0 = vertices[tri.i0]
            p1 = vertices[tri.i1]
            p2 = vertices[tri.i2]
            ax = p1[0] - p0[0]
            ay = p1[1] - p0[1]
            az = p1[2] - p0[2]
            bx = p2[0] - p0[0]
            by = p2[1] - p0[1]
            bz = p2[2] - p0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            normal_sq = nx * nx + ny * ny + nz * nz
            if normal_sq <= NORMAL_EPS:
                normals.append((0.0, 0.0, 0.0))
                continue
            inv_len = 1.0 / math.sqrt(normal_sq)
            normals.append((nx * inv_len, ny * inv_len, nz * inv_len))
        return normals

    @staticmethod
    def _compute_smooth_normals(
        vertices: list[tuple[float, float, float]],
        triangles: list[Triangle],
        flat_normals: list[tuple[float, float, float]],
        corner_normals: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
            | None
        ]
        | None,
    ) -> list[tuple[float, float, float]]:
        smooth: list[tuple[float, float, float]] = []

        if corner_normals is not None and any(entry is not None for entry in corner_normals):
            # Authored vn data wins: average the three corner normals per triangle.
            for tri_idx, entry in enumerate(corner_normals):
                if entry is None:
                    smooth.append(flat_normals[tri_idx])
                    continue
                n0, n1, n2 = entry
                averaged = normalize(
                    (n0[0] + n1[0] + n2[0], n0[1] + n1[1] + n2[1], n0[2] + n1[2] + n2[2])
                )
                smooth.append(averaged if averaged != (0.0, 0.0, 0.0) else flat_normals[tri_idx])
            return smooth

        # Angle-weighted vertex normals (Thurmer & Wuthrich 1998), then a
        # crease test per corner so hard edges stay hard.
        accum_x = [0.0] * len(vertices)
        accum_y = [0.0] * len(vertices)
        accum_z = [0.0] * len(vertices)
        for tri_idx, tri in enumerate(triangles):
            fn = flat_normals[tri_idx]
            if fn == (0.0, 0.0, 0.0):
                continue
            p0 = vertices[tri.i0]
            p1 = vertices[tri.i1]
            p2 = vertices[tri.i2]
            for corner, pa, pb, pc in (
                (tri.i0, p0, p1, p2),
                (tri.i1, p1, p2, p0),
                (tri.i2, p2, p0, p1),
            ):
                weight = _corner_angle(pa, pb, pc)
                accum_x[corner] += fn[0] * weight
                accum_y[corner] += fn[1] * weight
                accum_z[corner] += fn[2] * weight

        vertex_normals: list[tuple[float, float, float]] = [
            normalize((accum_x[idx], accum_y[idx], accum_z[idx]))
            for idx in range(len(vertices))
        ]

        crease_cos = math.cos(math.radians(CREASE_ANGLE_DEG))
        for tri_idx, tri in enumerate(triangles):
            fn = flat_normals[tri_idx]
            if fn == (0.0, 0.0, 0.0):
                smooth.append(fn)
                continue
            sum_x = sum_y = sum_z = 0.0
            for corner in (tri.i0, tri.i1, tri.i2):
                vn = vertex_normals[corner]
                if vn[0] * fn[0] + vn[1] * fn[1] + vn[2] * fn[2] >= crease_cos:
                    sum_x += vn[0]
                    sum_y += vn[1]
                    sum_z += vn[2]
                else:
                    sum_x += fn[0]
                    sum_y += fn[1]
                    sum_z += fn[2]
            averaged = normalize((sum_x, sum_y, sum_z))
            smooth.append(averaged if averaged != (0.0, 0.0, 0.0) else fn)
        return smooth

    def compute_stats(self) -> dict[str, object]:
        if self._stats is not None:
            return self._stats

        area = 0.0
        volume = 0.0
        edge_counts: dict[tuple[int, int], int] = {}
        for tri in self.triangles:
            p0 = self.vertices[tri.i0]
            p1 = self.vertices[tri.i1]
            p2 = self.vertices[tri.i2]
            ax = p1[0] - p0[0]
            ay = p1[1] - p0[1]
            az = p1[2] - p0[2]
            bx = p2[0] - p0[0]
            by = p2[1] - p0[1]
            bz = p2[2] - p0[2]
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
            # Signed tetrahedron volume (divergence theorem); valid when closed.
            volume += (
                p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
                - p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
                + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
            ) / 6.0
            for ia, ib in ((tri.i0, tri.i1), (tri.i1, tri.i2), (tri.i2, tri.i0)):
                key = (ia, ib) if ia < ib else (ib, ia)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
        nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
        min_x, max_x, min_y, max_y, min_z, max_z = self._compute_bounds(self.vertices)

        self._stats = {
            "vertices": len(self.vertices),
            "triangles": len(self.triangles),
            "size": (max_x - min_x, max_y - min_y, max_z - min_z),
            "surface_area": area,
            "volume": abs(volume),
            "edges": len(edge_counts),
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            "watertight": boundary_edges == 0 and nonmanifold_edges == 0,
        }
        return self._stats

    @classmethod
    def load(cls, filepath: str) -> "OBJModel":
        if Path(filepath).suffix.lower() == ".stl":
            return cls.from_stl(filepath)
        return cls.from_obj(filepath)

    @classmethod
    def from_stl(cls, filepath: str) -> "OBJModel":
        stl_path = Path(filepath)
        if not stl_path.exists():
            raise ValueError("STL file was not found.")

        try:
            raw = stl_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Failed to read STL file: {exc}") from exc

        facets: list[tuple[tuple[float, float, float], ...]] = []
        is_binary = False
        if len(raw) >= 84:
            (declared_count,) = struct.unpack_from("<I", raw, 80)
            if 84 + declared_count * 50 == len(raw):
                is_binary = True

        if is_binary:
            (count,) = struct.unpack_from("<I", raw, 80)
            offset = 84
            for _ in range(count):
                values = struct.unpack_from("<12fH", raw, offset)
                offset += 50
                facets.append(
                    (
                        (values[3], values[4], values[5]),
                        (values[6], values[7], values[8]),
                        (values[9], values[10], values[11]),
                    )
                )
        else:
            current: list[tuple[float, float, float]] = []
            for raw_line in raw.decode("ascii", errors="ignore").splitlines():
                parts = raw_line.split()
                if len(parts) >= 4 and parts[0] == "vertex":
                    try:
                        current.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        continue
                    if len(current) == 3:
                        facets.append((current[0], current[1], current[2]))
                        current = []

        if not facets:
            raise ValueError("No facets were found in this STL file.")

        # Weld duplicated facet vertices so smooth shading has shared indices.
        vertex_index: dict[tuple[float, float, float], int] = {}
        vertices: list[tuple[float, float, float]] = []
        triangles: list[Triangle] = []
        for facet in facets:
            indices: list[int] = []
            for point in facet:
                existing = vertex_index.get(point)
                if existing is None:
                    existing = len(vertices)
                    vertex_index[point] = existing
                    vertices.append(point)
                indices.append(existing)
            if indices[0] != indices[1] and indices[1] != indices[2] and indices[0] != indices[2]:
                triangles.append(
                    Triangle(i0=indices[0], i1=indices[1], i2=indices[2], color=DEFAULT_COLOR)
                )

        if not triangles:
            raise ValueError("No valid triangles were generated from this STL file.")

        return cls(vertices=vertices, triangles=triangles)

    @classmethod
    def from_obj(cls, filepath: str) -> "OBJModel":
        obj_path = Path(filepath)
        if not obj_path.exists():
            raise ValueError("OBJ file was not found.")

        vertices: list[tuple[float, float, float]] = []
        vertex_colors: list[tuple[float, float, float] | None] = []
        normals: list[tuple[float, float, float]] = []
        polygons: list[
            tuple[tuple[int, ...], tuple[int | None, ...], str | None]
        ] = []
        mtl_libraries: list[str] = []
        current_material: str | None = None

        try:
            with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.startswith("v "):
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                        except ValueError:
                            continue
                        vertices.append((x, y, z))
                        # Non-standard but common extension: v x y z r g b
                        color: tuple[float, float, float] | None = None
                        if len(parts) >= 7:
                            try:
                                color = (
                                    clamp(float(parts[4]), 0.0, 1.0),
                                    clamp(float(parts[5]), 0.0, 1.0),
                                    clamp(float(parts[6]), 0.0, 1.0),
                                )
                            except ValueError:
                                color = None
                        vertex_colors.append(color)
                        continue

                    if line.startswith("vn "):
                        parts = line.split()
                        if len(parts) < 4:
                            continue
                        try:
                            normals.append(
                                normalize((float(parts[1]), float(parts[2]), float(parts[3])))
                            )
                        except ValueError:
                            continue
                        continue

                    if line.startswith("mtllib "):
                        raw_library = line[7:].strip()
                        if raw_library:
                            full_library = (obj_path.parent / raw_library).expanduser()
                            if full_library.exists():
                                mtl_libraries.append(raw_library)
                            else:
                                mtl_libraries.extend(raw_library.split())
                        continue

                    if line.startswith("usemtl "):
                        material_name = line[7:].strip()
                        current_material = material_name if material_name else None
                        continue

                    if line.startswith("f "):
                        tokens = line.split()[1:]
                        if len(tokens) < 3:
                            continue

                        indices: list[int] = []
                        normal_indices: list[int | None] = []
                        vertex_total = len(vertices)
                        normal_total = len(normals)
                        for token in tokens:
                            pieces = token.split("/")
                            vertex_token = pieces[0]
                            if not vertex_token:
                                continue
                            try:
                                raw_index = int(vertex_token)
                            except ValueError:
                                continue

                            if raw_index < 0:
                                resolved_index = vertex_total + raw_index
                            else:
                                resolved_index = raw_index - 1
                            indices.append(resolved_index)

                            normal_index: int | None = None
                            if len(pieces) >= 3 and pieces[2]:
                                try:
                                    raw_normal = int(pieces[2])
                                except ValueError:
                                    raw_normal = 0
                                if raw_normal < 0:
                                    normal_index = normal_total + raw_normal
                                elif raw_normal > 0:
                                    normal_index = raw_normal - 1
                                if normal_index is not None and not (
                                    0 <= normal_index < normal_total
                                ):
                                    normal_index = None
                            normal_indices.append(normal_index)

                        if len(indices) >= 3:
                            polygons.append(
                                (tuple(indices), tuple(normal_indices), current_material)
                            )
        except OSError as exc:
            raise ValueError(f"Failed to read OBJ file: {exc}") from exc

        if not vertices:
            raise ValueError("No vertices were found in this OBJ file.")
        if not polygons:
            raise ValueError("No faces were found in this OBJ file.")

        material_colors = _load_mtl_colors(obj_path.parent, mtl_libraries)
        vertex_count = len(vertices)

        triangles: list[Triangle] = []
        corner_normals: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
            | None
        ] = []

        for polygon, polygon_normals, material_name in polygons:
            material_color = material_colors.get(material_name or "")
            i0 = polygon[0]
            n0 = polygon_normals[0]
            for index in range(1, len(polygon) - 1):
                i1 = polygon[index]
                i2 = polygon[index + 1]
                if (
                    i0 < 0
                    or i1 < 0
                    or i2 < 0
                    or i0 >= vertex_count
                    or i1 >= vertex_count
                    or i2 >= vertex_count
                ):
                    continue

                color = material_color
                if color is None:
                    c0 = vertex_colors[i0]
                    c1 = vertex_colors[i1]
                    c2 = vertex_colors[i2]
                    if c0 is not None and c1 is not None and c2 is not None:
                        color = to_hex_color(
                            (
                                int((c0[0] + c1[0] + c2[0]) / 3.0 * 255),
                                int((c0[1] + c1[1] + c2[1]) / 3.0 * 255),
                                int((c0[2] + c1[2] + c2[2]) / 3.0 * 255),
                            )
                        )
                if color is None:
                    color = DEFAULT_COLOR

                triangles.append(Triangle(i0=i0, i1=i1, i2=i2, color=color))

                n1 = polygon_normals[index]
                n2 = polygon_normals[index + 1]
                if n0 is not None and n1 is not None and n2 is not None:
                    corner_normals.append((normals[n0], normals[n1], normals[n2]))
                else:
                    corner_normals.append(None)

        if not triangles:
            raise ValueError("No valid triangles were generated from this OBJ file.")

        return cls(vertices=vertices, triangles=triangles, corner_normals=corner_normals)


class ModelViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"PyOBJViewer {__version__} - Advanced 3D Model Viewer")
        self.geometry("1220x820")
        self.minsize(860, 600)

        self.model: OBJModel | None = None
        self.current_path: Path | None = None

        self.rotation_x = math.radians(-12.0)
        self.rotation_y = math.radians(20.0)
        self.rotation_z = 0.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.base_scale = 260.0
        self.camera_distance = 3.8
        self.focal_length = 1.65

        self._drag_left: tuple[int, int] | None = None
        self._drag_right: tuple[int, int] | None = None
        self._render_pending = False
        self._interaction_deadline = 0.0
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0
        self._inertia_x = 0.0
        self._inertia_y = 0.0
        self._last_drag_event = 0.0
        self._last_render_capture: dict[str, object] | None = None

        initial_light = normalize(LIGHT_DIRECTION)
        initial_azimuth = math.degrees(math.atan2(initial_light[2], initial_light[0]))
        initial_elevation = math.degrees(math.asin(clamp(initial_light[1], -1.0, 1.0)))
        self._light_dir = initial_light
        self._linear_cache: dict[str, tuple[float, float, float]] = {}
        self._phong_cache: dict[tuple[str, int, int], str] = {}
        self._heat_cache: dict[int, str] = {}

        self.fill_var = tk.BooleanVar(value=True)
        self.wireframe_var = tk.BooleanVar(value=True)
        self.shading_var = tk.BooleanVar(value=True)
        self.smooth_var = tk.BooleanVar(value=True)
        self.shadow_var = tk.BooleanVar(value=True)
        self.culling_var = tk.BooleanVar(value=False)
        self.invert_culling_var = tk.BooleanVar(value=False)
        self.depth_sort_var = tk.BooleanVar(value=True)
        self.ortho_var = tk.BooleanVar(value=False)
        self.auto_spin_var = tk.BooleanVar(value=False)
        self.show_axes_var = tk.BooleanVar(value=True)
        self.show_bounds_var = tk.BooleanVar(value=False)
        self.show_normals_var = tk.BooleanVar(value=False)
        self.dark_bg_var = tk.BooleanVar(value=False)
        self.fullscreen_var = tk.BooleanVar(value=False)

        self.quality_var = tk.StringVar(value="Auto")
        self.color_mode_var = tk.StringVar(value="Material")
        self.ambient_var = tk.DoubleVar(value=0.18)
        self.diffuse_var = tk.DoubleVar(value=0.85)
        self.ambient_value_var = tk.StringVar(value=f"{self.ambient_var.get():.2f}")
        self.diffuse_value_var = tk.StringVar(value=f"{self.diffuse_var.get():.2f}")
        self.specular_var = tk.DoubleVar(value=0.22)
        self.gloss_var = tk.DoubleVar(value=32.0)
        self.specular_value_var = tk.StringVar(value=f"{self.specular_var.get():.2f}")
        self.gloss_value_var = tk.StringVar(value=f"{self.gloss_var.get():.0f}")
        self.near_clip_var = tk.DoubleVar(value=NEAR_CLIP)
        self.near_clip_value_var = tk.StringVar(value=f"{self.near_clip_var.get():.2f}")
        self.normal_len_var = tk.DoubleVar(value=0.09)
        self.normal_len_value_var = tk.StringVar(value=f"{self.normal_len_var.get():.2f}")
        self.wire_width_var = tk.DoubleVar(value=1.0)
        self.wire_width_value_var = tk.StringVar(value=f"{self.wire_width_var.get():.1f}px")
        self.light_azimuth_var = tk.DoubleVar(value=initial_azimuth)
        self.light_elevation_var = tk.DoubleVar(value=initial_elevation)
        self.light_azimuth_value_var = tk.StringVar(value=f"{self.light_azimuth_var.get():.0f}\u00b0")
        self.light_elevation_value_var = tk.StringVar(value=f"{self.light_elevation_var.get():.0f}\u00b0")
        self.spin_speed_var = tk.DoubleVar(value=1.0)
        self.spin_speed_value_var = tk.StringVar(value=f"{self.spin_speed_var.get():.2f}x")

        self.status_var = tk.StringVar(value="Load an OBJ or STL file to begin.")
        self._overlay_color = "#2c3642"
        self._hint_color = "#49596c"
        self._shadow_color = "#d3dae4"

        self._on_light_direction_change(render=False)
        self._on_wire_width_change(render=False)
        self._on_spin_speed_change(render=False)
        self._on_render_tuning_change(render=False)
        self._configure_theme()
        self._build_ui()
        self._bind_events()
        self._apply_background_theme()
        self.after(16, self._animation_tick)
        self.request_render()

    def _configure_theme(self) -> None:
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        self.configure(bg="#e6ecf4")
        self.style.configure("App.TFrame", background="#e6ecf4")
        self.style.configure("Controls.TFrame", background="#e6ecf4")
        self.style.configure("Card.TFrame", background="#f9fbff")
        self.style.configure("Status.TFrame", background="#dae2ee")

        self.style.configure(
            "TLabel",
            background="#e6ecf4",
            foreground="#293341",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Heading.TLabel",
            background="#e6ecf4",
            foreground="#1f2937",
            font=("Segoe UI Semibold", 10),
        )
        self.style.configure(
            "Status.TLabel",
            background="#dae2ee",
            foreground="#223042",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "TButton",
            padding=(11, 6),
            font=("Segoe UI Semibold", 10),
            relief="flat",
        )
        self.style.map(
            "TButton",
            background=[("active", "#d9e4f5"), ("pressed", "#c5d6ef")],
        )
        self.style.configure(
            "Accent.TButton",
            padding=(13, 6),
            foreground="white",
            background="#2f6fda",
            bordercolor="#2f6fda",
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", "#275fc0"), ("pressed", "#204f9f")],
            foreground=[("disabled", "#e6ecf4")],
        )
        self.style.configure(
            "TCheckbutton",
            background="#e6ecf4",
            foreground="#2a3645",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            arrowsize=14,
            padding=4,
        )
        self.style.configure("Horizontal.TScale", background="#e6ecf4")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, style="App.TFrame", padding=(12, 10, 12, 10))
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, style="Controls.TFrame")
        controls.pack(fill="x")

        buttons_row = ttk.Frame(controls, style="Controls.TFrame")
        buttons_row.pack(fill="x", pady=(0, 6))

        ttk.Button(
            buttons_row, text="Open OBJ", style="Accent.TButton", command=self.open_model
        ).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Reload", command=self.reload_model).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Fit View", command=self.fit_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Reset View", command=self.reset_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Front", command=self.set_front_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Back", command=self.set_back_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Left", command=self.set_left_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Top", command=self.set_top_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Right", command=self.set_right_view).pack(side="left", padx=(0, 6))
        ttk.Button(buttons_row, text="Save Snapshot", command=self.save_snapshot).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(buttons_row, text="Model Info", command=self.show_model_info).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(buttons_row, text="Fullscreen", command=self.toggle_fullscreen).pack(
            side="left", padx=(0, 10)
        )
        ttk.Checkbutton(
            buttons_row,
            text="Dark Background",
            variable=self.dark_bg_var,
            command=self._apply_background_theme,
        ).pack(side="left", padx=(6, 0))

        toggles_row = ttk.Frame(controls, style="Controls.TFrame")
        toggles_row.pack(fill="x", pady=(0, 6))

        for label, variable in (
            ("Fill", self.fill_var),
            ("Wireframe", self.wireframe_var),
            ("Shading", self.shading_var),
            ("Smooth", self.smooth_var),
            ("Shadow", self.shadow_var),
            ("Backface Culling", self.culling_var),
            ("Invert Culling", self.invert_culling_var),
            ("Depth Sort", self.depth_sort_var),
            ("Orthographic", self.ortho_var),
            ("Auto Spin", self.auto_spin_var),
            ("Axes", self.show_axes_var),
            ("Bounds", self.show_bounds_var),
            ("Normals", self.show_normals_var),
        ):
            ttk.Checkbutton(
                toggles_row,
                text=label,
                variable=variable,
                command=self.request_render,
            ).pack(side="left", padx=(0, 8))

        tuning_row = ttk.Frame(controls, style="Controls.TFrame")
        tuning_row.pack(fill="x")

        ttk.Label(tuning_row, text="Quality", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        self.quality_combo = ttk.Combobox(
            tuning_row,
            textvariable=self.quality_var,
            values=("Auto", "High", "Balanced", "Fast"),
            state="readonly",
            width=11,
        )
        self.quality_combo.pack(side="left", padx=(0, 18))
        self.quality_combo.bind("<<ComboboxSelected>>", lambda _event: self.request_render())

        ttk.Label(tuning_row, text="Mode", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        self.color_mode_combo = ttk.Combobox(
            tuning_row,
            textvariable=self.color_mode_var,
            values=("Material", "Depth Heatmap", "Normal Tint"),
            state="readonly",
            width=14,
        )
        self.color_mode_combo.pack(side="left", padx=(0, 18))
        self.color_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.request_render())

        ttk.Label(tuning_row, text="Ambient", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        ambient_scale = ttk.Scale(
            tuning_row,
            from_=0.05,
            to=0.90,
            variable=self.ambient_var,
            command=self._on_lighting_change,
            length=150,
        )
        ambient_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            tuning_row,
            textvariable=self.ambient_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left", padx=(0, 16))

        ttk.Label(tuning_row, text="Diffuse", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        diffuse_scale = ttk.Scale(
            tuning_row,
            from_=0.20,
            to=1.30,
            variable=self.diffuse_var,
            command=self._on_lighting_change,
            length=150,
        )
        diffuse_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            tuning_row,
            textvariable=self.diffuse_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left")

        ttk.Label(tuning_row, text="Wire", style="Heading.TLabel").pack(side="left", padx=(16, 5))
        wire_scale = ttk.Scale(
            tuning_row,
            from_=0.5,
            to=3.5,
            variable=self.wire_width_var,
            command=self._on_wire_width_change,
            length=120,
        )
        wire_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            tuning_row,
            textvariable=self.wire_width_value_var,
            width=5,
            style="Heading.TLabel",
        ).pack(side="left")

        light_row = ttk.Frame(controls, style="Controls.TFrame")
        light_row.pack(fill="x", pady=(6, 0))

        ttk.Label(light_row, text="Light Az", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        az_scale = ttk.Scale(
            light_row,
            from_=-180.0,
            to=180.0,
            variable=self.light_azimuth_var,
            command=self._on_light_direction_change,
            length=170,
        )
        az_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            light_row,
            textvariable=self.light_azimuth_value_var,
            width=6,
            style="Heading.TLabel",
        ).pack(side="left", padx=(0, 14))

        ttk.Label(light_row, text="Light El", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        el_scale = ttk.Scale(
            light_row,
            from_=-80.0,
            to=80.0,
            variable=self.light_elevation_var,
            command=self._on_light_direction_change,
            length=170,
        )
        el_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            light_row,
            textvariable=self.light_elevation_value_var,
            width=6,
            style="Heading.TLabel",
        ).pack(side="left")

        ttk.Label(light_row, text="Spin", style="Heading.TLabel").pack(side="left", padx=(16, 5))
        spin_scale = ttk.Scale(
            light_row,
            from_=0.10,
            to=3.00,
            variable=self.spin_speed_var,
            command=self._on_spin_speed_change,
            length=150,
        )
        spin_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            light_row,
            textvariable=self.spin_speed_value_var,
            width=5,
            style="Heading.TLabel",
        ).pack(side="left")

        render_row = ttk.Frame(controls, style="Controls.TFrame")
        render_row.pack(fill="x", pady=(6, 0))

        ttk.Label(render_row, text="Near", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        near_scale = ttk.Scale(
            render_row,
            from_=0.02,
            to=0.70,
            variable=self.near_clip_var,
            command=self._on_render_tuning_change,
            length=145,
        )
        near_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            render_row,
            textvariable=self.near_clip_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left", padx=(0, 16))

        ttk.Label(render_row, text="Spec", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        spec_scale = ttk.Scale(
            render_row,
            from_=0.00,
            to=1.20,
            variable=self.specular_var,
            command=self._on_render_tuning_change,
            length=145,
        )
        spec_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            render_row,
            textvariable=self.specular_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left", padx=(0, 16))

        ttk.Label(render_row, text="Gloss", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        gloss_scale = ttk.Scale(
            render_row,
            from_=4.0,
            to=96.0,
            variable=self.gloss_var,
            command=self._on_render_tuning_change,
            length=145,
        )
        gloss_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            render_row,
            textvariable=self.gloss_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left", padx=(0, 16))

        ttk.Label(render_row, text="N Len", style="Heading.TLabel").pack(side="left", padx=(0, 5))
        normal_len_scale = ttk.Scale(
            render_row,
            from_=0.02,
            to=0.24,
            variable=self.normal_len_var,
            command=self._on_render_tuning_change,
            length=130,
        )
        normal_len_scale.pack(side="left", padx=(0, 4))
        ttk.Label(
            render_row,
            textvariable=self.normal_len_value_var,
            width=4,
            style="Heading.TLabel",
        ).pack(side="left")

        help_text = (
            "Controls: left-drag rotate (release to glide), Shift+left-drag roll, right-drag pan, wheel zooms to cursor, "
            "R reset, F fit, 1/2/3 front-top-right, 4 back, 5 left, W wireframe, S shading, T smooth, H shadow, C culling, "
            "I invert culling, D depth sort, O orthographic, M render mode, N normals, A axes, G bounds, B bg, Space spin, "
            "F1 model info, F5 reload, F11 fullscreen"
        )
        ttk.Label(
            outer,
            text=help_text,
            style="TLabel",
            anchor="w",
        ).pack(fill="x", pady=(8, 8))

        viewport = ttk.Frame(outer, style="Card.TFrame", padding=5)
        viewport.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            viewport,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.canvas.pack(fill="both", expand=True)

        status_frame = ttk.Frame(outer, style="Status.TFrame", padding=(8, 6))
        status_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
        ).pack(fill="x")

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)

        self.canvas.bind("<ButtonPress-2>", self._on_right_press)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_right_release)

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", lambda _event: self.request_render())

        self.bind("<KeyPress-r>", lambda _event: self.reset_view())
        self.bind("<KeyPress-f>", lambda _event: self.fit_view())
        self.bind("<KeyPress-w>", lambda _event: self._toggle_and_render(self.wireframe_var))
        self.bind("<KeyPress-s>", lambda _event: self._toggle_and_render(self.shading_var))
        self.bind("<KeyPress-t>", lambda _event: self._toggle_and_render(self.smooth_var))
        self.bind("<KeyPress-h>", lambda _event: self._toggle_and_render(self.shadow_var))
        self.bind("<KeyPress-c>", lambda _event: self._toggle_and_render(self.culling_var))
        self.bind("<KeyPress-i>", lambda _event: self._toggle_and_render(self.invert_culling_var))
        self.bind("<KeyPress-d>", lambda _event: self._toggle_and_render(self.depth_sort_var))
        self.bind("<KeyPress-o>", lambda _event: self._toggle_and_render(self.ortho_var))
        self.bind("<KeyPress-m>", lambda _event: self._cycle_color_mode())
        self.bind("<KeyPress-n>", lambda _event: self._toggle_and_render(self.show_normals_var))
        self.bind("<KeyPress-a>", lambda _event: self._toggle_and_render(self.show_axes_var))
        self.bind("<KeyPress-g>", lambda _event: self._toggle_and_render(self.show_bounds_var))
        self.bind("<KeyPress-b>", lambda _event: self._toggle_background())
        self.bind("<KeyPress-1>", lambda _event: self.set_front_view())
        self.bind("<KeyPress-2>", lambda _event: self.set_top_view())
        self.bind("<KeyPress-3>", lambda _event: self.set_right_view())
        self.bind("<KeyPress-4>", lambda _event: self.set_back_view())
        self.bind("<KeyPress-5>", lambda _event: self.set_left_view())
        self.bind("<space>", lambda _event: self._toggle_auto_spin())
        self.bind("<F1>", lambda _event: self.show_model_info())
        self.bind("<F5>", lambda _event: self.reload_model())
        self.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.bind("<Escape>", lambda _event: self._exit_fullscreen())
        self.focus_set()

    def _on_lighting_change(self, _value: str | None = None) -> None:
        self.ambient_value_var.set(f"{self.ambient_var.get():.2f}")
        self.diffuse_value_var.set(f"{self.diffuse_var.get():.2f}")
        self.request_render()

    def _on_wire_width_change(
        self, _value: str | None = None, render: bool = True
    ) -> None:
        self.wire_width_value_var.set(f"{self.wire_width_var.get():.1f}px")
        if render:
            self.request_render()

    def _on_light_direction_change(
        self, _value: str | None = None, render: bool = True
    ) -> None:
        azimuth = self.light_azimuth_var.get()
        elevation = self.light_elevation_var.get()
        self.light_azimuth_value_var.set(f"{azimuth:.0f}\u00b0")
        self.light_elevation_value_var.set(f"{elevation:.0f}\u00b0")
        self._light_dir = direction_from_angles(azimuth, elevation)
        if render:
            self.request_render()

    def _on_spin_speed_change(
        self, _value: str | None = None, render: bool = True
    ) -> None:
        self.spin_speed_value_var.set(f"{self.spin_speed_var.get():.2f}x")
        if render:
            self.request_render()

    def _on_render_tuning_change(
        self, _value: str | None = None, render: bool = True
    ) -> None:
        self.near_clip_value_var.set(f"{self.near_clip_var.get():.2f}")
        self.specular_value_var.set(f"{self.specular_var.get():.2f}")
        self.gloss_value_var.set(f"{self.gloss_var.get():.0f}")
        self.normal_len_value_var.set(f"{self.normal_len_var.get():.2f}")
        if render:
            self.request_render()

    def _cycle_color_mode(self) -> None:
        modes = ("Material", "Depth Heatmap", "Normal Tint")
        try:
            current_index = modes.index(self.color_mode_var.get())
        except ValueError:
            current_index = 0
        self.color_mode_var.set(modes[(current_index + 1) % len(modes)])
        self.request_render()

    def _toggle_and_render(self, variable: tk.BooleanVar) -> None:
        variable.set(not variable.get())
        self.request_render()

    def _toggle_auto_spin(self) -> None:
        self.auto_spin_var.set(not self.auto_spin_var.get())
        self.request_render()

    def _toggle_background(self) -> None:
        self.dark_bg_var.set(not self.dark_bg_var.get())
        self._apply_background_theme()

    def toggle_fullscreen(self) -> None:
        self.fullscreen_var.set(not self.fullscreen_var.get())
        self.attributes("-fullscreen", self.fullscreen_var.get())

    def _exit_fullscreen(self) -> None:
        self.fullscreen_var.set(False)
        self.attributes("-fullscreen", False)

    def _apply_background_theme(self) -> None:
        if self.dark_bg_var.get():
            self.canvas.configure(bg="#111a24")
            self._overlay_color = "#d7e2ee"
            self._hint_color = "#8aa0b7"
            self._shadow_color = "#0a1018"
        else:
            self.canvas.configure(bg="#f7f9fc")
            self._overlay_color = "#2d3b4d"
            self._hint_color = "#4f6178"
            self._shadow_color = "#d3dae4"
        self.request_render()

    def _on_left_press(self, event: tk.Event) -> None:
        self._drag_left = (event.x, event.y)
        self._inertia_x = 0.0
        self._inertia_y = 0.0

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._drag_left is None:
            return
        last_x, last_y = self._drag_left
        dx = event.x - last_x
        dy = event.y - last_y

        if event.state & 0x0001:
            self.rotation_z += dx * 0.01
        else:
            self.rotation_y += dx * 0.01
            self.rotation_x += dy * 0.01
            self._inertia_x = dy * 0.01
            self._inertia_y = dx * 0.01
            self._last_drag_event = time.perf_counter()

        self._drag_left = (event.x, event.y)
        self._mark_interacting()
        self.request_render()

    def _on_left_release(self, _event: tk.Event) -> None:
        self._drag_left = None
        # Only keep gliding when the pointer was still moving at release time.
        if time.perf_counter() - self._last_drag_event > 0.06:
            self._inertia_x = 0.0
            self._inertia_y = 0.0

    def _on_right_press(self, event: tk.Event) -> None:
        self._drag_right = (event.x, event.y)

    def _on_right_drag(self, event: tk.Event) -> None:
        if self._drag_right is None:
            return
        last_x, last_y = self._drag_right
        dx = event.x - last_x
        dy = event.y - last_y
        self.pan_x += dx
        self.pan_y += dy
        self._drag_right = (event.x, event.y)
        self._mark_interacting()
        self.request_render()

    def _on_right_release(self, _event: tk.Event) -> None:
        self._drag_right = None

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        wheel_up = False
        if hasattr(event, "delta") and event.delta != 0:
            wheel_up = event.delta > 0
        elif hasattr(event, "num"):
            wheel_up = event.num == 4

        new_zoom = clamp(self.zoom * (1.10 if wheel_up else 0.91), 0.2, 9.0)
        applied = new_zoom / self.zoom
        if applied != 1.0:
            # Keep the point under the cursor fixed while zooming.
            half_w = self.canvas.winfo_width() * 0.5
            half_h = self.canvas.winfo_height() * 0.5
            self.pan_x = event.x - half_w - applied * (event.x - half_w - self.pan_x)
            self.pan_y = event.y - half_h - applied * (event.y - half_h - self.pan_y)
        self.zoom = new_zoom
        self._mark_interacting()
        self.request_render()

    def _draw_empty_state(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
            width / 2,
            height / 2,
            text="Open an OBJ or STL file to preview the model",
            fill=self._hint_color,
            font=("Segoe UI Semibold", 14),
        )

    def _animation_tick(self) -> None:
        if self.auto_spin_var.get() and self.model is not None:
            self.rotation_y += 0.013 * self.spin_speed_var.get()
            self.request_render()
        if (
            self._drag_left is None
            and self.model is not None
            and (abs(self._inertia_x) > 4e-4 or abs(self._inertia_y) > 4e-4)
        ):
            self.rotation_x += self._inertia_x
            self.rotation_y += self._inertia_y
            self._inertia_x *= 0.92
            self._inertia_y *= 0.92
            self._mark_interacting(0.05)
            self.request_render()
        self.after(16, self._animation_tick)

    def _mark_interacting(self, duration: float = 0.16) -> None:
        self._interaction_deadline = time.perf_counter() + duration

    def _is_interacting(self) -> bool:
        return time.perf_counter() < self._interaction_deadline

    def _shade_color_phong(
        self, base_color: str, diffuse_bucket: int, specular_bucket: int
    ) -> str:
        diffuse_bucket = max(0, min(255, diffuse_bucket))
        specular_bucket = max(0, min(255, specular_bucket))
        key = (base_color, diffuse_bucket, specular_bucket)
        cached = self._phong_cache.get(key)
        if cached is not None:
            return cached

        linear = self._linear_cache.get(base_color)
        if linear is None:
            rgb = from_hex_color(base_color)
            linear = (
                _SRGB_TO_LINEAR[rgb[0]],
                _SRGB_TO_LINEAR[rgb[1]],
                _SRGB_TO_LINEAR[rgb[2]],
            )
            self._linear_cache[base_color] = linear

        # Light in linear space, encode back to sRGB; specular adds white light.
        diffuse_factor = diffuse_bucket / 255.0
        specular_factor = specular_bucket / 255.0
        shaded = (
            _linear_to_srgb_byte(linear[0] * diffuse_factor + specular_factor),
            _linear_to_srgb_byte(linear[1] * diffuse_factor + specular_factor),
            _linear_to_srgb_byte(linear[2] * diffuse_factor + specular_factor),
        )
        result = to_hex_color(shaded)
        self._phong_cache[key] = result
        return result

    def _depth_color_bucket(self, bucket: int) -> str:
        bucket = max(0, min(255, bucket))
        cached = self._heat_cache.get(bucket)
        if cached is not None:
            return cached
        color = depth_heat_color(bucket / 255.0)
        self._heat_cache[bucket] = color
        return color

    def _triangle_stride(self, total_triangles: int, interacting: bool) -> int:
        mode = self.quality_var.get()
        if mode == "High":
            budget = 24000
        elif mode == "Balanced":
            budget = 12000
        elif mode == "Fast":
            budget = 4500
        else:
            budget = 5500 if interacting else 18000

        if total_triangles <= budget:
            return 1
        return max(1, (total_triangles + budget - 1) // budget)

    def _draw_axis_gizmo(
        self,
        width: int,
        cos_x: float,
        sin_x: float,
        cos_y: float,
        sin_y: float,
        cos_z: float,
        sin_z: float,
    ) -> None:
        if not self.show_axes_var.get():
            return

        origin_x = width - 74
        origin_y = 76
        axis_len = 42

        def rotate_vector(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
            y1 = vy * cos_x - vz * sin_x
            z1 = vy * sin_x + vz * cos_x
            x2 = vx * cos_y + z1 * sin_y
            z2 = -vx * sin_y + z1 * cos_y
            x3 = x2 * cos_z - y1 * sin_z
            y3 = x2 * sin_z + y1 * cos_z
            return (x3, y3, z2)

        for axis_vector, axis_color, label in (
            ((1.0, 0.0, 0.0), "#d04444", "X"),
            ((0.0, 1.0, 0.0), "#2f9e5f", "Y"),
            ((0.0, 0.0, 1.0), "#2d6fd3", "Z"),
        ):
            rx, ry, _rz = rotate_vector(*axis_vector)
            x2 = origin_x + rx * axis_len
            y2 = origin_y - ry * axis_len
            self.canvas.create_line(origin_x, origin_y, x2, y2, fill=axis_color, width=2)
            self.canvas.create_text(
                x2 + (6 if rx >= 0 else -6),
                y2 + (6 if ry < 0 else -6),
                text=label,
                fill=axis_color,
                font=("Segoe UI Semibold", 9),
            )

        self.canvas.create_oval(
            origin_x - 3,
            origin_y - 3,
            origin_x + 3,
            origin_y + 3,
            fill=self._overlay_color,
            outline="",
        )

    def _draw_bounds_overlay(
        self,
        width: int,
        height: int,
        half_w: float,
        half_h: float,
        scale: float,
        cos_x: float,
        sin_x: float,
        cos_y: float,
        sin_y: float,
        cos_z: float,
        sin_z: float,
        use_ortho: bool,
        near_clip: float,
    ) -> None:
        if not self.show_bounds_var.get() or self.model is None:
            return

        min_x, max_x, min_y, max_y, min_z, max_z = self.model.normalized_bounds
        corners = [
            (min_x, min_y, min_z),
            (max_x, min_y, min_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (min_x, min_y, max_z),
            (max_x, min_y, max_z),
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
        ]
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        cam_corners: list[tuple[float, float, float]] = []
        for vx, vy, vz in corners:
            y1 = vy * cos_x - vz * sin_x
            z1 = vy * sin_x + vz * cos_x
            x2 = vx * cos_y + z1 * sin_y
            z2 = -vx * sin_y + z1 * cos_y
            x3 = x2 * cos_z - y1 * sin_z
            y3 = x2 * sin_z + y1 * cos_z
            cam_corners.append((x3, y3, z2 + self.camera_distance))

        def project_point(point: tuple[float, float, float]) -> tuple[float, float]:
            if use_ortho:
                return (
                    point[0] * scale + half_w + self.pan_x,
                    -point[1] * scale + half_h + self.pan_y,
                )
            perspective = self.focal_length / point[2]
            return (
                point[0] * scale * perspective + half_w + self.pan_x,
                -point[1] * scale * perspective + half_h + self.pan_y,
            )

        bounds_color = "#3d79c9" if not self.dark_bg_var.get() else "#86b6ff"
        for i0, i1 in edges:
            p0 = cam_corners[i0]
            p1 = cam_corners[i1]
            if use_ortho:
                x0, y0 = project_point(p0)
                x1, y1 = project_point(p1)
            else:
                if p0[2] <= near_clip and p1[2] <= near_clip:
                    continue
                if p0[2] <= near_clip:
                    p0 = _intersect_near_plane(p0, p1, near_clip)
                elif p1[2] <= near_clip:
                    p1 = _intersect_near_plane(p1, p0, near_clip)
                x0, y0 = project_point(p0)
                x1, y1 = project_point(p1)

            if (
                max(x0, x1) < -8
                or max(y0, y1) < -8
                or min(x0, x1) > width + 8
                or min(y0, y1) > height + 8
            ):
                continue
            self.canvas.create_line(x0, y0, x1, y1, fill=bounds_color, width=1, dash=(4, 3))

    def request_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self.after_idle(self._render_now)

    def _load_model_from_path(self, filepath: str) -> bool:
        try:
            loaded = OBJModel.load(filepath)
        except Exception as exc:
            messagebox.showerror("Load error", f"Unable to open model:\n{exc}")
            return False

        self.model = loaded
        self.current_path = Path(filepath)
        self._phong_cache.clear()
        self._heat_cache.clear()
        self._linear_cache.clear()

        self.reset_view(render=False)
        self.fit_view(render=False)
        self.request_render()
        return True

    def open_model(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Open 3D model",
            filetypes=[
                ("3D models", "*.obj *.stl"),
                ("OBJ files", "*.obj"),
                ("STL files", "*.stl"),
                ("All files", "*.*"),
            ],
        )
        if not filepath:
            return

        self._load_model_from_path(filepath)

    def reload_model(self) -> None:
        if self.current_path is None:
            messagebox.showinfo("No model", "Load a model before reloading.")
            return

        reloaded = self._load_model_from_path(str(self.current_path))
        if reloaded:
            self.status_var.set(f"Reloaded: {self.current_path.name}")

    def fit_view(self, render: bool = True) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.base_scale = 0.56 * min(width, height)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if render:
            self.request_render()

    def reset_view(self, render: bool = True) -> None:
        self.rotation_x = math.radians(-12.0)
        self.rotation_y = math.radians(20.0)
        self.rotation_z = 0.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        if render:
            self.request_render()

    def set_front_view(self) -> None:
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.request_render()

    def set_back_view(self) -> None:
        self.rotation_x = 0.0
        self.rotation_y = math.radians(180.0)
        self.rotation_z = 0.0
        self.request_render()

    def set_left_view(self) -> None:
        self.rotation_x = 0.0
        self.rotation_y = math.radians(90.0)
        self.rotation_z = 0.0
        self.request_render()

    def set_top_view(self) -> None:
        self.rotation_x = math.radians(-90.0)
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.request_render()

    def set_right_view(self) -> None:
        self.rotation_x = 0.0
        self.rotation_y = math.radians(-90.0)
        self.rotation_z = 0.0
        self.request_render()

    def save_snapshot(self) -> None:
        if self.model is None:
            messagebox.showinfo("No model", "Load a model before saving a snapshot.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save snapshot",
            defaultextension=".svg",
            filetypes=[
                ("SVG vector image", "*.svg"),
                ("PNG image (needs Pillow)", "*.png"),
                ("PostScript", "*.ps"),
                ("All files", "*.*"),
            ],
        )
        if not output_path:
            return

        suffix = Path(output_path).suffix.lower()
        try:
            if suffix == ".svg":
                self._write_svg_snapshot(output_path)
            elif suffix == ".png":
                self._write_png_snapshot(output_path)
            else:
                self.canvas.postscript(file=output_path, colormode="color")
        except Exception as exc:
            messagebox.showerror("Snapshot error", f"Unable to save snapshot:\n{exc}")
            return
        self.status_var.set(f"Snapshot saved: {Path(output_path).name}")

    def _write_svg_snapshot(self, output_path: str) -> None:
        capture = self._last_render_capture
        if capture is None:
            raise ValueError("Nothing rendered yet.")

        width = capture["width"]
        height = capture["height"]
        outline = capture["outline"]
        wire_width = capture["wire_width"]
        stroke = (
            f' stroke="{outline}" stroke-width="{wire_width}" stroke-linejoin="round"'
            if outline
            else ""
        )

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{capture["bg"]}"/>',
        ]
        shadow_polys, shadow_color = capture["shadow"]
        for coords in shadow_polys:
            points = " ".join(
                f"{coords[i]:.2f},{coords[i + 1]:.2f}" for i in range(0, len(coords), 2)
            )
            parts.append(f'<polygon points="{points}" fill="{shadow_color}"/>')
        for coords, fill_color in capture["polys"]:
            points = " ".join(
                f"{coords[i]:.2f},{coords[i + 1]:.2f}" for i in range(0, len(coords), 2)
            )
            fill = fill_color if fill_color else "none"
            parts.append(f'<polygon points="{points}" fill="{fill}"{stroke}/>')
        parts.append("</svg>")

        Path(output_path).write_text("\n".join(parts), encoding="utf-8")

    def _write_png_snapshot(self, output_path: str) -> None:
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise ValueError(
                "PNG export needs Pillow (pip install pillow). "
                "SVG export works without extra packages."
            ) from exc

        self.update_idletasks()
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        ImageGrab.grab(bbox=(x0, y0, x1, y1)).save(output_path)

    def show_model_info(self) -> None:
        if self.model is None:
            messagebox.showinfo("No model", "Load a model to inspect its statistics.")
            return

        stats = self.model.compute_stats()
        size = stats["size"]
        watertight = "yes" if stats["watertight"] else "no"
        volume_note = "" if stats["watertight"] else " (open mesh - approximate)"
        info = (
            f"File: {self.current_path.name if self.current_path else 'untitled'}\n"
            f"Vertices: {stats['vertices']:,}\n"
            f"Triangles: {stats['triangles']:,}\n"
            f"Edges: {stats['edges']:,}\n"
            f"Size (X x Y x Z): {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g}\n"
            f"Surface area: {stats['surface_area']:.6g}\n"
            f"Volume: {stats['volume']:.6g}{volume_note}\n"
            f"Boundary edges: {stats['boundary_edges']:,}\n"
            f"Non-manifold edges: {stats['nonmanifold_edges']:,}\n"
            f"Watertight: {watertight}"
        )
        messagebox.showinfo("Model info", info)

    def _render_now(self) -> None:
        self._render_pending = False
        self.canvas.delete("all")

        if self.model is None:
            self._draw_empty_state()
            return

        frame_start = time.perf_counter()

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        half_w = width * 0.5
        half_h = height * 0.5
        scale = self.base_scale * self.zoom

        vertices = self.model.normalized_vertices
        vertex_count = len(vertices)
        triangles = self.model.triangles
        triangle_count = len(triangles)

        cos_x, sin_x = math.cos(self.rotation_x), math.sin(self.rotation_x)
        cos_y, sin_y = math.cos(self.rotation_y), math.sin(self.rotation_y)
        cos_z, sin_z = math.cos(self.rotation_z), math.sin(self.rotation_z)

        # Combined XYZ rotation as one matrix (also reused for normals).
        m00 = cos_y * cos_z
        m01 = sin_x * sin_y * cos_z - cos_x * sin_z
        m02 = cos_x * sin_y * cos_z + sin_x * sin_z
        m10 = cos_y * sin_z
        m11 = sin_x * sin_y * sin_z + cos_x * cos_z
        m12 = cos_x * sin_y * sin_z - sin_x * cos_z
        m20 = -sin_y
        m21 = sin_x * cos_y
        m22 = cos_x * cos_y

        use_ortho = self.ortho_var.get()
        near_clip = self.near_clip_var.get()
        offset_x = half_w + self.pan_x
        offset_y = half_h + self.pan_y
        rotated_min_y = 0.0

        use_numpy = (
            _np is not None
            and self.model._np_vertices is not None
            and vertex_count >= NUMPY_VERTEX_THRESHOLD
        )
        if use_numpy:
            rot_matrix = _np.array(
                ((m00, m01, m02), (m10, m11, m12), (m20, m21, m22)), dtype=_np.float64
            )
            rot_np = self.model._np_vertices @ rot_matrix.T
            rotated = rot_np.tolist()
            rotated_min_y = float(rot_np[:, 1].min())
            z_cam_np = rot_np[:, 2] + self.camera_distance
            camera_points = _np.column_stack(
                (rot_np[:, 0], rot_np[:, 1], z_cam_np)
            ).tolist()
            if use_ortho:
                projected = _np.column_stack(
                    (
                        rot_np[:, 0] * scale + offset_x,
                        -rot_np[:, 1] * scale + offset_y,
                        z_cam_np,
                    )
                ).tolist()
            else:
                safe_z = _np.where(z_cam_np > near_clip, z_cam_np, 1.0)
                perspective_scale = (self.focal_length * scale) / safe_z
                projected = _np.column_stack(
                    (
                        rot_np[:, 0] * perspective_scale + offset_x,
                        -rot_np[:, 1] * perspective_scale + offset_y,
                        z_cam_np,
                    )
                ).tolist()
                for idx in _np.nonzero(z_cam_np <= near_clip)[0].tolist():
                    projected[idx] = None
        else:
            rotated = [(0.0, 0.0, 0.0)] * vertex_count
            min_y = math.inf
            for idx, (vx, vy, vz) in enumerate(vertices):
                ry = m10 * vx + m11 * vy + m12 * vz
                rotated[idx] = (
                    m00 * vx + m01 * vy + m02 * vz,
                    ry,
                    m20 * vx + m21 * vy + m22 * vz,
                )
                if ry < min_y:
                    min_y = ry
            rotated_min_y = 0.0 if math.isinf(min_y) else min_y

            camera_points = [(0.0, 0.0, 0.0)] * vertex_count
            projected = [None] * vertex_count
            if use_ortho:
                for idx, (x, y, z) in enumerate(rotated):
                    z_cam = z + self.camera_distance
                    camera_points[idx] = (x, y, z_cam)
                    projected[idx] = (
                        x * scale + offset_x,
                        -y * scale + offset_y,
                        z_cam,
                    )
            else:
                for idx, (x, y, z) in enumerate(rotated):
                    z_cam = z + self.camera_distance
                    camera_points[idx] = (x, y, z_cam)
                    if z_cam <= near_clip:
                        continue
                    perspective = self.focal_length / z_cam
                    projected[idx] = (
                        x * scale * perspective + offset_x,
                        -y * scale * perspective + offset_y,
                        z_cam,
                    )

        interacting = self._is_interacting()
        stride = self._triangle_stride(triangle_count, interacting)

        fill_enabled = self.fill_var.get()
        wire_enabled = self.wireframe_var.get()
        if interacting and self.quality_var.get() == "Auto" and triangle_count > 18000 and fill_enabled:
            wire_enabled = False
        shading_enabled = self.shading_var.get()
        smooth_enabled = self.smooth_var.get()
        shadow_enabled = self.shadow_var.get()
        smooth_normals = self.model.smooth_normals
        culling_enabled = self.culling_var.get()
        invert_culling = self.invert_culling_var.get()
        depth_sort_enabled = self.depth_sort_var.get()
        color_mode = self.color_mode_var.get()
        depth_heatmap_mode = color_mode == "Depth Heatmap"
        normal_tint_mode = color_mode == "Normal Tint"
        show_normals = self.show_normals_var.get()
        specular_strength = self.specular_var.get()
        gloss_power = max(1.0, self.gloss_var.get())
        normal_length = self.normal_len_var.get()
        cull_sign = -1.0 if invert_culling else 1.0

        if not fill_enabled and not wire_enabled:
            self.canvas.create_text(
                width / 2,
                28,
                text="Nothing to render: enable Fill or Wireframe",
                fill=self._hint_color,
                font=("Segoe UI", 11),
            )
            return

        ambient = self.ambient_var.get()
        diffuse_strength = self.diffuse_var.get()
        lx, ly, lz = self._light_dir

        draw_queue: list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float, float, float, float],
                str,
            ]
        ] = []
        queue_append = draw_queue.append
        normal_lines: list[tuple[float, float, float, float]] = []
        normal_budget = 360 if interacting else 1300
        normal_stride = max(1, (triangle_count + normal_budget - 1) // normal_budget)

        def project_camera_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
            perspective = self.focal_length / point[2]
            return (
                point[0] * scale * perspective + half_w + self.pan_x,
                -point[1] * scale * perspective + half_h + self.pan_y,
                point[2],
            )

        for tri_idx in range(0, triangle_count, stride):
            tri = triangles[tri_idx]
            i0, i1, i2 = tri.i0, tri.i1, tri.i2

            if i0 >= vertex_count or i1 >= vertex_count or i2 >= vertex_count:
                continue

            color = tri.color

            p0 = rotated[i0]
            p1 = rotated[i1]
            p2 = rotated[i2]

            normal_sq = 0.0
            nx = ny = nz = 0.0
            nx_unit = ny_unit = nz_unit = 0.0
            center_ready = False
            cx = cy = cz = 0.0
            vx = vy = vz = 0.0

            if culling_enabled or shading_enabled or normal_tint_mode or show_normals:
                ax = p1[0] - p0[0]
                ay = p1[1] - p0[1]
                az = p1[2] - p0[2]
                bx = p2[0] - p0[0]
                by = p2[1] - p0[1]
                bz = p2[2] - p0[2]

                nx = ay * bz - az * by
                ny = az * bx - ax * bz
                nz = ax * by - ay * bx

                normal_sq = nx * nx + ny * ny + nz * nz
                if culling_enabled and normal_sq <= NORMAL_EPS:
                    continue

                if normal_sq > NORMAL_EPS:
                    inv_normal = 1.0 / math.sqrt(normal_sq)
                    nx_unit = nx * inv_normal
                    ny_unit = ny * inv_normal
                    nz_unit = nz * inv_normal
                    if culling_enabled and invert_culling:
                        nx_unit = -nx_unit
                        ny_unit = -ny_unit
                        nz_unit = -nz_unit

                if culling_enabled:
                    cx = (p0[0] + p1[0] + p2[0]) / 3.0
                    cy = (p0[1] + p1[1] + p2[1]) / 3.0
                    cz = (p0[2] + p1[2] + p2[2]) / 3.0
                    center_ready = True
                    vx = -cx
                    vy = -cy
                    vz = -self.camera_distance - cz
                    if (nx * vx + ny * vy + nz * vz) * cull_sign <= 0.0:
                        continue

                shade_x, shade_y, shade_z = nx_unit, ny_unit, nz_unit
                shade_valid = normal_sq > NORMAL_EPS
                if smooth_enabled and (shading_enabled or normal_tint_mode):
                    sn = smooth_normals[tri_idx]
                    if sn[0] != 0.0 or sn[1] != 0.0 or sn[2] != 0.0:
                        smooth_x = m00 * sn[0] + m01 * sn[1] + m02 * sn[2]
                        smooth_y = m10 * sn[0] + m11 * sn[1] + m12 * sn[2]
                        smooth_z = m20 * sn[0] + m21 * sn[1] + m22 * sn[2]
                        if culling_enabled and invert_culling:
                            smooth_x = -smooth_x
                            smooth_y = -smooth_y
                            smooth_z = -smooth_z
                        shade_x, shade_y, shade_z = smooth_x, smooth_y, smooth_z
                        shade_valid = True

                if normal_tint_mode:
                    if shade_valid:
                        color = to_hex_color(
                            (
                                int(abs(shade_x) * 255),
                                int(abs(shade_y) * 255),
                                int(abs(shade_z) * 255),
                            )
                        )
                    else:
                        color = "#808080"
                elif shading_enabled and shade_valid:
                    light_dot = shade_x * lx + shade_y * ly + shade_z * lz
                    diffuse = max(0.0, light_dot if culling_enabled else abs(light_dot))
                    # Hemisphere ambient: surfaces facing up receive a bit more sky light.
                    ambient_term = ambient * (0.76 + 0.24 * shade_y)
                    diffuse_term = clamp(ambient_term + diffuse_strength * diffuse, 0.05, 1.0)

                    if not center_ready:
                        cx = (p0[0] + p1[0] + p2[0]) / 3.0
                        cy = (p0[1] + p1[1] + p2[1]) / 3.0
                        cz = (p0[2] + p1[2] + p2[2]) / 3.0
                        center_ready = True

                    specular_term = 0.0
                    if specular_strength > 0.0005:
                        vx = -cx
                        vy = -cy
                        vz = -self.camera_distance - cz
                        view_sq = vx * vx + vy * vy + vz * vz
                        if view_sq > 1e-20:
                            inv_view = 1.0 / math.sqrt(view_sq)
                            hx = lx + vx * inv_view
                            hy = ly + vy * inv_view
                            hz = lz + vz * inv_view
                            half_sq = hx * hx + hy * hy + hz * hz
                            if half_sq > 1e-20:
                                inv_half = 1.0 / math.sqrt(half_sq)
                                ndoth = max(
                                    0.0,
                                    shade_x * (hx * inv_half)
                                    + shade_y * (hy * inv_half)
                                    + shade_z * (hz * inv_half),
                                )
                                specular_term = (ndoth**gloss_power) * specular_strength

                    diffuse_bucket = int(clamp(diffuse_term, 0.0, 1.0) * 255)
                    specular_bucket = int(clamp(specular_term, 0.0, 1.0) * 255)
                    color = self._shade_color_phong(tri.color, diffuse_bucket, specular_bucket)

            if show_normals and normal_sq > NORMAL_EPS:
                sampled_index = tri_idx // stride
                if sampled_index % normal_stride == 0:
                    if not center_ready:
                        cx = (p0[0] + p1[0] + p2[0]) / 3.0
                        cy = (p0[1] + p1[1] + p2[1]) / 3.0
                        cz = (p0[2] + p1[2] + p2[2]) / 3.0

                    tip_x = cx + nx_unit * normal_length
                    tip_y = cy + ny_unit * normal_length
                    tip_z = cz + nz_unit * normal_length

                    if use_ortho:
                        start_x = cx * scale + half_w + self.pan_x
                        start_y = -cy * scale + half_h + self.pan_y
                        end_x = tip_x * scale + half_w + self.pan_x
                        end_y = -tip_y * scale + half_h + self.pan_y
                        normal_lines.append((start_x, start_y, end_x, end_y))
                    else:
                        start_z = cz + self.camera_distance
                        end_z = tip_z + self.camera_distance
                        if start_z > near_clip and end_z > near_clip:
                            start_p = self.focal_length / start_z
                            end_p = self.focal_length / end_z
                            start_x = cx * scale * start_p + half_w + self.pan_x
                            start_y = -cy * scale * start_p + half_h + self.pan_y
                            end_x = tip_x * scale * end_p + half_w + self.pan_x
                            end_y = -tip_y * scale * end_p + half_h + self.pan_y
                            normal_lines.append((start_x, start_y, end_x, end_y))

            projected_triangles: list[
                tuple[
                    tuple[float, float, float],
                    tuple[float, float, float],
                    tuple[float, float, float],
                ]
            ]
            if use_ortho:
                proj0 = projected[i0]
                proj1 = projected[i1]
                proj2 = projected[i2]
                if proj0 is None or proj1 is None or proj2 is None:
                    continue
                projected_triangles = [(proj0, proj1, proj2)]
            else:
                cam0 = camera_points[i0]
                cam1 = camera_points[i1]
                cam2 = camera_points[i2]

                if cam0[2] <= near_clip and cam1[2] <= near_clip and cam2[2] <= near_clip:
                    continue

                if cam0[2] > near_clip and cam1[2] > near_clip and cam2[2] > near_clip:
                    proj0 = projected[i0]
                    proj1 = projected[i1]
                    proj2 = projected[i2]
                    if proj0 is None or proj1 is None or proj2 is None:
                        continue
                    projected_triangles = [(proj0, proj1, proj2)]
                else:
                    clipped = clip_triangle_near_plane(cam0, cam1, cam2, near_clip)
                    if not clipped:
                        continue
                    projected_triangles = [
                        (
                            project_camera_point(cp0),
                            project_camera_point(cp1),
                            project_camera_point(cp2),
                        )
                        for cp0, cp1, cp2 in clipped
                    ]

            for proj0, proj1, proj2 in projected_triangles:
                coords = (
                    proj0[0],
                    proj0[1],
                    proj1[0],
                    proj1[1],
                    proj2[0],
                    proj2[1],
                )
                d0, d1, d2 = proj0[2], proj1[2], proj2[2]
                depth_key = (max(d0, d1, d2), (d0 + d1 + d2) / 3.0, min(d0, d1, d2))
                queue_append((depth_key, coords, color))

        if depth_sort_enabled and fill_enabled and len(draw_queue) > 1:
            draw_queue.sort(key=lambda item: item[0], reverse=True)

        # Planar projected shadow (Blinn-style): drop each vertex along the
        # light direction onto a ground plane just below the model.
        shadow_polys: list[tuple[float, ...]] = []
        skip_shadow = (
            interacting and self.quality_var.get() == "Auto" and triangle_count > 9000
        )
        if shadow_enabled and not skip_shadow and ly > SHADOW_MIN_LIGHT_Y:
            # The shadow is a soft visual cue; half resolution is plenty.
            shadow_stride = stride * 2 if triangle_count // stride > 2500 else stride
            ground_y = rotated_min_y - 0.045
            ground_screen_ortho_y = -ground_y * scale + offset_y
            inv_ly = 1.0 / ly
            for tri_idx in range(0, triangle_count, shadow_stride):
                tri = triangles[tri_idx]
                points: list[float] = []
                visible = True
                for vertex_index in (tri.i0, tri.i1, tri.i2):
                    px, py, pz = rotated[vertex_index]
                    travel = (py - ground_y) * inv_ly
                    shadow_x = px - travel * lx
                    shadow_z = pz - travel * lz
                    if use_ortho:
                        points.append(shadow_x * scale + offset_x)
                        points.append(ground_screen_ortho_y)
                    else:
                        z_cam = shadow_z + self.camera_distance
                        if z_cam <= near_clip:
                            visible = False
                            break
                        perspective = self.focal_length / z_cam
                        points.append(shadow_x * scale * perspective + offset_x)
                        points.append(-ground_y * scale * perspective + offset_y)
                if visible:
                    shadow_polys.append(tuple(points))

            shadow_fill = self._shadow_color
            for coords in shadow_polys:
                self.canvas.create_polygon(coords, fill=shadow_fill, outline="")

        outline_color = "#1f2a38" if not self.dark_bg_var.get() else "#9db0c6"
        wire_width = max(1, int(round(self.wire_width_var.get())))
        heat_min = 0.0
        heat_span = 1.0
        if fill_enabled and depth_heatmap_mode and draw_queue:
            depth_values = [depth_key[1] for depth_key, _coords, _color in draw_queue]
            heat_min = min(depth_values)
            heat_span = max(1e-9, max(depth_values) - heat_min)

        capture_polys: list[tuple[tuple[float, ...], str]] = []
        for depth_key, tri_coords, color in draw_queue:
            fill_color = color
            if fill_enabled and depth_heatmap_mode:
                heat_t = (depth_key[1] - heat_min) / heat_span
                fill_color = self._depth_color_bucket(int(clamp(heat_t, 0.0, 1.0) * 255))
            self.canvas.create_polygon(
                tri_coords,
                fill=fill_color if fill_enabled else "",
                outline=outline_color if wire_enabled else "",
                width=wire_width if wire_enabled else 0,
            )
            capture_polys.append((tri_coords, fill_color if fill_enabled else ""))

        self._last_render_capture = {
            "width": width,
            "height": height,
            "bg": self.canvas.cget("bg"),
            "shadow": (shadow_polys, self._shadow_color),
            "polys": capture_polys,
            "outline": outline_color if wire_enabled else "",
            "wire_width": wire_width if wire_enabled else 0,
        }

        if show_normals and normal_lines:
            normal_color = "#f29f32" if not self.dark_bg_var.get() else "#ffd07e"
            for x0, y0, x1, y1 in normal_lines:
                self.canvas.create_line(x0, y0, x1, y1, fill=normal_color, width=1)

        self._draw_bounds_overlay(
            width=width,
            height=height,
            half_w=half_w,
            half_h=half_h,
            scale=scale,
            cos_x=cos_x,
            sin_x=sin_x,
            cos_y=cos_y,
            sin_y=sin_y,
            cos_z=cos_z,
            sin_z=sin_z,
            use_ortho=use_ortho,
            near_clip=near_clip,
        )
        self._draw_axis_gizmo(width, cos_x, sin_x, cos_y, sin_y, cos_z, sin_z)

        now = time.perf_counter()
        frame_dt = now - self._last_frame_time
        self._last_frame_time = now
        if frame_dt > 0:
            fps_now = 1.0 / frame_dt
            self._fps = fps_now if self._fps == 0.0 else (self._fps * 0.86 + fps_now * 0.14)

        model_name = self.current_path.name if self.current_path else "untitled"
        rendered_triangles = len(draw_queue)
        sampled_triangles = max(1, (triangle_count + stride - 1) // stride)
        frame_ms = (now - frame_start) * 1000.0

        if culling_enabled and rendered_triangles < int(sampled_triangles * 0.14):
            self.canvas.create_text(
                width / 2,
                48,
                text="Most faces are culled. Try disabling Backface Culling or enabling Invert Culling.",
                fill=self._hint_color,
                font=("Segoe UI", 10),
            )

        self.canvas.create_text(
            10,
            10,
            anchor="nw",
            text=(
                f"{model_name} | zoom {self.zoom:.2f}x | "
                f"rx {math.degrees(self.rotation_x):.1f} ry {math.degrees(self.rotation_y):.1f} rz {math.degrees(self.rotation_z):.1f}"
            ),
            fill=self._overlay_color,
            font=("Segoe UI Semibold", 10),
        )
        self.canvas.create_text(
            10,
            28,
            anchor="nw",
            text=(
                f"tris {rendered_triangles:,}/{triangle_count:,} | "
                f"stride {stride} | {self.quality_var.get()} | mode {color_mode} | "
                f"near {near_clip:.2f} | normals {len(normal_lines)} | {self._fps:.1f} fps | {frame_ms:.1f} ms"
            ),
            fill=self._overlay_color,
            font=("Segoe UI", 9),
        )

        self.status_var.set(
            f"Loaded: {model_name} | Vertices: {vertex_count:,} | Triangles: {triangle_count:,} | "
            f"Rendered: {rendered_triangles:,} | Mode: {color_mode} | Near: {near_clip:.2f} | "
            f"FPS: {self._fps:.1f} | Frame: {frame_ms:.1f} ms"
        )


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="PyOBJViewer",
        description="Dependency-free 3D model viewer for OBJ and STL files.",
    )
    parser.add_argument("path", nargs="?", help="model file to open (.obj or .stl)")
    parser.add_argument(
        "--version", action="version", version=f"PyOBJViewer {__version__}"
    )
    parser.add_argument("--dark", action="store_true", help="start with the dark background")
    parser.add_argument("--ortho", action="store_true", help="start in orthographic projection")
    parser.add_argument("--spin", action="store_true", help="start with auto-spin enabled")
    parser.add_argument("--no-shadow", action="store_true", help="disable the ground shadow")
    parser.add_argument(
        "--quality",
        choices=("Auto", "High", "Balanced", "Fast"),
        help="initial render quality preset",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = _parse_args(sys.argv[1:])
    _enable_windows_dpi_awareness()
    app = ModelViewer()
    if cli_args.dark:
        app.dark_bg_var.set(True)
        app._apply_background_theme()
    if cli_args.ortho:
        app.ortho_var.set(True)
    if cli_args.spin:
        app.auto_spin_var.set(True)
    if cli_args.no_shadow:
        app.shadow_var.set(False)
    if cli_args.quality:
        app.quality_var.set(cli_args.quality)
    if cli_args.path:
        app.after(60, lambda: app._load_model_from_path(cli_args.path))
    app.mainloop()
