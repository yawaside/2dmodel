#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal, self-contained MOC3 (Cubism 3.00 / format version 1) writer.

Layout facts verified against a real model (haru_greeter_t03.moc3, MOC3 v1)
and against Cubism Core 4.2.2 (moc revived + model initialized + updated):

  * header (64B) + section offset table (160 x u32 = 640B) + padding to 1984
  * SOT[0] -> count info (23 x i32, padded to 128B)
  * SOT[1] -> canvas info (5 x f32 + 1 byte, padded to 64B)
  * SOT[2..] -> body sections, each padded to 64B
  * one keyform binding (curve) per parameter, in parameter order
  * keyform count of a drawable/deformer = product of the key counts of all
    bindings in its binding band (full tensor grid)
  * every keyform position block is padded up to a multiple of 16 floats
  * `art_mesh.position_index_counts` holds the VERTEX count,
    `art_mesh.vertex_counts` holds the INDEX (triangle corner) count
"""

from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MAGIC = b"MOC3"
HEADER_SIZE = 64
SOT_COUNT = 160
SOT_SIZE = 640
COUNT_INFO_MAX = 23
COUNT_INFO_SIZE = 128
DEFAULT_OFFSET = 1984
ALIGN = 64
VERSION_V300 = 1

# count-info indices
C_PARTS, C_DEFORMERS, C_WARP, C_ROT, C_ARTMESH, C_PARAMS, C_PART_KF, C_WARP_KF, \
    C_ROT_KF, C_ART_KF, C_KF_POS, C_KBIDX, C_BANDS, C_BINDINGS, C_KEYS, C_UVS, \
    C_POSIDX, C_MASKS, C_DOG, C_DOGO, C_GLUES, C_GLUEINFO, C_GLUEKF = range(23)


def _pad(n: int, a: int = ALIGN) -> int:
    return n if n % a == 0 else n + (a - n % a)


class _W:
    """Tiny little-endian binary writer."""

    def __init__(self) -> None:
        self.buf = bytearray()

    @property
    def pos(self) -> int:
        return len(self.buf)

    def raw(self, data: bytes) -> None:
        self.buf.extend(data)

    def i32(self, *vals: int) -> None:
        self.buf.extend(struct.pack("<%di" % len(vals), *vals))

    def u32(self, *vals: int) -> None:
        self.buf.extend(struct.pack("<%dI" % len(vals), *vals))

    def f32(self, *vals: float) -> None:
        self.buf.extend(struct.pack("<%df" % len(vals), *vals))

    def i16(self, *vals: int) -> None:
        self.buf.extend(struct.pack("<%dh" % len(vals), *vals))

    def u8(self, *vals: int) -> None:
        self.buf.extend(bytes(vals))

    def bools(self, vals: Sequence[bool]) -> None:
        self.i32(*[1 if v else 0 for v in vals])

    def str64(self, s: str) -> None:
        b = s.encode("utf-8")
        assert len(b) < 64, "id too long: %s" % s
        self.buf.extend(b)
        self.buf.extend(b"\x00" * (64 - len(b)))

    def strs(self, vals: Sequence[str]) -> None:
        for s in vals:
            self.str64(s)

    def align(self, a: int = ALIGN) -> None:
        n = _pad(self.pos, a) - self.pos
        if n:
            self.buf.extend(b"\x00" * n)

    def pad_bytes(self, n: int) -> None:
        self.buf.extend(b"\x00" * n)


# --------------------------------------------------------------------------- #
# model description classes
# --------------------------------------------------------------------------- #

@dataclass
class Param:
    id: str
    min_value: float
    max_value: float
    default_value: float
    keys: List[float] = field(default_factory=list)
    repeat: bool = False
    decimals: int = 2


@dataclass
class KeyformSet:
    """Vertices/grid of one keyform."""
    positions: List[Tuple[float, float]]
    opacity: float = 1.0


@dataclass
class ArtMesh:
    id: str
    verts: List[Tuple[float, float]]
    uvs: List[Tuple[float, float]]
    tris: List[Tuple[int, int, int]]
    draw_order: float = 0.0
    texture_index: int = 0
    parent_part: int = 0
    parent_deformer: int = -1
    params: List[str] = field(default_factory=list)
    pos_fn: Optional[Callable[[Dict[str, float]], List[Tuple[float, float]]]] = None
    opa_fn: Optional[Callable[[Dict[str, float]], float]] = None
    draw_order_fn: Optional[Callable[[Dict[str, float]], float]] = None


@dataclass
class WarpDeformer:
    id: str
    rows: int          # number of cells vertically
    cols: int          # number of cells horizontally
    grid: List[Tuple[float, float]]   # (rows+1)*(cols+1) points, row-major
    parent_part: int = 0
    parent_deformer: int = -1
    params: List[str] = field(default_factory=list)
    pos_fn: Optional[Callable[[Dict[str, float]], List[Tuple[float, float]]]] = None


@dataclass
class Part:
    id: str
    draw_order: float = 0.0
    parent: int = -1


class ModelBuilder:
    def __init__(self, canvas_w: float, canvas_h: float, ppu: float,
                 origin_x: float, origin_y: float) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.ppu = ppu
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.params: List[Param] = []
        self._pidx: Dict[str, int] = {}
        self.parts: List[Part] = []
        self.deformers: List[WarpDeformer] = []
        self.meshes: List[ArtMesh] = []
        self._bands: List[Tuple[int, ...]] = []

    # -- description ------------------------------------------------------- #

    def add_param(self, pid: str, min_value: float, max_value: float,
                  default_value: float, keys: Sequence[float]) -> Param:
        assert pid not in [p.id for p in self.params], "duplicate param %s" % pid
        p = Param(pid, float(min_value), float(max_value), float(default_value),
                  sorted(float(k) for k in keys))
        self._pidx[pid] = len(self.params)
        self.params.append(p)
        return p

    def add_part(self, pid: str, draw_order: float = 0.0, parent: int = -1) -> int:
        self.parts.append(Part(pid, float(draw_order), parent))
        return len(self.parts) - 1

    def add_warp_deformer(self, **kw) -> int:
        d = WarpDeformer(**kw)
        assert len(d.grid) == (d.rows + 1) * (d.cols + 1), "grid size mismatch"
        self.deformers.append(d)
        return len(self.deformers) - 1

    def add_art_mesh(self, **kw) -> int:
        m = ArtMesh(**kw)
        assert len(m.verts) == len(m.uvs)
        assert len(m.verts) < 65536, "too many vertices (i16 index limit)"
        for t in m.tris:
            assert max(t) < len(m.verts)
        self.meshes.append(m)
        return len(self.meshes) - 1

    # -- band bookkeeping -------------------------------------------------- #

    def _band_id(self, param_ids: Sequence[str]) -> int:
        idx = tuple(sorted(self._pidx[p] for p in param_ids))
        if idx not in self._bands:
            self._bands.append(idx)
        return self._bands.index(idx)

    # -- keyform enumeration ----------------------------------------------- #

    def _states(self, param_ids: Sequence[str]):
        """Yield (state_dict) over the full tensor grid, last param fastest."""
        if not param_ids:
            yield {}
            return
        # Core flattens the tensor grid with the FIRST binding varying fastest:
        #   index = i0 + i1*k0 + i2*k0*k1 ...
        # The bindings of a band are ordered by parameter index, so the caller's
        # parameter order has to be normalised before enumerating.
        order = sorted(param_ids, key=lambda p: self._pidx[p])
        key_lists = [self.params[self._pidx[p]].keys for p in order]
        for combo in itertools.product(*reversed(key_lists)):
            yield dict(zip(order, reversed(combo)))

    # -- build -------------------------------------------------------------- #

    def build(self) -> bytes:
        params = self.params
        n_param = len(params)
        n_part = len(self.parts)
        n_def = len(self.deformers)
        n_mesh = len(self.meshes)

        # ---- bindings: one per parameter, identical index ----------------- #
        binding_keys: List[List[float]] = [p.keys for p in params]
        keys_flat: List[float] = []
        binding_keys_begin: List[int] = []
        binding_keys_count: List[int] = []
        for ks in binding_keys:
            binding_keys_begin.append(len(keys_flat))
            binding_keys_count.append(len(ks))
            keys_flat.extend(ks)

        # ---- bands -------------------------------------------------------- #
        for d in self.deformers:
            self._band_id(d.params)
        for m in self.meshes:
            self._band_id(m.params)
        if () not in self._bands:
            self._bands.insert(0, ())
        bands: List[Tuple[int, ...]] = self._bands
        band_begin: List[int] = []
        band_count: List[int] = []
        kbi_flat: List[int] = []
        for b in bands:
            band_begin.append(len(kbi_flat))
            band_count.append(len(b))
            kbi_flat.extend(b)

        def band_of(param_ids) -> int:
            return bands.index(tuple(sorted(self._pidx[p] for p in param_ids)))

        # ---- keyform data -------------------------------------------------- #
        kf_pos: List[float] = []          # padded flat keyform positions
        art_kf_begin: List[int] = []
        art_kf_opa: List[float] = []
        art_kf_do: List[float] = []
        art_kf_counts: List[int] = []
        art_kf_first: List[int] = []
        art_kf_band: List[int] = []
        art_vcount: List[int] = []
        art_idxlen: List[int] = []

        def push_positions(pts: Sequence[Tuple[float, float]], base_len: int) -> int:
            begin = len(kf_pos)
            assert len(pts) == base_len, "keyform vertex count mismatch (%d vs %d)" % (len(pts), base_len)
            for (x, y) in pts:
                kf_pos.append(float(x))
                kf_pos.append(float(y))
            while len(kf_pos) % 16:
                kf_pos.append(0.0)
            return begin

        for m in self.meshes:
            art_kf_first.append(len(art_kf_opa))
            art_kf_band.append(band_of(m.params))
            nv = len(m.verts)
            art_vcount.append(nv)
            art_idxlen.append(3 * len(m.tris))
            cnt = 0
            for st in self._states(m.params):
                pts = m.pos_fn(st) if m.pos_fn else m.verts
                opa = m.opa_fn(st) if m.opa_fn else 1.0
                do = m.draw_order_fn(st) if m.draw_order_fn else m.draw_order
                art_kf_begin.append(push_positions(pts, nv))
                art_kf_opa.append(float(opa))
                art_kf_do.append(float(do))
                cnt += 1
            art_kf_counts.append(cnt)

        warp_kf_begin: List[int] = []
        warp_kf_opa: List[float] = []
        warp_kf_counts: List[int] = []
        warp_first: List[int] = []
        warp_band: List[int] = []
        warp_vcount: List[int] = []
        warp_rows: List[int] = []
        warp_cols: List[int] = []
        for d in self.deformers:
            warp_first.append(len(warp_kf_opa))
            warp_band.append(band_of(d.params))
            n = (d.rows + 1) * (d.cols + 1)
            warp_vcount.append(n)
            warp_rows.append(d.rows)
            warp_cols.append(d.cols)
            cnt = 0
            for st in self._states(d.params):
                pts = d.pos_fn(st) if d.pos_fn else d.grid
                warp_kf_begin.append(push_positions(pts, n))
                warp_kf_opa.append(1.0)
                cnt += 1
            warp_kf_counts.append(cnt)

        # ---- flat uv / index arrays ---------------------------------------- #
        uv_flat: List[float] = []
        uv_begin: List[int] = []
        for m in self.meshes:
            uv_begin.append(len(uv_flat))
            for (u, v) in m.uvs:
                uv_flat.append(float(u))
                uv_flat.append(float(v))
        idx_flat: List[int] = []
        idx_begin: List[int] = []
        for m in self.meshes:
            idx_begin.append(len(idx_flat))
            for t in m.tris:
                idx_flat.extend(t)

        counts = [0] * COUNT_INFO_MAX
        counts[C_PARTS] = n_part
        counts[C_DEFORMERS] = n_def
        counts[C_WARP] = n_def
        counts[C_ROT] = 0
        counts[C_ARTMESH] = n_mesh
        counts[C_PARAMS] = n_param
        counts[C_PART_KF] = n_part
        counts[C_WARP_KF] = len(warp_kf_opa)
        counts[C_ROT_KF] = 0
        counts[C_ART_KF] = len(art_kf_opa)
        counts[C_KF_POS] = len(kf_pos)
        counts[C_KBIDX] = len(kbi_flat)
        counts[C_BANDS] = len(bands)
        counts[C_BINDINGS] = n_param
        counts[C_KEYS] = len(keys_flat)
        counts[C_UVS] = len(uv_flat)
        counts[C_POSIDX] = len(idx_flat)

        # ---- draw order group: one group holding every art mesh ------------ #
        dog_begin = [0]
        dog_count = [n_mesh]
        dog_total = [n_mesh]
        # NB: these two fields are stored swapped in the file (the first one is
        # the *maximum*, the second one the *minimum*).  Core only derives
        # per-drawable render orders when every draw order lies inside the
        # [min, max] range, and render orders are what the Cubism renderer sorts
        # by - getting this wrong makes the model render in a broken order.
        _dos = [m.draw_order for m in self.meshes] or [0.0]
        dog_max = [int(min(_dos))]              # stored first  -> lower bound
        dog_min = [int(max(_dos)) + 1]          # stored second -> upper bound
        dogo_type = [0] * n_mesh
        dogo_index = list(range(n_mesh))
        dogo_group = [0] * n_mesh
        counts[C_DOG] = 1
        counts[C_DOGO] = n_mesh

        # ---- serialize ------------------------------------------------------ #
        w = _W()
        sot: List[int] = []

        def section(fn):
            w.align(ALIGN)
            sot.append(DEFAULT_OFFSET + w.pos)
            fn()

        # count info (SOT[0])
        sot.append(DEFAULT_OFFSET + w.pos)
        w.i32(*counts)
        w.pad_bytes(COUNT_INFO_SIZE - COUNT_INFO_MAX * 4)
        # canvas info (SOT[1])
        sot.append(DEFAULT_OFFSET + w.pos)
        w.f32(float(self.ppu), float(self.origin_x), float(self.origin_y),
              float(self.canvas_w), float(self.canvas_h))
        w.u8(0)
        w.pad_bytes(64 - (5 * 4 + 1))

        def rt(n: int):  # runtime space: 8 zero bytes per element
            w.pad_bytes(8 * n)

        # parts
        section(lambda: rt(n_part))
        section(lambda: w.strs([p.id for p in self.parts]))
        section(lambda: w.i32(*[band_of([]) for _ in self.parts]))
        section(lambda: w.i32(*list(range(n_part))))
        section(lambda: w.i32(*[1] * n_part))
        section(lambda: w.bools([True] * n_part))
        section(lambda: w.bools([True] * n_part))
        section(lambda: w.i32(*[p.parent for p in self.parts]))

        # deformers (warp only)
        section(lambda: rt(n_def))                                    # runtime
        section(lambda: w.strs([d.id for d in self.deformers]))       # ids
        section(lambda: w.i32(*warp_band))                            # band (generic)
        section(lambda: w.bools([True] * n_def))                      # visibles
        section(lambda: w.bools([True] * n_def))                      # enables
        section(lambda: w.i32(*[d.parent_part for d in self.deformers]))
        section(lambda: w.i32(*[d.parent_deformer for d in self.deformers]))
        section(lambda: w.i32(*[0] * n_def))                          # types: warp
        section(lambda: w.i32(*list(range(n_def))))                   # specific idx
        # warp deformer specifics
        section(lambda: w.i32(*warp_band))
        section(lambda: w.i32(*warp_first))
        section(lambda: w.i32(*warp_kf_counts))
        section(lambda: w.i32(*warp_vcount))
        section(lambda: w.i32(*warp_rows))
        section(lambda: w.i32(*warp_cols))
        # rotation deformer specifics (empty)
        for _ in range(4):
            section(lambda: None)

        # art meshes
        for _ in range(4):
            section(lambda: rt(n_mesh))
        section(lambda: w.strs([m.id for m in self.meshes]))
        section(lambda: w.i32(*art_kf_band))
        section(lambda: w.i32(*art_kf_first))
        section(lambda: w.i32(*art_kf_counts))
        section(lambda: w.bools([True] * n_mesh))
        section(lambda: w.bools([True] * n_mesh))
        section(lambda: w.i32(*[m.parent_part for m in self.meshes]))
        section(lambda: w.i32(*[m.parent_deformer for m in self.meshes]))
        section(lambda: w.i32(*[m.texture_index for m in self.meshes]))
        section(lambda: w.u8(*[4] * n_mesh))                 # drawable flags: double sided
        section(lambda: w.i32(*art_vcount))                  # "position_index_counts"
        section(lambda: w.i32(*uv_begin))
        section(lambda: w.i32(*idx_begin))
        section(lambda: w.i32(*art_idxlen))                  # "vertex_counts"
        section(lambda: w.i32(*[0] * n_mesh))                # mask begin
        section(lambda: w.i32(*[0] * n_mesh))                # mask counts

        # parameters
        section(lambda: rt(n_param))
        section(lambda: w.strs([p.id for p in params]))
        section(lambda: w.f32(*[p.max_value for p in params]))
        section(lambda: w.f32(*[p.min_value for p in params]))
        section(lambda: w.f32(*[p.default_value for p in params]))
        section(lambda: w.bools([p.repeat for p in params]))
        section(lambda: w.i32(*[p.decimals for p in params]))
        section(lambda: w.i32(*list(range(n_param))))        # binding begin (1:1)
        section(lambda: w.i32(*[1] * n_param))               # binding counts

        # part keyforms
        section(lambda: w.f32(*[p.draw_order for p in self.parts]))
        # warp keyforms
        section(lambda: w.f32(*warp_kf_opa))
        section(lambda: w.i32(*warp_kf_begin))
        # rotation keyforms (empty)
        for _ in range(7):
            section(lambda: None)
        # art mesh keyforms
        section(lambda: w.f32(*art_kf_opa))
        section(lambda: w.f32(*art_kf_do))
        section(lambda: w.i32(*art_kf_begin))
        # keyform positions
        section(lambda: w.f32(*kf_pos))
        # binding indices / bands / bindings / keys
        section(lambda: w.i32(*kbi_flat))
        section(lambda: w.i32(*band_begin))
        section(lambda: w.i32(*band_count))
        section(lambda: w.i32(*binding_keys_begin))
        section(lambda: w.i32(*binding_keys_count))
        section(lambda: w.f32(*keys_flat))
        # uv / position indices
        section(lambda: w.f32(*uv_flat))
        section(lambda: w.i16(*idx_flat))
        # masks
        section(lambda: None)
        # draw order groups / objects
        section(lambda: w.i32(*dog_begin))
        section(lambda: w.i32(*dog_count))
        section(lambda: w.i32(*dog_total))
        section(lambda: w.i32(*dog_min))
        section(lambda: w.i32(*dog_max))
        section(lambda: w.i32(*dogo_type))
        section(lambda: w.i32(*dogo_index))
        section(lambda: w.i32(*dogo_group))
        # glues (empty)
        for _ in range(12):
            section(lambda: None)

        body = bytes(w.buf)

        out = _W()
        out.raw(MAGIC)
        out.u8(VERSION_V300)
        out.u8(0)                      # little endian
        out.pad_bytes(HEADER_SIZE - 6)
        sot = sot + [0] * (SOT_COUNT - len(sot))
        out.u32(*sot[:SOT_COUNT])
        out.pad_bytes(DEFAULT_OFFSET - out.pos)
        out.raw(body)
        out.align(ALIGN)
        return bytes(out.buf)

    def save(self, path: str) -> int:
        data = self.build()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)

def make_localizer(grid_parent, rows, cols, mirror=True):
    """Build pixel/parent-space -> deformer-local coordinate converter.

    Cubism maps a deformer's local [0,1]^2 onto its neutral grid with
        local (lx, ly)  ->  mirror_y( bilinear(grid, u=lx*cols, v=ly*rows) )
    (verified against Cubism Core 4.2.2).  This helper returns the inverse of
    that map for an affine (rectangular) grid, so a vertex that should sit at
    parent-space position (X, Y) can be stored directly.

    `grid_parent` is the neutral grid, row-major, expressed in the parent space.
    `mirror` must be True when the parent space is the model space (root
    deformer) and False for nested deformers - the mirror is applied once when
    the root deformer maps into model space, not at every level.
    """
    P0 = (grid_parent[0][0], grid_parent[0][1])
    Pu = grid_parent[cols]
    Pv = grid_parent[rows * (cols + 1)]
    U = ((Pu[0] - P0[0]) / cols, (Pu[1] - P0[1]) / cols)
    V = ((Pv[0] - P0[0]) / rows, (Pv[1] - P0[1]) / rows)
    det = U[0] * V[1] - U[1] * V[0]
    if abs(det) < 1e-12:
        raise ValueError("degenerate deformer grid")

    def loc(X, Y):
        dx = X - P0[0]
        dy = (-Y if mirror else Y) - P0[1]
        u = (dx * V[1] - dy * V[0]) / det
        v = (-dx * U[1] + dy * U[0]) / det
        return (u / cols, v / rows)

    return loc

