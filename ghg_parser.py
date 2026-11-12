"""
Parser et extracteur pour les fichiers binaires Lego GHG (format NU20 TT Games).

Le format PC etudie ici n'est pas celui des documents jaandev (variante big-endian) :
- les chunks sont little-endian et leur champ `size` inclut les 8 octets d'en-tete ;
- la geometrie n'est pas decrite par MESH/HSEM mais par une section finale
  [u16 nVB][(u32 byteSize, data) * nVB][u16 nIB][u32 ibBytes][u16 indices] ;
- les sous-maillages sont decrits par des enregistrements de 56 octets dans DISP
  (index count, stride, vb offset/count, ib offset, vertex buffer index).
"""

import io
import os
import struct
import zipfile

import numpy as np
import trimesh
from PIL import Image

KNOWN_TAGS = (
    b"HEAD", b"NTBL", b"TREF", b"TST0", b"MS00", b"INID", b"FDNS",
    b"BNDS", b"CDAT", b"VBIB", b"DISP", b"SALI", b"DYNO", b"GSNH",
    b"PNTR", b"MESH", b"HSEM", b"ROTV", b"ANIM",
)

DRAW_RECORD_SIZE = 0x38

VERTEX_LAYOUTS = {
    28: {"pos": 0, "normal": 12, "tangent": 16, "uv": 20},
    32: {"pos": 0, "normal": 12, "tangent": 16, "uv": 24},
    36: {"pos": 0, "normal": 12, "tangent": 16, "uv": 28},
    40: {"pos": 0, "normal": 12, "tangent": 16, "uv": 24},
    44: {"pos": 0, "normal": 12, "tangent": 16, "uv": 28},
}

DXT_BLOCK_BYTES = {b"DXT1": 8, b"DXT5": 16}


class GHGModel:
    @staticmethod
    def _unpack(fmt, data, offset):
        """Unpack with endianness configurable."""
        if fmt.startswith("<"):
            fmt = fmt[1:]
            end = "<"
        else:
            end = ">"
        return struct.unpack_from(end + fmt, data, offset)

    def __init__(self, filepath_or_bytes, name="lego_model", big_endian=False):
        self.name = name
        self._big_endian = big_endian
        if isinstance(filepath_or_bytes, (str, bytes, bytearray)):
            if isinstance(filepath_or_bytes, str):
                self.filepath = filepath_or_bytes
                self.name = os.path.splitext(os.path.basename(filepath_or_bytes))[0]
                with open(filepath_or_bytes, "rb") as f:
                    self.data = f.read()
            else:
                self.filepath = None
                self.data = bytes(filepath_or_bytes)
        else:
            raise ValueError("Le parametre doit etre un chemin de fichier ou des octets.")

        self.size = len(self.data)
        self.chunks = []
        self.ntbl_strings = []
        self.textures = []
        self.mesh = None
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.faces = np.empty((0, 3), dtype=np.int32)
        self.uvs = np.empty((0, 2), dtype=np.float32)
        self.draws = []
        self.vertex_buffers = []
        self.geometry_offset = None
        self.warnings = []
        self.stats = {}
        self.texture_refs = {}
        self.skeleton_bones = []
        self.skeleton_parts = []

        self._parse()

    def _parse(self):
        if len(self.data) < 16 or self.data[:4] != b"NU20":
            raise ValueError("Signature invalide : le fichier n'est pas un fichier Lego NU20 (.ghg).")

        self._read_chunks()
        self._extract_ntbl()
        self._extract_textures()
        self._extract_geometry()
        self._build_stats()

    def _read_chunks(self):
        """Lit la chaine de chunks puis complete avec les tags valides isoles."""
        data = self.data
        found = {}

        def add(tag, offset, size):
            name = tag.decode("latin1", errors="ignore")
            found[offset] = {
                "tag": name,
                "offset": offset,
                "size": size,
                "data_offset": offset + 8,
                "data_size": max(0, size - 8),
                "hex_offset": f"0x{offset:06X}",
            }

        add(b"NU20", 0, 16)

        pos = 16
        while pos + 8 <= len(data):
            tag = data[pos:pos + 4]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if not all(32 <= b <= 126 for b in tag) or size < 8 or pos + size > len(data):
                break
            add(tag, pos, size)
            pos += size

        for tag in KNOWN_TAGS:
            start = 0
            while True:
                idx = data.find(tag, start)
                if idx < 0:
                    break
                start = idx + 1
                if idx + 8 > len(data):
                    continue
                size = struct.unpack_from("<I", data, idx + 4)[0]
                if size < 8 or idx + size > len(data):
                    continue
                if idx in found:
                    continue
                add(tag, idx, size)

        chunks = sorted(found.values(), key=lambda c: c["offset"])
        for chunk in chunks:
            chunk["depth"] = 0
            for parent in chunks:
                if parent is chunk:
                    continue
                inner_start = parent["data_offset"]
                inner_end = parent["offset"] + parent["size"]
                if inner_start <= chunk["offset"] < inner_end:
                    chunk["depth"] = parent["depth"] + 1
                    break
        self.chunks = chunks

    def _find_chunk(self, tag):
        for chunk in self.chunks:
            if chunk["tag"] == tag:
                return chunk
        return None

    def _extract_ntbl(self):
        chunk = self._find_chunk("NTBL")
        if chunk is None:
            return
        raw = self.data[chunk["data_offset"] + 4:chunk["offset"] + chunk["size"]]
        parts = raw.split(b"\x00")
        names = []
        for part in parts:
            try:
                text = part.decode("latin1")
            except UnicodeDecodeError:
                continue
            if len(text) > 1 and text.isprintable():
                names.append(text)
        self.ntbl_strings = names

    def _extract_textures(self):
        """Extraire les noms de textures depuis NTBL/TREF et decoder les DDS embarques."""
        self.textures = []
        for chunk in self.chunks:
            if chunk["tag"] == b"NTBL":
                try:
                    text_data = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
                    num_textures = len(text_data) // 4
                    for i in range(num_textures):
                        offset = struct.unpack_from("<I", text_data, i * 4)[0]
                        if 0 < offset < len(self.data) - 4:
                            end = self.data.index(0, offset) if 0 in self.data[offset:] else len(self.data)
                            name = self.data[offset:end].decode("latin1", errors="ignore").rstrip("\x00")
                            if name and name not in [t["name"] for t in self.textures]:
                                self.textures.append({"index": len(self.textures), "name": name, "image": None,
                                                      "width": 0, "height": 0, "format": "DDS"})
                except (struct.error, ValueError, IndexError):
                    pass
            elif chunk["tag"] == b"TREF":
                try:
                    text_data = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
                    num_textures = len(text_data) // 4
                    for i in range(num_textures):
                        offset = struct.unpack_from("<I", text_data, i * 4)[0]
                        if 0 < offset < len(self.data) - 4:
                            end = self.data.index(0, offset) if 0 in self.data[offset:] else len(self.data)
                            name = self.data[offset:end].decode("latin1", errors="ignore").rstrip("\x00")
                            if name and name not in [t["name"] for t in self.textures]:
                                self.textures.append({"index": len(self.textures), "name": name, "image": None,
                                                      "width": 0, "height": 0, "format": "DDS"})
                except (struct.error, ValueError, IndexError):
                    pass

        self._decode_embedded_dds()
        self.apply_texture_mapping()

    def _decode_embedded_dds(self):
        """Cherche et decode les blocs DDS (DXT1/DXT5) embarques dans le fichier."""
        data = self.data
        pos = 0
        tex_idx = 0
        while pos + 128 <= len(data):
            if data[pos:pos + 4] != b"DDS ":
                pos += 1
                continue
            try:
                height = struct.unpack_from("<I", data, pos + 12)[0]
                width = struct.unpack_from("<I", data, pos + 16)[0]
                mip_count = struct.unpack_from("<I", data, pos + 28)[0]
                fourcc = data[pos + 84:pos + 88]
            except struct.error:
                pos += 1
                continue
            if fourcc not in (b"DXT1", b"DXT5"):
                pos += 1
                continue
            block_bytes = DXT_BLOCK_BYTES[fourcc]
            total_size = 0
            w, h = width, height
            for _ in range(max(1, mip_count)):
                if w == 0:
                    w = 1
                if h == 0:
                    h = 1
                blocks_w = max(1, (w + 3) // 4)
                blocks_h = max(1, (h + 3) // 4)
                total_size += blocks_w * blocks_h * block_bytes
                w //= 2
                h //= 2
            dds_data = data[pos:pos + 128 + total_size]
            image = self._decode_dxt_to_image(dds_data, width, height, fourcc, mip_count)
            if tex_idx < len(self.textures):
                self.textures[tex_idx]["image"] = image
                self.textures[tex_idx]["width"] = width
                self.textures[tex_idx]["height"] = height
                self.textures[tex_idx]["format"] = fourcc.decode("latin1")
            else:
                self.textures.append({
                    "index": tex_idx,
                    "name": f"texture_{tex_idx}",
                    "image": image,
                    "width": width,
                    "height": height,
                    "format": fourcc.decode("latin1"),
                })
            tex_idx += 1
            pos += 128 + total_size
    @staticmethod
    def _decode_dxt_to_image(dds_data, width, height, fourcc, mip_count):
        """Decode un bloc DDS DXT1/DXT5 en image PIL RGBA (premier mipmap uniquement)."""
        data = dds_data[128:]
        block_bytes = DXT_BLOCK_BYTES[fourcc]
        blocks_w = max(1, (width + 3) // 4)
        blocks_h = max(1, (height + 3) // 4)
        pixels = np.zeros((height, width, 4), dtype=np.uint8)

        def rgb565_to_888(v):
            r = ((v >> 11) & 0x1F) * 255 // 31
            g = ((v >> 5) & 0x3F) * 255 // 63
            b = (v & 0x1F) * 255 // 31
            return (r, g, b)

        if fourcc == b"DXT1":
            for by in range(blocks_h):
                for bx in range(blocks_w):
                    block_offset = (by * blocks_w + bx) * block_bytes
                    block = data[block_offset:block_offset + 8]
                    if len(block) < 8:
                        continue
                    c0 = block[0] | (block[1] << 8)
                    c1 = block[2] | (block[3] << 8)
                    codes = block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24)
                    color0 = rgb565_to_888(c0)
                    color1 = rgb565_to_888(c1)
                    if c0 > c1:
                        colors = [
                            color0,
                            color1,
                            ((2 * color0[0] + color1[0]) // 3, (2 * color0[1] + color1[1]) // 3, (2 * color0[2] + color1[2]) // 3),
                            ((color0[0] + 2 * color1[0]) // 3, (color0[1] + 2 * color1[1]) // 3, (color0[2] + 2 * color1[2]) // 3),
                        ]
                    else:
                        colors = [
                            color0,
                            color1,
                            ((color0[0] + color1[0]) // 2, (color0[1] + color1[1]) // 2, (color0[2] + color1[2]) // 2),
                            (0, 0, 0),
                        ]
                    for py in range(4):
                        for px in range(4):
                            idx = (codes >> (2 * (py * 4 + px))) & 0x03
                            ox = bx * 4 + px
                            oy = by * 4 + py
                            if ox < width and oy < height:
                                c = colors[idx]
                                alpha = 0 if (c0 <= c1 and idx == 3) else 255
                                pixels[oy, ox] = [c[0], c[1], c[2], alpha]
        elif fourcc == b"DXT5":
            for by in range(blocks_h):
                for bx in range(blocks_w):
                    block_offset = (by * blocks_w + bx) * block_bytes
                    block = data[block_offset:block_offset + 16]
                    if len(block) < 16:
                        continue
                    alpha0 = block[0]
                    alpha1 = block[1]
                    alpha_codes = int.from_bytes(block[2:8], "little")
                    c0 = block[8] | (block[9] << 8)
                    c1 = block[10] | (block[11] << 8)
                    codes = block[12] | (block[13] << 8) | (block[14] << 16) | (block[15] << 24)
                    color0 = rgb565_to_888(c0)
                    color1 = rgb565_to_888(c1)
                    if c0 > c1:
                        colors = [
                            color0,
                            color1,
                            ((2 * color0[0] + color1[0]) // 3, (2 * color0[1] + color1[1]) // 3, (2 * color0[2] + color1[2]) // 3),
                            ((color0[0] + 2 * color1[0]) // 3, (color0[1] + 2 * color1[1]) // 3, (color0[2] + 2 * color1[2]) // 3),
                        ]
                    else:
                        colors = [
                            color0,
                            color1,
                            ((color0[0] + color1[0]) // 2, (color0[1] + color1[1]) // 2, (color0[2] + color1[2]) // 2),
                            (0, 0, 0),
                        ]
                    alphas = [alpha0, alpha1]
                    if alpha0 > alpha1:
                        for i in range(6):
                            alphas.append(((7 - i) * alpha0 + i * alpha1) // 7)
                    else:
                        for i in range(6):
                            alphas.append(((6 - i) * alpha0 + i * alpha1) // 6)
                        alphas.append(0)
                        alphas.append(255)
                    for py in range(4):
                        for px in range(4):
                            bit_index = py * 4 + px
                            acode = (alpha_codes >> (3 * bit_index)) & 0x07
                            a = alphas[acode]
                            ccode = (codes >> (2 * bit_index)) & 0x03
                            ox = bx * 4 + px
                            oy = by * 4 + py
                            if ox < width and oy < height:
                                c = colors[ccode]
                                pixels[oy, ox] = [c[0], c[1], c[2], a]
        return Image.fromarray(pixels, "RGBA")
    def _parse_tref(self):
        """Parse le chunk TREF (references textures vers dessins)."""
        chunk = self._find_chunk("TREF")
        if chunk is None:
            return {}
        raw = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
        refs = {}
        pos = 0
        while pos + 8 <= len(raw):
            draw_idx = struct.unpack_from("<I", raw, pos)[0]
            tex_idx = struct.unpack_from("<I", raw, pos + 4)[0]
            if draw_idx not in refs:
                refs[draw_idx] = []
            refs[draw_idx].append(tex_idx)
            pos += 8
        return refs

    def _parse_ms00(self):
        """Parse le chunk MS00 (proprietes materiel) et retourne une liste de
        dictionnaires par materiau trouve."""
        chunk = self._find_chunk("MS00")
        if chunk is None:
            return []
        raw = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
        materials = []
        pos = 0
        while pos + 12 <= len(raw):
            count = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            if count == 0:
                break
            end = min(pos + count * 12, len(raw))
            entries = []
            p2 = pos
            while p2 + 12 <= end:
                flag = struct.unpack_from("<I", raw, p2)[0]
                flags = struct.unpack_from("<I", raw, p2 + 4)[0]
                index = struct.unpack_from("<I", raw, p2 + 8)[0]
                entries.append({"flag": flag, "flags": flags, "index": index})
                p2 += 12
            materials.append(entries)
            pos = end
        return materials

    def apply_texture_mapping(self):
        """Construit le mapping associe a chaque draw sa texture (si disponible).

        Retourne un dict {draw_index: [texture_index]} stocke dans
        self.texture_refs, et un dictionnaire de materiaux.
        """
        tref = self._parse_tref()
        ms00 = self._parse_ms00()
        self.texture_refs = tref
        return tref, ms00

    def _parse_gsnh(self):
        """Parse le chunk GSNH (nom du squelette/character name)."""
        chunk = self._find_chunk("GSNH")
        if chunk is None:
            return None
        raw = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
        try:
            name = raw.split(b"\x00")[0].decode("latin1", errors="ignore")
        except Exception:
            name = None
        return name

    def _parse_pntr(self):
        """Parse le chunk PNTR (pointeurs de bones/bone pointers)."""
        chunk = self._find_chunk("PNTR")
        if chunk is None:
            return []
        raw = self.data[chunk["data_offset"]:chunk["offset"] + chunk["size"]]
        num_entries = len(raw) // 4
        indices = []
        for i in range(num_entries):
            idx = struct.unpack_from("<I", raw, i * 4)[0]
            indices.append(idx)
        return indices
    def _extract_geometry(self):
        """Extraire la geometrie (vertices, faces, UVs) depuis la section finale du fichier."""
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.faces = np.empty((0, 3), dtype=np.int32)
        self.uvs = np.empty((0, 2), dtype=np.float32)
        self.draws = []
        self.vertex_buffers = []
        self.geometry_offset = None

        geometry_data = self._find_geometry_section()
        if geometry_data is None:
            self.warnings.append("Section geometrie introuvable en fin de fichier.")
            return

        mesh_data = geometry_data[0]
        self.geometry_offset = geometry_data[1]
        try:
            pos = 0
            nVB = struct.unpack_from("<H", mesh_data, pos)[0]
            pos += 2
            vertex_buffers = []
            for _ in range(nVB):
                if pos + 4 > len(mesh_data):
                    break
                byte_size = struct.unpack_from("<I", mesh_data, pos)[0]
                pos += 4
                if pos + byte_size > len(mesh_data):
                    break
                vertex_buffers.append({
                    "index": len(vertex_buffers),
                    "offset": self.geometry_offset + pos,
                    "size": byte_size,
                })
                pos += byte_size

            nIB = struct.unpack_from("<H", mesh_data, pos)[0]
            pos += 2
            ib_bytes = struct.unpack_from("<I", mesh_data, pos)[0]
            pos += 4
            num_indices = ib_bytes // 2
            indices = np.frombuffer(
                mesh_data,
                dtype="<u2",
                count=num_indices,
                offset=pos,
            ).astype(np.int64)
        except (struct.error, IndexError, ValueError):
            self.warnings.append("Section geometrie invalide.")
            return

        parts, bone_names, bone_world = self._parse_gsnh_parts()
        self.vertex_buffers = vertex_buffers
        draw_run = self._read_draw_records(len(indices), expected_parts=len(parts))
        if not draw_run:
            self.warnings.append("Aucun enregistrement de draw trouve dans DISP.")
            return

        if parts and any(bone >= 0 for bone in parts):
            if len(parts) != len(draw_run):
                self.warnings.append(
                    f"GSNH: {len(parts)} parties pour {len(draw_run)} draws ; mapping partiel."
                )
        else:
            parts = []
        self.skeleton_bones = bone_names
        self.skeleton_parts = parts

        vertices = []
        faces = []
        uvs = []
        offset = 0
        cache = {}

        for part_index, record in enumerate(draw_run):
            block = self._pick_vertex_buffer(vertex_buffers, record)
            if block is None:
                continue
            stride = record["stride"]
            count = block["size"] // stride
            if count <= 0 or record["vb_offset"] + record["vb_count"] > count:
                continue

            key = (block["offset"], stride)
            if key not in cache:
                cache[key] = self._decode_vertices(block["offset"], count, stride)
            positions, vnormals, vuvs = cache[key]

            start = record["vb_offset"]
            stop = start + record["vb_count"]
            sub_pos = positions[start:stop]
            sub_normals = vnormals[start:stop]
            sub_uvs = vuvs[start:stop]

            bone = parts[part_index] if part_index < len(parts) else -1
            if 0 <= bone < len(bone_world):
                sub_pos, sub_normals = self._apply_bone_transform(
                    sub_pos, sub_normals, bone_world[bone]
                )

            triangles = self._strip_triangles(
                indices, record["ib_offset"], record["index_count"], record["vb_count"]
            )
            if len(triangles) == 0:
                continue

            vertices.append(sub_pos)
            uvs.append(sub_uvs)
            faces.append(triangles + offset)
            offset += len(sub_pos)

            dot = self._normal_agreement(sub_pos, sub_normals, triangles)
            self.draws.append({
                "vertex_buffer": record["vb_index"],
                "stride": stride,
                "vertex_offset": record["vb_offset"],
                "vertex_count": record["vb_count"],
                "index_offset": record["ib_offset"],
                "index_count": record["index_count"],
                "triangles": len(triangles),
                "stored_normal_dot": round(float(dot), 4),
                "bone": bone,
                "bone_name": bone_names[bone] if 0 <= bone < len(bone_names) else None,
            })

        if not vertices:
            self.warnings.append("Aucun triangle decode.")
            return

        all_vertices = np.vstack(vertices)
        all_faces = np.vstack(faces)
        all_uvs = np.vstack(uvs)

        all_faces = self._dedupe_faces(all_vertices, all_faces)

        self.vertices = all_vertices
        self.faces = all_faces
        self.uvs = all_uvs

        used = np.unique(all_faces)
        self.vertices = all_vertices[used]
        remap = np.full(len(all_vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.faces = remap[all_faces]
        self.uvs = all_uvs[used]

        try:
            self.mesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                process=True,
            )
            self.vertices = np.asarray(self.mesh.vertices)
            self.faces = np.asarray(self.mesh.faces)
        except Exception as exc:
            self.warnings.append(f"Creation du maillage impossible : {exc}")

    def _parse_gsnh_parts(self):
        """Lire le squelette GSNH et la table layer -> os -> parties.

        Les pointeurs sont des offsets signes relatifs au champ qui les contient.
        Retourne (part_bones, bone_names, bone_world).
        """
        chunk = self._find_chunk("GSNH")
        if chunk is None:
            return [], [], []

        data = self.data
        g = chunk["offset"]

        def i32(addr):
            return struct.unpack_from("<i", data, addr)[0]

        def u32(addr):
            return struct.unpack_from("<I", data, addr)[0]

        def f32_matrix(addr):
            return np.array(struct.unpack_from("<16f", data, addr), dtype=np.float64).reshape(4, 4)

        def cstr(addr):
            end = data.find(b"\x00", addr)
            if end < 0:
                return ""
            return data[addr:end].decode("latin1", errors="ignore")

        try:
            p = g + 0x3C
            mesh_meta = p + i32(p)
            p += 4
            p += 0x130
            number_bones = u32(p)
            p += 4
            bones_off = p + i32(p)
            p += 4
            bones2_off = p + i32(p)
            p += 4
            p += 4  # bones3 (inverse bind), inutilise
            p += 24
            number_layers = u32(p)
            p += 4
            layers_off = p + i32(p)
        except struct.error:
            return [], [], []

        if not (0 < number_bones <= 1024 and 0 <= number_layers <= 256):
            return [], [], []
        if not (0 <= bones_off < len(data) and 0 <= bones2_off < len(data)):
            return [], [], []
        if not (0 <= layers_off < len(data)):
            return [], [], []

        names = []
        parents = []
        try:
            for i in range(number_bones):
                record = bones_off + i * 0x60
                name_field = record + 0x4C
                name_off = name_field + i32(name_field)
                if 0 <= name_off < len(data):
                    names.append(cstr(name_off))
                else:
                    names.append(f"bone{i}")
                parents.append(struct.unpack_from("<b", data, name_field + 4)[0])
            local = [f32_matrix(bones2_off + i * 0x40) for i in range(number_bones)]
        except struct.error:
            return [], [], []

        world = [None] * number_bones

        def resolve(index, stack):
            if world[index] is not None:
                return world[index]
            parent = parents[index]
            if 0 <= parent < number_bones and parent != index and parent not in stack:
                world[index] = local[index] @ resolve(parent, stack | {index})
            else:
                world[index] = local[index]
            return world[index]

        for i in range(number_bones):
            resolve(i, set())

        def read_pair(pair, default_bone):
            if not (0 <= pair <= len(data) - 0x10):
                return 0
            matrix_field = pair + 8
            matrix_target = matrix_field + i32(matrix_field)
            if not (0 <= matrix_target <= len(data) - 0xC0):
                return 0
            parts_field = matrix_target + 0xB0
            parts_target = parts_field + i32(parts_field)
            if not (0 <= parts_target <= len(data) - 4):
                return 0
            count = i32(parts_target)
            if count <= 0 or count > 100_000:
                return 0
            return count

        parts = []
        try:
            for layer in range(number_layers):
                base = layers_off + layer * 20
                pointers = {}
                for slot in range(1, 5):
                    field = base + slot * 4
                    value = i32(field)
                    if value != 0:
                        pointers[slot] = field + value

                for slot in (1, 2, 3, 4):
                    if slot not in pointers:
                        continue
                    if slot in (1, 3):
                        table = pointers[slot]
                        if not (0 <= table <= len(data) - number_bones * 4):
                            continue
                        for bone in range(number_bones):
                            field = table + bone * 4
                            value = i32(field)
                            if value == 0:
                                continue
                            count = read_pair(field + value, bone)
                            if count:
                                parts.extend([bone] * count)
                    else:
                        count = read_pair(pointers[slot], -1)
                        if count:
                            parts.extend([-1] * count)
        except struct.error:
            return [], [], []

        return parts, names, world

    def _read_draw_records(self, index_total, expected_parts=None):
        """Retrouver les enregistrements DISP valides (0x38 octets).

        Les draws sont normalement une suite contigue, mais certains modeles
        (Indiana Jones) ont quelques parties isolees avant la suite principale.
        Si leur nombre total correspond exactement aux parties GSNH, on les
        conserve toutes dans l'ordre du fichier.
        """
        chunk = self._find_chunk("DISP")
        if chunk is None:
            return []

        data = self.data
        start = chunk["data_offset"]
        end = chunk["offset"] + chunk["size"] - DRAW_RECORD_SIZE
        records = {}
        for pos in range(start, max(start, end)):
            if pos + DRAW_RECORD_SIZE > len(data):
                break
            try:
                attributes = struct.unpack_from("<I", data, pos)[0]
                index_count = struct.unpack_from("<I", data, pos + 4)[0]
                stride = struct.unpack_from("<H", data, pos + 8)[0]
                vb_offset = struct.unpack_from("<I", data, pos + 0x14)[0]
                vb_count = struct.unpack_from("<I", data, pos + 0x18)[0]
                ib_offset = struct.unpack_from("<I", data, pos + 0x1C)[0]
                vb_index = struct.unpack_from("<I", data, pos + 0x24)[0]
            except struct.error:
                break
            if not (1 <= attributes <= 16):
                continue
            if not (1 <= index_count <= 1_000_000):
                continue
            if stride not in VERTEX_LAYOUTS:
                continue
            if vb_index > 8 or vb_count <= 0 or vb_count > 4_000_000:
                continue
            if ib_offset + index_count > index_total + 8:
                continue
            records[pos] = {
                "offset": pos,
                "attributes": attributes,
                "index_count": index_count,
                "stride": stride,
                "vb_offset": vb_offset,
                "vb_count": vb_count,
                "ib_offset": ib_offset,
                "vb_index": vb_index,
            }

        runs = []
        pos = start
        while pos < end:
            if pos not in records:
                pos += 1
                continue
            run = [records[pos]]
            nxt = pos + DRAW_RECORD_SIZE
            while nxt in records:
                previous = run[-1]
                current = records[nxt]
                gap = current["ib_offset"] - (previous["ib_offset"] + previous["index_count"])
                if not (0 <= gap <= 8):
                    break
                run.append(current)
                nxt += DRAW_RECORD_SIZE
            runs.append(run)
            pos = nxt

        if not runs:
            return []

        valid = [records[pos] for pos in sorted(records)]
        if expected_parts and len(valid) == expected_parts:
            return valid
        return max(runs, key=lambda run: sum(rec["index_count"] for rec in run))

    @staticmethod
    def _pick_vertex_buffer(blocks, record):
        stride = record["stride"]
        index = record["vb_index"]
        if index < len(blocks):
            block = blocks[index]
            if block["size"] % stride == 0 and record["vb_offset"] + record["vb_count"] <= block["size"] // stride:
                return block
        for block in blocks:
            if block["size"] % stride == 0 and record["vb_offset"] + record["vb_count"] <= block["size"] // stride:
                return block
        return None

    def _decode_vertices(self, offset, count, stride):
        layout = VERTEX_LAYOUTS.get(stride)
        if layout is None:
            return (
                np.zeros((count, 3), dtype=np.float32),
                np.zeros((count, 3), dtype=np.float32),
                np.zeros((count, 2), dtype=np.float32),
            )
        raw = np.frombuffer(
            self.data, dtype=np.uint8, count=count * stride, offset=offset
        ).reshape(count, stride)
        positions = raw[:, layout["pos"]:layout["pos"] + 12].copy().view("<f4").reshape(count, 3).astype(np.float32)
        normals = raw[:, layout["normal"]:layout["normal"] + 3].astype(np.float32) / 127.5 - 1.0
        uv_offset = layout["uv"]
        if uv_offset + 8 <= stride:
            uvs = raw[:, uv_offset:uv_offset + 8].copy().view("<f4").reshape(count, 2).astype(np.float32)
        else:
            uvs = np.zeros((count, 2), dtype=np.float32)
        return positions, normals, uvs

    @staticmethod
    def _apply_bone_transform(positions, normals, matrix):
        if matrix is None:
            return positions, normals
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4):
            return positions, normals
        p = np.asarray(positions, dtype=np.float64)
        if len(p):
            p_h = np.concatenate((p, np.ones((len(p), 1), dtype=np.float64)), axis=1)
            p = (p_h @ m)[:, :3]
        n = np.asarray(normals, dtype=np.float64) @ m[:3, :3]
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(length, 1e-12)
        return p.astype(np.float32), n.astype(np.float32)

    @staticmethod
    def _strip_triangles(indices, ib_offset, index_count, vb_count):
        segment = indices[ib_offset:ib_offset + index_count]
        if len(segment) < 3:
            return np.empty((0, 3), dtype=np.int64)
        a = segment[:-2]
        b = segment[1:-1]
        c = segment[2:]
        flip = (np.arange(len(a)) % 2) == 1
        triangles = np.column_stack((a, b, c))
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        valid = (a != b) & (b != c) & (a != c)
        triangles = triangles[valid]
        valid = (triangles >= 0).all(axis=1) & (triangles < vb_count).all(axis=1)
        return triangles[valid]

    @staticmethod
    def _normal_agreement(positions, normals, triangles):
        if len(triangles) == 0:
            return 0.0
        v0 = positions[triangles[:, 0]]
        v1 = positions[triangles[:, 1]]
        v2 = positions[triangles[:, 2]]
        face = np.cross(v1 - v0, v2 - v0)
        length = np.linalg.norm(face, axis=1)
        valid = length > 1e-12
        if not valid.any():
            return 0.0
        face = face[valid] / length[valid, None]
        vertex = normals[triangles[valid]].mean(axis=1)
        vertex /= np.maximum(np.linalg.norm(vertex, axis=1, keepdims=True), 1e-12)
        return float(np.clip((face * vertex).sum(axis=1), -1.0, 1.0).mean())

    @staticmethod
    def _dedupe_faces(vertices, faces):
        if len(faces) == 0:
            return faces
        keys = np.round(vertices[faces], 4)
        order = np.lexsort((keys[:, :, 2], keys[:, :, 1], keys[:, :, 0]), axis=1)
        keys = np.take_along_axis(keys, order[:, :, None], axis=1)
        packed = np.ascontiguousarray(keys).view(np.uint8).reshape(len(faces), 36)
        _, unique = np.unique(packed, axis=0, return_index=True)
        return faces[np.sort(unique)]

    def _find_geometry_section(self):
        """Trouver la section geometrie en fin de fichier.

        Retourne (data, offset) ou None.
        """
        data = self.data
        file_len = len(data)

        last_dds_end = 16
        pos = 16
        while pos + 4 <= file_len:
            if data[pos:pos + 4] == b"DDS " and pos + 128 <= file_len:
                try:
                    height = struct.unpack_from("<I", data, pos + 12)[0]
                    width = struct.unpack_from("<I", data, pos + 16)[0]
                    mip_count = struct.unpack_from("<I", data, pos + 28)[0]
                    fourcc = data[pos + 84:pos + 88]
                    if fourcc in (b"DXT1", b"DXT5"):
                        block_bytes = DXT_BLOCK_BYTES[fourcc]
                        total_size = 0
                        w, h = width, height
                        for _ in range(max(1, mip_count)):
                            if w == 0:
                                w = 1
                            if h == 0:
                                h = 1
                            blocks_w = max(1, (w + 3) // 4)
                            blocks_h = max(1, (h + 3) // 4)
                            total_size += blocks_w * blocks_h * block_bytes
                            w //= 2
                            h //= 2
                        last_dds_end = pos + 128 + total_size
                        pos = last_dds_end
                        continue
                except struct.error:
                    pass
            pos += 1

        candidates = [last_dds_end + i for i in range(0, 8)]
        best = None
        best_score = -1

        for cand in candidates:
            if cand + 2 > file_len:
                continue
            try:
                nVB = struct.unpack_from("<H", data, cand)[0]
                if nVB == 0 or nVB > 32:
                    continue
                p = cand + 2
                total_vb_size = 0
                ok = True
                for _ in range(nVB):
                    if p + 4 > file_len:
                        ok = False
                        break
                    byte_size = struct.unpack_from("<I", data, p)[0]
                    p += 4
                    if p + byte_size > file_len:
                        ok = False
                        break
                    total_vb_size += byte_size
                    p += byte_size
                if not ok:
                    continue
                if p + 2 > file_len:
                    continue
                nIB = struct.unpack_from("<H", data, p)[0]
                p += 2
                if p + 4 > file_len:
                    continue
                ib_bytes = struct.unpack_from("<I", data, p)[0]
                p += 4
                if p + ib_bytes != file_len:
                    continue
                score = total_vb_size
                if score > best_score:
                    best_score = score
                    best = (data[cand:], cand)
            except struct.error:
                continue

        if best is None:
            for cand in range(file_len - 2, 0, -1):
                try:
                    nVB = struct.unpack_from("<H", data, cand)[0]
                    if nVB == 0 or nVB > 32:
                        continue
                    p = cand + 2
                    total_vb_size = 0
                    ok = True
                    for _ in range(nVB):
                        if p + 4 > file_len:
                            ok = False
                            break
                        byte_size = struct.unpack_from("<I", data, p)[0]
                        p += 4
                        if p + byte_size > file_len:
                            ok = False
                            break
                        total_vb_size += byte_size
                        p += byte_size
                    if not ok:
                        continue
                    if p + 2 > file_len:
                        continue
                    nIB = struct.unpack_from("<H", data, p)[0]
                    p += 2
                    if p + 4 > file_len:
                        continue
                    ib_bytes = struct.unpack_from("<I", data, p)[0]
                    p += 4
                    if p + ib_bytes != file_len:
                        continue
                    score = total_vb_size
                    if score > best_score:
                        best_score = score
                        best = (data[cand:], cand)
                except struct.error:
                    continue

        return best

    def _deduplicate_vertices(self):
        """Supprimer les vertices dupliques (cas biplane : deux copies identiques dans le VB 0)."""
        if len(self.vertices) == 0:
            return

        positions = self.vertices[:, :3]
        rounded = np.round(positions, 4)

        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)

        if len(unique_indices) == len(self.vertices):
            return

        self.warnings.append(f"De-doublonnage : {len(self.vertices)} -> {len(unique_indices)} vertices")

        remap = np.zeros(len(self.vertices), dtype=np.int32)
        for new_idx, old_idx in enumerate(unique_indices):
            remap[old_idx] = new_idx

        self.vertices = self.vertices[unique_indices]
        if len(self.uvs) == len(remap):
            self.uvs = self.uvs[unique_indices]

        new_faces = []
        for face in self.faces:
            nf = [remap[f] for f in face]
            if nf[0] != nf[1] and nf[1] != nf[2] and nf[0] != nf[2]:
                new_faces.append(nf)
        self.faces = np.array(new_faces, dtype=np.int32) if new_faces else np.empty((0, 3), dtype=np.int32)
    def _build_stats(self):
        """Construire le dictionnaire de statistiques (API stable de l'UI)."""
        bounds = None
        if self.mesh is not None and len(self.mesh.vertices):
            bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        elif len(self.vertices) > 0:
            bounds = np.asarray(
                [self.vertices[:, :3].min(axis=0), self.vertices[:, :3].max(axis=0)],
                dtype=np.float64,
            )

        self.stats = {
            "name": self.name,
            "file_size": f"{self.size / 1024:.1f} Ko",
            "num_chunks": len(self.chunks),
            "num_textures": len(self.textures),
            "num_draws": len(self.draws),
            "num_vertex_buffers": len(self.vertex_buffers),
            "num_vertices": len(self.vertices),
            "num_faces": len(self.faces),
            "ntbl_count": len(self.ntbl_strings),
            "geometry_offset": (
                f"0x{self.geometry_offset:06X}" if self.geometry_offset is not None else "-"
            ),
            "bounds": bounds,
            "has_mesh": self.mesh is not None,
        }

    def _build_mesh(self):
        """Construire un objet trimesh.Trimesh a partir des donnees extraites."""
        if self.mesh is not None:
            return
        if len(self.vertices) == 0 or len(self.faces) == 0:
            self.mesh = None
            return
        self.mesh = trimesh.Trimesh(
            vertices=self.vertices[:, :3],
            faces=self.faces,
            process=True,
        )
        if len(self.uvs) > 0 and len(self.uvs) == len(self.vertices):
            self.mesh.visual = trimesh.visual.TextureVisuals(uv=self.uvs)

    def export_glb(self, path):
        """Exporter le modele au format GLB (glTF 2.0 binaire)."""
        self._build_mesh()
        if self.mesh is None:
            raise ValueError("Aucun maillage a exporter.")
        self.mesh.export(path)

    def export_obj(self, path):
        """Exporter le modele au format OBJ."""
        self._build_mesh()
        if self.mesh is None:
            raise ValueError("Aucun maillage a exporter.")
        self.mesh.export(path)

    def export_textures_zip(self, path):
        """Exporter les textures decodees dans un fichier ZIP (PNG)."""
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for tex in self.textures:
                if tex["image"] is not None:
                    buf = io.BytesIO()
                    tex["image"].save(buf, format="PNG")
                    zf.writestr(f"{tex['name']}.png", buf.getvalue())
                else:
                    zf.writestr(f"{tex['name']}.txt", f"Texture {tex['name']} (pas d'image decodee)\n")
