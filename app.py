"""
Application Gradio pour la visualisation et l'exportation des modeles 3D LEGO (.ghg / NU20).
Visualiseur WebGL (Three.js via Model3D), galerie de textures, inspection des chunks et exports.
"""

import glob
import os
import zipfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from ghg_parser import GHGModel

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(WORKSPACE_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


def list_workspace_ghg_files():
    files = glob.glob(os.path.join(WORKSPACE_DIR, "*.ghg"))
    return sorted(os.path.basename(path) for path in files)


def _export_arrays(model):
    return (
        getattr(model, "_export_vertices", model.vertices),
        getattr(model, "_export_faces", model.faces),
        getattr(model, "_export_uvs", model.uvs),
        getattr(model, "_export_face_materials", model.face_materials),
    )


def build_material_rows(model):
    vertices, faces, uvs, face_materials = _export_arrays(model)
    rows = []
    for material in model.materials:
        index = material.get("index", -1)
        face_indices = np.where(face_materials == index)[0]
        uv_min = "-"
        uv_max = "-"
        if len(face_indices) and len(uvs) == len(vertices):
            used = np.unique(faces[face_indices])
            if len(used):
                material_uvs = uvs[used]
                uv_min = "({:.3f}, {:.3f})".format(*material_uvs.min(axis=0))
                uv_max = "({:.3f}, {:.3f})".format(*material_uvs.max(axis=0))

        texture_index = material.get("texture", -1)
        texture_label = "-"
        if 0 <= texture_index < len(model.textures):
            texture = model.textures[texture_index]
            texture_label = (
                f"#{texture_index} {texture['width']}x{texture['height']}"
            )

        diffuse = material.get("diffuse", (1.0, 1.0, 1.0, 1.0))
        diffuse_label = ", ".join(f"{channel:.3f}" for channel in diffuse[:3])
        alpha = diffuse[3] if len(diffuse) > 3 else 1.0
        if alpha <= 0.0:
            alpha = 1.0
        rows.append([
            index,
            material.get("id", -1),
            texture_label,
            diffuse_label,
            float(alpha),
            len(face_indices),
            uv_min,
            uv_max,
        ])
    return rows


def build_uv_previews(model):
    vertices, faces, uvs, face_materials = _export_arrays(model)
    if len(uvs) != len(vertices):
        return []
    items = []
    for material in model.materials:
        index = material.get("index", -1)
        face_indices = np.where(face_materials == index)[0]
        if len(face_indices) == 0:
            continue

        texture_index = material.get("texture", -1)
        image = None
        if 0 <= texture_index < len(model.textures):
            image = model.textures[texture_index].get("image")

        if image is not None:
            canvas = image.convert("RGBA").copy()
            canvas.thumbnail((512, 512), Image.Resampling.LANCZOS)
        else:
            canvas = Image.new("RGBA", (512, 512), (40, 42, 52, 255))

        width, height = canvas.size
        draw = ImageDraw.Draw(canvas, "RGBA")
        used = np.unique(faces[face_indices])
        remap = {int(vertex): position for position, vertex in enumerate(used)}
        group_uvs = uvs[used]
        line_color = (255, 70, 70, 210) if image is not None else (255, 200, 60, 220)

        for face in faces[face_indices]:
            points = []
            for vertex in face:
                u, v = group_uvs[remap[int(vertex)]]
                points.append((float(u) * width, float(v) * height))
            draw.line(points + [points[0]], fill=line_color, width=1)

        caption = f"Mat {index} - {len(face_indices)} faces"
        if texture_index >= 0:
            caption += f" - Tex #{texture_index}"
        items.append((canvas, caption))
    return items


def _empty_result(message):
    return None, message, [], [], [], "", None, None, None, [], []


def load_and_process_model(selected_file, uploaded_file):
    target_path = None
    if uploaded_file is not None:
        target_path = uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    elif selected_file:
        target_path = os.path.join(WORKSPACE_DIR, selected_file)

    if not target_path or not os.path.exists(target_path):
        return _empty_result("Aucun fichier selectionne ou fichier introuvable.")

    try:
        model = GHGModel(target_path)
        base_name = model.name

        glb_path = None
        obj_zip_path = None
        if model.mesh is not None:
            glb_path = os.path.join(EXPORTS_DIR, f"{base_name}.glb")
            model.export_glb(glb_path)
            obj_path = os.path.join(EXPORTS_DIR, f"{base_name}.obj")
            obj_files = model.export_obj(obj_path)
            obj_zip_path = os.path.join(EXPORTS_DIR, f"{base_name}_obj.zip")
            with zipfile.ZipFile(obj_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for obj_file in obj_files:
                    zf.write(obj_file, arcname=os.path.basename(obj_file))

        zip_path = os.path.join(EXPORTS_DIR, f"{base_name}_textures.zip")
        model.export_textures_zip(zip_path)

        gallery_items = []
        for texture in model.textures:
            if texture["image"] is not None:
                caption = (
                    f"Tex #{texture['index']} "
                    f"({texture['width']}x{texture['height']} {texture['format']})"
                )
                gallery_items.append((texture["image"], caption))

        stats = model.stats
        bounds = stats.get("bounds")
        dimensions = "-"
        if bounds is not None:
            size = bounds[1] - bounds[0]
            dimensions = f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m"

        info_md = f"""
### Statistiques du modele
* **Nom :** `{model.name}`
* **Taille fichier :** `{stats['file_size']}`
* **Sommets :** `{stats['num_vertices']:,}`
* **Triangles :** `{stats['num_faces']:,}`
* **Sous-maillages (draws) :** `{stats['num_draws']}`
* **Vertex buffers :** `{stats['num_vertex_buffers']}`
* **Materiaux :** `{len(model.materials)}`
* **Textures extraites :** `{stats['num_textures']}`
* **Entrees NTBL :** `{stats['ntbl_count']}`
* **Section geometrie :** `{stats['geometry_offset']}`
* **Dimensions (X x Y x Z) :** `{dimensions}`
"""
        if model.warnings:
            info_md += "\n**Avertissements :**\n" + "\n".join(f"* {w}" for w in model.warnings)

        chunks_data = [
            [
                "  " * chunk["depth"] + chunk["tag"],
                chunk["hex_offset"],
                f"{chunk['size']:,}",
                f"{chunk['data_size']:,}",
                chunk["depth"],
            ]
            for chunk in model.chunks
        ]

        draw_rows = [
            [
                draw["vertex_buffer"],
                draw["stride"],
                draw["vertex_offset"],
                draw["vertex_count"],
                draw["index_offset"],
                draw["index_count"],
                draw["triangles"],
                draw["stored_normal_dot"],
                draw.get("material", -1),
                draw.get("texture", -1),
            ]
            for draw in model.draws
        ]

        material_rows = build_material_rows(model)
        uv_previews = build_uv_previews(model)

        ntbl_text = " - ".join(model.ntbl_strings) if model.ntbl_strings else "Aucun nom extrait."

        return (
            glb_path,
            info_md,
            gallery_items,
            chunks_data,
            draw_rows,
            ntbl_text,
            glb_path,
            obj_zip_path,
            zip_path,
            material_rows,
            uv_previews,
        )

    except Exception as exc:
        return _empty_result(f"Erreur lors du chargement : {exc}")


custom_css = """
.container { max-width: 1300px; margin: auto; }
.model3d-box { min-height: 520px; border-radius: 12px; background: #1a1b26; }
"""

with gr.Blocks(title="LEGO GHG 3D Explorer") as demo:
    gr.Markdown("""
    # LEGO Model Viewer & Exporter (moteur TT Games NU20)
    Visualisez en 3D, inspectez la structure binaire et exportez les modeles `.ghg` des jeux LEGO
    (*LEGO Indiana Jones*, *LEGO Star Wars*, *LEGO Batman*, etc.).
    """)

    with gr.Row():
        with gr.Column(scale=4):
            available_files = list_workspace_ghg_files()
            file_dropdown = gr.Dropdown(
                label="Modeles inclus dans le depot",
                choices=available_files,
                value=available_files[0] if available_files else None,
                interactive=True,
            )
        with gr.Column(scale=5):
            file_upload = gr.File(
                label="Ou televersez votre propre fichier .ghg",
                file_types=[".ghg", ".bin"],
                type="filepath",
            )
        with gr.Column(scale=3):
            btn_load = gr.Button("Charger & visualiser", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown("### Visualiseur 3D interactif")
            model_viewer = gr.Model3D(
                label="Modele 3D (rotation clic-gauche, pan clic-droit, zoom molette)",
                elem_classes=["model3d-box"],
                display_mode="solid",
            )
            with gr.Row():
                download_glb = gr.File(label="Telecharger .GLB", interactive=False)
                download_obj = gr.File(
                    label="Telecharger .OBJ + MTL (.ZIP)", interactive=False
                )
                download_tex = gr.File(label="Textures (.ZIP)", interactive=False)

        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.TabItem("Resume"):
                    model_info = gr.Markdown(
                        "Selectionnez ou televersez un modele pour commencer l'analyse."
                    )

                with gr.TabItem("Textures DDS"):
                    tex_gallery = gr.Gallery(
                        label="Textures embarquees decompressees en PNG",
                        columns=3,
                        rows=2,
                        height=420,
                        object_fit="contain",
                    )

                with gr.TabItem("Materiaux / UV"):
                    materials_table = gr.Dataframe(
                        headers=[
                            "Materiau",
                            "ID",
                            "Texture",
                            "Diffuse RVB",
                            "Alpha",
                            "Faces",
                            "UV min",
                            "UV max",
                        ],
                        datatype=[
                            "number",
                            "number",
                            "str",
                            "str",
                            "number",
                            "number",
                            "str",
                            "str",
                        ],
                        label="Materiaux, textures et emprise UV",
                        interactive=False,
                        wrap=True,
                    )
                    uv_gallery = gr.Gallery(
                        label="Apercu des UV projetes sur chaque texture",
                        columns=2,
                        rows=2,
                        height=380,
                        object_fit="contain",
                    )

                with gr.TabItem("Sous-maillages"):
                    draws_table = gr.Dataframe(
                        headers=[
                            "VB",
                            "Stride",
                            "VBO",
                            "Sommets",
                            "IBO",
                            "Indices",
                            "Triangles",
                            "Dot normales",
                            "Materiau",
                            "Texture",
                        ],
                        datatype=["number"] * 10,
                        label="Draws extraits de DISP",
                        interactive=False,
                    )

                with gr.TabItem("Noms (NTBL)"):
                    ntbl_display = gr.Textbox(
                        label="Noeuds, os et sous-maillages",
                        lines=12,
                        interactive=False,
                    )

                with gr.TabItem("Chunks binaires"):
                    chunks_table = gr.Dataframe(
                        headers=["Tag 4CC", "Offset", "Taille", "Donnees", "Profondeur"],
                        datatype=["str", "str", "str", "str", "number"],
                        label="Sections detectees dans le fichier GHG",
                        interactive=False,
                    )

    outputs = [
        model_viewer,
        model_info,
        tex_gallery,
        chunks_table,
        draws_table,
        ntbl_display,
        download_glb,
        download_obj,
        download_tex,
        materials_table,
        uv_gallery,
    ]

    btn_load.click(fn=load_and_process_model, inputs=[file_dropdown, file_upload], outputs=outputs)
    file_dropdown.change(
        fn=load_and_process_model, inputs=[file_dropdown, file_upload], outputs=outputs
    )


def main():
    print("Demarrage du serveur Gradio sur http://localhost:7860 ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="amber"),
        css=custom_css,
    )


if __name__ == "__main__":
    main()
