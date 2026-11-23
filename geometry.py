import gmsh
import numpy as np
import os

def load_and_mesh_step(step_filepath: str, mesh_size: float = 3.0):
    """
    Imports an AP214 .step file, generates a finite element mesh, 
    and extracts 0-indexed NumPy arrays in native STEP coordinates.
    """
    if not os.path.exists(step_filepath):
        raise FileNotFoundError(f"Could not find STEP file at: {step_filepath}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("step_imported_part")

    shape_dim_tags = gmsh.model.occ.importShapes(step_filepath)
    gmsh.model.occ.synchronize()

    volumes = [tag for dim, tag in shape_dim_tags if dim == 3]
    surfaces = [tag for dim, tag in shape_dim_tags if dim == 2]

    if volumes:
        domain_dim = 3
        domain_tag = volumes[0]
    elif surfaces:
        domain_dim = 2
        domain_tag = surfaces[0]
    else:
        raise ValueError("Could not find any valid 2D surfaces or 3D volumes in the STEP file.")

    # Configure Mesh Sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)

    if domain_dim == 3:
        gmsh.model.mesh.generate(3)
        elem_type = 4
        elem_name = "Tetrahedra"
    else:
        gmsh.model.mesh.generate(2)
        elem_type = 2
        elem_name = "Triangles"

    node_tags, node_coords_raw, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords_raw.reshape(-1, 3)

    tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

    _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)
    elements_raw = elem_node_tags.reshape(-1, elem_type)
    elements = np.vectorize(tag_to_idx.get)(elements_raw)

    meta = {
        "num_nodes": len(nodes),
        "num_elements": len(elements),
        "elem_type": elem_name,
        "domain_dim": domain_dim
    }

    gmsh.finalize()

    return {
        "nodes": nodes,
        "elements": elements,
        "meta": meta
    }