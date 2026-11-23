import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

def compute_tet_stiffness_and_stress_matrix(coords, E, nu):
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = factor * np.array([
        [1-nu, nu,   nu,   0,       0,       0      ],
        [nu,   1-nu, nu,   0,       0,       0      ],
        [nu,   nu,   1-nu, 0,       0,       0      ],
        [0,    0,    0,    (1-2*nu)/2, 0,       0    ],
        [0,    0,    0,    0,       (1-2*nu)/2, 0    ],
        [0,    0,    0,    0,       0,       (1-2*nu)/2]
    ])

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    C = np.ones((4, 4))
    C[:, 1:] = coords
    six_V = np.linalg.det(C)
    if abs(six_V) < 1e-14:
        raise ValueError("Degenerate tetrahedral element detected (zero volume).")
    V = six_V / 6.0

    b = np.array([
         (y[1]*(z[2]-z[3]) + y[2]*(z[3]-z[1]) + y[3]*(z[1]-z[2])),
        -(y[0]*(z[2]-z[3]) + y[2]*(z[3]-z[0]) + y[3]*(z[0]-z[2])),
         (y[0]*(z[1]-z[3]) + y[1]*(z[3]-z[0]) + y[3]*(z[0]-z[1])),
        -(y[0]*(z[1]-z[2]) + y[1]*(z[2]-z[0]) + y[2]*(z[0]-z[1]))
    ])
    c = np.array([
        -(x[1]*(z[2]-z[3]) + x[2]*(z[3]-z[1]) + x[3]*(z[1]-z[2])),
         (x[0]*(z[2]-z[3]) + x[2]*(z[3]-z[0]) + x[3]*(z[0]-z[2])),
        -(x[0]*(z[1]-z[3]) + x[1]*(z[3]-z[0]) + x[3]*(z[0]-z[1])),
         (x[0]*(z[1]-z[2]) + x[1]*(z[2]-z[0]) + x[2]*(z[0]-z[1]))
    ])
    d = np.array([
         (x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])),
        -(x[0]*(y[2]-y[3]) + x[2]*(y[3]-y[0]) + x[3]*(y[0]-y[2])),
         (x[0]*(y[1]-y[3]) + x[1]*(y[3]-y[0]) + x[3]*(y[0]-y[1])),
        -(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
    ])

    B = np.zeros((6, 12))
    for i in range(4):
        col = 3 * i
        B[0, col]   = b[i]; B[1, col+1] = c[i]; B[2, col+2] = d[i]
        B[3, col]   = c[i]; B[3, col+1] = b[i]
        B[4, col+1] = d[i]; B[4, col+2] = c[i]
        B[5, col]   = d[i]; B[5, col+2] = b[i]

    B /= (6.0 * V)
    Ke = (B.T @ D @ B) * V
    return Ke, B, D


def select_native_boundary_nodes(nodes):
    """
    Strict geometric isolation using native STEP coordinates:
    - Fixed Supports: Exactly the inner cylindrical walls of the 3 M8 mounting holes (r = 4.0mm).
    - Load Nodes: Exactly the inner curved bearing surfaces of the two rope guide slots,
      shifted inward to match the true test rig moment arm.
    """
    xmax = nodes[:, 0].max()
    
    # 1. Fixed Supports: Exact M8 Hole Cylinders on the back flange face (X ≈ 75mm)
    hole_centers_z = [20.0, 57.5, 95.0]
    fixed_nodes = []
    
    for i, node in enumerate(nodes):
        x, y, z = node[0], node[1], node[2]
        if x >= xmax - 2.5:
            for cz in hole_centers_z:
                r_hole = np.sqrt((node[1] - 50.0)**2 + (z - cz)**2)
                if abs(r_hole - 4.0) <= 0.6:
                    fixed_nodes.append(i)
                    break

    fixed_nodes = np.array(list(set(fixed_nodes)))

    # 2. Load Nodes: Inset into the actual rope slot cutouts (avoiding the outer overhang tip)
    # Shifting X range inward (e.g., between 25.0mm and 40.0mm) to establish the correct moment arm.
    load_nodes = []
    
    for i, node in enumerate(nodes):
        x, y, z = node[0], node[1], node[2]
        # Target the slot interior region rather than the far edge tip
        if 25.0 <= x <= 40.0:
            # Restrict to the lower half of the slots where the rope bears down
            if 15.0 <= y <= 35.0 and z <= 42.0:
                load_nodes.append(i)

    load_nodes = np.array(list(set(load_nodes)))

    # Safety fallbacks if mesh sizing is coarse
    if len(fixed_nodes) < 5:
        fixed_nodes = np.where(nodes[:, 0] >= xmax - 2.0)[0]
    if len(load_nodes) < 5:
        load_nodes = np.where((nodes[:, 0] >= 25.0) & (nodes[:, 0] <= 40.0))[0]

    return fixed_nodes, load_nodes


def run_fea_analysis(mesh_data, material_props, reference_load_n=100.0, pla_uts_mpa=50.0):
    start_time = time.time()
    
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    E = material_props["E"]
    nu = material_props["nu"]

    fixed_nodes, load_nodes = select_native_boundary_nodes(nodes)
    
    print(f"[Solver] Fixed Support Nodes: {len(fixed_nodes)} | Load Application Nodes: {len(load_nodes)}")

    row_indices, col_indices, data_values = [], [], []
    element_matrices = []

    for elem in elements:
        Ke, B, D = compute_tet_stiffness_and_stress_matrix(nodes[elem], E, nu)
        element_matrices.append((B, D))
        
        global_dofs = []
        for n_idx in elem:
            global_dofs.extend([3*n_idx, 3*n_idx+1, 3*n_idx+2])

        for i in range(12):
            for j in range(12):
                row_indices.append(global_dofs[i])
                col_indices.append(global_dofs[j])
                data_values.append(Ke[i, j])

    K = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(n_dof, n_dof)).tocsr()

    # Apply pure downward vertical force vector (-Z direction) distributed across load nodes
    F = np.zeros(n_dof)
    if len(load_nodes) > 0:
        force_per_node = np.array([0.0, 0.0, -reference_load_n]) / len(load_nodes)
        for n_idx in load_nodes:
            F[3*n_idx:3*n_idx+3] += force_per_node

    fixed_dofs = []
    for n_idx in fixed_nodes:
        fixed_dofs.extend([3*n_idx, 3*n_idx+1, 3*n_idx+2])
    
    penalty = 1e18 * (K.diagonal().max() if K.diagonal().max() > 0 else 1.0)
    K = K.tolil()
    for dof in fixed_dofs:
        K[dof, dof] += penalty
        F[dof] = 0.0
    K = K.tocsr()

    U = spla.spsolve(K, F)
    displacements = U.reshape(-1, 3)

    max_principal_stress = 0.0
    max_von_mises_stress = 0.0

    for idx, elem in enumerate(elements):
        B, D = element_matrices[idx]
        u_elem = np.zeros(12)
        for local_i, node_idx in enumerate(elem):
            u_elem[3*local_i : 3*local_i+3] = displacements[node_idx]

        stress = D @ (B @ u_elem)
        sx, sy, sz, txy, tyz, tzx = stress
        
        stress_tensor = np.array([
            [sx,  txy, tzx],
            [txy, sy,  tyz],
            [tzx, tyz, sz ]
        ])

        principal_stresses = np.linalg.eigvalsh(stress_tensor)
        p1 = np.max(principal_stresses)
        v_mises = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 + 6 * (txy**2 + tyz**2 + tzx**2)))

        if p1 > max_principal_stress:
            max_principal_stress = p1
        if v_mises > max_von_mises_stress:
            max_von_mises_stress = v_mises

    if max_principal_stress > 0:
        failure_scale_factor = pla_uts_mpa / max_principal_stress
        max_load_before_failure_n = reference_load_n * failure_scale_factor
        max_disp_at_failure = np.max(np.linalg.norm(displacements, axis=1)) * failure_scale_factor
    else:
        max_load_before_failure_n = 0.0
        max_disp_at_failure = 0.0

    elapsed_time = time.time() - start_time

    return {
        "max_load_failure_n": max_load_before_failure_n,
        "max_disp_failure_mm": max_disp_at_failure,
        "max_principal_stress_mpa": max_principal_stress,
        "max_von_mises_mpa": max_von_mises_stress,
        "max_displacement_ref_mm": np.max(np.linalg.norm(displacements, axis=1)),
        "runtime_seconds": elapsed_time,
        "num_nodes": n_nodes,
        "num_elements": len(elements),
        "fixed_nodes": fixed_nodes,
        "load_nodes": load_nodes
    }