import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geometry import load_and_mesh_step
from solver import run_fea_analysis

# Path to your STEP file
STEP_PATH = r"C:\Users\Brock Riley\Downloads\partstepimport.STEP"

# PLA Material Properties
pla_material = {
    "E": 3000.0,   # MPa (Elastic Modulus)
    "nu": 0.35     # Poisson's Ratio
}

PLA_UTS_MPA = 50.0 
N_TO_LBS = 0.224808943870961

# Expanded mesh resolution sweep (mm) from coarse to fine
mesh_resolution_sweep = [10.0, 8.0, 5.0, 4.0, 3.0, 2.0]


def get_exterior_surface_triangles(elements):
    """
    Extracts only the outer boundary triangular faces of a 3D tetrahedral mesh.
    """
    f0 = elements[:, [0, 1, 2]]
    f1 = elements[:, [0, 1, 3]]
    f2 = elements[:, [1, 2, 3]]
    f3 = elements[:, [0, 2, 3]]
    
    all_faces = np.vstack((f0, f1, f2, f3))
    sorted_faces = np.sort(all_faces, axis=1)
    
    _, unique_idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    boundary_faces = all_faces[unique_idx[counts == 1]]
    return boundary_faces


def update_live_dashboard(fig, ax_3d, ax_conv, nodes, elements, fixed_nodes, load_nodes, study_results):
    ax_3d.cla()
    ax_conv.cla()

    current = study_results[-1]
    
    # --- LEFT SUBPLOT: 3D Exterior Mesh & Boundary Conditions ---
    boundary_faces = get_exterior_surface_triangles(elements)
    verts = nodes[boundary_faces]
    
    mesh_poly = Poly3DCollection(verts, alpha=0.35, facecolor='#4A90E2', edgecolor='#1B365D', linewidth=0.25)
    ax_3d.add_collection3d(mesh_poly)

    if len(fixed_nodes) > 0:
        ax_3d.scatter(nodes[fixed_nodes, 0], nodes[fixed_nodes, 1], nodes[fixed_nodes, 2],
                      color='blue', s=18, label='Fixed Supports (M8 Holes)', zorder=5)
    if len(load_nodes) > 0:
        ax_3d.scatter(nodes[load_nodes, 0], nodes[load_nodes, 1], nodes[load_nodes, 2],
                      color='red', s=18, label='Load Nodes (Rope Slots)', zorder=5)

    ax_3d.set_xlim(nodes[:, 0].min() - 5, nodes[:, 0].max() + 5)
    ax_3d.set_ylim(nodes[:, 1].min() - 5, nodes[:, 1].max() + 5)
    ax_3d.set_zlim(nodes[:, 2].min() - 5, nodes[:, 2].max() + 5)
    ax_3d.set_box_aspect([
        np.ptp(nodes[:, 0]) + 10,
        np.ptp(nodes[:, 1]) + 10,
        np.ptp(nodes[:, 2]) + 10
    ])
    
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title(
        f"Live Mesh: {current['mesh_size']:.2f} mm | Nodes: {current['nodes']:,} | Elements: {current['elements']:,}",
        fontsize=11, fontweight='bold'
    )
    ax_3d.legend(loc='upper right', fontsize=8)

    # --- RIGHT SUBPLOT: Live Failure Load Convergence ---
    elem_counts = [r["elements"] for r in study_results]
    loads_lbs = [r["max_load_lbs"] for r in study_results]
    runtimes = [r["runtime"] for r in study_results]

    color_load = '#D9534F'
    ax_conv.plot(elem_counts, loads_lbs, marker='o', color=color_load, linewidth=2, label='Max Load (lbs)')
    ax_conv.set_xlabel("Total Tetrahedral Elements", fontweight='bold')
    ax_conv.set_ylabel("Predicted Max Load (lbs)", color=color_load, fontweight='bold')
    ax_conv.tick_params(axis='y', labelcolor=color_load)
    ax_conv.grid(True, linestyle='--', alpha=0.5)
    ax_conv.set_title("Mesh Convergence & Solver Runtime", fontsize=11, fontweight='bold')

    for r in study_results:
        ax_conv.annotate(f"{r['mesh_size']}mm\n({r['max_load_lbs']:.1f} lbs)",
                         (r["elements"], r["max_load_lbs"]),
                         textcoords="offset points", xytext=(0, 8), ha='center', fontsize=7)

    ax_time = ax_conv.twinx()
    color_time = '#5CB85C'
    ax_time.plot(elem_counts, runtimes, marker='s', linestyle='--', color=color_time, linewidth=1.5, label='Runtime (s)')
    ax_time.set_ylabel("Solve Runtime (s)", color=color_time, fontweight='bold')
    ax_time.tick_params(axis='y', labelcolor=color_time)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.25)


# === MAIN EXECUTION ===

plt.ion()
fig = plt.figure(figsize=(14, 6))
ax_3d = fig.add_subplot(121, projection='3d')
ax_conv = fig.add_subplot(122)
ax_3d.view_init(elev=25, azim=-45)

study_results = []
for m_size in mesh_resolution_sweep:
    print(f"\nRunning simulation for mesh size: {m_size:.2f} mm...")
    
    mesh_data = load_and_mesh_step(STEP_PATH, mesh_size=m_size)
    results = run_fea_analysis(mesh_data, pla_material, reference_load_n=100.0, pla_uts_mpa=PLA_UTS_MPA)
    
    max_load_n = results["max_load_failure_n"]
    max_load_lbs = max_load_n * N_TO_LBS
    
    study_results.append({
        "mesh_size": m_size,
        "nodes": results["num_nodes"],
        "elements": results["num_elements"],
        "max_load_n": max_load_n,
        "max_load_lbs": max_load_lbs,
        "max_disp_fail": results["max_disp_failure_mm"],
        "max_principal": results["max_principal_stress_mpa"],
        "runtime": results["runtime_seconds"]
    })

    update_live_dashboard(
        fig, ax_3d, ax_conv,
        mesh_data["nodes"], mesh_data["elements"],
        results["fixed_nodes"], results["load_nodes"],
        study_results
    )

print("\n" + "="*135)
print(f"{'Mesh (mm)':<10} | {'Nodes':<7} | {'Elements':<9} | {'Max Load (N)':<18} | {'Max Load (lbs)':<18} | {'Max Disp Fail (mm)':<18} | {'Max Principal (MPa)':<20} | {'Runtime (s)':<10}")
print("-" * 135)

for res in study_results:
    print(
        f"{res['mesh_size']:<10.4f} | "
        f"{res['nodes']:<7} | "
        f"{res['elements']:<9} | "
        f"{res['max_load_n']:<18.6f} | "
        f"{res['max_load_lbs']:<18.6f} | "
        f"{res['max_disp_fail']:<18.6f} | "
        f"{res['max_principal']:<20.6f} | "
        f"{res['runtime']:<10.4f}"
    )
print("="*135)

print("\nStudy complete! Close the Matplotlib window to exit.")
plt.ioff()
plt.show()