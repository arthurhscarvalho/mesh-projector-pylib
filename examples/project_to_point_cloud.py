import open3d as o3d
import mesh_projector as mp

if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(
        shading_mode="blinn_phong",
        resolution=1024,
        verbose=True,
    )
    # Lighting configuration
    light_config = mp.PointLights(
        positions=[
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
        ],
        global_ambient=(0.5, 0.5, 0.5),
        global_diffuse=(1.0, 1.0, 1.0),
        global_specular=(0.0, 0.0, 0.0),
    )
    # Read the mesh
    mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    # Project to images
    points, colors = mp.project_to_point_cloud(
        mesh,
        light_config,
        density_factor=1.0,
    )
    # Save the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
