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
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
        ],
        global_ambient=(0.3, 0.3, 0.3),
        global_diffuse=(0.7, 0.7, 0.7),
        global_specular=(0.0, 0.0, 0.0),
    )
    # Read the mesh
    mesh = mp.read_mesh(
        "assets/nier/Untitled.obj",
        render_config,
        light_config,
    )
    # Project to images
    points, colors = mp.project_to_point_cloud(
        mesh,
        density_factor=1.0,
    )
    # Save the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
