import mesh_projector as mp
import open3d as o3d

if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(resolution=512, verbose=True)
    # Projection configuration (lighting + density + verbosity)
    projection_config = mp.PointCloudProjectionConfig(verbose=True)
    # Read the mesh
    mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    # Project to images
    points, colors = mp.project_to_point_cloud(mesh, projection_config)
    # Save the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
