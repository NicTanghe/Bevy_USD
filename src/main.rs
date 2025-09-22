// vim: set filetype=rust:
//! A simple 3D scene with light shining over a cube sitting on a plane.

mod usdish;
use usdish::meshdata_to_bevy;

mod open_rs_loader;
use open_rs_loader::{MeshInstance, fetch_stage_usd};

use bevy::{prelude::*, render::mesh::MeshTag};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

const USD_STAGE_PATH: &str = "C:/Users/Nicol/CGI/year5/slay/usd/Helmet_bus_3.usdc";

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, setup)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // import USD data without baking transforms into vertex data
    let scene = fetch_stage_usd(USD_STAGE_PATH);

    // cache Mesh handles so instances can reuse geometry
    let mesh_handles: Vec<Handle<Mesh>> = scene
        .meshes
        .iter()
        .map(|mesh| meshes.add(meshdata_to_bevy(mesh)))
        .collect();

    let material_handle = materials.add(Color::srgb(0.7, 0.7, 0.7));

    for instance in &scene.instances {
        if let Some(mesh_handle) = mesh_handles.get(instance.mesh_index) {
            commands.spawn((
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d(material_handle.clone()),
                MeshTag(instance.mesh_index as u32),
                instance_to_transform(instance),
            ));
        }
    }

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 8.0, -4.0),
    ));

    // camera
    commands.spawn((
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn instance_to_transform(instance: &MeshInstance) -> Transform {
    let mat = Mat4::from_cols_array(&[
        instance.transform[0][0],
        instance.transform[1][0],
        instance.transform[2][0],
        instance.transform[3][0],
        instance.transform[0][1],
        instance.transform[1][1],
        instance.transform[2][1],
        instance.transform[3][1],
        instance.transform[0][2],
        instance.transform[1][2],
        instance.transform[2][2],
        instance.transform[3][2],
        instance.transform[0][3],
        instance.transform[1][3],
        instance.transform[2][3],
        instance.transform[3][3],
    ]);

    Transform::from_matrix(mat)
}
