// vim: set filetype=rust:
//! A simple 3D scene with light shining over a cube sitting on a plane.

mod usdish;
use usdish::{meshdata_to_bevy, spawn_custom_mesh};

mod openRsLoader;
use openRsLoader::fetch_stage_usd;

use bevy::{prelude::*, render::mesh::MeshTag};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

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

    // import USD custom type
    //let (custom_meshes, instanced_meshes) = fetch_stage_usd("C:/Users/Nicol/dev/rust/usd/cutofqube.usdc");
    // import USD custom type
    let (custom_meshes, instanced_meshes) =
        fetch_stage_usd("C:/Users/Nicol/CGI/year5/slay/usd/Helmet_bus_2.usdc");

    //convert to bevy type
    let bevys_meshes: Vec<Mesh> = custom_meshes
        .into_iter()
        .map(|m| meshdata_to_bevy(&m))
        .collect();

    // spawn instanced meshes
    for (index, inst) in instanced_meshes.iter().enumerate() {
        let mesh_handle = meshes.add(meshdata_to_bevy(&inst.mesh));
        let material_handle = materials.add(Color::srgb(0.7, 0.7, 0.7));

        for (inst_index, pos) in inst.positions.iter().enumerate() {
            let translation = Vec3::from(*pos);

            let scale = inst
                .scales
                .get(inst_index)
                .map(|s| Vec3::new(s[0] as f32, s[1] as f32, s[2] as f32))
                .unwrap_or(Vec3::ONE);

            //if i just load identity it is the same,wtf ?
            let rotation = inst
                .rotations
                .get(inst_index)
                .map(|o| Quat::from_xyzw(o[0], o[1], o[2], o[3]))
                .unwrap_or(Quat::IDENTITY);

            commands.spawn((
                Mesh3d(mesh_handle.clone()),
                MeshMaterial3d(material_handle.clone()),
                MeshTag(index as u32), // outer loop index = instancer id
                Transform {
                    translation,
                    rotation,
                    scale,
                },
            ));
        }
    }

    // Spawn each one
    for custom_mesh in bevys_meshes {
        spawn_custom_mesh(&mut commands, &mut meshes, &mut materials, custom_mesh);
    }

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // camera
    commands.spawn((
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
