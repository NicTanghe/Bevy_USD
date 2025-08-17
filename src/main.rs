// vim: set filetype=rust:
//! A simple 3D scene with light shining over a cube sitting on a plane.

mod usdish;
use usdish::{spawn_custom_mesh,meshdata_to_bevy};

mod openRsLoader;
use openRsLoader::fetch_stage_usd;

use bevy_panorbit_camera::{PanOrbitCamera,PanOrbitCameraPlugin};
use bevy::prelude::*;

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
    let custom_meshes = fetch_stage_usd("C:/Users/Nicol/dev/rust/usd/robberto.usdc");
    //convert to bevy type
    let bevys_meshes: Vec<Mesh> = custom_meshes
        .into_iter()
        .map(|m| meshdata_to_bevy(&m))
        .collect();

    
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

