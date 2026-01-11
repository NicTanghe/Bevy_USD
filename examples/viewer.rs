use bevy::prelude::*;
use bevy::light::DirectionalLightShadowMap;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use bevy_usd_lib::{UsdPlugin, UsdSceneBundle};

const USD_STAGE_PATH: &str = "USD_scenes/Helmet_bus_3.usdc";

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(UsdPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene: Handle<bevy_usd_lib::UsdScene> =
        asset_server.load(format!("{USD_STAGE_PATH}#Scene0"));
    commands.spawn(UsdSceneBundle::new(scene));

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_rotation(
            Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)
                * Quat::from_rotation_y(std::f32::consts::PI),
        ),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.6405, 0.822, 1.0035),
        brightness: 200.0,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
