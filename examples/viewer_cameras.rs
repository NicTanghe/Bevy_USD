use bevy::input::mouse::MouseMotion;
use bevy::light::DirectionalLightShadowMap;
use bevy::prelude::*;
use glam::{DMat4, DQuat, DVec3};
use openusd_rs::{gf, sdf, tf::Token, usd, usd_geom, vt};
use std::path::PathBuf;

use bevy_usd_lib::{UsdPlugin, UsdScene, UsdSceneBundle};

const USD_STAGE_PATH: &str = "USD_scenes/Helmet_bus_cam_3.usdc";

#[derive(Resource)]
struct CameraList {
    cameras: Vec<UsdCameraInfo>,
}

#[derive(Debug, Clone)]
struct UsdCameraInfo {
    prim_path: String,
    prim_name: String,
    transform: Transform,
    projection: UsdCameraProjection,
    clipping_range: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
enum UsdCameraProjection {
    Perspective { vertical_fov: f32, aspect_ratio: f32 },
    Orthographic,
}

#[derive(Component)]
struct ViewerCamera;

#[derive(Component)]
struct FlyCamera {
    speed: f32,
    boost: f32,
    sensitivity: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            speed: 6.0,
            boost: 4.0,
            sensitivity: 0.003,
        }
    }
}

#[derive(Component)]
struct CameraButton {
    index: usize,
}

#[derive(Component)]
struct CameraUiRoot;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(UsdPlugin)
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (build_camera_ui, camera_button_system, fly_camera_system),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene: Handle<UsdScene> = asset_server.load(format!("{USD_STAGE_PATH}#Scene0"));
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
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCamera::default(),
        ViewerCamera,
    ));

    let cameras = load_usd_cameras(USD_STAGE_PATH);
    commands.insert_resource(CameraList { cameras });
}

fn build_camera_ui(
    mut commands: Commands,
    cameras: Res<CameraList>,
    existing: Query<Entity, With<CameraUiRoot>>,
) {
    if !existing.is_empty() {
        return;
    }

    let root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                top: Val::Px(12.0),
                padding: UiRect::all(Val::Px(8.0)),
                row_gap: Val::Px(6.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.05, 0.85)),
            CameraUiRoot,
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        parent.spawn((
            Text::new("USD Cameras"),
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
        ));

        if cameras.cameras.is_empty() {
            parent.spawn((
                Text::new("No cameras found"),
                TextColor(Color::srgb(0.7, 0.7, 0.7)),
            ));
            return;
        }

        for (index, camera) in cameras.cameras.iter().enumerate() {
            let label = if camera.prim_name.is_empty() {
                camera.prim_path.clone()
            } else {
                camera.prim_name.clone()
            };

            parent
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(220.0),
                        height: Val::Px(28.0),
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                    CameraButton { index },
                ))
                .with_children(|button| {
                    button.spawn((Text::new(label), TextColor(Color::srgb(0.9, 0.9, 0.9))));
                });
        }
    });
}

fn camera_button_system(
    mut interactions: Query<(&Interaction, &CameraButton), Changed<Interaction>>,
    cameras: Res<CameraList>,
    mut viewer_camera: Query<(&mut Transform, &mut Projection), With<ViewerCamera>>,
) {
    let Ok((mut transform, mut projection)) = viewer_camera.single_mut() else {
        return;
    };

    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(camera) = cameras.cameras.get(button.index) else {
            continue;
        };

        *transform = camera.transform;
        transform.scale = Vec3::ONE;
        if transform.rotation.length_squared() <= f32::EPSILON {
            transform.rotation = Quat::IDENTITY;
        } else {
            transform.rotation = transform.rotation.normalize();
        }

        *projection = projection_from_usd(camera);
    }
}

fn projection_from_usd(camera: &UsdCameraInfo) -> Projection {
    let near = camera.clipping_range[0].max(0.001);
    let mut far = camera.clipping_range[1];
    if far <= near {
        far = near + 1.0;
    }

    match camera.projection {
        UsdCameraProjection::Perspective {
            vertical_fov,
            aspect_ratio,
        } => Projection::Perspective(PerspectiveProjection {
            fov: vertical_fov,
            aspect_ratio,
            near,
            far,
        }),
        UsdCameraProjection::Orthographic => {
            let mut projection = OrthographicProjection::default_3d();
            projection.near = near;
            projection.far = far;
            Projection::Orthographic(projection)
        }
    }
}

fn fly_camera_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut viewer_camera: Query<(&mut Transform, &FlyCamera), With<ViewerCamera>>,
) {
    let Ok((mut transform, settings)) = viewer_camera.single_mut() else {
        return;
    };

    let mut delta = Vec2::ZERO;
    for event in motion.read() {
        delta += event.delta;
    }

    if buttons.pressed(MouseButton::Right) && delta.length_squared() > 0.0 {
        let yaw = -delta.x * settings.sensitivity;
        let pitch = -delta.y * settings.sensitivity;
        transform.rotation = Quat::from_rotation_y(yaw) * transform.rotation;
        transform.rotation = transform.rotation * Quat::from_rotation_x(pitch);
    }

    let mut axis = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        axis += *transform.forward();
    }
    if keys.pressed(KeyCode::KeyS) {
        axis -= *transform.forward();
    }
    if keys.pressed(KeyCode::KeyA) {
        axis -= *transform.right();
    }
    if keys.pressed(KeyCode::KeyD) {
        axis += *transform.right();
    }
    if keys.pressed(KeyCode::KeyE) {
        axis += *transform.up();
    }
    if keys.pressed(KeyCode::KeyQ) {
        axis -= *transform.up();
    }

    if axis.length_squared() > 0.0 {
        let mut speed = settings.speed;
        if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            speed *= settings.boost;
        }
        transform.translation += axis.normalize() * speed * time.delta_secs();
    }
}

fn load_usd_cameras(stage_path: &str) -> Vec<UsdCameraInfo> {
    let Some(stage_path) = resolve_stage_path(stage_path) else {
        eprintln!("USD stage not found: {stage_path}");
        return Vec::new();
    };

    let stage = usd::Stage::open(&stage_path);
    let mut cameras = Vec::new();
    let identity = DMat4::IDENTITY;
    collect_cameras(&stage, &stage.pseudo_root(), identity, &mut cameras);
    cameras
}

fn resolve_stage_path(path: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() && candidate.exists() {
        return Some(candidate);
    }

    let cwd = std::env::current_dir().ok()?;
    let direct = cwd.join(path);
    if direct.exists() {
        return Some(direct);
    }

    let assets = cwd.join("assets").join(path);
    if assets.exists() {
        return Some(assets);
    }

    None
}

fn collect_cameras(stage: &usd::Stage, prim: &usd::Prim, parent_xf: DMat4, out: &mut Vec<UsdCameraInfo>) {
    let (local, reset_stack) = get_local_transform(prim);
    let local = local.unwrap_or(DMat4::IDENTITY);
    let world_xf = if reset_stack { local } else { parent_xf * local };

    if prim.type_name().as_str() == "Camera" {
        out.push(get_camera_data(prim, world_xf));
    }

    for child in prim.children() {
        collect_cameras(stage, &child, world_xf, out);
    }
}

fn get_camera_data(prim: &usd::Prim, xf: DMat4) -> UsdCameraInfo {
    const DEFAULT_FOCAL_LENGTH: f32 = 50.0;
    const DEFAULT_HORIZONTAL_APERTURE: f32 = 20.955;
    const DEFAULT_VERTICAL_APERTURE: f32 = 15.2908;
    const DEFAULT_CLIPPING_RANGE: [f32; 2] = [0.1, 1_000_000.0];

    let prim_path = prim.path().to_string();
    let prim_name = prim.name().as_str().to_string();
    let projection = get_camera_projection(prim);
    let focal_length = get_camera_scalar_attr(prim, "focalLength", DEFAULT_FOCAL_LENGTH);
    let horizontal_aperture =
        get_camera_scalar_attr(prim, "horizontalAperture", DEFAULT_HORIZONTAL_APERTURE);
    let vertical_aperture =
        get_camera_scalar_attr(prim, "verticalAperture", DEFAULT_VERTICAL_APERTURE);
    let clipping_range = get_camera_vec2_attr(prim, "clippingRange", DEFAULT_CLIPPING_RANGE);

    let aspect_ratio = if vertical_aperture.abs() > f32::EPSILON {
        horizontal_aperture / vertical_aperture
    } else {
        1.0
    };
    let vertical_fov = if focal_length > f32::EPSILON && vertical_aperture > f32::EPSILON {
        2.0 * ((vertical_aperture * 0.5) / focal_length).atan()
    } else {
        std::f32::consts::FRAC_PI_3
    };

    let projection = match projection {
        UsdCameraProjection::Perspective { .. } => UsdCameraProjection::Perspective {
            vertical_fov,
            aspect_ratio,
        },
        UsdCameraProjection::Orthographic => UsdCameraProjection::Orthographic,
    };

    let mut transform = transform_from_dmat4(xf);
    transform.scale = Vec3::ONE;
    if transform.rotation.length_squared() <= f32::EPSILON {
        transform.rotation = Quat::IDENTITY;
    } else {
        transform.rotation = transform.rotation.normalize();
    }

    UsdCameraInfo {
        prim_path,
        prim_name,
        transform,
        projection,
        clipping_range,
    }
}

fn transform_from_dmat4(matrix: DMat4) -> Transform {
    let cols = matrix.to_cols_array();
    let mut cols_f32 = [0.0f32; 16];
    for (idx, value) in cols.iter().enumerate() {
        cols_f32[idx] = *value as f32;
    }
    Transform::from_matrix(Mat4::from_cols_array(&cols_f32))
}

fn get_camera_projection(prim: &usd::Prim) -> UsdCameraProjection {
    let projection_attr = prim.attribute(&Token::new("projection"));
    if let Some(token) = projection_attr.try_get::<Token>() {
        if token.as_str() == "orthographic" {
            UsdCameraProjection::Orthographic
        } else {
            UsdCameraProjection::Perspective {
                vertical_fov: 0.0,
                aspect_ratio: 1.0,
            }
        }
    } else {
        UsdCameraProjection::Perspective {
            vertical_fov: 0.0,
            aspect_ratio: 1.0,
        }
    }
}

fn get_camera_scalar_attr(prim: &usd::Prim, name: &str, default: f32) -> f32 {
    let attr = prim.attribute(&Token::new(name));
    let Some(value) = get_attr_value(&attr) else {
        return default;
    };

    value
        .get::<f32>()
        .or_else(|| value.get::<f64>().map(|v| v as f32))
        .unwrap_or(default)
}

fn get_camera_vec2_attr(prim: &usd::Prim, name: &str, default: [f32; 2]) -> [f32; 2] {
    let attr = prim.attribute(&Token::new(name));
    let Some(value) = get_attr_value(&attr) else {
        return default;
    };

    if let Some(v) = value.get::<gf::Vec2f>() {
        [v.x, v.y]
    } else if let Some(v) = value.get::<gf::Vec2d>() {
        [v.x as f32, v.y as f32]
    } else if let Some(v) = value.get::<gf::Vec2h>() {
        [v.x.into(), v.y.into()]
    } else {
        default
    }
}

fn get_attr_value(attr: &usd::Attribute) -> Option<vt::Value> {
    if let Some(value) = attr.get_value() {
        return Some(value);
    }

    let samples = attr.metadata::<sdf::TimeSampleMap>(&sdf::FIELD_KEYS.time_samples)?;
    if let Some((_, value)) = samples.iter().find(|(time, _)| *time == 0.0) {
        return Some(value.clone());
    }

    samples.first().map(|(_, value)| value.clone())
}

#[derive(Clone, Copy)]
enum XformOpType {
    TranslateX,
    TranslateY,
    TranslateZ,
    Translate,
    ScaleX,
    ScaleY,
    ScaleZ,
    Scale,
    RotateX,
    RotateY,
    RotateZ,
    RotateXYZ,
    RotateXZY,
    RotateYXZ,
    RotateYZX,
    RotateZXY,
    RotateZYX,
    Orient,
    Transform,
}

impl TryFrom<&str> for XformOpType {
    type Error = ();
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Ok(match s {
            "translateX" => XformOpType::TranslateX,
            "translateY" => XformOpType::TranslateY,
            "translateZ" => XformOpType::TranslateZ,
            "translate" => XformOpType::Translate,
            "scaleX" => XformOpType::ScaleX,
            "scaleY" => XformOpType::ScaleY,
            "scaleZ" => XformOpType::ScaleZ,
            "scale" => XformOpType::Scale,
            "rotateX" => XformOpType::RotateX,
            "rotateY" => XformOpType::RotateY,
            "rotateZ" => XformOpType::RotateZ,
            "rotateXYZ" => XformOpType::RotateXYZ,
            "rotateXZY" => XformOpType::RotateXZY,
            "rotateYXZ" => XformOpType::RotateYXZ,
            "rotateYZX" => XformOpType::RotateYZX,
            "rotateZXY" => XformOpType::RotateZXY,
            "rotateZYX" => XformOpType::RotateZYX,
            "orient" => XformOpType::Orient,
            "transform" => XformOpType::Transform,
            _ => return Err(()),
        })
    }
}

fn get_op_transform_degrees(op_type: XformOpType, value: vt::Value, is_inverse: bool) -> Option<DMat4> {
    use XformOpType::*;

    let get_scalar = || -> Option<f64> {
        value
            .get::<f64>()
            .or_else(|| value.get::<f32>().map(|v| v.into()))
    };

    let get_vec3 = || -> Option<DVec3> {
        value
            .get::<gf::Vec3d>()
            .or_else(|| value.get::<gf::Vec3f>().map(|v| v.into()))
            .or_else(|| value.get::<gf::Vec3h>().map(|v| v.into()))
            .map(|v| DVec3::new(v.x, v.y, v.z))
    };

    let get_quat = || -> Option<DQuat> {
        value
            .get::<gf::Quatd>()
            .or_else(|| value.get::<gf::Quatf>().map(|v| v.into()))
            .or_else(|| value.get::<gf::Quath>().map(|v| v.into()))
            .map(|v| DQuat::from_xyzw(v.i, v.j, v.k, v.w))
    };

    Some(match op_type {
        TranslateX if is_inverse => DMat4::from_translation(DVec3::new(-get_scalar()?, 0.0, 0.0)),
        TranslateY if is_inverse => DMat4::from_translation(DVec3::new(0.0, -get_scalar()?, 0.0)),
        TranslateZ if is_inverse => DMat4::from_translation(DVec3::new(0.0, 0.0, -get_scalar()?)),
        Translate if is_inverse => DMat4::from_translation(-get_vec3()?),

        TranslateX => DMat4::from_translation(DVec3::new(get_scalar()?, 0.0, 0.0)),
        TranslateY => DMat4::from_translation(DVec3::new(0.0, get_scalar()?, 0.0)),
        TranslateZ => DMat4::from_translation(DVec3::new(0.0, 0.0, get_scalar()?)),
        Translate => DMat4::from_translation(get_vec3()?),

        ScaleX if is_inverse => DMat4::from_scale(DVec3::new(1.0 / get_scalar()?, 1.0, 1.0)),
        ScaleY if is_inverse => DMat4::from_scale(DVec3::new(1.0, 1.0 / get_scalar()?, 1.0)),
        ScaleZ if is_inverse => DMat4::from_scale(DVec3::new(1.0, 1.0, 1.0 / get_scalar()?)),
        Scale if is_inverse => DMat4::from_scale(1.0 / get_vec3()?),

        ScaleX => DMat4::from_scale(DVec3::new(get_scalar()?, 1.0, 1.0)),
        ScaleY => DMat4::from_scale(DVec3::new(1.0, get_scalar()?, 1.0)),
        ScaleZ => DMat4::from_scale(DVec3::new(1.0, 1.0, get_scalar()?)),
        Scale => DMat4::from_scale(get_vec3()?),

        RotateX | RotateY | RotateZ => {
            let angle = get_scalar()?.to_radians();
            let angle = if is_inverse { -angle } else { angle };
            match op_type {
                RotateX => DMat4::from_rotation_x(angle),
                RotateY => DMat4::from_rotation_y(angle),
                RotateZ => DMat4::from_rotation_z(angle),
                _ => unreachable!(),
            }
        }

        RotateXYZ | RotateXZY | RotateYXZ | RotateYZX | RotateZXY | RotateZYX => {
            let vec = get_vec3()?;
            let vec = if is_inverse { -vec } else { vec };
            let vec = DVec3::new(vec.x.to_radians(), vec.y.to_radians(), vec.z.to_radians());

            let rot_x = DQuat::from_axis_angle(DVec3::X, vec.x);
            let rot_y = DQuat::from_axis_angle(DVec3::Y, vec.y);
            let rot_z = DQuat::from_axis_angle(DVec3::Z, vec.z);

            let rot = match op_type {
                RotateXYZ if is_inverse => rot_z * rot_y * rot_x,
                RotateXZY if is_inverse => rot_y * rot_z * rot_x,
                RotateYXZ if is_inverse => rot_z * rot_x * rot_y,
                RotateYZX if is_inverse => rot_x * rot_z * rot_y,
                RotateZXY if is_inverse => rot_y * rot_x * rot_z,
                RotateZYX if is_inverse => rot_x * rot_y * rot_z,

                RotateXYZ => rot_x * rot_y * rot_z,
                RotateXZY => rot_x * rot_z * rot_y,
                RotateYXZ => rot_y * rot_x * rot_z,
                RotateYZX => rot_y * rot_z * rot_x,
                RotateZXY => rot_z * rot_x * rot_y,
                RotateZYX => rot_z * rot_y * rot_x,
                _ => unreachable!(),
            };

            DMat4::from_quat(rot)
        }
        Orient => DMat4::from_quat(if is_inverse {
            get_quat()?.inverse()
        } else {
            get_quat()?
        }),
        Transform => {
            let mat = value.get::<gf::Matrix4d>()?;
            let mat = DMat4::from_cols_array_2d(&mat.data);
            if is_inverse { mat.inverse() } else { mat }
        }
    })
}

fn get_local_transform_degrees(prim: &usd::Prim) -> (Option<DMat4>, bool) {
    if !prim.has_attribute(&usd_geom::TOKENS.xform_op_order) {
        return (None, false);
    }

    let op_order = prim
        .attribute(&usd_geom::TOKENS.xform_op_order)
        .get::<vt::Array<Token>>();

    let mut reset_stack = false;
    let mut transform = DMat4::IDENTITY;
    for op in op_order.iter().rev() {
        let op = op.as_str();
        if op == "!resetXformStack!" || op == "resetXformStack" {
            reset_stack = true;
            break;
        }

        let (op, is_inverse) = op
            .strip_prefix("!invert!")
            .map_or((op, false), |s| (s, true));

        let op_type_name = op.trim_start_matches("xformOp:");
        let op_type_name = op_type_name.split(':').next().unwrap_or(op_type_name);
        let op_type = XformOpType::try_from(op_type_name).ok();
        let op_value = get_attr_value(&prim.attribute(&Token::new(op)));

        if let (Some(op_type), Some(op_value)) = (op_type, op_value) {
            if let Some(op_transform) = get_op_transform_degrees(op_type, op_value, is_inverse) {
                transform *= op_transform;
            }
        }
    }

    (Some(transform), reset_stack)
}

fn get_local_transform(prim: &usd::Prim) -> (Option<DMat4>, bool) {
    let (matrix, reset_stack) = get_local_transform_degrees(prim);
    if matrix.is_some() {
        return (matrix, reset_stack);
    }

    let single_tok = Token::new("xformOp:transform");
    if prim.has_attribute(&single_tok) {
        let attr = prim.attribute(&single_tok);
        let value = get_attr_value(&attr).and_then(|v| v.get::<gf::Matrix4d>());
        if let Some(mat) = value {
            let mat = DMat4::from_cols_array_2d(&mat.data);
            return (Some(mat), reset_stack);
        }
    }

    (None, reset_stack)
}
