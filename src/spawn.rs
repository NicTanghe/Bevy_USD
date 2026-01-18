use std::collections::HashMap;

use bevy::asset::AssetId;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::prelude::AlphaMode;

use crate::instancing::{InstanceData, UsdInstanceData};
use crate::types::{UsdPrimName, UsdPrimPath, UsdScene, UsdSceneRoot, UsdSceneSpawned};

struct DebugAssets {
    mesh: Handle<Mesh>,
    bounds_material: Handle<StandardMaterial>,
    min_material: Handle<StandardMaterial>,
    center_material: Handle<StandardMaterial>,
    max_material: Handle<StandardMaterial>,
    origin_material: Handle<StandardMaterial>,
}

impl DebugAssets {
    fn new(meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>) -> Self {
        let mesh = meshes.add(Cuboid::default());
        let bounds_material = materials.add(StandardMaterial {
            base_color: Color::srgba(0.1, 0.9, 1.0, 0.12),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            cull_mode: None,
            ..default()
        });
        let min_material = materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.2, 0.2),
            unlit: true,
            ..default()
        });
        let center_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 1.0, 0.2),
            unlit: true,
            ..default()
        });
        let max_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.4, 1.0),
            unlit: true,
            ..default()
        });
        let origin_material = materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.9, 0.1),
            unlit: true,
            ..default()
        });

        Self {
            mesh,
            bounds_material,
            min_material,
            center_material,
            max_material,
            origin_material,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BatchKey {
    mesh: AssetId<Mesh>,
    material: AssetId<StandardMaterial>,
}

struct InstanceBatch {
    mesh: Handle<Mesh>,
    color: [f32; 4],
    instances: Vec<InstanceData>,
}

fn material_color(
    materials: &Assets<StandardMaterial>,
    handle: &Handle<StandardMaterial>,
) -> [f32; 4] {
    materials
        .get(handle)
        .map(|material| material.base_color.to_linear().to_f32_array())
        .unwrap_or_else(|| Color::srgb(0.7, 0.4, 1.0).to_linear().to_f32_array())
}

fn spawn_debug_markers(
    parent: &mut ChildSpawnerCommands,
    assets: &DebugAssets,
    min: Vec3,
    max: Vec3,
) {
    let center = (min + max) * 0.5;
    let mut extents = (max - min).abs();
    if !extents.is_finite() {
        extents = Vec3::ONE;
    }

    let max_extent = extents.max_element().max(1.0);
    let marker_size = (max_extent * 0.03).clamp(0.05, 2.0);
    let marker_scale = Vec3::splat(marker_size);
    let bounds_scale = extents.max(Vec3::splat(marker_size));

    parent.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.bounds_material.clone()),
        Transform {
            translation: center,
            scale: bounds_scale,
            ..default()
        },
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        NoFrustumCulling,
        Name::new("usd_debug_bounds"),
    ));

    parent.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.min_material.clone()),
        Transform {
            translation: min,
            scale: marker_scale,
            ..default()
        },
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        NoFrustumCulling,
        Name::new("usd_debug_min"),
    ));

    parent.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.center_material.clone()),
        Transform {
            translation: center,
            scale: marker_scale,
            ..default()
        },
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        NoFrustumCulling,
        Name::new("usd_debug_center"),
    ));

    parent.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.max_material.clone()),
        Transform {
            translation: max,
            scale: marker_scale,
            ..default()
        },
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        NoFrustumCulling,
        Name::new("usd_debug_max"),
    ));
}

fn spawn_debug_origin(parent: &mut ChildSpawnerCommands, assets: &DebugAssets) {
    parent.spawn((
        Mesh3d(assets.mesh.clone()),
        MeshMaterial3d(assets.origin_material.clone()),
        Transform {
            translation: Vec3::ZERO,
            scale: Vec3::splat(0.5),
            ..default()
        },
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        NoFrustumCulling,
        Name::new("usd_debug_origin"),
    ));
}

pub fn spawn_usd_scenes(
    mut commands: Commands,
    scenes: Res<Assets<UsdScene>>,
    roots: Query<
        (Entity, &UsdSceneRoot, &Transform, Option<&GlobalTransform>),
        Without<UsdSceneSpawned>,
    >,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let debug_log = std::env::var_os("OPENUSD_DEBUG_STAGE").is_some();
    let debug_draw = std::env::var_os("OPENUSD_DEBUG_DRAW").is_some();
    let debug_origin = std::env::var_os("OPENUSD_DEBUG_ORIGIN").is_some();
    let debug_nocull = std::env::var_os("OPENUSD_DEBUG_NOCULL").is_some() || debug_draw;
    let debug_skip_instances = std::env::var_os("OPENUSD_DEBUG_SKIP_INSTANCES").is_some();
    let debug_entity_instances =
        std::env::var_os("OPENUSD_DEBUG_ENTITY_INSTANCES").is_some();
    let debug_instance_limit = std::env::var("OPENUSD_DEBUG_INSTANCE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let debug_instance_stride = std::env::var("OPENUSD_DEBUG_INSTANCE_STRIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 1);
    let debug_assets = if debug_draw {
        Some(DebugAssets::new(&mut meshes, &mut materials))
    } else {
        None
    };
    for (entity, root, root_transform, root_global) in &roots {
        let Some(scene) = scenes.get(&root.scene) else {
            continue;
        };
        let root_matrix = root_global
            .map(|transform| transform.to_matrix())
            .unwrap_or_else(|| root_transform.to_matrix());

        let bounds = if debug_log || debug_draw {
            let count = scene.instances.len();
            if count > 0 {
                let mut min = Vec3::splat(f32::INFINITY);
                let mut max = Vec3::splat(f32::NEG_INFINITY);
                for instance in &scene.instances {
                    let pos = instance.transform.translation;
                    min = min.min(pos);
                    max = max.max(pos);
                }
                Some((min, max))
            } else {
                None
            }
        } else {
            None
        };

        if debug_log {
            if let Some((min, max)) = bounds {
                println!(
                    "USD spawn: scene={} instances={} bounds min={:?} max={:?}",
                    scene.label,
                    scene.instances.len(),
                    min,
                    max
                );
            } else {
                println!("USD spawn: scene={} instances=0", scene.label);
            }
        }

        commands.entity(entity).insert(UsdSceneSpawned);
        commands.entity(entity).with_children(|parent| {
            if let (Some((min, max)), Some(assets)) = (bounds, debug_assets.as_ref()) {
                spawn_debug_markers(parent, assets, min, max);
            }
            if debug_origin {
                if let Some(assets) = debug_assets.as_ref() {
                    spawn_debug_origin(parent, assets);
                } else {
                    let assets = DebugAssets::new(&mut meshes, &mut materials);
                    spawn_debug_origin(parent, &assets);
                }
            }

            if debug_skip_instances {
                return;
            }

            let max_instances = debug_instance_limit.unwrap_or(scene.instances.len());
            let stride = debug_instance_stride.unwrap_or(1);
            let mut spawned = 0usize;
            let mut skipped_stride = 0usize;
            if debug_entity_instances {
                for (idx, instance) in scene.instances.iter().enumerate() {
                    if idx >= max_instances {
                        break;
                    }
                    if stride > 1 && idx % stride != 0 {
                        skipped_stride += 1;
                        continue;
                    }
                    let mut child = parent.spawn((
                        Mesh3d(instance.mesh.clone()),
                        MeshMaterial3d(instance.material.clone()),
                        instance.transform,
                        GlobalTransform::default(),
                        Visibility::default(),
                        InheritedVisibility::default(),
                        ViewVisibility::default(),
                        UsdPrimPath(instance.prim_path.clone()),
                        UsdPrimName(instance.prim_name.clone()),
                    ));

                    if !instance.prim_name.is_empty() {
                        child.insert(Name::new(instance.prim_name.clone()));
                    }
                    if debug_nocull {
                        child.insert(NoFrustumCulling);
                    }
                    spawned += 1;
                }

                if debug_log && stride > 1 {
                    println!(
                        "USD spawn: scene={} stride={} spawned={} skipped_stride={}",
                        scene.label, stride, spawned, skipped_stride
                    );
                }
                return;
            }

            let mut batches: HashMap<BatchKey, InstanceBatch> = HashMap::new();
            for (idx, instance) in scene.instances.iter().enumerate() {
                if idx >= max_instances {
                    break;
                }
                if stride > 1 && idx % stride != 0 {
                    skipped_stride += 1;
                    continue;
                }

                let key = BatchKey {
                    mesh: instance.mesh.id(),
                    material: instance.material.id(),
                };
                let batch = batches.entry(key).or_insert_with(|| InstanceBatch {
                    mesh: instance.mesh.clone(),
                    color: material_color(materials.as_ref(), &instance.material),
                    instances: Vec::new(),
                });

                let matrix = root_matrix * instance.transform.to_matrix();
                let cols = matrix.to_cols_array_2d();
                batch.instances.push(InstanceData {
                    model_col0: cols[0],
                    model_col1: cols[1],
                    model_col2: cols[2],
                    model_col3: cols[3],
                    color: batch.color,
                });
                spawned += 1;
            }

            let batch_count = batches.len();
            for batch in batches.into_values() {
                if batch.instances.is_empty() {
                    continue;
                }
                parent.spawn((
                    Mesh3d(batch.mesh),
                    UsdInstanceData::new(batch.instances),
                    Transform::default(),
                    GlobalTransform::default(),
                    Visibility::default(),
                    InheritedVisibility::default(),
                    ViewVisibility::default(),
                    NoFrustumCulling,
                ));
            }

            if debug_log {
                println!(
                    "USD spawn: scene={} batches={} instances={} stride={} skipped_stride={}",
                    scene.label, batch_count, spawned, stride, skipped_stride
                );
            }
        });
    }
}
