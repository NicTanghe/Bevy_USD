use bevy::prelude::*;

use crate::types::{UsdPrimName, UsdPrimPath, UsdScene, UsdSceneRoot, UsdSceneSpawned};

pub fn spawn_usd_scenes(
    mut commands: Commands,
    scenes: Res<Assets<UsdScene>>,
    roots: Query<(Entity, &UsdSceneRoot), Without<UsdSceneSpawned>>,
) {
    for (entity, root) in &roots {
        let Some(scene) = scenes.get(&root.scene) else {
            continue;
        };

        commands.entity(entity).insert(UsdSceneSpawned);
        commands.entity(entity).with_children(|parent| {
            for instance in &scene.instances {
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
            }
        });
    }
}
