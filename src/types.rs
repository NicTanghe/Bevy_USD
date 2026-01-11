use bevy::asset::{Asset, Handle};
use bevy::prelude::{
    Bundle, Component, GlobalTransform, InheritedVisibility, Mesh, StandardMaterial, Transform,
    ViewVisibility, Visibility,
};
use bevy::reflect::TypePath;
use std::collections::HashMap;

#[derive(Asset, Debug, TypePath)]
pub struct Usd {
    pub scenes: HashMap<String, Handle<UsdScene>>,
    pub meshes: HashMap<String, Handle<Mesh>>,
    pub materials: HashMap<String, Handle<StandardMaterial>>,
    pub prim_names: HashMap<String, String>,
    pub default_scene: Option<Handle<UsdScene>>,
}

#[derive(Asset, Debug, Clone, TypePath)]
pub struct UsdScene {
    pub label: String,
    pub root_path: String,
    pub instances: Vec<UsdInstance>,
}

#[derive(Debug, Clone)]
pub struct UsdInstance {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
    pub transform: Transform,
    pub prim_path: String,
    pub prim_name: String,
}

#[derive(Component, Debug, Clone)]
pub struct UsdPrimPath(pub String);

#[derive(Component, Debug, Clone)]
pub struct UsdPrimName(pub String);

#[derive(Component, Debug, Clone)]
pub struct UsdSceneRoot {
    pub scene: Handle<UsdScene>,
}

#[derive(Component, Debug, Default, Clone)]
pub struct UsdSceneSpawned;

#[derive(Bundle, Debug)]
pub struct UsdSceneBundle {
    pub scene: UsdSceneRoot,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub visibility: Visibility,
    pub inherited_visibility: InheritedVisibility,
    pub view_visibility: ViewVisibility,
}

impl UsdSceneBundle {
    pub fn new(scene: Handle<UsdScene>) -> Self {
        Self {
            scene: UsdSceneRoot { scene },
            transform: Transform::default(),
            global_transform: GlobalTransform::default(),
            visibility: Visibility::default(),
            inherited_visibility: InheritedVisibility::default(),
            view_visibility: ViewVisibility::default(),
        }
    }
}
