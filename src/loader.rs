use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::asset::{AssetLoader, LoadContext};
use bevy::prelude::{Color, Handle, Mat4, Mesh, StandardMaterial, Transform};
use serde::{Deserialize, Serialize};

use crate::labels::{material_label_for_path, mesh_label_for_path, scene_label_for_path, USD_SCENE0_LABEL};
use crate::open_rs_loader::{load_stage, MeshData, MeshInstance, StageData};
use crate::types::{Usd, UsdInstance, UsdScene};
use crate::usdish::meshdata_to_bevy;

pub struct UsdLoader;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct UsdLoaderSettings {
    pub root_path: Option<String>,
}

#[derive(Debug)]
pub enum UsdLoadError {
    Io(std::io::Error),
    MissingFile(String),
    StagePanic(String),
}

impl std::fmt::Display for UsdLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsdLoadError::Io(err) => write!(f, "io error: {err}"),
            UsdLoadError::MissingFile(path) => write!(f, "usd file not found: {path}"),
            UsdLoadError::StagePanic(path) => write!(f, "failed to load usd stage: {path}"),
        }
    }
}

impl std::error::Error for UsdLoadError {}

impl From<std::io::Error> for UsdLoadError {
    fn from(err: std::io::Error) -> Self {
        UsdLoadError::Io(err)
    }
}

impl AssetLoader for UsdLoader {
    type Asset = Usd;
    type Settings = UsdLoaderSettings;
    type Error = UsdLoadError;

    async fn load(
        &self,
        _reader: &mut dyn bevy::asset::io::Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let stage_path = resolve_stage_path(load_context, settings)?;
        let stage_path_str = stage_path.to_string_lossy().to_string();
        let stage_data = std::panic::catch_unwind(|| load_stage(&stage_path_str))
            .map_err(|_| UsdLoadError::StagePanic(stage_path_str.clone()))?;

        build_assets(stage_data, load_context)
    }

    fn extensions(&self) -> &[&str] {
        &["usd", "usda", "usdc", "usdz"]
    }
}

fn resolve_stage_path(
    load_context: &LoadContext<'_>,
    settings: &UsdLoaderSettings,
) -> Result<PathBuf, UsdLoadError> {
    let path = load_context.path();
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    if let Some(root) = &settings.root_path {
        let candidate = Path::new(root).join(path);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    let cwd = std::env::current_dir()?;
    let candidate = cwd.join(path);
    if candidate.exists() {
        return Ok(candidate);
    }

    let assets_candidate = cwd.join("assets").join(path);
    if assets_candidate.exists() {
        return Ok(assets_candidate);
    }

    Err(UsdLoadError::MissingFile(path.display().to_string()))
}

fn build_assets(
    stage_data: StageData,
    load_context: &mut LoadContext<'_>,
) -> Result<Usd, UsdLoadError> {
    let mut mesh_handles: Vec<Handle<Mesh>> = Vec::with_capacity(stage_data.meshes.len());
    let mut material_handles: Vec<Handle<StandardMaterial>> =
        Vec::with_capacity(stage_data.meshes.len());

    let mut meshes = HashMap::default();
    let mut materials = HashMap::default();
    let mut prim_names = HashMap::default();

    for mesh_data in &stage_data.meshes {
        let mesh_label = mesh_label_for_path(&mesh_data.prim_path);
        let material_label = material_label_for_path(&mesh_data.prim_path);

        let mesh = meshdata_to_bevy(mesh_data);
        let mesh_handle = load_context.add_labeled_asset(mesh_label.clone(), mesh);

        let material = material_from_mesh(mesh_data);
        let material_handle = load_context.add_labeled_asset(material_label.clone(), material);

        mesh_handles.push(mesh_handle.clone());
        material_handles.push(material_handle.clone());
        meshes.insert(mesh_label, mesh_handle);
        materials.insert(material_label, material_handle);
        prim_names.insert(mesh_data.prim_path.clone(), mesh_data.prim_name.clone());
    }

    let mut scenes = HashMap::default();
    let mut default_scene = None;

    for scene_data in &stage_data.scenes {
        let label = scene_label_for_scene(scene_data);
        let instances = build_instances(scene_data, &stage_data.meshes, &mesh_handles, &material_handles);

        let scene = UsdScene {
            label: label.clone(),
            root_path: scene_data.root_path.clone(),
            instances,
        };
        let handle = load_context.add_labeled_asset(label.clone(), scene);

        if label == USD_SCENE0_LABEL {
            default_scene = Some(handle.clone());
        }

        scenes.insert(label, handle);
    }

    Ok(Usd {
        scenes,
        meshes,
        materials,
        prim_names,
        default_scene,
    })
}

fn scene_label_for_scene(scene: &crate::open_rs_loader::SceneData) -> String {
    if scene.root_path == "/" || scene.root_path.is_empty() {
        USD_SCENE0_LABEL.to_string()
    } else {
        scene_label_for_path(&scene.root_path)
    }
}

fn build_instances(
    scene: &crate::open_rs_loader::SceneData,
    meshes: &[MeshData],
    mesh_handles: &[Handle<Mesh>],
    material_handles: &[Handle<StandardMaterial>],
) -> Vec<UsdInstance> {
    scene
        .instances
        .iter()
        .map(|instance| {
            let mesh_data = &meshes[instance.mesh_index];
            let transform = instance_to_transform(instance);

            UsdInstance {
                mesh: mesh_handles[instance.mesh_index].clone(),
                material: material_handles[instance.mesh_index].clone(),
                transform,
                prim_path: mesh_data.prim_path.clone(),
                prim_name: mesh_data.prim_name.clone(),
            }
        })
        .collect()
}

fn material_from_mesh(mesh: &MeshData) -> StandardMaterial {
    let mut material = StandardMaterial::from(Color::srgb(0.7, 0.4, 1.0));
    material.double_sided = mesh.double_sided;
    if mesh.double_sided {
        material.cull_mode = None;
    }
    material
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
