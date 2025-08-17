
use bevy::{
    prelude::*,
    math::{vec3,vec2},
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
};


use crate::openRsLoader::{MeshData,fetch_stage_usd};


fn triangulate(
    counts: &[usize],
    indices: &[u32],
    positions_ref: &[Vec3],
    normals_ref: &[Vec3],
    uvs_ref: &[Vec2],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
    let mut new_positions = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();
    let mut tri_indices = Vec::new();

    let mut wedge_idx = 0;
    for &n in counts {
        for i in 0..(n.saturating_sub(2)) {
            let idxs = [
                indices[wedge_idx] as usize,
                indices[wedge_idx + i + 2] as usize,
                indices[wedge_idx + i + 1] as usize,
            ];
            for &idx in &idxs {
                tri_indices.push(new_positions.len() as u32);
                new_positions.push(positions_ref[idx].to_array());
                new_normals.push(normals_ref[idx].to_array());
                new_uvs.push(uvs_ref[idx].to_array());
            }
        }
        wedge_idx += n;
    }

    (new_positions, new_normals, new_uvs, tri_indices)
}

pub fn meshdata_to_bevy(mesh: &MeshData) -> Mesh {
    let positions: Vec<Vec3> = mesh.positions.iter().map(|p| Vec3::from(*p)).collect();

    let normals: Vec<Vec3> = match &mesh.normals {
        Some(n) => n.iter().map(|nn| Vec3::from(*nn)).collect(),
        None => Vec::new(),
    };

    let uvs: Vec<Vec2> = match &mesh.uvs {
        Some(uvs) => uvs.iter().map(|uv| Vec2::from(*uv)).collect(),
        None => Vec::new(),
    };

    let wedge_positions: Vec<Vec3> = mesh
        .face_vertex_indices
        .iter()
        .map(|&i| positions[i])
        .collect();

    let wedge_normals: Vec<Vec3> = if !normals.is_empty() {
        normals
    } else {
        vec![Vec3::Y; wedge_positions.len()] // fallback normal
    };

    let wedge_uvs: Vec<Vec2> = if !uvs.is_empty() {
        uvs
    } else {
        vec![Vec2::ZERO; wedge_positions.len()] // fallback uv
    };

    let wedge_indices: Vec<u32> = (0..wedge_positions.len() as u32).collect();

    let (flat_positions, flat_normals, flat_uvs, tri_indices) = triangulate(
        &mesh.face_vertex_counts,
        &wedge_indices,
        &wedge_positions,
        &wedge_normals,
        &wedge_uvs,
    );

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, flat_positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, flat_uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, flat_normals)
        .with_inserted_indices(Indices::U32(tri_indices))
}

/// Convert a Vec<MeshData> into a Vec<Mesh>
pub fn meshdata_vec_to_bevy(meshes: Vec<MeshData>) -> Vec<Mesh> {
    meshes.into_iter().map(|m| meshdata_to_bevy(&m)).collect()
}

//pub fn load_usd_as_resource(mut commands: Commands, stage: &str) {
//    let custom_meshes = fetch_stage_usd(stage);
//
//    let bevys_meshes: Vec<Mesh> = custom_meshes
//        .into_iter()
//        .map(|m| meshdata_to_bevy(&m))
//        .collect();
//
//    commands.insert_resource(bevys_meshes);
//}


pub fn spawn_custom_mesh(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    custom_mesh: Mesh,
) {
    commands.spawn((
        Mesh3d(meshes.add(custom_mesh)),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
}
