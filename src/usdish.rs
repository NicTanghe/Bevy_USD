use bevy::{
    math::{vec2, vec3},
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        render_asset::RenderAssetUsages,
        render_resource::PrimitiveTopology,
    },
};

use crate::openRsLoader::{fetch_stage_usd, MeshData};

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
    // positions (vertex array)
    let positions_vtx: Vec<Vec3> = mesh.positions.iter().map(|&p| Vec3::from(p)).collect();

    // face indices (to vertex positions)
    let fv_idx: Vec<usize> = mesh
        .face_vertex_indices
        .iter()
        .map(|&i| i as usize)
        .collect();

    // quick sanity checks (turn into proper errors if you prefer)
    let vtx_len = positions_vtx.len();
    if let Some(bad) = fv_idx.iter().position(|&i| i >= vtx_len) {
        panic!(
            "face_vertex_indices[{}] = {} out of range (positions.len() = {})",
            bad, fv_idx[bad], vtx_len
        );
    }
    let sum_counts: usize = mesh.face_vertex_counts.iter().sum();
    debug_assert_eq!(
        sum_counts,
        fv_idx.len(),
        "sum(face_vertex_counts) != face_vertex_indices.len()"
    );

    // expand to wedge-local attributes (one per face-vertex)
    let wedge_positions: Vec<Vec3> = fv_idx.iter().map(|&i| positions_vtx[i]).collect();

    let wedge_normals: Vec<Vec3> = match &mesh.normals {
        // already per-wedge
        Some(n) if n.len() == fv_idx.len() => n.iter().map(|&nn| Vec3::from(nn)).collect(),
        // per-vertex -> expand by indices
        Some(n) if n.len() == vtx_len => fv_idx.iter().map(|&i| Vec3::from(n[i])).collect(),
        // missing / unexpected -> fallback
        _ => vec![Vec3::Y; wedge_positions.len()],
    };

    let wedge_uvs: Vec<Vec2> = if let Some(uvs) = &mesh.uvs {
        if uvs.len() == fv_idx.len() {
            // already per-wedge
            uvs.iter().map(|&u| Vec2::from(u)).collect()
        } else if uvs.len() == vtx_len {
            // per-vertex → expand
            fv_idx.iter().map(|&i| Vec2::from(uvs[i])).collect()
        } else {
            // fallback
            vec![Vec2::ZERO; wedge_positions.len()]
        }
    } else {
        // no UVs at all → fallback
        vec![Vec2::ZERO; wedge_positions.len()]
    };

    // sequential wedge ids for triangulation fan
    let wedge_ids: Vec<u32> = (0..wedge_positions.len() as u32).collect();

    // triangulate using wedge-local data
    let (flat_positions, flat_normals, flat_uvs, tri_indices) = triangulate(
        &mesh.face_vertex_counts,
        &wedge_ids,
        &wedge_positions,
        &wedge_normals,
        &wedge_uvs,
    );

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, flat_positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, flat_normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, flat_uvs)
    .with_inserted_indices(Indices::U32(tri_indices))
}
/// Convert a Vec<MeshData> into a Vec<Mesh>
pub fn meshdata_vec_to_bevy(meshes: Vec<MeshData>) -> Vec<Mesh> {
    meshes.into_iter().map(|m| meshdata_to_bevy(&m)).collect()
}

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
