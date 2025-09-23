use bevy::{
    prelude::*,
    render::{mesh::Indices, render_asset::RenderAssetUsages, render_resource::PrimitiveTopology},
};

use crate::open_rs_loader::{MeshData, PrimvarInterpolation};

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

fn generate_wedge_normals(
    positions: &[Vec3],
    face_vertex_counts: &[usize],
    face_vertex_indices: &[usize],
) -> Vec<Vec3> {
    let mut wedge_normals = Vec::with_capacity(face_vertex_indices.len());
    let mut cursor = 0;

    for &count in face_vertex_counts {
        if count < 3 {
            for _ in 0..count {
                wedge_normals.push(Vec3::Y);
            }
            cursor += count;
            continue;
        }

        let first_index = face_vertex_indices[cursor];
        let p0 = positions[first_index];
        let mut face_normal = Vec3::ZERO;

        for tri_offset in 1..(count - 1) {
            let i1 = face_vertex_indices[cursor + tri_offset];
            let i2 = face_vertex_indices[cursor + tri_offset + 1];
            let edge1 = positions[i1] - p0;
            let edge2 = positions[i2] - p0;
            face_normal += edge2.cross(edge1);
        }

        if face_normal.length_squared() <= f32::EPSILON {
            face_normal = Vec3::Y;
        } else {
            face_normal = face_normal.normalize();
        }

        for _ in 0..count {
            wedge_normals.push(face_normal);
        }

        cursor += count;
    }

    wedge_normals
}

fn sample_normals(values: &[Vec3], indices: &[usize]) -> Option<Vec<Vec3>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let value = *values.get(idx)?;
        out.push(value);
    }
    Some(out)
}

fn expand_normals_to_wedges(mesh: &MeshData, positions: &[Vec3], fv_idx: &[usize]) -> Vec<Vec3> {
    let wedge_count = fv_idx.len();
    let vertex_count = positions.len();
    let face_count = mesh.face_vertex_counts.len();

    let values = match &mesh.normals {
        Some(normals) => normals
            .iter()
            .map(|&n| Vec3::from(n))
            .collect::<Vec<Vec3>>(),
        None => return generate_wedge_normals(positions, &mesh.face_vertex_counts, fv_idx),
    };

    let indices_opt = mesh.normal_indices.as_deref();
    let interpolation = mesh
        .normal_interpolation
        .unwrap_or(PrimvarInterpolation::Vertex);

    let result = match interpolation {
        PrimvarInterpolation::FaceVarying => {
            if let Some(indices) = indices_opt {
                if indices.len() == wedge_count {
                    sample_normals(&values, indices)
                } else {
                    None
                }
            } else if values.len() == wedge_count {
                Some(values.clone())
            } else {
                None
            }
        }
        PrimvarInterpolation::Vertex | PrimvarInterpolation::Varying => {
            if let Some(indices) = indices_opt {
                if indices.len() == vertex_count {
                    sample_normals(&values, indices).map(|per_vertex| {
                        fv_idx
                            .iter()
                            .map(|&v_idx| *per_vertex.get(v_idx).unwrap_or(&Vec3::Y))
                            .collect()
                    })
                } else if indices.len() == wedge_count {
                    sample_normals(&values, indices)
                } else {
                    None
                }
            } else if values.len() == vertex_count {
                Some(
                    fv_idx
                        .iter()
                        .map(|&v_idx| *values.get(v_idx).unwrap_or(&Vec3::Y))
                        .collect(),
                )
            } else if values.len() == wedge_count {
                Some(values.clone())
            } else {
                None
            }
        }
        PrimvarInterpolation::Uniform => {
            let per_face = if let Some(indices) = indices_opt {
                if indices.len() == face_count {
                    sample_normals(&values, indices)
                } else {
                    None
                }
            } else if values.len() == face_count {
                Some(values.clone())
            } else {
                None
            };

            per_face.map(|per_face_vals| {
                let mut out = Vec::with_capacity(wedge_count);
                for (face_idx, &count) in mesh.face_vertex_counts.iter().enumerate() {
                    let normal = *per_face_vals.get(face_idx).unwrap_or(&Vec3::Y);
                    for _ in 0..count {
                        out.push(normal);
                    }
                }
                out
            })
        }
        PrimvarInterpolation::Constant => {
            if let Some(indices) = indices_opt {
                if !indices.is_empty() {
                    sample_normals(&values, &indices[..1]).map(|vals| vec![vals[0]; wedge_count])
                } else {
                    None
                }
            } else if !values.is_empty() {
                Some(vec![values[0]; wedge_count])
            } else {
                None
            }
        }
        PrimvarInterpolation::Unknown => None,
    };

    result.unwrap_or_else(|| generate_wedge_normals(positions, &mesh.face_vertex_counts, fv_idx))
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

    let wedge_normals = expand_normals_to_wedges(mesh, &positions_vtx, &fv_idx);

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
