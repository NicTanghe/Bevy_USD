use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use openusd_rs::{
    gf::{self, Matrix4d},
    tf::Token,
    usd, usd_geom, vt,
};

// -------- Data structs --------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimvarInterpolation {
    Vertex,
    Varying,
    FaceVarying,
    Uniform,
    Constant,
    Unknown,
}

impl PrimvarInterpolation {
    fn from_token(token: &str) -> Self {
        match token {
            "vertex" => PrimvarInterpolation::Vertex,
            "varying" => PrimvarInterpolation::Varying,
            "faceVarying" => PrimvarInterpolation::FaceVarying,
            "uniform" => PrimvarInterpolation::Uniform,
            "constant" => PrimvarInterpolation::Constant,
            _ => PrimvarInterpolation::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<usize>,
    pub face_vertex_indices: Vec<usize>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub normal_indices: Option<Vec<usize>>,
    pub normal_interpolation: Option<PrimvarInterpolation>,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub double_sided: bool,
}

#[derive(Debug, Clone)]
pub struct MeshInstance {
    pub mesh_index: usize,
    pub transform: [[f32; 4]; 4],
}

#[derive(Debug, Default, Clone)]
pub struct SceneData {
    pub meshes: Vec<MeshData>,
    pub instances: Vec<MeshInstance>,
}

// -------- Local transform --------
fn get_local_transform(prim: &usd::Prim) -> Option<Matrix4d> {
    if let Some(matrix) = usd_geom::XformOp::get_local_transform_matrix(prim) {
        return Some(matrix);
    }

    let single_tok = Token::new("xformOp:transform");
    if prim.has_attribute(&single_tok) {
        let attr = prim.attribute(&single_tok);
        return Some(attr.get::<Matrix4d>());
    }

    None
}

// -------- Mesh data --------
fn get_mesh_data(prim: &usd::Prim) -> MeshData {
    let path = prim.path().clone();
    let stage = prim.stage();
    let mesh = usd_geom::Mesh::define(&stage, path);

    let double_sided = {
        let prim = mesh.prim();
        let double_sided_tok = Token::new("doubleSided");

        if prim.has_property("doubleSided") {
            let prop = prim.property(&double_sided_tok);
            if let Some(val) = prop.get_value() {
                if let Some(b) = val.get::<bool>() {
                    // println!("Mesh {} doubleSided = {}", prim.path().clone(), b);
                    b
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    };

    let positions = if mesh.has_points_attr() {
        let arr: vt::Array<gf::Vec3f> = mesh.points_attr().get();
        arr.iter().map(|p| [p.x, p.y, p.z]).collect()
    } else {
        Vec::new()
    };

    let face_vertex_counts = if mesh.has_face_vertex_counts_attr() {
        let arr: vt::Array<i32> = mesh.face_vertex_counts_attr().get();
        arr.iter().map(|&c| c as usize).collect()
    } else {
        Vec::new()
    };

    let face_vertex_indices = if mesh.has_face_vertex_indices_attr() {
        let arr: vt::Array<i32> = mesh.face_vertex_indices_attr().get();
        arr.iter().map(|&i| i as usize).collect()
    } else {
        Vec::new()
    };

    let normals_token = Token::new("normals");
    let has_normals_attr = mesh.has_normals_attr();

    let (normals, normal_indices, normal_interpolation) = if has_normals_attr {
        let attr = mesh.normals_attr();
        let arr: vt::Array<gf::Vec3f> = attr.get();
        let normals = if !arr.is_empty() {
            Some(arr.iter().map(|n| [n.x, n.y, n.z]).collect())
        } else {
            None
        };

        let interpolation = attr
            .metadata::<Token>(&Token::new("interpolation"))
            .map(|token| PrimvarInterpolation::from_token(token.as_str()));

        let indices_token = Token::new("normals:indices");
        let normal_indices = if mesh.prim().has_attribute(&indices_token) {
            let idx_attr = mesh.prim().attribute(&indices_token);
            let idx_arr: vt::Array<i32> = idx_attr.get();
            if !idx_arr.is_empty() {
                Some(idx_arr.iter().map(|&i| i as usize).collect())
            } else {
                None
            }
        } else {
            None
        };

        (normals, normal_indices, interpolation)
    } else if mesh.has_primvar(&normals_token) {
        let attr = mesh.primvar(&normals_token);
        let normals = attr
            .get_value()
            .and_then(|value| value.get::<vt::Array<gf::Vec3f>>())
            .and_then(|arr| {
                if arr.is_empty() {
                    None
                } else {
                    Some(arr.iter().map(|n| [n.x, n.y, n.z]).collect())
                }
            });

        let interpolation = attr
            .metadata::<Token>(&Token::new("interpolation"))
            .map(|token| PrimvarInterpolation::from_token(token.as_str()));

        let indices_token = Token::new("primvars:normals:indices");
        let normal_indices = if mesh.prim().has_attribute(&indices_token) {
            let idx_attr = mesh.prim().attribute(&indices_token);
            let idx_arr: vt::Array<i32> = idx_attr.get();
            if !idx_arr.is_empty() {
                Some(idx_arr.iter().map(|&i| i as usize).collect())
            } else {
                None
            }
        } else {
            None
        };

        (normals, normal_indices, interpolation)
    } else {
        (None, None, None)
    };

    let uvs = if mesh.has_primvar(&Token::new("st")) {
        let uv_attr = mesh.primvar(&Token::new("st"));
        if let Some(val) = uv_attr.get_value() {
            if let Some(arr) = val.get::<vt::Array<gf::Vec2f>>() {
                if arr.len() > 0 {
                    Some(arr.iter().map(|uv| [uv.x, uv.y]).collect())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    MeshData {
        positions,
        face_vertex_counts,
        face_vertex_indices,
        normals,
        normal_indices,
        normal_interpolation,
        uvs,
        double_sided,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn matrix4d_to_mat4(matrix: &Matrix4d) -> Mat4 {
    let src = matrix.as_array();
    let mut cols = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            cols[col * 4 + row] = src[row][col] as f32;
        }
    }
    Mat4::from_cols_array(&cols)
}

fn mat4_to_matrix4d(mat: Mat4) -> Matrix4d {
    let cols = mat.to_cols_array();
    let mut data = [[0.0f64; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            data[row][col] = cols[col * 4 + row] as f64;
        }
    }
    Matrix4d::from_array(data)
}

fn matrix4d_to_f32_array(matrix: &Matrix4d) -> [[f32; 4]; 4] {
    let src = matrix.as_array();
    let mut out = [[0.0f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = src[row][col] as f32;
        }
    }
    out
}

fn make_trs_matrix(pos: [f32; 3], rot: [f32; 4], scale: [f32; 3]) -> Matrix4d {
    let translation = Vec3::from_array(pos);
    let mut rotation = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
    if rotation.length_squared() <= f32::EPSILON {
        rotation = Quat::IDENTITY;
    } else {
        rotation = rotation.normalize();
    }
    let scale = Vec3::from_array(scale);

    let mat = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    mat4_to_matrix4d(mat)
}

// -------- Scene builder --------
struct SceneBuilder {
    data: SceneData,
    mesh_lookup: HashMap<String, usize>,
}

impl SceneBuilder {
    fn new() -> Self {
        Self {
            data: SceneData::default(),
            mesh_lookup: HashMap::new(),
        }
    }

    fn get_or_insert_mesh(&mut self, prim: &usd::Prim) -> usize {
        let key = prim.path().to_string();
        if let Some(&idx) = self.mesh_lookup.get(&key) {
            return idx;
        }

        let mesh_data = get_mesh_data(prim);
        let index = self.data.meshes.len();
        self.data.meshes.push(mesh_data);
        self.mesh_lookup.insert(key, index);
        index
    }

    fn push_instance(&mut self, mesh_index: usize, xf: &Matrix4d) {
        let transform = matrix4d_to_f32_array(xf);
        self.data.instances.push(MeshInstance {
            mesh_index,
            transform,
        });
    }

    fn into_scene(self) -> SceneData {
        self.data
    }
}

// -------- Recursively expand prims --------
fn expand_prim(
    stage: &usd::Stage,
    prim: &usd::Prim,
    parent_xf: &Matrix4d,
    scene: &mut SceneBuilder,
) {
    let local = get_local_transform(prim)
        .map(|m| m.transpose())
        .unwrap_or_else(Matrix4d::identity);
    let world_xf = parent_xf.post_mult(&local);

    match prim.type_name().as_str() {
        "Mesh" => {
            let mesh_index = scene.get_or_insert_mesh(prim);
            scene.push_instance(mesh_index, &world_xf);
        }
        "PointInstancer" => {
            let inst = usd_geom::PointInstancer::define(&stage, prim.path().clone());

            let indices: Vec<usize> = inst
                .proto_indices_attr()
                .get::<vt::Array<i32>>()
                .iter()
                .map(|&i| i as usize)
                .collect();

            let positions: Vec<[f32; 3]> = inst
                .positions_attr()
                .get::<vt::Array<gf::Vec3f>>()
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect();

            let scales: Vec<[f32; 3]> = inst
                .scales_attr()
                .get::<vt::Array<gf::Vec3f>>()
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect();

            let rotations: Vec<[f32; 4]> = match inst.orientations_attr().get_value() {
                Some(val) => {
                    if let Some(arr) = val.get::<vt::Array<gf::Quatf>>() {
                        arr.iter().map(|q| [q.i, q.j, q.k, q.w]).collect()
                    } else if let Some(arr) = val.get::<vt::Array<gf::Quatd>>() {
                        arr.iter()
                            .map(|q| [q.i as f32, q.j as f32, q.k as f32, q.w as f32])
                            .collect()
                    } else if let Some(arr) = val.get::<vt::Array<gf::Quath>>() {
                        arr.iter()
                            .map(|q| [q.i.into(), q.j.into(), q.k.into(), q.w.into()])
                            .collect()
                    } else {
                        vec![]
                    }
                }
                None => vec![],
            };

            let targets = inst.prototypes_rel().targets();
            for (proto_idx, path) in targets.iter().enumerate() {
                let proto = stage.prim_at_path(path.clone());
                for (point_idx, &pi) in indices.iter().enumerate() {
                    if pi == proto_idx {
                        let pos = *positions.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0]);
                        let scale = *scales.get(point_idx).unwrap_or(&[1.0, 1.0, 1.0]);
                        let rot = *rotations.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0, 1.0]);
                        let xf = make_trs_matrix(pos, rot, scale);
                        let new_xf = world_xf.post_mult(&xf);
                        expand_prim(stage, &proto, &new_xf, scene);
                    }
                }
            }
        }
        _ => {
            for child in prim.children() {
                expand_prim(stage, &child, &world_xf, scene);
            }
        }
    }
}

// -------- Entry point --------
pub fn fetch_stage_usd(stagep: &str) -> SceneData {
    let stage = usd::Stage::open(stagep);
    let mut builder = SceneBuilder::new();

    expand_prim(
        &stage,
        &stage.pseudo_root(),
        &Matrix4d::identity(),
        &mut builder,
    );

    builder.into_scene()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn make_trs_matches_glam() {
        let pos = [1.0, -2.5, 3.25];
        let rot = [0.1, 0.2, 0.3, 0.9];
        let scale = [2.0, 3.0, 0.5];

        let usd_matrix = make_trs_matrix(pos, rot, scale);
        let glam_rot = {
            let mut q = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
            if q.length_squared() <= f32::EPSILON {
                q = Quat::IDENTITY;
            } else {
                q = q.normalize();
            }
            q
        };
        let glam_mat = Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            glam_rot,
            Vec3::from_array(pos),
        );

        let converted = matrix4d_to_mat4(&usd_matrix);
        let diff = glam_mat - converted;
        for v in diff.to_cols_array() {
            assert!(v.abs() < 1e-4, "matrix mismatch: {:?}", diff);
        }
    }

    #[test]
    fn matrix_roundtrip() {
        let mat = Mat4::from_scale_rotation_translation(
            Vec3::new(1.5, -2.0, 0.75),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_3),
            Vec3::new(4.0, 5.0, -6.0),
        );
        let usd = mat4_to_matrix4d(mat);
        let back = matrix4d_to_mat4(&usd);
        let diff = mat - back;
        for value in diff.to_cols_array() {
            assert!(approx_eq(value, 0.0));
        }
    }
}
