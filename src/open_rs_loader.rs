use std::collections::HashMap;

use glam::{DMat4, DQuat, DVec3, Mat4, Quat, Vec3};
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
    pub prim_path: String,
    pub prim_name: String,
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

#[derive(Debug, Clone)]
pub struct SceneData {
    pub root_path: String,
    pub instances: Vec<MeshInstance>,
}

#[derive(Debug, Default, Clone)]
pub struct StageData {
    pub meshes: Vec<MeshData>,
    pub scenes: Vec<SceneData>,
}

// -------- Local transform --------
#[derive(Clone, Copy)]
enum XformOpType {
    // Scalar
    TranslateX,
    TranslateY,
    TranslateZ,
    // Vec3
    Translate,
    // Scalar
    ScaleX,
    ScaleY,
    ScaleZ,
    // Vec3
    Scale,
    // Scalar
    RotateX,
    RotateY,
    RotateZ,
    // Vec3
    RotateXYZ,
    RotateXZY,
    RotateYXZ,
    RotateYZX,
    RotateZXY,
    RotateZYX,
    // Quat
    Orient,
    // Matrix4
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
            // USD uses row-major, glam uses column-major, so transpose here
            let mat = DMat4::from_cols_array_2d(&mat.data).transpose();
            if is_inverse { mat.inverse() } else { mat }
        }
    })
}

fn get_local_matrix_degrees(prim: &usd::Prim) -> Option<Matrix4d> {
    if !prim.has_attribute(&usd_geom::TOKENS.xform_op_order) {
        return None;
    }

    let op_order = prim
        .attribute(&usd_geom::TOKENS.xform_op_order)
        .get::<vt::Array<Token>>();

    let mut transform = DMat4::IDENTITY;
    for op in op_order.iter().rev() {
        let (op, is_inverse) = op
            .as_str()
            .strip_prefix("!invert!")
            .map_or((op.as_str(), false), |s| (s, true));

        if op == "resetXformStack" {
            continue;
        }

        let op_type = XformOpType::try_from(op.trim_start_matches("xformOp:")).ok();
        let op_value = prim.attribute(&Token::new(op)).get_value();

        if let (Some(op_type), Some(op_value)) = (op_type, op_value) {
            if let Some(op_transform) = get_op_transform_degrees(op_type, op_value, is_inverse) {
                transform *= op_transform;
            }
        }
    }

    Some(dmat4_to_matrix4d(transform))
}

fn get_local_transform(prim: &usd::Prim) -> Option<Matrix4d> {
    if let Some(matrix) = get_local_matrix_degrees(prim) {
        return Some(matrix);
    }

    let single_tok = Token::new("xformOp:transform");
    if prim.has_attribute(&single_tok) {
        let attr = prim.attribute(&single_tok);
        return attr.try_get::<Matrix4d>();
    }

    None
}

// -------- Mesh data --------
fn get_mesh_data(prim: &usd::Prim) -> MeshData {
    let prim_path = prim.path().to_string();
    let prim_name = prim.name().as_str().to_string();
    let path = prim.path().clone();
    let stage = prim.stage();
    let mesh = usd_geom::Mesh::define(&stage, path);
    let primvars = usd_geom::PrimvarsApi::new(mesh.prim());

    // --- doubleSided
    let double_sided = {
        let double_sided_tok = Token::new("doubleSided");
        mesh.prim()
            .attribute(&double_sided_tok)
            .try_get::<bool>()
            .unwrap_or(false)
    };

    // --- positions
    let positions_arr = mesh
        .points_attr()
        .try_get::<vt::Array<gf::Vec3f>>()
        .unwrap_or_default();
    let positions = positions_arr
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();

    // --- faceVertexCounts
    let fvc_arr = mesh
        .face_vertex_counts_attr()
        .try_get::<vt::Array<i32>>()
        .unwrap_or_default();
    let face_vertex_counts = fvc_arr.iter().map(|&c| c as usize).collect();

    // --- faceVertexIndices
    let fvi_arr = mesh
        .face_vertex_indices_attr()
        .try_get::<vt::Array<i32>>()
        .unwrap_or_default();
    let face_vertex_indices = fvi_arr.iter().map(|&i| i as usize).collect();

    // --- normals / primvars:normals
    let normals_token = Token::new("normals");
    let normals_attr = mesh.normals_attr();
    let normals_arr = normals_attr.try_get::<vt::Array<gf::Vec3f>>();

    let (normals, normal_indices, normal_interpolation) = if let Some(arr) = normals_arr {
        // direct normals attr
        let normals = if !arr.is_empty() {
            Some(arr.iter().map(|n| [n.x, n.y, n.z]).collect())
        } else {
            None
        };

        let interpolation = normals_attr
            .metadata::<Token>(&Token::new("interpolation"))
            .map(|token| PrimvarInterpolation::from_token(token.as_str()));

        let indices_token = Token::new("normals:indices");
        let normal_indices = mesh
            .prim()
            .attribute(&indices_token)
            .try_get::<vt::Array<i32>>()
            .and_then(|idx_arr| {
                if idx_arr.is_empty() {
                    None
                } else {
                    Some(idx_arr.iter().map(|&i| i as usize).collect())
                }
            });

        (normals, normal_indices, interpolation)
    } else {
        // primvar normals
        let primvar = primvars.primvar(&normals_token);
        if primvar.attr.is_valid() {
            let normals = primvar
                .attr
                .try_get::<vt::Array<gf::Vec3f>>()
                .and_then(|arr| {
                    if arr.is_empty() {
                        None
                    } else {
                        Some(arr.iter().map(|n| [n.x, n.y, n.z]).collect())
                    }
                });

            let interpolation = primvar
                .attr
                .metadata::<Token>(&Token::new("interpolation"))
                .map(|token| PrimvarInterpolation::from_token(token.as_str()));

            let normal_indices = primvar.indices().and_then(|idx_arr| {
                if idx_arr.is_empty() {
                    None
                } else {
                    Some(idx_arr.iter().map(|&i| i as usize).collect())
                }
            });

            (normals, normal_indices, interpolation)
        } else {
            (None, None, None)
        }
    };

    // --- UVs
    let uv_primvar = primvars.primvar(&Token::new("st"));
    let uvs = if uv_primvar.attr.is_valid() {
        uv_primvar
            .attr
            .try_get::<vt::Array<gf::Vec2f>>()
            .and_then(|arr| {
                if arr.is_empty() {
                    None
                } else {
                    Some(arr.iter().map(|uv| [uv.x, uv.y]).collect())
                }
            })
    } else {
        None
    };

    MeshData {
        prim_path,
        prim_name,
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
    let src = &matrix.data;
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
    Matrix4d { data }
}

fn matrix4d_to_f32_array(matrix: &Matrix4d) -> [[f32; 4]; 4] {
    let src = &matrix.data;
    let mut out = [[0.0f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = src[row][col] as f32;
        }
    }
    out
}

fn matrix4d_to_dmat4(matrix: &Matrix4d) -> DMat4 {
    DMat4::from_cols_array_2d(&matrix.data).transpose()
}

fn dmat4_to_matrix4d(matrix: DMat4) -> Matrix4d {
    Matrix4d {
        data: matrix.transpose().to_cols_array_2d(),
    }
}

fn matrix4d_identity() -> Matrix4d {
    Matrix4d {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

fn matrix4d_mul(lhs: &Matrix4d, rhs: &Matrix4d) -> Matrix4d {
    dmat4_to_matrix4d(matrix4d_to_dmat4(lhs) * matrix4d_to_dmat4(rhs))
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
struct MeshStore {
    meshes: Vec<MeshData>,
    mesh_lookup: HashMap<String, usize>,
}

impl MeshStore {
    fn new() -> Self {
        Self {
            meshes: Vec::new(),
            mesh_lookup: HashMap::new(),
        }
    }

    fn get_or_insert_mesh(&mut self, prim: &usd::Prim) -> usize {
        let key = prim.path().to_string();
        if let Some(&idx) = self.mesh_lookup.get(&key) {
            return idx;
        }

        let mesh_data = get_mesh_data(prim);
        let index = self.meshes.len();
        self.meshes.push(mesh_data);
        self.mesh_lookup.insert(key, index);
        index
    }
}

struct SceneBuilder {
    instances: Vec<MeshInstance>,
}

impl SceneBuilder {
    fn new() -> Self {
        Self {
            instances: Vec::new(),
        }
    }

    fn push_instance(&mut self, mesh_index: usize, xf: &Matrix4d) {
        let transform = matrix4d_to_f32_array(xf);
        self.instances.push(MeshInstance {
            mesh_index,
            transform,
        });
    }

    fn into_scene(self, root_path: String) -> SceneData {
        SceneData {
            root_path,
            instances: self.instances,
        }
    }
}

// -------- Recursively expand prims --------
fn expand_prim(
    stage: &usd::Stage,
    prim: &usd::Prim,
    parent_xf: &Matrix4d,
    mesh_store: &mut MeshStore,
    scene: &mut SceneBuilder,
) {
    let local = get_local_transform(prim).unwrap_or_else(matrix4d_identity);
    let world_xf = matrix4d_mul(parent_xf, &local);

    match prim.type_name().as_str() {
        "Mesh" => {
            let mesh_index = mesh_store.get_or_insert_mesh(prim);
            scene.push_instance(mesh_index, &world_xf);
        }
        "PointInstancer" => {
            let inst = usd_geom::PointInstancer::define(&stage, prim.path().clone());

            let indices_arr = inst
                .proto_indices_attr()
                .try_get::<vt::Array<i32>>()
                .unwrap_or_default();
            let indices: Vec<usize> = indices_arr.iter().map(|&i| i as usize).collect();

            let positions_arr = inst
                .positions_attr()
                .try_get::<vt::Array<gf::Vec3f>>()
                .unwrap_or_default();
            let positions: Vec<[f32; 3]> =
                positions_arr.iter().map(|p| [p.x, p.y, p.z]).collect();

            let scales_arr = inst
                .scales_attr()
                .try_get::<vt::Array<gf::Vec3f>>()
                .unwrap_or_default();
            let scales: Vec<[f32; 3]> = scales_arr.iter().map(|p| [p.x, p.y, p.z]).collect();

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
                        let new_xf = matrix4d_mul(&world_xf, &xf);
                        expand_prim(stage, &proto, &new_xf, mesh_store, scene);
                    }
                }
            }
        }
        _ => {
            for child in prim.children() {
                expand_prim(stage, &child, &world_xf, mesh_store, scene);
            }
        }
    }
}

fn build_scene_data(stage: &usd::Stage, root: &usd::Prim, mesh_store: &mut MeshStore) -> SceneData {
    let mut builder = SceneBuilder::new();
    let identity = matrix4d_identity();

    expand_prim(stage, root, &identity, mesh_store, &mut builder);

    builder.into_scene(root.path().to_string())
}

// -------- Entry point --------
pub fn load_stage(stage_path: &str) -> StageData {
    let stage = usd::Stage::open(stage_path);
    build_stage_data(&stage)
}

pub fn build_stage_data(stage: &usd::Stage) -> StageData {
    let mut mesh_store = MeshStore::new();
    let mut scenes = Vec::new();

    scenes.push(build_scene_data(stage, &stage.pseudo_root(), &mut mesh_store));
    for child in stage.pseudo_root().children() {
        scenes.push(build_scene_data(stage, &child, &mut mesh_store));
    }

    StageData {
        meshes: mesh_store.meshes,
        scenes,
    }
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
