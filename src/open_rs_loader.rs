use std::collections::{HashMap, HashSet};

use glam::{DMat4, DQuat, DVec3, Mat4, Quat, Vec3};
use openusd_rs::{
    gf::{self, Matrix4d},
    sdf, tf::Token,
    usd, usd_geom, vt,
};
use std::cmp::Ordering;

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
    #[allow(dead_code)]
    pub double_sided: bool,
    pub left_handed: bool,
}

#[derive(Debug, Clone)]
pub struct MeshInstance {
    pub mesh_index: usize,
    pub transform: [[f32; 4]; 4],
}

#[derive(Debug, Clone, Copy)]
struct ProtoMeshInstance {
    mesh_index: usize,
    transform: Matrix4d,
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

#[derive(Default)]
struct StageSummary {
    root_children: usize,
    total_prims: usize,
    mesh_prims: usize,
    instancer_prims: usize,
    xform_prims: usize,
    other_prims: usize,
    prims_with_refs: usize,
    prims_with_payloads: usize,
    mesh_with_points: usize,
    mesh_missing_points: usize,
    mesh_with_indices: usize,
    mesh_missing_indices: usize,
    mesh_samples: Vec<MeshSample>,
}

struct MeshSample {
    path: String,
    points_len: usize,
    indices_len: usize,
}

struct InstancerSample {
    path: String,
    targets: usize,
    indices: usize,
    positions: usize,
    scales: usize,
    rotations: usize,
    proto_children: usize,
    proto_indices_type: String,
    positions_type: String,
    scales_type: String,
    orientations_type: String,
    orientationsf_type: String,
    target_preview: Option<String>,
    proto_preview: Option<String>,
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

fn attr_value_or_first_sample(attr: &usd::Attribute) -> Option<vt::Value> {
    if let Some(value) = attr.get_value() {
        return Some(value);
    }

    let samples = attr.metadata::<sdf::TimeSampleMap>(&sdf::FIELD_KEYS.time_samples)?;
    let sample = samples
        .iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))?;
    Some(sample.1.clone())
}

fn attr_typed_or_first_sample<T: vt::ValueType>(attr: &usd::Attribute) -> Option<T> {
    attr_value_or_first_sample(attr).and_then(|value| value.get::<T>())
}

fn read_index_array(attr: &usd::Attribute) -> Vec<usize> {
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<i32>>(attr) {
        return arr
            .iter()
            .filter_map(|&v| usize::try_from(v).ok())
            .collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<i64>>(attr) {
        return arr
            .iter()
            .filter_map(|&v| usize::try_from(v).ok())
            .collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<u32>>(attr) {
        return arr
            .iter()
            .filter_map(|&v| usize::try_from(v).ok())
            .collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<u64>>(attr) {
        return arr
            .iter()
            .filter_map(|&v| usize::try_from(v).ok())
            .collect();
    }
    Vec::new()
}

fn read_vec3_array(attr: &usd::Attribute) -> Vec<[f32; 3]> {
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<gf::Vec3f>>(attr) {
        return arr.iter().map(|v| [v.x, v.y, v.z]).collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<gf::Vec3d>>(attr) {
        return arr
            .iter()
            .map(|v| [v.x as f32, v.y as f32, v.z as f32])
            .collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<gf::Vec3h>>(attr) {
        return arr
            .iter()
            .map(|v| [v.x.into(), v.y.into(), v.z.into()])
            .collect();
    }
    if let Some(arr) = attr_typed_or_first_sample::<vt::Array<gf::Vec3i>>(attr) {
        return arr
            .iter()
            .map(|v| [v.x as f32, v.y as f32, v.z as f32])
            .collect();
    }
    Vec::new()
}

fn read_quat_array(value: vt::Value) -> Vec<[f32; 4]> {
    if let Some(arr) = value.get::<vt::Array<gf::Quatf>>() {
        return arr.iter().map(|q| [q.i, q.j, q.k, q.w]).collect();
    }
    if let Some(arr) = value.get::<vt::Array<gf::Quatd>>() {
        return arr
            .iter()
            .map(|q| [q.i as f32, q.j as f32, q.k as f32, q.w as f32])
            .collect();
    }
    if let Some(arr) = value.get::<vt::Array<gf::Quath>>() {
        return arr
            .iter()
            .map(|q| [q.i.into(), q.j.into(), q.k.into(), q.w.into()])
            .collect();
    }
    Vec::new()
}

fn read_quat_attr(attr: &usd::Attribute) -> Vec<[f32; 4]> {
    attr_value_or_first_sample(attr)
        .map(read_quat_array)
        .unwrap_or_default()
}

fn collect_prototype_children(prim: &usd::Prim) -> Vec<(Token, sdf::Path)> {
    for child in prim.children() {
        let name_token = child.name();
        let name = name_token.as_str();
        if name == "Prototypes" || name == "prototypes" {
            return child
                .children()
                .map(|proto| (proto.name(), proto.path().clone()))
                .collect();
        }
    }
    Vec::new()
}

fn collect_stage_summary(stage: &usd::Stage) -> StageSummary {
    let root = stage.pseudo_root();
    let mut summary = StageSummary::default();
    summary.root_children = root.children().count();
    for child in root.children() {
        summarize_prim(stage, &child, &mut summary);
    }
    summary
}

fn summarize_prim(stage: &usd::Stage, prim: &usd::Prim, summary: &mut StageSummary) {
    summary.total_prims += 1;

    match prim.type_name().as_str() {
        "Mesh" => {
            summary.mesh_prims += 1;

            let mesh = usd_geom::Mesh::define(stage, prim.path().clone());
            let points_len = attr_typed_or_first_sample::<vt::Array<gf::Vec3f>>(&mesh.points_attr())
                .map(|arr| arr.len())
                .unwrap_or(0);
            let indices_len =
                attr_typed_or_first_sample::<vt::Array<i32>>(&mesh.face_vertex_indices_attr())
                    .map(|arr| arr.len())
                    .unwrap_or(0);

            if points_len == 0 {
                summary.mesh_missing_points += 1;
            } else {
                summary.mesh_with_points += 1;
            }

            if indices_len == 0 {
                summary.mesh_missing_indices += 1;
            } else {
                summary.mesh_with_indices += 1;
            }

            if summary.mesh_samples.len() < 3 {
                summary.mesh_samples.push(MeshSample {
                    path: prim.path().to_string(),
                    points_len,
                    indices_len,
                });
            }
        }
        "PointInstancer" => summary.instancer_prims += 1,
        "Xform" => summary.xform_prims += 1,
        _ => summary.other_prims += 1,
    }

    if let Some(op) = prim.metadata::<sdf::ReferenceListOp>(&sdf::FIELD_KEYS.references) {
        if !op.ordered_elements().is_empty() {
            summary.prims_with_refs += 1;
        }
    }

    if let Some(op) = prim.metadata::<sdf::PayloadListOp>(&sdf::FIELD_KEYS.payload) {
        if !op.ordered_elements().is_empty() {
            summary.prims_with_payloads += 1;
        }
    }

    for child in prim.children() {
        summarize_prim(stage, &child, summary);
    }
}

fn log_stage_summary(stage_path: &str, stage: &usd::Stage, stage_data: &StageData) {
    let debug = std::env::var_os("OPENUSD_DEBUG_STAGE").is_some();
    if !debug && !stage_data.meshes.is_empty() {
        return;
    }

    let summary = collect_stage_summary(stage);
    let instance_count: usize = stage_data.scenes.iter().map(|scene| scene.instances.len()).sum();

    println!("USD stage summary: path={stage_path}");
    println!(
        "  assets: scenes={}, meshes={}, instances={}",
        stage_data.scenes.len(),
        stage_data.meshes.len(),
        instance_count
    );
    println!(
        "  prims: total={}, Mesh={}, PointInstancer={}, Xform={}, other={}, root_children={}",
        summary.total_prims,
        summary.mesh_prims,
        summary.instancer_prims,
        summary.xform_prims,
        summary.other_prims,
        summary.root_children
    );
    println!(
        "  arcs: prims with references={}, payloads={}",
        summary.prims_with_refs, summary.prims_with_payloads
    );
    println!(
        "  mesh data: with points={}, missing points={}, with indices={}, missing indices={}",
        summary.mesh_with_points,
        summary.mesh_missing_points,
        summary.mesh_with_indices,
        summary.mesh_missing_indices
    );

    if !summary.mesh_samples.is_empty() {
        let preview = summary
            .mesh_samples
            .iter()
            .map(|sample| {
                format!(
                    "{}(points={}, indices={})",
                    sample.path, sample.points_len, sample.indices_len
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        println!("  mesh samples: {preview}");
    }
}

fn collect_instancer_samples(
    stage: &usd::Stage,
    prim: &usd::Prim,
    samples: &mut Vec<InstancerSample>,
    max_samples: usize,
) {
    if samples.len() >= max_samples {
        return;
    }

    if prim.type_name().as_str() == "PointInstancer" {
        let inst = usd_geom::PointInstancer::define(stage, prim.path().clone());

        let targets = inst.prototypes_rel().targets();
        let indices_len = read_index_array(&inst.proto_indices_attr()).len();
        let positions_len = read_vec3_array(&inst.positions_attr()).len();
        let scales_len = read_vec3_array(&inst.scales_attr()).len();
        let mut rotations_len = read_quat_attr(&inst.orientations_attr()).len();
        if rotations_len == 0 {
            rotations_len = read_quat_attr(&inst.orientationsf_attr()).len();
        }

        let mut proto_children = 0usize;
        let mut proto_preview = None;
        for child in prim.children() {
            let name_token = child.name();
            let name = name_token.as_str();
            if name == "Prototypes" || name == "prototypes" {
                let mut proto_iter = child.children();
                if let Some(first) = proto_iter.next() {
                    proto_preview = Some(first.path().to_string());
                    proto_children = 1 + proto_iter.count();
                }
                break;
            }
        }

        let proto_indices_type = inst.proto_indices_attr().type_name();
        let positions_type = inst.positions_attr().type_name();
        let scales_type = inst.scales_attr().type_name();
        let orientations_type = inst.orientations_attr().type_name();
        let orientationsf_type = inst.orientationsf_attr().type_name();

        samples.push(InstancerSample {
            path: prim.path().to_string(),
            targets: targets.len(),
            indices: indices_len,
            positions: positions_len,
            scales: scales_len,
            rotations: rotations_len,
            proto_children,
            proto_indices_type: proto_indices_type.as_str().to_string(),
            positions_type: positions_type.as_str().to_string(),
            scales_type: scales_type.as_str().to_string(),
            orientations_type: orientations_type.as_str().to_string(),
            orientationsf_type: orientationsf_type.as_str().to_string(),
            target_preview: targets.get(0).map(|path| path.to_string()),
            proto_preview,
        });
    }

    for child in prim.children() {
        collect_instancer_samples(stage, &child, samples, max_samples);
        if samples.len() >= max_samples {
            return;
        }
    }
}

fn log_instancer_summary(stage: &usd::Stage, stage_data: &StageData) {
    let instance_count: usize = stage_data.scenes.iter().map(|scene| scene.instances.len()).sum();
    if instance_count > 0 {
        return;
    }

    let mut samples = Vec::new();
    for child in stage.pseudo_root().children() {
        collect_instancer_samples(stage, &child, &mut samples, 3);
        if samples.len() >= 3 {
            break;
        }
    }

    if samples.is_empty() {
        return;
    }

    println!("  instancer samples={}", samples.len());
    for sample in samples {
        println!(
            "  instancer: {} targets={} indices={} positions={} scales={} rotations={} proto_children={}",
            sample.path,
            sample.targets,
            sample.indices,
            sample.positions,
            sample.scales,
            sample.rotations,
            sample.proto_children
        );
        println!(
            "    types: protoIndices={} positions={} scales={} orientations={} orientationsf={}",
            sample.proto_indices_type,
            sample.positions_type,
            sample.scales_type,
            sample.orientations_type,
            sample.orientationsf_type
        );
        if let Some(target) = sample.target_preview.as_deref() {
            println!("    first target: {}", target);
        }
        if let Some(proto) = sample.proto_preview.as_deref() {
            println!("    first prototype: {}", proto);
        }
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
        let op_value = attr_value_or_first_sample(&prim.attribute(&Token::new(op)));

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
        return attr_typed_or_first_sample::<Matrix4d>(&attr);
    }

    None
}

type LocalTransformCache = HashMap<String, Option<Matrix4d>>;
type PrototypeCache = HashMap<String, Vec<ProtoMeshInstance>>;
type PrototypeStack = HashSet<String>;

fn get_local_transform_cached(
    prim: &usd::Prim,
    cache: &mut LocalTransformCache,
) -> Option<Matrix4d> {
    let key = prim.path().to_string();
    if let Some(value) = cache.get(&key) {
        return *value;
    }

    let value = get_local_transform(prim);
    cache.insert(key, value);
    value
}

fn get_proto_meshes<'a>(
    stage: &usd::Stage,
    prim: &usd::Prim,
    mesh_store: &mut MeshStore,
    xform_cache: &mut LocalTransformCache,
    proto_cache: &'a mut PrototypeCache,
    proto_stack: &mut PrototypeStack,
) -> &'a [ProtoMeshInstance] {
    let key = prim.path().to_string();
    if !proto_cache.contains_key(&key) {
        if !proto_stack.insert(key.clone()) {
            return &[];
        }

        let mut out = Vec::new();
        let identity = matrix4d_identity();
        collect_proto_meshes(
            stage,
            prim,
            &identity,
            mesh_store,
            xform_cache,
            proto_cache,
            proto_stack,
            &mut out,
        );

        proto_stack.remove(&key);
        proto_cache.insert(key.clone(), out);
    }

    proto_cache.get(&key).map(|list| list.as_slice()).unwrap_or(&[])
}

fn collect_proto_meshes(
    stage: &usd::Stage,
    prim: &usd::Prim,
    parent_xf: &Matrix4d,
    mesh_store: &mut MeshStore,
    xform_cache: &mut LocalTransformCache,
    proto_cache: &mut PrototypeCache,
    proto_stack: &mut PrototypeStack,
    out: &mut Vec<ProtoMeshInstance>,
) {
    let local = get_local_transform_cached(prim, xform_cache).unwrap_or_else(matrix4d_identity);
    let world_xf = matrix4d_mul(parent_xf, &local);

    match prim.type_name().as_str() {
        "Mesh" => {
            let mesh_index = mesh_store.get_or_insert_mesh(prim);
            out.push(ProtoMeshInstance {
                mesh_index,
                transform: world_xf,
            });
        }
        "PointInstancer" => {
            let inst = usd_geom::PointInstancer::define(&stage, prim.path().clone());

            let indices = read_index_array(&inst.proto_indices_attr());
            let positions = read_vec3_array(&inst.positions_attr());
            let scales = read_vec3_array(&inst.scales_attr());

            let mut rotations = read_quat_attr(&inst.orientations_attr());
            if rotations.is_empty() {
                rotations = read_quat_attr(&inst.orientationsf_attr());
            }

            let proto_children = collect_prototype_children(prim);

            let mut targets = inst.prototypes_rel().targets();
            if targets.is_empty() {
                targets = proto_children
                    .iter()
                    .map(|(_, path)| path.clone())
                    .collect();
            } else if !proto_children.is_empty() {
                let mut remapped = Vec::with_capacity(targets.len());
                for target in &targets {
                    let proto = stage.prim_at_path(target.clone());
                    if proto.is_valid() {
                        remapped.push(target.clone());
                        continue;
                    }

                    let target_name = target.name_token();
                    if let Some((_, path)) = proto_children
                        .iter()
                        .find(|(name, _)| name.as_str() == target_name.as_str())
                    {
                        remapped.push(path.clone());
                    }
                }
                if !remapped.is_empty() {
                    targets = remapped;
                }
            }

            for (proto_idx, path) in targets.iter().enumerate() {
                let proto = stage.prim_at_path(path.clone());
                if !proto.is_valid() {
                    continue;
                }
                let proto_meshes = get_proto_meshes(
                    stage,
                    &proto,
                    mesh_store,
                    xform_cache,
                    proto_cache,
                    proto_stack,
                );
                if proto_meshes.is_empty() {
                    continue;
                }
                for (point_idx, &pi) in indices.iter().enumerate() {
                    if pi == proto_idx {
                        let pos = *positions.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0]);
                        let scale = *scales.get(point_idx).unwrap_or(&[1.0, 1.0, 1.0]);
                        let rot = *rotations.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0, 1.0]);
                        let xf = make_trs_matrix(pos, rot, scale);
                        let inst_xf = matrix4d_mul(&world_xf, &xf);
                        for proto_mesh in proto_meshes {
                            let mesh_xf = matrix4d_mul(&inst_xf, &proto_mesh.transform);
                            out.push(ProtoMeshInstance {
                                mesh_index: proto_mesh.mesh_index,
                                transform: mesh_xf,
                            });
                        }
                    }
                }
            }
        }
        _ => {
            for child in prim.children() {
                collect_proto_meshes(
                    stage,
                    &child,
                    &world_xf,
                    mesh_store,
                    xform_cache,
                    proto_cache,
                    proto_stack,
                    out,
                );
            }
        }
    }
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

    // --- orientation (defaults to rightHanded)
    let left_handed = {
        let orient_tok = Token::new("orientation");
        mesh.prim()
            .attribute(&orient_tok)
            .try_get::<Token>()
            .map(|token| token.as_str() == "leftHanded")
            .unwrap_or(false)
    };

    // --- positions
    let positions_arr = attr_typed_or_first_sample::<vt::Array<gf::Vec3f>>(&mesh.points_attr())
        .unwrap_or_default();
    let positions = positions_arr
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();

    // --- faceVertexCounts
    let fvc_arr =
        attr_typed_or_first_sample::<vt::Array<i32>>(&mesh.face_vertex_counts_attr())
            .unwrap_or_default();
    let face_vertex_counts = fvc_arr.iter().map(|&c| c as usize).collect();

    // --- faceVertexIndices
    let fvi_arr =
        attr_typed_or_first_sample::<vt::Array<i32>>(&mesh.face_vertex_indices_attr())
            .unwrap_or_default();
    let face_vertex_indices = fvi_arr.iter().map(|&i| i as usize).collect();

    // --- normals / primvars:normals
    let normals_token = Token::new("normals");
    let normals_attr = mesh.normals_attr();
    let normals_arr = attr_typed_or_first_sample::<vt::Array<gf::Vec3f>>(&normals_attr);

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
        let normal_indices =
            attr_typed_or_first_sample::<vt::Array<i32>>(&mesh.prim().attribute(&indices_token))
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
            let normals = attr_typed_or_first_sample::<vt::Array<gf::Vec3f>>(&primvar.attr)
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
        attr_typed_or_first_sample::<vt::Array<gf::Vec2f>>(&uv_primvar.attr).and_then(|arr| {
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
        left_handed,
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
    xform_cache: &mut LocalTransformCache,
    proto_cache: &mut PrototypeCache,
    proto_stack: &mut PrototypeStack,
    scene: &mut SceneBuilder,
) {
    let local = get_local_transform_cached(prim, xform_cache).unwrap_or_else(matrix4d_identity);
    let world_xf = matrix4d_mul(parent_xf, &local);

    match prim.type_name().as_str() {
        "Mesh" => {
            let mesh_index = mesh_store.get_or_insert_mesh(prim);
            scene.push_instance(mesh_index, &world_xf);
        }
        "PointInstancer" => {
            let inst = usd_geom::PointInstancer::define(&stage, prim.path().clone());

            let indices = read_index_array(&inst.proto_indices_attr());
            let positions = read_vec3_array(&inst.positions_attr());
            let scales = read_vec3_array(&inst.scales_attr());

            let mut rotations = read_quat_attr(&inst.orientations_attr());
            if rotations.is_empty() {
                rotations = read_quat_attr(&inst.orientationsf_attr());
            }

            let proto_children = collect_prototype_children(prim);

            let mut targets = inst.prototypes_rel().targets();
            if targets.is_empty() {
                targets = proto_children
                    .iter()
                    .map(|(_, path)| path.clone())
                    .collect();
            } else if !proto_children.is_empty() {
                let mut remapped = Vec::with_capacity(targets.len());
                for target in &targets {
                    let proto = stage.prim_at_path(target.clone());
                    if proto.is_valid() {
                        remapped.push(target.clone());
                        continue;
                    }

                    let target_name = target.name_token();
                    if let Some((_, path)) = proto_children
                        .iter()
                        .find(|(name, _)| name.as_str() == target_name.as_str())
                    {
                        remapped.push(path.clone());
                    }
                }
                if !remapped.is_empty() {
                    targets = remapped;
                }
            }

            for (proto_idx, path) in targets.iter().enumerate() {
                let proto = stage.prim_at_path(path.clone());
                if !proto.is_valid() {
                    continue;
                }
                let proto_meshes = get_proto_meshes(
                    stage,
                    &proto,
                    mesh_store,
                    xform_cache,
                    proto_cache,
                    proto_stack,
                );
                if proto_meshes.is_empty() {
                    continue;
                }
                for (point_idx, &pi) in indices.iter().enumerate() {
                    if pi == proto_idx {
                        let pos = *positions.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0]);
                        let scale = *scales.get(point_idx).unwrap_or(&[1.0, 1.0, 1.0]);
                        let rot = *rotations.get(point_idx).unwrap_or(&[0.0, 0.0, 0.0, 1.0]);
                        let xf = make_trs_matrix(pos, rot, scale);
                        let inst_xf = matrix4d_mul(&world_xf, &xf);
                        for proto_mesh in proto_meshes {
                            let mesh_xf = matrix4d_mul(&inst_xf, &proto_mesh.transform);
                            scene.push_instance(proto_mesh.mesh_index, &mesh_xf);
                        }
                    }
                }
            }
        }
        _ => {
            for child in prim.children() {
                expand_prim(
                    stage,
                    &child,
                    &world_xf,
                    mesh_store,
                    xform_cache,
                    proto_cache,
                    proto_stack,
                    scene,
                );
            }
        }
    }
}

fn build_scene_data(
    stage: &usd::Stage,
    root: &usd::Prim,
    mesh_store: &mut MeshStore,
    xform_cache: &mut LocalTransformCache,
    proto_cache: &mut PrototypeCache,
    proto_stack: &mut PrototypeStack,
) -> SceneData {
    let mut builder = SceneBuilder::new();
    let identity = matrix4d_identity();

    expand_prim(
        stage,
        root,
        &identity,
        mesh_store,
        xform_cache,
        proto_cache,
        proto_stack,
        &mut builder,
    );

    builder.into_scene(root.path().to_string())
}

// -------- Entry point --------
pub fn load_stage(stage_path: &str) -> StageData {
    let stage = usd::Stage::open(stage_path);
    let stage_data = build_stage_data(&stage);
    log_stage_summary(stage_path, &stage, &stage_data);
    log_instancer_summary(&stage, &stage_data);
    stage_data
}

pub fn build_stage_data(stage: &usd::Stage) -> StageData {
    let mut mesh_store = MeshStore::new();
    let mut scenes = Vec::new();
    let mut xform_cache = LocalTransformCache::new();
    let mut proto_cache = PrototypeCache::new();
    let mut proto_stack = PrototypeStack::new();

    scenes.push(build_scene_data(
        stage,
        &stage.pseudo_root(),
        &mut mesh_store,
        &mut xform_cache,
        &mut proto_cache,
        &mut proto_stack,
    ));
    for child in stage.pseudo_root().children() {
        scenes.push(build_scene_data(
            stage,
            &child,
            &mut mesh_store,
            &mut xform_cache,
            &mut proto_cache,
            &mut proto_stack,
        ));
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
