use downcast_rs::Downcast;
use openusd_rs::{
    gf::{self, Matrix4d},
    sdf,
    tf::Token,
    usd, usd_geom, vt,
};

// -------- Data structs --------
#[derive(Debug, Clone)]
pub struct InstancedMesh {
    pub mesh: MeshData,
    pub positions: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>, // stored as [x,y,z,w]
    pub scales: Vec<[f32; 3]>,
}

#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<usize>,
    pub face_vertex_indices: Vec<usize>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
}

impl MeshData {
    pub fn apply_transform(&mut self, xf: &openusd_rs::gf::Matrix4d) {
        let m: [[f64; 4]; 4] = *xf.as_array();
        self.positions = self
            .positions
            .iter()
            .map(|&p| transform_point(&m, p))
            .collect();

        if let Some(normals) = &mut self.normals {
            *normals = normals
                .iter()
                .map(|&n| transform_direction(&m, n))
                .collect();
        }
    }
}

// -------- Local transform --------
fn get_local_transform(prim: &usd::Prim) -> Option<Matrix4d> {
    let order_tok = Token::new("xformOpOrder");
    if prim.has_attribute(&order_tok) {
        let attr = prim.attribute(&order_tok);
        let order: vt::Array<Token> = attr.get::<vt::Array<Token>>();

        let mut local = Matrix4d::identity();
        for op_name in order.iter() {
            if prim.has_attribute(op_name) {
                let op_attr = prim.attribute(op_name);
                let m = op_attr.get::<Matrix4d>();
                local *= m;
            }
        }
        return Some(local);
    }

    let single_tok = Token::new("xformOp:transform");
    if prim.has_attribute(&single_tok) {
        let attr = prim.attribute(&single_tok);
        return Some(attr.get::<Matrix4d>());
    }

    None
}

// -------- Transform points --------
#[inline]
fn transform_point(m: &[[f64; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let x = p[0] as f64;
    let y = p[1] as f64;
    let z = p[2] as f64;

    let tx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    let ty = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    let tz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    let tw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];

    let inv_w = if tw.abs() > f64::EPSILON {
        1.0 / tw
    } else {
        1.0
    };

    [
        (tx * inv_w) as f32,
        (ty * inv_w) as f32,
        (tz * inv_w) as f32,
    ]
}

#[inline]
fn transform_direction(m: &[[f64; 4]; 4], v: [f32; 3]) -> [f32; 3] {
    let x = v[0] as f64;
    let y = v[1] as f64;
    let z = v[2] as f64;

    let tx = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    let ty = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    let tz = m[2][0] * x + m[2][1] * y + m[2][2] * z;

    let len = (tx * tx + ty * ty + tz * tz).sqrt();
    if len.abs() > f64::EPSILON {
        [(tx / len) as f32, (ty / len) as f32, (tz / len) as f32]
    } else {
        [x as f32, y as f32, z as f32]
    }
}

// -------- Mesh data --------
fn get_mesh_data(prim: &usd::Prim) -> MeshData {
    let path = prim.path().clone();
    let stage = prim.stage();
    let mesh = usd_geom::Mesh::define(&stage, path);

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

    let normals = if mesh.has_normals_attr() {
        let arr: vt::Array<gf::Vec3f> = mesh.normals_attr().get();
        if arr.len() > 0 {
            Some(arr.iter().map(|n| [n.x, n.y, n.z]).collect())
        } else {
            None
        }
    } else {
        None
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
        uvs,
    }
}

fn make_trs_matrix(pos: [f32; 3], rot: [f32; 4], scale: [f32; 3]) -> Matrix4d {
    let [x, y, z, w] = rot; // <-- keep consistent with extraction
    let (x, y, z, w) = (x as f64, y as f64, z as f64, w as f64);

    let rot_m: [[f64; 4]; 4] = [
        [
            1.0 - 2.0 * y * y - 2.0 * z * z,
            2.0 * x * y - 2.0 * z * w,
            2.0 * x * z + 2.0 * y * w,
            0.0,
        ],
        [
            2.0 * x * y + 2.0 * z * w,
            1.0 - 2.0 * x * x - 2.0 * z * z,
            2.0 * y * z - 2.0 * x * w,
            0.0,
        ],
        [
            2.0 * x * z - 2.0 * y * w,
            2.0 * y * z + 2.0 * x * w,
            1.0 - 2.0 * x * x - 2.0 * y * y,
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let mut m = Matrix4d::from_array(rot_m);

    // scale individual columns so we apply scale before rotation
    for row in 0..4 {
        m[row][0] *= scale[0] as f64;
        m[row][1] *= scale[1] as f64;
        m[row][2] *= scale[2] as f64;
    }

    // translate (column-major convention)
    m[0][3] = pos[0] as f64;
    m[1][3] = pos[1] as f64;
    m[2][3] = pos[2] as f64;

    m
}

// -------- Recursively expand prims --------
fn expand_prim(
    stage: &usd::Stage,
    prim: &usd::Prim,
    parent_xf: &Matrix4d,
    meshes_out: &mut Vec<MeshData>,
    instanced_out: &mut Vec<InstancedMesh>,
) {
    let local = get_local_transform(prim).unwrap_or(Matrix4d::identity());
    let world_xf = parent_xf.post_mult(&local);

    match prim.type_name().as_str() {
        "Mesh" => {
            let mut mesh = get_mesh_data(prim);
            mesh.apply_transform(&world_xf);
            meshes_out.push(mesh);
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

            // ✅ store quats as [x,y,z,w]
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
                        expand_prim(stage, &proto, &new_xf, meshes_out, instanced_out);
                    }
                }
            }
        }
        _ => {
            for child in prim.children() {
                expand_prim(stage, &child, &world_xf, meshes_out, instanced_out);
            }
        }
    }
}

// -------- Entry point --------
pub fn fetch_stage_usd(stagep: &str) -> (Vec<MeshData>, Vec<InstancedMesh>) {
    let stage = usd::Stage::open(stagep);
    let mut meshes = Vec::new();
    let mut instanced = Vec::new();

    expand_prim(
        &stage,
        &stage.pseudo_root(),
        &Matrix4d::identity(),
        &mut meshes,
        &mut instanced,
    );

    (meshes, instanced)
}
