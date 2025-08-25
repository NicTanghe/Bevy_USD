use openusd_rs::{
    gf::{self, Matrix4d},
    sdf,
    tf::Token,
    usd, usd_geom, vt,
};
use std::collections::HashSet;
use downcast_rs::Downcast;
fn is_in_prototypes_subtree(path: &sdf::Path) -> bool {
    let s = path.to_string();
    s.contains("/Prototypes/") || s.ends_with("/Prototypes")
}

fn is_point_instancer(prim: &usd::Prim) -> bool {
    prim.type_name() == Token::new("PointInstancer")
}

fn collect_leaves_and_instancers(stage: &usd::Stage) -> (Vec<sdf::Path>, Vec<sdf::Path>) {
    let root = stage.pseudo_root();
    let mut stack: Vec<usd::Prim> = root.children().collect();
    let mut leaves: Vec<sdf::Path> = Vec::new();
    let mut instancers: Vec<sdf::Path> = Vec::new();

    let mut seen_inst: HashSet<sdf::Path> = HashSet::new();
    let mut seen_leaf: HashSet<sdf::Path> = HashSet::new();

    while let Some(prim) = stack.pop() {
        let p = prim.path().clone();

        if is_in_prototypes_subtree(&p) {
            continue;
        }
        if is_point_instancer(&prim) {
            if seen_inst.insert(p.clone()) {
                instancers.push(p);
            }
            continue;
        }

        let child_paths: Vec<sdf::Path> = prim
            .children()
            .map(|c| c.path().clone())
            .filter(|cp| !is_in_prototypes_subtree(cp))
    .collect();

        if child_paths.is_empty() {
            if seen_leaf.insert(p.clone()) {
                leaves.push(p);
            }
            continue;
        }
        for cp in child_paths {
            let child_prim = stage.prim_at_path(cp);
            stack.push(child_prim);
        }
    }
    (leaves, instancers)
}

// --- compose local xform using xformOpOrder ---
fn get_local_transform(prim: &usd::Prim) -> Option<Matrix4d> {
    // 1) Try xformOpOrder first
    let order_tok = Token::new("xformOpOrder");
    if prim.has_attribute(&order_tok) {
        let attr = prim.attribute(&order_tok);

        // NOTE: token[] comes back as vt::Array<Token> in openusd-rs
        let order: vt::Array<Token> = attr.get::<vt::Array<Token>>();

        let mut local = Matrix4d::identity();
        for op_name in order.iter() {
            // Each entry is something like "xformOp:transform:stagemanager1"
            if prim.has_attribute(op_name) {
                let op_attr = prim.attribute(op_name);

                // In USDA the ops are declared as "matrix4d xformOp:transform:...".
                // read them as Matrix4d and multiply in listed order.
                let m = op_attr.get::<Matrix4d>();
                local *= m; // apply in-order
            }
        }
        // If order existed but had no valid ops, keep identity but return Some for clarity
        return Some(local);
    }

    // 2) Fallback: single consolidated transform
    let single_tok = Token::new("xformOp:transform");
    if prim.has_attribute(&single_tok) {
        let attr = prim.attribute(&single_tok);
        let m = attr.get::<Matrix4d>();
        return Some(m);
    }

    // No local xform
    None
}


fn accumulate_transforms(stage: &usd::Stage, start: &usd::Prim) -> Matrix4d {
    let mut total = Matrix4d::identity();

    // 1) apply start's local transform
    if let Some(local_xf) = get_local_transform(start) {
        //eprintln!("xf {:?}", local_xf);
        total *= local_xf;
    }

    // 2) climb parents continuously
    let mut path = start.path().parent_path();
    while !path.is_empty() {
        let prim = stage.prim_at_path(path);           // consume current path
        if let Some(local_xf) = get_local_transform(&prim) {
            //eprintln!("xf {:?}", local_xf);
            total *= local_xf;                         // child-first accumulation
        }
        path = prim.path().parent_path();              // next parent
    }

    total
}


//just check if usd is column or row. this can,t actually be correct.
#[inline]
fn transform_point_auto(m: &[[f64; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let x = p[0] as f64;
    let y = p[1] as f64;
    let z = p[2] as f64;

    // Heuristic: where is translation authored?
    // Row-vector => last row has T, Column-vector => last column has T.
    let row_t_mag = m[3][0].abs() + m[3][1].abs() + m[3][2].abs();
    let col_t_mag = m[0][3].abs() + m[1][3].abs() + m[2][3].abs();
    let use_row_vector = row_t_mag >= col_t_mag;

    if use_row_vector {
        // Row-vector multiply: [x y z 1] * M
        let tx = x * m[0][0] + y * m[1][0] + z * m[2][0] + 1.0 * m[3][0];
        let ty = x * m[0][1] + y * m[1][1] + z * m[2][1] + 1.0 * m[3][1];
        let tz = x * m[0][2] + y * m[1][2] + z * m[2][2] + 1.0 * m[3][2];
        let tw = x * m[0][3] + y * m[1][3] + z * m[2][3] + 1.0 * m[3][3];
        let inv_w = if tw != 0.0 { 1.0 / tw } else { 1.0 };
        [
            (tx * inv_w) as f32,
            (ty * inv_w) as f32,
            (tz * inv_w) as f32,
        ]
    } else {
        // Column-vector multiply: M * [x y z 1]^T
        let tx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * 1.0;
        let ty = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * 1.0;
        let tz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * 1.0;
        let tw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * 1.0;
        let inv_w = if tw != 0.0 { 1.0 / tw } else { 1.0 };
        [
            (tx * inv_w) as f32,
            (ty * inv_w) as f32,
            (tz * inv_w) as f32,
        ]
    }
}

//perhaps its best to not implement clone to prevent duplication instead of referancing.
//mesh here probably also needs to be an array although i'm not sure if it is possible to do
//that?()

#[derive(Debug, Clone)]
pub struct InstancedMesh {
    pub mesh: MeshData,
    pub positions: Vec<[f32; 3]>,
    pub rotations: Vec<[f32;4]>,
    pub scales: Vec<[f32;3]>
}
#[derive(Debug, Clone)]
pub struct MeshData {
    pub id: Option<String>, // <-- optional ID
    pub positions: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<usize>,
    pub face_vertex_indices: Vec<usize>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
}


impl MeshData {
    pub fn apply_transform(&mut self, xf: &openusd_rs::gf::Matrix4d) {
        // Get a reference to the internal 4x4 array
        let arr_ref: &[[f64; 4]; 4] = xf.as_array();

        // Copy into a local array so we can pass by value
        let m: [[f64; 4]; 4] = *arr_ref;

        self.positions = self
            .positions
            .iter()
            .map(|&p| transform_point_auto(&m, p))
            .collect();
    }
}


//seems to be loading all meshes on all positions instead of just the 1 mesh per thing.
//makes sence we`re not using proto_indeces


fn get_mesh_data(prim: &usd::Prim) -> MeshData {
    let path  = prim.path().clone();
    let stage = prim.stage();

    // Safer: use get instead of define if available.
    let mesh = usd_geom::Mesh::define(&stage, path);

    // Positions
    let positions = if mesh.has_points_attr() {
        let points_array: vt::Array<gf::Vec3f> = mesh.points_attr().get();
        points_array.iter().map(|p| [p.x, p.y, p.z]).collect()
    } else {
        Vec::new()
    };

    // Face vertex counts
    let face_vertex_counts = if mesh.has_face_vertex_counts_attr() {
        let counts_array: vt::Array<i32> = mesh.face_vertex_counts_attr().get();
        counts_array.iter().map(|&c| c as usize).collect()
    } else {
        Vec::new()
    };

    // Face vertex indices
    let face_vertex_indices = if mesh.has_face_vertex_indices_attr() {
        let indices_array: vt::Array<i32> = mesh.face_vertex_indices_attr().get();
        indices_array.iter().map(|&i| i as usize).collect()
    } else {
        Vec::new()
    };

    // Normals
    let normals = if mesh.has_normals_attr() {
        let normals_array: vt::Array<gf::Vec3f> = mesh.normals_attr().get();
        if normals_array.len() > 0 {
            Some(normals_array.iter().map(|n| [n.x, n.y, n.z]).collect())
        } else {
            None
        }
    } else {
        None
    };

    // UVs (Texcoords)
    let uvs: Option<Vec<[f32; 2]>> = if mesh.has_primvar(&Token::new("st")) {
        let uv_attr = mesh.primvar(&Token::new("st"));

        if let Some(val) = uv_attr.get_value() {
            if let Some(uv_arr) = val.get::<vt::Array<gf::Vec2f>>() {
                if uv_arr.len() > 0 {
                    Some(uv_arr.iter().map(|uv| [uv.x, uv.y]).collect())
                } else {
                    // exists but empty → None
                    None
                }
            } else {
                // wrong type → None
                None
            }
        } else {
            // authored attr but no value → None
            None
        }
    } else {
        // attribute missing entirely → None
        None
    };

    MeshData {
        id: Some(prim.path().to_string()),
        positions,
        face_vertex_counts,
        face_vertex_indices,
        normals,
        uvs,
    }
}

// Helper: inverse-transpose of the upper-left 3x3 (from [[f64;4];4])
fn inverse_transpose3x3(m: &[[f64; 4]; 4]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let co00 = (e * i - f * h);
    let co01 = -(d * i - f * g);
    let co02 = (d * h - e * g);
    let co10 = -(b * i - c * h);
    let co11 = (a * i - c * g);
    let co12 = -(a * h - b * g);
    let co20 = (b * f - c * e);
    let co21 = -(a * f - c * d);
    let co22 = (a * e - b * d);

    let det = a * co00 + b * co01 + c * co02;
    if det.abs() < 1e-12 {
        // Singular: just return transpose (better than garbage)
        return [[a, d, g], [b, e, h], [c, f, i]];
    }
    let inv_det = 1.0 / det;

    // inverse = adjugate / det; then transpose -> inverse-transpose = (inverse)^T = adj^T / det ^T == adj / det (since adj is already transposed cofactors)
    [
        [co00 * inv_det, co10 * inv_det, co20 * inv_det],
        [co01 * inv_det, co11 * inv_det, co21 * inv_det],
        [co02 * inv_det, co12 * inv_det, co22 * inv_det],
    ]
}

fn add_vec3(a: &[f32;3], b: &[f32;3]) -> [f32;3] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
}
fn mul_vec3(a: &[f32;3], b: &[f32;3]) -> [f32;3] {
    [a[0]*b[0], a[1]*b[1], a[2]*b[2]]
}
fn mul_quat(a: &[f32;4], b: &[f32;4]) -> [f32;4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
    [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]
}
fn rotate_vec3(v: &[f32;3], q: &[f32;4]) -> [f32;3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let qv = [0.0, v[0], v[1], v[2]];
    let q_conj = [w, -x, -y, -z];
    let res = mul_quat(&mul_quat(q, &qv), &q_conj);
    [res[1], res[2], res[3]]
}


fn read_orientations(instancer: &usd_geom::PointInstancer) -> Vec<[f32; 4]> {
    let attr = instancer.orientations_attr();
    match attr.get_value() {
        Some(val) => {
            if let Some(rot_array) = val.get::<vt::Array<gf::Quath>>() {
                rot_array.iter().map(|p| [f32::from(p.w), f32::from(p.i), f32::from(p.j), f32::from(p.k)]).collect()
            } else if let Some(rot_array) = val.get::<vt::Array<gf::Quatf>>() {
                rot_array.iter().map(|p| [p.w, p.i, p.j, p.k]).collect()
            } else if let Some(rot_array) = val.get::<vt::Array<gf::Quatd>>() {
                rot_array.iter().map(|p| [p.w as f32, p.i as f32, p.j as f32, p.k as f32]).collect()
            } else {
                vec![[1.0, 0.0, 0.0, 0.0]] // identity
            }
        }
        None => vec![[1.0, 0.0, 0.0, 0.0]], // identity
    }
}

fn collect_instanced_meshes(
    stage: &usd::Stage,
    prim: &usd::Prim,
    parent_pos: [f32; 3],
    parent_rot: [f32; 4],
    parent_scale: [f32; 3],
) -> Vec<InstancedMesh> {
    
    let type_token = prim.type_name();     // keep the owned token alive
    let type_name = type_token.as_str();   // borrow from it safely

    if type_name == "Mesh" {
        // Leaf case: just return one mesh with parent transform baked in
        let mesh = get_mesh_data(prim);
        return vec![InstancedMesh {
            mesh,
            positions: vec![parent_pos],
            rotations: vec![parent_rot],
            scales: vec![parent_scale],
        }];
    }

    if type_name == "PointInstancer" {
        let instancer = usd_geom::PointInstancer::define(stage, prim.path().clone());

        // indices
        let proto_indices: vt::Array<i32> = instancer.proto_indices_attr().get();
        let indices: Vec<usize> = proto_indices.iter().map(|&i| i as usize).collect();

        // local attrs
        let local_positions: Vec<[f32; 3]> =
            instancer.positions_attr().get::<vt::Array<gf::Vec3f>>()
                .iter().map(|p| [p.x, p.y, p.z]).collect();

        let local_scales: Vec<[f32; 3]> =
            instancer.scales_attr().get::<vt::Array<gf::Vec3f>>()
                .iter().map(|p| [p.x, p.y, p.z]).collect();

        let local_rotations = read_orientations(&instancer);

        let proto_paths: vt::Array<sdf::Path> = instancer.prototypes_rel().targets().into();

        // Storage for merged results
        let mut out: Vec<InstancedMesh> = Vec::new();

        for (i, proto_index) in indices.iter().enumerate() {
            if *proto_index < proto_paths.len() {
                let proto_path = &proto_paths[*proto_index];

                let proto_prim = stage.prim_at_path(proto_path.clone());



                // unwrap common wrapper prims like Xform
                let child_opt = proto_prim.children().next(); // keep this alive
                let effective_prim: &usd::Prim = if proto_prim.type_name().as_str() == "Xform" {
                    child_opt.as_ref().unwrap_or(&proto_prim)
                } else {
                    &proto_prim
                };


                // compose parent + local
                let lp = local_positions[i];
                let lr = local_rotations[i];
                let ls = local_scales[i];

                let composed_rot = mul_quat(&parent_rot, &lr);
                let composed_scale = mul_vec3(&parent_scale, &ls);

                let rotated = rotate_vec3(&lp, &parent_rot);
                let scaled = mul_vec3(&rotated, &parent_scale);
                let composed_pos = add_vec3(&parent_pos, &scaled);

                // recurse
                let children = collect_instanced_meshes(
                    stage,
                    &proto_prim,
                    composed_pos,
                    composed_rot,
                    composed_scale,
                );

                // merge children into `out`
                for child in children {
                    // try to find an existing entry with same mesh
                    if let Some(existing) = out.iter_mut().find(|m| m.mesh.id == child.mesh.id) {
                        existing.positions.extend(child.positions);
                        existing.rotations.extend(child.rotations);
                        existing.scales.extend(child.scales);
                    } else {
                        out.push(child);
                    }
                }
            }
        }

        return out;
    }

    // Unknown prim type
    Vec::new()
}

pub fn fetch_stage_usd(stagep: &str) -> (Vec<MeshData>, Vec<InstancedMesh>) {
    let stage = usd::Stage::open(stagep);
    let (leaves, instancers) = collect_leaves_and_instancers(&stage);

    // --- direct meshes (non-instanced) ---
    let mut meshes_out = Vec::new();
    for p in &leaves {
        let prim = stage.prim_at_path(p.clone());
        let type_token = prim.type_name();
        if type_token.as_str() == "Mesh" {
            let xf = accumulate_transforms(&stage, &prim);
            let mut mesh = get_mesh_data(&prim);
            mesh.apply_transform(&xf);
            meshes_out.push(mesh);
        }
    }

    // --- instanced meshes (recursive) ---
    let mut instanced_meshes = Vec::new();
    for p in &instancers {
        let prim = stage.prim_at_path(p.clone());
        instanced_meshes.extend(collect_instanced_meshes(
            &stage,
            &prim,
            [0.0, 0.0, 0.0],         // root pos
            [1.0, 0.0, 0.0, 0.0],    // identity quat (w=1)
            [1.0, 1.0, 1.0],         // unit scale
        ));
    }

    (meshes_out, instanced_meshes)
}
