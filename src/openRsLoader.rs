use openusd_rs::{
    gf::{self, Matrix4d},
    sdf,
    tf::Token,
    usd, usd_geom, vt,
};
use std::collections::HashSet;

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
    let mut current: usd::Prim = stage.prim_at_path(start.path().clone());

    loop {
        //define parent
        let parent_path = current.path().parent_path();

        //aply and print local transform
        if let Some(local_xf) = get_local_transform(&current) {
            // child-first accumulation (total = total * local)
            total *= local_xf;
        }

        //stop if root reloop if not
        if parent_path.is_empty() {
            break;
        }
        current = stage.prim_at_path(parent_path);
    }

    total
}

pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub face_vertex_counts: Vec<usize>,
    pub face_vertex_indices: Vec<usize>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
}

fn get_mesh_data(prim: &usd::Prim) -> MeshData {
    let mesh = usd_geom::Mesh::define(&prim.stage(), prim.path().clone());

    // Positions
    let points_array: vt::Array<gf::Vec3f> = mesh.points_attr().get();
    let positions: Vec<[f32; 3]> = points_array.iter().map(|p| [p.x, p.y, p.z]).collect();

    // Face vertex counts
    let counts_array: vt::Array<i32> = mesh.face_vertex_counts_attr().get();
    let face_vertex_counts: Vec<usize> = counts_array.iter().map(|&c| c as usize).collect();

    // Face vertex indices
    let indices_array: vt::Array<i32> = mesh.face_vertex_indices_attr().get();
    let face_vertex_indices: Vec<usize> = indices_array.iter().map(|&i| i as usize).collect();

    // Normals
    let normals_array: vt::Array<gf::Vec3f> = mesh.normals_attr().get();
    let normals = if normals_array.len() > 0 {
        Some(normals_array.iter().map(|n| [n.x, n.y, n.z]).collect())
    } else {
        None
    };

    // UVs not handled yet
    let uvs = None;

    MeshData {
        positions,
        face_vertex_counts,
        face_vertex_indices,
        normals,
        uvs,
    }
}



pub fn fetch_stage_usd(stagep: &str) -> Vec<MeshData> {
    let stage = usd::Stage::open(stagep);
    let (leaves, instancers) = collect_leaves_and_instancers(&stage);

    let mut meshes_out = Vec::new();

    for p in &leaves {
        let prim = stage.prim_at_path(p.clone());
        if prim.type_name().as_str() == "Mesh" {
            let mesh = get_mesh_data(&prim);
            meshes_out.push(mesh);
        }
    }

    // You can decide whether to handle instancers here or not.
    // For now, we only return MeshData from leaf prims.

    meshes_out
}

