pub const USD_SCENE0_LABEL: &str = "Scene0";

pub fn scene_label_for_path(path: &str) -> String {
    format!("Scene:{path}")
}

pub fn mesh_label_for_path(path: &str) -> String {
    format!("Mesh:{path}")
}

pub fn material_label_for_path(path: &str) -> String {
    format!("Material:{path}")
}
